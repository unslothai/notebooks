# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""Dependency parity gate (AC3 / R-deps — broad-catalog scope).

Parametrised over molab/*.py files that ACTUALLY EXIST ON DISK so that a
partial generation run (best-effort) does not red the whole suite.

For every generated molab/*.py file:
  1. Source install packages extracted from the corresponding
     ``nb/<name>.ipynb`` post-injection file must all appear in the generated
     file — either as PEP 723 inline metadata or inside a marimo setup cell.
  2. Every dropped dependency must be explicitly listed via the real
     ``molab_dependencies`` API (``get_dropped_deps(nb)`` or module-level
     dicts).  There is NO silent comment-scan fallback — if molab_dependencies
     is present on disk but lacks ``get_dropped_deps``, the test FAILS loudly
     (DA-02: a test that silently degrades is a test that doesn't test).
  3. No per-notebook pin may be silently collapsed into a single global list.

Tests skip gracefully when:
- The corresponding nb/<name>.ipynb post-injection file is absent.
- molab_dependencies.py does NOT YET EXIST on disk (pre-generator-landing).
  Once it exists, the API contract is mandatory.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_manifest as mm  # noqa: E402

# ---------------------------------------------------------------------------
# Parametrise over files actually on disk — tolerates partial generation.
# ---------------------------------------------------------------------------

_MOLAB_DIR = REPO_ROOT / "molab"
_GENERATED_FILES: list[Path] = (
    sorted(_MOLAB_DIR.glob("*.py")) if _MOLAB_DIR.exists() else []
)

if not _GENERATED_FILES:
    _GENERATED_FILES = [nb.output for nb in mm.get_active_notebooks()]

# Map stem -> MolabNotebook for dependency lookup.
_STEM_TO_NB: dict[str, mm.MolabNotebook] = {
    nb.output.stem: nb for nb in mm.MOLAB_NOTEBOOKS
}


# ---------------------------------------------------------------------------
# Source notebook helpers (extract pip packages from nb/<name>.ipynb)
# ---------------------------------------------------------------------------

# Matches only shell-invocation pip install lines: lines that start with !pip
# or %pip (Jupyter shell escapes).  Plain prose containing "pip install" (e.g.
# "run pip install or uv pip install") is excluded by requiring the line to
# start with the shell escape.  The `re.MULTILINE` flag makes ^ match
# start-of-line so the anchor works correctly in multi-line cell source.
_PIP_LINE_RE = re.compile(
    r"^[!%]pip\s+install\s+([^\n]+)", re.IGNORECASE | re.MULTILINE
)
# Also match subprocess.run(["pip", "install", ...]) style calls inside cells.
_SUBPROCESS_PIP_RE = re.compile(
    r'subprocess\.run\s*\(\s*\[.*?["\']pip["\'].*?["\']install["\']',
    re.IGNORECASE | re.DOTALL,
)
_PKG_NAME_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_\-]*)")

# Tokens that look like package names but are actually CLI noise or prose —
# single-character flags, common English words used as connectives/qualifiers.
_NON_PACKAGE_TOKENS: frozenset[str] = frozenset(
    {"or", "and", "uv", "pip", "install", "upgrade", "quiet", "q", "qqq", "no",
     "deps", "pre", "user", "system", "index", "url", "extra", "find", "links",
     # VCS prefixes that appear as first token in 'git+https://...' package specs.
     "git", "hg", "svn", "bzr"}
)


def _extract_pip_packages_from_nb(nb_path: Path) -> set[str]:
    """Return normalised lowercase package names from shell pip install lines
    in nb_path (a post-injection .ipynb Jupyter notebook).

    Only lines starting with ``!pip`` or ``%pip`` are matched — this excludes
    prose sentences that contain the words "pip install" but are not actual
    install commands.
    """
    if not nb_path.exists():
        return set()
    raw = nb_path.read_text(encoding="utf-8")
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return set()

    packages: set[str] = set()
    cells = data.get("cells", [])
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        text = "".join(source_lines) if isinstance(source_lines, list) else source_lines
        for match in _PIP_LINE_RE.finditer(text):
            args = match.group(1)
            # Strip inline comments: everything from the first bare '#' onward
            # is a comment (e.g. "protobuf==3.20.3 # required" → "protobuf==3.20.3").
            # A '#' that appears inside a URL fragment (git+https://...#egg=name)
            # is handled by the _PKG_NAME_RE below since the token starts with
            # 'git+', which fails the name-char check.
            comment_pos = args.find(" #")
            if comment_pos != -1:
                args = args[:comment_pos]
            for token in args.split():
                token = token.strip().rstrip("\\,;")
                if not token:
                    continue
                if token.startswith("-"):
                    continue
                # Skip f-string expressions and quoted flags.
                if token.startswith("{") or token.startswith('"') or token.startswith("'"):
                    continue
                pkg_match = _PKG_NAME_RE.match(token)
                if pkg_match:
                    name = pkg_match.group(1).lower().replace("-", "_")
                    if name not in _NON_PACKAGE_TOKENS and len(name) >= 2:
                        packages.add(name)
    return packages


# ---------------------------------------------------------------------------
# Generated file helpers (extract packages from PEP 723 + setup cell)
# ---------------------------------------------------------------------------

_PEP723_BLOCK_RE = re.compile(
    r"^#\s*///\s*script\b.*?^#\s*///", re.MULTILINE | re.DOTALL
)

# Spec -> name extraction.  Handles all PEP 508 / PEP 723 spec forms the
# generator emits:
#   "foo"                                                  -> "foo"
#   "foo>=1.0"                                             -> "foo"
#   "foo[extra,extra2]"                                    -> "foo"
#   "foo[extra] @ git+https://repo.git#subdirectory=..."   -> "foo"
#   "git+https://github.com/user/repo.git@main"            -> "repo"  (last URL segment)
_NAMED_SPEC_RE = re.compile(
    r"^\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~]|@|;|$)"
)
_BARE_VCS_RE = re.compile(
    r"^\s*(?:git|hg|svn|bzr)\+[^\s@]+/(?P<name>[A-Za-z0-9_\-]+?)(?:\.git)?(?:@|$|#|\s)"
)


def _spec_to_name(spec: str) -> str | None:
    """Extract the package name from a PEP 508 / PEP 723 dep spec.

    Returns ``None`` if the spec is unparseable so the caller can skip it.
    See the regex docstrings above for the supported forms.
    """
    spec = spec.strip()
    m = _NAMED_SPEC_RE.match(spec)
    if m:
        return m.group(1)
    m = _BARE_VCS_RE.match(spec)
    if m:
        return m.group("name")
    return None


def _extract_pep723_packages(text: str) -> set[str]:
    """Return normalised package names from PEP 723 ``# /// script`` blocks.

    PEP 723 inline metadata is TOML wrapped in ``# `` comment prefixes.
    Strip the prefixes and parse with ``tomllib`` so direct-reference
    specs (``name[extras] @ url``) and bracketed extras survive the
    parse — the previous regex was confused by literal ``]`` inside
    extras or URL fragments (review finding #8, #3).
    """
    packages: set[str] = set()
    for block_match in _PEP723_BLOCK_RE.finditer(text):
        block = block_match.group()
        # Strip the `# ` comment prefix from each TOML line.  Drop the
        # `# /// script` / `# ///` delimiters themselves.
        toml_lines = []
        for line in block.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("# ///"):
                continue
            toml_lines.append(re.sub(r"^#\s?", "", line))
        toml_body = "\n".join(toml_lines)
        try:
            import tomllib

            meta = tomllib.loads(toml_body)
        except Exception:
            continue
        for spec in meta.get("dependencies", []) or []:
            name = _spec_to_name(spec)
            if name:
                packages.add(name.lower().replace("-", "_"))
    return packages


def _extract_setup_cell_packages(text: str) -> set[str]:
    """Return normalised package names from pip install lines in the file."""
    packages: set[str] = set()
    for match in _PIP_LINE_RE.finditer(text):
        args = match.group(1)
        for token in args.split():
            token = token.strip()
            if token.startswith("-"):
                continue
            pkg_match = _PKG_NAME_RE.match(token)
            if pkg_match:
                packages.add(pkg_match.group(1).lower().replace("-", "_"))
    return packages


# ---------------------------------------------------------------------------
# Load molab_dependencies for drop-reason registry (DA-02: no silent fallback)
# ---------------------------------------------------------------------------

#: Sentinel returned when molab_dependencies.py is not yet on disk.
_DEP_MOD_ABSENT = object()


def _import_dependencies():
    """Import molab_dependencies from scripts/.

    Returns the module if present, or the ``_DEP_MOD_ABSENT`` sentinel when
    the file does not yet exist (pre-generator-landing — skip is acceptable).

    FAILS (raises ImportError) if the file EXISTS but cannot be imported or
    does not expose ``get_dropped_deps``.  A present-but-broken module is a
    real error, not a graceful skip (DA-02).
    """
    dep_path = REPO_ROOT / "scripts" / "molab_dependencies.py"
    if not dep_path.exists():
        return _DEP_MOD_ABSENT

    # File exists — import must succeed and expose the required API.
    spec = importlib.util.spec_from_file_location(
        "molab_dependencies", str(dep_path)
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise ImportError(
            f"molab_dependencies.py exists but failed to import: {exc}"
        ) from exc

    if not hasattr(mod, "get_dropped_deps"):
        raise ImportError(
            "molab_dependencies.py is present but does not expose "
            "'get_dropped_deps'.  The dependency-parity test requires this "
            "function (DA-02).  Add it or broadcast the correct API name."
        )

    return mod


# Module-level import: fails at collection time if the module is broken,
# which surfaces the error clearly rather than silently passing every test.
_DEP_MOD = _import_dependencies()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_dependency_parity(py_file: Path) -> None:
    """Every source install package must appear in the generated molab file
    or be listed in the dependency module's drop-reason registry (AC3).

    Source install commands are taken from ``nb/<stem>.ipynb`` (post-injection).
    Generated packages come from PEP 723 metadata + any setup cell pip lines.

    Drop reasons MUST come from the real molab_dependencies API.  There is no
    silent comment-scan fallback — DA-02: a test that silently passes when the
    API is absent is a test that doesn't test.
    """
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")

    if _DEP_MOD is _DEP_MOD_ABSENT:
        pytest.skip(
            "scripts/molab_dependencies.py not yet committed; "
            "dependency parity check deferred until generator-implementer lands."
        )

    nb = _STEM_TO_NB.get(py_file.stem)

    # Resolve the post-injection source notebook.
    nb_name = nb.source.name if nb is not None else f"{py_file.stem}.ipynb"
    source_nb_path = REPO_ROOT / "nb" / nb_name
    if not source_nb_path.exists():
        pytest.skip(
            f"Post-injection notebook {source_nb_path.name} not found; "
            "dependency parity check deferred until nb/ is present."
        )

    source_packages = _extract_pip_packages_from_nb(source_nb_path)
    if not source_packages:
        pytest.skip(
            f"No pip install lines found in {source_nb_path.name}; "
            "nothing to check."
        )

    generated_text = py_file.read_text(encoding="utf-8")
    generated_packages = (
        _extract_pep723_packages(generated_text)
        | _extract_setup_cell_packages(generated_text)
    )

    # Collect drop reasons from the REAL API only — no silent fallback (DA-02).
    drop_reasons: dict[str, str] = {}
    if nb is not None:
        drop_reasons = _DEP_MOD.get_dropped_deps(nb) or {}

    # Also check module-level dicts if present (supplementary, not a fallback).
    # Keys may be bare package names (str) or (stem, pkg) tuples.
    stem = nb.source.stem if nb is not None else py_file.stem
    module_level_dropped: set[str] = set()
    for attr in ("DROPPED_DEPS", "UNSUPPORTED_DEPS", "IGNORED_DEPS"):
        mapping = getattr(_DEP_MOD, attr, None)
        if not isinstance(mapping, dict):
            continue
        for key in mapping:
            if isinstance(key, str):
                module_level_dropped.add(key)
            elif isinstance(key, tuple) and len(key) == 2 and key[0] == stem:
                module_level_dropped.add(key[1])

    missing: list[str] = []

    for pkg in sorted(source_packages):
        if pkg in generated_packages:
            continue
        if pkg in drop_reasons:
            continue
        if pkg in module_level_dropped:
            continue
        # No comment-scan fallback — if it reaches here, it's genuinely missing.
        missing.append(pkg)

    if missing:
        pytest.fail(
            f"DEPENDENCY DRIFT: {py_file.name} is missing packages from the "
            f"source install cell of {source_nb_path.name}:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nEach dropped package must be represented in:\n"
            "  (a) PEP 723 inline metadata (# /// script ... /// block)\n"
            "  (b) a marimo setup cell pip install line\n"
            "  (c) molab_dependencies.get_dropped_deps(nb) return value\n"
            "  (d) molab_dependencies.DROPPED_DEPS / UNSUPPORTED_DEPS / "
            "IGNORED_DEPS module-level dict\n"
            "\nNOTE: comment-scan fallback has been removed (DA-02). "
            "Add the package to the API or include it in the generated output."
        )


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_no_global_dep_list_without_per_notebook_entries(py_file: Path) -> None:
    """Per-notebook PEP 723 metadata must not be a single shared global list.

    Each generated molab/*.py must have at least one package in its own
    inline metadata block — not a zero-length list."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")

    generated_text = py_file.read_text(encoding="utf-8")
    packages = _extract_pep723_packages(generated_text)

    if not packages:
        pytest.skip(
            f"{py_file.name} has no PEP 723 metadata; "
            "this check is only relevant when metadata is present."
        )

    assert len(packages) >= 1, (
        f"{py_file.name} PEP 723 block lists zero packages. "
        "At minimum 'unsloth' should be present."
    )
