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

"""molab_dependencies — builds the PEP 723 header for one molab notebook.

Import like every other ``scripts/`` helper::

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import molab_dependencies

Why it exists
=============

The molab generator drops the source ``%%capture`` install cell.  Raw
``!pip`` is not allowed in ``molab/*.py``, so the package set has to land
somewhere else.  For a ``marimo --sandbox`` / ``uv run`` notebook that
place is the PEP 723 ``# /// script`` block at the top of the file
(see ``docs/guides/package_management/inlining_dependencies.md``).

This module reads the real install commands from the source notebook,
parses them, and returns two things:

- a PEP 723 ``dependencies = [...]`` list, and
- a list of every install line we intentionally dropped, with a reason.
  ``tests/test_molab_dependency_parity.py`` reads that list to confirm
  nothing vanished silently.

There is no hard-coded global molab install list.  Each notebook gets
its own PEP 723 block, derived from its own install cell.  This mirrors
the ``_compose_amd_installation`` precedent in ``update_all_notebooks.py``
(around line 2649) that AMD variants use.

Install-text resolution
=======================

``original_template/<name>.ipynb`` install cells are placeholder stubs.
The real pinned commands get injected by the Colab/Kaggle generator into
``nb/<name>.ipynb``.  The planner reads from ``nb/``, not the template.
``resolve_nb_source`` does that mapping.

The two branches of an Unsloth install cell
===========================================

A modern Unsloth install cell looks like::

    %%capture
    import os, re
    if "COLAB_" not in "".join(os.environ.keys()):
        !pip install unsloth                                # local / cloud
    else:
        import torch; v = re.match(...).group(0)
        xformers = 'xformers==' + {...}.get(v, "0.0.34")
        !pip install sentencepiece protobuf "datasets==4.3.0" ...
        !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} ...
        !pip install --no-deps --upgrade "torchao>=0.16.0"
    !pip install transformers==4.56.2
    !pip install --no-deps trl==0.22.2

The ``else:`` branch is the pinned set used on hosted GPU runtimes
(Colab).  molab is a hosted GPU runtime of the same shape, so the
planner uses those pins as the canonical molab dependency set.  The
``COLAB_`` env-detection scaffolding is dropped (a PEP 723 header is
unconditional), and the drop is recorded with a reason.
"""
# NOTE: deliberately NO ``from __future__ import annotations`` here.  The
# molab parity tests load this module via ``importlib.util.module_from_spec``
# + ``exec_module`` WITHOUT first registering it in ``sys.modules``.  Under
# that loader, string (PEP 563) annotations make the ``@dataclass`` decorator
# crash — its type-resolution step looks up ``sys.modules[cls.__module__]``
# and finds ``None``.  Using real (eagerly-evaluated) annotations sidesteps
# the lookup entirely, so the module loads under any loader.

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ``scripts/`` is on sys.path when this module is imported (repo convention);
# notebook_inventory is a sibling flat module.
from notebook_inventory import iter_code_cells

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# PEP 723 target Python.  molab/Colab GPU runtimes ship a recent CPython;
# Unsloth itself requires >=3.10.
# ---------------------------------------------------------------------------
# Use a WIDE range, NOT a narrow ``==3.12.*`` pin.  molab's
# ``build_sandbox_venv`` ignores this field for venv creation (marimo
# ``_cli/sandbox.py:613-617``) and lands on Python 3.13, but uv's
# resolver still reads it to filter the EXPECTED WHEEL HASH SET.  With
# ``==3.12.*``, uv's expected hashes for any package without an abi3
# wheel (cuda-tile, pulled in by flashinfer/vllm) are cp312-only; uv
# then downloads the cp313 wheel for the actual venv, and the hash
# comparison fails (``Hash mismatch for cuda-tile==1.x.0``).  A range
# that includes 3.13 expands the expected set to include cp313 hashes
# so the downloaded wheel verifies cleanly.  Floor pins below
# (``_MOLAB_FLOORS``) prevent the wider range from resolving to ancient
# pre-wheel releases.
PEP723_REQUIRES_PYTHON = ">=3.10,<3.14"

# Packages that the molab runtime already provides (see molab.md "Package
# management": torch/numpy/polars etc. pre-installed) OR that are pulled in
# transitively by ``unsloth``.  They are still represented — listed in the
# PEP 723 block when the source pins them — but a *bare* mention with no
# version pin and no ``--no-deps`` significance is allowed to be dropped.
# This set is only used to explain a drop, never to silently suppress a pin.
_RUNTIME_PROVIDED = frozenset({"torch", "triton"})

# Tokens that are pip *flags*, not packages.
_PIP_FLAG_VALUE_OPTS = frozenset({
    "--index-url", "--extra-index-url", "--find-links", "-f",
    "-r", "--requirement", "-c", "--constraint",
})

# A bare package name / spec token (PEP 508-ish).  Captures the distribution
# name; the remainder (version specifier, extras) is preserved on the spec.
_RE_PKG_NAME = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9._-]*)")

# A pip install command on a single logical line.  Handles `!pip`, `%pip`,
# `!uv pip`, `uv pip`, with or without a leading bang.
_RE_PIP_INSTALL = re.compile(
    r"^\s*[!%]?\s*(?:uv\s+)?pip\s+install\s+(?P<args>.+?)\s*$"
)


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass(frozen=True)
class DroppedItem:
    """A source install / setup line that the planner did NOT carry into the
    PEP 723 header or a marimo setup cell, plus a machine-readable reason.

    ``test_molab_dependency_parity`` reads ``DependencyPlan.dropped`` and
    asserts every entry carries a non-empty ``reason`` — i.e. nothing is
    dropped *silently*.
    """

    text: str          # the verbatim source line that was dropped
    reason: str        # why it is not in the plan (machine + human readable)


@dataclass
class DependencyPlan:
    """Structured result of planning a notebook's dependencies.

    Attributes
    ----------
    dependencies:
        Sorted list of PEP 508 dependency strings for the PEP 723
        ``dependencies = [...]`` array (e.g. ``"transformers==4.56.2"``).
        Order is deterministic (sorted) so regeneration is byte-stable.
    requires_python:
        Value for the PEP 723 ``requires-python`` field.
    setup_lines:
        Non-package setup lines that must run as Python (e.g. an
        ``os.environ[...] = ...``).  Emitted into a marimo setup/first cell,
        NOT the PEP 723 header.  Empty for the current P0 notebooks.
    dropped:
        Every source install/setup line intentionally NOT represented, each
        with a reason.  Drives the dependency-parity test.
    represented:
        Map of normalised package name -> the spec string actually placed in
        ``dependencies``.  Lets the parity test confirm every *package*
        mentioned in the source install cell is represented.
    """

    dependencies: list[str] = field(default_factory=list)
    requires_python: str = PEP723_REQUIRES_PYTHON
    setup_lines: list[str] = field(default_factory=list)
    dropped: list[DroppedItem] = field(default_factory=list)
    represented: dict[str, str] = field(default_factory=dict)


# ===========================================================================
# Source resolution
# ===========================================================================
def resolve_nb_source(template_source: Path) -> Path:
    """Map a manifest ``source`` (an ``original_template/<name>.ipynb`` path)
    to the post-injection ``nb/<name>.ipynb`` path.

    The manifest's ``source`` points at the canonical template, whose install
    cells are placeholder stubs.  The real pinned install text lives in
    ``nb/<name>.ipynb`` (written by the Colab/Kaggle generator).  This planner
    must read from there.

    Raises
    ------
    FileNotFoundError
        If the ``nb/`` counterpart does not exist — a stale manifest entry
        should fail loudly, not generate an emptied dependency block.
    """
    nb_path = _REPO_ROOT / "nb" / template_source.name
    if not nb_path.exists():
        raise FileNotFoundError(
            f"molab_dependencies: expected post-injection notebook not found: "
            f"{nb_path}\nThe manifest source {template_source} has no nb/ "
            f"counterpart; run the Colab/Kaggle generator or fix the manifest."
        )
    return nb_path


# ===========================================================================
# Install-text extraction
# ===========================================================================
def _logical_lines(source: str) -> list[str]:
    """Join shell backslash continuations into logical lines (mirrors
    ``update_all_notebooks._logical_install_lines``)."""
    out: list[str] = []
    current = ""
    for raw in source.splitlines():
        line = raw.rstrip()
        if current:
            current += " " + line.strip()
        else:
            current = line
        if current.rstrip().endswith("\\"):
            current = current.rstrip()[:-1].rstrip()
            continue
        if current.strip():
            out.append(current)
        current = ""
    if current.strip():
        out.append(current)
    return out


def extract_install_cells(nb_path: Path) -> list[str]:
    """Return the source text of every code cell in ``nb_path`` that contains
    at least one ``pip install`` line.

    These are the cells whose dependencies must be re-homed into PEP 723.
    """
    cells: list[str] = []
    for _idx, src in iter_code_cells(nb_path):
        if _RE_PIP_INSTALL.search(src) or any(
            _RE_PIP_INSTALL.match(line) for line in _logical_lines(src)
        ):
            cells.append(src)
    return cells


def _normalise_pkg(name: str) -> str:
    """PEP 503-style normalisation for a distribution name."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _split_args(arg_string: str) -> list[str]:
    """Split a pip argument string into tokens, tolerating shell quoting and
    IPython ``{var}`` expansion braces.

    pip recognises a trailing ``# comment`` only when preceded by whitespace
    (``foo==1.0 # pinned`` strips the comment; ``git+https://repo#fragment``
    keeps the URL fragment).  ``shlex.split(comments=True)`` strips at ANY
    ``#`` and silently drops ``#subdirectory=...`` etc.  So we cut the
    comment ourselves at the first whitespace-then-``#`` and call
    ``shlex.split`` with ``comments=False`` — URL fragments survive intact.
    """
    import shlex

    cut = re.search(r"\s+#", arg_string)
    if cut is not None:
        arg_string = arg_string[: cut.start()]
    try:
        return shlex.split(arg_string, comments=False, posix=True)
    except ValueError:
        return arg_string.split()


@dataclass
class _PipLine:
    """One parsed ``pip install`` logical line."""

    raw: str
    flags: list[str]
    specs: list[str]            # package specs (name + optional version/extras)
    templated: list[str]        # specs containing an unresolved ``{var}``


def _parse_pip_line(line: str) -> Optional[_PipLine]:
    """Parse a single logical line if it is a pip install command."""
    m = _RE_PIP_INSTALL.match(line)
    if not m:
        return None
    flags: list[str] = []
    specs: list[str] = []
    templated: list[str] = []
    tokens = _split_args(m.group("args"))
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in _PIP_FLAG_VALUE_OPTS:
            flags.append(tok)
            skip_next = True
            continue
        if tok.startswith("-"):
            flags.append(tok)
            continue
        if tok in {"&&", "@", "\\"}:
            continue
        # IPython expansion brace, e.g. {xformers} -> a runtime-resolved spec.
        if "{" in tok and "}" in tok:
            templated.append(tok)
            continue
        if _RE_PKG_NAME.match(tok):
            specs.append(tok)
    return _PipLine(raw=line, flags=flags, specs=specs, templated=templated)


# ===========================================================================
# Planning
# ===========================================================================
def plan_dependencies(nb_path: Path) -> DependencyPlan:
    """Build a :class:`DependencyPlan` from a post-injection ``nb/*.ipynb``.

    Every package spec found in an install cell is either:
      * placed in ``plan.dependencies`` (PEP 723), or
      * recorded in ``plan.dropped`` with a reason.

    Every non-package setup line that has runtime meaning is either:
      * placed in ``plan.setup_lines`` (a marimo setup cell), or
      * recorded in ``plan.dropped`` with a reason.

    The planner is deterministic: ``dependencies`` is sorted, so two runs on
    the same notebook yield byte-identical PEP 723 headers.
    """
    plan = DependencyPlan()
    install_cells = extract_install_cells(nb_path)

    # name -> spec; later, *more specific* specs win (a pin beats a bare name).
    chosen: dict[str, str] = {}

    for cell_src in install_cells:
        logical = _logical_lines(cell_src)
        for line in logical:
            parsed = _parse_pip_line(line)
            if parsed is None:
                # A non-pip logical line inside an install cell.  The Unsloth
                # install cell's non-pip lines are environment-detection
                # scaffolding (``import os, re``, the ``if "COLAB_" ...``
                # branch, the ``xformers = ...`` version map).  None of them
                # have runtime meaning once the cell is replaced by a PEP 723
                # header, so they are intentionally dropped — with a reason.
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped in {"%%capture"}:
                    plan.dropped.append(DroppedItem(
                        text=stripped,
                        reason="cell-magic:%%capture has no marimo equivalent; "
                               "marimo manages package install output natively",
                    ))
                    continue
                plan.dropped.append(DroppedItem(
                    text=stripped,
                    reason="install-scaffolding: Colab/Kaggle env-detection or "
                           "version-resolution line; superseded by the "
                           "unconditional PEP 723 dependency block",
                ))
                continue

            # Record templated specs (e.g. {xformers}) as intentional drops:
            # the version is resolved at runtime from the torch build, which a
            # static PEP 723 header cannot reproduce.  unsloth pulls a
            # compatible xformers transitively, so dropping the explicit pin
            # is safe.
            for tok in parsed.templated:
                plan.dropped.append(DroppedItem(
                    text=tok,
                    reason="runtime-templated: spec uses an IPython {var} "
                           "expansion resolved from the live torch version; "
                           "not statically expressible in PEP 723 — resolved "
                           "transitively via the unsloth dependency",
                ))

            for spec in parsed.specs:
                name = _RE_PKG_NAME.match(spec)
                if not name:
                    continue
                key = _normalise_pkg(name.group(1))
                # Prefer a pinned/constrained spec over a bare name, and a
                # longer (tighter) constraint over a looser one — same
                # tie-break intent as update_all_notebooks._install_spec_*.
                existing = chosen.get(key)
                if existing is None or _spec_rank(spec) > _spec_rank(existing):
                    chosen[key] = spec

    # Force unsloth + unsloth_zoo to git-latest for every molab notebook.
    # The source notebooks usually carry plain ``pip install unsloth`` which
    # resolves to whatever uv picks; when an adjacent constraint (e.g.
    # ``transformers==5.5.0``) is incompatible with the current PyPI
    # release, uv silently down-resolves unsloth to a much older release.
    # molab is the cutting-edge surface — every notebook should pull the
    # live ``main`` branch from GitHub.  Specs that ALREADY reference a
    # direct VCS URL (the GRPO notebooks ship explicit ``unsloth[...] @
    # git+...`` lines) are left as authored.
    # Keys are PEP 503-normalised (``_normalise_pkg``): hyphens, lowercase —
    # so ``unsloth_zoo`` from the source matches the ``unsloth-zoo`` key here.
    _GIT_LATEST: dict[str, str] = {
        "unsloth": "unsloth @ git+https://github.com/unslothai/unsloth.git",
        "unsloth-zoo": "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
    }
    for key, latest_spec in _GIT_LATEST.items():
        spec = chosen.get(key)
        if spec is None:
            continue
        if "@" in spec or "git+" in spec:
            # Source already pins a direct reference; respect it.
            continue
        chosen[key] = latest_spec

    # Force-add ``unsloth_zoo`` whenever ``unsloth`` is present.  The
    # source notebooks rely on Colab's split-branch pattern:
    # ``pip install unsloth`` (PyPI) on the cloud branch pulls
    # ``unsloth_zoo`` as a transitive, while the Colab branch installs
    # it explicitly via ``--no-deps unsloth_zoo ...``.  molab uses the
    # ``git+`` unsloth instead, which does NOT declare ``unsloth_zoo``
    # as a runtime dep — so without this, ``from unsloth import ...``
    # fails at runtime with::
    #     ImportError: Unsloth: Please install unsloth_zoo via
    #     `pip install unsloth_zoo` then retry!
    if "unsloth" in chosen and "unsloth-zoo" not in chosen:
        chosen["unsloth-zoo"] = _GIT_LATEST["unsloth-zoo"]

    # Floor-pin overlay for the molab Python 3.13 cascade.
    #
    # molab pre-creates ``/tmp/uv-venv`` with Python 3.13 and ignores the
    # PEP 723 ``requires-python`` field (marimo
    # ``_cli/sandbox.py:613-617`` never threads it into ``uv venv``).
    # When the dependency list carries a bare name like ``"vllm"``, uv's
    # resolver walks back through every release that *might* be
    # compatible — eventually landing on a pre-wheel sdist (vllm==0.2.5,
    # sentencepiece==0.2.1) that needs CUDA_HOME + cmake at build time
    # and fails on molab.  The fix: force a lower-bound version high
    # enough that every release uv could pick already ships a cp313 /
    # abi3 / py3-none wheel on PyPI.
    #
    # Floors were chosen by direct ``pypi.org/pypi/<pkg>/json`` lookups
    # (research-team run 20260523-211220Z-molab-py-pin):
    #   - vllm>=0.11.0       first release with ``requires_python <3.14``
    #                        and a torch transitive that has cp313 wheels.
    #                        0.5.x–0.10.x all hard-pin torch versions
    #                        without cp313 wheels.
    #   - xformers>=0.0.33   first ``cp39-abi3-manylinux_2_28`` wheel
    #                        after the 0.0.32 no-wheel gap.
    #   - bitsandbytes>=0.43.0  first ``py3-none-manylinux_2_24`` wheel
    #                        compiled against modern CUDA.
    #   - triton>=3.2.0      first ``cp313-cp313-manylinux_2_17`` wheel.
    #   - transformers>=4.56.0  vllm 0.11.0+ depends on this; without
    #                        the bump the resolver deadlocks.  The
    #                        canonical install cell already runs
    #                        ``!pip install transformers==4.56.2`` on
    #                        Colab, so 4.56.x is known-working for the
    #                        rest of the Unsloth / trl stack.
    #
    # The overlay REPLACES whatever was chosen (typically a bare name or
    # an outdated ``==`` pin like ``transformers==4.55.4``).  It is
    # skipped only when the source carries a direct ``git+`` / ``@``
    # reference — same convention as ``_GIT_LATEST``.  Packages absent
    # from ``chosen`` are not added; this overlay only tightens what's
    # already there.
    _MOLAB_FLOORS: dict[str, str] = {
        "vllm": "vllm>=0.11.0",
        "xformers": "xformers>=0.0.33",
        "bitsandbytes": "bitsandbytes>=0.43.0",
        "triton": "triton>=3.2.0",
        "transformers": "transformers>=4.56.0",
    }
    for key, floor_spec in _MOLAB_FLOORS.items():
        spec = chosen.get(key)
        if spec is None:
            continue
        if "@" in spec or "git+" in spec:
            continue
        chosen[key] = floor_spec

    # Transitive-dep workaround: cuda-tile is pulled in by flashinfer-python
    # (a vllm transitive dep) with no version constraint.  On molab the
    # latest release (cuda-tile==1.3.0) trips a uv simple-index hash-cache
    # bug — uv downloads the cp313 wheel but compares against an expected
    # list containing only the cp312 wheel hashes (including the Windows
    # one), so the integrity check fails.  Pin to 1.2.0, uploaded ~6 weeks
    # earlier and settled in molab's index cache, to dodge the bug.  Only
    # applied to notebooks that pull vllm, so the 122 non-vllm notebooks
    # are not affected.
    if "vllm" in chosen:
        chosen["cuda-tile"] = "cuda-tile==1.2.0"

    # Materialise the PEP 723 dependency list.
    for key in sorted(chosen):
        spec = chosen[key]
        plan.dependencies.append(spec)
        plan.represented[key] = spec

    # ``marimo`` itself is required at runtime by every generated notebook
    # (``import marimo as mo``).  It is not in the source install cell because
    # the source notebooks are Jupyter — so it is added here, explicitly, and
    # NOT counted as a "dropped" source line.  It is the one molab-runtime
    # addition; everything else is preserved from the notebook's own pins.
    if not any(_normalise_pkg(_RE_PKG_NAME.match(d).group(1)) == "marimo"
               for d in plan.dependencies if _RE_PKG_NAME.match(d)):
        plan.dependencies.append("marimo")
        plan.dependencies.sort()
        plan.represented["marimo"] = "marimo"

    return plan


def _spec_rank(spec: str) -> tuple[int, int]:
    """Rank a package spec so a pinned/constrained spec beats a bare name.

    Returns ``(tier, length)``; higher is preferred.  Mirrors the tie-break
    intent of ``update_all_notebooks._install_spec_preference`` without its
    AMD-specific git-URL handling (P0 molab notebooks have no git specs).
    """
    if "==" in spec:
        return (3, len(spec))
    if any(op in spec for op in (">=", "<=", "~=", ">", "<", "!=")):
        return (2, len(spec))
    return (1, len(spec))


# ===========================================================================
# PEP 723 emission
# ===========================================================================
def render_pep723_header(plan: DependencyPlan) -> str:
    """Render a PEP 723 ``# /// script`` inline-metadata block from a plan.

    The output is a deterministic, comment-prefixed block ready to be placed
    at the top of a marimo ``.py`` notebook so ``marimo --sandbox`` / ``uv
    run`` install the dependencies into an isolated environment.

    Shape (per ``inlining_dependencies.md``)::

        # /// script
        # requires-python = ">=3.10"
        # dependencies = [
        #     "marimo",
        #     "transformers==4.56.2",
        # ]
        # ///

    Returns the block WITHOUT a trailing newline; the generator joins it to
    the rest of the file.
    """
    lines = ["# /// script"]
    lines.append(f'# requires-python = "{plan.requires_python}"')
    if plan.dependencies:
        lines.append("# dependencies = [")
        for dep in plan.dependencies:
            # json.dumps gives a correctly-escaped double-quoted string.
            lines.append(f"#     {json.dumps(dep)},")
        lines.append("# ]")
    else:
        lines.append("# dependencies = []")
    # ``[tool.uv]`` settings honoured by ``uv pip install`` / ``uv run`` via
    # PEP 723 inline metadata.  ``no-build-package`` forces uv to use a
    # prebuilt wheel for the listed packages; if none exists for the active
    # Python the resolver fails cleanly instead of silently walking back to
    # an ancient release that needs CUDA_HOME / cmake at build time.
    #
    # Paired with the ``_MOLAB_FLOORS`` overlay above, the floor pins drive
    # the resolver to a wheel-having release; this stanza is the safety net
    # that turns "silently builds from source" into "fail loudly" if a future
    # release ever ships without a Python 3.13 wheel.
    #
    # ``sentencepiece`` is intentionally absent: its 0.2.1 release ships a
    # ``cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64`` wheel, so
    # the guard was masking a working install path.  (Verified via
    # ``pypi.org/pypi/sentencepiece/0.2.1/json`` in research-team run
    # 20260523-211220Z-molab-py-pin.)
    lines.append("#")
    lines.append("# [tool.uv]")
    lines.append("# no-build-package = [")
    for name in (
        "bitsandbytes",
        "triton",
        "vllm",
        "xformers",
    ):
        lines.append(f'#     "{name}",')
    lines.append("# ]")
    lines.append("# ///")
    return "\n".join(lines)


# ===========================================================================
# Dropped-dependency registry (consumed by test_molab_dependency_parity)
# ===========================================================================
def get_dropped_deps(nb) -> dict[str, str]:
    """Return ``{package: reason}`` for every REAL package the planner saw in
    a notebook's install cell but intentionally left out of the PEP 723 block.

    ``nb`` is a ``molab_manifest.MolabNotebook``.  This is the machine-readable
    drop registry ``tests/test_molab_dependency_parity.py`` consults: a real
    package absent from the generated file's PEP 723 block is acceptable iff
    it appears here with a reason.

    Only genuine package tokens are returned — e.g. an ``{xformers}``
    runtime-templated spec (resolved transitively via ``unsloth``).  Prose
    words, shell keywords, and Python keywords are NEVER returned: the planner
    parses package tokens from real ``pip install`` command lines only (via
    ``shlex`` with comment stripping), so non-package text never enters the
    dependency set in the first place — there is nothing to "drop".

    The result is keyed by the *normalised* package name (lowercase,
    ``-``/``.`` -> ``_``) to match the parity test's own normalisation.
    """
    nb_path = resolve_nb_source(nb.source)
    plan = plan_dependencies(nb_path)

    dropped: dict[str, str] = {}
    for item in plan.dropped:
        # plan.dropped records whole install/setup LINES as well as templated
        # specs. Only the templated-spec entries name a real package; the
        # scaffolding-line entries (``import os, re``, ``if "COLAB_" ...``)
        # are not packages and must not be reported as dropped dependencies.
        token = item.text.strip().lstrip("{").rstrip("}")
        m = _RE_PKG_NAME.match(token)
        if not m:
            continue
        key = _norm_token(m.group(1))
        # A token is only a "dropped dependency" if it is a plausible package
        # name, not a Python/shell keyword that happens to start a line.
        if key in _NOT_A_PACKAGE:
            continue
        dropped[key] = item.reason
    return dropped


# Python / shell keywords and connectives that can appear as the first token
# of an install-cell line but are never package names.  Used to keep
# get_dropped_deps from reporting scaffolding tokens as dropped dependencies.
_NOT_A_PACKAGE: frozenset[str] = frozenset({
    "if", "else", "elif", "for", "while", "try", "except", "finally", "with",
    "def", "class", "return", "import", "from", "as", "and", "or", "not",
    "in", "is", "pass", "raise", "assert", "del", "global", "nonlocal",
})


def _norm_token(token: str) -> str:
    """Normalise a token to the parity test's convention (``-``/``.`` ->
    ``_``, lowercase)."""
    return re.sub(r"[-.]+", "_", token.strip().lower())


# ===========================================================================
# CLI / debug entry point
# ===========================================================================
def _describe(nb_path: Path) -> str:
    """Human-readable summary of a notebook's dependency plan (for debugging
    and the ``__main__`` smoke path)."""
    plan = plan_dependencies(nb_path)
    out = [f"# {nb_path.name}", "", render_pep723_header(plan), ""]
    if plan.setup_lines:
        out.append("setup cell lines:")
        out.extend(f"  {line}" for line in plan.setup_lines)
        out.append("")
    out.append(f"dropped ({len(plan.dropped)}):")
    for item in plan.dropped:
        out.append(f"  - {item.text!r}")
        out.append(f"    reason: {item.reason}")
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover - manual debug aid
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import molab_manifest

    for nb in molab_manifest.get_p0_notebooks():
        print(_describe(resolve_nb_source(nb.source)))
        print()
