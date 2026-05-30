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

"""Generation parity gate (AC2 — broad-catalog scope).

Parametrises over two axes:

1. MANIFEST COVERAGE — all active manifest entries:
   Verifies every non-skipped manifest entry has a committed ``molab/<name>.py``
   OR is explicitly marked skip=True with a notes reason explaining the gap.
   Best-effort: a missing file for a skip=True entry is not a failure.

2. GENERATED FILES ON DISK — glob ``molab/*.py``:
   For every file actually committed under molab/, verifies:
   - It parses as valid Python.
   - Re-running the generator produces byte-identical output (R-drift).

Tests are STATIC (no torch / unsloth). The generator itself must not
require GPU presence to run.

If ``molab_generate`` is not yet importable (generator not yet landed), the
byte-identical regeneration test skips gracefully.

generation_status.json
----------------------
The generator writes ``molab/generation_status.json`` after each run:

    {
        "<notebook_stem>": "ok",
        "<notebook_stem>": "failed: <reason>",
        ...
    }

Parametrised tests over generated files use this file to skip notebooks
that the generator recorded as failed (with the recorded reason in the skip
message), so a single bad notebook does not red the whole suite.  When the
file is absent (generator has not yet run), tests that depend on generated
artifacts are hard-red — that is the correct pre-generation baseline.
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_manifest as mm  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture collections
# ---------------------------------------------------------------------------

_ACTIVE = mm.get_active_notebooks()

# Files that actually exist on disk (best-effort — may be a subset of manifest).
_MOLAB_DIR = REPO_ROOT / "molab"
_GENERATED_FILES: list[Path] = (
    sorted(_MOLAB_DIR.glob("*.py")) if _MOLAB_DIR.exists() else []
)

# ---------------------------------------------------------------------------
# generation_status.json helpers (FIX 3 / DA-03 / consolidated in P2 / ARCH-03)
# ---------------------------------------------------------------------------

from tests._molab_test_utils import (  # noqa: E402
    GENERATION_STATUS as _GENERATION_STATUS,
    skip_if_generation_failed as _skip_if_generation_failed,
)


# ---------------------------------------------------------------------------
# Helper: attempt to import the generator; skip if not yet available.
# ---------------------------------------------------------------------------


def _import_generator():
    """Import molab_generate from scripts/; skip calling test if not on disk."""
    generate_path = REPO_ROOT / "scripts" / "molab_generate.py"
    if not generate_path.exists():
        pytest.skip(
            "scripts/molab_generate.py not yet committed; "
            "generation-parity tests will run once generator-implementer lands."
        )
    spec = importlib.util.spec_from_file_location(
        "molab_generate", str(generate_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper: look up a manifest entry for a generated file (by stem).
# ---------------------------------------------------------------------------

_STEM_TO_NB: dict[str, mm.MolabNotebook] = {
    nb.output.stem: nb for nb in mm.MOLAB_NOTEBOOKS
}


# ---------------------------------------------------------------------------
# Test 1: manifest coverage (best-effort — skip entries are acceptable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nb", _ACTIVE, ids=lambda nb: nb.source.stem)
def test_committed_molab_file_exists(nb: mm.MolabNotebook) -> None:
    """Every non-skipped active manifest entry must have a committed molab/*.py,
    OR the entry must be explicitly marked skip=True with a notes reason (AC2).

    Best-effort: a notebook that failed to convert should have skip=True + notes
    explaining the failure.  The generator must NOT silently omit files.
    """
    if nb.skip:
        # Entry is explicitly marked as skipped — notes must explain why.
        assert nb.notes.strip(), (
            f"MANIFEST: {nb.source.stem} is skip=True but has no notes. "
            "Document the skip reason in the notes field of MOLAB_NOTEBOOKS."
        )
        if not nb.output.exists():
            pytest.xfail(
                f"{nb.source.stem} is skipped (notes: {nb.notes!r}); "
                "no generated file expected."
            )
        return

    if not nb.output.exists():
        # An active (skip=False) manifest entry MUST have a committed file.
        # The previous behaviour skipped on "failed: ..." status in
        # generation_status.json, but that silently masks new genuine
        # breakage — the only acceptable way to silence a missing file is
        # to mark skip=True in MOLAB_NOTEBOOKS with a notes reason.
        # A "failed:" status without an explicit manifest skip is a HARD
        # failure (review finding #5).
        stem = nb.output.stem
        status = _GENERATION_STATUS.get(stem)
        if status is not None and status != "ok":
            reason = status if status.startswith("failed:") else f"failed: {status}"
            pytest.fail(
                f"GENERATION FAILED: {nb.source.stem} -- {reason}\n"
                "An active (skip=False) manifest entry recorded a generation "
                "failure.  Either fix the source notebook so it converts, or "
                "add an explicit entry in scripts/molab_manifest.py "
                "(_SKIP_STEMS) with a notes reason -- silently failed generation "
                "must not pass CI."
            )
        pytest.fail(
            f"GENERATION MISSING: {nb.output} does not exist and the manifest "
            f"entry is not marked skip=True.\n"
            f"Run the generator (python scripts/molab_generate.py) and commit "
            f"the output,\n"
            f"or add the stem to _SKIP_STEMS in scripts/molab_manifest.py with "
            f"a notes reason if this notebook cannot be converted."
        )


# ---------------------------------------------------------------------------
# Test 2: committed files parse as valid Python
# Parametrised over files actually on disk (not manifest) — tolerates partial gen.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_molab_file_parses_as_python(py_file: Path) -> None:
    """Each committed molab/*.py must be syntactically valid Python (AC2)."""
    _skip_if_generation_failed(py_file)
    source_text = py_file.read_text(encoding="utf-8")
    try:
        ast.parse(source_text, filename=str(py_file))
    except SyntaxError as exc:
        pytest.fail(
            f"PARSE ERROR: {py_file.name} is not valid Python.\n"
            f"SyntaxError at line {exc.lineno}: {exc.msg}\n"
            "Fix the generator so molab/*.py files are always valid Python."
        )


# ---------------------------------------------------------------------------
# Test 3: no generated file is an orphan (not in manifest)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_generated_file_is_in_manifest(py_file: Path) -> None:
    """Every molab/*.py on disk must correspond to a manifest entry (AC1 / AC2).

    Orphan files not referenced by the manifest indicate either a stale
    generated file or a manifest that was not updated after generation."""
    _skip_if_generation_failed(py_file)
    nb = _STEM_TO_NB.get(py_file.stem)
    assert nb is not None, (
        f"ORPHAN FILE: {py_file.name} exists in molab/ but has no matching entry "
        "in MOLAB_NOTEBOOKS (matched by output.stem == file.stem). "
        "Either add a manifest entry or remove the file."
    )


# ---------------------------------------------------------------------------
# Test 4: byte-identical regeneration (determinism / R-drift)
# Parametrised over files actually on disk.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_generation_is_deterministic(py_file: Path) -> None:
    """Re-running the generator must produce byte-identical output (AC2, R-drift).

    Strategy: call molab_generate.generate_notebook(nb) -> str and compare
    with the committed file's text.

    If the generator module or function is not yet present, the test is skipped.
    """
    _skip_if_generation_failed(py_file)

    nb = _STEM_TO_NB.get(py_file.stem)
    if nb is None:
        pytest.skip(f"{py_file.stem} has no manifest entry; orphan test covers it.")

    gen = _import_generator()

    if not hasattr(gen, "generate_notebook"):
        pytest.skip(
            "molab_generate.generate_notebook not available; "
            "determinism check deferred until generator API is finalised."
        )

    committed_text = py_file.read_text(encoding="utf-8")
    try:
        regenerated_text = gen.generate_notebook(nb)
    except FileNotFoundError as exc:
        # Source notebook missing is a setup gap, not a drift bug.
        pytest.skip(f"source notebook missing for {py_file.stem}: {exc}")
    except Exception as exc:
        # Any other exception is a real generator bug. Skipping here would
        # silently disable the R-drift guard exactly when it should fire,
        # so fail loudly instead (review B1 / TC-01).
        pytest.fail(
            f"generate_notebook raised {type(exc).__name__}: {exc}. "
            "R-drift guard cannot run; fix the generator before merging."
        )

    if regenerated_text is None:
        pytest.skip(
            "generate_notebook returned None (disk-write API?). "
            "Determinism check requires a str return value. "
            "Update generate_notebook to return the rendered source string."
        )

    if committed_text != regenerated_text:
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                committed_text.splitlines(keepends=True),
                regenerated_text.splitlines(keepends=True),
                fromfile=f"committed:{py_file.name}",
                tofile="regenerated",
                n=4,
            )
        )
        diff_excerpt = "".join(diff_lines[:60])
        pytest.fail(
            f"DRIFT DETECTED: regenerating {py_file.name} produces a diff.\n"
            "The generator is non-deterministic or the committed file is stale.\n"
            "Re-run: python scripts/molab_generate.py and commit the result.\n\n"
            + textwrap.indent(diff_excerpt, "  ")
        )
