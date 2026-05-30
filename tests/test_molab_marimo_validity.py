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

"""Marimo-validity gate (AC2 / AC4 — broad-catalog scope).

Parametrised over molab/*.py files that ACTUALLY EXIST ON DISK so that a
partial generation run (best-effort) does not red the whole suite.

For every generated molab/*.py file:
  1. Source text contains the required marimo structural markers:
       - ``import marimo``
       - ``marimo.App(``
       - ``@app.cell``
       - ``app.run()``
  2. ``app.run()`` appears inside an ``if __name__ == '__main__':`` guard.
  3. If the ``marimo`` CLI is on PATH, run ``marimo check <file>`` and assert
     it exits 0.  If the CLI is absent, that sub-test is skipped (gracefully).

No torch / unsloth / marimo is imported at module load time.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_manifest as mm  # noqa: E402  # used for ids only (display_name)

# ---------------------------------------------------------------------------
# Parametrise over files on disk — tolerates partial generation.
# ---------------------------------------------------------------------------

_MOLAB_DIR = REPO_ROOT / "molab"
_GENERATED_FILES: list[Path] = (
    sorted(_MOLAB_DIR.glob("*.py")) if _MOLAB_DIR.exists() else []
)

# If molab/ doesn't exist yet, fall back to the active manifest entries
# so the parametrise list is non-empty (all tests will skip individually).
if not _GENERATED_FILES:
    _GENERATED_FILES = [nb.output for nb in mm.get_active_notebooks()]


# ---------------------------------------------------------------------------
# generation_status.json skip helper (FIX 3 / DA-03 / consolidated in P2 / ARCH-03)
# ---------------------------------------------------------------------------

from tests._molab_test_utils import (  # noqa: E402
    GENERATION_STATUS as _GENERATION_STATUS,
    skip_if_generation_failed as _skip_if_generation_failed,
)


# ---------------------------------------------------------------------------
# Required marimo structural markers
# ---------------------------------------------------------------------------

_MARIMO_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("import marimo", re.compile(r"\bimport marimo\b")),
    ("marimo.App(", re.compile(r"\bmarimo\.App\s*\(")),
    ("@app.cell", re.compile(r"@app\.cell")),
    ("app.run()", re.compile(r"\bapp\.run\s*\(\s*\)")),
]


# ---------------------------------------------------------------------------
# Structural-marker tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_marimo_import_marker(py_file: Path) -> None:
    """Generated file must contain ``import marimo`` (AC2)."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)
    text = py_file.read_text(encoding="utf-8")
    _, pattern = _MARIMO_MARKERS[0]
    assert pattern.search(text), (
        f"MARKER MISSING: {py_file.name} does not contain 'import marimo'. "
        "All generated molab files must be valid marimo notebooks."
    )


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_marimo_app_instantiation_marker(py_file: Path) -> None:
    """Generated file must contain ``marimo.App(`` (AC2)."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)
    text = py_file.read_text(encoding="utf-8")
    _, pattern = _MARIMO_MARKERS[1]
    assert pattern.search(text), (
        f"MARKER MISSING: {py_file.name} does not contain 'marimo.App('. "
        "All generated molab files must instantiate a marimo App."
    )


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_marimo_app_cell_decorator_marker(py_file: Path) -> None:
    """Generated file must contain at least one ``@app.cell`` decorator (AC2)."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)
    text = py_file.read_text(encoding="utf-8")
    _, pattern = _MARIMO_MARKERS[2]
    assert pattern.search(text), (
        f"MARKER MISSING: {py_file.name} does not contain '@app.cell'. "
        "All generated molab files must define at least one marimo cell."
    )


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_marimo_app_run_marker(py_file: Path) -> None:
    """Generated file must contain ``app.run()`` (AC2)."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)
    text = py_file.read_text(encoding="utf-8")
    _, pattern = _MARIMO_MARKERS[3]
    assert pattern.search(text), (
        f"MARKER MISSING: {py_file.name} does not contain 'app.run()'. "
        "All generated molab files must call app.run() at the bottom."
    )


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_marimo_app_run_in_main_guard(py_file: Path) -> None:
    """``app.run()`` must be inside an ``if __name__ == '__main__':`` guard (AC2).

    This prevents the notebook from auto-running when imported by the
    generator or test suite.
    """
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)
    text = py_file.read_text(encoding="utf-8")
    main_guard = re.compile(
        r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', re.MULTILINE
    )
    run_call = re.compile(r"\bapp\.run\s*\(\s*\)")

    if not main_guard.search(text):
        pytest.fail(
            f"MARKER MISSING: {py_file.name} has no "
            "'if __name__ == \"__main__\":' guard. "
            "app.run() must be guarded to prevent auto-execution on import."
        )

    main_match = main_guard.search(text)
    run_match = run_call.search(text)
    if run_match is None:
        pytest.fail(
            f"MARKER MISSING: {py_file.name} has no 'app.run()' call."
        )
    if run_match.start() < main_match.start():
        pytest.fail(
            f"ORDERING ERROR: {py_file.name} has app.run() BEFORE the "
            "__main__ guard. app.run() must be inside the guard."
        )


# ---------------------------------------------------------------------------
# CLI check: ``marimo check <file>`` (skipped if marimo not on PATH)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_marimo_cli_check(py_file: Path) -> None:
    """Run ``marimo check <file>`` to validate the notebook structure.

    Skipped gracefully when:
    - The ``marimo`` CLI is not on PATH.
    - The generated file does not yet exist.
    - The generator recorded a failure for this notebook.
    """
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)

    marimo_exe = shutil.which("marimo")
    if marimo_exe is None:
        pytest.skip(
            "marimo CLI not found on PATH; "
            "install marimo (pip install marimo) to enable this check."
        )

    try:
        result = subprocess.run(
            [marimo_exe, "check", str(py_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        # `marimo check` is parsing-only and should be near-instant; a 60s
        # wallclock blow-up indicates a hang. Without the timeout one frozen
        # invocation would burn the entire 10-minute CI budget for the
        # 104-notebook parametrise list (review P3 / TC-02).
        pytest.fail(
            f"marimo check TIMED OUT for {py_file.name} after {exc.timeout}s — "
            "the CLI hung. Investigate marimo before re-running."
        )
    if result.returncode != 0:
        pytest.fail(
            f"marimo check FAILED for {py_file.name} "
            f"(exit code {result.returncode}).\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}\n"
            "Fix the generated notebook so it passes marimo check."
        )
