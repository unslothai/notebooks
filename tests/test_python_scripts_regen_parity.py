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

"""Stronger gate than per-cell AST: run update_all_notebooks.py's
``convert_notebook_to_script`` on every ``nb/*.ipynb`` and assert:

  1. Every checked-in ``python_scripts/<name>.py`` ast.parses as a
     complete module. Catches drift the per-cell test misses
     (cross-cell forward references, top-level statement order, etc.).

  2. The live conversion of ``nb/<name>.ipynb`` is byte-identical to
     the committed ``python_scripts/<name>.py``. Catches the failure
     mode where someone edits an .ipynb but forgets to regenerate the
     .py mirror (or vice-versa).

  3. The two trees hold the same set of names, in BOTH directions.
     Check 2 is parametrized over ``nb/``, so it can only see a
     notebook that is missing its mirror; a mirror whose notebook was
     deleted or renamed is invisible to it and keeps shipping to users
     as a script with no source. Nothing covered that until a
     hand-uploaded notebook landed in ``nb/`` with no mirror and sat on
     main for four days, which prompted looking at the other side too.

This is exactly what the generator does at regen time -- we just lift
it into a pytest gate so per-PR CI fails red if the two trees diverge.

Cost: nbconvert's PythonExporter runs in ~50ms per notebook on
ubuntu-latest; the full 422-notebook sweep is ~30 sec sequential and
much less with pytest-xdist. Cheap enough to gate every PR.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


def _load_generator():
    """Load update_all_notebooks.py by file path so the test does not
    depend on the repo root being on sys.path."""
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location("_update_all_notebooks", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_update_all_notebooks"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()


_PYTHON_SCRIPTS = sorted((REPO_ROOT / "python_scripts").glob("*.py"))
_NB_NOTEBOOKS = sorted((REPO_ROOT / "nb").glob("*.ipynb"))

# Notebooks whose committed python_scripts/<name>.py mirror is known to
# diverge from the live conversion of nb/<name>.ipynb at PR-open time.
# Each entry is a real drift: someone edited the .ipynb without running
# `python update_all_notebooks.py --to_main_repo` to refresh the .py
# mirror. Tracked as follow-up; remove the entry once the notebook is
# regenerated.
_KNOWN_REGEN_MIRRORS_STALE: set[str] = {
    "AMD-Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb",
    "Qwen3_5_MoE.ipynb",
    "Qwen3_6_MoE.ipynb",
}


@pytest.mark.parametrize("script_path", _PYTHON_SCRIPTS, ids=lambda p: p.name)
def test_python_script_ast_parses(script_path: Path) -> None:
    """Every committed ``python_scripts/*.py`` must ast.parse as a
    complete module. The per-cell test in
    test_notebook_json_validity.py parses each code cell in isolation;
    this one catches drift that only surfaces when the cells are
    concatenated (e.g. a cell that defines a name used by a previous
    cell, top-level statement order, syntax that crosses cell
    boundaries)."""
    src = script_path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        pytest.fail(
            f"DRIFT DETECTED: {script_path.relative_to(REPO_ROOT)} fails "
            f"ast.parse as a complete module at line {exc.lineno}: "
            f"{exc.msg}. Regenerate via "
            f"`python update_all_notebooks.py --to_main_repo`."
        )


@pytest.mark.parametrize("nb_path", _NB_NOTEBOOKS, ids=lambda p: p.name)
def test_live_conversion_matches_committed_script(nb_path: Path, tmp_path: Path) -> None:
    """Convert ``nb/<name>.ipynb`` via the generator's own
    ``convert_notebook_to_script`` (nbconvert PythonExporter +
    remove_unwanted_section post-pass) and assert the result is
    byte-identical to the committed ``python_scripts/<name>.py``."""
    expected = REPO_ROOT / "python_scripts" / (nb_path.stem + ".py")
    if not expected.is_file():
        pytest.fail(
            f"DRIFT DETECTED: {nb_path.relative_to(REPO_ROOT)} has no "
            f"matching python_scripts/{nb_path.stem}.py mirror. "
            f"Regenerate via `python update_all_notebooks.py --to_main_repo`."
        )

    fresh = tmp_path / (nb_path.stem + ".py")
    try:
        _GEN.convert_notebook_to_script(str(nb_path), str(fresh))
    except Exception as exc:
        pytest.fail(
            f"DRIFT DETECTED: convert_notebook_to_script crashed on "
            f"{nb_path.relative_to(REPO_ROOT)}: {exc!r}. The generator "
            f"itself cannot regenerate this notebook's .py mirror."
        )

    actual_bytes = fresh.read_bytes()
    expected_bytes = expected.read_bytes()
    if actual_bytes == expected_bytes:
        return

    if nb_path.name in _KNOWN_REGEN_MIRRORS_STALE:
        pytest.xfail(
            f"{nb_path.name} has a stale python_scripts/ mirror pending "
            f"`python update_all_notebooks.py --to_main_repo`."
        )

    import difflib

    a = expected_bytes.decode("utf-8", errors="replace").splitlines()
    b = actual_bytes.decode("utf-8", errors="replace").splitlines()
    diff = list(
        difflib.unified_diff(
            a, b,
            fromfile=str(expected.relative_to(REPO_ROOT)) + " (committed)",
            tofile=str(nb_path.relative_to(REPO_ROOT)) + " (live convert)",
            n=2,
        )
    )
    snippet = "\n".join(diff[:60])
    pytest.fail(
        f"DRIFT DETECTED: live conversion of "
        f"{nb_path.relative_to(REPO_ROOT)} no longer matches the "
        f"committed python_scripts mirror. Either regenerate via "
        f"`python update_all_notebooks.py --to_main_repo` to refresh "
        f"the .py mirror, or revert the unintended change to the "
        f".ipynb. First diff lines:\n{snippet}"
    )


def test_no_python_script_without_its_notebook() -> None:
    """A ``python_scripts/*.py`` whose ``nb/*.ipynb`` is gone must fail.

    The parity test above walks ``nb/``, so a mirror left behind by a deleted or
    renamed notebook is invisible to it: the script keeps shipping, is never
    regenerated again, and silently drifts from a source that no longer exists.
    Both trees are generated from the same names, so the sets are equal or
    something was moved by hand.
    """
    notebooks = {p.stem for p in (REPO_ROOT / "nb").glob("*.ipynb")}
    scripts = {p.stem for p in (REPO_ROOT / "python_scripts").glob("*.py")}
    assert notebooks, "no nb/*.ipynb found -- the glob or the layout changed"

    stray = sorted(scripts - notebooks)
    assert not stray, (
        "DRIFT DETECTED: python_scripts/ holds mirrors whose nb/ notebook does not "
        "exist. Delete them, or restore the notebook they were generated from: "
        f"{stray}"
    )


def test_no_notebook_without_its_python_script() -> None:
    """The forward direction, as one assertion naming every offender at once.

    The parametrized test reports these one notebook per test id, which is the
    right granularity when a mirror has drifted. When a batch is uploaded by
    hand it is easier to act on the whole list, and this states the invariant
    the two trees are supposed to satisfy in one place.
    """
    notebooks = {p.stem for p in (REPO_ROOT / "nb").glob("*.ipynb")}
    scripts = {p.stem for p in (REPO_ROOT / "python_scripts").glob("*.py")}

    missing = sorted(notebooks - scripts)
    assert not missing, (
        "DRIFT DETECTED: nb/ notebooks have no python_scripts/ mirror. Regenerate "
        "via `python update_all_notebooks.py --to_main_repo`: "
        f"{missing}"
    )
