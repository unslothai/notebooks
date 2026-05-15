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

"""Parametrised gate: every .ipynb under original_template/, nb/, kaggle/
must be valid JSON, every cell must have a known cell_type, and every
non-empty code cell must parse as Python after Jupyter-magic stripping.

Mirrors update_all_notebooks.py::validate_notebook_syntax (lines
1478-1535) but as a HARD pytest gate so a syntax-bad notebook fails CI
red instead of being logged-and-skipped during in-process regeneration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


_ALL_NOTEBOOKS = list(ni.iter_notebooks())


@pytest.mark.parametrize("path", _ALL_NOTEBOOKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_is_valid_json(path: Path) -> None:
    """Loads cleanly with json.loads; nothing else."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            json.load(fh)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"DRIFT DETECTED: {path.relative_to(REPO_ROOT)} is not valid JSON: "
            f"{exc!r}. The notebook is unloadable by jupyter / nbformat / "
            f"update_all_notebooks.py; users will see a 500 on Colab open."
        )


@pytest.mark.parametrize("path", _ALL_NOTEBOOKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_cells_have_known_types(path: Path) -> None:
    """Every cell.cell_type is in {code, markdown}; nothing exotic."""
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    bad: list[tuple[int, str]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        kind = cell.get("cell_type", "")
        if kind not in {"code", "markdown"}:
            bad.append((i, kind))
    if bad:
        pytest.fail(
            f"DRIFT DETECTED: {path.relative_to(REPO_ROOT)} contains unknown "
            f"cell types {bad}; jupyter/nbformat only handles "
            f"code + markdown reliably."
        )


@pytest.mark.parametrize("path", _ALL_NOTEBOOKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_code_cells_parse_as_python(path: Path) -> None:
    """Strip Jupyter magics, then ast.parse every code cell. Mirrors
    update_all_notebooks.py::validate_notebook_syntax."""
    errs: list[tuple[int, int, str]] = []
    for cell_idx, src in ni.iter_code_cells(path):
        exc = ni.parse_cell_or_collect_error(src)
        if exc is not None:
            errs.append((cell_idx, exc.lineno or 0, str(exc)))
    if errs:
        rel = path.relative_to(REPO_ROOT)
        lines = "\n".join(
            f"    cell {i}, line {ln}: {msg}" for i, ln, msg in errs
        )
        pytest.fail(
            f"DRIFT DETECTED: {rel} has {len(errs)} code cell(s) that fail "
            f"to ast.parse after stripping Jupyter magics:\n{lines}"
        )
