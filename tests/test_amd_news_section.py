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

"""An nb/-sourced AMD variant still gets the News section.

News lives in `original_template/` only, so once the AMD generator started
minting hand-maintained notebooks from `nb/`, `AMD-Ministral_3_(3B)_
Reinforcement_Learning_Sudoku_Game` lost the `### News` heading and the
announcement it had carried. `_is_stale_amd_announcement` never put it back:
it only rewrites the opening Colab announcements.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"
TEMPLATE_DIR = REPO_ROOT / "original_template"

NEWS = "### News"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "update_all_notebooks", REPO_ROOT / "update_all_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_generator()


def _markdown(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", [])).strip()
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]


def _hand_maintained_with_amd_and_template(gen):
    for name in gen.DONT_UPDATE_EXCEPTIONS:
        if (
            (NB_DIR / f"AMD-{name}").is_file()
            and (NB_DIR / name).is_file()
            and (TEMPLATE_DIR / name).is_file()
        ):
            yield name


def test_a_hand_maintained_source_without_news_still_exists(gen):
    """The pair that motivates this file. If every hand-maintained source grows
    a News section the check below would pass for the wrong reason."""
    stranded = [
        name
        for name in _hand_maintained_with_amd_and_template(gen)
        if NEWS in _markdown(TEMPLATE_DIR / name)
        and NEWS not in _markdown(NB_DIR / name)
    ]
    assert stranded, (
        "no hand-maintained nb/ source lacks the News its template carries"
    )


def test_the_amd_variant_keeps_the_news_its_template_carries(gen):
    """Sourcing content from nb/ must not drop generator-owned boilerplate."""
    missing = [
        name
        for name in _hand_maintained_with_amd_and_template(gen)
        if NEWS in _markdown(TEMPLATE_DIR / name)
        and NEWS not in _markdown(NB_DIR / f"AMD-{name}")
    ]
    assert not missing, (
        f"AMD variants lost the News section their template defines: {missing}"
    )


@pytest.mark.parametrize(
    "path", sorted(NB_DIR.glob("AMD-*.ipynb")), ids=lambda p: p.name)
def test_news_appears_once_and_above_installation(path):
    """Two News headings would mean the restore ran on top of an existing one."""
    cells = _markdown(path)
    assert cells.count(NEWS) <= 1, f"{path.name} has {cells.count(NEWS)} News headings"
    if NEWS not in cells:
        pytest.skip("no News section")
    headings = [c.splitlines()[0].lstrip("#").strip().lower() for c in cells if c]
    if not any(h.startswith("install") for h in headings):
        pytest.skip("no installation heading")
    news_at = cells.index(NEWS)
    install_at = next(
        i for i, c in enumerate(cells)
        if c and c.splitlines()[0].lstrip("#").strip().lower().startswith("install")
    )
    assert news_at < install_at, f"{path.name} puts News below Installation"


def _write(path, sources):
    path.write_text(
        json.dumps({
            "cells": [
                {"cell_type": kind, "metadata": {}, "source": [text]}
                for kind, text in sources
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )


def test_restore_inserts_news_above_installation(gen, tmp_path):
    template = tmp_path / "T.ipynb"
    amd = tmp_path / "AMD-T.ipynb"
    _write(template, [("markdown", NEWS), ("markdown", "Placeholder"),
                      ("markdown", "### Installation"), ("code", "!pip install unsloth")])
    _write(amd, [("markdown", "# Goal: train a model"),
                 ("markdown", "# Installation\nWe'll be using Unsloth."),
                 ("code", "!pip install unsloth")])

    assert gen._restore_news_section(amd, str(template), "Studio is out.")

    cells = _markdown(amd)
    assert cells == ["# Goal: train a model", NEWS, "Studio is out.",
                     "# Installation\nWe'll be using Unsloth."]


def test_restore_is_idempotent(gen, tmp_path):
    """The AMD copy is rebuilt from nb/ every run, but a second pass over the
    same file must not stack a second News section on top."""
    template = tmp_path / "T.ipynb"
    amd = tmp_path / "AMD-T.ipynb"
    _write(template, [("markdown", NEWS), ("markdown", "Placeholder"),
                      ("markdown", "### Installation")])
    _write(amd, [("markdown", "# Goal: train a model"),
                 ("markdown", "### Installation")])

    assert gen._restore_news_section(amd, str(template), "Studio is out.")
    first = amd.read_text(encoding="utf-8")
    assert not gen._restore_news_section(amd, str(template), "Studio is out.")
    assert amd.read_text(encoding="utf-8") == first


def test_restore_declines_when_the_template_has_no_news(gen, tmp_path):
    """The nb/-only notebooks have no template to take News from, and none of
    them carries a News section on main. They must stay that way."""
    template = tmp_path / "T.ipynb"
    amd = tmp_path / "AMD-T.ipynb"
    _write(template, [("markdown", "### Installation")])
    _write(amd, [("markdown", "### Installation")])

    assert not gen._restore_news_section(amd, str(template), "Studio is out.")
    assert NEWS not in _markdown(amd)


def test_restore_declines_when_there_is_no_template(gen, tmp_path):
    amd = tmp_path / "AMD-T.ipynb"
    _write(amd, [("markdown", "### Installation")])

    assert not gen._restore_news_section(amd, str(tmp_path / "missing.ipynb"), "x")
    assert NEWS not in _markdown(amd)
