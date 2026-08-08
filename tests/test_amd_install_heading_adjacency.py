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

"""A second install cell behind its own heading must still be AMD-stripped.

`_adjacent_install_like_code_cells` stopped at the first non-code cell, so an
install cell introduced by its own markdown heading was invisible to it.
Qwen3_5_MoE and Qwen3_6_MoE put a CUDA-wheel resolver behind "### Install
flash-linear-attention and causal-conv-1d", so it survived into the AMD variant
and `_assert_amd_install_runtime` then refused the entire `--amd` run -- which
blocked regenerating all 153 AMD notebooks, not just those two.

The existing gate in test_notebook_amd_install_parity.py could not catch this:
it reads the committed `nb/AMD-*.ipynb`, which were stale but valid. The bad
content only appears when the generator actually regenerates them.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "update_all_notebooks", REPO_ROOT / "update_all_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def gen():
    return _load_generator()


def _code(source):
    return {"cell_type": "code", "source": source, "metadata": {}, "outputs": [],
            "execution_count": None}


def _md(source):
    return {"cell_type": "markdown", "source": source, "metadata": {}}


CUDA_INSTALL = '%%capture\n!uv pip install -qqq "torch>=2.10" cu12-wheel\n'


def test_an_install_cell_behind_a_heading_is_collected(gen):
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md("### Install flash-linear-attention and causal-conv-1d"),
        _code(CUDA_INSTALL),
        _md("### Unsloth"),
        _code("from unsloth import FastLanguageModel\n"),
    ]
    found = gen._adjacent_install_like_code_cells(cells, 1)
    indices = [index for index, _text in found]
    assert 3 in indices, "the install cell behind its heading was not collected"
    assert 2 in indices, "the heading was left behind, pointing at a deleted cell"
    assert 5 not in indices, "a non-install code cell was swept up"


def test_the_heading_is_marked_so_it_never_reaches_the_install_text(gen):
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md("### Install extra wheels"),
        _code(CUDA_INSTALL),
    ]
    found = dict(gen._adjacent_install_like_code_cells(cells, 1))
    # None, not the heading text: the AMD recipe is built from these strings.
    assert found[2] is None
    assert found[3] == CUDA_INSTALL


def test_prose_between_cells_still_stops_the_scan(gen):
    """Only a one-line heading is stepped over. Real prose ends the install."""
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md("Now we explain what the model does.\n\nAt length, over lines."),
        _code(CUDA_INSTALL),
    ]
    found = gen._adjacent_install_like_code_cells(cells, 1)
    assert found == [], "prose was treated as an install heading"


def test_a_trailing_heading_with_no_install_after_it_is_not_consumed(gen):
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md("### Unsloth"),
        _code("from unsloth import FastLanguageModel\n"),
    ]
    found = gen._adjacent_install_like_code_cells(cells, 1)
    assert found == [], "a heading was consumed without an install cell behind it"


def test_back_to_back_install_cells_still_work(gen):
    """The original behaviour, with no heading in the way, is unchanged."""
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _code(CUDA_INSTALL),
        _code("from unsloth import FastLanguageModel\n"),
    ]
    found = gen._adjacent_install_like_code_cells(cells, 1)
    assert [index for index, _text in found] == [2]


@pytest.mark.parametrize("heading", [
    "### Setup the model",
    "### Start Unsloth Studio",
    "## Setup",
])
def test_a_non_install_heading_stops_the_scan(gen, heading):
    """Stepping over any heading let the cell behind it qualify as an install
    on the strength of that heading alone, and the caller deletes what this
    returns. `### Start Unsloth Studio` collected the Studio launch code."""
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md(heading),
        _code('import sys\nsys.path.insert(0, "/content/unsloth")\n'
              'from colab import launch_unsloth_studio\n'),
    ]
    assert gen._adjacent_install_like_code_cells(cells, 1) == []


@pytest.mark.parametrize("heading", [
    "### Install flash-linear-attention and causal-conv-1d",
    "### Installation",
    "## 2. Install dependencies",
    "#### install extras",
])
def test_a_dependency_heading_is_still_stepped_over(gen, heading):
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md(heading),
        _code(CUDA_INSTALL),
    ]
    assert [i for i, _t in gen._adjacent_install_like_code_cells(cells, 1)] == [2, 3]


def test_a_heading_cannot_make_ordinary_code_an_install(gen):
    """`_is_install_like_cell` answers yes on the strength of the markdown
    above a cell, so an install-shaped heading over ordinary code qualified it
    and the caller deletes both. After a heading the cell has to carry the
    evidence itself."""
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md("### Install the trainer"),
        _code("trainer = SFTTrainer(model = model, args = config)\n"),
    ]
    assert gen._adjacent_install_like_code_cells(cells, 1) == []


def test_the_first_cell_still_uses_the_wider_test(gen):
    """Nothing was stepped over yet, so the section heading above the canonical
    install block is exactly the signal that identifies it."""
    cells = [
        _md("### Installation"),
        _code("%%capture\nimport os\nfrom setup import bootstrap\n"),
    ]
    assert gen._is_install_like_cell(cells, 1, _cell_text(cells[1]))


def _cell_text(cell):
    return "".join(cell["source"])
