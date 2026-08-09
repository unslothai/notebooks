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
install cell introduced by its own markdown heading was invisible. Qwen3_5_MoE
and Qwen3_6_MoE put a CUDA-wheel resolver behind "### Install
flash-linear-attention and causal-conv-1d", it survived into the AMD variant,
and `_assert_amd_install_runtime` refused the whole `--amd` run -- blocking all
153 AMD notebooks.

test_notebook_amd_install_parity.py cannot catch this: it reads the committed
`nb/AMD-*.ipynb`, stale but valid. The bad content appears only on regeneration.
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
    """Only a one-line heading is stepped over; prose ends the install."""
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
    """Stepping over any heading let the cell behind it qualify on that
    heading alone, and the caller deletes what this returns: `### Start Unsloth
    Studio` collected the Studio launch code."""
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
    """`_is_install_like_cell` answers yes on the markdown above a cell, so an
    install-shaped heading over ordinary code qualified it and both got deleted.
    After a heading the cell must carry the evidence itself."""
    cells = [
        _md("### Installation"),
        _code("%%capture\n!pip install unsloth\n"),
        _md("### Install the trainer"),
        _code("trainer = SFTTrainer(model = model, args = config)\n"),
    ]
    assert gen._adjacent_install_like_code_cells(cells, 1) == []


def test_the_first_cell_still_uses_the_wider_test(gen):
    """Nothing stepped over yet, so the section heading above the canonical
    install block is the signal that identifies it."""
    cells = [
        _md("### Installation"),
        _code("%%capture\nimport os\nfrom setup import bootstrap\n"),
    ]
    assert gen._is_install_like_cell(cells, 1, _cell_text(cells[1]))


def _cell_text(cell):
    return "".join(cell["source"])


# --- the exported script's install block, at whatever heading depth ----------


_SCRIPT_BODY = (
    "#!/usr/bin/env python\n"
    "# coding: utf-8\n"
    "\n"
    "{heading}\n"
    "# We'll be using Unsloth to do RL.\n"
    "\n"
    "# In[ ]:\n"
    "\n"
    "\n"
    "get_ipython().system('uv pip install --system -qqq unsloth')\n"
    "\n"
    "\n"
    "# ### Unsloth\n"
    "\n"
    "from unsloth import FastLanguageModel\n"
)


@pytest.mark.parametrize(
    "heading",
    ["# ### Installation", "# ## Installation", "# # Installation"],
)
def test_the_install_block_is_commented_at_any_heading_depth(gen, heading):
    """A template exports `# ### Installation`, a hand-maintained `nb/` source
    `# # Installation`. Matching only the first left the AMD script with live
    `get_ipython()` calls that raise NameError under plain python."""
    out = gen.remove_unwanted_section(_SCRIPT_BODY.format(heading = heading))
    live = [line for line in out.splitlines() if line.startswith("get_ipython")]
    assert live == [], f"install calls still executable under {heading!r}: {live}"
    assert "# get_ipython().system(" in out, "the call was dropped rather than commented"


def test_the_code_after_the_unsloth_heading_is_untouched(gen):
    """Commenting past the end marker would silently disable the script."""
    out = gen.remove_unwanted_section(_SCRIPT_BODY.format(heading = "# # Installation"))
    assert "\nfrom unsloth import FastLanguageModel\n" in out


def test_a_script_with_no_install_section_is_returned_unchanged(gen):
    """No markers, so it must not comment the whole file."""
    body = "#!/usr/bin/env python\nfrom unsloth import FastLanguageModel\n"
    assert gen.remove_unwanted_section(body) == body


# An intro heading, as `Falcon_H1_(0.5B)-Alpaca` opens: a level-1 `# Unsloth`
# title well above the install block.
_SCRIPT_WITH_INTRO = (
    "# # Unsloth\n"
    "#\n"
    "# Visit our docs for model uploads and notebooks.\n"
    "\n"
) + _SCRIPT_BODY


def test_an_intro_unsloth_heading_is_not_mistaken_for_the_end_marker(gen):
    """Searched from the Installation heading onwards. From the top it lands on
    the intro title, before the start, so the range is discarded and every
    install call stays live -- which the old three-hash literal dodged by luck."""
    out = gen.remove_unwanted_section(_SCRIPT_WITH_INTRO.format(heading = "# ### Installation"))
    live = [line for line in out.splitlines() if line.startswith("get_ipython")]
    assert live == [], f"intro heading swallowed the section: {live}"
    assert out.startswith("# # Unsloth\n"), "the intro itself must not be commented"


def test_the_intro_case_holds_at_every_heading_depth(gen):
    for heading in ["# ### Installation", "# ## Installation", "# # Installation"]:
        out = gen.remove_unwanted_section(_SCRIPT_WITH_INTRO.format(heading = heading))
        live = [line for line in out.splitlines() if line.startswith("get_ipython")]
        assert live == [], f"{heading!r} left live calls beneath an intro heading: {live}"


def test_a_later_unsloth_heading_is_still_the_end_marker(gen):
    """Searching forward must not skip past the real terminator."""
    out = gen.remove_unwanted_section(_SCRIPT_WITH_INTRO.format(heading = "# # Installation"))
    assert "\nfrom unsloth import FastLanguageModel\n" in out
    assert out.count("# ### Unsloth") == 1, "the end marker was commented or duplicated"
