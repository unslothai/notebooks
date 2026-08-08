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

"""An AMD variant must not be minted from an abandoned template.

For `DONT_UPDATE_EXCEPTIONS`, `nb/` is the source of truth and the
`original_template/` copy is left behind. `Advanced_Llama3_1_(3B)_GRPO_LoRA`
had drifted to `weight_decay = 0.1` under `nb/` while the template still said
`0.001`, so the AMD reader was handed hyperparameters nobody chose. Three of
the eleven exceptions have both a stale template and an AMD variant.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"
TEMPLATE_DIR = REPO_ROOT / "original_template"


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


def _code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", []))
            for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]


def _hand_maintained_with_amd(gen):
    for name in gen.DONT_UPDATE_EXCEPTIONS:
        if (NB_DIR / f"AMD-{name}").is_file() and (NB_DIR / name).is_file():
            yield name


def test_the_drifted_pair_is_still_present(gen):
    """If the template is ever refreshed this test is the one that says so,
    rather than the check below quietly passing for the wrong reason."""
    drifted = [name for name in _hand_maintained_with_amd(gen)
               if (TEMPLATE_DIR / name).is_file()
               and _code_cells(TEMPLATE_DIR / name) != _code_cells(NB_DIR / name)]
    assert drifted, "no exception notebook drifts from its template any more"


def test_an_amd_variant_follows_nb_not_the_template(gen):
    """Every code cell of the AMD variant that is not install machinery has to
    come from `nb/`. The install cell is rewritten for ROCm by design."""
    mismatched = []
    for name in _hand_maintained_with_amd(gen):
        if not (TEMPLATE_DIR / name).is_file():
            continue
        template_only = set(_code_cells(TEMPLATE_DIR / name)) - set(_code_cells(NB_DIR / name))
        amd = set(_code_cells(NB_DIR / f"AMD-{name}"))
        leaked = template_only & amd
        if leaked:
            mismatched.append((name, sorted(leaked)[0][:120]))
    assert not mismatched, (
        f"AMD variants carry cells that exist only in the abandoned template: "
        f"{mismatched}"
    )


_COLAB_BADGE = (
    '<a href="https://colab.research.google.com/github/unslothai/notebooks/'
    'blob/main/nb/X.ipynb" target="_parent">'
    '<img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>'
)


def test_a_bare_colab_badge_counts_as_a_stale_announcement(gen):
    """It carries no "to run this, press", so it walked past the check and the
    AMD variant kept a button pointing at the CUDA notebook."""
    assert gen._is_stale_amd_announcement(_COLAB_BADGE)


@pytest.mark.parametrize("text", [
    '<a href="https://unsloth.ai/"><img src="logo.png"/></a>',
    '<a href="https://colab.research.google.com/x">run it</a>\n\nThen train.',
    "# Goal: solve Sudoku with reinforcement learning",
])
def test_real_content_is_not_mistaken_for_a_badge(gen, text):
    assert not gen._is_stale_amd_announcement(text)


@pytest.mark.parametrize("path", sorted(NB_DIR.glob("AMD-*.ipynb")), ids=lambda p: p.name)
def test_every_amd_notebook_opens_on_dev_cloud(path):
    """Not one may open with a Colab button, whatever its source."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first = next((c for c in notebook.get("cells", [])
                  if c.get("cell_type") == "markdown"), None)
    if first is None:
        pytest.skip("no markdown cell")
    text = "".join(first.get("source", []))
    assert "colab.research.google.com/github" not in text, (
        f"{path.name} opens with a Colab badge pointing at the CUDA notebook"
    )
