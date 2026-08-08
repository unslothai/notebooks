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

"""The AMD MoE notebooks turn MoE autotuning off, and regeneration keeps it.

`UNSLOTH_MOE_DISABLE_AUTOTUNE=1` makes `unsloth.kernels.moe.autotune_cache`
hand back heuristic configs instead of searching per device capability. Set by
hand on `AMD-Qwen3_5_MoE` / `AMD-Qwen3_6_MoE` and absent from every CUDA
source, it was composed away once the generator started rewriting the
follow-up install cell. It now comes from the generator's own variant extras,
the only place a regeneration cannot lose it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"

GUARD = "UNSLOTH_MOE_DISABLE_AUTOTUNE"


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


def _cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        ("".join(cell.get("source", [])), cell.get("cell_type"))
        for cell in notebook.get("cells", [])
    ]


def _amd_moe_notebooks(gen):
    return [p for p in sorted(NB_DIR.glob("AMD-*.ipynb")) if gen._is_qwen3_moe_path(str(p))]


def test_both_amd_moe_notebooks_are_present(gen):
    assert [p.name for p in _amd_moe_notebooks(gen)] == [
        "AMD-Qwen3_5_MoE.ipynb",
        "AMD-Qwen3_6_MoE.ipynb",
    ]


@pytest.mark.parametrize("name", ["AMD-Qwen3_5_MoE.ipynb", "AMD-Qwen3_6_MoE.ipynb"])
def test_the_guard_is_set_before_unsloth_is_imported(name):
    cells = _cells(NB_DIR / name)
    guard_at = next((i for i, (text, _) in enumerate(cells) if GUARD in text), None)
    assert guard_at is not None, f"{name} no longer sets {GUARD}"
    import_at = next(
        (i for i, (text, kind) in enumerate(cells)
         if kind == "code" and "from unsloth import" in text),
        None,
    )
    assert import_at is not None, f"{name} never imports unsloth"
    assert guard_at < import_at, (
        f"{name} sets {GUARD} at cell {guard_at}, after the unsloth import at "
        f"{import_at}, where autotune_cache has already read the environment"
    )


def test_no_other_notebook_picks_the_guard_up(gen):
    """Two notebooks carry it on main. The Qwen3.5 Vision notebooks sit next to
    them and match a looser family test, so widening the predicate shows here."""
    carriers = sorted(
        path.name for path in NB_DIR.glob("*.ipynb")
        if GUARD in path.read_text(encoding="utf-8")
    )
    assert carriers == ["AMD-Qwen3_5_MoE.ipynb", "AMD-Qwen3_6_MoE.ipynb"]


@pytest.mark.parametrize("path, expected", [
    ("nb/AMD-Qwen3_5_MoE.ipynb", True),
    ("nb/AMD-Qwen3_6_MoE.ipynb", True),
    ("nb/AMD-Qwen3_5_(4B)_Vision_GRPO.ipynb", False),
    ("nb/AMD-Qwen3_5_(2B)_Vision.ipynb", False),
    ("nb/AMD-Qwen3_VL_(8B)-Vision-GRPO.ipynb", False),
])
def test_only_the_moe_pair_selects_the_variant(gen, path, expected):
    assert gen._is_qwen3_moe_path(path) is expected


def test_the_variant_extras_carry_the_guard(gen):
    install, extras = gen._compose_amd_installation("nb/AMD-Qwen3_5_MoE.ipynb", [])
    assert extras is not None and GUARD in extras
    install, extras = gen._compose_amd_installation("nb/AMD-Qwen3_5_(2B)_Vision.ipynb", [])
    assert extras is None or GUARD not in extras
