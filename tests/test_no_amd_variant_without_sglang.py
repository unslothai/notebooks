# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
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
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""An AMD notebook may not import a package the ROCm recipe cannot install.

sglang is the concrete case: it publishes no ROCm wheel (see
`AMD_SKIP_NOTEBOOKS` in update_all_notebooks.py), and leaving its install out
of the ROCm cell is what produced the shape this file guards.
`AMD-Gemma3N_(2B)-Inference` shipped with no sglang anywhere and a first code
cell reading `from sglang.utils import wait_for_server`, so Run All stopped on
ModuleNotFoundError before reaching any inference. The answer is not to mint
that AMD variant at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "original_template"
NB_DIR = REPO_ROOT / "nb"


def _load_generator():
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location(
        "_update_all_notebooks_amd_skip", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_update_all_notebooks_amd_skip"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "path", sorted(TEMPLATE_DIR.glob("*.ipynb")), ids=lambda p: p.name)
def test_sglang_serving_templates_are_skipped_for_amd(path):
    """A template that imports sglang must be on the AMD skip list."""
    if not _GEN._notebook_imports_sglang(_load(path)):
        return
    assert path.name in _GEN.AMD_SKIP_NOTEBOOKS, (
        f"{path.name} imports sglang, which has no ROCm wheel, so its AMD "
        f"variant would ship an install cell that never installs sglang and a "
        f"code cell that imports it. Add it to AMD_SKIP_NOTEBOOKS in "
        f"update_all_notebooks.py, or give the AMD recipe a real ROCm sglang "
        f"install."
    )


@pytest.mark.parametrize(
    "path", sorted(NB_DIR.glob("AMD-*.ipynb")), ids=lambda p: p.name)
def test_no_shipped_amd_notebook_imports_sglang(path):
    """The artifact, not the intent: nothing under nb/AMD-* may import sglang."""
    assert not _GEN._notebook_imports_sglang(_load(path)), (
        f"{path.name} imports sglang but no ROCm recipe installs it, so the "
        f"notebook stops on ModuleNotFoundError."
    )


def test_the_amd_generator_does_not_mint_a_skipped_notebook(monkeypatch, tmp_path):
    """End to end over the real selection in copy_and_update_amd_notebooks.

    Only the per-notebook rewrite and the file copy are neutralised, so what is
    measured is which notebooks the generator chose, on the real
    original_template/ and nb/ trees.
    """
    minted = []
    monkeypatch.setattr(_GEN, "update_notebook_sections", lambda *a, **k: None)
    monkeypatch.setattr(_GEN, "_cache_original_outputs", lambda *a, **k: None)
    monkeypatch.setattr(_GEN, "_cache_notebook_format", lambda *a, **k: None)
    monkeypatch.setattr(_GEN, "_set_file_permissions", lambda *a, **k: None)
    monkeypatch.setattr(
        _GEN.shutil, "copyfile", lambda src, dst: minted.append(Path(dst).name))
    monkeypatch.chdir(REPO_ROOT)

    _GEN.copy_and_update_amd_notebooks(
        str(TEMPLATE_DIR), str(NB_DIR), "", "", "", "")

    assert minted, "the probe neutralised too much; nothing was selected"
    for basename in _GEN.AMD_SKIP_NOTEBOOKS:
        assert f"AMD-{basename}" not in minted, (
            f"AMD-{basename} is on AMD_SKIP_NOTEBOOKS but the generator still "
            f"mints it, so a regen puts the broken notebook back."
        )
