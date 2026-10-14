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

"""The Studio notebook is Colab-only and must stay the live launcher.

`original_template/Unsloth_Studio.ipynb` was the 2024 chat notebook that clones
the deleted `unslothai/studio` repo, so every regeneration overwrote the
hand-uploaded Colab notebook and minted Kaggle / AMD variants that hang on a
hidden GitHub username prompt (unslothai/unsloth#4944). The current notebook
launches through Studio's Colab helper (`/content`, `from colab import start`),
which fails on Kaggle and AMD Dev Cloud, so neither gets a variant.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "original_template" / "Unsloth_Studio.ipynb"
NB_DIR = REPO_ROOT / "nb"
LGPL = "This notebook and all Unsloth notebooks are licensed [LGPL-3.0]"


def _load_generator():
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location(
        "_update_all_notebooks_studio", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_update_all_notebooks_studio"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()


def _source(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def _generate(tmp_path, template):
    template_dir = tmp_path / "original_template"
    out_dir = tmp_path / "nb"
    template_dir.mkdir()
    out_dir.mkdir()
    shutil.copyfile(template, template_dir / template.name)
    _GEN.copy_and_update_notebooks(
        str(template_dir),
        str(out_dir),
        _GEN.general_announcement_content,
        _GEN.installation_content,
        _GEN.installation_kaggle_content,
        _GEN.new_announcement,
    )
    return out_dir


def test_studio_template_is_the_live_launcher():
    text = _source(TEMPLATE)
    assert "github.com/unslothai/studio" not in text
    assert "from colab import start" in text


def test_studio_gets_no_kaggle_or_amd_notebook(tmp_path):
    out_dir = _generate(tmp_path, TEMPLATE)
    assert (out_dir / "Unsloth_Studio.ipynb").is_file()
    assert not (out_dir / "Kaggle-Unsloth_Studio.ipynb").exists()
    for prefix in ("Kaggle-", "AMD-"):
        assert not (NB_DIR / f"{prefix}Unsloth_Studio.ipynb").exists()


def test_footer_with_its_own_licence_gets_no_lgpl_line(tmp_path):
    out_dir = _generate(tmp_path, TEMPLATE)
    footer = _source(out_dir / "Unsloth_Studio.ipynb")
    assert "This notebook is licensed" in footer
    assert LGPL not in footer


def test_footer_without_a_licence_still_gets_lgpl_line(tmp_path):
    plain = REPO_ROOT / "original_template" / "Llama3.1_(8B)-Alpaca.ipynb"
    out_dir = _generate(tmp_path, plain)
    assert LGPL in _source(out_dir / plain.name)
