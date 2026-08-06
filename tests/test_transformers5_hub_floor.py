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

"""A notebook installing transformers 5.x must raise huggingface_hub with it.

transformers 5.x requires `huggingface-hub>=1.5.0,<2.0`; 4.x requires `<1.0`.
The Gemma 4 install cells pin transformers with `--no-deps`, so pip never
enforces that, and the `huggingface_hub>=0.34.0` line beside them is satisfied
by Colab's preinstalled 0.36.2. The very next import then stopped with

    found 0.36.2, needs >=1.5.0

which is the whole notebook, at cell 6, before a single line of training. A
lower bound that a stale environment already satisfies buys nothing.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

# `--no-deps` on the same command is the whole point: without it pip intersects
# transformers 5.x's own `huggingface-hub>=1.5.0` with whatever floor the line
# asks for and installs something that works. With it, nothing does.
_NO_DEPS_TRANSFORMERS_5 = re.compile(
    r"^[^#\n]*--no-deps[^\n]*transformers\s*[=>]=\s*5\.", re.MULTILINE)
_HUB_FLOOR = re.compile(r"huggingface[-_]hub\s*>=\s*1\.(?:[5-9]|\d\d)")


def _install_text(path):
    """Every code cell, joined. Installs are not always in cell 0."""
    notebook = json.loads(path.read_text(encoding = "utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _notebooks_pinning_transformers_5():
    for path in sorted(NB_DIR.glob("*.ipynb")):
        text = _install_text(path)
        if _NO_DEPS_TRANSFORMERS_5.search(text):
            yield path.name, text


def test_some_notebook_installs_transformers_5_without_deps():
    """Or the check below is vacuously green forever."""
    assert list(_notebooks_pinning_transformers_5())


@pytest.mark.parametrize("name, text", list(_notebooks_pinning_transformers_5()),
                         ids = lambda v: v if isinstance(v, str) and len(v) < 80 else "")
def test_transformers_5_notebooks_raise_the_hub_floor(name, text):
    assert _HUB_FLOOR.search(text), (
        f"{name} installs transformers 5.x with --no-deps and never raises "
        "huggingface_hub to 1.5.0; Colab's preinstalled 0.36.2 satisfies a "
        ">=0.34.0 floor and the next import fails"
    )
