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

"""`fast_inference = True` only loads if the notebook installed vLLM.

Unsloth refuses the flag outright when vLLM is missing
(`unsloth/models/loader.py`, both the language and the vision loader):

    ImportError: Unsloth: Please install vLLM before enabling
    `fast_inference`!

That fires on the from_pretrained call, so a notebook whose install cells never
mention vllm is dead at its first model load. There is nothing to catch at
runtime and nothing a library fix can do about it -- the install cell has to
name vllm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

_FAST_INFERENCE_ON = re.compile(r"fast_inference\s*=\s*True")
# Either a literal `vllm` in a pip/uv install line, or the `_vllm` variable the
# generated GRPO install cells expand a pinned version into.
_VLLM_INSTALL = re.compile(
    r"^[^\n]*\b(?:pip install|pip3 install)\b[^\n]*\b(?:vllm|\{_vllm\})",
    re.MULTILINE | re.IGNORECASE,
)
_VLLM_PIN_ASSIGNMENT = re.compile(r"_vllm\s*(?:,[^=\n]*)?=\s*[^\n]*vllm==")


def _notebooks_enabling_fast_inference():
    found = []
    for path in sorted(NB_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = [
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        joined = "\n".join(code)
        if _FAST_INFERENCE_ON.search(joined):
            found.append((path.name, joined))
    return found


_NOTEBOOKS = _notebooks_enabling_fast_inference()


def test_some_notebook_enables_fast_inference():
    """Guard against the parametrisation below silently collecting nothing."""
    assert _NOTEBOOKS, (
        "no notebook under nb/ enables fast_inference any more; this file is "
        "measuring nothing, so retire it or repoint it."
    )


@pytest.mark.parametrize("name,code", _NOTEBOOKS, ids=[n for n, _ in _NOTEBOOKS])
def test_fast_inference_notebook_installs_vllm(name, code):
    installs = bool(_VLLM_INSTALL.search(code)) or bool(
        _VLLM_PIN_ASSIGNMENT.search(code)
    )
    assert installs, (
        f"{name} passes fast_inference = True but no install cell installs "
        f"vllm. from_pretrained raises ImportError(\"Unsloth: Please install "
        f"vLLM before enabling `fast_inference`!\") and the notebook cannot "
        f"get past its model load."
    )
