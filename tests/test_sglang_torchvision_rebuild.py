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

"""Installing sglang replaces torch, so its companions have to be replaced too.

sglang pins one exact torch (`torch==2.11.0` for 0.5.16), which resolves to the
default PyPI wheel, a different CUDA build from the session's. torchvision and
torchaudio are version-compatible either way -- PEP 440 matches `==2.11.0`
against a preinstalled `2.11.0+cu128` -- so pip leaves the old builds in place
and importing them fails on a CUDA mismatch:

  torchvision: "PyTorch has CUDA Version=13.0 and torchvision has CUDA
               Version=12.8"
  torchaudio:  "Detected that PyTorch and TorchAudio were compiled with
               different CUDA versions"

sglang imports both, so either one takes down `sglang.launch_server`. Only
`--force-reinstall` replaces an already-satisfied requirement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

_SGLANG_INSTALL = re.compile(r"^\s*!.*\bpip install\b.*\bsglang\b", re.MULTILINE)

# The torch companions sglang drags along that ship compiled operators linked
# against libtorch, so a CUDA build split is fatal at import.
_COMPANIONS = ("torchvision", "torchaudio")


def _cells_installing_sglang():
    found = []
    for path in sorted(NB_DIR.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if _SGLANG_INSTALL.search(source):
                found.append((f"{path.name}:cell{index}", source))
    return found


_CELLS = _cells_installing_sglang()


def test_some_notebook_installs_sglang():
    """Guard against the parametrisation below silently collecting nothing."""
    assert _CELLS, (
        "no notebook install cell installs sglang any more; this file is "
        "measuring nothing, so retire it or repoint it."
    )


@pytest.mark.parametrize("companion", _COMPANIONS)
@pytest.mark.parametrize("cell_id,source", _CELLS, ids=[c for c, _ in _CELLS])
def test_sglang_install_replaces_torch_companion(cell_id, source, companion):
    pattern = re.compile(
        rf"^\s*!.*\bpip install\b.*\b{companion}\b.*$", re.MULTILINE
    )
    lines = [line.strip() for line in pattern.findall(source)]
    assert lines, (
        f"{cell_id} installs sglang, which pins one exact torch and so swaps "
        f"the CUDA build under the session, but never reinstalls {companion}. "
        f"`import {companion}` then fails on a CUDA version mismatch."
    )
    forced = [line for line in lines if "--force-reinstall" in line]
    assert forced, (
        f"{cell_id} reinstalls {companion} without --force-reinstall: "
        f"{lines}. The version constraint is already satisfied by the "
        f"pre-installed build, so pip does nothing and the mismatch survives."
    )
    for line in forced:
        assert re.search(rf"{companion}==\d", line), (
            f"{cell_id} force-reinstalls {companion} unpinned: {line}. Pip "
            f"would take the newest {companion}, which pairs with a newer "
            f"torch than sglang pinned."
        )
