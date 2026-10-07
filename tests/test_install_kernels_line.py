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

"""`!unsloth install-kernels` replaces the hand-written xformers / causal_conv1d wheel picks.

The command runs the `unsloth` CLI, so it only works after the cell installed unsloth. ROCm
has no prebuilt wheels, so AMD cells keep their own lines and never call it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS = re.compile(r"^\s*!unsloth install-kernels\b", re.MULTILINE)
INSTALLS_UNSLOTH = re.compile(r'^\s*!(?:uv )?pip install\b.*(?:\s|")unsloth(?:\s|"|>|=|\[|$)', re.MULTILINE)
RETIRED = (
    "xformers = 'xformers=='",
    "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl",
    "{xformers}",
    "causal-conv1d/releases/download/v1.7.0",
)


def _code_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = cell["source"]
            yield "".join(src) if isinstance(src, list) else src


def _notebooks():
    return sorted((REPO_ROOT / "nb").glob("*.ipynb"))


def test_retired_wheel_picks_are_gone():
    offenders = [
        (path.name, text)
        for path in _notebooks()
        for cell in _code_cells(path)
        for text in RETIRED
        if text in cell
    ]
    assert offenders == []


def test_kernel_line_follows_the_unsloth_install():
    seen = 0
    for path in _notebooks():
        if path.name.startswith("AMD-"):
            continue
        earlier = ""
        for cell in _code_cells(path):
            for match in KERNELS.finditer(cell):
                seen += 1
                # Join `\`-continued pip lines, which name unsloth on a later physical line.
                before = (earlier + cell[: match.start()]).replace("\\\n", " ")
                assert INSTALLS_UNSLOTH.search(before), (path.name, cell)
            earlier += cell + "\n"
    assert seen > 100


def test_ssm_notebooks_no_longer_build_kernels_from_source():
    offenders = [
        path.name
        for path in _notebooks()
        if not path.name.startswith("AMD-")
        for cell in _code_cells(path)
        if re.search(r"--no-build-isolation[^\n]*(mamba|causal)", cell)
    ]
    assert offenders == []


def test_amd_cells_never_call_it():
    amd = [path for path in _notebooks() if path.name.startswith("AMD-")]
    assert amd
    assert [p.name for p in amd for cell in _code_cells(p) if KERNELS.search(cell)] == []


def test_molab_runs_the_kernel_install_as_its_own_cell():
    molab = sorted((REPO_ROOT / "molab").glob("*.py"))
    texts = {p.name: p.read_text(encoding="utf-8") for p in molab}
    # marimo cannot run `!` magics, and the PEP 723 header cannot pick a wheel per torch.
    assert [n for n, t in texts.items() if "#! unsloth install-kernels" in t] == []
    assert sum('subprocess.run(["unsloth", "install-kernels"])' in t for t in texts.values()) > 100
