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

"""XCodec2 must encode real speech codes, or Llasa finetunes on silence (notebooks#131).

transformers >= 4.49 stopped renaming the checkpoint's SnakeBeta `act.beta` to `act.bias`, so
the encoder's bias stays uninitialized, `encode_code` returns all-zero codes, training loss
collapses to ~0.02 and inference outputs noise. xcodec2 also declares
vector-quantize-pytorch==1.17.8, which `--no-deps` skips.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADS_CODEC = "XCodec2Model.from_pretrained("
INSTALLS_XCODEC2 = re.compile(r"^\s*!(?:uv )?pip install\b.*\bxcodec2==", re.MULTILINE)
VQ_PIN = re.compile(r'(?:^|[\s"])vector_quantize_pytorch==1\.17\.8(?:[\s"]|$)')


def _code_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = cell["source"]
            yield "".join(src) if isinstance(src, list) else src


def _notebooks():
    return sorted((REPO_ROOT / "nb").glob("*.ipynb")) + sorted((REPO_ROOT / "original_template").glob("*.ipynb"))


def test_codec_load_restores_snakebeta_bias():
    seen = 0
    for path in _notebooks():
        for cell in _code_cells(path):
            if LOADS_CODEC in cell:
                seen += 1
                assert '.endswith(".act.beta")' in cell and '"bias"' in cell, path.name
    assert seen >= 8


def test_xcodec2_install_pins_vector_quantize():
    seen = 0
    for path in _notebooks():
        for cell in _code_cells(path):
            for line in cell.splitlines():
                if INSTALLS_XCODEC2.match(line):
                    seen += 1
                    assert VQ_PIN.search(line), (path.name, line)
    assert seen >= 6
