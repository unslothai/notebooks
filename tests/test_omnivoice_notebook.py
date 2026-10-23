# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OmniVoice trains with a 4D bidirectional mask, which Flash Attention cannot take.

Where FA2 is installed (L4, A100, H100) FastModel picks it unless asked for SDPA, and training
dies with `cu_seqlens_q must have shape (batch_size + 1)` or a device-side assert.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_PIN = re.compile(r"^\s*!(?:uv )?pip install\b.*\btransformers==(\d+)\.(\d+)", re.MULTILINE)


def _omnivoice_notebooks():
    for folder in ("nb", "original_template"):
        for path in sorted((REPO_ROOT / folder).glob("*.ipynb")):
            cells = [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c.get("cell_type") == "code"]
            code = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells]
            if any("from omnivoice import" in c for c in code):
                yield path, code


def test_backbone_loads_with_sdpa():
    seen = 0
    for path, code in _omnivoice_notebooks():
        for cell in code:
            if "FastModel.from_pretrained(" in cell:
                seen += 1
                assert re.search(r'attn_implementation\s*=\s*"sdpa"', cell), path.name
    assert seen >= 3


def test_installs_omnivoice_and_transformers_5_3_or_newer():
    seen = 0
    for path, code in _omnivoice_notebooks():
        if path.parent.name != "nb":
            continue
        install = "\n".join(code)
        pins = TRANSFORMERS_PIN.findall(install)
        assert len(pins) == 1 and tuple(map(int, pins[0])) >= (5, 3), (path.name, pins)
        assert re.search(r"pip install --no-deps omnivoice==", install), path.name
        seen += 1
    assert seen >= 2
