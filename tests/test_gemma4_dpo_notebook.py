# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gemma 4 DPO notebooks must install transformers >= 5.5.2, not the 5.5.0 the other Gemma 4 notebooks pin.

On 5.5.0-5.5.1, unsloth_zoo carries E2B/E4B cross-layer shared K/V in a module-scoped carrier.
DPOTrainer's reference forward runs under no_grad between the policy forward and its backward, and
replaces then deletes that carrier, so the checkpointed recompute traces a different graph and the
first backward dies with `torch.utils.checkpoint.CheckpointError: Recomputed values for the following
tensors have different metadata`. 5.5.2+ passes shared_kv_states natively and has no carrier.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Colab / Kaggle pin `transformers==X.Y.Z`; the AMD extras cell sets a floor, `transformers>=X.Y.Z`.
TRANSFORMERS_SPEC = re.compile(
    r"^\s*!(?:uv )?pip install\b.*?\btransformers(==|>=)(\d+)\.(\d+)\.(\d+)", re.MULTILINE
)


def _gemma4_dpo_notebooks():
    for path in sorted((REPO_ROOT / "nb").glob("*Gemma4*DPO*.ipynb")):
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        yield path, "\n".join(
            "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            for c in cells
            if c.get("cell_type") == "code"
        )


def test_installs_transformers_5_5_2_or_newer():
    seen = 0
    for path, code in _gemma4_dpo_notebooks():
        specs = TRANSFORMERS_SPEC.findall(code)
        assert len(specs) == 1, (path.name, specs)
        assert tuple(map(int, specs[0][1:])) >= (5, 5, 2), (path.name, specs)
        seen += 1
    assert seen >= 3  # Colab, Kaggle and AMD copies
