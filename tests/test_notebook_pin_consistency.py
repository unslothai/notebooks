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

"""Pin-consistency gate: every ``pkg==version`` pinned inside a
notebook's ``!pip install`` cell must also appear somewhere in
update_all_notebooks.py (either as a PIN_* constant or inside a
per-model override branch). Catches the failure mode where someone
hand-edits a generated nb/*.ipynb to bump a pin and forgets to update
the source-of-truth generator.

Pins outside the PINNED_PACKAGES set (e.g. one-off model-specific
deps like xcodec2, mamba-ssm) are not checked here -- they would be
caught by test_notebook_template_regeneration on demand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


_ALL_NOTEBOOKS = list(ni.iter_notebooks())
_PIN_CONSTANTS = ni.read_pin_constants()
_OVERRIDE_PINS = ni.extract_per_model_override_pins()
_INTERESTING = set(ni.PINNED_PACKAGES)

# Notebooks with pins that do not appear in update_all_notebooks.py but
# are known special-purpose one-offs (Whisper audio + experimental GRPO
# variants). The right long-term fix is to either lift these into the
# generator as per-model overrides, or regenerate the notebook from a
# template that uses the canonical PIN_* constants. Tracked as
# follow-up to the CI mirror PR; remove an entry when its notebook is
# brought back into the canonical / override set.
_KNOWN_PIN_OUTLIERS: set[tuple[str, str, str]] = {
    # (notebook-rel-path, pkg, ver)
    ("original_template/Whisper.ipynb", "trl", "0.21.0"),
    ("original_template/Whisper.ipynb", "transformers", "4.51.3"),
    ("nb/Openenv_wordle_grpo.ipynb", "trl", "0.29.1"),
    ("nb/AMD-Openenv_wordle_grpo.ipynb", "trl", "0.29.1"),
    ("nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb", "transformers", "4.57.0"),
    ("nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb", "trl", "0.26.2"),
}


def _legitimate_pin(pkg: str, ver: str, rel_path: str) -> bool:
    """A (pkg, ver) is legitimate if any of:
       * the canonical PIN_* constant matches, or
       * the pair appears anywhere in the generator's per-model override
         branches, or
       * (notebook, pkg, ver) is explicitly grandfathered."""
    canonical = _PIN_CONSTANTS.get(pkg)
    if canonical is not None and canonical == ver:
        return True
    if (pkg, ver) in _OVERRIDE_PINS:
        return True
    return (rel_path, pkg, ver) in _KNOWN_PIN_OUTLIERS


def test_pin_constants_were_extracted() -> None:
    """Sanity: read_pin_constants found at least transformers + trl."""
    missing = {"transformers", "trl"} - set(_PIN_CONSTANTS)
    assert not missing, (
        f"DRIFT DETECTED: notebook_inventory.read_pin_constants did not "
        f"resolve PIN_TRANSFORMERS / PIN_TRL from update_all_notebooks.py. "
        f"Lines 107-117 were the source of truth at branch fork; if they "
        f"moved or were renamed, the pin-consistency gate is reading the "
        f"wrong constants. Missing: {sorted(missing)}. "
        f"Found: {_PIN_CONSTANTS}"
    )


@pytest.mark.parametrize("path", _ALL_NOTEBOOKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_notebook_pip_pins_match_generator_canonical_or_override(path: Path) -> None:
    """Walk every code cell, collect ``pkg==X`` pins for the packages
    we care about, assert each is in the union of canonical PIN_*
    constants + the per-model override pins extracted from the
    generator."""
    rel_path = str(path.relative_to(REPO_ROOT))
    bad: list[tuple[int, str, str]] = []
    for cell_idx, src in ni.iter_code_cells(path):
        for pkg, ver in ni.extract_pip_pins_from_cell(src):
            if pkg not in _INTERESTING:
                continue
            if not _legitimate_pin(pkg, ver, rel_path):
                bad.append((cell_idx, pkg, ver))
    if bad:
        rel = path.relative_to(REPO_ROOT)
        lines = "\n".join(
            f"    cell {i}: {pkg}=={ver}" for i, pkg, ver in bad
        )
        canon = ", ".join(f"{k}=={v}" for k, v in sorted(_PIN_CONSTANTS.items()))
        pytest.fail(
            f"DRIFT DETECTED: {rel} pins package versions that are not in "
            f"the generator's canonical / override pin set:\n{lines}\n"
            f"Canonical PIN_* constants: {canon}\n"
            f"Either update update_all_notebooks.py to mention the new "
            f"version, or regenerate the notebook from the template."
        )
