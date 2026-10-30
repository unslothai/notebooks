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

import re
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
    # Minted from the nb/ copy above, so it inherits that pin.
    ("nb/AMD-Qwen3_VL_(8B)-Vision-GRPO.ipynb", "trl", "0.26.2"),
    # Muse Glimmer needs the device map planner and the GRPO log-softmax fix,
    # which are in trl 1.9.2 and in no release the canonical pin can move to
    # without dragging every other notebook along. The template carries the
    # pin, and the three generated copies inherit it.
    ("original_template/Muse_Glimmer_(30B)-GRPO.ipynb", "trl", "1.9.2"),
    ("nb/Muse_Glimmer_(30B)-GRPO.ipynb", "trl", "1.9.2"),
    ("nb/Kaggle-Muse_Glimmer_(30B)-GRPO.ipynb", "trl", "1.9.2"),
    ("nb/HuggingFace Course-Muse_Glimmer_(30B)-GRPO.ipynb", "trl", "1.9.2"),
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
    # as_posix, not str: on Windows str() gives `nb\X.ipynb`, which matches no
    # key in _KNOWN_PIN_OUTLIERS, so every allowed outlier fires as drift.
    rel_path = path.relative_to(REPO_ROOT).as_posix()
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


_DATASETS_SPEC = re.compile(r"""["']?\bdatasets((?:\s*,?\s*(?:==|!=|<=|>=|~=|<|>)\s*[0-9][0-9A-Za-z.*]*)+)""")
_DATASETS_PROBES = [f"{a}.{b}.{c}" for a in range(2, 6) for b in range(25) for c in range(8)]


def _datasets_specs(src: str) -> list[str]:
    out = []
    for line in src.splitlines():
        if re.match(r"\s*!(?:uv )?pip install\b", line):
            out += [m.group(1).replace(" ", "") for m in _DATASETS_SPEC.finditer(line)]
    return out


@pytest.mark.parametrize(
    "path",
    sorted((REPO_ROOT / "nb").glob("Kaggle-*.ipynb")),
    ids=lambda p: p.name,
)
def test_kaggle_install_cell_never_pins_two_disjoint_datasets_specs(path: Path) -> None:
    # The Kaggle install cell is one straight-line run of pip calls, so a later
    # datasets spec that excludes an earlier one makes pip install one version
    # only to replace it a few lines down.
    from packaging.specifiers import SpecifierSet

    for cell_idx, src in ni.iter_code_cells(path):
        specs = _datasets_specs(src)
        for i, a in enumerate(specs):
            for b in specs[i + 1:]:
                both = SpecifierSet(a) & SpecifierSet(b)
                assert any(both.contains(v) for v in _DATASETS_PROBES), (
                    f"{path.name} cell {cell_idx}: datasets{a} then datasets{b}"
                )
