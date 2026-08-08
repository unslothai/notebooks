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

"""The AMD composer must resolve the runtime-templated kernel pins.

The Colab install cell picks mamba_ssm / causal_conv1d at runtime from the
GPU's compute capability, so what the AMD composer receives is `{_mamba}` and
`{_conv}`, not a spec. `_package_key_from_install_token` refuses any token
starting with `{`, so an unresolved variable keys to nothing and
`_extract_install_package_groups` drops it: the whole
`--no-build-isolation` group disappears and the AMD notebook ships with no
mamba_ssm at all. `_AMD_VARIABLE_PACKAGE_FALLBACKS` is where that resolution
lives.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

# Colab notebook -> its AMD counterpart. These are the notebooks whose install
# cell reaches for the templated kernel pair.
KERNEL_NOTEBOOKS = [
    ("Granite4.0.ipynb", "AMD-Granite4.0.ipynb"),
    ("Granite4.0_350M.ipynb", "AMD-Granite4.0_350M.ipynb"),
    (
        "Nemotron-3-Nano-30B-A3B_A100.ipynb",
        "AMD-Nemotron-3-Nano-30B-A3B_A100.ipynb",
    ),
    (
        "Nemotron-Nano-3-30B-A3B_A100.ipynb",
        "AMD-Nemotron-Nano-3-30B-A3B_A100.ipynb",
    ),
]


def _load_generator():
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location(
        "_update_all_notebooks_kernel_pins", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_update_all_notebooks_kernel_pins"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()


def _install_cell_text(path):
    """The source notebook's install cell: the one that names the kernels."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "mamba" in source:
            return source
    return None


def _composed_amd_kernel_line(colab_name, amd_name):
    source = _install_cell_text(NB_DIR / colab_name)
    assert source, f"{colab_name} no longer has a kernel install cell"
    _, extras = _GEN._compose_amd_installation(
        str(NB_DIR / amd_name), [source])
    lines = [
        line for line in (extras or "").splitlines()
        if "--no-build-isolation" in line
    ]
    return source, lines


@pytest.mark.parametrize(
    "colab_name,amd_name", KERNEL_NOTEBOOKS, ids=[n for n, _ in KERNEL_NOTEBOOKS])
def test_amd_composer_resolves_the_templated_kernel_pins(colab_name, amd_name):
    source, lines = _composed_amd_kernel_line(colab_name, amd_name)
    assert "{_mamba}" in source and "{_conv}" in source, (
        f"{colab_name} no longer templates the kernel pins, so this test is "
        f"measuring nothing; retire it or repoint it."
    )
    assert lines, (
        f"composing {amd_name} produced no --no-build-isolation install line. "
        f"{colab_name} asks for {{_mamba}} and {{_conv}}; unresolved they key "
        f"to nothing and the group is dropped, so the AMD notebook ships with "
        f"no mamba_ssm. Add the resolutions to "
        f"_AMD_VARIABLE_PACKAGE_FALLBACKS."
    )
    line = " ".join(lines)
    assert "{_" not in line, (
        f"{amd_name} would carry an unexpanded template variable into a pip "
        f"call: {line}"
    )
    assert "mamba_ssm==" in line and "causal_conv1d==" in line, (
        f"{amd_name} kernel install line lost a package: {line}"
    )


@pytest.mark.parametrize(
    "colab_name,amd_name", KERNEL_NOTEBOOKS, ids=[n for n, _ in KERNEL_NOTEBOOKS])
def test_committed_amd_notebook_matches_what_the_composer_produces(
        colab_name, amd_name):
    """The artifact and the generator may not drift apart.

    A hand-edited AMD pin that no regeneration reproduces is the failure this
    catches, in either direction.
    """
    _, lines = _composed_amd_kernel_line(colab_name, amd_name)
    assert lines, f"composing {amd_name} produced no kernel install line"
    committed = (NB_DIR / amd_name).read_text(encoding="utf-8")
    committed_specs = json.loads(committed)
    blob = "\n".join(
        "".join(cell.get("source", []))
        for cell in committed_specs.get("cells", [])
    )
    for line in lines:
        assert line in blob, (
            f"{amd_name} does not contain the line a regeneration would "
            f"write:\n  {line}\nRegenerate with `python update_all_notebooks.py "
            f"--amd`, or fix the composer."
        )
