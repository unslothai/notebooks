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

"""The QAT numpy pin has to come from the generator, not the artifacts.

fbgemm-gpu-genai depends on an unpinned numpy, so the force-reinstall in the
QAT install cell fetches the newest one while `import torch` two lines up has
already loaded the old one. The next import then stops with "numpy was upgraded
mid-session" and the notebook cannot continue without a restart.

The pin was first applied to the generated notebooks only, which meant the next
`update_all_notebooks.py` run would have removed it again, and the AMD notebook
computed `_qat_numpy` and then never passed it to the install command at all.
Both are the same mistake in different places, so this checks the generator and
the committed artifacts together.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_generator():
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location("_qat_pin_generator", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_qat_pin_generator"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()

QAT_NOTEBOOKS = [
    "Qwen3_(4B)_Instruct-QAT",
    "AMD-Qwen3_(4B)_Instruct-QAT",
    "Kaggle-Qwen3_(4B)_Instruct-QAT",
]


def _install_cell(name):
    path = REPO_ROOT / "nb" / f"{name}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        if "_qat_fbgemm" in source:
            return source
    pytest.fail(f"{name} has no QAT install cell")


# ---- the generator -------------------------------------------------------

def test_the_colab_block_pins_numpy():
    block = _GEN.build_qat_native_install_block()
    assert "_qat_numpy" in block
    assert "{_qat_numpy}" in block.rsplit("\n", 1)[-1], (
        "the pin must be in the install command, not merely computed")


def test_the_amd_block_pins_numpy():
    assert "_qat_numpy" in _GEN._build_qat_version_vars_block()


def test_both_blocks_share_one_definition():
    """Two copies drift, and this one was already worded two different ways."""
    colab = _GEN.build_qat_native_install_block()
    amd = _GEN._build_qat_version_vars_block()
    assert _GEN.QAT_NUMPY_PIN_BLOCK in colab
    assert _GEN.QAT_NUMPY_PIN_BLOCK in amd


def test_the_pin_lands_in_the_fbgemm_command():
    """A later command is too late: pip has already installed a newer numpy by
    then and the kernel is holding the old one."""
    groups = {("--upgrade", "--force-reinstall"):
              ["fbgemm-gpu-genai=={_qat_fbgemm}"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert groups[("--upgrade", "--force-reinstall")] == [
        "fbgemm-gpu-genai=={_qat_fbgemm}", "{_qat_numpy}",
        "torchao=={_qat_torchao}"]


def test_the_torchao_pin_rides_along_when_amd_left_it_unpinned():
    """AMD seeds a bare `torchao` with lock = True, dropping the source's pin."""
    groups = {("--force-reinstall",): ["fbgemm-gpu-genai=={_qat_fbgemm}"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert "torchao=={_qat_torchao}" in groups[("--force-reinstall",)]


def test_a_torchao_the_variant_did_pin_is_left_alone():
    """Only fill the gap; a deliberate AMD pin outranks the computed one."""
    groups = {("--force-reinstall",):
              ["fbgemm-gpu-genai=={_qat_fbgemm}", "torchao==0.14.0"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert "torchao=={_qat_torchao}" not in groups[("--force-reinstall",)]
    assert "torchao==0.14.0" in groups[("--force-reinstall",)]


def test_pinning_twice_does_not_duplicate():
    groups = {("--force-reinstall",): ["fbgemm-gpu-genai=={_qat_fbgemm}"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert groups[("--force-reinstall",)].count("{_qat_numpy}") == 1
    assert groups[("--force-reinstall",)].count("torchao=={_qat_torchao}") == 1


def test_a_group_without_fbgemm_is_untouched():
    groups = {("--no-deps",): ["accelerate", "peft"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert groups == {("--no-deps",): ["accelerate", "peft"]}


# ---- the PEFT floor ------------------------------------------------------

def _as_tuple(version):
    return tuple(int(part) for part in version.split("."))


@pytest.mark.parametrize("table", ["QAT_TORCHAO_BY_TORCH_VERSION",
                                   "QAT_TORCHAO_BY_TORCH_MINOR"])
def test_no_torchao_pin_falls_below_the_peft_floor(table):
    """peft 0.19 raises ImportError from `is_torchao_available()` under 0.16.0,
    and `get_peft_model` reaches it, so a lower pin kills the run outright.
    Picking the torch-matched release is only worth doing above that line."""
    floor = _as_tuple(_GEN.QAT_PEFT_TORCHAO_FLOOR)
    for torch_version, torchao in getattr(_GEN, table).items():
        assert _as_tuple(torchao) >= floor, f"{table}[{torch_version}] = {torchao}"


@pytest.mark.parametrize("table", ["QAT_TORCHAO_BY_TORCH_VERSION",
                                   "QAT_TORCHAO_BY_TORCH_MINOR"])
def test_no_torch_below_2_11_is_pinned_to_an_unimportable_torchao(table):
    """torchao 0.17.0 and up import `ScalingType` from `torch.nn.functional`,
    which arrives in torch 2.11, so `import torchao` itself fails below that.
    Measured: torch 2.8.0 and 2.9.1 with torchao 0.18.0 both raise ImportError.
    The floor test alone would happily pick 0.18.0 and kill the notebook."""
    newest = _as_tuple(_GEN.QAT_TORCHAO_NEWEST_TORCH + ".0")[:2]
    for torch_version, torchao in getattr(_GEN, table).items():
        parts = _as_tuple(torch_version + ".0" * (2 - torch_version.count(".")))
        if parts[:2] >= newest:
            continue
        assert _as_tuple(torchao) < (0, 17, 0), \
            f"{table}[{torch_version}] = {torchao} cannot be imported on that torch"


def test_exactly_one_release_satisfies_both_bounds():
    """0.16.0 is the whole answer below torch 2.11: the peft floor and the
    torch ceiling leave no other choice, so every such row must be it."""
    newest = _as_tuple(_GEN.QAT_TORCHAO_NEWEST_TORCH + ".0")[:2]
    rows = [v for k, v in _GEN.QAT_TORCHAO_BY_TORCH_VERSION.items()
            if _as_tuple(k)[:2] < newest]
    assert rows and set(rows) == {_GEN.QAT_PEFT_TORCHAO_FLOOR}


def test_the_default_pin_clears_the_floor_too():
    """It is what every torch the tables have not seen gets."""
    assert _as_tuple(_GEN.QAT_DEFAULT_TORCHAO_VERSION) >= \
        _as_tuple(_GEN.QAT_PEFT_TORCHAO_FLOOR)


def _resolve_emitted(torch_version):
    """Run the emitted resolution exactly as a notebook kernel would.

    The tables are only half the logic; the fallback decides what an unlisted
    torch gets, and that is reachable only by executing the block.
    """
    block = _GEN.build_qat_native_install_block()
    body = block.split("_qat_fbgemm_map")[0]
    # Drop the torch-detection preamble and inject the version under test.
    body = "\n".join(
        line for line in body.splitlines()
        if not line.startswith(("try:", "    import torch;", "except Exception:",
                                "    _qat_torch_version"))
    )
    minor = torch_version.rsplit(".", 1)[0] if torch_version else ""
    ns = {"_qat_torch_version": torch_version, "_qat_torch_minor": minor}
    exec(body, ns)
    return ns["_qat_torchao"]


@pytest.mark.parametrize("torch_version", ["2.4.0", "2.5.1", "2.6.0", "2.7.0"])
def test_an_unlisted_older_torch_gets_an_importable_torchao(torch_version):
    """The tables cover 2.8-2.11. Anything older fell through to the newest
    pin, so a local torch 2.7 was handed torchao 0.18.0, which reads
    `torch.nn.functional.ScalingType` -- added in 2.11 -- and dies on import.
    The failure lands at `import torchao`, long after pip has reported success."""
    assert _as_tuple(_resolve_emitted(torch_version)) < (0, 17, 0), \
        f"torch {torch_version} would get an unimportable torchao"


@pytest.mark.parametrize("torch_version", ["2.11.0", "2.12.0", "2.13.0"])
def test_an_unlisted_newer_torch_still_gets_the_newest_pin(torch_version):
    """The fix must not drag future torches down to the floor."""
    assert _resolve_emitted(torch_version) == _GEN.QAT_DEFAULT_TORCHAO_VERSION


def test_a_listed_torch_still_comes_from_the_table():
    for torch_version, expected in _GEN.QAT_TORCHAO_BY_TORCH_VERSION.items():
        assert _resolve_emitted(torch_version) == expected


def test_an_undetectable_torch_keeps_the_old_default():
    """Nothing was learned, so nothing is assumed."""
    assert _resolve_emitted("") == _GEN.QAT_DEFAULT_TORCHAO_VERSION


def test_the_emitted_notebook_pins_clear_the_floor():
    """The tables are interpolated into the cell as JSON, so a floor kept only
    in the generator would not survive a hand edit of the emitted block."""
    import re
    block = _GEN.build_qat_native_install_block()
    floor = _as_tuple(_GEN.QAT_PEFT_TORCHAO_FLOOR)
    found = re.findall(r'"(\d+\.\d+\.\d+)"\s*:\s*"(\d+\.\d+\.\d+)"', block)
    assert found, "no torchao mapping found in the emitted block"
    for torch_version, torchao in found:
        assert _as_tuple(torchao) >= floor, f"{torch_version} -> {torchao}"


# ---- the committed notebooks --------------------------------------------

@pytest.mark.parametrize("name", QAT_NOTEBOOKS)
def test_the_notebook_passes_the_pin_to_pip(name):
    source = _install_cell(name)
    install = [line for line in source.splitlines()
               if "fbgemm-gpu-genai" in line and line.lstrip().startswith("!")]
    assert install, f"{name} has no fbgemm install command"
    for line in install:
        assert "{_qat_numpy}" in line, line


@pytest.mark.parametrize("name", ["Qwen3_(4B)_Instruct-QAT",
                                  "Kaggle-Qwen3_(4B)_Instruct-QAT"])
def test_the_notebook_matches_the_generator(name):
    """These two cells are emitted whole, so drift is directly checkable."""
    assert _install_cell(name).rstrip("\n") == \
        _GEN.installation_qat_content.rstrip("\n")


def test_the_amd_notebook_carries_the_generator_block():
    """The AMD cell is composed rather than emitted whole, so only the block
    it takes from the generator can be compared."""
    assert _GEN._build_qat_version_vars_block() in \
        _install_cell("AMD-Qwen3_(4B)_Instruct-QAT")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
