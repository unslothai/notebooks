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
        "fbgemm-gpu-genai=={_qat_fbgemm}", "{_qat_numpy}"]


def test_pinning_twice_does_not_duplicate():
    groups = {("--force-reinstall",): ["fbgemm-gpu-genai=={_qat_fbgemm}"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert groups[("--force-reinstall",)].count("{_qat_numpy}") == 1


def test_a_group_without_fbgemm_is_untouched():
    groups = {("--no-deps",): ["accelerate", "peft"]}
    _GEN._pin_qat_numpy_beside_fbgemm(groups)
    assert groups == {("--no-deps",): ["accelerate", "peft"]}


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
