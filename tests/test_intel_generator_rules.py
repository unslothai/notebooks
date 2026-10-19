"""Intel- notebooks get an XPU install recipe, XPU devices and if listed, a multi-GPU device_map.

The first install cell is the shared XPU cell, or the vLLM XPU cell when the
notebook uses `fast_inference = True`. The second keeps the source notebook's own
pins, drops what has no XPU build, and ends with the triton -> triton-xpu swap. The
generator runs `update_notebook_sections` twice, so the composed cells must come out
the same when the second pass reads them back as source.

`_apply_intel_cell_rules` converts `torch.cuda.*` and `"cuda"` into their XPU
equivalents and gives the first `from_pretrained` of each notebook in
`INTEL_MULTI_GPU_NOTEBOOKS` a `device_map = "balanced"`. `_validate_intel_notebook`
fails a generated notebook if those rules were missed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CUDA_INSTALL = """%%capture
!uv pip install -qqq "torch==2.8.0" "triton>=3.3.0" xformers==0.0.32.post2 \\
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
!uv pip install -qqq --no-deps "torchcodec==0.7.0"
!uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
!uv pip install --no-build-isolation causal_conv1d==1.6.0
!unsloth install-kernels
"""

GRPO_INSTALL = """%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
!pip install unsloth vllm
_vllm, _triton = ('vllm==0.11.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')
!uv pip install -qqq --upgrade {_vllm} torchvision bitsandbytes xformers unsloth
!uv pip install transformers==4.57.6
!uv pip install --no-deps trl==0.22.2
"""

MOE_FROM_PRETRAINED = """model, processor = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3.5-35B-A3B",
    load_in_4bit = False,
    fast_inference = False, # Not supported for MoE (yet!)
)
"""

GRPO_FROM_PRETRAINED = """model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    fast_inference = True, # Enable vllm fast inference
)
"""


@pytest.fixture(scope = "module")
def gen():
    spec = importlib.util.spec_from_file_location("update_all_notebooks", REPO_ROOT / "update_all_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code(source):
    return {"cell_type": "code", "source": source, "metadata": {}, "outputs": [], "execution_count": None}


def _write_notebook(path, cells):
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return str(path)


@pytest.fixture
def moe_notebook(tmp_path):
    """The recipe depends on the notebook's code, so compose reads a file."""
    return _write_notebook(tmp_path / "Intel-Qwen3_5_MoE.ipynb", [_code(MOE_FROM_PRETRAINED)])


@pytest.fixture
def grpo_notebook(tmp_path):
    return _write_notebook(tmp_path / "Intel-Qwen3_(4B)-GRPO.ipynb", [_code(GRPO_FROM_PRETRAINED)])


def test_keeps_source_pins_and_drops_cuda_packages(gen, moe_notebook):
    install, extras = gen._compose_intel_installation(moe_notebook, [CUDA_INSTALL])
    assert install == gen.installation_intel_cell
    assert '"trl==0.22.2"' in extras and '"transformers==5.2.0"' in extras
    assert "git+https://github.com/unslothai/unsloth-zoo" in extras
    for cuda_only in ("xformers", "causal_conv1d", "torchcodec", "install-kernels", '"torch==2.8.0"'):
        assert cuda_only not in extras, cuda_only
    # The triton swap is last, after every install that could pull generic triton back.
    assert extras.rstrip().endswith(gen.INTEL_XPU_FINALIZE.rstrip())
    assert extras.index("!uv pip uninstall -qqq triton") > extras.index('"transformers==5.2.0"')


def test_fast_inference_notebook_gets_the_vllm_recipe(gen, grpo_notebook):
    """The vLLM cell owns vLLM, torch and transformers, and vLLM's triton shim replaces the swap."""
    install, extras = gen._compose_intel_installation(grpo_notebook, [GRPO_INSTALL])
    assert install == gen.installation_intel_vllm_cell
    assert gen.INTEL_VLLM_SPEC in install and f"wheels.vllm.ai/{gen.INTEL_VLLM_VERSION}/xpu" in install
    for owned in ("vllm", "transformers==4.57.6", "torchvision", "xformers"):
        assert owned not in extras, owned
    assert '"trl==0.22.2"' in extras
    assert 'os.environ["UNSLOTH_VLLM_STANDBY"] = "1"' in extras
    assert "triton" not in extras


@pytest.mark.parametrize("notebook, source", [("moe_notebook", CUDA_INSTALL), ("grpo_notebook", GRPO_INSTALL)])
def test_second_pass_reads_its_own_cells_back_unchanged(gen, notebook, source, request):
    path = request.getfixturevalue(notebook)
    install, first = gen._compose_intel_installation(path, [source])
    _install, second = gen._compose_intel_installation(path, [source, install, first or ""])
    assert second == first


def test_followup_drops_template_placeholder_and_previous_extras(gen):
    """The vLLM extras cell has no pip line, so only this rule stops the second pass from
    inserting it again; the template's second `# Placeholder` has no Intel use."""
    extras = 'import os\nos.environ["UNSLOTH_VLLM_STANDBY"] = "1"\n'
    cells = [_code("# install"), _code(extras), _code("# Placeholder"), _code(GRPO_FROM_PRETRAINED)]
    assert [idx for idx, _ in gen._intel_stale_followup_cells(cells, 0, extras)] == [1, 2]
    # No extras: an empty cell is not mistaken for one.
    assert gen._intel_stale_followup_cells([_code("# install"), _code("")], 0, None) == []


def test_multi_gpu_extras_only_for_listed_notebooks(gen, moe_notebook, grpo_notebook):
    env = "%env PYTORCH_ALLOC_CONF=expandable_segments:False"
    _install, moe_extras = gen._compose_intel_installation(moe_notebook, [CUDA_INSTALL])
    _install, grpo_extras = gen._compose_intel_installation(grpo_notebook, [GRPO_INSTALL])
    assert env in moe_extras
    assert env not in (grpo_extras or "")


def test_cuda_devices_are_rewritten_to_xpu(gen):
    src = ('x.to("cuda")\ny.to(\'cuda:1\')\ntorch.cuda.empty_cache()\n'
           'p = torch.cuda.get_device_properties(0)\nprint(torch.version.cuda)\n')
    out = gen._rewrite_cuda_to_xpu(src)
    assert 'x.to("xpu")' in out and "y.to('xpu:1')" in out
    assert "torch.xpu.empty_cache()" in out and "torch.xpu.get_device_properties(0)" in out
    assert "torch.version.cuda" in out


def test_device_map_goes_after_the_model_name_line(gen):
    cells = [_code(MOE_FROM_PRETRAINED)]
    assert gen._apply_intel_cell_rules("nb/Intel-Qwen3_5_MoE.ipynb", cells)
    assert '    "unsloth/Qwen3.5-35B-A3B",\n    device_map = "balanced",' in gen._cell_source_text(cells[0])
    # A second pass adds nothing.
    assert not gen._apply_intel_cell_rules("nb/Intel-Qwen3_5_MoE.ipynb", cells)


def test_device_map_goes_to_listed_notebooks_only(gen):
    """Listed by notebook, not model: whether one 24 GB card is enough depends on the
    notebook's own settings (4-bit Llama 3.2 11B Vision still needs two cards to train)."""
    cells = [_code('model, tokenizer = FastVisionModel.from_pretrained(\n'
                   '    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct",\n    load_in_4bit = True,\n)\n')]
    assert gen._apply_intel_cell_rules("nb/Intel-Llama3.2_(11B)-Vision.ipynb", cells)
    assert 'model_name = "unsloth/Llama-3.2-11B-Vision-Instruct",\n    device_map = "balanced",' in gen._cell_source_text(cells[0])
    # The same model in a notebook that is not listed is left alone.
    assert not gen._apply_intel_cell_rules("nb/Intel-Some_New_Notebook.ipynb", [_code(MOE_FROM_PRETRAINED)])


def test_device_map_skips_model_lists_and_later_reloads(gen):
    """Vision keeps a list of model names above the call, and reloads its saved adapters with
    a second from_pretrained further down."""
    load = ('fourbit_models = [\n    "unsloth/Pixtral-12B-2409-bnb-4bit",\n]\n'
            'model, tokenizer = FastVisionModel.from_pretrained(\n'
            '    "unsloth/Llama-3.2-11B-Vision-Instruct",\n    load_in_4bit = True,\n)\n')
    reload = 'model, tokenizer = FastVisionModel.from_pretrained(\n    model_name = "lora_model",\n)\n'
    cells = [_code(load), _code(reload)]
    gen._apply_intel_cell_rules("nb/Intel-Llama3.2_(11B)-Vision.ipynb", cells)
    first, second = (gen._cell_source_text(c) for c in cells)
    assert '"unsloth/Pixtral-12B-2409-bnb-4bit",\n]' in first
    assert '"unsloth/Llama-3.2-11B-Vision-Instruct",\n    device_map = "balanced",' in first
    assert "device_map" not in second


def test_device_map_rule_raises_when_it_cannot_place_it(gen):
    cells = [_code('model, processor = FastLanguageModel.from_pretrained("unsloth/Qwen3.5-35B-A3B", load_in_4bit = False)\n')]
    with pytest.raises(RuntimeError, match = "_RE_MODEL_NAME_LINE"):
        gen._apply_intel_cell_rules("nb/Intel-Qwen3_5_MoE.ipynb", cells)


def test_validator_flags_leftover_cuda_and_missing_device_map(gen, tmp_path):
    path = _write_notebook(tmp_path / "Intel-Qwen3_5_MoE.ipynb", [
        _code(gen.installation_intel_cell), _code(gen.INTEL_XPU_FINALIZE), _code(MOE_FROM_PRETRAINED), _code('x.to("cuda")\n'),
    ])
    issues = gen._validate_intel_notebook(path)
    assert "contains '\"cuda\"'" in issues
    assert 'is missing device_map = "balanced"' in issues


def test_validator_flags_fast_inference_without_vllm_recipe(gen, tmp_path):
    path = _write_notebook(tmp_path / "Intel-Qwen3_(4B)-GRPO.ipynb", [
        _code(gen.installation_intel_cell), _code(gen.INTEL_XPU_FINALIZE), _code(GRPO_FROM_PRETRAINED),
    ])
    assert "uses fast_inference = True but has no Intel vLLM install" in gen._validate_intel_notebook(path)
