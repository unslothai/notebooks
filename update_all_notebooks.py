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
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import ast
import concurrent.futures
import concurrent.futures.process
import copy
import json
import multiprocessing
import os
import pickle
import platform
import re
import shutil
import stat
import subprocess
import sys
import csv
import hashlib
from datetime import datetime, timezone
from glob import glob
from nbconvert import PythonExporter
import nbformat
from spellchecker import SpellChecker

try:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
except Exception:
    HfApi = None
    RepositoryNotFoundError = Exception

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

new_announcement = """Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)

<table><tr>
<td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
<td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
</tr></table>

Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)

Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)

New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)

Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).""".strip()

hf_course_name = "HuggingFace Course"

announcement_separation = '<div class="align-center">'

general_announcement_content = """To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)"""

general_announcement_content_a100 = general_announcement_content.replace("on a **free** Tesla T4 Google Colab instance!", "on your A100 Google Colab Pro instance!")
general_announcement_content_l4 = general_announcement_content.replace("on a **free** Tesla T4 Google Colab instance!", "on your L4 Google Colab Pro instance!")
general_announcement_content_fp8 = general_announcement_content_l4  # backwards compat alias

general_announcement_content_hf_course = general_announcement_content.split(announcement_separation)
general_announcement_content_hf_course = general_announcement_content_hf_course[0] + announcement_separation + '<a href="https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt"><img src="https://github.com/unslothai/notebooks/raw/main/assets/hf%20course.png" width="165"></a>' + general_announcement_content_hf_course[1]
general_announcement_content_hf_course = general_announcement_content_hf_course.split("To install Unsloth")
hf_additional_string_announcement = "In this [Hugging Face](https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt) and Unsloth notebook, you will learn to transform {full_model_name} into a Reasoning model using GRPO."
general_announcement_content_hf_course = (
    general_announcement_content_hf_course[0] + 
    hf_additional_string_announcement + 
    "\n\n" +
    "To install Unsloth" + general_announcement_content_hf_course[1]
)

general_announcement_content_meta = general_announcement_content.split(announcement_separation)
general_announcement_content_meta = general_announcement_content_meta[0] + "\n\n" + '<a href="https://github.com/meta-llama/synthetic-data-kit"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/meta%20round%20logo.png" width="137"></a>' + general_announcement_content_meta[1]

# CONSTANT
PIN_TRANSFORMERS = "!pip install transformers==4.56.2"
UV_PIN_TRANSFORMERS = PIN_TRANSFORMERS.replace("pip", "uv pip")

PIN_TRL = "!pip install --no-deps trl==0.22.2"
UV_PIN_TRL = PIN_TRL.replace("pip", "uv pip")
SPACES = " " * 4

XFORMERS_INSTALL = """xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")"""

QAT_TORCHAO_BY_TORCH_MINOR = {
    "2.10": "0.16.0",
    "2.9": "0.15.0",
    "2.8": "0.14.1",
}
QAT_DEFAULT_TORCHAO_VERSION = "0.16.0"
QAT_FBGEMM_GENAI_BY_TORCH_MINOR = {
    "2.10": "1.5.0",
    "2.9": "1.4.2",
    "2.8": "1.3.0",
}
QAT_DEFAULT_FBGEMM_GENAI_VERSION = "1.5.0"


def build_qat_native_install_block(
    torchao_by_torch_minor=None,
    default_torchao=QAT_DEFAULT_TORCHAO_VERSION,
    fbgemm_genai_by_torch_minor=None,
    default_fbgemm_genai=QAT_DEFAULT_FBGEMM_GENAI_VERSION,
):
    """Build runtime torchao/fbgemm native install block for QAT notebooks."""
    if torchao_by_torch_minor is None:
        torchao_by_torch_minor = QAT_TORCHAO_BY_TORCH_MINOR
    if fbgemm_genai_by_torch_minor is None:
        fbgemm_genai_by_torch_minor = QAT_FBGEMM_GENAI_BY_TORCH_MINOR
    torchao_mapping = json.dumps(
        torchao_by_torch_minor, sort_keys=True, separators=(",", ":")
    )
    fbgemm_mapping = json.dumps(
        fbgemm_genai_by_torch_minor, sort_keys=True, separators=(",", ":")
    )
    return f"""try:
    import torch; _qat_torch_minor = re.match(r"[0-9]{{1,}}\\.[0-9]{{1,}}", str(torch.__version__)).group(0)
except Exception:
    _qat_torch_minor = ""
_qat_torchao_map = {torchao_mapping}
_qat_torchao = _qat_torchao_map.get(_qat_torch_minor, "{default_torchao}")
_qat_fbgemm_map = {fbgemm_mapping}
_qat_fbgemm = _qat_fbgemm_map.get(_qat_torch_minor, "{default_fbgemm_genai}")
!pip install --upgrade --force-reinstall torchao=={{_qat_torchao}} fbgemm-gpu-genai=={{_qat_fbgemm}}"""


def update_or_append_pip_install(base_content, package_name, new_install_line):
    pattern = re.compile(rf"^!(uv )?pip install .*?{package_name}.*$", re.MULTILINE)

    updated_content, substitutions_count = pattern.subn(new_install_line, base_content)

    if substitutions_count == 0:
        output = base_content.strip() + "\n" + new_install_line
    else:
        output = updated_content
    return output

installation_content = """%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    __XFORMERS_INSTALL__
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
""".replace("__XFORMERS_INSTALL__", XFORMERS_INSTALL)
installation_content = update_or_append_pip_install(
    installation_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_content = update_or_append_pip_install(
    installation_content,
    "trl",
    PIN_TRL,
)

installation_kaggle_content = """%%capture
import os

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install --upgrade transformers "huggingface_hub>=0.34.0" "datasets==4.3.0"
"""

installation_kaggle_content = update_or_append_pip_install(
    installation_kaggle_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_kaggle_content = update_or_append_pip_install(
    installation_kaggle_content,
    "trl",
    PIN_TRL,
)

installation_grpo_content = """%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # [NEW] Extra 30% context lengths!
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install or uv pip install
    !pip install unsloth vllm
else:
    pass # For Colab / Kaggle, we need extra instructions hidden below \\/"""

installation_extra_grpo_content = r"""#@title Colab Extra Install { display-mode: "form" }
%%capture
import os
!pip install --upgrade -qqq uv
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install!
    !pip install unsloth vllm
else:
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}"""

installation_extra_grpo_content = update_or_append_pip_install(
    installation_extra_grpo_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)
installation_extra_grpo_content = update_or_append_pip_install(
    installation_extra_grpo_content,
    "trl",
    UV_PIN_TRL,
)


installation_grpo_kaggle_content = """%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # [NEW] Extra 30% context lengths!
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
!uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
!uv pip install -qqq {_triton} "huggingface_hub>=0.34.0" "datasets==4.3.0"
"""

installation_grpo_kaggle_content = update_or_append_pip_install(
    installation_grpo_kaggle_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)

installation_grpo_kaggle_content = update_or_append_pip_install(
    installation_grpo_kaggle_content,
    "trl",
    UV_PIN_TRL,
)

installation_synthetic_data_content = """%%capture
import os
!pip install --upgrade -qqq uv
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install!
    !pip install unsloth vllm synthetic-data-kit==0.0.3
else:
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}
    !uv pip install synthetic-data-kit==0.0.3"""

installation_synthetic_data_content = update_or_append_pip_install(
    installation_synthetic_data_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)

installation_synthetic_data_content = update_or_append_pip_install(
    installation_synthetic_data_content,
    "trl",
    UV_PIN_TRL,
)

installation_grpo_synthetic_data_content = """%%capture
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
!uv pip install -qqq --upgrade unsloth {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers
!uv pip install -qqq {_triton}
!uv pip install "huggingface_hub>=0.34.0" "datasets==4.3.0"
!uv pip install synthetic-data-kit==0.0.3"""
installation_grpo_synthetic_data_content = update_or_append_pip_install(
    installation_grpo_synthetic_data_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)
installation_grpo_synthetic_data_content = update_or_append_pip_install(
    installation_grpo_synthetic_data_content,
    "trl",
    UV_PIN_TRL,
)

# Add install snac under install unsloth
installation_orpheus_content = installation_content + """\n!pip install snac torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_orpheus_kaggle_content = installation_kaggle_content + """\n!pip install snac torchcodec \"datasets>=3.4.1,<4.0.0\""""

installation_whisper_content = installation_content + """\n!pip install librosa soundfile evaluate jiwer torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_whisper_kaggle_content = installation_kaggle_content + """\n!pip install librosa soundfile evaluate jiwer torchcodec \"datasets>=3.4.1,<4.0.0\""""

installation_spark_content = installation_content + """\n!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_spark_kaggle_content = installation_kaggle_content + """\n!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx torchcodec \"datasets>=3.4.1,<4.0.0\""""

installation_gpt_oss_content = r"""%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):    
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \
        "torch>=2.8.0" "triton>=3.4.0" {_numpy} {_pil} torchvision bitsandbytes "transformers==4.56.2" \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
        git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo"""

# installation_gpt_oss_content = update_or_append_pip_install(
#     installation_gpt_oss_content,
#     "transformers",
#     "!uv pip install transformers==4.56.2",
# )
# installation_gpt_oss_content = update_or_append_pip_install(
#     installation_gpt_oss_content,
#     "trl",
#     UV_PIN_TRL,
# )

installation_gpt_oss_kaggle_content = installation_gpt_oss_content

installation_oute_content = installation_content + """\n!pip install omegaconf einx
!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS
import os
os.remove("/content/OuteTTS/outetts/models/gguf_model.py")
os.remove("/content/OuteTTS/outetts/interface.py")
os.remove("/content/OuteTTS/outetts/__init__.py")
!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy torchcodec \"datasets>=3.4.1,<4.0.0\"
!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""

installation_oute_kaggle_content = installation_kaggle_content + """\n!pip install omegaconf einx
!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS
import os
os.remove("/content/OuteTTS/outetts/models/gguf_model.py")
os.remove("/content/OuteTTS/outetts/interface.py")
os.remove("/content/OuteTTS/outetts/__init__.py")
!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy torchcodec \"datasets>=3.4.1,<4.0.0\"
!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""

# Llasa Need Unsloth==2025.4.1, Transformers==4.48 to running stable, and trl ==0.15.2
# installation_llasa_content = re.sub(r'\bunsloth\b(==[\d\.]*)?', 'unsloth==2025.4.1', installation_content)
installation_llasa_content = installation_content
installation_llasa_content = re.sub(r'\btrl\b(==[\d\.]*)?', 'trl==0.15.2', installation_llasa_content)

installation_llasa_content += """\

!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps
!pip install omegaconf torchcodec \"datasets>=3.4.1,<4.0.0\"
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""
installation_llasa_content = update_or_append_pip_install(
    installation_llasa_content,
    "transformers",
    "!pip install transformers==4.56.1",
)

installation_llasa_kaggle_content = installation_kaggle_content + """\n!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps
!pip install omegaconf torchcodec \"datasets>=3.4.1,<4.0.0\"
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""
installation_llasa_kaggle_content = update_or_append_pip_install(
    installation_llasa_kaggle_content,
    "transformers",
    "!pip install transformers==4.48",
)
installation_llasa_kaggle_content = update_or_append_pip_install(
    installation_llasa_kaggle_content,
    "trl",
    PIN_TRL,
)

installation_tool_calling_content = installation_content + """\n!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg"""
installation_tool_calling_kaggle_content = installation_kaggle_content + """\n!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg"""

installation_sesame_csm_content = installation_content + """\n!pip install torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_sesame_csm_content = update_or_append_pip_install(
    installation_sesame_csm_content,
    "transformers",
    "!pip install transformers==4.52.3",
)
installation_sesame_csm_content = update_or_append_pip_install(
    installation_sesame_csm_content,
    "trl",
    PIN_TRL
)

installation_sesame_csm_kaggle_content = installation_kaggle_content + """\n!pip install torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_sesame_csm_kaggle_content = update_or_append_pip_install(
    installation_sesame_csm_kaggle_content,
    "transformers",
    "!pip install transformers==4.52.3 torchcodec",
)
installation_sesame_csm_kaggle_content = update_or_append_pip_install(
    installation_sesame_csm_kaggle_content,
    "trl",
    PIN_TRL
)

installation_llama_vision_content = installation_content
installation_llama_vision_content = update_or_append_pip_install(
    installation_llama_vision_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_llama_vision_content = update_or_append_pip_install(
    installation_llama_vision_content,
    "trl",
    PIN_TRL
)


installation_llama_vision_kaggle_content = installation_kaggle_content
installation_llama_vision_kaggle_content = update_or_append_pip_install(
    installation_llama_vision_kaggle_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_llama_vision_kaggle_content = update_or_append_pip_install(
    installation_llama_vision_kaggle_content,
    "trl",
    PIN_TRL
)

gemma3n_extra_content = """\

!pip install torchcodec
import torch; torch._dynamo.config.recompile_limit = 64;"""
installation_gemma3n_content = installation_content
installation_gemma3n_content += gemma3n_extra_content

installation_gemma3n_kaggle_content = installation_kaggle_content
installation_gemma3n_kaggle_content += gemma3n_extra_content

# Gemma 4 needs transformers==5.5.0 (with --no-deps), torchcodec, and
# torch._dynamo recompile_limit. Do NOT go through update_or_append_pip_install
# here because Gemma 4 must not get the default transformers==4.56.2 pin or
# the trl==0.22.2 --no-deps downgrade.
installation_gemma4_content = """%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)
    __XFORMERS_INSTALL__
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
!pip install --no-deps transformers==5.5.0
!pip install torchcodec
import torch; torch._dynamo.config.recompile_limit = 64;""".replace("__XFORMERS_INSTALL__", XFORMERS_INSTALL)

gemma3n_extra_content = """\

!pip install torchcodec
import torch; torch._dynamo.config.recompile_limit = 64;"""
installation_qwen3_vl_content = installation_content 
installation_qwen3_vl_content = update_or_append_pip_install(
    installation_qwen3_vl_content,
    "transformers",
    "!pip install transformers==4.57.1",
)

installation_qwen3_vl_kaggle_content  = installation_kaggle_content
installation_qwen3_vl_kaggle_content  = update_or_append_pip_install(
    installation_qwen3_vl_kaggle_content,
    "transformers",
    "!pip install transformers==4.57.1",
)

installation_qwen3_5_content = """%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \\
        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes xformers==0.0.32.post2 \\
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
# causal_conv1d is supported only on torch==2.8.0. If you have newer torch versions, please wait 10 minutes!
!uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0"""

installation_qwen3_5_kaggle_content = installation_qwen3_5_content

installation_sglang_content = """%%capture
import sys
import os
!git clone https://github.com/sgl-project/sglang.git && cd sglang && pip install -e "python[all]"
!pip install -U transformers==4.53.0
sys.path.append(f'{os.getcwd()}/sglang/')
sys.path.append(f'{os.getcwd()}/sglang/python')"""
installation_sglang_kaggle_content = installation_sglang_content

installation_deepseek_ocr_content = installation_content
installation_deepseek_ocr_content += """\n!pip install jiwer
!pip install einops addict easydict"""

installation_deepseek_ocr_kaggle_content = installation_kaggle_content
installation_deepseek_ocr_kaggle_content += """\n!pip install jiwer
!pip install einops addict easydict"""

installation_ernie_4_5_vl_content = installation_content
installation_ernie_4_5_vl_content += """\n!pip install decord"""

installation_ernie_4_5_vl_kaggle_content = installation_kaggle_content
installation_ernie_4_5_vl_kaggle_content += """\n!pip install decord"""

installation_nemotron_nano_content = """%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):    
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \\
        "torch==2.7.1" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes "transformers==4.56.2" \\
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo

# Mamba is supported only on torch==2.7.1. If you have newer torch versions, please wait 30 minutes!
!uv pip install --no-build-isolation mamba_ssm==2.2.5 causal_conv1d==1.5.2"""

installation_nemotron_nano_kaggle_content = installation_nemotron_nano_content

installation_qat_content = """%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
    __XFORMERS_INSTALL__
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
__QAT_NATIVE_INSTALL__
!pip install transformers==4.55.4 && pip install --no-deps trl==0.22.2""".replace(
    "__XFORMERS_INSTALL__", XFORMERS_INSTALL
).replace(
    "__QAT_NATIVE_INSTALL__", build_qat_native_install_block()
)
installation_qat_kaggle_content = installation_qat_content

installation_ministral_content = installation_content
installation_ministral_content = update_or_append_pip_install(
    installation_ministral_content,
    "transformers",
    "!pip install transformers==5.3.0"
)

installation_ministral_kaggle_content = installation_kaggle_content
installation_ministral_kaggle_content = update_or_append_pip_install(
    installation_ministral_kaggle_content,
    "transformers",
    "!pip install transformers==5.3.0"
)

installation_glm_flash_content = installation_content
installation_glm_flash_content = update_or_append_pip_install(
    installation_glm_flash_content,
    "transformers",
    "!pip install transformers==5.3.0"
)

installation_glm_flash_kaggle_content = installation_kaggle_content
installation_glm_flash_kaggle_content = update_or_append_pip_install(
    installation_glm_flash_kaggle_content,
    "transformers",
    "!pip install transformers==5.3.0"
)

installation_phone_content = installation_content
installation_phone_content = update_or_append_pip_install(
    installation_phone_content,
    "transformers",
    "!pip install transformers==4.57.3"
)
installation_phone_content = update_or_append_pip_install(
    installation_phone_content,
    "trl",
    "!pip install --no-deps trl==0.25.1"
)
installation_phone_content += """\n!pip install torchao==0.15.0 optimum==1.24.0 pytorch-tokenizers executorch==1.1.0
!pip install git+https://github.com/huggingface/optimum-executorch.git@v0.1.0 --no-deps"""

installation_phone_kaggle_content = installation_kaggle_content
installation_phone_kaggle_content = update_or_append_pip_install(
    installation_phone_kaggle_content,
    "transformers",
    "!pip install transformers==4.57.3"
)
installation_phone_kaggle_content = update_or_append_pip_install(
    installation_phone_kaggle_content,
    "trl",
    "!pip install --no-deps trl==0.25.1"
)
installation_phone_kaggle_content += """\n!pip install torchao==0.15.0 optimum==1.24.0 pytorch-tokenizers executorch==1.1.0
!pip install git+https://github.com/huggingface/optimum-executorch.git@v0.1.0 --no-deps"""

OTHER_RESOURCES = """Some other resources:
1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install-and-update) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
"""

text_for_last_cell_gguf = """And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

__OTHER_RESOURCES__
<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️

  <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
</div>""".replace("__OTHER_RESOURCES__", OTHER_RESOURCES)

text_for_last_cell_ollama = text_for_last_cell_gguf.replace("Now, ", "You can also ", 1)

text_for_last_cell_non_gguf = """And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

__OTHER_RESOURCES__
<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️

  This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
</div>""".replace("__OTHER_RESOURCES__", OTHER_RESOURCES)

# Pre-compiled regex patterns (used in hot paths per-cell per-notebook)
_RE_HTML_TAGS = re.compile(r'<[^>]+>')
_RE_URLS = re.compile(r'https?://\S+')
_RE_MD_LINKS = re.compile(r'\[([^\]]*)\]\([^\)]*\)')
_RE_ENGLISH_WORDS = re.compile(r'\b[a-zA-Z]{3,}\b')
_RE_DOUBLE_EXCL = re.compile(r"!{2,}")
_RE_VERSION = re.compile(r"[\d]{4}\.[\d]{1,2}\.[\d]{1,2}([^\d])")
_RE_PACKING = re.compile(
    r"(\n[ \t]*)packing\s*=\s*(True|False).*?\n(\1args\s*=\s*SFTConfig\(\n)"
)
_RE_GGUF_USAGE = re.compile(
    r"Now, use the `[^`]+\.Q8_0\.gguf` file or `[^`]+\.Q4_K_M\.gguf` file in llama\.cpp\."
)
_RE_HUGGINGFACE_BACKTICK = re.compile(r"Huggingface  (`[^`]+`)")
_RE_NOCOMMIT = re.compile(r'\[@nocommit[^\]]*\]\([^\)]*\)\.?')
_RE_FOOTER_NUM = re.compile(r'\n6\. See notebooks for DPO')
_RE_DUP_DOCS = re.compile(
    r'(See \[our docs\]\([^)]+\) for more deployment options\.)\s*\1'
)
_RE_NEMO_GYM = re.compile(r'\bNemo Gym\b')
_RE_SAVE_GGUF = re.compile(
    r"(save_pretrained_gguf\(\s*)([\"\'])([^\"\']*)([\"\'])",
    re.DOTALL,
)
_RE_PUSH_GGUF = re.compile(
    r"(push_to_hub_gguf\(\s*)([\"\'])([^\"\']*)([\"\'])",
    re.DOTALL,
)
_RE_SAVE_MERGED = re.compile(
    r"(save_pretrained_merged\(\s*)([\"\'])([^\"\']*)([\"\'])(.*?save_method\s*=\s*[\"\'])(merged_16bit|merged_4bit|mxfp4)([\"\'])",
    re.DOTALL,
)
_RE_PUSH_MERGED = re.compile(
    r"(push_to_hub_merged\(\s*)([\"\'])([^\"\']*)([\"\'])(.*?save_method\s*=\s*[\"\'])(merged_16bit|merged_4bit|mxfp4)([\"\'])",
    re.DOTALL,
)
_RE_SAVE_LORA = re.compile(
    r"(\b(?:model|tokenizer|processor)\.save_pretrained\(\s*)([\"\'])([^\"\']*)([\"\'])"
)
_RE_PUSH_LORA = re.compile(
    r"(\b(?:model|tokenizer|processor)\.push_to_hub\(\s*)([\"\'])([^\"\']*)([\"\'])"
)
_RE_LORA_LOAD = re.compile(
    r"(model_name\s*=\s*)([\"\'])([^\"\']*)([\"\'])([^\n]*YOUR MODEL YOU USED FOR TRAINING)"
)
_RE_LORA_LOAD2 = re.compile(
    r"([\"\'])([^\"\']*)([\"\'])([^\n]*YOUR MODEL YOU USED FOR TRAINING)"
)
_RE_LORA_MODEL = re.compile(r"([\"\'])lora_model([\"\'])")
_RE_FINETUNED_MODEL = re.compile(r"([\"\'])finetuned_model([\"\'])")
_RE_AUTO_LORA = re.compile(
    r"(Auto(?:PeftModel\w*|Tokenizer|Model\w*)\.from_pretrained\(\s*)([\"\'])([^\"\']*_lora[^\"\']*)([\"\'])"
)
_RE_TOKEN = re.compile(r'(\btoken\s*=\s*)([\"\'])([^\"\']*)([\"\'])')
_RE_EOS_TOKEN = re.compile(r"unsloth_eos_token\s*=\s*[\"\']YOUR_HF_TOKEN[\"\']")
_RE_PATCH_TOKEN = re.compile(r"patch_token\s*=\s*[\"\']YOUR_HF_TOKEN[\"\']")
_RE_DTYPE_LINE = re.compile(r"^[ \t]*dtype\s*=\s*None\s*#.*$")
_RE_DTYPE_PARAM = re.compile(r"(\bdtype\s*=\s*)dtype\b\s*,?")
_RE_VLLM = re.compile(r'\bvLLM\b')
_RE_VLLM_UPPER = re.compile(r'\bVLLM\b(?!_)')
_RE_GATED_COMMENT = re.compile(r"# use one if using gated models.*")
_RE_GEMMA3N = re.compile(r"gemma[-_]?3n")
_RE_GEMMA3 = re.compile(r"gemma[-_]?3")
_RE_GEMMA4 = re.compile(r"gemma[-_]?4")
_RE_STOP_BRACKET = re.compile(r"[\(\[\{]")
_RE_MULTI_UNDERSCORE = re.compile(r"__+")
_RE_ALPHA_ONLY = re.compile(r"[A-Za-z]+")
_RE_ALPHA_DIGIT = re.compile(r"[A-Za-z][0-9]")
_RE_ALPHA_LEAD = re.compile(r"[A-Za-z]+")
# Global fix pass patterns
_RE_GATED_GLOBAL = re.compile(r"# use one if using gated models[^\n]*")
_RE_HUGGINGFACE_GLOBAL = re.compile(r"Huggingface  (`[^`]+`)")
_RE_NOCOMMIT_GLOBAL = re.compile(r'\[@nocommit[^\]]*\]\([^\)]*\)\.?')
_RE_FOOTER_NUM_NL = re.compile(r'\n6\. See notebooks for DPO')
_RE_FOOTER_NUM_Q = re.compile(r'"6\. See notebooks for DPO')
_RE_DUP_DOCS_GLOBAL = re.compile(
    r'(See \[our docs\]\([^)]+\) for more deployment options\.)\s*\1'
)
_RE_TRANSFORMERS_V5_PIN = re.compile(
    r"(?<![A-Za-z0-9_.-])transformers\s*==\s*5(?:\.\d+){0,2}(?![A-Za-z0-9_.-])"
)
_RE_TRANSFORMERS_V5_PIN_CAPTURE = re.compile(
    r"(?<![A-Za-z0-9_.-])transformers\s*==\s*(5(?:\.\d+){0,2})(?![A-Za-z0-9_.-])"
)

_TARGET_TRANSFORMERS_V5 = "5.3.0"

def _parse_version_tuple(v):
    """Parse '5.3.0' into (5, 3, 0) for comparison."""
    return tuple(int(x) for x in v.split("."))

def _normalize_transformers_v5_pin(text):
    """Normalize transformers 5.x pins to the target version, but never downgrade."""
    target = _parse_version_tuple(_TARGET_TRANSFORMERS_V5)
    def _replace(m):
        existing = m.group(1)
        if _parse_version_tuple(existing) > target:
            return m.group(0)  # keep the higher version
        return f"transformers=={_TARGET_TRANSFORMERS_V5}"
    return _RE_TRANSFORMERS_V5_PIN_CAPTURE.sub(_replace, text)


_ALL_NB_FIXES = {
    "fibonnaci": "fibonacci",
    "Fibonnaci": "Fibonacci",
    "SHould": "Should",
    "GTP-OSS": "GPT-OSS",
    "stratgegy": "strategy",
    "verifer": "verifier",
    "verisons": "versions",
    "datases": "datasets",
    "Huggingface's": "Hugging Face's",
    "Huggingface TRL's": "Hugging Face TRL's",
    "Prime and Prejudice": "Pride and Prejudice",
    "2x Telsa T4s": "2x Tesla T4s",
    "float32 s disable": "float32 so disable",
    "and its amazing": "and it's amazing",
    "look like this:": "looks like this:",
    "AutoModelForPeftCausalLM": "AutoPeftModelForCausalLM",
    "<|start_of_role|>user|end_of_role|>": "<|start_of_role|>user<|end_of_role|>",
    # New fixes
    "[Open Math Reasoning]()": "[Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)",
    "Some other links:": "Some other resources:",
    "unsloth.ai/docs/get-started/installing-+-updating": "unsloth.ai/docs/get-started/install",
    "unsloth.ai/docs/get-started/install-and-update": "unsloth.ai/docs/get-started/install",
    # Also handle old domain format that may be in exception files
    "docs.unsloth.ai/get-started/installing-+-updating": "unsloth.ai/docs/get-started/install",
    "docs.unsloth.ai/get-started/install-and-update": "unsloth.ai/docs/get-started/install",
    # Handle intermediate format (domain changed but path not)
    "unsloth.ai/get-started/installing-+-updating": "unsloth.ai/docs/get-started/install",
    "unsloth.ai/get-started/install-and-update": "unsloth.ai/docs/get-started/install",
    "Nemo Gym": "NeMo Gym",
    # Fix old domain for exception files
    "https://docs.unsloth.ai/": "https://unsloth.ai/docs/",
    # Fix ExecuTorch dangling sentence left after @nocommit removal
    "ExecuTorch.  Follow the directions \\n": "ExecuTorch.\\n",
}

ARCHITECTURE_MAPPING = {
    # Gemma Family
    # NOTE: "gemma4" must appear before "gemma" so that the longest-key-first
    # match in extract_model_info_refined routes Gemma 4 notebooks to their
    # own section. Other Gemma* notebooks (Gemma3, Gemma3N, Gemma2,
    # FunctionGemma, EmbeddingGemma, CodeGemma) still resolve to "Gemma".
    "gemma4": "Gemma 4",
    "gemma": "Gemma",
    "codegemma": "Gemma", # Explicitly map specific models if needed

    # Llama Family
    "llama": "Llama",
    "tinylama": "Llama",

    # Qwen Family
    "qwen": "Qwen",

    # Phi Family
    "phi": "Phi",

    # Mistral Family
    "mistral": "Mistral",
    "pixtral": "Mistral",
    "zephyr": "Mistral",
    "Magistral" : "Mistral",
    "Ministral" : "Mistral",

    # Whisper
    "whisper": "Speech-to-Text (STT)",

    # Text-to-Speech Models (Group or keep separate?)
    "oute": "Text-to-Speech (TTS)",
    "llasa": "Text-to-Speech (TTS)",
    "spark": "Text-to-Speech (TTS)",
    "orpheus": "Text-to-Speech (TTS)",
    "sesame": "Text-to-Speech (TTS)",

    # gpt oss
    "gpt oss": "GPT-OSS",

    # Hybrid Attention (SSM / linear-attention hybrids, Mamba-style models)
    "falcon": "Hybrid Attention",
    "liquid": "Hybrid Attention",
    "lfm": "Hybrid Attention",

    # Deepseek
    "deepseek": "Deepseek",

    # Granite
    "granite": "Granite",

    # ERNIE
    "ernie": "ERNIE",

    # Nemotron
    "nemotron": "Nemotron",

    # Paddle
    "paddle": "Paddle",

    # GLM
    "glm": "GLM",

    # Bert
    "bert": "BERT",
    "modernbert": "BERT",
    "bge": "Embedding",
    "minilm": "Embedding",

    # Other Models (Assign architecture or keep specific)
    # 'codeforces': 'CodeForces Model', # Example
    # 'unsloth': 'Unsloth Model',     # Example
    "meta synthetic data": "Llama",
}

TYPE_MAPPING = {
    "Gemma3N" : {
        "Conversational" : "Multimodal"
    },
    "Meta Synthetic Data" : {
        "Synthetic Data" : "GRPO",
        "GRPO LoRA" : "GRPO"
    },
}

KNOWN_TYPES_ORDERED = [
    "Tool Calling",          
    "Text Completion",       
    "Synthetic Data",        
    "Reasoning Conversational",
    "Vision GRPO",
    "Fine Tuning",
    "500K Context",
    "QAT",
    
    "Conversational",
    "Alpaca",
    "Vision",
    "Reasoning",
    "Completion",
    "Finetune",             
    "Studio",               
    "Coder",                
    "Inference",            
    "Ollama",               
    "Audio",                
    "Thinking",

    # FP8 GRPO
    "FP8 GRPO",

    # GPT OSS
    "GRPO 2048",
    "GRPO Sudoku",
    
    "ORPO",
    "GRPO",
    "DPO",
    "CPT",
    "TTS",                  
    "LoRA",
    "VL",                   
    "RAFT",

    # Deepseek OCR
    "Evaluation",
    "Eval",

    # BERT, ModernBERT,
    "Classification",
    "Mobile Actions",
]

# Notebooks excluded from automated updates. Each entry is preserved
# exactly as the author wrote it (custom install cells, bespoke content, etc.).
DONT_UPDATE_EXCEPTIONS = [
    "Falcon_H1-Alpaca.ipynb",                                         # Custom Falcon H1 hybrid architecture setup
    "Liquid_LFM2-Conversational.ipynb",                                # Custom Liquid Foundation Model install
    "Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb",                          # Hand-tuned advanced GRPO notebook
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb",            # Custom GPT-OSS RL environment
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb",  # DGX Spark variant of GPT-OSS RL
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb",       # BF16 variant of GPT-OSS RL
    "Qwen3_VL_(8B)-Vision-GRPO.ipynb",                                 # Vision GRPO with custom reward function
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb",    # OpenEnv variant of GPT-OSS RL
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb", # OpenEnv BF16 variant
    "Synthetic_Data_Hackathon.ipynb",                                  # Hackathon-specific notebook
    "Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb",       # Custom Sudoku RL environment
]

# Notebooks excluded from automatic README.md listing. You can use a basename,
# repo-relative path, or absolute path.
README_SKIP_NOTEBOOKS = [
    "Meta-Synthetic-Data-Llama3.1_(8B).ipynb",
    "Meta_Synthetic_Data_Llama3_2_(3B).ipynb"
]

# Per-notebook overrides for the Model column in README.md tables. Keyed by
# the on-disk basename (with .ipynb). The value is the literal Markdown text
# rendered between the surrounding ** ** bold markers, so HTML tags such as
# <br> can be embedded for multi-line cells. Only set this for notebooks
# whose computed model name is too long to fit on a single README row.
README_MODEL_NAME_OVERRIDES = {
    "CodeForces-cot-Finetune_for_Reasoning_on_CodeForces.ipynb":
        "CodeForces cot Finetune<br>for Reasoning on CodeForces",
    "Kaggle-CodeForces-cot-Finetune_for_Reasoning_on_CodeForces.ipynb":
        "CodeForces cot Finetune<br>for Reasoning on CodeForces",
}


FIRST_MAPPING_NAME = {
    "gpt-oss-(20B)-Fine-tuning.ipynb" : "gpt_oss_(20B)-Fine-tuning.ipynb",
    "Qwen2_5_7B_VL_GRPO.ipynb" : "Qwen2.5_VL_(7B)-Vision-GRPO.ipynb",
    "Qwen3_(4B)-Instruct.ipynb" : "Qwen3_(4B)-Conversational.ipynb",
    "Qwen3_(4B)_Instruct-QAT.ipynb" : "Qwen3_(4B)-QAT.ipynb",

    # GPT OSS 
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb" : "(DGX Spark)-gpt-oss-(20B)-GRPO-2048.ipynb",
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb" : "gpt-oss-(20B)-GRPO-2048.ipynb",
    "Deepseek_OCR_(3B).ipynb" : "Deepseek_OCR_(3B)-Fine-Tuning.ipynb",
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb" : "(OpenEnv)-gpt-oss-BF16-(20B)-GRPO-2048.ipynb",
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb" : "gpt-oss-BF16-(20B)-GRPO-2048.ipynb",
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb" : "(OpenEnv)-gpt-oss-(20B)-GRPO-2048.ipynb",
    "GPT_OSS_BNB_(20B)-Inference.ipynb" : "gpt-oss-BNB-(20B)-Inference.ipynb",
    "GPT_OSS_MXFP4_(20B)-Inference.ipynb" : "gpt-oss-MXFP4-(20B)-Inference.ipynb",
    "gpt_oss_(20B)_500K_Context_Fine_tuning" : "gpt_oss_(20B)-500K-Context.ipynb",

    # Gemma
    "Gemma3_(4B).ipynb" : "Gemma3_(4B)-Conversational.ipynb",
    "Gemma3_(270M).ipynb" : "Gemma3_(270M)-Conversational.ipynb",
    # Gemma 4 Text notebooks: the on-disk filenames use the "-Text" suffix
    # which is not a known type, so the README row generator would render
    # them with an empty Type column. Map them to "-Conversational" so the
    # type extractor picks "Conversational" (matching how Gemma 3 Text
    # notebooks are labelled).
    "Gemma4_(E2B)-Text.ipynb" : "Gemma4_(E2B)-Conversational.ipynb",
    "Gemma4_(E4B)-Text.ipynb" : "Gemma4_(E4B)-Conversational.ipynb",
    "Gemma4_(31B)-Text.ipynb" : "Gemma4_(31B)-Conversational.ipynb",
    "Gemma4_(26B_A4B)-Text.ipynb" : "Gemma4_(26B_A4B)-Conversational.ipynb",

    # Granite
    "Granite4.0_350M.ipynb" : "Granite4.0_(350M)-Conversational.ipynb",
    "Granite4.0.ipynb" : "Granite4.0_(3B)-Conversational.ipynb",

    # Bert
    "bert_classification.ipynb" : "ModernBERT_(Large)-Classification.ipynb",

    # Whisper
    "Whisper.ipynb" : "Whisper_(Large)-Fine-Tuning.ipynb",

    # Spark
    "Spark_TTS_(0_5B).ipynb" : "Spark_TTS_(0.5B)-TTS.ipynb",

    # FP8
    "Qwen3_8B_FP8_GRPO.ipynb" : "Qwen3_(8B)-FP8-GRPO.ipynb",
    "Llama_FP8_GRPO.ipynb" : "Llama3.2_(1B)-FP8-GRPO.ipynb",

    # Ministral
    "Ministral_3_VL_(3B)_Vision.ipynb" : "Ministral3_VL_(3B)-Vision.ipynb",
    "Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb" : "Ministral3_(3B)-GRPO-Sudoku.ipynb",

    # FunctionGemma
    "FunctionGemma_(270M).ipynb" : "FunctionGemma_(270M)-Conversational.ipynb",
    "FunctionGemma_(270M)-LMStudio.ipynb" : "FunctionGemma_(270M)-Inference.ipynb",
}


def _set_file_permissions(filepath):
    """Set file permissions to 0o644 on non-Windows platforms."""
    if platform.system() != "Windows":
        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def _should_skip_readme_notebook(path):
    """Return True when a notebook is configured to be omitted from README.md."""
    normalized_path = path.replace("\\", "/")
    absolute_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), normalized_path)
    ).replace("\\", "/")
    basename = os.path.basename(normalized_path)
    base_variants = {
        normalized_path,
        absolute_path,
        basename,
    }

    if basename.startswith("Kaggle-"):
        kaggle_stripped = basename[len("Kaggle-"):]
        base_variants.add(kaggle_stripped)
        base_variants.add(
            normalized_path[: -len(basename)] + kaggle_stripped
        )
        base_variants.add(
            absolute_path[: -len(basename)] + kaggle_stripped
        )

    for skipped in README_SKIP_NOTEBOOKS:
        normalized_skipped = skipped.replace("\\", "/")
        if normalized_skipped in base_variants:
            return True

    return False


_NOTEBOOK_FORMAT_CACHE = {}
_ORIGINAL_OUTPUTS_CACHE = {}


def _detect_notebook_indent(filepath):
    """Detect the JSON indent level used in a notebook file."""
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            for line in f:
                stripped = line.lstrip()
                if stripped and not stripped.startswith("{"):
                    indent = len(line) - len(stripped)
                    return indent if indent > 0 else 1
    except (OSError, UnicodeDecodeError):
        pass
    return 1


def _file_has_trailing_newline(filepath):
    """Check if a file ends with a newline character."""
    try:
        with open(filepath, "rb") as f:
            f.seek(-1, 2)
            return f.read(1) == b"\n"
    except OSError:
        return True


def _cache_notebook_format(filepath):
    """Cache the original indent and EOF-newline of a notebook file (first call wins)."""
    if filepath not in _NOTEBOOK_FORMAT_CACHE:
        _NOTEBOOK_FORMAT_CACHE[filepath] = (
            _detect_notebook_indent(filepath),
            _file_has_trailing_newline(filepath),
        )
    return _NOTEBOOK_FORMAT_CACHE[filepath]


def _source_lines(text):
    """Split text into a Jupyter source array.

    CRLF sequences are normalized to LF before splitting.
    Each line keeps its trailing ``\\n`` except the very last one,
    matching the convention used by ``nbformat``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def _write_notebook(filepath, content):
    """Write notebook JSON, preserving original indent and trailing newline."""
    indent, trailing_nl = _cache_notebook_format(filepath)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        json.dump(content, f, indent=indent, ensure_ascii=False)
        if trailing_nl:
            f.write("\n")
    _set_file_permissions(filepath)


def _ensure_cell_ids(notebook_content):
    """Ensure every notebook cell has an id (required by newer nbformat validation)."""
    changed = False
    for idx, cell in enumerate(notebook_content.get("cells", [])):
        if not isinstance(cell, dict):
            continue
        if not cell.get("id"):
            src = "".join(cell.get("source", []))
            cell["id"] = hashlib.md5(f"{idx}:{src}".encode()).hexdigest()[:12]
            changed = True
    return changed


def _cache_original_outputs(filepath):
    """Cache output cells, widget state, and cell IDs from a notebook before it is overwritten (first call wins)."""
    if filepath not in _ORIGINAL_OUTPUTS_CACHE:
        try:
            with open(filepath, "r", encoding="utf-8", newline="") as f:
                nb = json.load(f)
            cells = nb.get("cells", [])
            outputs = {idx: cell["outputs"] for idx, cell in enumerate(cells) if cell.get("outputs")}
            widget_state = nb.get("metadata", {}).get("widgets", None)
            cell_ids = {idx: cell["id"] for idx, cell in enumerate(cells) if cell.get("id")}
            _ORIGINAL_OUTPUTS_CACHE[filepath] = (len(cells), outputs, widget_state, cell_ids)
        except Exception:
            _ORIGINAL_OUTPUTS_CACHE[filepath] = (0, {}, None, {})


def _restore_original_outputs(filepath):
    """Restore output cells, widget state, and cell IDs from the cached original if cell count matches."""
    if filepath not in _ORIGINAL_OUTPUTS_CACHE:
        return
    orig_count, orig_outputs, orig_widgets, orig_cell_ids = _ORIGINAL_OUTPUTS_CACHE[filepath]
    if not orig_outputs and orig_widgets is None and not orig_cell_ids:
        return
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            nb = json.load(f)
        if len(nb.get("cells", [])) != orig_count:
            return
        for idx, outputs in orig_outputs.items():
            if idx < len(nb["cells"]):
                nb["cells"][idx]["outputs"] = outputs
        for idx, cell_id in orig_cell_ids.items():
            if idx < len(nb["cells"]):
                nb["cells"][idx]["id"] = cell_id
        if orig_widgets is not None:
            nb.setdefault("metadata", {})["widgets"] = orig_widgets
        elif "widgets" in nb.get("metadata", {}):
            del nb["metadata"]["widgets"]
        _write_notebook(filepath, nb)
    except Exception:
        pass


def _normalize_lgpl_blank_line(filepath):
    """Ensure a blank line before the LGPL marker in the last cell's source array."""
    lgpl_marker = "This notebook and all Unsloth notebooks are licensed [LGPL-3.0]"
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            nb = json.load(f)
        last_cell = nb.get("cells", [{}])[-1]
        if last_cell.get("cell_type") != "markdown":
            return
        source = last_cell.get("source", [])
        for j, line in enumerate(source):
            if lgpl_marker in line and j > 0 and source[j - 1] != "\n":
                source.insert(j, "\n")
                _write_notebook(filepath, nb)
                return
    except Exception:
        pass


def _rmtree_robust(path):
    """Remove a directory tree, handling read-only files on Windows."""
    def _on_error(func, fpath, exc_info):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)
    def _on_exc(func, fpath, exc):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_exc)
    else:
        shutil.rmtree(path, onerror=_on_error)


SPELL_IGNORE_WORDS = {
    "unsloth", "qwen", "llama", "gemma", "lora", "gguf", "vllm", "grpo",
    "kaggle", "colab", "alpaca", "qlora", "peft", "sft", "dpo", "orpo",
    "bnb", "bitsandbytes", "xformers", "triton", "cuda", "pytorch",
    "tokenizer", "huggingface", "finetune", "finetuning", "bf16", "fp16",
    "fp8", "int4", "int8", "eos", "vram", "gpu", "cpu", "trl", "sdpa",
    "ipynb", "ggml", "ollama", "mistral", "deepseek", "pixtral", "qat",
    "nemotron", "magistral", "ministral", "granite", "ernie", "bert",
    "roberta", "xlm", "matmul", "autocast", "dtype", "warmup",
    "pretrained", "instruct", "mergekit", "wandb", "tensorboard", "lmstudio",
    "venv", "conda", "repo", "param",
    "numpy", "scipy", "sklearn", "tokenizers", "datasets",
    "checkpointing", "logits", "softmax", "quantized", "quantize",
    "quantization", "backprop", "embeddings", "hyperparameters", "trainable",
    "nemo", "nvidia", "multimodal", "env", "linux", "macos", "runpod",
    "eval", "cot", "codeforces", "completions",
    # HTML/markdown tags and attributes commonly found in notebooks
    "img", "src", "href", "div", "png", "svg", "alt", "https", "http",
    "html", "css", "url", "readme", "github", "runtime", "cpp", "natively",
    "pretraining", "finetunes", "tts", "llms", "vlm", "vlms", "gpt", "oss",
    "dataset", "nli", "finetuned", "tutoring", "tutored",
    "unslothai", "nbsp", "executorch", "regex",
    "prequantized", "prepend", "prepended", "hugging", "submodule",
    "repo", "repos", "txt", "csv", "json", "yaml", "toml",
    "subfolder", "subdirectory", "gradio", "chatbot", "natively",
    # Common words in notebooks that are valid but not in dictionary
    "etc", "pre", "multi", "chatml", "vicuna", "labonne", "maxime",
    "maths", "tokenized", "workflow", "functiongemma", "templating",
    "tomaarsen", "miriad", "langid", "bahasa",
    "electroglyph", "runpod",
    # GitHub usernames, package names, tech terms
    "willccbb", "sglang", "thytu", "vicgalle", "kadirnar", "saibo",
    "etherl", "mithex", "pydantic", "scikit", "jsonl", "docstrings",
    "tokenization", "tokenize", "prepending", "customizable", "chatbots",
    "modelfile", "subprocess", "app", "bot", "dict", "globals", "configs",
    "shouldn", "backticks", "analyse", "filepath", "pclass", "skp",
    "pte", "uncomment", "entrypoint", "pid", "resize",
    "alibaba", "moby", "ebooks", "pdf", "ppt", "docx", "num",
    "doesn", "removeprefix", "multiturn", "rechne", "direkt", "ich",
}

SPELL_KNOWN_FIXES = {
    "Optinal": "Optional",
    "trainig": "training",
    "competive": "competitive",
    "whicht": "which",
    "simpilicity": "simplicity",
    "managable": "manageable",
    "randomnly": "randomly",
    "enclused": "enclosed",
    "effecient": "efficient",
    "fibonnaci": "fibonacci",
    "Fibonnaci": "Fibonacci",
    "SHould": "Should",
    "GTP-OSS": "GPT-OSS",
    "stratgegy": "strategy",
    "verifer": "verifier",
    "verisons": "versions",
    "datases": "datasets",
    "optimisations": "optimizations",
    "initialised": "initialized",
    "optimisation": "optimization",
    "initialise": "initialize",
}


def check_spelling(notebook_content, notebook_name, spell=None):
    """Check spelling in markdown cells and code comments. Auto-fix known misspellings."""
    if spell is None:
        spell = SpellChecker()
        spell.word_frequency.load_words(SPELL_IGNORE_WORDS)
    issues = []
    fixed = False
    for i, cell in enumerate(notebook_content.get("cells", [])):
        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]
        text = "".join(source)

        # Apply known fixes
        new_text = text
        for wrong, right in SPELL_KNOWN_FIXES.items():
            if wrong in new_text:
                new_text = new_text.replace(wrong, right)
        if new_text != text:
            cell["source"] = _source_lines(new_text)
            fixed = True

        # Check for unknown misspellings in markdown cells (use new_text which has known fixes applied)
        if cell.get("cell_type") == "markdown":
            # Strip HTML tags and URLs before extracting words
            clean_text = _RE_HTML_TAGS.sub(' ', new_text)
            clean_text = _RE_URLS.sub(' ', clean_text)
            clean_text = _RE_MD_LINKS.sub(r'\1', clean_text)
            words = _RE_ENGLISH_WORDS.findall(clean_text)
            # Filter out code identifiers (camelCase, snake_case, ALL_CAPS)
            english_words = [
                w for w in words
                if w == w.lower() or w == w.capitalize()
            ]
            lower_words = [w.lower() for w in english_words]
            misspelled = spell.unknown(lower_words)
            misspelled -= SPELL_IGNORE_WORDS
            if misspelled:
                issues.append((i, misspelled))
    return fixed, issues


def validate_notebook_syntax(notebook_path):
    """Validate Python syntax of all code cells in a notebook."""
    try:
        with open(notebook_path, "r", encoding="utf-8", newline="") as f:
            nb = json.load(f)
    except Exception:
        return []

    errors = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        # Remove IPython magics and shell commands for AST parsing
        # Replace with 'pass' to avoid empty blocks (e.g., if COLAB: !pip install)
        clean_lines = []
        in_shell_continuation = False
        in_cell_magic = False
        shell_block_indent = ""
        for line in source.splitlines():
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            if in_cell_magic:
                clean_lines.append(shell_block_indent + "pass")
                continue
            if in_shell_continuation:
                clean_lines.append(shell_block_indent + "pass")
                in_shell_continuation = line.rstrip().endswith("\\")
                if not in_shell_continuation:
                    shell_block_indent = ""
                continue
            if stripped.startswith("%%"):
                shell_block_indent = indent
                clean_lines.append(shell_block_indent + "pass")
                in_cell_magic = True
                continue
            if stripped.startswith(("!", "%")):
                shell_block_indent = indent
                clean_lines.append(shell_block_indent + "pass")
                in_shell_continuation = line.rstrip().endswith("\\")
                if not in_shell_continuation:
                    shell_block_indent = ""
                continue
            clean_lines.append(line)
        clean_source = "\n".join(clean_lines)

        if not clean_source.strip():
            continue

        try:
            ast.parse(clean_source)
        except SyntaxError as e:
            errors.append((i, e.lineno, str(e)))

    return errors


_RE_FAST_INFERENCE_TRUE = re.compile(r"\bfast_inference\s*=\s*true\b", re.IGNORECASE)
_RE_INSTALL_SECTION_MD = re.compile(r"\b(installation|install|setup)\b", re.IGNORECASE)


def _cell_source_text(cell):
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return str(source)


def _is_install_like_cell(cells, idx, source_text):
    lower = source_text.lower()
    if "pip install" in lower or "uv pip install" in lower or "pip3_autoremove" in lower:
        return True
    prev_md = ""
    if idx > 0 and cells[idx - 1].get("cell_type") == "markdown":
        prev_md = _cell_source_text(cells[idx - 1])
    if _RE_INSTALL_SECTION_MD.search(prev_md):
        return True
    # Most install blocks live near the top and include setup/capture boilerplate.
    if idx <= 6 and ("%%capture" in lower or "colab" in lower) and ("unsloth" in lower or "pip" in lower):
        return True
    return False


def _validate_vllm_install_usage(notebook_path):
    try:
        with open(notebook_path, "r", encoding="utf-8", newline="") as f:
            nb = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"WARNING: Could not read or parse notebook '{notebook_path}': {e}")
        return None

    cells = nb.get("cells", [])
    install_vllm_cells = []
    has_fast_inference_true = False
    has_vllm_mention_outside_install = False

    for idx, cell in enumerate(cells):
        text = _cell_source_text(cell)
        lower = text.lower()
        if _RE_FAST_INFERENCE_TRUE.search(text):
            has_fast_inference_true = True
        if "vllm" not in lower:
            continue
        if cell.get("cell_type") == "code" and _is_install_like_cell(cells, idx, text):
            install_vllm_cells.append(idx)
        else:
            has_vllm_mention_outside_install = True

    if install_vllm_cells and not (has_fast_inference_true or has_vllm_mention_outside_install):
        return {
            "notebook": os.path.basename(notebook_path),
            "cells": install_vllm_cells,
        }
    return None


def _assert_vllm_install_usage_or_fast_inference(notebook_files, max_workers=1, executor_type="process"):
    issues = [
        issue for issue in _map_with_executor(
            _validate_vllm_install_usage,
            notebook_files,
            max_workers=max_workers,
            executor_type=executor_type,
            progress_desc="Validate vllm usage",
        )
        if issue is not None
    ]

    if not issues:
        return

    print("\nERROR: Found notebooks with vllm install cells but no fast_inference=True and no non-install vllm usage:")
    for issue in issues:
        print(f"  - {issue['notebook']} (install cells: {issue['cells']})")
    raise RuntimeError("vllm install validation failed")


def _get_base_name_from_filename(filename):
    """Extract a base name from the notebook filename for dynamic model naming."""
    name = os.path.splitext(os.path.basename(filename))[0]
    for prefix in ("Kaggle-", "HuggingFace Course-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    lower = name.lower()
    if _RE_GEMMA3N.match(lower):
        return "gemma_3n"
    if _RE_GEMMA3.match(lower):
        return "gemma_3"
    if _RE_GEMMA4.match(lower):
        return "gemma_4"

    stop_match = _RE_STOP_BRACKET.search(name)
    trimmed = name[:stop_match.start()] if stop_match else name
    trimmed = trimmed.strip(" _-") or name

    segments = re.split(r"[^A-Za-z0-9]+", trimmed)
    segments = [s for s in segments if s]
    if not segments:
        base = trimmed.lower()
        base = base.replace("-", "_")
        base = _RE_MULTI_UNDERSCORE.sub("_", base)
        return base.strip("_")

    max_len = 24
    parts = []
    for seg in segments:
        if _RE_ALPHA_ONLY.fullmatch(seg):
            token = seg.lower()
        elif _RE_ALPHA_DIGIT.fullmatch(seg):
            token = seg.lower()
        else:
            if not parts:
                lead = _RE_ALPHA_LEAD.match(seg)
                if lead:
                    token = lead.group(0).lower()
                    parts.append(token)
            break
        candidate = "_".join(parts + [token]) if parts else token
        if len(candidate) <= max_len:
            parts.append(token)
        else:
            break

    base = "_".join(parts) if parts else segments[0].lower()
    return base


def _strip_extra_trailing_blank_lines(lines):
    """Remove consecutive trailing blank lines, keeping at most one."""
    while len(lines) > 1 and lines[-1].strip() == "" and lines[-2].strip() == "":
        lines.pop()
    return lines


def _space_equals_in_code(text):
    """Add spaces around = in code, but preserve compound operators (+=, -=, etc.)."""
    # Characters that form compound assignment operators when followed by =
    # e.g., +=, -=, *=, /=, //=, **=, %=, |=, &=, ^=, :=, @=
    COMPOUND_OP_CHARS = ("+", "-", "*", "/", "%", "|", "&", "^", ":", "@")

    new_lines = []
    in_shell_command = False
    for line in text.splitlines(True):
        stripped = line.lstrip()
        # Track multi-line shell commands (lines starting with ! or continuations)
        if stripped.startswith("!"):
            in_shell_command = True
        # Skip shell commands - they have their own syntax (pip URLs, version specs, etc.)
        # Also skip lines containing URL fragments like #subdirectory= or #egg=
        if in_shell_command or "#subdirectory=" in line or "#egg=" in line:
            new_lines.append(line)
            # Check if this line continues (ends with backslash)
            if in_shell_command and not line.rstrip().endswith("\\"):
                in_shell_command = False
            continue
        in_quote = None
        escaped = False
        out = []
        for i, ch in enumerate(line):
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if in_quote:
                out.append(ch)
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ("\"", "'"):
                out.append(ch)
                in_quote = ch
                continue

            if ch == "=":
                prev_char = line[i - 1] if i > 0 else ""
                next_char = line[i + 1] if i + 1 < len(line) else ""
                # Don't add space before = if it's part of ==, <=, >=, !=
                # or a compound operator like +=, -=, *=, /=, etc.
                if prev_char not in ("=", "<", ">", "!") and prev_char not in COMPOUND_OP_CHARS and next_char != "=":
                    if out and out[-1] not in (" ", "\t"):
                        out.append(" ")
                    out.append("=")
                    if next_char not in (" ", "\t", "\n", ""):
                        out.append(" ")
                    continue
            out.append(ch)
        new_lines.append("".join(out))
    return "".join(new_lines)


def update_old_unsloth(filename):
    """Update notebook with various fixes using JSON-based cell manipulation."""
    try:
        with open(filename, "r", encoding="utf-8", newline="") as f:
            notebook_content = json.load(f)
    except Exception:
        return

    base = _get_base_name_from_filename(filename)
    if base.endswith("_finetune"):
        base_gguf = base
        base_lora = f"{base}_lora"
        base_16 = f"{base}_16bit"
        base_4 = f"{base}_4bit"
    else:
        base_gguf = f"{base}_finetune"
        base_lora = f"{base}_lora"
        base_16 = f"{base}_finetune_16bit"
        base_4 = f"{base}_finetune_4bit"

    def replace_hf_prefix(name, new_name):
        if "/" in name:
            prefix = name.split("/", 1)[0]
            if prefix == "hf":
                prefix = "HF_USERNAME"
            return f"{prefix}/{new_name}"
        return new_name

    def replace_common(text):
        """Apply common text replacements for both code and markdown cells."""
        if "qwen3_5" not in filename.lower():
            text = _normalize_transformers_v5_pin(text)
        text = text.replace("</a></a>", "</a>")
        text = _RE_DOUBLE_EXCL.sub("!", text)
        text = text.replace("ee notice", "we notice")

        # Convert versions like X.X.X to 2026.2.1
        text = _RE_VERSION.sub(r"2026.2.1\1", text)

        # Change gguf-quantization-options link
        text = text.replace(
            "https://github.com/unslothai/unsloth/wiki#gguf-quantization-options",
            "https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf",
        )
        text = text.replace("https://docs.unsloth.ai/", "https://unsloth.ai/docs/")

        # Redirect Alpaca dataset
        text = text.replace(
            "https://huggingface.co/datasets/yahma/alpaca-cleaned",
            "https://huggingface.co/datasets/unsloth/alpaca-cleaned",
        )
        text = text.replace("yahma/alpaca-cleaned", "unsloth/alpaca-cleaned")
        text = text.replace("Alpaca dataset from [yahma]", "[Alpaca dataset]")

        # Train on completions
        text = text.replace(
            "TRL's docs [here](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).",
            "our docs [here](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#training-on-completions-only-masking-out-inputs)",
        )

        # Fix incorrect conversational link pointing to Alpaca notebook
        text = text.replace(
            "conversational [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Alpaca.ipynb)",
            "conversational [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Conversational.ipynb)",
        )

        # Fix Meta-Llama
        text = text.replace("unsloth/Meta-Llama", "unsloth/Llama")

        # TRL's `DPOTrainer`
        text = text.replace("TRL's `DPOTrainer`", "`DPOTrainer` and `GRPOTrainer` for reinforcement learning!")

        # Move packing = ...
        text = _RE_PACKING.sub(
            r"\3\1    packing = \2, # Makes training 2-5x faster for short sequences,\n",
            text,
        )

        # Ensure GGUF usage line matches base name used in code
        text = _RE_GGUF_USAGE.sub(
            f"Now, use the `{base_gguf}.Q8_0.gguf` file or `{base_gguf}.Q4_K_M.gguf` file in llama.cpp.",
            text,
        )

        # Fix concatenated markdown line if it slipped in
        text = text.replace("Unsloth!Now, use the", "Unsloth!\nNow, use the")

        # Update docs domain
        text = text.replace("docs.unsloth.ai", "unsloth.ai/docs")
        text = text.replace("[Wiki page]", "[docs page]")
        text = text.replace("[wiki page]", "[docs page]")

        text = text.replace(
            "You can go to https://huggingface.co/settings/tokens for your personal tokens.",
            "You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.",
        )

        # GGUF filename references
        text = text.replace("model-unsloth-Q4_K_M.gguf", f"{base_gguf}.Q4_K_M.gguf")
        text = text.replace("model-unsloth.Q4_K_M.gguf", f"{base_gguf}.Q4_K_M.gguf")
        text = text.replace("model-unsloth.Q8_0.gguf", f"{base_gguf}.Q8_0.gguf")
        text = text.replace("model-unsloth.gguf", f"{base_gguf}.Q8_0.gguf")

        # Fix "Huggingface" -> "Hugging Face" (only capitalized, not in URLs/packages)
        text = text.replace("Huggingface's", "Hugging Face's")
        text = _RE_HUGGINGFACE_BACKTICK.sub(r"Hugging Face \1", text)
        text = text.replace("Huggingface TRL's", "Hugging Face TRL's")

        # Fix instruction_part missing < before |end_of_role|>
        text = text.replace(
            '<|start_of_role|>user|end_of_role|>',
            '<|start_of_role|>user<|end_of_role|>',
        )

        # Fix typos in specific phrases
        text = text.replace("Prime and Prejudice", "Pride and Prejudice")
        text = text.replace("2x Telsa T4s", "2x Tesla T4s")
        text = text.replace("float32 s disable", "float32 so disable")
        text = text.replace("and its amazing", "and it's amazing")
        text = text.replace("look like this:", "looks like this:")
        text = text.replace("AutoModelForPeftCausalLM", "AutoPeftModelForCausalLM")

        # Remove @nocommit placeholders
        text = _RE_NOCOMMIT.sub('', text)

        # Fix empty Open Math Reasoning URL
        text = text.replace(
            "[Open Math Reasoning]()",
            "[Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)"
        )

        # Fix footer heading
        text = text.replace("Some other links:", "Some other resources:")

        # Fix old installation URL paths (both variants)
        text = text.replace(
            "unsloth.ai/docs/get-started/installing-+-updating",
            "unsloth.ai/docs/get-started/install"
        )
        text = text.replace(
            "unsloth.ai/docs/get-started/install-and-update",
            "unsloth.ai/docs/get-started/install"
        )

        # Fix footer numbering (6. → 4.)
        text = _RE_FOOTER_NUM.sub(r'\n4. See notebooks for DPO', text)

        # Fix duplicate "See our docs" sentences (same line duplicates)
        text = _RE_DUP_DOCS.sub(r'\1', text)

        # Fix Nemo → NeMo capitalization (but not Mistral-Nemo model names)
        text = _RE_NEMO_GYM.sub('NeMo Gym', text)

        return text

    def replace_code(text):
        """Apply code-specific replacements."""
        # Update gguf save/push names
        text = _RE_SAVE_GGUF.sub(
            rf"\1\2{base_gguf}\4",
            text,
        )

        def _replace_push_gguf(match):
            new_name = replace_hf_prefix(match.group(3), base_gguf)
            return f"{match.group(1)}{match.group(2)}{new_name}{match.group(4)}"

        text = _RE_PUSH_GGUF.sub(
            _replace_push_gguf,
            text,
        )

        # Update merged save/push names
        def _replace_save_merged(match):
            method = match.group(6)
            new_name = base_16 if method == "merged_16bit" else base_4
            return f"{match.group(1)}{match.group(2)}{new_name}{match.group(4)}{match.group(5)}{method}{match.group(7)}"

        text = _RE_SAVE_MERGED.sub(
            _replace_save_merged,
            text,
        )

        def _replace_push_merged(match):
            method = match.group(6)
            new_name = base_16 if method == "merged_16bit" else base_4
            replaced = replace_hf_prefix(match.group(3), new_name)
            return f"{match.group(1)}{match.group(2)}{replaced}{match.group(4)}{match.group(5)}{method}{match.group(7)}"

        text = _RE_PUSH_MERGED.sub(
            _replace_push_merged,
            text,
        )

        # Update LoRA save/push names (skip phone_model names)
        def _replace_save_lora(match):
            if "phone_model" in match.group(3):
                return match.group(0)
            return f"{match.group(1)}{match.group(2)}{base_lora}{match.group(4)}"

        text = _RE_SAVE_LORA.sub(_replace_save_lora, text)

        def _replace_push_lora(match):
            if "phone_model" in match.group(3):
                return match.group(0)
            new_name = replace_hf_prefix(match.group(3), base_lora)
            return f"{match.group(1)}{match.group(2)}{new_name}{match.group(4)}"

        text = _RE_PUSH_LORA.sub(
            _replace_push_lora,
            text,
        )

        # LoRA load snippets
        text = _RE_LORA_LOAD.sub(
            rf"\1\2{base_lora}\4\5",
            text,
        )
        text = _RE_LORA_LOAD2.sub(
            rf"\1{base_lora}\3\4",
            text,
        )
        text = _RE_LORA_MODEL.sub(rf"\1{base_lora}\2", text)
        text = _RE_FINETUNED_MODEL.sub(rf"\1{base_lora}\2", text)

        # Also handle AutoPeftModelForCausalLM.from_pretrained("xxx_lora")
        # and AutoTokenizer.from_pretrained("xxx_lora") for load-back consistency
        text = _RE_AUTO_LORA.sub(
            rf"\1\2{base_lora}\4",
            text,
        )

        # Update hf/ to HF_USERNAME/ in quoted strings
        text = text.replace('"hf/', '"HF_USERNAME/')
        text = text.replace("'hf/", "'HF_USERNAME/")

        # Update tokens - only match string literals to avoid breaking token = get_token()
        text = _RE_TOKEN.sub(
            r'\1"YOUR_HF_TOKEN"',
            text,
        )

        # Preserve special tokens that should not be replaced by HF token
        text = _RE_EOS_TOKEN.sub(
            'unsloth_eos_token = "eos_token"',
            text,
        )
        text = _RE_PATCH_TOKEN.sub(
            'patch_token = "<|IMAGE_PLACEHOLDER|>"',
            text,
        )

        # If dtype=None helper line is directly before from_pretrained and dtype=dtype is used,
        # drop the helper line and inline dtype=None with the standard comment.
        dtype_comment = "None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+"

        lines = text.splitlines(True)
        updated_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if _RE_DTYPE_LINE.match(line) and i + 1 < len(lines) and ".from_pretrained" in lines[i + 1]:
                # Try to update dtype within the from_pretrained call
                replaced = False
                depth = 0
                j = i + 1
                while j < len(lines):
                    current = lines[j]
                    if j == i + 1 and ".from_pretrained" not in current:
                        break
                    new_current, count = _RE_DTYPE_PARAM.subn(
                        r"\1None, # " + dtype_comment,
                        current,
                    )
                    if count:
                        replaced = True
                    lines[j] = new_current
                    depth += current.count("(") - current.count(")")
                    if depth <= 0 and ".from_pretrained" in lines[i + 1]:
                        break
                    j += 1
                if replaced:
                    # Drop the dtype helper line and continue from the call
                    i += 1
                    continue
            updated_lines.append(line)
            i += 1
        text = "".join(updated_lines)

        # Normalize vLLM naming in code where it is used as a package/path
        # Use word boundary to preserve UNSLOTH_VLLM_STANDBY env var
        text = _RE_VLLM.sub('vllm', text)
        text = _RE_VLLM_UPPER.sub('vllm', text)

        # Simplify gated models comment
        text = _RE_GATED_COMMENT.sub(
            "# HF Token for gated models",
            text,
        )

        # Fix A=A to A = A in code
        text = _space_equals_in_code(text)

        return text

    updated = False
    for cell in notebook_content.get("cells", []):
        if not isinstance(cell.get("source"), list):
            continue
        is_code = cell.get("cell_type") == "code"
        text = "".join(cell["source"])
        new_text = replace_common(text)
        if is_code:
            new_text = replace_code(new_text)
        if new_text != text:
            updated = True
            cell["source"] = _strip_extra_trailing_blank_lines(_source_lines(new_text))

    if updated:
        _write_notebook(filename, notebook_content)
pass


_RE_PIP_INSTALL_LINE = re.compile(r"^!(?:uv )?pip install\s+(.+)$", re.MULTILINE)
_RE_PIP_PKG_TOKEN = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9._-]*)")
_RE_TRANSFORMERS_EQ_PIN = re.compile(
    r"(?<![A-Za-z0-9_.-])(transformers\s*==\s*)(\d+(?:\.\d+){0,2})(?![A-Za-z0-9_.-])"
)
_RE_TRANSFORMERS_EQ_PIN_5 = re.compile(
    r"(?<![A-Za-z0-9_.-])transformers\s*==\s*5(?:\.\d+){0,2}(?![A-Za-z0-9_.-])"
)
_INSTALL_FLAG_PREFIXES = ("--", "-", "{", "\"", "'")
_INSTALL_GUARD_IGNORE = frozenset({
    # Standard packages present in the default install cell; safe to swap
    "unsloth", "unsloth_zoo", "bitsandbytes", "accelerate", "xformers",
    "peft", "trl", "triton", "cut_cross_entropy", "sentencepiece",
    "protobuf", "datasets", "huggingface_hub", "hf_transfer",
    "transformers", "pip3_autoremove", "torch", "torchvision",
    "torchaudio", "pip",
})

def _extract_pip_packages(install_text):
    """Extract base package names from pip install lines in install cell text."""
    packages = set()
    for m in _RE_PIP_INSTALL_LINE.finditer(install_text):
        for token in m.group(1).split():
            if any(token.startswith(p) for p in _INSTALL_FLAG_PREFIXES):
                continue
            if "git+" in token or "://" in token:
                # git+https://... -- use the repo name as identifier
                repo = token.rstrip("/").rsplit("/", 1)[-1]
                repo = repo.split(".git")[0].split("@")[0]
                packages.add(repo.lower().replace("-", "_"))
                continue
            pm = _RE_PIP_PKG_TOKEN.match(token)
            if pm:
                packages.add(pm.group(1).lower().replace("-", "_"))
    return packages

def _warn_dropped_packages(notebook_path, old_cell_text, new_cell_text):
    """Warn if the new install cell is missing packages that the old cell had."""
    old_pkgs = _extract_pip_packages(old_cell_text) - _INSTALL_GUARD_IGNORE
    new_pkgs = _extract_pip_packages(new_cell_text) - _INSTALL_GUARD_IGNORE
    dropped = old_pkgs - new_pkgs
    if dropped:
        print(
            f"WARNING: {notebook_path} -- install cell dropped packages: "
            f"{', '.join(sorted(dropped))}. "
            f"Add a dedicated installation_* entry in the script."
        )

def _preserve_transformers_v5_pin(old_cell_text, new_cell_text):
    """Preserve transformers 5.x pin from the old cell, never downgrading the version."""
    old_match = _RE_TRANSFORMERS_V5_PIN_CAPTURE.search(old_cell_text)
    if not old_match:
        return new_cell_text
    old_ver = _parse_version_tuple(old_match.group(1))
    target = _parse_version_tuple(_TARGET_TRANSFORMERS_V5)
    pin_ver = old_match.group(1) if old_ver > target else _TARGET_TRANSFORMERS_V5
    return _RE_TRANSFORMERS_EQ_PIN.sub(rf"\g<1>{pin_ver}", new_cell_text)


badge_section = '<a href="{link_colab}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'


def is_path_contains_any(file_path, words):
    return any(re.search(word, file_path, re.IGNORECASE) for word in words)

def extract_model_info_refined(filename, architecture_mapping, known_types_ordered):
    if not filename.endswith(".ipynb"):
        return {'name': filename, 'size': None, 'type': None, 'architecture': None}
    stem = filename[:-len(".ipynb")]

    requires_a100 = False
    if 'A100' in stem:
        requires_a100 = True
        stem = stem.replace('_A100', '')

    original_stem_parts = stem.replace('+', '_').split('_') 
    type_ = None
    stem_searchable = stem.lower().replace('_', ' ').replace('+', ' ')
    found_type_indices = [] 

    for type_keyword in known_types_ordered:
        kw_lower = type_keyword.lower()
        pattern = r'\b' + re.escape(kw_lower) + r'\b'
        match = re.search(pattern, stem_searchable)
        if match:
            type_ = type_keyword 
            try:
                 
                 kw_parts = type_keyword.split(' ')
                 for i in range(len(original_stem_parts) - len(kw_parts) + 1):
                     match_parts = True
                     for j in range(len(kw_parts)):
                         if original_stem_parts[i+j].lower() != kw_parts[j].lower():
                             match_parts = False
                             break
                     if match_parts:
                         found_type_indices = list(range(i, i + len(kw_parts)))
                         break
            except Exception:
                pass 
            break 
    size = None
    size_match = re.search(r'_\((.*?)\)', stem)
    size_start_index = -1
    if size_match:
        size = size_match.group(1)
        size_start_index = size_match.start() 
    name = None
    if size_start_index != -1:
        name_part = stem[:size_start_index]
        name = name_part.replace('_', ' ').strip()
        if not name:
             post_size_part = stem[size_match.end():]
             if post_size_part.startswith('_'): post_size_part = post_size_part[1:]
             if post_size_part.startswith('+'): post_size_part = post_size_part[1:]
             name = post_size_part.replace('_', ' ').replace('+', ' ').strip()
    else:
        name = stem.replace('_', ' ').strip()
        if type_ and name.lower().endswith(type_.lower()):
            name = name[:-len(type_)].strip()

    if not name:
        name_parts_filtered = [p for i, p in enumerate(original_stem_parts) if i not in found_type_indices]
        name = ' '.join(name_parts_filtered).strip()
        if not name: 
             name = stem.replace('_',' ').strip()

    architecture = None
    if name: 
        name_lower_for_mapping = name.lower()
        sorted_keys = sorted(architecture_mapping.keys(), key=len, reverse=True)
        for key in sorted_keys:
            
            pattern = r'\b' + re.escape(key.lower()) + r'\b'
            if re.search(pattern, name_lower_for_mapping):
                architecture = architecture_mapping[key]
                break
            elif key.lower() in name_lower_for_mapping and architecture is None:
               architecture = architecture_mapping[key]

    for key in TYPE_MAPPING:
        if key.lower() in name.lower():
            type_ = TYPE_MAPPING[key].get(type_, type_)
            break
    for key in TYPE_MAPPING:
        kaggle_key = f"Kaggle {key}"
        if kaggle_key.lower() in name.lower():
            type_ = TYPE_MAPPING.get(kaggle_key, {}).get(type_, type_)
            break

    if "kaggle" in name.lower():
        # Remove "kaggle" from the name
        name = name.replace("Kaggle", "").strip()

    return {'name': name,
            'size': size,
            'type': type_,
            'architecture': architecture,
            'requires_a100': requires_a100}

extracted_info_refined = {}
original_template_path = os.path.abspath("original_template")
list_files = [f for f in os.listdir(original_template_path) if f.endswith(".ipynb")]
standardized_name = [f.replace("-", "_") for f in list_files]

standard_to_original_name = {
    k : v for k, v in zip(standardized_name, list_files)
}
original_to_standard_name = {
    v : k for k, v in zip(standardized_name, list_files)
}
list_files = [f for f in os.listdir(original_template_path) if f.endswith(".ipynb")]
for std_name in standard_to_original_name:
    extracted_info_refined[std_name] = extract_model_info_refined(
        std_name,
        ARCHITECTURE_MAPPING,
        KNOWN_TYPES_ORDERED  
    )

def extract_version_from_row(row):
    """Extracts the version number from a row string for sorting."""
    match = re.search(r"\| (.*?) \|", row)  # Match content between first "|" and " |"
    if match:
        model_name = match.group(1)
        return extract_version(model_name)
    else:
        return (0, 0)

def extract_version(model_name):
    """Extracts the architecture version number for sorting.

    Handles cases like:
        - Phi 3 Medium
        - Phi 3.5 Mini
        - Phi 4
        - Gemma4 (E4B)        -> 4 (size E4B is ignored)
        - (A100) Gemma3 (27B) -> 3 (A100 prefix and 27B size are ignored)
        - Llama3.1 (8B)       -> (3, 1)
    Returns a tuple of (major version, minor version) for proper sorting.
    Returns (0, 0) if no version is found in the architecture name.

    The size suffix (e.g. "**(7B)**", "**(E4B)**") and the "(A100)" prefix
    are stripped before searching for the version digit, otherwise the
    parameter count or "100" from "A100" would be picked up as the version.
    """
    name = model_name
    # Strip a trailing parenthesised size suffix wrapped in markdown bold,
    # e.g. "**Gemma4** **(E4B)**" -> "**Gemma4** "
    name = re.sub(r"\*\*\([^)]*\)\*\*\s*$", "", name).strip()
    # Strip a trailing parenthesised size suffix without bold,
    # e.g. "Gemma4 (E4B)" -> "Gemma4"
    name = re.sub(r"\([^)]*\)\s*$", "", name).strip()
    # Strip an "(A100)" or "(A100) " prefix anywhere
    name = re.sub(r"\(A100\)\s*", "", name).strip()
    # Strip markdown bold markers
    name = name.replace("**", "").strip()
    match = re.search(r"(\d+(\.\d+)?)", name)
    if match:
        version_str = match.group(1)
        if "." in version_str:
            major, minor = version_str.split(".")
            return (int(major), int(minor))
        else:
            return (int(version_str), 0)
    else:
        return (0, 0)


# ============================================================================
# Model reference extraction + created_at cache (for README sorting)
# ============================================================================
#
# The README row order within each section is computed from the HF Hub
# popularity (downloads + likes*1000) of the models each notebook actually
# loads. To avoid hitting the Hub on every run we maintain a CSV at
#   scripts/model_created_at.csv
# which maps <org>/<repo> to its created_at, downloads, likes, fetched_at
# and status. Entries that cannot be resolved (datasets, 404s, placeholders
# that escaped the blocklist) are cached with status=not_found so we do not
# re-query them. ok rows older than _MODEL_CACHE_OK_TTL_DAYS days get
# refreshed automatically since downloads/likes drift over time.

# <org>/<repo> where both pieces look like HF repo IDs. Anchored by one of:
# start-of-string, whitespace, quote, open-paren, open-bracket, open-brace,
# comma, colon, equals. Ends at a similar boundary.
_HF_MODEL_REF_RE = re.compile(
    r"""(?P<before>^|['"\s\(\[\{,:=])
        (?P<org>[A-Za-z][A-Za-z0-9._-]{0,95})
        /
        (?P<repo>[A-Za-z0-9][A-Za-z0-9._-]{0,95})
        (?=['"\s\)\]\},:]|$)
    """,
    re.VERBOSE,
)

# Matches the primary model assignment: model_name = "org/repo" (or single-quoted).
# This is the model the notebook actually loads. Many notebooks also list
# alternative models in a `gemma4_models = [...]` style block; those show up
# in the generic ref set but are not what the notebook trains on, so we
# prefer assignments for sorting.
_HF_MODEL_NAME_ASSIGN_RE = re.compile(
    r"""model_name\s*=\s*(?P<quote>['"])
        (?P<org>[A-Za-z][A-Za-z0-9._-]{0,95})
        /
        (?P<repo>[A-Za-z0-9][A-Za-z0-9._-]{0,95})
        (?P=quote)
    """,
    re.VERBOSE,
)

# Matches a positional repo passed as the first arg of .from_pretrained(...).
# Example: FastVisionModel.from_pretrained("unsloth/gemma-4-E4B-it", ...)
# This is used when the notebook author omits the model_name= keyword.
_HF_FROM_PRETRAINED_POSITIONAL_RE = re.compile(
    r"""\.from_pretrained\s*\(\s*(?P<quote>['"])
        (?P<org>[A-Za-z][A-Za-z0-9._-]{0,95})
        /
        (?P<repo>[A-Za-z0-9][A-Za-z0-9._-]{0,95})
        (?P=quote)
    """,
    re.VERBOSE,
)

# Strings that look like <org>/<repo> in the notebook source but are never
# real Hugging Face repos. These get filtered at extraction time so we do
# not pollute the cache with thousands of useless rows.
_HF_MODEL_REF_PLACEHOLDER_ORGS = {
    # Placeholders users are supposed to edit before running
    "HF_USERNAME", "HF_ACCOUNT", "YOUR_USERNAME", "your_name",
    "hf", "HuggingFaceOrganization", "HuggingFaceUser",
    # Python variable names that happen to collide
    "repo_id", "prompt",
    # Local save paths used in post-training snippets
    "grpo_lora", "grpo_saved_lora",
    # Comment fragments
    "TrackIO",   # "# Use TrackIO/WandB etc"
    "data",      # file path fragments
    "python",    # e.g. "python/triton_kernels"
    "Colab",
    "pros",
}

# Cache file relative to the repo root.
_MODEL_CREATED_CACHE_PATH = os.path.join("scripts", "model_created_at.csv")


# Matches `fast_inference = True` (any whitespace around =, any case). This
# is Unsloth's flag for enabling vLLM during GRPO training, so GRPO notebooks
# that set it get rendered with a "GRPO + vLLM" type in the README.
_FAST_INFERENCE_TRUE_RE = re.compile(
    r"\bfast_inference\s*=\s*True\b"
)


# TRL trainer class names that identify an RL / preference-optimization
# training run. Ordered from most to least "RL-like" so detect_trainer_class
# can pick the most indicative one when a notebook imports several.
_TRAINER_CLASS_RE = re.compile(
    r"\b(GRPOTrainer|DPOTrainer|ORPOTrainer|KTOTrainer|RewardTrainer|PPOTrainer)\b"
)
_TRAINER_CLASS_PRIORITY = [
    "GRPOTrainer",
    "DPOTrainer",
    "ORPOTrainer",
    "KTOTrainer",
    "RewardTrainer",
    "PPOTrainer",
]


def detect_trainer_class(notebook_path):
    """Return the TRL trainer class the notebook trains with, or None.

    Scans every code cell for one of the known trainer class names. If
    multiple classes appear (e.g. an SFT warm-up before GRPO), returns
    the highest-priority one from _TRAINER_CLASS_PRIORITY so the final
    RL phase wins. Returns None on I/O errors or when no class matches.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    found = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src_list = cell.get("source", [])
        src = src_list if isinstance(src_list, str) else "".join(src_list)
        for m in _TRAINER_CLASS_RE.finditer(src):
            found.add(m.group(1))
    for cls in _TRAINER_CLASS_PRIORITY:
        if cls in found:
            return cls
    return None


def notebook_uses_fast_inference(notebook_path):
    """Return True if any code cell contains `fast_inference = True`.

    Returns False on I/O or parse errors.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src_list = cell.get("source", [])
        src = src_list if isinstance(src_list, str) else "".join(src_list)
        if _FAST_INFERENCE_TRUE_RE.search(src):
            return True
    return False


def detect_rl_task(notebook_path):
    """Inspect an RL/GRPO notebook and classify the task it trains on.

    Returns a short human-readable task label, or None if no task could
    be inferred. Detection uses two passes:

    1. Datasets referenced by `load_dataset("...")` calls in code cells.
       This catches the generic math GRPO notebooks (GSM8K, DAPO,
       MathVista) that share the same boilerplate markdown structure.
    2. Markdown headers and filename keywords for environment-specific
       notebooks (2048, Wordle, Sudoku, Multi Environment, kernels).

    Dataset detection runs first so a Vision GRPO notebook that happens
    to mention "sudoku" in a comment still gets the "Vision Math" label.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    datasets = set()
    markdown_text = []
    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        src_list = cell.get("source", [])
        src = src_list if isinstance(src_list, str) else "".join(src_list)
        if ctype == "code":
            for m in _LOAD_DATASET_RE.finditer(src):
                datasets.add(m.group("repo").strip())
        elif ctype == "markdown":
            markdown_text.append(src.lower())

    # --- Dataset-based classification (most reliable) --------------------
    # Any MathVista reference marks this as a Vision Math RL notebook.
    if any("mathvista" in d.lower() for d in datasets):
        return "Vision Math"
    # DAPO math / OpenMathReasoning (Unsloth's newer GRPO reasoning recipe)
    if any("dapo" in d.lower() or "openmathreasoning" in d.lower() for d in datasets):
        return "DAPO Math"
    # Classic GSM8K math-word-problem GRPO notebooks
    if any("gsm8k" in d.lower() for d in datasets):
        return "GSM8K Math"

    # --- Markdown / filename based classification ------------------------
    md_joined = "\n".join(markdown_text)
    basename_lower = os.path.basename(notebook_path).lower()

    if "wordle" in md_joined or "wordle" in basename_lower:
        return "Wordle"
    if (
        "multi-environment" in md_joined
        or "multi environment" in md_joined
        or "multi-environment" in basename_lower
    ):
        return "Multi Environment"
    if (
        "faster kernels" in md_joined
        or "optimized matrix multiplication" in md_joined
    ):
        return "Auto Kernel Creation"
    if (
        "2048 game" in md_joined
        or "play 2048" in md_joined
        or "2048" in basename_lower
    ):
        return "2048 Game"
    if "minesweeper" in md_joined or "minesweeper" in basename_lower:
        return "Minesweeper Game"
    if "sudoku" in md_joined or "sudoku" in basename_lower:
        return "Sudoku"
    return None


# Matches `load_dataset("...")` and `load_dataset('...')` with a single
# positional repo id. Used by detect_rl_task to pull the training dataset
# out of each notebook's code cells.
_LOAD_DATASET_RE = re.compile(
    r"""load_dataset\s*\(\s*
        (?P<quote>['"])
        (?P<repo>[^'"]+)
        (?P=quote)
    """,
    re.VERBOSE,
)


def extract_hf_model_refs_from_notebook(notebook_path):
    """Scan a notebook's code cells for <org>/<repo> Hugging Face model refs.

    Returns (all_refs, assigned_refs):
        all_refs      : set of every "org/repo" string found in the code cells.
        assigned_refs : ordered list (deduped, insertion order) of the values
                        assigned to `model_name = "..."`. These are the models
                        the notebook actually loads and should take precedence
                        in the sort key.

    Placeholders (HF_USERNAME, etc.), URL path fragments, and anything that
    looks like a local file path are filtered out. Returns (set(), []) on
    I/O or parse errors.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set(), []

    refs = set()
    assigned = []  # preserve order, first assignment wins ties
    seen_assigned = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src_list = cell.get("source", [])
        if isinstance(src_list, str):
            src = src_list
        else:
            src = "".join(src_list)

        # Pull out the primary model loads first: model_name="..." assignments
        # and positional first-arg .from_pretrained("...") calls. These are
        # what the notebook actually loads. We still add them to the generic
        # refs set below.
        for primary_re in (_HF_MODEL_NAME_ASSIGN_RE, _HF_FROM_PRETRAINED_POSITIONAL_RE):
            for m in primary_re.finditer(src):
                org = m.group("org")
                repo = m.group("repo")
                if "." in org or org in _HF_MODEL_REF_PLACEHOLDER_ORGS:
                    continue
                ref = f"{org}/{repo}"
                if ref not in seen_assigned:
                    assigned.append(ref)
                    seen_assigned.add(ref)

        for m in _HF_MODEL_REF_RE.finditer(src):
            org = m.group("org")
            repo = m.group("repo")
            # Skip URLs (match preceded by "://")
            start = m.start("org")
            if start >= 3 and src[start - 3:start] == "://":
                continue
            # Orgs containing a dot are always domain-ish, never HF orgs
            if "." in org:
                continue
            if org in _HF_MODEL_REF_PLACEHOLDER_ORGS:
                continue
            refs.add(f"{org}/{repo}")
    return refs, assigned


def _load_model_created_cache(cache_path=_MODEL_CREATED_CACHE_PATH):
    """Load the model popularity CSV into a dict keyed by model_repo.

    Each value is a dict with keys:
        created_at : str (ISO 8601 UTC) or ""
        downloads  : int (0 if missing/unknown)
        likes      : int (0 if missing/unknown)
        base_model : str (single upstream repo) or ""
        fetched_at : str (ISO 8601 UTC)
        status     : "ok" | "not_found" | "error"

    Older CSV files without one or more of these columns are still supported:
    missing numeric columns default to 0 and missing string columns to "".

    Returns an empty dict if the file is missing or unreadable.
    """
    cache = {}
    if not os.path.exists(cache_path):
        return cache
    try:
        with open(cache_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                repo = (row.get("model_repo") or "").strip()
                if not repo:
                    continue
                def _to_int(v):
                    try:
                        return int((v or "").strip() or 0)
                    except (TypeError, ValueError):
                        return 0
                cache[repo] = {
                    "created_at": (row.get("created_at") or "").strip(),
                    "downloads": _to_int(row.get("downloads")),
                    "likes": _to_int(row.get("likes")),
                    "base_model": (row.get("base_model") or "").strip(),
                    "fetched_at": (row.get("fetched_at") or "").strip(),
                    "status": (row.get("status") or "").strip() or "ok",
                }
    except Exception as e:
        print(f"  [WARN] Could not parse {cache_path}: {e}")
    return cache


def _write_model_created_cache(cache, cache_path=_MODEL_CREATED_CACHE_PATH):
    """Write the cache dict out as CSV, sorted alphabetically for stable diffs."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_repo", "created_at", "downloads", "likes",
            "base_model", "fetched_at", "status",
        ])
        for repo in sorted(cache.keys()):
            entry = cache[repo]
            writer.writerow([
                repo,
                entry.get("created_at", ""),
                entry.get("downloads", 0),
                entry.get("likes", 0),
                entry.get("base_model", ""),
                entry.get("fetched_at", ""),
                entry.get("status", ""),
            ])


def _extract_base_model(info):
    """Pull a single upstream repo from model_info card_data.base_model.

    The HF card field can be:
      * a string ("Qwen/Qwen2.5-VL-7B-Instruct")
      * a list of strings (["Qwen/Qwen2.5-VL-7B-Instruct"])
      * missing entirely
    We return the first valid <org>/<repo> we find, or "" if none.
    """
    cd = getattr(info, "card_data", None)
    if cd is None:
        return ""
    base = getattr(cd, "base_model", None)
    if base is None and isinstance(cd, dict):
        base = cd.get("base_model")
    if not base:
        return ""
    if isinstance(base, str):
        candidates = [base]
    elif isinstance(base, (list, tuple)):
        candidates = [b for b in base if isinstance(b, str)]
    else:
        return ""
    for c in candidates:
        c = c.strip()
        if "/" in c and c.count("/") == 1:
            org, repo = c.split("/", 1)
            if org and repo and "." not in org:
                return c
    return ""


def _fetch_model_info(repo):
    """Query HF Hub for model popularity.

    Returns (created_at, downloads, likes, base_model, status). status in
    {"ok", "not_found", "error"}. On non-ok, numeric fields are 0 and
    string fields are "".

    Uses HF_TOKEN from the environment when present so that gated/private
    repos resolve instead of bouncing as 404s.
    """
    if HfApi is None:
        return ("", 0, 0, "", "error")
    try:
        token = os.environ.get("HF_TOKEN") or None
        api = HfApi(token=token)
        info = api.model_info(repo, timeout=15, token=token)
    except RepositoryNotFoundError:
        return ("", 0, 0, "", "not_found")
    except Exception:
        return ("", 0, 0, "", "error")

    created_at = getattr(info, "created_at", None)
    if created_at is None:
        created_at_str = ""
    elif hasattr(created_at, "astimezone"):
        try:
            created_at_str = created_at.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except Exception:
            created_at_str = str(created_at)
    else:
        created_at_str = str(created_at)

    def _safe_int(v):
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    downloads = _safe_int(getattr(info, "downloads", 0))
    likes = _safe_int(getattr(info, "likes", 0))
    base_model = _extract_base_model(info)
    return (created_at_str, downloads, likes, base_model, "ok")


# How long an "ok" row is allowed to stay in the cache before its
# downloads/likes are refreshed against the Hub. created_at is immutable so
# this only matters for the popularity counters.
_MODEL_CACHE_OK_TTL_DAYS = 14


def _ok_row_is_stale(entry, ttl_days=_MODEL_CACHE_OK_TTL_DAYS):
    """Return True if an ok-status cache row is older than ttl_days.

    Rows that are missing fetched_at or have an unparseable timestamp are
    considered stale so they get refreshed on the next run.
    """
    fetched_at = _parse_iso8601_utc(entry.get("fetched_at"))
    if fetched_at is None:
        return True
    age = datetime.now(timezone.utc) - fetched_at
    return age.total_seconds() > ttl_days * 86400


def refresh_model_created_cache(notebook_paths, cache_path=_MODEL_CREATED_CACHE_PATH):
    """Scan notebooks, populate/refresh the cache, and return (cache, refs_by_nb, assigned_by_nb).

    refs_by_nb     : dict[notebook_path, set[model_repo]]
    assigned_by_nb : dict[notebook_path, list[model_repo]]  # model_name="..." hits
    cache          : dict[model_repo, {"created_at", "downloads", "likes", "fetched_at", "status"}]

    Refresh policy:
      * Repos not in the cache: always fetched.
      * Status "error": always re-fetched (transient failures retry).
      * Status "ok" and fetched_at older than _MODEL_CACHE_OK_TTL_DAYS days:
        re-fetched so downloads/likes stay reasonably current.
      * Status "ok" within the TTL window: skipped (cheap no-op runs).
      * Status "not_found": sticky, never re-queried.
    """
    cache = _load_model_created_cache(cache_path)

    refs_by_nb = {}
    assigned_by_nb = {}
    all_refs = set()
    for path in notebook_paths:
        refs, assigned = extract_hf_model_refs_from_notebook(path)
        refs_by_nb[path] = refs
        assigned_by_nb[path] = assigned
        all_refs.update(refs)

    # Decide which repos still need a fetch
    to_fetch = []
    for repo in sorted(all_refs):
        entry = cache.get(repo)
        if entry is None:
            to_fetch.append(repo)
            continue
        status = entry.get("status")
        if status == "error":
            to_fetch.append(repo)
        elif status == "ok" and _ok_row_is_stale(entry):
            to_fetch.append(repo)
        # status == "not_found": skip
        # status == "ok" and fresh: skip

    def _do_fetch_pass(repos, label):
        if not repos:
            return
        print(f"  Fetching popularity for {len(repos)} {label} repo(s) from HF Hub...")
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ok = not_found = errors = 0
        for i, repo in enumerate(repos, 1):
            created_at, downloads, likes, base_model, status = _fetch_model_info(repo)
            # Preserve the existing created_at/base_model if the new fetch
            # lost them but we had them before (defensive: card_data can
            # disappear or fail to parse on individual fetches).
            prev = cache.get(repo) or {}
            if not created_at and prev.get("created_at"):
                created_at = prev["created_at"]
            if not base_model and prev.get("base_model"):
                base_model = prev["base_model"]
            cache[repo] = {
                "created_at": created_at,
                "downloads": downloads,
                "likes": likes,
                "base_model": base_model,
                "fetched_at": now_iso,
                "status": status,
            }
            if status == "ok":
                ok += 1
            elif status == "not_found":
                not_found += 1
            else:
                errors += 1
            if i % 25 == 0 or i == len(repos):
                print(
                    f"    {i}/{len(repos)} "
                    f"(ok={ok} not_found={not_found} errors={errors})"
                )

    if to_fetch:
        _do_fetch_pass(to_fetch, "notebook-referenced")
    else:
        print("  Notebook-referenced popularity cache is up to date.")

    # Second pass: any base_model upstream we discovered that isn't already
    # in the cache. This pulls in upstream repos like Qwen/Qwen2.5-VL-7B-Instruct
    # that the notebooks themselves never reference but that drive the real
    # popularity of the model family.
    base_to_fetch = []
    seen_base = set()
    for repo, entry in cache.items():
        if entry.get("status") != "ok":
            continue
        bm = (entry.get("base_model") or "").strip()
        if not bm or bm in seen_base or bm in cache:
            continue
        seen_base.add(bm)
        base_to_fetch.append(bm)
    base_to_fetch.sort()
    if base_to_fetch:
        _do_fetch_pass(base_to_fetch, "upstream base_model")

    if to_fetch or base_to_fetch:
        _write_model_created_cache(cache, cache_path)

    return cache, refs_by_nb, assigned_by_nb


# Each like is worth this many downloads in the popularity score. Likes are
# rare relative to downloads (a popular model has thousands of downloads but
# usually <100 likes), so the multiplier is what makes them actually move
# the ordering.
_LIKE_WEIGHT = 1000

# Freshness boost to counteract the cold-start problem: a brand-new model
# has 0 downloads but should still appear high in the README for its first
# few weeks so readers can find it. The boost is added on top of the raw
# downloads+likes score, linearly decaying from _NEW_MODEL_BOOST_MAGNITUDE
# at day 0 to 0 at day _NEW_MODEL_BOOST_WINDOW_DAYS. The magnitude is sized
# to comfortably exceed the most popular Vision/Llama upstream scores
# (~5-6M) so a day-0 release can land at the top of cross-cutting
# sections like Vision (Multimodal).
_NEW_MODEL_BOOST_WINDOW_DAYS = 30
_NEW_MODEL_BOOST_MAGNITUDE = 15_000_000


def _parse_iso8601_utc(s):
    """Parse a stored ISO8601 string back into an aware UTC datetime.

    Returns None on anything unparseable. Accepts both "...Z" and
    "+00:00" suffixes (matching the two formats the cache can hold).
    """
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _freshness_boost(entry):
    """Decaying score boost for a recently-created model.

    Returns 0 outside the boost window or when created_at is missing. The
    boost scales linearly from _NEW_MODEL_BOOST_MAGNITUDE at age 0 to 0
    at age _NEW_MODEL_BOOST_WINDOW_DAYS days.
    """
    if not entry:
        return 0
    dt = _parse_iso8601_utc(entry.get("created_at"))
    if dt is None:
        return 0
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    if age_days < 0 or age_days >= _NEW_MODEL_BOOST_WINDOW_DAYS:
        return 0
    decay = 1.0 - (age_days / _NEW_MODEL_BOOST_WINDOW_DAYS)
    return int(_NEW_MODEL_BOOST_MAGNITUDE * decay)


def _entry_self_score(entry):
    """Popularity score for one cache entry: downloads + likes*1000 + freshness boost.

    Returns 0 for missing/non-ok entries. Does NOT follow base_model.
    """
    if not entry or entry.get("status") != "ok":
        return 0
    downloads = entry.get("downloads", 0) or 0
    likes = entry.get("likes", 0) or 0
    return downloads + likes * _LIKE_WEIGHT + _freshness_boost(entry)


def _popularity_score(entry, cache=None, _seen=None):
    """Popularity score for a cache entry, following base_model upstream.

    If the entry has a base_model that resolves to an ok cache row, the
    returned score is the MAX of (this entry, the upstream entry's score).
    This way notebooks that load `unsloth/X-bnb-4bit` inherit the popularity
    of the canonical upstream `Qwen/X` (or whatever the base_model is).

    Cycles are guarded by _seen.
    """
    if not entry or entry.get("status") != "ok":
        return 0
    score = _entry_self_score(entry)
    if cache is None:
        return score
    bm = (entry.get("base_model") or "").strip()
    if not bm:
        return score
    if _seen is None:
        _seen = set()
    if bm in _seen:
        return score
    _seen.add(bm)
    upstream = cache.get(bm)
    if upstream is None:
        return score
    return max(score, _popularity_score(upstream, cache, _seen))


def notebook_created_at_key(notebook_path, refs_by_nb, cache, assigned_by_nb=None):
    """Return the sort key for one notebook: (popularity_score, count_ok).

    The name is kept for back-compat but the key is driven by HF Hub
    popularity (downloads + likes*1000), with base_model upstream lookup.

    Preference order for *which* model the score comes from:
      1. If the notebook has one or more `model_name = "org/repo"` (or
         positional `from_pretrained("...")`) loads that resolved to ok,
         take the MAX score across THOSE (each follows its base_model).
      2. Otherwise, fall back to the MAX score across every HF ref
         discovered in the notebook.

    count_ok is the number of resolved repos in the chosen pool, used as a
    minor tiebreaker. Notebooks with no resolvable refs return (0, 0).
    """
    def _scores(repos):
        out = []
        for repo in repos:
            entry = cache.get(repo)
            if entry and entry.get("status") == "ok":
                out.append(_popularity_score(entry, cache))
        return out

    if assigned_by_nb:
        assigned = assigned_by_nb.get(notebook_path, [])
        assigned_scores = _scores(assigned)
        if assigned_scores:
            return (max(assigned_scores), len(assigned_scores))

    refs = refs_by_nb.get(notebook_path, set())
    ref_scores = _scores(refs)
    if not ref_scores:
        return (0, 0)
    return (max(ref_scores), len(ref_scores))


def _update_news_only(notebook_path, new_announcement):
    """Update ONLY the '### News' section in a notebook, leaving everything else untouched."""
    try:
        with open(notebook_path, "r", encoding="utf-8", newline="") as f:
            notebook_content = json.load(f)
    except Exception:
        return False

    _cache_notebook_format(notebook_path)
    updated = False

    for i, cell in enumerate(notebook_content["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        source_str = "".join(cell["source"]).strip()
        if source_str == "### News":
            if (
                i + 1 < len(notebook_content["cells"])
                and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
            ):
                announcement = new_announcement.strip()
                notebook_content["cells"][i + 1]["source"] = _source_lines(announcement)
                updated = True
            break

    if updated:
        _write_notebook(notebook_path, notebook_content)
    return updated


def update_notebook_sections(
    notebook_path,
    general_announcement,
    installation_steps,
    installation_steps_kaggle,
    new_announcement,
):
    try:
        with open(notebook_path, "r", encoding="utf-8", newline="") as f:
            notebook_content = json.load(f)

        updated = False

        first_markdown_index = -1
        news_markdown_index = -1

        for i, cell in enumerate(notebook_content["cells"]):
            if cell["cell_type"] == "markdown":
                if first_markdown_index == -1:
                    first_markdown_index = i

                source_str = "".join(cell["source"]).strip()

                if "###" in source_str:
                    news_markdown_index = i
                    break

        # Select announcement based on notebook type and GPU
        gpu_type = notebook_content.get("metadata", {}).get("colab", {}).get("gpuType", "T4")
        if f"{hf_course_name}-" in notebook_path:
            full_model_name = os.path.basename(notebook_path).replace(".ipynb", "")
            full_model_name = full_model_name.split("-")
            full_model_name = " ".join(full_model_name[1:]).replace("_", " ")
            general_announcement = general_announcement_content_hf_course.format(full_model_name=full_model_name)
        elif "Meta" in notebook_path:
            general_announcement = general_announcement_content_meta
        elif gpu_type == "A100":
            general_announcement = general_announcement_content_a100
        elif gpu_type == "L4":
            general_announcement = general_announcement_content_l4
        elif "A100" in notebook_path:
            general_announcement = general_announcement_content_a100

        # Fix GPU text in category-specific templates (HF Course, Meta) that default to T4
        if gpu_type == "A100":
            general_announcement = general_announcement.replace(
                "on a **free** Tesla T4 Google Colab instance!", "on your A100 Google Colab Pro instance!")
        elif gpu_type == "L4":
            general_announcement = general_announcement.replace(
                "on a **free** Tesla T4 Google Colab instance!", "on your L4 Google Colab Pro instance!")

        # Update the general announcement section
        if first_markdown_index != -1:
            if news_markdown_index == first_markdown_index:
                # "# News" is the first markdown, insert above it
                if first_markdown_index >= 0:
                    notebook_content["cells"].insert(
                        first_markdown_index,
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": _source_lines(general_announcement),
                        },
                    )
                    updated = True
                    news_markdown_index += 1  # Adjust index since a new cell is added
                else:
                    notebook_content["cells"][first_markdown_index]["source"] = _source_lines(general_announcement)
                    updated = True
            elif not "".join(
                notebook_content["cells"][first_markdown_index]["source"]
            ).strip():
                # First markdown is empty, replace it
                notebook_content["cells"][first_markdown_index]["source"] = _source_lines(general_announcement)
                updated = True

        i = 0 if news_markdown_index == -1 else news_markdown_index

        is_gguf = False
        is_ollama = False
        is_gemma3 = is_path_contains_any(notebook_path.lower(), ["gemma3"])
        is_llama = is_path_contains_any(notebook_path.lower(), ["llama"])
        is_vision = is_path_contains_any(notebook_path.lower(), ["vision"])
        is_qwen3 = is_path_contains_any(notebook_path.lower(), ["qwen3"])

        while i < len(notebook_content["cells"]):
            cell = notebook_content["cells"][i]

            if cell["cell_type"] == "markdown":
                source_str = "".join(cell["source"]).strip()

                if "### Ollama Support" in source_str:
                    is_ollama = True
                elif "gguf" in source_str and not is_gemma3:
                    is_gguf = True

                if source_str == "### News":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
                    ):
                        announcement = new_announcement.strip()
                        notebook_content["cells"][i + 1]["source"] = _source_lines(announcement)
                        updated = True
                        i += 1
                elif source_str == "### Installation":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "code"
                    ):
                        if is_path_contains_any(notebook_path, ["kaggle"]):
                            installation = installation_steps_kaggle
                        else:
                            installation = installation_steps

                        # GRPO INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["grpo"]) and not is_path_contains_any(notebook_path.lower(), ["gpt_oss", "gpt-oss"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_grpo_kaggle_content
                                # Kaggle will delete the second cell instead -> Need to check
                                del notebook_content["cells"][i + 2]
                            else:
                                installation = installation_grpo_content
                                # TODO: Remove after GRPO numpy bug fixed!
                                # Error : ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                                notebook_content["cells"][i + 2]["source"] = installation_extra_grpo_content

                        # META INSTALLATION
                        elif is_path_contains_any(notebook_path.lower(), ["Meta"]): 
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_grpo_synthetic_data_content
                                # Kaggle will delete the second cell instead -> Need to check
                                del notebook_content["cells"][i + 2]
                            else:
                                installation = installation_synthetic_data_content
                                # TODO: Remove after GRPO numpy bug fixed!
                                # Error : ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                                notebook_content["cells"][i + 2]["source"] = installation_extra_grpo_content
                        
                        # ORPHEUS INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["orpheus"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_orpheus_kaggle_content
                            else:
                                installation = installation_orpheus_content

                        # WHISPER INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["whisper"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_whisper_kaggle_content
                            else:
                                installation = installation_whisper_content

                        # SPARK INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["spark"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_spark_kaggle_content
                            else:
                                installation = installation_spark_content

                        # OUTE INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["oute"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_oute_kaggle_content
                            else:
                                installation = installation_oute_content

                        # LLASA INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["llasa"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_llasa_kaggle_content
                            else:
                                installation = installation_llasa_content

                        # TOOL CALLING INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["tool_calling"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_tool_calling_kaggle_content
                            else:
                                installation = installation_tool_calling_content

                        # SESAME CSM INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["sesame_csm"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_sesame_csm_kaggle_content
                            else:
                                installation = installation_sesame_csm_content

                        # SGLANG INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["sglang"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_sglang_kaggle_content
                            else:
                                installation = installation_sglang_content

                        # QAT INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["qat"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_qat_kaggle_content
                            else:
                                installation = installation_qat_content
                                
                        # GPT OSS INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["gpt_oss", "gpt-oss"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_gpt_oss_kaggle_content
                            else:
                                installation = installation_gpt_oss_content

                        # Llama Vision INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["llama"]) and is_vision:
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_llama_vision_kaggle_content
                            else:
                                installation = installation_llama_vision_content

                        # Gemma3N INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["gemma3n"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_gemma3n_kaggle_content
                            else:
                                installation = installation_gemma3n_content

                        # Gemma4 INSTALLATION: preserve the custom
                        # transformers==5.5.0 --no-deps + torchcodec block.
                        if is_path_contains_any(notebook_path.lower(), ["gemma4"]):
                            installation = installation_gemma4_content

                        # ERNIE VL INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["ernie_4_5_vl"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_ernie_4_5_vl_kaggle_content
                            else:
                                installation = installation_ernie_4_5_vl_content
                                
                        # Deepseek OCR INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["deepseek_ocr"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_deepseek_ocr_kaggle_content
                            else:
                                installation = installation_deepseek_ocr_content
                                
                        # Qwen3VL INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["qwen3"]) and is_vision:
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_qwen3_vl_kaggle_content
                            else:
                                installation = installation_qwen3_vl_content
                                
                        # Qwen3.5 INSTALLATION (must come after Qwen3VL to override for qwen3_5).
                        # Match both "qwen3_5" and "qwen_3_5" so that the inconsistently-named
                        # Qwen_3_5_27B_A100(80GB).ipynb also hits this branch instead of falling
                        # through to the default installation_content (which would clobber the
                        # custom torch 2.8.0 / flash-linear-attention / causal_conv1d block).
                        if is_path_contains_any(notebook_path.lower(), ["qwen3_5", "qwen_3_5"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_qwen3_5_kaggle_content
                            else:
                                installation = installation_qwen3_5_content

                        # Nemotron Nano 3 INSTALLATION also Granite has mamba
                        if is_path_contains_any(notebook_path.lower(), ["nemotron-3-nano","nemotron-nano-3", "granite4"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_nemotron_nano_kaggle_content
                            else:
                                installation = installation_nemotron_nano_content

                        # MINISTRAL INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["ministral"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_ministral_kaggle_content
                            else:
                                installation = installation_ministral_content

                        # GLM FLASH INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["glm_flash", "glm-flash"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_glm_flash_kaggle_content
                            else:
                                installation = installation_glm_flash_content

                        # PHONE DEPLOYMENT INSTALLATION (ExecuTorch)
                        if is_path_contains_any(notebook_path.lower(), ["phone_deployment", "phone-deployment"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_phone_kaggle_content
                            else:
                                installation = installation_phone_content

                        # Guard: warn if the replacement drops packages
                        old_install_src = notebook_content["cells"][i + 1].get("source", "")
                        if isinstance(old_install_src, list):
                            old_install_src = "".join(old_install_src)
                        if isinstance(installation, list):
                            new_install_text = "".join(installation)
                        else:
                            new_install_text = installation
                        if not is_path_contains_any(notebook_path.lower(), ["qwen3_5"]):
                            new_install_text = _preserve_transformers_v5_pin(old_install_src, new_install_text)
                        _warn_dropped_packages(notebook_path, old_install_src, new_install_text)

                        notebook_content["cells"][i + 1]["source"] = new_install_text
                        updated = True
                        # TODO: Remove after GRPO numpy bug fixed! 
                        # Error: ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                        if is_path_contains_any(notebook_path.lower(), ["grpo"]) and not is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                            i += 2
                        else:
                            i += 1

            i += 1

        # Add text to the last cell
        if notebook_content["cells"]:
            last_cell = notebook_content["cells"][-1]
            if is_ollama:
                text_for_last_cell = text_for_last_cell_ollama
            elif is_gguf:
                text_for_last_cell = text_for_last_cell_gguf
            else:
                text_for_last_cell = text_for_last_cell_non_gguf

            if last_cell["cell_type"] == "markdown":
                # Check if the last cell already contains footer content using key markers
                existing_text = "".join(last_cell["source"])
                # Key markers that indicate footer content already exists
                footer_markers = [
                    "And we're done! If you have any questions on Unsloth",
                    "Train your own reasoning model - Llama GRPO notebook",
                    "This notebook and all Unsloth notebooks are licensed"
                ]
                # Specific check for LGPL license line
                lgpl_marker = "This notebook and all Unsloth notebooks are licensed [LGPL-3.0]"

                # Check if notebook has partial footer content but missing LGPL line
                has_partial_footer = any(marker in existing_text for marker in footer_markers[:2])  # First two markers only
                has_lgpl = lgpl_marker in existing_text

                # Add content if:
                # 1. No footer markers at all, OR
                # 2. Has partial footer but missing LGPL license line
                if not any(marker in existing_text for marker in footer_markers) or (has_partial_footer and not has_lgpl):
                    # If there's partial footer but missing LGPL, only add the LGPL line
                    if has_partial_footer and not has_lgpl:
                        # Add just the LGPL license line
                        if last_cell["source"] and not last_cell["source"][-1].endswith("\n"):
                            last_cell["source"][-1] += "\n"
                        last_cell["source"].extend(_source_lines("\n  This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)."))
                    else:
                        # Add complete footer
                        if last_cell["source"] and not last_cell["source"][-1].endswith("\n"):
                            last_cell["source"][-1] += "\n"
                        last_cell["source"].extend(
                            _source_lines(text_for_last_cell)
                        )
                    updated = True  # Mark as updated only if content was added
            else:
                notebook_content["cells"].append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": _source_lines(text_for_last_cell),
                    }
                )
                updated = True

        # Ensure GPU metadata is set for Colab
        if "metadata" not in notebook_content:
            notebook_content["metadata"] = {}
        if "accelerator" not in notebook_content["metadata"]:
            notebook_content["metadata"]["accelerator"] = "GPU"
            updated = True
        if "colab" not in notebook_content["metadata"]:
            notebook_content["metadata"]["colab"] = {"provenance": [], "gpuType" : "T4", "include_colab_link": True}
            updated = True
        # Override gpuType for A100 notebooks (filename-based fallback)
        if "A100" in notebook_path and notebook_content["metadata"]["colab"].get("gpuType", "T4") == "T4":
            notebook_content["metadata"]["colab"]["gpuType"] = "A100"
            updated = True
        if "kernelspec" not in notebook_content["metadata"]:
            notebook_content["metadata"]["kernelspec"] = {
                "display_name": "Python 3",
                "name": "python3",
            }
            updated = True
        # Fix rendering in github
        if "widgets" not in notebook_content["metadata"]:
            notebook_content["metadata"]["widgets"] = {
                "application/vnd.jupyter.widget-state+json" : {
                    "state" : {}
                }
            }
            updated = True
        if notebook_content["metadata"]["widgets"].get("application/vnd.jupyter.widget-state+json", None) is not None:
            if notebook_content["metadata"]["widgets"]["application/vnd.jupyter.widget-state+json"].get("state") != {}:
                notebook_content["metadata"]["widgets"]["application/vnd.jupyter.widget-state+json"]["state"] = {}
                updated = True

        if updated:
            _write_notebook(notebook_path, notebook_content)
            print(f"Updated: {notebook_path}")
        else:
            print(f"No sections found to update in: {notebook_path}")

    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in notebook at {notebook_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {notebook_path}: {e}")


def replace(text, g, f):
    text = text.replace("(", r"\(")
    text = text.replace(")", r"\)")
    if g == "":
        g = g + "\n"
    else:
        g = "\1" + g + "\2"
    f = re.sub(
        r"([\s]{1,})([\"\'][ ]{0,})" + text + r"(\\n[\"\']\,\n)",
        g,
        f,
        flags = re.MULTILINE,
    )
    if " = " not in text:
        # Also replace x=x and x = x
        text = text.replace("=", " = ")
        f = re.sub(
            r"([\s]{1,})([\"\'][ ]{0,})" + text + r"(\\n[\"\']\,\n)",
            g,
            f,
            flags = re.MULTILINE,
        )
    return f
pass

def update_unsloth_config(filename):
    with open(filename, "r", encoding="utf-8", newline="") as f: f = f.read()
    if "from transformers import TrainingArguments\\n" not in f: return
    if "from trl import SFTTrainer\\n" not in f: return
    if "SFTConfig" in f: return
    if "UnslothTrainingArguments" in f: return

    f = replace("from unsloth import is_bfloat16_supported", "", f)
    f = replace("from transformers import TrainingArguments", "", f)
    f = f.replace("from trl import SFTTrainer", "from trl import SFTTrainer, SFTConfig")
    f = f.replace("TrainingArguments(\\n", "SFTConfig(\\n")
    f = replace("fp16=not is_bfloat16_supported(),", "", f)
    f = replace("bf16=is_bfloat16_supported(),", "", f)
    f = replace("fp16 = not is_bfloat16_supported(),", "", f)
    f = replace("bf16 = is_bfloat16_supported(),", "", f)
    f = replace("logging_steps=1,", "", f)
    f = replace("logging_steps = 1,", "", f)
    f = replace("dataset_num_proc=2,", "", f)
    f = replace("dataset_num_proc=4,", "", f)
    f = replace("dataset_num_proc = 2,", "", f)
    f = replace("dataset_num_proc = 4,", "", f)

    # Fix all spacings x=x to x = x
    spaces = r'(\"[ ]{4,}[^\<\n]{1,}[^ \=\'\"])\=([^ \=\'\"].*?\,\n)'
    f = re.sub(spaces, r"\1 = \2", f)

    with open(filename, "w", encoding="utf-8", newline="") as w: w.write(f)
pass


_MODEL_NAME_PREFIX_CACHE = {}

_RE_MODEL_NAME_ASSIGN = re.compile(
    r'(model_name\s*=\s*["\'])([A-Za-z0-9_-]+)/([A-Za-z0-9._-]+)(["\'])'
)
_RE_FROM_PRETRAINED_INLINE = re.compile(
    r'(from_pretrained\(\s*["\'])([A-Za-z0-9_-]+)/([A-Za-z0-9._-]+)(["\'])'
)
_RE_FROM_PRETRAINED_MULTILINE = re.compile(
    r'(from_pretrained\(\s*\n\s*["\'])([A-Za-z0-9_-]+)/([A-Za-z0-9._-]+)(["\'])'
)

_MODEL_PREFIX_SKIP_ORGS = {"unsloth", "LiquidAI"}
_MODEL_PREFIX_SKIP_EXACT = {"meta-llama/Llama-3.2-3B-Instruct"}
_MODEL_PREFIX_SKIP_PATTERNS = {"bert-base-uncased"}
_PROGRESS_ENABLED = False
_WINDOWS_PROCESSPOOL_MAX_WORKERS = 61


def _set_progress(enabled):
    """Enable/disable progress bars globally (best-effort if tqdm is unavailable)."""
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)
    if _PROGRESS_ENABLED and _tqdm is None:
        print("  [WARN] tqdm is not installed; progress bars are disabled.")
        _PROGRESS_ENABLED = False


def _progress_iter(iterable, total=None, desc=None):
    """Wrap an iterable with tqdm when progress is enabled."""
    if _PROGRESS_ENABLED and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
    return iterable


def _effective_worker_count(requested_workers, total_items, executor_type, platform_name=None, cpu_count=None):
    """Clamp worker counts for stability and platform limits."""
    if platform_name is None:
        platform_name = os.name
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1

    requested = max(1, int(requested_workers))
    workers = requested
    workers = min(workers, max(1, int(cpu_count)))

    if total_items is not None and total_items > 0:
        workers = min(workers, int(total_items))

    if executor_type == "process" and platform_name == "nt":
        workers = min(workers, _WINDOWS_PROCESSPOOL_MAX_WORKERS)

    return max(1, workers)


def _should_fallback_process_error(exc):
    """Return True for process-pool bootstrap/pickling errors suitable for thread fallback."""
    if isinstance(exc, concurrent.futures.process.BrokenProcessPool):
        return True
    if isinstance(exc, (pickle.PicklingError, OSError)):
        return True

    msg = str(exc).lower()
    markers = (
        "can't pickle",
        "cannot pickle",
        "pickl",
        "brokenprocesspool",
        "freeze_support",
        "bootstrapping phase",
        "cannot find",
        "__main__",
    )
    return any(marker in msg for marker in markers)


def _can_use_process_executor():
    """Return False for interactive/non-file entrypoints where spawn cannot import __main__."""
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if not main_file:
        return False
    bad_markers = ("<stdin>", "<string>")
    return not any(marker in str(main_file) for marker in bad_markers)


def _unsloth_model_exists(model_name):
    """Check if unsloth/<model_name> exists on HF Hub. Results are cached."""
    if model_name in _MODEL_NAME_PREFIX_CACHE:
        return _MODEL_NAME_PREFIX_CACHE[model_name]
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.model_info(f"unsloth/{model_name}")
        _MODEL_NAME_PREFIX_CACHE[model_name] = True
    except Exception:
        _MODEL_NAME_PREFIX_CACHE[model_name] = False
    return _MODEL_NAME_PREFIX_CACHE[model_name]


def fix_model_name_prefix(notebook_file):
    """Replace non-unsloth model name prefixes with unsloth/ where the model exists."""
    try:
        with open(notebook_file, "r", encoding="utf-8", newline="") as f:
            nb = json.load(f)
    except Exception:
        return False

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]
        text = "".join(source)

        new_text = text
        for pattern in [_RE_MODEL_NAME_ASSIGN, _RE_FROM_PRETRAINED_INLINE,
                        _RE_FROM_PRETRAINED_MULTILINE]:
            def _replace_model(m):
                prefix, org, model, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
                full_name = f"{org}/{model}"
                if org in _MODEL_PREFIX_SKIP_ORGS:
                    return m.group(0)
                if full_name in _MODEL_PREFIX_SKIP_EXACT:
                    return m.group(0)
                if any(p in model.lower() for p in _MODEL_PREFIX_SKIP_PATTERNS):
                    return m.group(0)
                if _unsloth_model_exists(model):
                    return f"{prefix}unsloth/{model}{suffix}"
                return m.group(0)
            new_text = pattern.sub(_replace_model, new_text)

        if new_text != text:
            cell["source"] = _source_lines(new_text)
            changed = True

    if changed:
        _write_notebook(notebook_file, nb)
    return changed


def _process_single_notebook(notebook_file):
    """Process a single notebook: update sections, fix content, check spelling & syntax."""
    _cache_notebook_format(notebook_file)
    update_notebook_sections(
        notebook_file,
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        new_announcement,
    )
    update_old_unsloth(notebook_file)
    fix_model_name_prefix(notebook_file)

    spell_issues = []
    syntax_errors = []
    spell_fixed = False

    # Spelling check (create SpellChecker per call for thread safety)
    try:
        with open(notebook_file, "r", encoding="utf-8", newline="") as f:
            nb_content = json.load(f)
        spell = SpellChecker()
        spell.word_frequency.load_words(SPELL_IGNORE_WORDS)
        fixed, issues = check_spelling(nb_content, os.path.basename(notebook_file), spell=spell)
        if fixed:
            _write_notebook(notebook_file, nb_content)
            spell_fixed = True
        spell_issues = issues
    except Exception:
        pass

    # AST syntax check
    errors = validate_notebook_syntax(notebook_file)
    syntax_errors = errors

    return notebook_file, spell_fixed, spell_issues, syntax_errors


def _map_with_executor(func, items, max_workers=1, executor_type="process", progress_desc=None):
    items = list(items)
    if not items:
        return []

    effective_workers = _effective_worker_count(max_workers, len(items), executor_type)
    if effective_workers != max_workers:
        print(
            f"  [INFO] Adjusted worker count from {max_workers} to {effective_workers} "
            f"for executor={executor_type} (items={len(items)})."
        )

    if effective_workers <= 1:
        return [func(item) for item in _progress_iter(items, total=len(items), desc=progress_desc)]

    if executor_type == "process":
        if not _can_use_process_executor():
            print("  [WARN] Process executor unavailable for interactive/__main__ context; using thread executor.")
        else:
            chunksize = max(1, len(items) // max(1, effective_workers * 8))
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=effective_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    mapped = executor.map(func, items, chunksize=chunksize)
                    return list(_progress_iter(mapped, total=len(items), desc=progress_desc))
            except Exception as e:
                if not _should_fallback_process_error(e):
                    raise
                print(f"WARNING: process executor failed ({e}); falling back to thread executor.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        mapped = executor.map(func, items)
        return list(_progress_iter(mapped, total=len(items), desc=progress_desc))


def main(max_workers=1, executor_type="process"):
    notebook_directory = "nb"
    notebook_pattern = "*.ipynb"

    notebook_files = glob(os.path.join(notebook_directory, notebook_pattern))
    print(f"Found {len(notebook_files)} notebooks")
    # filter out the DONT_UPDATE_EXCEPTIONS
    notebook_files = [x for x in notebook_files if os.path.basename(x) not in DONT_UPDATE_EXCEPTIONS]
    print(f"Filtered out {len(DONT_UPDATE_EXCEPTIONS)} notebooks")
    print(f"Remaining {len(notebook_files)} notebooks")

    if not notebook_files:
        print(
            f"No notebooks found in the directory: {notebook_directory} with pattern: {notebook_pattern}"
        )
        return

    spell_issues_found = False
    syntax_issues_found = False

    results = _map_with_executor(
        _process_single_notebook,
        notebook_files,
        max_workers=max_workers,
        executor_type=executor_type,
        progress_desc="Notebook processing",
    )

    for notebook_file, spell_fixed, spell_issues, syntax_errors in results:
        if spell_fixed:
            print(f"  AUTO-FIXED spelling in {os.path.basename(notebook_file)}")
        if spell_issues:
            spell_issues_found = True
            for cell_idx, words in spell_issues:
                print(f"  SPELLING: {os.path.basename(notebook_file)} cell {cell_idx}: {words}")
        if syntax_errors:
            syntax_issues_found = True
            for cell_idx, lineno, msg in syntax_errors:
                print(f"  SYNTAX: {os.path.basename(notebook_file)} cell {cell_idx} line {lineno}: {msg}")

    print("\n=== Spelling Check ===")
    if not spell_issues_found:
        print("  No spelling issues found.")

    print("\n=== AST Syntax Check ===")
    if not syntax_issues_found:
        print("  No syntax issues found.")


# NOTE: add_colab_badge is not part of the main pipeline; kept for potential external use.
def add_colab_badge(notebooks_dir):
    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    for path in paths:
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"])
        is_colab = not is_kaggle
        if is_colab:
            with open(path, "r", encoding="utf-8", newline="") as f:
                notebook_content = json.load(f)

            badge = badge_section.format(link_colab=(f"https://colab.research.google.com/github/unslothai/notebooks/blob/main/"+path).replace(" ", "%20"))
            notebook_content["cells"].insert(
                0,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": _source_lines(badge),
                },
            )

            _write_notebook(path, notebook_content)


def update_readme(
    args,
    readme_path,
    notebooks_dir,
    architecture_mapping, 
    known_types_ordered,  
    type_order=None,      
    kaggle_accelerator="nvidiaTeslaT4",
):
    base_url_colab = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/"
    base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/"

    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    # Scan notebooks for HF model refs and refresh the model_created_at cache
    # so we can sort rows by the most recently created referenced model.
    try:
        model_created_cache, refs_by_nb, assigned_by_nb = refresh_model_created_cache(paths)
    except Exception as e:
        print(f"  [WARN] Could not refresh model created_at cache: {e}")
        model_created_cache, refs_by_nb, assigned_by_nb = {}, {}, {}

    # Priority sections appear first in the README, in this order
    priority_sections = [
        "GRPO & Reinforcement Learning",
        "Tool Calling",
        "Text-to-Speech (TTS)",
        "Vision (Multimodal)",
        "Embedding",
        "Speech-to-Text (STT)",
        "OCR",
    ]

    unique_architectures = sorted(list(set(architecture_mapping.values())))
    # Build section list: priority first, then remaining architectures alphabetically
    list_models = list(priority_sections)
    for arch in unique_architectures:
        if arch not in list_models:
            list_models.append(arch)

    # Place "Gemma 4" immediately before "Gemma" so the newer family appears
    # above its predecessor instead of after it (default alphabetical sort
    # would order "Gemma" before "Gemma 4" because "Gemma" is a prefix).
    if "Gemma 4" in list_models and "Gemma" in list_models:
        list_models.remove("Gemma 4")
        list_models.insert(list_models.index("Gemma"), "Gemma 4")

    # Cross-cutting sections (notebooks can appear in multiple sections)
    for cross_section in ["Vision (Multimodal)", "Embedding", "OCR"]:
        if cross_section not in list_models:
            list_models.append(cross_section)

    # "Text Completion / Continued Pretraining" collects notebooks whose
    # primary purpose is base-model continued pretraining or raw text
    # completion, sitting just before "Other" at the end of the README.
    _TEXT_COMPLETION_SECTION = "Text Completion / Continued Pretraining"
    if _TEXT_COMPLETION_SECTION not in list_models:
        list_models.append(_TEXT_COMPLETION_SECTION)

    list_models.append('Other')

    sections = {}
    for section in list_models:
        sections[section] = {
            "Colab": {"header": f"### {section} Notebooks\n", "rows": []},
            "Kaggle": {"header": f"### {section} Notebooks\n", "rows": []},
        }

    colab_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"
    kaggle_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"

    notebook_data = []

    print(f"Processing {len(paths)} notebooks...")
    for path in paths:
        # Ignore HF course and Advanced notebooks
        if is_path_contains_any(path.lower(), [hf_course_name.lower(), "Advanced".lower()]):
            continue
        if _should_skip_readme_notebook(path):
            continue

        notebook_name = os.path.basename(path)
        old_notebook_name = notebook_name
        check = False
        if notebook_name in FIRST_MAPPING_NAME:
            notebook_name = FIRST_MAPPING_NAME[notebook_name]
            check = True
        
        # For Kaggle
        if notebook_name.lstrip("Kaggle-") in FIRST_MAPPING_NAME:
            notebook_name = FIRST_MAPPING_NAME[notebook_name.lstrip("Kaggle-")]
            notebook_name = "Kaggle-" + notebook_name

        std_notebook_name = notebook_name.replace("-", "_")
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"]) 

        try:
            info = extract_model_info_refined(
                std_notebook_name,
                architecture_mapping,
                known_types_ordered
            )
        except Exception as e:
            print(f"Error processing {notebook_name}: {e}")
            info = {'name': notebook_name.replace('.ipynb',''), 'size': None, 'type': 'Error', 'architecture': None, 'requires_a100': False} # Fallback

        model_name = info['name'] if info and info['name'] else notebook_name.replace('.ipynb','')
        # Apply per-notebook display-name override (keyed by on-disk basename)
        # so notebooks with very long auto-derived names can wrap onto multiple
        # lines in the rendered Markdown table.
        on_disk_basename = os.path.basename(path)
        if on_disk_basename in README_MODEL_NAME_OVERRIDES:
            model_name = README_MODEL_NAME_OVERRIDES[on_disk_basename]
        model_type = info['type'] if info and info['type'] else ""
        # Classify RL/GRPO notebooks by the task they actually train on
        # (GSM8K Math, DAPO Math, Vision Math, Wordle, Sudoku, 2048 Game,
        # Auto Kernel Creation, Multi Environment, ...). This is driven by
        # the dataset name or markdown headers, not the filename, and
        # overrides the generic "GRPO" label from the filename classifier.
        # The "GRPO" prefix is redundant with the section header, so we
        # drop it entirely when a task is detected.
        basename_lower_for_rl = os.path.basename(path).lower()
        # Detect the TRL trainer class the notebook trains with. This is
        # the authoritative signal for RL-style training (GRPO / DPO /
        # ORPO / KTO / Reward / PPO). Notebooks that import one of these
        # classes land in the GRPO & Reinforcement Learning section, and
        # a "(GRPO RL)" suffix is added to the Type when a GRPO notebook
        # is rendered in a non-GRPO cross-cutting section.
        trainer_class = detect_trainer_class(path)
        is_grpo_trainer = trainer_class == "GRPOTrainer"
        is_dpo_trainer = trainer_class == "DPOTrainer"
        is_orpo_trainer = trainer_class == "ORPOTrainer"
        # GRPO / RL notebooks: identified by trainer class, model_type OR
        # filename. The trainer-class check catches notebooks whose type
        # the filename classifier assigned as "Vision GRPO" / "FP8 GRPO"
        # (GRPO as a suffix) or even as "DPO" / "ORPO".
        is_in_grpo_section = (
            is_grpo_trainer
            or is_dpo_trainer
            or is_orpo_trainer
            or model_type.startswith("GRPO")
            or "grpo" in basename_lower_for_rl
            or "nemo-gym" in basename_lower_for_rl
            or "nemo_gym" in basename_lower_for_rl
            or "reinforcement_learning" in basename_lower_for_rl
        )
        if is_dpo_trainer:
            # Pure DPO preference optimization. Route to GRPO section and
            # force the Type to "DPO" regardless of what the classifier
            # inferred from the filename.
            model_type = "DPO"
        elif is_orpo_trainer:
            model_type = "ORPO"
        elif is_in_grpo_section:
            task = detect_rl_task(path)
            if task:
                model_type = task
            elif not model_type:
                model_type = "GRPO"
        # Notebooks that enable Unsloth's fast_inference flag are using
        # vLLM under the hood. Surface that in the Type column so readers
        # can tell at a glance which variants ship vLLM. We only append
        # the suffix for GRPO-class training so DPO / ORPO rows stay
        # clean.
        if is_grpo_trainer and notebook_uses_fast_inference(path):
            model_type = (model_type or "GRPO") + " + vLLM"
        architecture = info['architecture'] if info else None
        size = info['size']
        size = size.replace(r"_", " ") if size else None
        size = f"**({size})**" if size else ""

        requires_a100 = info.get('requires_a100', False)

        # Primary section (architecture-based)
        section_name = "Other"
        # Force-route notebooks whose filename signals a GRPO / RL environment
        # even though the classifier did not tag them with model_type='GRPO'.
        # Examples: NeMo-Gym-Sudoku.ipynb, NeMo-Gym-Multi-Environment.ipynb.
        basename_lower = os.path.basename(path).lower()
        is_forced_grpo = any(
            kw in basename_lower for kw in ["nemo-gym", "nemo_gym"]
        )
        # Force-route text completion / continued pretraining notebooks so
        # they land in the dedicated section instead of the architecture one.
        # We key off the classified type and the filename because some
        # notebooks have type="" in the cache but a clear filename.
        is_text_completion = (
            model_type in ("Text Completion", "CPT")
            or "text_completion" in basename_lower
            or "-cpt" in basename_lower
            or "_cpt" in basename_lower
        )
        # Force-route tool-calling notebooks to the dedicated Tool Calling
        # section. All FunctionGemma notebooks belong here regardless of
        # their filename subtype (Multi-Turn-Tool-Calling, LMStudio,
        # Mobile-Actions, etc.) since FunctionGemma is purpose-built for
        # function calling. Other notebooks match on type or filename.
        is_tool_calling = (
            "tool calling" in model_type.lower()
            or "functiongemma" in basename_lower
            or "tool_calling" in basename_lower
            or "tool-calling" in basename_lower
        )
        # Use the precomputed is_in_grpo_section flag instead of checking
        # model_type here -- by this point the RL classifier may have
        # already renamed model_type to "GSM8K Math", "Wordle", etc. which
        # would no longer start with "GRPO".
        if is_in_grpo_section or is_forced_grpo:
            section_name = 'GRPO & Reinforcement Learning'
        elif is_tool_calling:
            section_name = 'Tool Calling'
        elif is_text_completion:
            section_name = _TEXT_COMPLETION_SECTION
        elif architecture and architecture in list_models:
            section_name = architecture

        # Build list of sections this notebook belongs to (primary + cross-cutting)
        sections_for_notebook = [section_name]

        # Cross-cutting: Vision (Multimodal) -- excludes OCR notebooks
        name_lower = os.path.basename(path).lower()
        is_ocr = "ocr" in name_lower
        if not is_ocr and any(kw in name_lower for kw in ["vision", "_vl_", "_vl.", "-vl-", "-vl.", "multimodal"]):
            if "Vision (Multimodal)" not in sections_for_notebook:
                sections_for_notebook.append("Vision (Multimodal)")

        # Cross-cutting: OCR
        if is_ocr:
            if "OCR" not in sections_for_notebook:
                sections_for_notebook.append("OCR")

        # Cross-cutting: Embedding
        if any(kw in name_lower for kw in ["embedding", "minilm", "bge", "modernbert", "bert_classification"]):
            if "Embedding" not in sections_for_notebook:
                sections_for_notebook.append("Embedding")

        link_base = base_url_kaggle if is_kaggle else base_url_colab
        link_url = f"{link_base}{path}"

        if is_kaggle:
            image_src = "https://kaggle.com/static/images/open-in-kaggle.svg"
            image_alt = "Open in Kaggle"
            if kaggle_accelerator:
                link_url += f"&accelerator={kaggle_accelerator}"
        else:
            image_src = "https://colab.research.google.com/assets/colab-badge.svg"
            image_alt = "Open In Colab"
        link = f'<a href="{link_url}" target="_blank" rel="noopener noreferrer"><img src="{image_src}" alt="{image_alt}"></a>'

        created_at_key = notebook_created_at_key(
            path, refs_by_nb, model_created_cache, assigned_by_nb=assigned_by_nb
        )

        notebook_data.append(
            {
                "model": model_name,
                "type": model_type,
                "link": link,
                "sections": sections_for_notebook,
                "path": path,
                "architecture" : architecture,
                "size" : size,
                "requires_a100": requires_a100,
                "created_at_key": created_at_key,
                "is_grpo_trainer": is_grpo_trainer,
            }
        )

    def get_sort_key(x):
        version_key = extract_version(x["model"])

        type_sort_val = float("inf") 
        current_type = x["type"].strip('*') 
        if type_order and current_type in type_order:
            type_sort_val = type_order.index(current_type)
        elif current_type: 
             type_sort_val = current_type

        return version_key

    notebook_data.sort(key=get_sort_key)

    _grpo_section_name = "GRPO & Reinforcement Learning"
    for data in notebook_data:
        model_prefix = "(A100) " if data.get('requires_a100', False) else ""
        platform = "Kaggle" if "kaggle" in data['link'].lower() else "Colab"
        raw_type = data.get("type") or ""
        # Strip the " + vLLM" suffix to get the base task type for grouping.
        # Both "GSM8K Math" and "GSM8K Math + vLLM" should group together
        # when we interleave the GRPO section by task type below.
        task_type = raw_type.replace(" + vLLM", "").strip() or "Other"
        is_grpo_trainer_row = data.get("is_grpo_trainer", False)
        for section_name in data["sections"]:
            # Cross-section suffix: a GRPOTrainer notebook appearing in a
            # non-GRPO cross-cutting section (e.g. Vision (Multimodal))
            # gets "(GRPO RL)" appended so readers can tell it is an RL
            # notebook and not an SFT Vision notebook that happened to
            # share the same model family.
            display_type = raw_type
            if is_grpo_trainer_row and section_name != _grpo_section_name:
                if display_type:
                    display_type = f"{display_type} (GRPO RL)"
                else:
                    display_type = "(GRPO RL)"
            row = (
                f"| **{model_prefix}{data['model']}** {data['size']} | "
                f"{display_type} | {data['link']} |\n"
            )
            row_entry = {
                "row": row,
                "popularity_key": data.get("created_at_key", (0, 0)),
                # Boolean flag -- GRPO + vLLM rows sort to the top of any
                # section they appear in, regardless of raw popularity.
                "has_vllm": "vLLM" in raw_type,
                # Base task type (no " + vLLM" suffix) for the GRPO
                # section round-robin interleave. Used only by
                # _interleave_by_task.
                "task_type": task_type,
            }
            sections[section_name][platform]["rows"].append(row_entry)

    def _section_row_sort_key(entry):
        # Top bucket: rows whose Type column mentions vLLM (e.g. "GRPO + vLLM")
        # always sort above non-vLLM rows in the same section. This puts the
        # vLLM-enabled GRPO notebooks at the top of the GRPO section.
        # Secondary: HF Hub popularity score (downloads + likes*1000) of the
        # model the notebook actually loads. Notebooks with no resolvable
        # model get 0 which sorts below every real score in a descending
        # sort.
        # Tertiary: count of ok-status refs that contributed to the score.
        # Quaternary: version-from-name fallback so Gemma 4 > Gemma 3 when
        # popularity ties (e.g. brand-new releases with 0 downloads).
        # Quinary: the row string itself for stable ordering.
        popularity, count_ok = entry["popularity_key"]
        return (
            1 if entry.get("has_vllm") else 0,
            popularity,
            count_ok,
            extract_version_from_row(entry["row"]),
            entry["row"],
        )

    def _unique_types_first(entries):
        """Order rows so every distinct task type appears at the top first,
        then duplicates fall in behind, both phases sorted by popularity.

        Concretely: sort the incoming list by popularity globally (with
        vLLM as a tiebreaker so Llama3.1 GSM8K + vLLM beats Gemma3 GSM8K
        when GSM8K's representative is picked). Then walk the sorted list
        once -- the first time we see a given task_type, that row becomes
        the representative of its type; every later row with the same
        type is a duplicate. Concatenate all representatives (in
        popularity order) followed by all duplicates (also in popularity
        order).

        The result: rows 1..N of the section show N distinct task types,
        ordered by the best row in each type, and rows N+1.. are the
        leftover duplicates in popularity order.
        """
        def _rank(entry):
            # Tuple used in descending sort. Popularity first, has_vllm
            # as a minor tiebreaker so vLLM rows outrank non-vLLM rows
            # with the same popularity. popularity_key is already a
            # (score, count_ok) tuple from the upstream sort.
            pop_score, pop_count = entry.get("popularity_key", (0, 0))
            return (pop_score, pop_count, 1 if entry.get("has_vllm") else 0)

        sorted_entries = sorted(entries, key=_rank, reverse=True)
        seen_types = set()
        representatives = []
        duplicates = []
        for e in sorted_entries:
            task = e.get("task_type") or "Other"
            if task in seen_types:
                duplicates.append(e)
            else:
                seen_types.add(task)
                representatives.append(e)
        return representatives + duplicates

    for section in sections:
        try:
            sections[section]["Colab"]["rows"].sort(key=_section_row_sort_key, reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Colab rows for section '{section}': {e}")
        try:
            sections[section]["Kaggle"]["rows"].sort(key=_section_row_sort_key, reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Kaggle rows for section '{section}': {e}")

    # Re-order the GRPO section so every distinct task type appears once
    # at the top (each row of phase 1 is a different type, sorted by the
    # popularity of the best row in that type), then duplicates come after
    # (also sorted by popularity). This only applies to "GRPO &
    # Reinforcement Learning"; every other section stays in pure
    # popularity order.
    _grpo_section = "GRPO & Reinforcement Learning"
    if _grpo_section in sections:
        for platform in ("Colab", "Kaggle"):
            sections[_grpo_section][platform]["rows"] = _unique_types_first(
                sections[_grpo_section][platform]["rows"]
            )

    # Flatten row entries back into raw strings for the rendering step below.
    for section in sections:
        for platform in ("Colab", "Kaggle"):
            sections[section][platform]["rows"] = [
                e["row"] if isinstance(e, dict) else e
                for e in sections[section][platform]["rows"]
            ]

    try:
        with open(readme_path, "r", encoding="utf-8", newline="") as f:
            readme_content = f.read()

        start_marker = "<!-- START OF EDITING -->"
        start_index = readme_content.find(start_marker)
        if start_index == -1:
            raise ValueError(f"Start marker '{start_marker}' not found in README.")
        start_index += len(start_marker)

        end_marker_alt = None
        end_marker = "<!-- End of Notebook Links -->"
        end_index = readme_content.find(end_marker)
        if end_index == -1:
            end_marker_alt = "# 📒 Kaggle Notebooks"
            end_index = readme_content.find(end_marker_alt)
            if end_index == -1:
                raise ValueError(f"End marker '{end_marker}' or '{end_marker_alt}' not found in README.")
        content_before = readme_content[:start_index]
        content_after = readme_content[end_index:] 

        temp = (
            "(https://github.com/unslothai/notebooks/#-kaggle-notebooks).\n\n"
            if args.to_main_repo
            else "(https://github.com/unslothai/notebooks/#-kaggle-notebooks).\n\n"
        )

        colab_updated_notebooks_links = "\n"

        kaggle_updated_notebooks_links = (
            "# 📒 Kaggle Notebooks\n"
            "<details>\n  <summary>\n" 
            "    Click for all our Kaggle notebooks categorized by model:\n  "
            "</summary>\n\n"
        )

        # Static "Specific use-case Notebooks" section (Colab only, before "Other")
        specific_usecase_section = (
            "### Specific use-case Notebooks\n"
            "| Usecase | Model | Notebook Link |\n"
            "| --- | --- | --- |\n"
            '| Text Classification | Llama 3.1 (8B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb) |\n'
            '| Tool Calling | Qwen2.5-Coder (1.5B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb) |\n'
            '| Multiple Datasets | | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing) |\n'
            '| KTO | Qwen2.5-Instruct (1.5B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing) |\n'
            '| Inference Chat UI | LLaMa 3.2 Vision | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Unsloth_Studio.ipynb) |\n'
            '| Conversational | LLaMa 3.2 (1B and 3B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb) |\n'
            '| ChatML | Mistral (7B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing) |\n'
            '| Text Completion | Mistral (7B) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb) |\n'
            "\n"
        )

        for section in list_models:
            # Insert "Specific use-case Notebooks" just before "Other"
            if section == "Other":
                colab_updated_notebooks_links += specific_usecase_section

            if sections[section]["Colab"]["rows"]:
                colab_updated_notebooks_links += sections[section]["Colab"]["header"]
                colab_updated_notebooks_links += colab_table_header
                colab_updated_notebooks_links += "".join(sections[section]["Colab"]["rows"]) + "\n"

            if sections[section]["Kaggle"]["rows"]:
                kaggle_updated_notebooks_links += sections[section]["Kaggle"]["header"]
                kaggle_updated_notebooks_links += kaggle_table_header
                kaggle_updated_notebooks_links += "".join(sections[section]["Kaggle"]["rows"]) + "\n"

        kaggle_updated_notebooks_links += "</details>\n\n"

        now = datetime.now() 
        timestamp = f"\n"

        updated_readme_content = (
            content_before
            + colab_updated_notebooks_links
            + kaggle_updated_notebooks_links 
            + timestamp
            + content_after 
        )

        if end_marker_alt and end_index != -1:
             content_after = readme_content[end_index:]
             next_section_index = content_after.find("\n#")
             if next_section_index != -1:
                 content_after = content_after[next_section_index:] 
             else:
                  
                  explicit_end_marker_index = content_after.find("")
                  if explicit_end_marker_index != -1:
                      content_after = content_after[explicit_end_marker_index:]
                  else:
                      content_after = "" 

             updated_readme_content = ( 
                content_before
                + colab_updated_notebooks_links
                + kaggle_updated_notebooks_links 
                + timestamp
                + content_after
             )


        with open(readme_path, "w", encoding="utf-8", newline="") as f:
            f.write(updated_readme_content)

        print(f"Successfully updated {readme_path}")

    except FileNotFoundError:
        print(f"Error: README file '{readme_path}' not found.")
    except ValueError as ve:
        print(f"Error processing README: {ve}")
    except Exception as e:
        print(f"An error occurred while updating {readme_path}: {e}")
        import traceback
        traceback.print_exc()


def copy_and_update_notebooks(
    template_dir,
    destination_dir,
    general_announcement,
    installation,
    installation_kaggle,
    new_announcement,
):
    """Copies notebooks from template_dir to destination_dir, updates them, and renames them."""
    template_notebooks = glob(os.path.join(template_dir, "*.ipynb"))

    temp_location = os.path.join(destination_dir, ".temp_backup")
    if os.path.exists(destination_dir):
        if os.path.exists(temp_location):
            _rmtree_robust(temp_location)
        os.makedirs(temp_location, exist_ok=True)
        # Move everything currently in destination_dir into .temp_backup
        for entry in os.listdir(destination_dir):
            if entry == ".temp_backup":
                continue
            if entry not in DONT_UPDATE_EXCEPTIONS:
                continue
            src_path = os.path.join(destination_dir, entry)
            shutil.move(src_path, temp_location)
    else:
        os.makedirs(destination_dir, exist_ok=True)

    def _preserve_outputs(dest_path, template_path):
        """Copy template to dest, caching original outputs for deferred restoration."""
        _cache_original_outputs(dest_path)
        shutil.copyfile(template_path, dest_path)
        _set_file_permissions(dest_path)
        _cache_notebook_format(dest_path)

    for template_notebook_path in template_notebooks:
        notebook_name = os.path.basename(template_notebook_path)

        colab_notebook_name = notebook_name
        destination_notebook_path = os.path.join(destination_dir, colab_notebook_name)

        _preserve_outputs(destination_notebook_path, template_notebook_path)
        print(f"Copied '{colab_notebook_name}' to '{destination_dir}'")

        kaggle_notebook_name = "Kaggle-" + notebook_name
        destination_notebook_path = os.path.join(destination_dir, kaggle_notebook_name)

        _preserve_outputs(destination_notebook_path, template_notebook_path)

        print(f"Copied '{kaggle_notebook_name}' to '{destination_dir}'")

        if "GRPO" in template_notebook_path:
            hf_course_notebook_name = f"{hf_course_name}-" + notebook_name
            destination_notebook_path = os.path.join(destination_dir, hf_course_notebook_name)
            _preserve_outputs(destination_notebook_path, template_notebook_path)
            print(f"Copied f'{hf_course_name}-{notebook_name}' to '{destination_notebook_path}'")

        update_notebook_sections(
            os.path.join(destination_dir, colab_notebook_name),
            general_announcement,
            installation,
            installation_kaggle,
            new_announcement,
        )

        update_notebook_sections(
            destination_notebook_path,
            general_announcement,
            installation_kaggle,
            installation_kaggle,
            new_announcement,
        )

    # Move Exceptions back to destination_dir from temp_location
    for entry in DONT_UPDATE_EXCEPTIONS:
        src_path = os.path.join(temp_location, entry)
        dst_path = os.path.join(destination_dir, entry)
        if os.path.exists(src_path):
            # shutil.rmtree(dst_path)
            shutil.move(src_path, dst_path)
            print(f"Moved '{entry}' back to '{dst_path}'")
        else:
            print(f"Warning: '{entry}' not found in '{temp_location}'")
    
    # finally remove the temp_location
    _rmtree_robust(temp_location)

def missing_files(nb: str | os.PathLike, original_template: str | os.PathLike) -> list[str]:
    nb_abs = os.path.abspath(nb)
    original_template_abs = os.path.abspath(original_template)

    files_in_nb = {f for f in os.listdir(nb_abs) if os.path.isfile(os.path.join(nb_abs, f))}
    files_in_original_template = {f for f in os.listdir(original_template_abs) if os.path.isfile(os.path.join(original_template_abs, f))}

    files_in_nb = {f for f in files_in_nb if not (f.startswith("Kaggle") or f.startswith("HuggingFace Course"))}
    files_in_original_template = {f for f in files_in_original_template if not f.startswith("Kaggle")}

    only_in_nb = files_in_nb - files_in_original_template
    return sorted(list(only_in_nb))


def remove_unwanted_section(script_content):
    start_marker = "# ### Installation"
    end_marker = "# ### Unsloth"

    start_index = script_content.find(start_marker)
    end_index = script_content.find(end_marker)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        before_section = script_content[:start_index]
        section_to_comment = script_content[start_index:end_index]
        after_section = script_content[end_index:]

        lines = section_to_comment.split('\n')
        commented_lines = [f"# {line}" for line in lines]
        commented_section = '\n'.join(commented_lines)
        return before_section + commented_section + after_section
    else:
        return script_content

def convert_notebook_to_script(notebook_path: str, output_path: str):
    exporter = PythonExporter()
    with open(notebook_path, 'r', encoding='utf-8', newline='') as f:
        notebook_json = json.load(f)

    # Add cell IDs only to the in-memory copy for nbformat validation;
    # do not write them back to the .ipynb file on disk.
    notebook_for_export = copy.deepcopy(notebook_json)
    _ensure_cell_ids(notebook_for_export)

    notebook_content = nbformat.reads(json.dumps(notebook_for_export), as_version=4)

    (body, resources) = exporter.from_notebook_node(notebook_content)

    body = remove_unwanted_section(body)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        f.write(body)
    _set_file_permissions(output_path)

    print(f"Converted {notebook_path} to {output_path}")


def _convert_notebook_task(task):
    notebook_path, output_path = task
    convert_notebook_to_script(notebook_path, output_path)


def convert_folder(
    input_folder: str,
    output_folder: str,
    max_workers: int = 1,
    executor_type: str = "process",
):
    if os.path.exists(output_folder):
        _rmtree_robust(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tasks = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join(input_folder, filename)
            script_filename = filename.replace('.ipynb', '.py')
            output_path = os.path.join(output_folder, script_filename)
            tasks.append((notebook_path, output_path))

    _map_with_executor(
        _convert_notebook_task,
        tasks,
        max_workers=max_workers,
        executor_type=executor_type,
        progress_desc="Notebook -> script",
    )


def _ensure_memory_stats_hidden(nb_path):
    """Ensure memory stats cells have cellView='form' metadata so they are collapsed."""
    memory_markers = ["Show current memory stats", "Show final memory"]
    try:
        with open(nb_path, "r", encoding="utf-8", newline="") as f:
            nb = json.load(f)
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            text = "".join(cell.get("source", []))
            if any(marker in text for marker in memory_markers):
                if cell.get("metadata", {}).get("cellView") != "form":
                    cell.setdefault("metadata", {})["cellView"] = "form"
                    changed = True
        if changed:
            _write_notebook(nb_path, nb)
    except Exception:
        pass


def _apply_global_fixes(nb_path):
    try:
        with open(nb_path, "r", encoding="utf-8", newline="") as f:
            raw = f.read()
        new_raw = raw
        if "qwen3_5" not in nb_path.lower():
            new_raw = _normalize_transformers_v5_pin(new_raw)
        new_raw = _RE_GATED_GLOBAL.sub(
            "# HF Token for gated models",
            new_raw,
        )
        new_raw = _RE_HUGGINGFACE_GLOBAL.sub(
            r"Hugging Face \1",
            new_raw,
        )
        new_raw = _RE_NOCOMMIT_GLOBAL.sub(
            '',
            new_raw,
        )
        for wrong, right in _ALL_NB_FIXES.items():
            new_raw = new_raw.replace(wrong, right)
        # Fix footer numbering (various formats)
        new_raw = _RE_FOOTER_NUM_NL.sub(r'\n4. See notebooks for DPO', new_raw)
        new_raw = _RE_FOOTER_NUM_Q.sub(r'"4. See notebooks for DPO', new_raw)
        # Fix duplicate "See our docs" sentences
        new_raw = _RE_DUP_DOCS_GLOBAL.sub(r'\1', new_raw)
        # Fix broken #Save link ONLY if file has NO <a name="Save"> anchor
        if '[how to save it](#Save)' in new_raw and '<a name="Save">' not in new_raw:
            new_raw = new_raw.replace('[how to save it](#Save)', 'how to save it')
        if new_raw != raw:
            with open(nb_path, "w", encoding="utf-8", newline="") as f:
                f.write(new_raw)
    except Exception as e:
        print(f"WARNING: Failed to apply global fixes to {nb_path}: {e}")
    try:
        _set_file_permissions(nb_path)
    except OSError:
        pass


def _summarize_git_diff():
    """Print a categorized summary of git diff (staged + unstaged) after a run."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return
        changed_files = [f for f in result.stdout.strip().splitlines() if f]
        if not changed_files:
            print("\n=== Git Diff Summary: no changes ===")
            return

        news_only = []
        install_only = []
        readme_changes = []
        new_files = []
        other_changes = []

        for filepath in changed_files:
            basename = os.path.basename(filepath)
            if basename == "README.md":
                readme_changes.append(filepath)
                continue

            # Check untracked (new) files
            stat_result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", filepath],
                capture_output=True, text=True, timeout=10,
            )
            if stat_result.returncode != 0:
                new_files.append(filepath)
                continue

            if not filepath.endswith(".ipynb"):
                other_changes.append(filepath)
                continue

            # Inspect the unified diff to categorize
            diff_result = subprocess.run(
                ["git", "diff", "--unified=0", filepath],
                capture_output=True, text=True, timeout=30,
            )
            diff_text = diff_result.stdout
            # Strip diff header lines
            diff_lines = [
                line for line in diff_text.splitlines()
                if line.startswith("+") or line.startswith("-")
            ]
            diff_lines = [
                line for line in diff_lines
                if not line.startswith("+++") and not line.startswith("---")
            ]
            content = "\n".join(diff_lines)

            # Heuristic categorization
            is_news = False
            is_install = False
            if "### News" in content or "new_announcement" in content.lower():
                is_news = True
            if "pip install" in content or "%%capture" in content:
                is_install = True

            if is_news and not is_install:
                news_only.append(filepath)
            elif is_install and not is_news:
                install_only.append(filepath)
            elif is_news and is_install:
                install_only.append(filepath)
            else:
                other_changes.append(filepath)

        total = len(changed_files)
        print(f"\n=== Git Diff Summary ({total} files changed) ===")
        if news_only:
            print(f"  News-only changes: {len(news_only)}")
        if install_only:
            print(f"  Installation-only changes: {len(install_only)}")
        if readme_changes:
            print(f"  README changes: {len(readme_changes)}")
        if new_files:
            print(f"  New files: {len(new_files)}")
            for f in new_files:
                print(f"    + {os.path.basename(f)}")
        if other_changes:
            print(f"  Other changes: {len(other_changes)}")
            for f in other_changes:
                print(f"    ~ {os.path.basename(f)}")

    except FileNotFoundError:
        # git not available
        pass
    except Exception as e:
        print(f"\n=== Git Diff Summary: error ({e}) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to_main_repo",
        action="store_true",
        help="Whether update notebooks and README.md for Unsloth main repository or not. Default is False.",
    )
    parser.add_argument(
        "--check_missing_files",
        action="store_true",
        help="Check for missing files in the destination directory compared to the original template.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="If true, instead of checking from original_template to nb, it will check nb to original_template instead"
    )
    parser.add_argument(
        "--disable_convert_to_script",
        action="store_true",
        help="If true, it will not convert the notebooks to scripts",
    )
    parser.add_argument(
        "--news_only",
        action="store_true",
        help="Only update the News section in all notebooks. Skips installation, README, spelling, and all other updates.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto-detect from CPU count, 1 = sequential). Default: 0 (auto).",
    )
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="process",
        help="Executor backend for parallel sections. Default: process.",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars for long phases.",
    )
    progress_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    if args.workers == 0:
        args.workers = max(2, min(os.cpu_count() or 4, 32))

    if args.no_progress:
        _set_progress(False)
    elif args.progress:
        _set_progress(True)
    else:
        _set_progress(sys.stderr.isatty())

    if args.check_missing_files:
        original_template = "original_template"
        nb = "nb"
        if args.reverse:
            missing_files_list = missing_files(original_template, nb)
        else:
            missing_files_list = missing_files(nb, original_template)
        if not missing_files_list:
            print("No missing files.")
        else:
            print(f"Missing files in {nb} compared to {original_template}:")
            for file in missing_files_list:
                if file not in DONT_UPDATE_EXCEPTIONS:
                    print(file)
        exit(0)

    if args.news_only:
        notebook_files = glob(os.path.join("nb", "*.ipynb"))
        print(f"[--news_only] Found {len(notebook_files)} notebooks")
        count = 0
        for nb_path in notebook_files:
            if _update_news_only(nb_path, new_announcement):
                count += 1
        print(f"[--news_only] Updated news in {count} notebooks")
        _summarize_git_diff()
        exit(0)

    # Cache format of existing notebooks before they are replaced by templates.
    # This preserves the original indent / trailing-newline of files already in nb/.
    for _nb_path in glob(os.path.join("nb", "*.ipynb")):
        _cache_notebook_format(_nb_path)

    copy_and_update_notebooks(
        "original_template",
        "nb",
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        new_announcement,
    )
    main(max_workers=args.workers, executor_type=args.executor)
    all_nb_paths = glob(os.path.join("nb", "*.ipynb"))
    _assert_vllm_install_usage_or_fast_inference(
        all_nb_paths,
        max_workers=args.workers,
        executor_type=args.executor,
    )

    notebook_directory = "nb"
    readme_path = "README.md"
    type_order = [
        "Alpaca",
        "Conversational",
        "CPT",
        "DPO",
        "ORPO",
        "Text_Completion",
        "CSV",
        "Inference",
        "Unsloth_Studio",
        "GRPO"
    ]  # Define your desired order here
    update_readme(
        args, 
        readme_path, 
        notebook_directory, 
        ARCHITECTURE_MAPPING,
        KNOWN_TYPES_ORDERED,
        type_order
    )

    # Apply targeted fixes to ALL notebooks (including DONT_UPDATE_EXCEPTIONS)
    # These are safe fixes that should apply everywhere.
    _map_with_executor(
        _apply_global_fixes,
        all_nb_paths,
        max_workers=args.workers,
        executor_type=args.executor,
        progress_desc="Global notebook fixes",
    )

    # Ensure memory stats cells are hidden behind cellView form
    for nb_path in _progress_iter(all_nb_paths, total=len(all_nb_paths), desc="Hide memory stats"):
        _ensure_memory_stats_hidden(nb_path)

    # Normalize LGPL blank line after all source modifications (update_old_unsloth
    # joins and re-splits source arrays, which can merge the blank line away).
    for nb_path in _progress_iter(all_nb_paths, total=len(all_nb_paths), desc="Normalize LGPL"):
        _normalize_lgpl_blank_line(nb_path)

    # Strip trailing empty strings from source arrays and ensure the last
    # element does not end with \n (nbformat convention).
    for nb_path in _progress_iter(all_nb_paths, total=len(all_nb_paths), desc="Normalize sources"):
        try:
            with open(nb_path, "r", encoding="utf-8", newline="") as f:
                nb_data = json.load(f)
            changed = False
            for cell in nb_data.get("cells", []):
                source = cell.get("source", [])
                if isinstance(source, list):
                    while source and source[-1] == "":
                        source.pop()
                        changed = True
                    if source and source[-1].endswith("\n"):
                        source[-1] = source[-1][:-1]
                        changed = True
            if changed:
                _write_notebook(nb_path, nb_data)
        except Exception:
            pass

    # Restore original output cells now that all processing is done and cell
    # counts should match the originals (templates gain cells during processing).
    for nb_path in _progress_iter(all_nb_paths, total=len(all_nb_paths), desc="Restore outputs"):
        _restore_original_outputs(nb_path)

    # Fix HTML-like tags in outputs that GitHub's renderer would hide.
    # Must run AFTER _restore_original_outputs (processes restored outputs).
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
    from fix_html_tags import fix_outputs as _fix_html_outputs
    from fix_html_tags import fix_comments as _fix_html_comments

    for nb_path in _progress_iter(all_nb_paths, total=len(all_nb_paths), desc="Fix HTML tags"):
        try:
            with open(nb_path, "r", encoding="utf-8", newline="") as f:
                nb_data = json.load(f)
            output_fixes = _fix_html_outputs(nb_data)
            comment_fixes = _fix_html_comments(nb_data)
            if output_fixes + comment_fixes > 0:
                _write_notebook(nb_path, nb_data)
        except Exception:
            pass

    if not args.disable_convert_to_script:
        convert_folder(
            "nb",
            "python_scripts",
            max_workers=args.workers,
            executor_type=args.executor,
        )

    _summarize_git_diff()
