import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from glob import glob
from nbconvert import PythonExporter
import nbformat

DONT_UPDATE_EXCEPTIONS = [
    'Falcon_H1-Alpaca.ipynb',
    'Liquid_LFM2-Conversational.ipynb',
    'Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb', # Daniel's?
]

MODAL_NOTEBOOK_ALLOWLIST = {
    "gpt-oss-(20B)-Fine-tuning.ipynb",
    "Qwen3_(4B)-GRPO.ipynb",
    "Gemma3N_(4B)-Conversational.ipynb",
}

def get_current_git_branch():
    try:
        # Run the git command to get the current branch name
        # '--abbrev-ref HEAD' gives the branch name (e.g., 'main', 'feature/new-feature')
        # 'STDOUT' captures standard output, 'STDERR' redirects error output
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')
        return branch_name
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git branch: {e}")
        print(f"Command output: {e.output.decode('utf-8')}")
        return None
    except FileNotFoundError:
        print("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        return None


def update_or_append_pip_install(base_content, package_name, new_install_line):
    pattern = re.compile(rf"^!(uv )?pip install .*?{package_name}.*$", re.MULTILINE)

    updated_content, substitutions_count = pattern.subn(new_install_line, base_content)

    if substitutions_count == 0:
        return base_content.strip() + "\n" + new_install_line
    else:
        return updated_content

current_branch = get_current_git_branch()
# =======================================================
# GENERAL ANNOUNCEMENTS (THE VERY TOP)
# =======================================================

general_announcement_content = """To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)"""

general_announcement_content_a100 = general_announcement_content.replace("on a **free** Tesla T4 Google Colab instance!", "on your A100 Google Colab Pro instance!")

announcement_separation = '<div class="align-center">'

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
PIN_TRANSFORMERS = "!pip install transformers==4.55.4"
UV_PIN_TRANSFORMERS = PIN_TRANSFORMERS.replace("pip", "uv pip")

PIN_TRL = "!pip install --no-deps trl==0.22.2"
UV_PIN_TRL = PIN_TRL.replace("pip", "uv pip")
SPACES = " " * 4

# =======================================================
# INSTALLATION (MANY OF THIS IS SPECIFIC TO ONE OF THE NOTEBOOKS)
# =======================================================

installation_content = """%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9\\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
"""
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
!pip install --upgrade transformers "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.0"
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

installation_modal_content = installation_content

# =======================================================
# GRPO Notebook
# =======================================================

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
    try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
    except: get_numpy = "numpy"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    get_vllm, get_triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.10.2", "triton")
    !uv pip install -qqq --upgrade \
        unsloth {get_vllm} {get_numpy} torchvision bitsandbytes xformers
    !uv pip install -qqq {get_triton}"""

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
try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
except: get_numpy = "numpy"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
get_vllm, get_triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.10.2", "triton")
!uv pip install -qqq --upgrade \
    unsloth {get_vllm} {get_numpy} torchvision bitsandbytes xformers
!uv pip install -qqq {get_triton}
!uv pip install "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0."""

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

# =======================================================
# Meta Synthetic Data Kit Notebook
# =======================================================

installation_synthetic_data_content = """%%capture
import os
!pip install --upgrade -qqq uv
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install!
    !pip install unsloth vllm synthetic-data-kit==0.0.3
else:
    try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
    except: get_numpy = "numpy"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    get_vllm, get_triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.10.2", "triton")
    !uv pip install -qqq --upgrade \
        unsloth {get_vllm} {get_numpy} torchvision bitsandbytes xformers
    !uv pip install -qqq {get_triton}
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
try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
except: get_numpy = "numpy"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
get_vllm, get_triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.10.2", "triton")
!uv pip install -qqq --upgrade \
    unsloth {get_vllm} {get_numpy} torchvision bitsandbytes xformers
!uv pip install -qqq {get_triton}
!uv pip install "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.0
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

# =======================================================
# Orpheus Notebook
# =======================================================

# Add install snac under install unsloth
installation_orpheus_content = installation_content + """\n!pip install snac"""
installation_orpheus_kaggle_content = installation_kaggle_content + """\n!pip install snac"""

# =======================================================
# Whisper Notebook
# =======================================================

installation_whisper_content = installation_content + """\n!pip install librosa soundfile evaluate jiwer"""
installation_whisper_kaggle_content = installation_kaggle_content + """\n!pip install librosa soundfile evaluate jiwer"""

# =======================================================
# Spark Notebook
# =======================================================

installation_spark_content = installation_content + """\n!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx"""
installation_spark_kaggle_content = installation_kaggle_content + """\n!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx"""

# =======================================================
# GPT OSS Notebook
# =======================================================
installation_gpt_oss_content = r"""%%capture
!pip install --upgrade -qqq uv
import os; os.environ["UV_SYSTEM_PYTHON"]="1"
try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
except: get_numpy = "numpy"
!uv pip install -qqq \
    "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers>=4.55.3" \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
    git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers
!uv pip install --no-deps trl==0.22.2"""

# installation_gpt_oss_content = update_or_append_pip_install(
#     installation_gpt_oss_content,
#     "transformers",
#     "!uv pip install transformers==4.56.2",
# )
installation_gpt_oss_content = update_or_append_pip_install(
    installation_gpt_oss_content,
    "trl",
    UV_PIN_TRL,
)

installation_gpt_oss_kaggle_content = installation_gpt_oss_content

# =======================================================
# Oute Notebook
# =======================================================

installation_oute_content = installation_content + """\n!pip install omegaconf einx
!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS
import os
os.remove("/content/OuteTTS/outetts/models/gguf_model.py")
os.remove("/content/OuteTTS/outetts/interface.py")
os.remove("/content/OuteTTS/outetts/__init__.py")
!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy
!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""

installation_oute_kaggle_content = installation_kaggle_content + """\n!pip install omegaconf einx
!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS
import os
os.remove("/content/OuteTTS/outetts/models/gguf_model.py")
os.remove("/content/OuteTTS/outetts/interface.py")
os.remove("/content/OuteTTS/outetts/__init__.py")
!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy
!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""

# =======================================================
# Llasa Notebook
# =======================================================

# Llasa Need Unsloth==2025.4.1, Transformers==4.48 to running stable, and trl ==0.15.2
# installation_llasa_content = re.sub(r'\bunsloth\b(==[\d\.]*)?', 'unsloth==2025.4.1', installation_content)
installation_llasa_content = installation_content
installation_llasa_content = re.sub(r'\btrl\b(==[\d\.]*)?', 'trl==0.15.2', installation_llasa_content)

installation_llasa_content += """\

!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps
!pip install omegaconf
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""
installation_llasa_content = update_or_append_pip_install(
    installation_llasa_content,
    "transformers",
    "!pip install transformers==4.56.1",
)

installation_llasa_kaggle_content = installation_kaggle_content + """\n!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps
!pip install omegaconf\n%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""
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

# =======================================================
# Tool Calling Notebook
# =======================================================

installation_tool_calling_content = installation_content + """\n!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg"""
installation_tool_calling_kaggle_content = installation_kaggle_content + """\n!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg"""

# =======================================================
# Sesame CSM Notebook
# =======================================================
installation_sesame_csm_content = installation_content
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

installation_sesame_csm_kaggle_content = installation_kaggle_content
installation_sesame_csm_kaggle_content = update_or_append_pip_install(
    installation_sesame_csm_kaggle_content,
    "transformers",
    "!pip install transformers==4.52.3",
)
installation_sesame_csm_kaggle_content = update_or_append_pip_install(
    installation_sesame_csm_kaggle_content,
    "trl",
    PIN_TRL
)

# =======================================================
# Llama Vision Notebook
# =======================================================
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

# =======================================================
# Gemma3N Notebook
# =======================================================
gemma3n_extra_content = """\

import torch; torch._dynamo.config.recompile_limit = 64;
"""
installation_gemma3n_content = installation_content 
installation_gemma3n_content += gemma3n_extra_content

installation_gemma3n_kaggle_content = installation_kaggle_content
installation_gemma3n_kaggle_content += gemma3n_extra_content


# =======================================================
# SGLang Notebook
# =======================================================
installation_sglang_content = """%%capture
import sys
import os
!git clone https://github.com/sgl-project/sglang.git && cd sglang && pip install -e "python[all]"
!pip install -U transformers==4.53.0
sys.path.append(f'{os.getcwd()}/sglang/')
sys.path.append(f'{os.getcwd()}/sglang/python')"""
installation_sglang_kaggle_content = installation_sglang_content

# =======================================================
# NEWS (WILL KEEP CHANGING THIS)
# =======================================================

new_announcement = """
Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker).

[gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) is now supported with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which creates kernels!

Introducing [Vision](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) and [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) for RL! Train Qwen, Gemma etc. VLMs with GSPO - even faster with less VRAM.

Unsloth now supports Text-to-Speech (TTS) models. Read our [guide here](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning).

Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)."""

# =======================================================
# LAST BLOCK CLOSE STATEMENT
# =======================================================

text_for_last_cell_gguf = """Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.

And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>"""

text_for_last_cell_ollama = text_for_last_cell_gguf.replace("Now, ", "You can also ", 1)

text_for_last_cell_gemma3 = text_for_last_cell_gguf.replace("model-unsloth", "gemma-3-finetune")

text_for_last_cell_non_gguf = """And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>"""

hf_course_name = "HuggingFace Course"

ARCHITECTURE_MAPPING = {
    # Gemma Family
    'gemma': 'Gemma',
    'codegemma': 'Gemma', # Explicitly map specific models if needed

    # Llama Family
    'llama': 'Llama',
    'tinylama': 'Llama',

    # Qwen Family
    'qwen': 'Qwen',

    # Phi Family
    'phi': 'Phi',

    # Mistral Family
    'mistral': 'Mistral',
    'pixtral': 'Mistral',
    'zephyr': 'Mistral',

    # Whisper
    'whisper': 'Whisper',

    # Text-to-Speech Models (Group or keep separate?)
    'oute': 'Oute', 
    'llasa': 'Llama',
    'spark': 'Spark',
    'orpheus': 'Orpheus',

    # gpt oss
    'gpt oss': 'GPT-OSS',

    # Linear Attention
    'falcon' : 'Linear Attention',
    'liquid' : 'Linear Attention',

    # Other Models (Assign architecture or keep specific)
    # 'codeforces': 'CodeForces Model', # Example
    # 'unsloth': 'Unsloth Model',     # Example
    'meta synthetic data': 'Llama',
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
    'Tool Calling',          
    'Text Completion',       
    'Synthetic Data',        
    'Reasoning Conversational',
    'Vision GRPO',
    'GRPO LoRA',             
    
    'Conversational',
    'Alpaca',
    'Vision',
    'Reasoning',
    'Completion',
    'Finetune',             
    'Studio',               
    'Coder',                
    'Inference',            
    'Ollama',               
    'Audio',
    
    'ORPO',
    'GRPO',
    'DPO',
    'CPT',
    'TTS',                  
    'LoRA',
    'VL',                   
    'RAFT'
]

FIRST_MAPPING_NAME = {
    "gpt-oss-(20B)-Fine-tuning" : "GPT_OSS_(20B)-Fine-tuning",
    "Qwen2_5_7B_VL_GRPO" : "Qwen2.5_(7B)-VL-GRPO",
}

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

badge_section = '<a href="{link_colab}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'




def copy_folder(source_path, new_name, destination_path=None, replace=False):
    if destination_path is None:
        destination_path = os.path.dirname(source_path)

    new_path = os.path.join(destination_path, new_name)

    try:
        if replace and os.path.exists(new_path):
            shutil.rmtree(new_path)
            print(f"Removed existing folder: '{new_path}'")

        shutil.copytree(source_path, new_path)
        print(f"Successfully copied '{source_path}' to '{new_path}'")
    except FileNotFoundError:
        print(f"Error: Source folder '{source_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def is_path_contains_any(file_path, words):
    return any(re.search(word, file_path, re.IGNORECASE) for word in words)

def extract_version_from_row(row):
    """Extracts the version number from a row string for sorting."""
    match = re.search(r"\| (.*?) \|", row)  # Match content between first "|" and " |"
    if match:
        model_name = match.group(1)
        return extract_version(model_name)
    else:
        return (0, 0)

def extract_version(model_name):
    """Extracts the version number for sorting.

    Handles cases like:
        - Phi 3 Medium
        - Phi 3.5 Mini
        - Phi 4
    Returns a tuple of (major version, minor version) for proper sorting.
    Returns (0, 0) if no version is found.
    """
    match = re.search(r"(\d+(\.\d+)?)", model_name)
    if match:
        version_str = match.group(1)
        if "." in version_str:
            major, minor = version_str.split(".")
            return (int(major), int(minor))
        else:
            return (int(version_str), 0)
    else:
        return (0, 0)


def update_notebook_sections(
    notebook_path,
    general_announcement,
    installation_steps,
    installation_steps_kaggle,
    installation_steps_modal,
    new_announcement,
):
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
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

        if f"{hf_course_name}-" in notebook_path: 
            full_model_name = notebook_path.split("/")[-1].replace(".ipynb", "")
            full_model_name = full_model_name.split("-")
            full_model_name = " ".join(full_model_name[1:]).replace("_", " ")
            general_announcement = general_announcement_content_hf_course.format(full_model_name=full_model_name)
        elif "Meta" in notebook_path:
            general_announcement = general_announcement_content_meta
        elif "A100" in notebook_path:
            general_announcement = general_announcement_content_a100

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
                            "source": [
                                f"{line}\n"
                                for line in general_announcement.splitlines()
                            ],
                        },
                    )
                    updated = True
                    news_markdown_index += 1  # Adjust index since a new cell is added
                else:
                    notebook_content["cells"][first_markdown_index]["source"] = [
                        f"{line}\n" for line in general_announcement.splitlines()
                    ]
                    updated = True
            elif not "".join(
                notebook_content["cells"][first_markdown_index]["source"]
            ).strip():
                # First markdown is empty, replace it
                notebook_content["cells"][first_markdown_index]["source"] = [
                    f"{line}\n" for line in general_announcement.splitlines()
                ]
                updated = True

        i = 0 if news_markdown_index == -1 else news_markdown_index

        is_gguf = False
        is_ollama = False
        is_modal = is_path_contains_any(notebook_path.lower(), ["modal"])
        is_gemma3 = is_path_contains_any(notebook_path.lower(), ["gemma3"])
        is_llama = is_path_contains_any(notebook_path.lower(), ["llama"])
        is_vision = is_path_contains_any(notebook_path.lower(), ["vision"])

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
                        announcement = new_announcement
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in announcement.splitlines()
                        ]
                        updated = True
                        i += 1
                elif source_str == "### Installation":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "code"
                    ):
                        if is_path_contains_any(notebook_path, ["kaggle"]):
                            installation = installation_steps_kaggle
                        elif is_modal and installation_steps_modal is not None:
                            installation = installation_steps_modal
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
                                if is_modal:
                                    del notebook_content["cells"][i + 2]
                                else:
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


                        notebook_content["cells"][i + 1]["source"] = installation
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
            elif is_gemma3 and not is_vision and is_gguf: # Vision cannot be transformed to GGUF yet
                text_for_last_cell = text_for_last_cell_gemma3
            else:
                text_for_last_cell = text_for_last_cell_non_gguf

            if last_cell["cell_type"] == "markdown":
                # Check if the last cell already contains the text
                existing_text = "".join(last_cell["source"])
                if text_for_last_cell not in existing_text:
                  last_cell["source"].extend(
                      [f"{line}\n" for line in text_for_last_cell.splitlines()]
                  )
                  updated = True  # Mark as updated only if content was added
            else:
                notebook_content["cells"].append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f"{line}\n" for line in text_for_last_cell.splitlines()
                        ],
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
            notebook_content["metadata"]["widgets"]["application/vnd.jupyter.widget-state+json"]["state"] = {}
            updated = True

        if updated:
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=1)
            print(f"Updated: {notebook_path}")
        else:
            print(f"No sections found to update in: {notebook_path}")

    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in notebook at {notebook_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {notebook_path}: {e}")


import re

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
    with open(filename, "r", encoding = "utf-8") as f: f = f.read()
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

    with open(filename, "w", encoding = "utf-8") as w: w.write(f)
pass


def main():
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

    for notebook_file in notebook_files:
        update_notebook_sections(
            notebook_file,
            general_announcement_content,
            installation_content,
            installation_kaggle_content,
            installation_modal_content,
            new_announcement,
        )
        # update_unsloth_config(notebook_file)

def add_colab_badge(notebooks_dir):
    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    for path in paths:
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"])
        is_colab = not is_kaggle
        if is_colab:
            with open(path, "r", encoding="utf-8") as f:
                notebook_content = json.load(f)

            badge = badge_section.format(link_colab=(f"https://colab.research.google.com/github/unslothai/notebooks/blob/main/"+path).replace(" ", "%20"))
            notebook_content["cells"].insert(
                0,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"{line}\n"
                        for line in badge.splitlines()
                    ],
                },
            )

            with open(path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=1)


def update_readme(
    args,
    readme_path,
    notebooks_dir,
    architecture_mapping, 
    known_types_ordered,  
    type_order=None,      
    kaggle_accelerator="nvidiaTeslaT4",
):
    if args.to_main_repo:
        base_url_colab = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/"
        base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/"
        base_url_modal = "http://modal.com/notebooks/new/https://github.com/unslothai/notebooks/blob/main/"
    else:
        base_url_colab = f"https://colab.research.google.com/github/unslothai/notebooks/blob/{current_branch}/"
        base_url_kaggle = f"https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/{current_branch}/"
        base_url_modal = f"http://modal.com/notebooks/new/https://github.com/unslothai/notebooks/blob/{current_branch}/"

    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    list_models = ['GRPO'] 
    unique_architectures = sorted(list(set(architecture_mapping.values())))
    for arch in unique_architectures:
        if arch not in list_models:
            list_models.append(arch)
    list_models.append('Other') 

    sections = {}
    for section in list_models:
        sections[section] = {
            "Colab": {"header": f"### {section} Notebooks\n", "rows": []},
            "Kaggle": {"header": f"### {section} Notebooks\n", "rows": []},
            "Modal": {"header": f"### {section} Notebooks\n", "rows": []},
        }

    colab_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"
    kaggle_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"
    modal_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"

    notebook_data = []

    print(f"Processing {len(paths)} notebooks...")
    for path in paths:
        # Ignore HF course and Advanced notebooks
        if is_path_contains_any(path.lower(), [hf_course_name.lower(), "Advanced".lower()]):
            continue

        notebook_name = os.path.basename(path)
        std_notebook_name = notebook_name.replace("-", "_")
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"]) 
        is_modal = is_path_contains_any(path.lower(), ["modal"])

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
        model_type = info['type'] if info and info['type'] else "" 
        architecture = info['architecture'] if info else None
        size = info['size'] 
        size = size.replace(r"_", " ") if size else None 
        size = f"**({size})**" if size else ""

        requires_a100 = info.get('requires_a100', False)

        section_name = "Other" 
        if model_type == 'GRPO':
            section_name = 'GRPO'
        elif architecture and architecture in list_models:
             section_name = architecture
        platform = "Modal" if is_modal else ("Kaggle" if is_kaggle else "Colab")

        if is_modal:
            link_url = f"{base_url_modal}{path}"
            link = f'<a href="{link_url}" target="_blank" rel="noopener noreferrer">Open in Modal</a>'
        else:
            link_base = base_url_kaggle if is_kaggle else base_url_colab
            link_url = f"{link_base}{path}"

            if is_kaggle and kaggle_accelerator:
                link_url += f"&accelerator={kaggle_accelerator}"

            if is_kaggle:
                image_src = "https://kaggle.com/static/images/open-in-kaggle.svg"
                image_alt = "Open in Kaggle"
            else:
                image_src = "https://colab.research.google.com/assets/colab-badge.svg"
                image_alt = "Open In Colab"

            link = f'<a href="{link_url}" target="_blank" rel="noopener noreferrer"><img src="{image_src}" alt="{image_alt}"></a>'

        notebook_data.append(
            {
                "model": model_name,
                "type": model_type,
                "link": link,
                "section": section_name,
                "path": path, 
                "architecture" : architecture, 
                "size" : size, 
                "requires_a100": requires_a100,
                "platform": platform,
            }
        )

    def get_sort_key(x):
        section_index = list_models.index(x["section"])
        version_key = extract_version(x["model"]) 

        type_sort_val = float("inf") 
        current_type = x["type"].strip('*') 
        if type_order and current_type in type_order:
            type_sort_val = type_order.index(current_type)
        elif current_type: 
             type_sort_val = current_type

        return version_key

    notebook_data.sort(key=get_sort_key)

    for data in notebook_data:
        model_prefix = "(A100) " if data.get('requires_a100', False) else ""
        row = f"| **{model_prefix}{data['model']}** {data['size']} | {data['type']} | {data['link']} |\n"
        platform = data.get("platform", "Colab")
        if platform not in sections[data["section"]]:
            platform = "Colab"
        sections[data["section"]][platform]["rows"].append(row)

    for section in sections:
        try:
            sections[section]["Colab"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Colab rows for section '{section}' by version: {e}")
        try:
            sections[section]["Kaggle"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Kaggle rows for section '{section}' by version: {e}")
        try:
            sections[section]["Modal"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Modal rows for section '{section}' by version: {e}")

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
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

        modal_updated_notebooks_links = (
            "# 📒 Modal Notebooks\n"
            "<details>\n  <summary>\n"
            "    Click for all our Modal notebooks categorized by model:\n  "
            "</summary>\n\n"
        )

        for section in list_models:
            if sections[section]["Colab"]["rows"]:
                colab_updated_notebooks_links += sections[section]["Colab"]["header"]
                colab_updated_notebooks_links += colab_table_header
                colab_updated_notebooks_links += "".join(sections[section]["Colab"]["rows"]) + "\n"

            if sections[section]["Kaggle"]["rows"]:
                kaggle_updated_notebooks_links += sections[section]["Kaggle"]["header"]
                kaggle_updated_notebooks_links += kaggle_table_header
                kaggle_updated_notebooks_links += "".join(sections[section]["Kaggle"]["rows"]) + "\n"

            if sections[section]["Modal"]["rows"]:
                modal_updated_notebooks_links += sections[section]["Modal"]["header"]
                modal_updated_notebooks_links += modal_table_header
                modal_updated_notebooks_links += "".join(sections[section]["Modal"]["rows"]) + "\n"

        kaggle_updated_notebooks_links += "</details>\n\n"
        modal_updated_notebooks_links += "</details>\n\n"

        now = datetime.now() 
        timestamp = f"\n"

        updated_readme_content = (
            content_before
            + colab_updated_notebooks_links
            + kaggle_updated_notebooks_links 
            + modal_updated_notebooks_links
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
                + modal_updated_notebooks_links
                + timestamp
                + content_after
             )


        with open(readme_path, "w", encoding="utf-8") as f:
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
    installation_modal,
    new_announcement,
):
    """Copies notebooks from template_dir to destination_dir, updates them, and renames them."""
    template_notebooks = glob(os.path.join(template_dir, "*.ipynb"))

    temp_location = os.path.join(destination_dir, ".temp_backup")
    if os.path.exists(destination_dir):
        if os.path.exists(temp_location):
            shutil.rmtree(temp_location)
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

    for template_notebook_path in template_notebooks:
        notebook_name = os.path.basename(template_notebook_path)

        colab_notebook_name = notebook_name
        colab_destination_path = os.path.join(destination_dir, colab_notebook_name)

        shutil.copy2(template_notebook_path, colab_destination_path)
        print(f"Copied '{colab_notebook_name}' to '{destination_dir}'")

        kaggle_notebook_name = "Kaggle-" + notebook_name
        kaggle_destination_path = os.path.join(destination_dir, kaggle_notebook_name)

        shutil.copy2(template_notebook_path, kaggle_destination_path)
        print(f"Copied '{kaggle_notebook_name}' to '{destination_dir}'")

        modal_notebook_name = "Modal-" + notebook_name
        modal_destination_path = os.path.join(destination_dir, modal_notebook_name)
        should_create_modal = notebook_name in MODAL_NOTEBOOK_ALLOWLIST

        if should_create_modal:
            shutil.copy2(template_notebook_path, modal_destination_path)
            print(f"Copied '{modal_notebook_name}' to '{destination_dir}'")
        else:
            if os.path.exists(modal_destination_path):
                os.remove(modal_destination_path)
                print(f"Removed '{modal_notebook_name}' (not in Modal allowlist)")
            modal_destination_path = None

        if "GRPO" in template_notebook_path:
            hf_course_notebook_name = f"{hf_course_name}-" + notebook_name
            hf_destination_path = os.path.join(destination_dir, hf_course_notebook_name)
            shutil.copy2(template_notebook_path, hf_destination_path)
            print(f"Copied f'{hf_course_name}-{notebook_name}' to '{hf_destination_path}'")
        else:
            hf_destination_path = None

        update_notebook_sections(
            colab_destination_path,
            general_announcement,
            installation,
            installation_kaggle,
            installation_modal,
            new_announcement,
        )

        update_notebook_sections(
            kaggle_destination_path,
            general_announcement,
            installation,
            installation_kaggle,
            installation_modal,
            new_announcement,
        )

        if modal_destination_path:
            update_notebook_sections(
                modal_destination_path,
                general_announcement,
                installation,
                installation_kaggle,
                installation_modal,
                new_announcement,
            )

        if hf_destination_path:
            update_notebook_sections(
                hf_destination_path,
                general_announcement,
                installation,
                installation_kaggle,
                installation_modal,
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
    shutil.rmtree(temp_location)

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
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    (body, resources) = exporter.from_notebook_node(notebook_content)

    body = remove_unwanted_section(body)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)

    print(f"Converted {notebook_path} to {output_path}")

def convert_folder(input_folder: str, output_folder: str):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join(input_folder, filename)
            script_filename = filename.replace('.ipynb', '.py')
            output_path = os.path.join(output_folder, script_filename)
            convert_notebook_to_script(notebook_path, output_path)


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
    args = parser.parse_args()

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
                print(file)
        exit(0)
    copy_and_update_notebooks(
        "original_template",
        "nb",
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        installation_modal_content,
        new_announcement,
    )
    main()

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

    if not args.disable_convert_to_script:
        convert_folder("nb", "python_scripts")
