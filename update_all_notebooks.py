import argparse
import json
import os
import re
import shutil
from datetime import datetime
from glob import glob

badge_section = '<a href="{link_colab}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'

general_announcement_content = """To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)"""

hf_course_separation = '<div class="align-center">'

general_announcement_content_hf_course = general_announcement_content.split(hf_course_separation)
general_announcement_content_hf_course = general_announcement_content_hf_course[0] + hf_course_separation + '<a href="https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt"><img src="https://github.com/unslothai/notebooks/raw/main/assets/hf%20course.png" width="165"></a>' + general_announcement_content_hf_course[1]
general_announcement_content_hf_course = general_announcement_content_hf_course.split("To install Unsloth")
hf_additional_string_announcement = "In this [Hugging Face](https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt) and Unsloth notebook, you will learn to transform {full_model_name} into a Reasoning model using GRPO."
general_announcement_content_hf_course = (
    general_announcement_content_hf_course[0] + 
    hf_additional_string_announcement + 
    "\n\n" +
    "To install Unsloth" + general_announcement_content_hf_course[1]
)

installation_content = """%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"""

installation_kaggle_content = """%%capture

!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth"""

installation_grpo_content = """%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    !pip install --no-deps unsloth vllm"""

installation_extra_grpo_content = r"""#@title Colab Extra Install { display-mode: "form" }
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    !pip install --no-deps unsloth vllm
    # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    # Skip restarting message in Colab
    import sys, re, requests; modules = list(sys.modules.keys())
    for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    
    # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
    f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
    with open("vllm_requirements.txt", "wb") as file:
        file.write(re.sub(rb"(transformers|numpy|xformers)[^\n]{1,}\n", b"", f))
    !pip install -r vllm_requirements.txt"""


installation_grpo_kaggle_content = """%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth vllm
!pip install triton==3.1.0
!pip install -U pynvml"""

installation_gemma3_content = """%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
    !pip install --no-deps unsloth vllm
# Install latest Hugging Face for Gemma-3!
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"""

# Add install snac under install unsloth
installation_orpheus_content = installation_content + """\n!pip install snac"""
installation_orpheus_kaggle_content = installation_kaggle_content + """\n!pip install snac"""

installation_gemma3_kaggle_content = """%%capture
!pip install unsloth vllm
!pip install triton==3.1.0
!pip install -U pynvml
# Install latest Hugging Face for Gemma-3!
!pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"""

new_announcement_content_non_vlm = """**Read our [Gemma 3 blog](https://unsloth.ai/blog/gemma3) for what's new in Unsloth and our [Reasoning blog](https://unsloth.ai/blog/r1-reasoning) on how to train reasoning models.**

Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)."""

new_announcement_content_vlm = """**Read our [Gemma 3 blog](https://unsloth.ai/blog/gemma3) for what's new in Unsloth and our [Reasoning blog](https://unsloth.ai/blog/r1-reasoning) on how to train reasoning models.**

Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks)."""

text_for_last_cell_gguf = """Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

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

text_for_last_cell_non_gguf = """And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/u54VK8m8tk) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

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

naming_mapping = {
    "mistral": ["pixtral", "mistral"],
    "other notebooks": ["TinyLlama", "Advanced"],
    "llama": ["Llama", "Orpheus"],
    "grpo" : ["GRPO"],
}

hf_course_name = "HuggingFace Course"


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
    new_announcement_non_vlm,
    new_announcement_vlm,
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
        is_gemma3 = is_path_contains_any(notebook_path, ["gemma3"])

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
                        if is_path_contains_any(notebook_path, ["Vision"]):
                            announcement = new_announcement_vlm
                        else:
                            announcement = new_announcement_non_vlm
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
                        else:
                            installation = installation_steps

                        # GRPO specific installation
                        if is_path_contains_any(notebook_path.lower(), ["grpo", "gemma3"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_grpo_kaggle_content
                                # Kaggle will delete the second cell instead -> Need to check
                                del notebook_content["cells"][i + 2]
                            else:
                                installation = installation_grpo_content
                                # TODO: Remove after GRPO numpy bug fixed!
                                # Error : ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                                notebook_content["cells"][i + 2]["source"] = installation_extra_grpo_content


                        if is_path_contains_any(notebook_path.lower(), ["gemma3"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_gemma3_kaggle_content
                            else:
                                installation = installation_gemma3_content

                        if is_path_contains_any(notebook_path.lower(), ["orpheus"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_orpheus_kaggle_content
                            else:
                                installation = installation_orpheus_content


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
            elif is_gemma3:
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
    text = text.replace("(", "\(")
    text = text.replace(")", "\)")
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
    f = replace("logging_steps=1,", "", f)
    f = replace("dataset_num_proc=2,", "", f)

    # Fix all spacings x=x to x = x
    spaces = r'(\"[ ]{4,}[^\<\n]{1,}[^ \=\'\"])\=([^ \=\'\"].*?\,\n)'
    f = re.sub(spaces, r"\1 = \2", f)

    with open(filename, "w", encoding = "utf-8") as w: w.write(f)
pass


def main():
    notebook_directory = "nb"
    notebook_pattern = "*.ipynb"

    notebook_files = glob(os.path.join(notebook_directory, notebook_pattern))

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
            new_announcement_content_non_vlm,
            new_announcement_content_vlm,
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
    type_order=None,
    kaggle_accelerator="nvidiaTeslaT4",
):
    if args.to_main_repo:
        base_url_colab = (
            "https://colab.research.google.com/github/unslothai/unsloth/blob/main/nb/"
        )
        base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/unsloth/blob/main/nb/"
    else:
        base_url_colab = (
            "https://colab.research.google.com/github/unslothai/notebooks/blob/main/"
        )
        base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/"

    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    list_models = ["GRPO", "Llama", "Phi", "Mistral", "Qwen", "Gemma", "Other notebooks"]
    sections = {}
    for section in list_models:
        sections[section] = {
            "Colab": {
                "header": f"### {section} Notebooks\n",
                "rows": [],
            },
            "Kaggle": {"header": f"### {section} Notebooks\n", "rows": []},
        }

    colab_table_header = "| Model | Type | Colab Link | \n| --- | --- | --- | \n"
    kaggle_table_header = "| Model | Type | Kaggle Link | \n| --- | --- | --- | \n"

    notebook_data = []

    for path in paths:
        if is_path_contains_any(path.lower(), [hf_course_name.lower()]):
            continue
        notebook_name = os.path.basename(path)
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"])

        section_name = "Other notebooks"  # Default to Other Notebooks

        # Prioritize "Other Notebooks" section
        if is_path_contains_any(path.lower(), naming_mapping["other notebooks"]):
            section_name = "Other notebooks"
        # Prioritize GRPO
        elif is_path_contains_any(path.lower(), naming_mapping["grpo"]):
            section_name = "GRPO"
        else:
            for sect in sections:
                if sect.lower() in ["other notebooks", "grpo"]: # Skip these
                    continue

                check = naming_mapping.get(sect.lower(), [])
                if not check:
                    check = [sect.lower()]

                if is_path_contains_any(path.lower(), check):
                    section_name = sect
                    break

        if is_kaggle:
            link = f"[Open in Kaggle]({base_url_kaggle}{path}"
            # Force to use GPU on start for Kaggle
            if kaggle_accelerator:
                link += f"&accelerator={kaggle_accelerator})"
            else:
                link += ")"
        else:
            link = f"[Open in Colab]({base_url_colab}{path})"
        
        parts = notebook_name.replace(".ipynb", "").split("-")
        
        if is_kaggle:
            model = parts[1].replace("_", " ") if len(parts) > 1 else ""
            type_ = parts[2].replace("_", " ") if len(parts) > 2 else ""
        else:
            model = parts[0].replace("_", " ")
            type_ = parts[1].replace("_", " ") if len(parts) > 1 else ""

        # Add space before version number only if concatenated to the first word
        model_parts = model.split(" ", 1)  # Split into two parts at the first space
        if len(model_parts) > 1:
            model_parts[0] = re.sub(r"([A-Za-z])(\d)", r"\1 \2", model_parts[0])  # Apply regex to the first part only
            model = " ".join(model_parts)

        if is_path_contains_any(path.lower(), ["vision"]):
            type_ = f"**{type_}**"

        notebook_data.append(
            {
                "model": model,
                "type": type_,
                "link": link,
                "section": section_name,
                "path": path,
            }
        )

    if type_order:
        notebook_data.sort(
            key=lambda x: (
                list_models.index(x["section"]),
                type_order.index(x["type"])
                if x["type"] in type_order
                else float("inf"),
            )
        )
    else:
        notebook_data.sort(key=lambda x: (list_models.index(x["section"]), x["type"]))

    for section in sections:
        sections[section]["Colab"]["rows"] = []
        sections[section]["Kaggle"]["rows"] = []

    for data in notebook_data:
        version = extract_version(data["model"])
        data["sort_key"] = (list_models.index(data["section"]), version)

        if is_path_contains_any(data["path"].lower(), ["kaggle"]):
            sections[data["section"]]["Kaggle"]["rows"].append(
                f"| {data['model']} | {data['type']} | {data['link']}\n"
            )
        else:
            sections[data["section"]]["Colab"]["rows"].append(
                f"| {data['model']} | {data['type']} | {data['link']}\n"
            )

    for section in sections:
        sections[section]["Colab"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)
        sections[section]["Kaggle"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        start_marker = "# 📒 Fine-tuning Notebooks"
        start_index = readme_content.find(start_marker)
        if start_index == -1:
            raise ValueError(f"Start marker '{start_marker}' not found in README.")
        start_index += len(start_marker)

        end_marker = "<!-- End of Notebook Links -->"
        end_index = readme_content.find(end_marker)
        if end_index == -1:
            raise ValueError(f"End marker '{end_marker}' not found in README.")

        content_before = readme_content[:start_index]
        content_after = readme_content[end_index:]

        temp = (
            "(https://github.com/unslothai/unsloth/nb/#-kaggle-notebooks).\n\n"
            if args.to_main_repo
            else "(https://github.com/unslothai/notebooks/#-kaggle-notebooks).\n\n"
        )

        colab_updated_notebooks_links = (
            "Below are our notebooks for Google Colab categorized by model.\n"
            "You can also view our [Kaggle notebooks here]"
            f"{temp}"
        )

        kaggle_updated_notebooks_links = (
            "# 📒 Kaggle Notebooks\n"
            "<details>\n  <summary>   \n"
            "Click for all our Kaggle notebooks categorized by model:\n  "
            "</summary>\n\n"
        )

        for section in list_models:
            colab_updated_notebooks_links += (
                sections[section]["Colab"]["header"] + colab_table_header
            )
            colab_updated_notebooks_links += (
                "".join(sections[section]["Colab"]["rows"]) + "\n"
            )

            kaggle_updated_notebooks_links += (
                sections[section]["Kaggle"]["header"] + kaggle_table_header
            )
            kaggle_updated_notebooks_links += (
                "".join(sections[section]["Kaggle"]["rows"]) + "\n"
            )

        kaggle_updated_notebooks_links += "</details>\n\n"

        timestamp = f"<!-- Last updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"

        updated_readme_content = (
            content_before
            + "\n"
            + colab_updated_notebooks_links
            + kaggle_updated_notebooks_links
            + timestamp
            + content_after
        )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_readme_content)

        print(f"Successfully updated {readme_path}")

    except FileNotFoundError:
        print(f"Error: {readme_path} not found.")
    except Exception as e:
        print(f"An error occurred while updating {readme_path}: {e}")


def copy_and_update_notebooks(
    template_dir,
    destination_dir,
    general_announcement,
    installation,
    installation_kaggle,
    new_announcement_non_vlm,
    new_announcement_vlm,
):
    """Copies notebooks from template_dir to destination_dir, updates them, and renames them."""
    template_notebooks = glob(os.path.join(template_dir, "*.ipynb"))

    if os.path.exists(destination_dir):
        shutil.rmtree(destination_dir)
    os.makedirs(destination_dir, exist_ok=True)

    for template_notebook_path in template_notebooks:
        notebook_name = os.path.basename(template_notebook_path)

        colab_notebook_name = notebook_name
        destination_notebook_path = os.path.join(destination_dir, colab_notebook_name)

        shutil.copy2(template_notebook_path, destination_notebook_path)
        print(f"Copied '{colab_notebook_name}' to '{destination_dir}'")

        kaggle_notebook_name = "Kaggle-" + notebook_name
        destination_notebook_path = os.path.join(destination_dir, kaggle_notebook_name)

        shutil.copy2(template_notebook_path, destination_notebook_path)

        print(f"Copied '{kaggle_notebook_name}' to '{destination_dir}'")

        if "GRPO" in template_notebook_path:
            hf_course_notebook_name = f"{hf_course_name}-" + notebook_name
            destination_notebook_path = os.path.join(destination_dir, hf_course_notebook_name)
            shutil.copy2(template_notebook_path, destination_notebook_path)
            print(f"Copied f'{hf_course_name}-{notebook_name}' to '{destination_notebook_path}'")

        update_notebook_sections(
            os.path.join(destination_dir, colab_notebook_name),
            general_announcement,
            installation,
            installation_kaggle,
            new_announcement_non_vlm,
            new_announcement_vlm,
        )

        update_notebook_sections(
            destination_notebook_path,
            general_announcement,
            installation_kaggle,
            installation_kaggle,
            new_announcement_non_vlm,
            new_announcement_vlm,
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to_main_repo",
        action="store_true",
        help="Whether update notebooks and README.md for Unsloth main repository or not. Default is False.",
    )
    args = parser.parse_args()
    copy_and_update_notebooks(
        "original_template",
        "nb",
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        new_announcement_content_non_vlm,
        new_announcement_content_vlm,
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
    update_readme(args, readme_path, notebook_directory, type_order)