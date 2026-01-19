#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth your local device, follow [our guide](https://docs.unsloth.ai/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)

# ### News

# 
# New 3x faster training & 30% less VRAM. New kernels, padding-free & packing. [Blog](https://docs.unsloth.ai/new/3x-faster-training-packing)
# 
# You can now train with 500K context windows on a single 80GB GPU. [Blog](https://docs.unsloth.ai/new/500k-context-length-fine-tuning)
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker).
# 
# New in Reinforcement Learning: [FP8 RL](https://docs.unsloth.ai/new/fp8-reinforcement-learning) • [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) (faster, less VRAM RL) • [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.33.post1")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==4.56.2 && pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth

# In[ ]:


# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()


# In[ ]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit", # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# In[ ]:


# @title Alignment Handbook utils
import os
import re
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"] = "sft",
    assistant_prefix = "<|assistant|>\n",
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [
                [msg for msg in example["chosen"] if msg["role"] == "user"][0]
            ]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(
                example["text_chosen"], assistant_prefix
            )
            example["text_rejected"] = _strip_prefix(
                example["text_rejected"], assistant_prefix
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    data_config: dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(
    dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                seed=42
            )
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


# <a name="Data"></a>
# ### Data Prep
# We follow Huggingface's [Alignment Handbook](https://github.com/huggingface/alignment-handbook) for [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) and use the [Ultra Feedback dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), and sample 0.5% of it to speed things up. You can sample the full dataset for a full run.

# In[ ]:


raw_datasets = get_datasets(
    {"HuggingFaceH4/ultrafeedback_binarized" : 0.005}, # 0.5% sampled
    splits = ["train_prefs", "test_prefs"],
)
column_names = list(raw_datasets["train"].features)

raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
    num_proc = 12,
    remove_columns = column_names,
    desc = "Formatting comparisons with prompt template",
)

# Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )


# We shall print a random item from the dataset

# In[ ]:


import pprint

row = raw_datasets["train"][8]
pprint.pprint(row["prompt"])
pprint.pprint(row["chosen"])
pprint.pprint(row["rejected"])


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# <a name="Train"></a>
# ### Train the DPO model
# Now let's train our model. We do 3 epochs on 0.5% of the dataset to speed things up.

# In[ ]:


# One must patch the DPO Trainer first!
from unsloth import PatchDPOTrainer

PatchDPOTrainer()


# In[ ]:


from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
    beta = 0.1,
    train_dataset = raw_datasets["train"],
    # eval_dataset = raw_datasets["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)


# In[ ]:


dpo_trainer.train()


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://docs.unsloth.ai/get-started/install-and-update) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
# 2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide).
# 3. Read our guides and notebooks for [Text-to-speech (TTS)](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning) and [vision](https://docs.unsloth.ai/basics/vision-fine-tuning) model support.
# 4. Explore our [LLM Tutorials Directory](https://docs.unsloth.ai/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
# 5. Need help with Inference? Read our [Inference & Deployment page](https://docs.unsloth.ai/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
# 
