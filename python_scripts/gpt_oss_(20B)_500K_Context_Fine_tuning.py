#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt_oss_(20B)_500K_Context_Fine_tuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
# 

# ### News

# 
# Introducing FP8 precision training for faster RL inference. [Read Blog](https://docs.unsloth.ai/new/fp8-reinforcement-learning).
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker).
# 
# [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) is now supported with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which creates kernels!
# 
# Introducing [Vision](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) and [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) for RL! Train Qwen, Gemma etc. VLMs with GSPO - even faster with less VRAM.
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[1]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):    \n    try: import numpy, PIL; get_numpy = f"numpy=={numpy.__version__}"; get_pil = f"pillow=={PIL.__version__}"\n    except: get_numpy = "numpy"; get_pil = "pillow"\n    !uv pip install -qqq \\\n        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} {get_pil} torchvision bitsandbytes "transformers==4.56.2" \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\\n        git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth\n!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo\n')
# 
# 
# # ### Unsloth

# We're about to demonstrate the power of the new OpenAI GPT-OSS 20B model through a finetuning example. To use our `MXFP4` inference example, use this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_(20B)-Inference.ipynb) instead.

# In[3]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 500_000 # Set long context sequence length here!
dtype = None

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = dtype, # None for auto detection
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    unsloth_tiled_mlp = True, # Super long context >500K finetuning!
    # token = "hf_...", # use one if using gated models
)


# We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

# In[4]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# <a name="Data"></a>
# ### Data Prep

# We'll be using the https://www.gutenberg.org free Ebooks database. We'll get the most popular titles sorted by downloads as listed here: https://www.gutenberg.org/ebooks/search/?sort_order=downloads
# 
# We then will mimic long context training by combining "Moby Dick" and "Prime and Prejudice" together

# In[10]:


import requests
from dataclasses import dataclass, field
from typing import List
@dataclass
class Book:
    url: str
    text: str = ""
    input_ids: List[int] = field(default_factory = list)
    length: int = 0
books = {
    # "Frankenstein" : Book("https://www.gutenberg.org/ebooks/84.txt.utf-8"),
    "Moby Dick" : Book("https://www.gutenberg.org/ebooks/2701.txt.utf-8"),
    "Pride and Prejudice" : Book("https://www.gutenberg.org/ebooks/1342.txt.utf-8"),
    # "The King in Yellow" : Book("https://www.gutenberg.org/ebooks/8492.txt.utf-8"),
    # "Romeo and Juliet" : Book("https://www.gutenberg.org/ebooks/1513.txt.utf-8"),
}
for book_name, book_data in books.items():
    book_data.text = requests.get(book_data.url).text
    book_data.input_ids = tokenizer.encode(book_data.text)
    book_data.length = len(book_data.input_ids)
    print(f"The book '{book_name}' has context length = {book_data.length:,}")


# In[11]:


sum([book.length for book in books.values()])


# We combine the datasets and use gpt-oss's formatting:

# In[20]:


from datasets import Dataset
import itertools
messages = []
for book_name, book_data in books.items():
    messages += [
        {"role" : "user", "content" : f"Print out the contents for the book: '{book_name}'"},
        {"role" : "assistant", "content" : book_data.text},
    ]
dataset = Dataset.from_list([{"text" : tokenizer.apply_chat_template(messages, tokenize = False) + tokenizer.eos_token}])


# In[24]:


len(tokenizer.encode(dataset[0]["text"]))


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 1 step to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[25]:


from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 1,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 1,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# In[26]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# ---
# **NOTE**
# 
# Long context training can take VERY long to even do 1 step. Expect to wait 40 minutes for 500K context lengths or more.
# 
# ---

# In[27]:


trainer_stats = trainer.train()


# In[28]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** Currently finetunes can only be loaded via Unsloth in the meantime - we're working on vLLM and GGUF exporting!

# In[29]:


model.save_pretrained("finetuned_model")
# model.push_to_hub("hf_username/finetuned_model", token = "hf_...") # Save to HF


# ### Saving to float16 for VLLM or mxfp4
# 
# We also support saving to `float16` or `mxfp4` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[30]:


# Merge and push to hub in mxfp4 4bit format
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "mxfp4")
if False: model.push_to_hub_merged("repo_id/repo_name", tokenizer, token = "hf...", save_method = "mxfp4")

# Merge and push to hub in 16bit
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "merged_16bit")
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/gpt-oss-finetune", tokenizer, save_method = "merged_16bit", token = "")


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
