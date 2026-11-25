#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on your A100 Google Colab Pro instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i>
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it

# ### News

# Introducing **[Unsloth Desktop](https://unsloth.ai/docs/desktop)**, the first desktop app to run and train models. Free and open-source for macOS, Windows and Linux. [GitHub](https://github.com/unslothai/unsloth) • [Download](https://unsloth.ai/download)
# 
# <p>
# <a href="https://unsloth.ai/docs/desktop"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/unsloth-qwen3-8.png" width="350" alt="Introducing Unsloth Desktop"></a>
# </p>
# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    !pip install sentencepiece protobuf "datasets>=4.7.0,<5.0.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth\n    !unsloth install-kernels\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==5.24.0\n!pip install --no-deps trl==1.15.0\n!pip install --no-deps --force-reinstall "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo" "unsloth @ git+https://github.com/unslothai/unsloth"\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# ### What you are about to load
# 
# Qwen3.8-Flash-Next is Qwen's preview of the Qwen4 architecture: a 125B parameter MoE (6B active) with 512 experts
# per layer, plus a 51B parameter n-gram embedding table. We load the FP8 upload, 186 GB on disk instead of 360 GB
# in bfloat16, and Unsloth quantizes the experts to 4-bit while loading.
# 
# The n-gram table never enters GPU memory. Unsloth memory maps it from the checkpoint on disk and reads only the
# rows each batch needs, so it costs disk space, not VRAM.
# 
# **One A100 80GB (High-RAM) or one RTX PRO 6000 (96 GB), and about 200 GB of free disk.** On Colab the A100 High-RAM
# runtime has a 375 GB local NVMe at `/mnt/local-scratch`, so the next cell downloads the model there.

# In[ ]:


import os, shutil, torch

# The FP8 checkpoint is 186 GB. Colab's A100 High-RAM has a 375 GB NVMe disk at /mnt/local-scratch: download there.
if os.path.isdir("/mnt/local-scratch") and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/mnt/local-scratch/huggingface"
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(hf_home, exist_ok = True)

# A checkpoint already in the cache (a rerun) counts as disk in hand.
blobs = os.path.join(hf_home, "hub", "models--unsloth--Qwen3.8-Flash-Next-FP8", "blobs")
cached = sum(e.stat().st_size for e in os.scandir(blobs)) if os.path.isdir(blobs) else 0
free_disk = (shutil.disk_usage(hf_home).free + cached) / 1e9
total_vram = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"{torch.cuda.get_device_name(0)}: {total_vram:.1f} GB VRAM. {free_disk:.0f} GB of disk for the model at {hf_home}")
assert total_vram >= 75, "Qwen3.8-Flash-Next in 4-bit needs one 80 GB GPU (A100 80GB or RTX PRO 6000)."
assert free_disk >= 195, "The FP8 checkpoint is 186 GB: pick the A100 High-RAM runtime, which has a 375 GB disk."


# In[ ]:


from unsloth import FastModel

max_seq_length = 2048  # Qwen3.8-Flash-Next goes to 256K tokens

model, tokenizer = FastModel.from_pretrained(
    model_name        = "unsloth/Qwen3.8-Flash-Next-FP8",
    max_seq_length    = max_seq_length,
    load_in_4bit      = True,
    full_finetuning   = False,
)


# Unsloth chose where the n-gram table lives: `disk` or `ram` reads rows from the memory mapped checkpoint (`ram`
# also warms the page cache when there is room), `gpu` only for small tables.

# In[ ]:


from unsloth_zoo.frozen_table_store import iter_frozen_table_modules
for name, table in iter_frozen_table_modules(model):
    print(name, table.frozen_table_store)
print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 2**30:.1f} GB")


# We now add LoRA adapters so we only need to update a small amount of parameters.
# 
# Unsloth maps the usual names onto the attention layers, the 512 routed experts and the n-gram projections of every
# layer: about 1.4B trainable parameters. The QSA indexer gets no gradient from the loss, so it is left out, and the
# n-gram table and the vision tower stay frozen.
# 
# `offload_layers = "auto"` keeps as few decoder layers in host RAM as the training step needs and streams them in
# during each step. On an 80 GB A100 that is a few layers; on a 96 GB RTX PRO 6000 it may be none.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # The routed experts

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 30% less VRAM, fits 2x larger batch sizes
    random_state = 3407,
    offload_layers = "auto", # Keeps a few layers in host RAM only when the GPU is short
)


# <a name="Data"></a>
# ### Data Prep
# 
# We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in
# ShareGPT style.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")


# We now use `standardize_sharegpt` to convert ShareGPT style datasets into HuggingFace's generic format. This changes the dataset from looking like:
# ```
# {"from": "system", "value": "You are an assistant"}
# {"from": "human", "value": "What is 2+2?"}
# {"from": "gpt", "value": "It's 4."}
# ```
# to
# ```
# {"role": "system", "content": "You are an assistant"}
# {"role": "user", "content": "What is 2+2?"}
# {"role": "assistant", "content": "It's 4."}
# ```

# In[ ]:


from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)


# Let's see how row 100 looks like!

# In[ ]:


dataset[100]["conversations"]


# Qwen3.8-Flash-Next ships its own chat template, so we apply the tokenizer's template directly. It renders ChatML:
# `<|im_start|>role\n ... <|im_end|>`, with a `<think>` block opening every assistant turn.

# In[ ]:


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False,
        )
        for convo in convos
    ]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


# And we see how the chat template transformed these conversations.

# In[ ]:


dataset[100]["text"]


# ### Controlling how much the model thinks
# 
# The template writes a reasoning effort into the system block, `xhigh` by default. Pass `reasoning_effort` to
# `apply_chat_template`: `"low"`, `"medium"` or `"xhigh"`. Pass `enable_thinking = False` to skip thinking.
# Reasoning tokens count against `max_new_tokens`.

# In[ ]:


reasoning_effort = "low"  # low, medium, xhigh. The template default is xhigh.

messages = [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
    reasoning_effort = reasoning_effort,
)
print(text)


# <a name="Train"></a>
# ### Train the model
# 
# We do 30 steps to keep this quick; set `num_train_epochs = 1` and `max_steps = None` for a full run. Keep
# `per_device_train_batch_size = 1`: the 248K vocabulary makes the logits the largest allocation in the step.

# In[ ]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = max_seq_length,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# We also use Unsloth's `train_on_responses_only` method to only train on the assistant outputs and ignore the loss
# on the user's inputs. Qwen3.8 turns start with `<|im_start|>user\n` and `<|im_start|>assistant\n`.

# In[ ]:


from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)


# Let's verify masking the instruction part is done. Row 100, then the same row with everything the model is not
# trained on blanked out - you should see only the answer.

# In[ ]:


text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
space = text_tokenizer(" ", add_special_tokens = False).input_ids[0]
print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
print("=" * 30)
print(tokenizer.decode([space if x == -100 else x
                        for x in trainer.train_dataset[100]["labels"]]))


# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[ ]:


trainer_stats = trainer.train()


# In[ ]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# `offload_layers = "auto"` keeps 6 of the 48 layers in host RAM on Colab's RTX PRO 6000 (G4): training peaks at about
# 81 GB reserved and takes about 44 s per step. On an A100 80GB (High-RAM) it keeps 22 layers there, peaks at about
# 78 GB and takes about 140 s per step, mostly copying the swapped experts in. `finetune_mlp_modules = False` trains only
# the attention layers, which needs far less memory.

# <a name="Inference"></a>
# ### Inference
# Let's run the model via Unsloth native inference!

# In[ ]:


messages = [
    {"role": "user", "content": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
    reasoning_effort = "low",
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 256, # Thinking tokens count too
    temperature = 1.0, top_p = 0.95, top_k = 20, # Recommended for thinking mode
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or
# `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down.

# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("HF_ACCOUNT/qwen_lora") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/qwen_lora") # Online saving


# The adapters are small, and they do not contain the n-gram table: loading them needs the base model's checkpoint
# again. Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "qwen_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )


# ### Saving to float16
# 
# A merged 16-bit Qwen3.8-Flash-Next is about 360 GB (the experts in bfloat16 plus the decoded n-gram table), more than
# a Colab disk has next to the 186 GB checkpoint. On a machine with about 600 GB of free disk, set `False` to `True`:

# In[ ]:


if False: # Change to True to save finetune!
    model.save_pretrained_merged("qwen_finetune_16bit", tokenizer, save_method = "merged_16bit")

if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/qwen_finetune_16bit", tokenizer,
        save_method = "merged_16bit", token = "YOUR_HF_TOKEN", private = True,
    )


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i>
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
