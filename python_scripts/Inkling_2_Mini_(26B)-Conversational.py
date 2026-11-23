#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
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
# Inkling-2-Mini is Thinking Machines' 26.4B parameter MoE (3.4B active): 24 layers, the middle 12 looped three
# times, 128 experts per layer and a vision tower. It needs transformers 5.24.0. Unsloth quantizes the released
# NVFP4 / MXFP8 experts to 4-bit while loading.
# 
# **Two Tesla T4s, not one.** The 4-bit weights are about 14.4 GB, so use Kaggle's GPU T4 x2. Do not pass
# `device_map`: Unsloth's `"sequential"` default fills one card, then the other.

# In[ ]:


MODEL_NAME = "thinkingmachines/Inkling-2-Mini"


# In[ ]:


import torch

# About 14.4 GB of 4-bit weights plus training: two 16 GB cards (Kaggle GPU T4 x2) or one 40 GB card (A100).
total_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30
assert total_gb >= 28, (
    f"Inkling-2-Mini in 4-bit needs about 28 GB of GPU memory in total, this machine has {total_gb:.1f} GB. "
    "On Kaggle pick the T4 x2 accelerator (Settings -> Accelerator -> GPU T4 x2)."
)


# In[ ]:


from unsloth import FastModel

max_seq_length = 1024  # Inkling-2-Mini goes to 1M tokens, but 1024 is what two T4s train on

model, tokenizer = FastModel.from_pretrained(
    model_name        = MODEL_NAME,
    max_seq_length    = max_seq_length,
    load_in_4bit      = True,
    full_finetuning   = False,
)
print(f"placed over {len(set(getattr(model, 'hf_device_map', {'': 0}).values()))} devices")


# On a T4 Unsloth trains in float16 and keeps the routers, short convolutions and loop gates in float32, so do
# not pass `dtype` or set `fp16` / `bf16` on the trainer.

# We now add LoRA adapters so we only need to update a small amount of parameters.
# 
# Unsloth maps the usual names onto Inkling's attention Linears (q, k, v, o and the output gate) and the 128 routed
# experts of every layer: 202,555,392 trainable parameters. The vision tower stays frozen.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # q, k, v, o and the output gate
    finetune_mlp_modules       = True,  # The routed experts

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 30% less VRAM, fits 2x larger batch sizes
    random_state = 3407,
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


# Inkling ships its own chat template, so we apply the tokenizer's template directly. Turns look like
# `<|message_user|><|content_text|> ... <|end_message|>` and the assistant turn opens with `<|message_model|>`.

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
# The template writes a thinking effort into the system block, `0.9` by default. Pass `reasoning_effort` to
# `apply_chat_template`: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`, or 0 to 0.99.
# Reasoning tokens count against `max_new_tokens`.

# In[ ]:


reasoning_effort = "low"  # none, minimal, low, medium, high, xhigh, max, or 0.0 - 0.99. The template default is 0.9.

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
# `per_device_train_batch_size = 1`: the 200k vocabulary makes the logits the largest allocation in the step.

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
        # No fp16 / bf16 here: Unsloth picks the precision for this architecture.
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# We also use Unsloth's `train_on_responses_only` method to only train on the assistant outputs and ignore the loss
# on the user's inputs. Inkling's turns start with `<|message_user|>` and `<|message_model|>`.

# In[ ]:


from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|message_user|>",
    response_part = "<|message_model|>",
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
# Both cards, not just cuda:0 - the model is sharded and the two are not symmetric.
start_gpu_memory = []
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    reserved = torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024
    start_gpu_memory.append(reserved)
    print(f"cuda:{i} = {props.name}. Max memory = "
          f"{props.total_memory / 1024 / 1024 / 1024:.3f} GB. "
          f"{reserved:.3f} GB reserved.")


# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[ ]:


trainer_stats = trainer.train()


# In[ ]:


# @title Show final memory and time stats
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    used = torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024
    total = props.total_memory / 1024 / 1024 / 1024
    print(f"cuda:{i}: peak reserved {used:.3f} GB of {total:.3f} GB "
          f"({used / total * 100:.1f} %), {used - start_gpu_memory[i]:.3f} GB for training.")


# On Kaggle's two T4s this peaks at about 12.1 GB on the first card and 11.9 GB on the second, about 27 s per
# step. If you are close to the limit, lower `max_seq_length` or set `finetune_mlp_modules = False`.

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
    max_new_tokens = 128, # Thinking tokens count too; the first call also compiles
    temperature = 1.0,  # Inkling's generation_config
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or
# `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down.

# In[ ]:


model.save_pretrained("inkling_lora")  # Local saving
tokenizer.save_pretrained("inkling_lora")
# model.push_to_hub("HF_ACCOUNT/inkling_lora") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/inkling_lora") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "inkling_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 1024,
        load_in_4bit = True,
    )

messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
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
    max_new_tokens = 128, # Thinking tokens count too; the first call also compiles
    temperature = 1.0,  # Inkling's generation_config
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# ### Saving to float16
# 
# Set `if False` to `if True` to merge the LoRA adapters into a 16-bit model. A merged Inkling-2-Mini is about
# 53 GB on disk.

# In[ ]:


if False: # Change to True to save finetune!
    model.save_pretrained_merged("inkling_finetune_16bit", tokenizer, save_method = "merged_16bit")

if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/inkling_finetune_16bit", tokenizer,
        save_method = "merged_16bit", token = "YOUR_HF_TOKEN", private = True,
    )


# ### GGUF / llama.cpp Conversion
# 
# `save_pretrained_gguf` converts with Unsloth's prebuilt llama.cpp, which supports Inkling 2. The `q4_k_m` file is
# about 17.5 GB, but the export first writes the 53 GB merge and a 53 GB 16-bit GGUF, so it needs about 150 GB of
# free disk. A Kaggle session runs out of disk; a Colab A100 has the room and takes about 20 minutes.

# In[ ]:


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf("inkling_finetune", tokenizer, quantization_method = "q4_k_m")

if False: # Change to True to upload the GGUF
    model.push_to_hub_gguf(
        "HF_ACCOUNT/inkling_finetune", tokenizer,
        quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN",
    )


# Now run the GGUF with the llama.cpp Unsloth installed. We free the GPU memory first in case your llama.cpp build
# offloads to the GPU.

# In[ ]:


if False: # Change to True to run the GGUF in llama.cpp
    import gc, glob, os, subprocess
    del trainer, model; gc.collect(); torch.cuda.empty_cache()

    gguf_file = sorted(glob.glob("inkling_finetune_gguf/*Q4_K_M*.gguf"))[0]
    llama_cpp = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", os.path.expanduser("~/.unsloth/llama.cpp"))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"}],
        add_generation_prompt = True, tokenize = False, reasoning_effort = "low",
    )
    run = subprocess.run(
        [os.path.join(llama_cpp, "llama-completion"), "-m", gguf_file, "-p", prompt,
         "-n", "256", "-c", "4096", "-ngl", "99", "--temp", "1.0", "-no-cnv"],
        capture_output = True, text = True,
    )
    print(run.stdout)
    print("\n".join(line for line in run.stderr.splitlines() if "eval time" in line))


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
