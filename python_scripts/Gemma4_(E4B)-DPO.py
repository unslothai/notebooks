#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth\n    !unsloth install-kernels\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install --no-deps transformers==5.10.1 "tokenizers>=0.22.0,<=0.23.0"\n!pip install "huggingface_hub>=1.5.0,<2.0"\nimport importlib.metadata as _md\ntry: _tv = tuple(int(x) for x in _md.version("torch").split("+")[0].split(".")[:2])\nexcept Exception: _tv = (2, 11)\n_torchcodec = {(2, 11): "torchcodec>=0.11.0", (2, 10): "torchcodec==0.10.*", (2, 9): "torchcodec==0.9.*", (2, 8): "torchcodec==0.7.*", (2, 7): "torchcodec==0.5.*"}.get(_tv, "torchcodec>=0.12.0" if _tv >= (2, 12) else "torchcodec")\n!pip install "{_torchcodec}"\nimport torch; torch._dynamo.config.recompile_limit = 64;\n')
# 
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', '!pip install --no-deps --upgrade timm # For Gemma 4 vision/audio\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# In[ ]:


from unsloth import FastModel
import torch

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E4B-it",
    dtype = None, # None for auto detection
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# We use the `Gemma-4` chat template. Gemma-4 renders conversations like below:
# 
# ```
# <bos><|turn>user
# Hello<turn|>
# <|turn>model
# Hey there!<turn|>
# ```

# In[ ]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4",
)


# Let's see how the model answers a few prompts before DPO, so we can compare after training. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

# In[ ]:


from transformers import TextStreamer
# Helper function for inference
def do_gemma_4_inference(prompt, max_new_tokens = 256):
    messages = [{
        "role": "user",
        "content": [{"type" : "text", "text" : prompt}]
    }]
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

test_prompts = [
    "My code works but my teammate says it's unreadable. How do I make it better?",
    "Explain to a 10 year old why the sky is blue.",
]


# In[ ]:


# You might have to wait 1 minute for Unsloth's auto compiler
for prompt in test_prompts:
    print(f"### {prompt}")
    do_gemma_4_inference(prompt)


# We now add LoRA adapters so we only need to update a small amount of parameters! We only finetune the text layers.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for DPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


# <a name="Data"></a>
# ### Data Prep
# We use the [Ultra Feedback dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), the preference dataset Hugging Face's [Alignment Handbook](https://github.com/huggingface/alignment-handbook) used for [Zephyr](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). Each row has a prompt, a `chosen` response and a `rejected` response. We take the first 1000 rows to speed things up.
# 
# We keep each row in conversational format: `prompt`, `chosen` and `rejected` are lists of messages. `DPOTrainer` then applies the `Gemma-4` chat template itself, and each response ends with the template's own `<turn|>`, so no extra EOS token is added.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split = "train_prefs[:1000]")

def to_preference_format(example):
    return {
        "prompt"   : [{"role" : "user", "content" : example["prompt"]}],
        "chosen"   : [example["chosen"][-1]],   # The final assistant turn
        "rejected" : [example["rejected"][-1]],
    }
dataset = dataset.map(to_preference_format, remove_columns = dataset.column_names)


# Let's see how the first row looks like!

# In[ ]:


import pprint
pprint.pprint(dataset[0])


# <a name="Train"></a>
# ### Train the DPO model
# Now let's train our model with `DPOTrainer` and `GRPOTrainer` for reinforcement learning!. With `ref_model = None`, the reference model is the base model under the same LoRA, so we do not load a second copy. `beta = 0.1` controls how far the model may move away from the reference.
# 
# We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and remove `max_steps` or set it to `-1`.

# In[ ]:


from trl import DPOTrainer, DPOConfig
trainer = DPOTrainer(
    model = model,
    ref_model = None,
    processing_class = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = DPOConfig(
        beta = 0.1,
        max_length = 1024, # Prompt + response, longer rows are truncated
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 5e-5, # Reduce to 5e-6 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# Let's check the tokenized first row. The prompt ends with the `Gemma-4` generation prompt, and each response ends with a single `<turn|>`:

# In[ ]:


row = trainer.train_dataset[0]
print(tokenizer.decode(row["prompt_ids"]))
print("...", tokenizer.decode(row["chosen_ids"][-20:]))
print("...", tokenizer.decode(row["rejected_ids"][-20:]))


# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# # Let's train the model!
# 
# DPO logs `rewards/chosen` and `rewards/rejected` (how much more likely the model makes each response than the reference model did), `rewards/margins` (chosen minus rejected) and `rewards/accuracies` (how often chosen beats rejected). The loss starts at about 0.693 and the margins should grow as training goes on.
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
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the same prompts again after DPO! According to the `Gemma-4` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

# In[ ]:


for prompt in test_prompts:
    print(f"### {prompt}")
    do_gemma_4_inference(prompt)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("gemma_4_lora")  # Local saving
tokenizer.save_pretrained("gemma_4_lora")
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "gemma_4_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-4?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 128, # Increase for longer outputs!
    # Recommended Gemma-4 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-4-finetune`. Set `if False` to `if True` to let it run!

# In[ ]:


if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-4-finetune", tokenizer)


# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[ ]:


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-4-finetune", tokenizer,
        token = "YOUR_HF_TOKEN"
    )


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# In[ ]:


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma_4_finetune",
        tokenizer,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[ ]:


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "HF_ACCOUNT/gemma_4_finetune",
        tokenizer,
        quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
        token = "YOUR_HF_TOKEN",
    )


# Now, use the `gemma-4-finetune.gguf` file or `gemma-4-finetune-Q4_K_M.gguf` file in llama.cpp.
# 
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
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
