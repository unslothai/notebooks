#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-Mobile-Actions.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# New 3x faster training & 30% less VRAM. New kernels, padding-free & packing. [Blog](https://unsloth.ai/docs/new/3x-faster-training-packing)
# 
# You can now train with 500K context windows on a single 80GB GPU. [Blog](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://unsloth.ai/docs/new/how-to-train-llms-with-unsloth-and-docker).
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) (faster, less VRAM RL) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/all-our-models) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
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


from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Can choose any sequence length!
fourbit_models = [
    # 4bit Gemma 3 dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Function Gemma models
    "unsloth/functiongemma-270m-it",
    "unsloth/functiongemma-270m-it-unsloth-bnb-4bit",
    "unsloth/functiongemma-270m-it-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/functiongemma-270m-it",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    load_in_16bit = True, # [NEW!] Enables 16bit LoRA
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # use one if using gated models
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
# We now use Google's Mobile Actions dataset, which contains a list of tools a phone could call, like making a calendar invite, setting an alarm etc.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("google/mobile-actions", split = "train")


# Let's take a look at the first row for messages and the tools present.
# 
# We see the person is asking `Please set a reminder for a "Team Sync Meeting" this Friday, June 6th, 2025, at 2 PM.` so we expect the calendar or reminder tool to be used

# In[ ]:


dataset[0]["messages"]


# In[ ]:


dataset[0]["tools"]


# We can then apply FunctionGemma's chat template:

# In[ ]:


tokenizer.apply_chat_template(
    dataset[0]["messages"],
    tools = dataset[0]["tools"],
    tokenize = False,
    add_generation_prompt = False,
)


# We then create a dataset of just these mapped sequences.

# In[ ]:


def process_dataset(row, tokenizer):
    text = tokenizer.apply_chat_template(
        row["messages"],
        tools = row["tools"],
        tokenize = False,
        add_generation_prompt = False,
    )
    return {"text" : text}
dataset = dataset.map(process_dataset, fn_kwargs = {"tokenizer" : tokenizer})


# Look at the first row

# In[ ]:


dataset[0]["text"]


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 100 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[ ]:


# We split the dataset into some training and testing
split_dataset = dataset.train_test_split(test_size = 50, shuffle = True, seed = 3407)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = split_dataset["train"],
    eval_dataset = split_dataset["test"], # For evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 100,
        eval_steps = 2,
        eval_strategy = "steps",
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# We also mask all system and user instructions so our goal is to force the finetune to actually respond with the correct function calls. If not, the finetune might just learn the format, and not learn anything else.

# In[ ]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


# We print the full un-masked string:

# In[ ]:


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# See if we did the masking correctly - "-" is the mask

# In[ ]:


[tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, "-")]


# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

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
# Let's run the model via Unsloth native inference for the first message which should create a calendar reminder

# In[ ]:


dataset[0]["messages"][:2]


# Let's try inference:

# In[ ]:


text = tokenizer.apply_chat_template(
    dataset[0]["messages"][:2],
    tools = dataset[0]["tools"],
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
).removeprefix('<bos>')

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 1024,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
    top_p = 0.95, top_k = 64, temperature = 1.0,
)


# It looks correct!

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("functiongemma_lora")  # Local saving
tokenizer.save_pretrained("functiongemma_lora")
# model.push_to_hub("your_name/functiongemma_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/functiongemma_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "functiongemma_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = False,
    )


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("functiongemma_finetune_16bit", tokenizer, save_method = "merged_16bit")
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/functiongemma_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False:
    model.save_pretrained_merged("functiongemma_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/functiongemma_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("functiongemma_lora")
    tokenizer.save_pretrained("functiongemma_lora")
if False: # Pushing to HF Hub
    model.push_to_hub("HF_USERNAME/functiongemma_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/functiongemma_lora", token = "YOUR_HF_TOKEN")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# In[ ]:


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "functiongemma_finetune",
        tokenizer,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[ ]:


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "HF_ACCOUNT/functiongemma_finetune",
        tokenizer,
        quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
        token = "YOUR_HF_TOKEN",
    )


# Now, use the `functiongemma-finetune.gguf` file or `functiongemma-finetune-Q4_K_M.gguf` file in llama.cpp.
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
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
# 
