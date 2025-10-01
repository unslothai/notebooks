#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center"><a href="https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt"><img src="https://github.com/unslothai/notebooks/raw/main/assets/hf%20course.png" width="165"></a>
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# In this [Hugging Face](https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt) and Unsloth notebook, you will learn to transform gpt oss (20B) A100 GRPO into a Reasoning model using GRPO.
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# Unsloth now supports [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which automatically creates kernels!
# 
# [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) is now supported! Train Qwen2.5-VL, Gemma 3 etc. with GSPO or GRPO.
# 
# Introducing Unsloth [Standby for RL](https://docs.unsloth.ai/basics/memory-efficient-rl): GRPO is now faster, uses 30% less memory with 2x longer context.
# 
# Unsloth now supports Text-to-Speech (TTS) models. Read our [guide here](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning).
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', '!pip install --upgrade -qqq uv\ntry: import numpy; get_numpy = f"numpy=={numpy.__version__}"\nexcept: get_numpy = "numpy"\n!uv pip install -qqq \\\n    "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers>=4.55.3" \\\n    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n    "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\\n    git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels\n!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers\n!uv pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth

# We're about to demonstrate the power of the new OpenAI GPT-OSS 120B model through a finetuning example. To use our `MXFP4` inference example, use this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_(20B)-Inference.ipynb) instead.

# In[2]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 4096
dtype = None

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-120b",
    dtype = dtype, # None for auto detection
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


# We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

# In[3]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# ### Reasoning Effort
# The `gpt-oss` models from OpenAI include a feature that allows users to adjust the model's "reasoning effort." This gives you control over the trade-off between the model's performance and its response speed (latency) which by the amount of token the model will use to think.
# 
# ----
# 
# The `gpt-oss` models offer three distinct levels of reasoning effort you can choose from:
# 
# * **Low**: Optimized for tasks that need very fast responses and don't require complex, multi-step reasoning.
# * **Medium**: A balance between performance and speed.
# * **High**: Provides the strongest reasoning performance for tasks that require it, though this results in higher latency.

# In[4]:


from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "low", # **NEW!** Set reasoning effort to low, medium or high
).to("cuda")

_ = model.generate(**inputs, max_new_tokens = 64, streamer = TextStreamer(tokenizer))


# Changing the `reasoning_effort` to `medium` will make the model think longer. We have to increase the `max_new_tokens` to occupy the amount of the generated tokens but it will give better and more correct answer

# In[5]:


from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "medium", # **NEW!** Set reasoning effort to low, medium or high
).to("cuda")

_ = model.generate(**inputs, max_new_tokens = 64, streamer = TextStreamer(tokenizer))


# Lastly we will test it using `reasoning_effort` to `high`

# In[6]:


from transformers import TextStreamer

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "high", # **NEW!** Set reasoning effort to low, medium or high
).to("cuda")

_ = model.generate(**inputs, max_new_tokens = 64, streamer = TextStreamer(tokenizer))


# <a name="Data"></a>
# ### Data Prep

# The `HuggingFaceH4/Multilingual-Thinking` dataset will be utilized as our example. This dataset, available on Hugging Face, contains reasoning chain-of-thought examples derived from user questions that have been translated from English into four other languages. It is also the same dataset referenced in OpenAI's [cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) for fine-tuning. The purpose of using this dataset is to enable the model to learn and develop reasoning capabilities in these four distinct languages.

# In[7]:


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
dataset


# To format our dataset, we will apply our version of the GPT OSS prompt

# In[8]:


from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)


# Let's take a look at the dataset, and check what the 1st example shows

# In[9]:


print(dataset[0]['text'])


# What is unique about GPT-OSS is that it uses OpenAI [Harmony](https://github.com/openai/harmony) format which support conversation structures, reasoning output, and tool calling.

# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[10]:


from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


# In[11]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[12]:


trainer_stats = trainer.train()


# In[13]:


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
# Let's run the model! You can change the instruction and input - leave the output blank!

# In[14]:


messages = [
    {"role": "system", "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems."},
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "medium",
).to("cuda")
from transformers import TextStreamer
_ = model.generate(**inputs, max_new_tokens = 64, streamer = TextStreamer(tokenizer))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** Currently finetunes can only be loaded via Unsloth in the meantime - we're working on vLLM and GGUF exporting!

# In[ ]:


model.save_pretrained("gpt-oss-lora")
tokenizer.save_pretrained("gpt-oss-lora")
# model.push_to_hub("hf_username/gpt-oss-lora", token = "hf_...") # Save to HF
# tokenizer.push_to_hub("hf_username/gpt-oss-lora", token = "hf_...") # Save to HF


# To run the finetuned model, you can do the below after setting `if False` to `if True` in a new instance.

# In[16]:


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gpt-oss-lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )

messages = [
    {"role": "system", "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems."},
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "high",
).to("cuda")
from transformers import TextStreamer
_ = model.generate(**inputs, max_new_tokens = 64, streamer = TextStreamer(tokenizer))


# ### Saving to float16 for VLLM or mxfp4
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16, `merged_4bit` for int4, and `mxfp4` for mxfp4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
# 
# **[NOTE]** Due to the disk limit in Colab, we have to save it using `mxfp4` format (4-bit) or we can use 'push_to_hub_merged' function instead.

# In[ ]:


# Merge to mxfp 4bit
if False:
    model.save_pretrained_merged("gpt-oss-finetune", tokenizer, save_method = "mxfp4",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/gpt-oss-finetune", tokenizer, save_method = "mxfp4", token = "")

# Merge and push to hub in 16bit
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/gpt-oss-finetune", tokenizer, save_method = "merged_16bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("gpt-oss-finetune")
    tokenizer.save_pretrained("gpt-oss-finetune")
if False: # Pushing to HF Hub
    model.push_to_hub("hf/gpt-oss-finetune", token = "")
    tokenizer.push_to_hub("hf/gpt-oss-finetune", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[18]:


# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("gpt-oss-finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("gpt-oss-finetune", tokenizer, quantization_method = "f16")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("hf/gpt-oss-finetune", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("gpt-oss-finetune", tokenizer, quantization_method = "q4_k_m")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("hf/gpt-oss-finetune", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/gpt-oss-finetune", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )


# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.
# 
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
