#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# 
# 
# <a href="https://github.com/meta-llama/synthetic-data-kit"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/meta%20round%20logo.png" width="137"></a>
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it
# 

# ### News

# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss faster with 32% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# You can now train embedding models 1.8-3.3x faster with 20% less VRAM. [Blog](https://unsloth.ai/docs/new/embedding-finetuning)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# 3x faster LLM training with 30% less VRAM and 500K context. [3x faster](https://unsloth.ai/docs/new/3x-faster-training-packing) • [500K Context](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os\n!pip install --upgrade -qqq uv\nif "COLAB_" not in "".join(os.environ.keys()):\n    # If you\'re not in Colab, just use pip install!\n    !pip install unsloth vllm synthetic-data-kit==0.0.3\nelse:\n    try: import numpy, PIL; _numpy = f\'numpy=={numpy.__version__}\'; _pil = f\'pillow=={PIL.__version__}\'\n    except: _numpy = "numpy"; _pil = "pillow"\n    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))\n    except: is_t4 = False\n    _vllm, _triton = (\'vllm==0.9.2\', \'triton==3.2.0\') if is_t4 else (\'vllm==0.10.2\', \'triton\')\n    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth\n    !uv pip install -qqq {_triton}\n    !uv pip install synthetic-data-kit==0.0.3\n!uv pip install transformers==4.56.2\n!uv pip install --no-deps trl==0.22.2\n')
# 
# 
# # In[ ]:
# 
# 
# #@title Colab Extra Install { display-mode: "form" }
# get_ipython().run_line_magic('%capture', '')
# import os
# get_ipython().system('pip install --upgrade -qqq uv')
# if "COLAB_" not in "".join(os.environ.keys()):
#     # If you're not in Colab, just use pip install!
#     get_ipython().system('pip install unsloth vllm')
# else:
#     try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
#     except: _numpy = "numpy"; _pil = "pillow"
#     try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
#     except: is_t4 = False
#     _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.10.2', 'triton')
#     get_ipython().system('uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth')
#     get_ipython().system('uv pip install -qqq {_triton}')
# get_ipython().system('uv pip install transformers==4.56.2')
# get_ipython().system('uv pip install --no-deps trl==0.22.2')
# 
# 
# # ### Unsloth

# ## Primary Goal
# Our goal is to make Llama 3.2 3B understand the "Byte Latent Transformer: Patches Scale Better Than Tokens" [research paper](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/) that was published in December 2024.
# 
# We'll use https://github.com/meta-llama/synthetic-data-kit to generate question and answer pairs **fully locally** which will be used for finetuning Llama 3.2 3B!

# In[4]:


from unsloth.dataprep import SyntheticDataKit

generator = SyntheticDataKit.from_pretrained(
    # Choose any model from https://huggingface.co/unsloth
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048, # Longer sequence lengths will be slower!
)


# ## Generate QA Pairs + Auto clean data
# We now use synthetic data kit for question answer pair generation:

# In[5]:


generator.prepare_qa_generation(
    output_folder = "data", # Output location of synthetic data
    temperature = 0.7, # Higher temp makes more diverse datasets
    top_p = 0.95,
    overlap = 64, # Overlap portion during chunking
    max_generation_tokens = 512, # Can increase for longer QA pairs
)


# Check if it succeeded:

# In[6]:


get_ipython().system('synthetic-data-kit system-check')


# ## Document Parsing (PDF, CSV, HTML etc.)
# Now, let's take the Byte Latent Transformer: Patches Scale Better Than Tokens research paper found at https://arxiv.org/abs/2412.09871 and covert it to Q&A pairs in order to finetune Llama 3.2!

# In[7]:


# Byte Latent Transformer: Patches Scale Better Than Tokens paper in HTML format
get_ipython().system('synthetic-data-kit      -c synthetic_data_kit_config.yaml      ingest "https://arxiv.org/html/2412.09871v1"')

# Truncate document
filenames = generator.chunk_data("data/output/arxiv_org.txt")
print(len(filenames), filenames[:3])


# We see around 37 chunks of data. We now call synthetic-data-kit to create some pairs of data for 3 of our chunks.
# 
# You can process more chunks, but it'll be much slower!
# 
# Using `--num-pairs` will generate **approximately** that many QA pairs. However it might be shorter or longer depending on the `max_seq_length` of the loaded up model. So if you specify 100, you might only get 10 since the model's max sequence length is capped.

# In[8]:


import time
# Process 3 chunks for now -> can increase but slower!
for filename in filenames[:3]:
    get_ipython().system('synthetic-data-kit          -c synthetic_data_kit_config.yaml          create {filename}          --num-pairs 25          --type "qa"')
    time.sleep(2) # Sleep some time to leave some room for processing


# Optionally, you can clean up the data via pruning "bad" or low quality examples and convert the rest to JSON format for finetuning!

# In[ ]:


# !synthetic-data-kit \
#     -c synthetic_data_kit_config.yaml \
#     curate --threshold 0.0 \
#     "data/generated/arxiv_org_0_qa_pairs.json"


# We now convert the generated datasets into QA formats so we can load it for finetuning:

# In[12]:


qa_pairs_filenames = [
    f"data/generated/arxiv_org_{i}_qa_pairs.json"
    for i in range(len(filenames[:3]))
]
for filename in qa_pairs_filenames:
    get_ipython().system('synthetic-data-kit          -c synthetic_data_kit_config.yaml          save-as {filename} -f ft')


# Let's load up the data and see what the synthetic data looks like!

# In[25]:


from datasets import Dataset
import pandas as pd
final_filenames = [
    f"data/final/arxiv_org_{i}_qa_pairs_ft.json"
    for i in range(len(filenames[:3]))
]
conversations = pd.concat([
    pd.read_json(name) for name in final_filenames
]).reset_index(drop = True)

dataset = Dataset.from_pandas(conversations)


# In[27]:


dataset[0]


# In[28]:


dataset[1]


# In[31]:


dataset[-1]


# Finally free vLLM process to save memory and to allow for finetuning!

# In[32]:


generator.cleanup()


# ### Fine-tuning Synthetic Dataset with Unsloth

# In[33]:


from unsloth import FastLanguageModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Qwen3 new models
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # Other very popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[34]:


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
# We now use the `Llama-3.2` format for conversation style finetunes. The chat template renders conversations like below: (Cutting Knowledge Date is by default there!)
# 
# ```
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# 
# Cutting Knowledge Date: December 2023
# Today Date: 01 May 2025
# 
# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
# 
# What is 1+1?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# 
# 2<|eot_id|>
# ```

# In[35]:


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# Get our previous dataset and format it:
dataset = dataset.map(formatting_prompts_func, batched = True,)


# See result of the first row:

# In[36]:


dataset[0]


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[54]:


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# In[38]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[41]:


trainer_stats = trainer.train()


# In[42]:


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
# Let's run the model! Use `apply_chat_template` with `add_generation_prompt` set to `True` for inference.

# In[43]:


messages = [
    {"role": "user", "content": "What is the Byte Latent Transformer?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer,
                   max_new_tokens = 256, temperature = 0.1)


# The model learns about the research paper!

# In[45]:


messages = [
    {"role": "user", "content": "What are some benefits of the BLT?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer,
                   max_new_tokens = 256, temperature = 0.1)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[46]:


model.save_pretrained("meta_synthetic_data_lora")  # Local saving
tokenizer.save_pretrained("meta_synthetic_data_lora")
# model.push_to_hub("your_name/meta_synthetic_data_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/meta_synthetic_data_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[49]:


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta_synthetic_data_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
messages = [
    {"role": "user", "content": "What is so special about BLT's tokenization?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer,
                   max_new_tokens = 256, temperature = 0.1)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("meta_synthetic_data_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: # Change to True to upload finetune
    model.push_to_hub_merged("HF_USERNAME/meta_synthetic_data_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False:
    model.save_pretrained_merged("meta_synthetic_data_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: # Change to True to upload finetune
    model.push_to_hub_merged("HF_USERNAME/meta_synthetic_data_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("meta_synthetic_data_lora")
    tokenizer.save_pretrained("meta_synthetic_data_lora")
if False: # Change to True to upload finetune
    model.push_to_hub("HF_USERNAME/meta_synthetic_data_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/meta_synthetic_data_lora", token = "YOUR_HF_TOKEN")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# In[53]:


# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("meta_synthetic_data_finetune", tokenizer,)
if False: # Change to True to upload finetune
    model.push_to_hub_gguf("HF_USERNAME/meta_synthetic_data_finetune", tokenizer, token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("meta_synthetic_data_finetune", tokenizer, quantization_method = "f16")
if False: # Change to True to upload finetune
    model.push_to_hub_gguf("HF_USERNAME/meta_synthetic_data_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("meta_synthetic_data_finetune", tokenizer, quantization_method = "q4_k_m")
if False: # Change to True to upload finetune
    model.push_to_hub_gguf("HF_USERNAME/meta_synthetic_data_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
# 2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
# 3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
# 4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
# 5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# 
#   <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
# </div>
# 
