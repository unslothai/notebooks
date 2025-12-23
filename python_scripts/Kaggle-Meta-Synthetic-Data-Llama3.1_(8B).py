#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# 
# 
# <a href="https://github.com/meta-llama/synthetic-data-kit"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/meta%20round%20logo.png" width="137"></a>
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
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', '!pip install --upgrade -qqq uv\ntry: import numpy, PIL; get_numpy = f"numpy=={numpy.__version__}"; get_pil = f"pillow=={PIL.__version__}"\nexcept: get_numpy = "numpy"; get_pil = "pillow"\ntry: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))\nexcept: is_t4 = False\nget_vllm, get_triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.10.2", "triton")\n!uv pip install -qqq --upgrade     unsloth {get_vllm} {get_numpy} {get_pil} torchvision bitsandbytes xformers\n!uv pip install -qqq {get_triton}\n!uv pip install "huggingface_hub>=0.34.0" "datasets==4.3.0"\n!uv pip install synthetic-data-kit==0.0.3\n!uv pip install transformers==4.56.2\n!uv pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Synthetic-data-kit
# 
# # In[3]:
# 
# 
# # Load and run the model using vllm
# # we prepend "nohup" and postpend "&" to make the Colab cell run in background
# get_ipython().system(' nohup python -m vllm.entrypoints.openai.api_server                    --model unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit                    --trust-remote-code                    --dtype half                    --quantization bitsandbytes                    --max-model-len 10000                    --tensor-parallel-size 1                    --gpu-memory-utilization 0.7                    --enable-chunked-prefill                    --port 8000                    > vllm.log &')
# 
# 
# # In[4]:
# 
# 
# # tail vllm logs. Check server has been started correctly
# get_ipython().system('while ! grep -q "Application startup complete" vllm.log; do tail -n 1 vllm.log; sleep 5; done')
# 
# 
# # Optional: Function to check if vllm server is running. Change False to True and run cell
# 
# # In[6]:
# 
# 
# if False:
#   def is_vllm_server_running(api_base_url=None):
#       """Simply check if VLLM server is running and reachable."""
#       print(api_base_url)
#       try:
#           response = requests.get(f"{api_base_url}/models", timeout=2)
#           return response.status_code == 200
#       except:
#           return False
#   is_running = is_vllm_server_running("http://localhost:8000/v1")
#   if is_running:
#       print(f"VLLM server is running.")
#   else:
#       print(f"VLLM server is not available.")
# 
# 
# # Create data directories
# 
# # In[5]:
# 
# 
# get_ipython().system('mkdir -p data/{pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final}')
# 
# 
# # ### Ingest source file
# # 
# # Ingest source file "https://ai.meta.com/blog/llama-4-multimodal-intelligence/" . Can also use pdf, docx, ppt and youtube video
# 
# # In[6]:
# 
# 
# from synthetic_data_kit.core.ingest import process_file
# import os
# 
# # Set variables directly
# doc_source = "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"
# output_dir = "data/output"
# name = None  # Let the process determine the filename automatically
# config = ctx.config if 'ctx' in locals() else None  # Use ctx if available, otherwise None
# 
# try:
#     # Call process_file directly
#     output_path = process_file(doc_source, output_dir, name, config)
#     print(f"Text successfully extracted to {output_path}")
# except Exception as e:
#     print(f"Error: {e}")
# 
# 
# # ### Generate QA pairs
# # 
# # Generate QA pairs with the help of vllm and Llama-3.1-8B-Instruct-unsloth-bnb-4bit.
# # set num_pairs to the number of required pairs
# 
# # In[9]:
# 
# 
# from synthetic_data_kit.core.create import process_file
# import os
# import requests
# import json
# 
# # Set parameters
# input_file = "data/output/ai_meta_com.txt"
# output_dir = "data/generated"
# config_path = ctx.config_path if 'ctx' in locals() else None  # Use ctx if available
# api_base = "http://localhost:8000/v1"  # Default VLLM API endpoint
# model = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
# content_type = "qa"
# num_pairs = 10
# verbose = False
# 
# # Read the content of the input file
# with open(input_file, 'r') as f:
#     text_content = f.read()
# 
# 
# print("\nGenerating QA pairs...")
# try:
#     # Call process_file directly with all parameters
#     output_path = process_file(
#         input_file,
#         output_dir,
#         config_path,
#         api_base,
#         model,
#         content_type,
#         num_pairs,
#         verbose
#     )
# 
#     if output_path:
#         print(f"Content saved to {output_path}")
# 
#         # Additionally, print the content of the generated file
#         try:
#             with open(output_path, 'r') as f:
#                 output_content = f.read()
#             print("\nGenerated content (first 500 chars):")
#             print(output_content[:500] + "..." if len(output_content) > 500 else output_content)
#         except Exception as e:
#             print(f"Could not read generated file: {e}")
#     else:
#         print("No output was generated")
# except Exception as e:
#     print(f"Error: {e}")
# 
# 
# # ### Curate Data Pairs
# 
# # In[10]:
# 
# 
# from synthetic_data_kit.core.curate import curate_qa_pairs
# 
# # Set all parameters directly
# input_file = "data/generated/ai_meta_com_qa_pairs.json"
# cleaned_dir = "data/cleaned"
# base_name = os.path.splitext(os.path.basename(input_file))[0]
# output = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")
# 
# threshold = None  # Use default threshold
# config_path = ctx.config_path if 'ctx' in locals() else None  # Use ctx if available
# verbose = False
# 
# print("\nCurating generated pairs...")
# 
# try:
#     # Call curate_qa_pairs directly
#     result_path = curate_qa_pairs(
#         input_file,
#         output,
#         threshold,
#         api_base,
#         model,
#         config_path,
#         verbose
#     )
# 
#     print(f"Cleaned content saved to {result_path}")
# 
#     # Display the content of the cleaned file
#     try:
#         with open(result_path, 'r') as f:
#             output_content = f.read()
#         print("\nGenerated content (first 500 chars):")
#         print(output_content[:500] + "..." if len(output_content) > 500 else output_content)
#     except Exception as e:
#         print(f"Could not read cleaned file: {e}")
# except Exception as e:
#     print(f"Error: {e}")
# 
# 
# # ### Save to chatML format
# 
# # In[11]:
# 
# 
# from synthetic_data_kit.core.save_as import convert_format
# import os
# import json
# 
# # Set all parameters directly
# input_file = "data/cleaned/ai_meta_com_qa_pairs_cleaned.json"
# format_type = "ft"  # OpenAI fine-tuning format
# storage_format = "json"  # Default storage format
# 
# # Set up output path
# final_dir = "data/final"
# #os.makedirs(final_dir, exist_ok=True)
# base_name = os.path.splitext(os.path.basename(input_file))[0]
# 
# # Determine output file path
# if storage_format == "hf":
#     output_path = os.path.join(final_dir, f"{base_name}_{format_type}_hf")
# else:
#     if format_type == "jsonl":
#         output_path = os.path.join(final_dir, f"{base_name}.jsonl")
#     else:
#         output_path = os.path.join(final_dir, f"{base_name}_{format_type}.json")
# 
# # Load config if available
# config = ctx.config if 'ctx' in locals() else None
# 
# try:
#     # Call convert_format directly
#     result_path = convert_format(
#         input_file,
#         output_path,
#         format_type,
#         config,
#         storage_format=storage_format
#     )
# 
#     print(f"Converted to {format_type} format and saved to {result_path}")
# 
#     # Display the content of the converted file
#     try:
#         if os.path.isfile(result_path):
#             with open(result_path, 'r') as f:
#                 output_content = f.read()
#             print("\nConverted content (first 500 chars):")
#             print(output_content[:500] + "..." if len(output_content) > 500 else output_content)
#         else:
#             # For HF datasets, it's a directory
#             print(f"\nSaved as HF dataset directory at {result_path}")
#             if os.path.exists(os.path.join(result_path, "dataset_info.json")):
#                 with open(os.path.join(result_path, "dataset_info.json"), 'r') as f:
#                     info = json.load(f)
#                 print(f"Dataset info: {info}")
#     except Exception as e:
#         print(f"Could not read converted file: {e}")
# 
# except Exception as e:
#     print(f"Error: {e}")
# 
# 
# # In[12]:
# 
# 
# # kill vllm server. Takes around 5 seconds.
# print("Attempting to terminate the VLLM server")
# get_ipython().system('pkill -f "vllm.entrypoints.openai.api_server"')
# 
# 
# # ### Unsloth

# In[16]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[17]:


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
# We now use the `ChatML` format for conversation style finetunes. We use [Open Assistant conversations](https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style) in ShareGPT style. ChatML renders multi turn conversations like below:
# 
# ```
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# What's the capital of France?<|im_end|>
# <|im_start|>assistant
# Paris.
# ```
# 
# **[NOTE]** To train only on completions (ignoring the user's input) read TRL's docs [here](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).
# 
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old` and our own optimized `unsloth` template.
# 
# Normally one has to train `<|im_start|>` and `<|im_end|>`. We instead map `<|im_end|>` to be the EOS token, and leave `<|im_start|>` as is. This requires no additional training of additional tokens.
# 
# Note ShareGPT uses `{"from": "human", "value" : "Hi"}` and not `{"role": "user", "content" : "Hi"}`, so we use `mapping` to map it.
# 
# For text completions like novel writing, try this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb).

# In[18]:


from unsloth.chat_templates import get_chat_template

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
#     mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
#     map_eos_token = True, # Maps <|im_end|> to </s> instead
# )

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

from datasets import load_dataset, Dataset
dataset = Dataset.from_json("/content/data/final/ai_meta_com_qa_pairs_cleaned_ft.json")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# In[19]:


dataset[1]["messages"]


# In[20]:


print(dataset[1]["text"])


# If you're looking to make your own chat template, that also is possible! You
# 
# ---
# 
# must use the Jinja templating regime. We provide our own stripped down version of the `Unsloth template` which we find to be more efficient, and leverages ChatML, Zephyr and Alpaca styles.
# 
# More info on chat templates on [our wiki page!](https://github.com/unslothai/unsloth/wiki#chat-templates)

# In[21]:


unsloth_template = \
    "{{ bos_token }}"\
    "{{ 'You are a helpful assistant to the user\n' }}"\
    "{% for message in messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ '>>> User: ' + message['content'] + '\n' }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}"\
        "{{ '>>> Assistant: ' }}"\
    "{% endif %}"
unsloth_eos_token = "eos_token"

if False:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = (unsloth_template, unsloth_eos_token,), # You must provide a template and EOS token
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )


# [link text](https://)<a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!

# In[22]:


from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
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


# In[23]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[24]:


trainer_stats = trainer.train()


# In[25]:


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


# *italicized text*<a name="Inference"></a>
# ### Inference
# Let's run the model! Since we're using `ChatML`, use `apply_chat_template` with `add_generation_prompt` set to `True` for inference.

# In[26]:


from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


# You can also use a TextStreamer for continuous inference - so you can see the generation token by token, instead of waiting the whole time!
# 
# 
# 

# In[27]:


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128, use_cache = True)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[28]:


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# In[29]:


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "What is a famous tall tower in Paris?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128, use_cache = True)


# You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.

# In[30]:


if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoModelForPeftCausalLM
    from transformers import AutoTokenizer

    model = AutoModelForPeftCausalLM.from_pretrained(
        "lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[31]:


# Merge to 16bit
if False: model.save_pretrained_merged("model-merged", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model-merged", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model-merged-4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model-merged-4bit", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
if False:
    model.push_to_hub("hf/lora_model", token = "")
    tokenizer.push_to_hub("hf/lora_model", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# In[32]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")


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
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# </div>
# 
