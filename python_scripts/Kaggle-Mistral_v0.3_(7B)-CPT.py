#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
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
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.55.4\n!pip install "git+https://github.com/Erland366/unsloth.git@feat/distributed_with_lock"\n!pip install "git+https://github.com/Erland366/unsloth-zoo.git@feat/distributed_with_lock"\n!pip install git+https://github.com/Erland366/nbdistributed.git\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # In[ ]:
# 
# 
# # Let's start our distributed runtime
# get_ipython().run_line_magic('load_ext', 'nbdistributed')
# get_ipython().run_line_magic('dist_init', '-n 2')
# import time
# 
# time.sleep(5)
# get_ipython().run_line_magic('dist_status', '')
# 
# 
# # ### Unsloth

# In[ ]:


get_ipython().run_line_magic('env', 'UNSLOTH_RETURN_LOGITS=1 # Run this to disable CCE since it is not supported for CPT')


# In[ ]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
# 
# We also add `embed_tokens` and `lm_head` to allow the model to learn out of distribution data.

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# <a name="Data"></a>
# ### Data Prep
# We now use the Korean subset of the [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) to first continually pretrain the model. You can use **any language** you like! Go to [Wikipedia's List of Languages](https://en.wikipedia.org/wiki/List_of_Wikipedias) to find your own language!
# 
# **[NOTE]** To train only on completions (ignoring the user's input) read TRL's docs [here](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).
# 
# **[NOTE]** Remember to add the **EOS_TOKEN** to the tokenized output!! Otherwise you'll get infinite generations!
# 
# If you want to use the `llama-3` template for ShareGPT datasets, try our conversational [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb)
# 
# For text completions like novel writing, try this [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb).
# 
# **[NOTE]** Use https://translate.google.com to translate from English to Korean!

# In[ ]:


# Wikipedia provides a title and an article text.
# Use https://translate.google.com!
_wikipedia_prompt = """Wikipedia Article
### Title: {}

### Article:
{}"""
# becomes:
wikipedia_prompt = """위키피디아 기사
### 제목: {}

### 기사:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = examples["title"]
    texts  = examples["text"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = wikipedia_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return { "text" : outputs, }
pass


# We only use 1% of the dataset to speed things up! Use more for longer runs!

# In[ ]:


from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split = "train",)

# We select 1% of the data to make training faster!
dataset = dataset.train_test_split(train_size = 0.01)["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)


# <a name="Train"></a>
# ### Continued Pretraining
# Now let's use Unsloth's `UnslothTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 20 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
# 
# Also set `embedding_learning_rate` to be a learning rate at least 2x or 10x smaller than `learning_rate` to make continual pretraining work!

# In[ ]:


from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        max_steps = 120,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[ ]:


trainer_stats = trainer.train()


# ### Instruction Finetuning
# 
# We now use the [Alpaca in GPT4 Dataset](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-korean) but translated in Korean!
# 
# Go to [vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) for the original GPT4 dataset for Alpaca or [MultilingualSIFT project](https://github.com/FreedomIntelligence/MultilingualSIFT) for other translations of the Alpaca dataset.

# In[ ]:


from datasets import load_dataset

alpaca_dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-korean", split="train")


# We print 1 example:

# In[ ]:


print(alpaca_dataset[0])


# We again use https://translate.google.com/ to translate the Alpaca format into Korean

# In[ ]:


_alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
# Becomes:
alpaca_prompt = """다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

### 지침:
{}

### 응답:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(conversations):
    texts = []
    conversations = conversations["conversations"]
    for convo in conversations:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(convo[0]["value"], convo[1]["value"]) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

alpaca_dataset = alpaca_dataset.map(formatting_prompts_func, batched = True,)


# We again employ `UnslothTrainer` and do instruction finetuning!

# In[ ]:


from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = alpaca_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use num_train_epochs and warmup_ratio for longer runs!
        max_steps = 120,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


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
# Let's run the model! You can change the instruction and input - leave the output blank!
# 
# Remember to use https://translate.google.com/!

# In[ ]:


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,", # instruction
        "피보나치 수열을 계속하세요: 1, 1, 2, 3, 5, 8,", # instruction
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# In[ ]:


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "What is Korean music like?"
        "한국음악은 어떤가요?", # instruction
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# By using https://translate.google.com/ we get
# ```
# Korean music is classified into many types of music genres.
# 
# This genre is classified into different music genres such as pop songs,
# 
# rock songs, classical songs and pop songs, music groups consisting of drums, fans, instruments and singers
# ```

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
    [
        alpaca_prompt.format(
            # "Describe the planet Earth extensively.", # instruction
            "지구를 광범위하게 설명하세요.",
            "",  # output - leave this blank for generation!
        ),
    ],
    return_tensors="pt",
).to("cuda")


from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    **inputs, streamer=text_streamer, max_new_tokens=128, repetition_penalty=0.1
)


# By using https://translate.google.com/ we get
# ```
# Earth refers to all things including natural disasters such as local derailment
# 
# and local depletion that occur in one space along with the suppression of water, gases, and living things.
# 
# Most of the Earth's water comes from oceans, atmospheric water, underground water layers, and rivers and rivers.
# ```
# 
# Yikes the language model is a bit whacky! Change the temperature and using sampling will definitely make the output much better!

# You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.

# In[ ]:


if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# In[ ]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q5_k_m", token = "")


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
