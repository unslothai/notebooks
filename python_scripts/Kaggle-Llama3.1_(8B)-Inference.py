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
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.55.4\n!pip uninstall -y unsloth unsloth_zoo\n!pip install "git+https://github.com/Erland366/unsloth.git@feat/distributed_with_lock"\n!pip install "git+https://github.com/Erland366/unsloth-zoo.git@feat/distributed_with_lock"\n!pip install git+https://github.com/Erland366/nbdistributed.git\n!pip install --no-deps trl==0.22.2\n')
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

# If you want to finetune Llama-3 2x faster and use 70% less VRAM, go to our [finetuning notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Alpaca.ipynb)!

# In[ ]:


from unsloth import FastLanguageModel

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    use_exact_model_name = True,
    max_seq_length = 8192,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# In[ ]:


from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


# Change the "value" part to call the model!
# 
# Unsloth makes inference natively 2x faster!! No need to change or do anything!

# In[ ]:


messages = [
                               # EDIT HERE!
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)


# In[ ]:


messages = [
    {"from": "human", "value": "Describe the tallest tower in the world."},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)


# In[ ]:


messages = [
    {"from": "human", "value": "What is Unsloth?"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)


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
