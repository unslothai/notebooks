#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-Multi-Turn-Tool-Calling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
# Long-Context GRPO for reinforcement learning - train stably at massive sequence lengths. Fine-tune models with up to 7x more context length efficiently. [Read Blog](https://unsloth.ai/docs/new/grpo-long-context)
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
# # In[1]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.33.post1")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==4.56.2 && pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth

# In[3]:


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


# # Define tools for FunctionGemma
# 
# We'll define multiple tools which FunctionGemma can call, including the below:
# ```
# get_today_date
# get_current_weather
# add_numbers
# multiply_numbers
# ```

# In[4]:


def get_today_date():
    """
    Gets today's date

    Returns:
        today_date: Today's date in format 18 December 2025
    """
    from datetime import datetime
    today_date = datetime.today().strftime("%d %B %Y")
    return {"today_date": today_date}

def get_current_weather(location: str, unit: str = "celsius"):
    """
    Gets the current weather in a given location.

    Args:
        location: The city and state, e.g. "San Francisco, CA, USA" or "Sydney, Australia"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        temperature: The current temperature in the given location
        weather: The current weather in the given location
    """
    if "San Francisco" in location.title():
        return {"temperature": 15, "weather": "sunny"}
    elif "Sydney" in location.title():
        return {"temperature": 25, "weather": "cloudy"}
    else:
        return {"temperature": 30, "weather": "rainy"}

def add_numbers(x: float | str, y: float | str):
    """
    Adds 2 numbers together

    Args:
        x: First number
        y: Second number

    Returns:
        result: x + y
    """
    return {"result" : float(x) + float(y)}

def multiply_numbers(x: float | str, y: float | str):
    """
    Multiplies 2 numbers together

    Args:
        x: First number
        y: Second number

    Returns:
        result: x * y
    """
    return {"result" : float(x) * float(y)}


# We save all the functions to a mapping and as tools

# In[5]:


FUNCTION_MAPPING = {
    "get_today_date" : get_today_date,
    "get_current_weather" : get_current_weather,
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
}
TOOLS = list(FUNCTION_MAPPING.values())


# We then make some parsing code for FunctionGemma which we hide

# In[6]:


#@title FunctionGemma parsing code (expandible)
import re
def extract_tool_calls(text):
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]

def process_tool_calls(output, messages):
    calls = extract_tool_calls(output)
    if not calls: return messages
    messages.append({
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": call} for call in calls]
    })
    results = [
        {"name": c['name'], "response": FUNCTION_MAPPING[c['name']](**c['arguments'])}
        for c in calls
    ]
    messages.append({ "role": "tool", "content": results })
    return messages

def _do_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages, tools = TOOLS, add_generation_prompt = True, return_dict = True, return_tensors = "pt",
    )
    output = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens = False)

    out = model.generate(**inputs.to(model.device), max_new_tokens = max_new_tokens,
                         top_p = 0.95, top_k = 64, temperature = 1.0,)
    generated_tokens = out[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(generated_tokens, skip_special_tokens = True)

def do_inference(model, messages, print_assistant = True, max_new_tokens = 128):
    output = _do_inference(model, messages, max_new_tokens = max_new_tokens)
    messages = process_tool_calls(output, messages)
    if messages[-1]["role"] == "tool":
        output = _do_inference(model, messages, max_new_tokens = max_new_tokens)
    messages.append({"role": "assistant", "content": output})
    if print_assistant: print(output)
    return messages


# Let's call the model!

# In[7]:


messages = []
messages.append({"role": "user", "content": "What's today's date?"})
messages = do_inference(model, messages, max_new_tokens = 128)


# In[8]:


messages.append({"role" : "user", "content" : "What's the weather like in San Francisco?"})
messages = do_inference(model, messages, max_new_tokens = 128)


# In[9]:


messages.append({"role" : "user", "content" : "What's the weather like in Sydney, Australia?"})
messages = do_inference(model, messages, max_new_tokens = 128)


# In[10]:


messages.append({"role" : "user", "content" : "Add 112358 and 123456"})
messages = do_inference(model, messages, max_new_tokens = 128)


# In[11]:


messages.append({"role" : "user", "content" : "Multiply 112358 and 123456"})
messages = do_inference(model, messages, max_new_tokens = 128)


# In[12]:


messages.append({"role" : "user", "content" : "Do the addition of 2 and 231.111"})
messages = do_inference(model, messages, max_new_tokens = 128)


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
