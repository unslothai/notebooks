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
# %%capture
# import os
# 
# !pip install pip3-autoremove
# !pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
# !pip install unsloth
# !pip install --no-deps --upgrade "torchao>=0.16.0"
# !pip install transformers==4.57.6
# !pip install --no-deps trl==0.22.2
# !pip install protobuf==3.20.3 # required
# !pip install --no-deps transformers-cfg
# 
# # ### Unsloth

# Goal: teach `Qwen3-4B-Instruct-2507` to call tools by finetuning on **multi turn tool calling conversations**: the user asks something, the assistant calls one or more tools, the tools answer, and the assistant replies using the results.
# 
# This is also a good warm up before reinforcement learning with tools (see our GRPO tool use notebook), since the model then already knows the tool calling format.

# In[ ]:


from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = 4096, # Tool schemas and tool results make conversations long
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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

# <a name="Data"></a>
# ### Data Prep
# We use NousResearch's [Hermes Function Calling dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (Apache 2.0). Each row has a list of tools and a conversation where the assistant calls tools and then answers with their results.
# 
# The dataset writes tool calls as text, so we convert each conversation into the standard `messages` format with `tool_calls` and `tool` messages. The model's own chat template then renders the tools and tool calls in the exact format Qwen3 was trained on.

# We use our `get_chat_template` function to get the `Qwen-3` instruct chat template. It renders tools, tool calls and tool results exactly like the official Qwen3 template.

# In[ ]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen3-instruct",
)

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling", split = "train")
dataset

# In[ ]:


import json, re

tool_call_regex     = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
tool_response_regex = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)

def to_messages(example):
    messages = []
    for turn in example["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "system":
            continue # The chat template writes its own tool instructions
        elif role == "human":
            messages.append({"role" : "user", "content" : text})
        elif role == "gpt":
            calls = [json.loads(call) for call in tool_call_regex.findall(text)]
            message = {"role" : "assistant", "content" : tool_call_regex.sub("", text).strip()}
            if calls:
                message["tool_calls"] = [
                    {"type" : "function", "function" : {"name" : call["name"], "arguments" : call["arguments"]}}
                    for call in calls
                ]
            messages.append(message)
        elif role == "tool":
            for response in tool_response_regex.findall(text):
                messages.append({"role" : "tool", "content" : response})
    return messages, json.loads(example["tools"])

def formatting_prompts_func(examples):
    texts = []
    for conversations, tools in zip(examples["conversations"], examples["tools"]):
        try:
            messages, tools = to_messages({"conversations" : conversations, "tools" : tools})
            text = tokenizer.apply_chat_template(messages, tools = tools, tokenize = False)
        except Exception:
            text = "" # Some rows write tool calls as Python text, not JSON: we drop them below
        texts.append(text)
    return {"text" : texts}

dataset = dataset.map(formatting_prompts_func, batched = True)
dataset = dataset.filter(lambda x: len(x["text"]) != 0)
dataset

# Let's see how the chat template renders a conversation with tool calls:

# In[ ]:


print(dataset[0]["text"])

# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[ ]:


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
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

# We only train on what the **assistant** writes: its tool calls and its replies. The user turns and the tool results are masked out of the loss, since the model should learn to call tools and use their answers, not to predict what the tools return.
# 
# Qwen3 sends tool results back inside a user turn, so Unsloth's `train_on_responses_only` masks them together with the user's messages.

# In[ ]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# Let's verify the masking. Only the assistant's tool calls and replies should be left:

# In[ ]:


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " ")

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
# Let's give the model a tool it has never seen. We write a small agent loop: generate, run any tool calls the model makes, append the results, and generate again until the model answers.

# In[ ]:


import json, re

tool_call_regex = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def run_tools(messages, tools, generate, max_rounds = 5):
    available = {tool.__name__ : tool for tool in tools}
    for _ in range(max_rounds):
        text = tokenizer.apply_chat_template(
            messages,
            tools = tools,
            add_generation_prompt = True,
            tokenize = False,
        )
        reply = generate(text)
        calls = []
        for raw in tool_call_regex.findall(reply):
            try: calls.append(json.loads(raw))
            except json.JSONDecodeError: pass
        content = tool_call_regex.sub("", reply).replace("<|im_end|>", "").strip()
        if not calls:
            messages.append({"role" : "assistant", "content" : content})
            return messages
        messages.append({"role" : "assistant", "content" : content, "tool_calls" : [
            {"type" : "function", "function" : {"name" : call.get("name"), "arguments" : call.get("arguments", {})}}
            for call in calls
        ]})
        for call in calls:
            try: result = available[call["name"]](**call.get("arguments", {}))
            except Exception as error: result = {"error" : str(error)}
            print(f"Tool call: {call.get('name')}({call.get('arguments')}) -> {result}")
            messages.append({"role" : "tool", "name" : call.get("name"), "content" : json.dumps(result)})
    return messages

# In[ ]:


def get_stock_price(ticker: str) -> dict:
    """
    Get the latest stock price for a ticker symbol.

    Args:
        ticker: The stock ticker symbol, for example AAPL.

    Returns:
        The ticker and its latest price in US dollars.
    """
    prices = {"AAPL" : 231.4, "NVDA" : 182.9, "MSFT" : 512.3} # A fake price feed for the demo
    return {"ticker" : ticker, "price" : prices.get(ticker.upper(), 100.0)}

def generate(text):
    inputs = tokenizer(text, return_tensors = "pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,
        temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = False)

messages = run_tools(
    [{"role" : "user", "content" : "How much would 10 shares of NVDA and 5 shares of AAPL cost right now?"}],
    [get_stock_price],
    generate,
)
print(messages[-1]["content"])

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving

# <a name="Save"></a>
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("qwen_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("HF_USERNAME/qwen_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("qwen_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/qwen_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
if False:
    model.push_to_hub("HF_USERNAME/qwen_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/qwen_lora", token = "YOUR_HF_TOKEN")

# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# The chat template saved with the model keeps its tool calling support, so llama.cpp and Ollama can call tools with it too.

# In[ ]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("qwen_finetune", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

# Now, use the `qwen_finetune.Q8_0.gguf` file or `qwen_finetune.Q4_K_M.gguf` file in llama.cpp.
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Teach a model to call tools with SFT first. [Tool calling SFT notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Tool_Calling.ipynb)
# 2. Reinforcement learning with tools. [GRPO tool use notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO-Tool_Use.ipynb)
# 3. Train your own reasoning model - Qwen3 GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
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
