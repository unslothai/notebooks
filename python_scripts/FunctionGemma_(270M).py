#!/usr/bin/env python
# coding: utf-8

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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 

# In[2]:


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


# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[3]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128*2,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Formatting is very important in the `functiongemma` for tool-calling.

# For the general conversation,  each role (`developer`, `user`, `model`) is wrapped with `<start_of_turn>{role} ... <end_of_turn>`
# 

# In[4]:


messages_1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"},
]

rendered_1 = tokenizer.apply_chat_template(
    messages_1,
    tools = [], # no tools
    add_generation_prompt = False,
    tokenize = False,
)

print("=== Example 1: Basic turns ===")
print(rendered_1)


# For tool calling, in the `developer` turn, `<start_function_declaration>declaration:get_weather{...}<end_function_declaration>` encodes the full function spec (name, description, parameters) so the model knows *what* tools it can call and how to format arguments.
# 

# In[5]:


tools_2 = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo'.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

messages_2 = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

rendered_2 = tokenizer.apply_chat_template(
    messages_2,
    tools = tools_2,
    add_generation_prompt = False,
    tokenize = False,
)

print("=== Example 2: Tool declarations ===")
print(rendered_2)


# For tool calling + result of the tool call + LLM answer, the user turn is plain text; the next `model` turn embeds `<start_function_call>call:get_weather{...}<end_function_call>` and a matching `<start_function_response>response:get_weather{...}<end_function_response>`, modeling the whole “call tool → receive result → answer user” loop inside the prompt.
# 

# In[6]:


messages_3 = [
    {
        "role": "system",
        "content": "You are a weather assistant.",
    },
    {
        "role": "user",
        "content": "What is the weather in Tokyo?",
    },
    # Assistant issues a tool call
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Tokyo"},
                },
            }
        ],
    },
    # Tool (infrastructure) responds
    {
        "role": "tool",
        "name": "get_weather",
        "tool_call_id": "call_1",
        "content": '{"city": "Tokyo", "temp_c": 25, "condition": "sunny"}',
    },
    # Assistant gives final natural-language answer
    {
        "role": "assistant",
        "content": "It is currently 25°C and sunny in Tokyo.",
    },
]

rendered_3 = tokenizer.apply_chat_template(
    messages_3,
    tools = tools_2,
    add_generation_prompt = False,
    tokenize = False,
)

print("=== Example 3: User → Model → Tool → Model ===")
print(rendered_3)


# Lastly, for our thinking process which we will do in this notebook, since they did not support it in the chat template, we will put it inside the assistant response instead. We will use the tag `<think>...</think>`.  The `<think>...</think>` blocks inside the `model` turn are just tagged text that represent internal reasoning; they appear before and after the tool interaction.

# In[7]:


tools_4 = [
    {
        "type": "function",
        "function": {
            "name": "get_amazon_product_details",
            "description": (
                "Retrieves comprehensive product information from Amazon, "
                "including title, price, description, specifications, and availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asin": {
                        "type": "string",
                        "description": "The Amazon ASIN of the product.",
                    }
                },
                "required": ["asin"],
            },
        },
    }
]

messages_4 = [
    {
        "role": "system",
        "content": (
            "You are a shopping assistant. Use tools when you need detailed "
            "Amazon product data such as price and specifications."
        ),
    },
    {
        "role": "user",
        "content": "Is the espresso machine with ASIN B0XYZ12345 any good for home use?",
    },
    {
        "role": "assistant",
        "content": (
            "<think>"
            "User is asking for an opinion, but I need factual product details first "
            "such as price, features, and reviews. I should call the Amazon product "
            "details tool with the provided ASIN."
            "</think>"
        ),
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_amazon_product_details",
                    "arguments": {
                        "asin": "B0XYZ12345"
                    },
                },
            }
        ],
    },
    {
        "role": "tool",
        "name": "get_amazon_product_details",
        "tool_call_id": "call_1",
        "content": (
            '{"title": "Home Pro Espresso 3000", '
            '"price": 199.99, '
            '"pressure_bar": 15, '
            '"features": ["steam wand", "single and double shot baskets"], '
            '"pros": ["good crema", "compact"], '
            '"cons": ["a bit noisy"]}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<think>"
            "Tool response shows a mid-range price and standard 15 bar pressure. "
            "Features and pros/cons indicate it’s fine for home espresso but not "
            "a high-end machine for enthusiasts."
            "</think>\n"
            "Based on the product details, the Home Pro Espresso 3000 (ASIN B0XYZ12345) "
            "is a solid option for home use. It offers 15-bar pressure, a steam wand, "
            "and both single and double shot baskets, which are enough for everyday "
            "lattes and cappuccinos. It’s compact and produces good crema, but it can "
            "be a bit noisy. If you want a convenient, reasonably priced home machine, "
            "it should work well; if you’re very picky about espresso or plan to upgrade "
            "grinders and accessories, you might eventually want something more advanced."
        ),
    },
]

rendered_prompt = tokenizer.apply_chat_template(
    messages_4,
    tools = tools_4,
    add_generation_prompt = False,  # True if you want to open a fresh model turn for generation
    tokenize = False,
)

print("=== Thinking + Tools ===")
print(rendered_prompt)


# <a name="Data"></a>
# ### Data Prep
# We now use the built-in `functiongemma` format for conversation style finetunes. We use [TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts) dataset. This dataset contains SFT dataset for tool-calling with thinking reasoning, which the `functiongemma` original model did not have this thinking capability.
# 
# Since this dataset contains more than 1 million examples, we will take only the first 50000 using `streaming=True`, bypassing downloading the entire dataset.

# In[8]:


from datasets import load_dataset, Dataset
dataset = load_dataset("LLM360/TxT360-3efforts", name = "agent", split = "medium", streaming = True)
dataset = Dataset.from_list(list(dataset.take(50000)))


# Let's check one example of the dataset.

# In[9]:


dataset[0]["messages"]


# If we look closely, our dataset is in the form of string. Secondly, the format of this dataset is different with the one that is required for `functiongemma`. Especially the one when we need to pass the tools. We will use our defined function to do this :
# 
# 

# In[10]:


#@title Helper Function: prepare_messages_and_tools

import json

THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"

def prepare_messages_and_tools(example):
    raw = json.loads(example["messages"])
    msgs = [dict(m) for m in raw]

    # 1) Extract tools (same as before)
    tools_raw = []
    if msgs and isinstance(msgs[0], dict):
        tlist = msgs[0].get("tools")
        if isinstance(tlist, list) and tlist:
            tools_raw = tlist
            msgs[0].pop("tools", None)

    # 2) Merge assistant["think"] into ["content"]
    THINK_KEYS = ["think", "think_fast", "think_faster"]

    # TRACKER: Check if we successfully added thoughts
    has_valid_thought = False

    for m in msgs:
        if m.get("role") == "assistant":
            # Find the first available thinking key
            found_key = next((k for k in THINK_KEYS if m.get(k)), None)

            if found_key:
                think_text = m[found_key]
                content = m.get("content")
                think_block = f"{THINK_TAG_OPEN}{think_text}{THINK_TAG_CLOSE}"

                if isinstance(content, str) and content:
                    m["content"] = think_block + "\n" + content
                else:
                    m["content"] = think_block

                has_valid_thought = True

                # Clean up keys
                for k in THINK_KEYS:
                    m.pop(k, None)
            else:
                # If an assistant message HAS NO THOUGHT,
                # this example is "poison" for your goal.
                # We mark it as invalid to filter it out later.
                return None, None

    # If the conversation had no assistant turns at all (rare, but possible), skip it
    if not has_valid_thought:
        return None, None
    # 3) Normalize tool_calls to HF-style {type:'function', function:{name, arguments}}
    for m in msgs:
        if "tool_calls" not in m or not m["tool_calls"]:
            continue

        new_tool_calls = []
        for tc in m["tool_calls"]:
            if not isinstance(tc, dict):
                continue

            # Already has function dict?
            if "function" in tc and isinstance(tc["function"], dict):
                new_tool_calls.append(tc)
                continue

            fn_name = tc.get("name", "")
            args = tc.get("arguments", {})

            # Try to parse JSON string arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            new_tool_calls.append(
                {
                    "id": tc.get("id") or tc.get("tool_call_id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn_name,
                        "arguments": args,
                    },
                }
            )

        m["tool_calls"] = new_tool_calls

    # 3b) Build map from tool_call_id -> function name for later tool responses
    id_to_name = {}
    for m in msgs:
        for tc in m.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if tc_id and name:
                id_to_name[tc_id] = name

    # 3c) Ensure tool response messages have a 'name'
    for m in msgs:
        if m.get("role") == "tool":
            if not m.get("name"):
                # Try to infer from tool_call_id using previous map
                tc_id = m.get("tool_call_id")
                inferred = id_to_name.get(tc_id) if tc_id else None
                m["name"] = inferred or "unknown_tool"

    # 4) Normalize tool schemas to HF-style {type:'function', function:{...}}
    adapted_tools = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue

        if "function" in t and isinstance(t["function"], dict):
            adapted_tools.append(t)
            continue

        name = t.get("name", "")
        description = t.get("description", "")
        parameters = t.get("parameters") or {
            "type": "object",
            "properties": {},
        }

        adapted_tools.append(
            {
                "type": t.get("type", "function"),
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    # Delete empty system message
    first_message = msgs[0]
    if first_message["role"] == "system" and "content" not in first_message:
        msgs.pop(0)

    return msgs, adapted_tools


# This will separate the conversation and the tools which we will pass to `tokenizer.apply_chat_template`.
# 
# Let's transform all of our dataset

# In[11]:


def format_example(example):
    messages, tools = prepare_messages_and_tools(example)

    # FILTER: If the preparation returned None, this example was bad.
    if messages is None or len(messages) == 0:
        return {"text": None}

    chat_str = tokenizer.apply_chat_template(
        messages,
        tools = tools,
        add_generation_prompt = False,
        tokenize = False,
    ).removeprefix("<bos>")

    return {
        "text": chat_str,
    }

# Apply the map
train_dataset = dataset.map(format_example)

# Filter out the None values
train_dataset = train_dataset.filter(lambda x: x["text"] is not None)

print(f"Dataset size after filtering: {len(train_dataset)}")


# Let's see the resulting output

# In[12]:


train_dataset[0]["text"]


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 500 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[13]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2, # Use GA to mimic batch size!
        warmup_steps = 10,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 500,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

# In[14]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


# Let's verify masking the instruction part is done! Let's print the 100th row again.

# In[15]:


tokenizer.decode(trainer.train_dataset[-1]["input_ids"])


# Now let's print the masked out example - you should see only the answer is present:

# In[16]:


[tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, "-")]


# In[17]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[18]:


trainer_stats = trainer.train()


# In[19]:


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
# Let's run the model via Unsloth native inference!

# We will take only the first two `messages`, which is the `system` role and the `user` role while also passing the `tools` to the prompt.

# In[20]:


messages, tools = prepare_messages_and_tools(train_dataset[0])

text = tokenizer.apply_chat_template(
    messages[:1],
    tools = tools,
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


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[21]:


model.save_pretrained("functiongemma_lora")  # Local saving
tokenizer.save_pretrained("functiongemma_lora")
# model.push_to_hub("your_name/functiongemma_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/functiongemma_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[22]:


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "functiongemma_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = False,
    )


# ### Saving to float16 for vLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[23]:


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

# In[24]:


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "functiongemma_finetune",
        tokenizer,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[25]:


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
