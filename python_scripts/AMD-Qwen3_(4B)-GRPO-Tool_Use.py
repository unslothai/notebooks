#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Run*" and press "*Run All*" on **AMD Dev Cloud**!
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
# %%bash
# python -m pip install -qU uv --root-user-action=ignore
# 
# ROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || { [ -r /opt/rocm/.info/version ] && awk -F. '{print "rocm"$1"."$2; exit}' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F': *' '/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f='${Version}\n' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F'[.-]' '{print "rocm"$1"."$2; exit}' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf '%{VERSION}\n' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F'[.-]' '{print "rocm"$1"."$2; exit}' <<<"$ver"; })"
# [ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }
# case "$ROCM_TAG" in
#   rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;
#   rocm6.*) T="rocm6.4" ;;
#   *) T="rocm7.1" ;;
# esac
# pip install bitsandbytes
# PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"
# uv pip install --system -U --force-reinstall \
#     torch torchvision torchaudio triton-rocm \
#     --index-url "$PYTORCH_INDEX_URL"
# uv pip install --system cut-cross-entropy torchao --no-deps
# uv pip install --system -U --no-deps "unsloth[amd]" "unsloth_zoo[amd]"
# uv pip install --system --no-deps -r "$(python -c 'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))')" torchao
# uv pip install --system --no-deps -U "tokenizers>=0.22.0,<=0.23.0"
# 
# 
# # In[ ]:
# 
# 
# import os; os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
# 
# !uv pip install --system -qqq vllm "transformers==4.57.6"
# !uv pip install --system -qqq --no-deps "trl==0.22.2"
# 
# 
# # In[ ]:
# 
# 
# # Placeholder
# 
# # ### Unsloth

# Goal: teach `Qwen3-4B-Instruct-2507` to solve arithmetic problems by **calling calculator tools** with GRPO.
# 
# With `tools = [...]`, TRL's `GRPOTrainer` runs a multi turn agent loop for every rollout:
# 1. The model writes a tool call like `<tool_call>{"name": "multiply", "arguments": {"a": 4821, "b": 97}}</tool_call>`.
# 2. TRL runs your Python function and appends the result as a `tool` message.
# 3. The model continues, calling more tools or giving the final answer.
# 
# The tool results are written by your Python code, not by the model, so Unsloth masks those tokens out of the loss: the model is only trained on the tokens it generated itself. Every turn is generated with Unsloth's fast vLLM inference using the LoRA being trained.

# In[ ]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Covers the prompt plus every tool call and tool result
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vllm fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

# We use the `qwen3-instruct` chat template, which formats tools and tool calls the same way as the official Qwen3-Instruct-2507 template, so TRL can parse the model's tool calls.

# In[ ]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template = "qwen3-instruct")

# ### Tools
# A tool is a plain Python function with **type hints** and a **Google style docstring**. TRL converts each one into a JSON schema that the chat template shows to the model, so the docstring is what the model reads to decide which tool to call.
# 
# To add your own tool (a search engine, a database lookup, a code runner), write another function like these and add it to `tools`. Keep tools deterministic and fast, since they run inside every training step. If a tool raises an error, TRL sends the error message back to the model instead of crashing.

# In[ ]:


def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum a + b.
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """
    Subtract one integer from another.

    Args:
        a: The integer to subtract from.
        b: The integer to subtract.

    Returns:
        The difference a - b.
    """
    return a - b

def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product a * b.
    """
    return a * b

tools = [add, subtract, multiply]

# Let's see how the chat template shows the tools to the model:

# In[ ]:


print(tokenizer.apply_chat_template(
    [{"role" : "user", "content" : "What is 4821 * 97?"}],
    tools = tools,
    add_generation_prompt = True,
    tokenize = False,
))

# <a name="Data"></a>
# ### Data Prep
# We generate word problems with large numbers, which are hard to do reliably in your head but easy with a calculator. Each row has a `prompt` and the true `answer`, which the reward functions receive.

# In[ ]:


import random
from datasets import Dataset

system_prompt = """You are a careful math assistant with calculator tools.
Use the tools for every arithmetic step. Never compute large numbers in your head.
When you are done, reply with the final answer on its own line as: Answer: <number>"""

def make_problem(rng):
    a, b, c = rng.randint(100, 9999), rng.randint(10, 999), rng.randint(1000, 99999)
    kind = rng.randrange(3)
    if kind == 0:
        question = f"A warehouse has {a} boxes with {b} items each. It then receives {c} more items. How many items does it have?"
        answer = a * b + c
    elif kind == 1:
        c = c % (a * b) # Fewer failures than parts made
        question = f"A factory makes {a} parts per day for {b} days, and {c} of the parts fail inspection. How many parts pass?"
        answer = a * b - c
    else:
        question = f"Train A carries {a} passengers and train B carries {c}. Each passenger pays {b} dollars. How many dollars are collected in total?"
        answer = (a + c) * b
    return {
        "prompt" : [
            {"role" : "system", "content" : system_prompt},
            {"role" : "user",   "content" : question},
        ],
        "answer" : str(answer),
    }

rng = random.Random(3407)
dataset = Dataset.from_list([make_problem(rng) for _ in range(2000)])
dataset[0]

# ### Reward functions
# With tools, each completion is a **list of messages**: the assistant's tool calls, the `tool` results, and the final assistant reply. So reward functions can check both the final answer and how the tools were used.
# 
# We give 3 points for the correct final answer, and a small bonus for using the tools without errors.

# In[ ]:


import re
answer_regex = re.compile(r"Answer:\s*\$?\s*(-?[\d,]+)")

def final_reply(completion):
    # The last assistant message with text is the model's final answer
    for message in reversed(completion):
        if message["role"] == "assistant" and message.get("content"):
            return message["content"]
    return ""

global PRINTED_TIMES
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5

def correct_answer(prompts, completions, answer, **kwargs):
    scores = []
    for completion, true_answer in zip(completions, answer):
        match = answer_regex.search(final_reply(completion))
        guess = match.group(1).replace(",", "") if match else None
        if guess is None: scores.append(-1.0)
        elif guess == true_answer: scores.append(3.0)
        else: scores.append(0.0)

    global PRINTED_TIMES
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print("*" * 20, f"Question:\n{prompts[0][-1]['content']}\nAnswer: {answer[0]}")
        for message in completions[0]:
            print(f"[{message['role']}]", message.get("tool_calls") or message.get("content"))
    PRINTED_TIMES += 1
    return scores

def tool_use(completions, **kwargs):
    scores = []
    for completion in completions:
        results = [message for message in completion if message["role"] == "tool"]
        errors = sum('"error"' in str(message.get("content", "")) for message in results)
        score = 0.5 if results else -0.5 # Encourage calling the calculator
        score -= 0.5 * errors            # Penalize malformed tool calls
        scores.append(score)
    return scores

# <a name="Train"></a>
# ### Train the model
# 
# Now set up the GRPO Trainer. The tool specific settings are:
# * `tools = tools` in `GRPOTrainer` turns on the agent loop.
# * `max_tool_calling_iterations` caps how many rounds of tool calls one rollout can make.
# * `max_completion_length` counts every turn of the rollout, including the tool results.

# In[ ]:


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_completion_length = 1024,
    max_tool_calling_iterations = 6, # At most 6 rounds of tool calls per rollout
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 100,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

# And let's run the trainer! The goal is to see the `reward` column increase. The `tools/call_frequency` and `tools/failure_frequency` columns show how often the model calls tools and how often those calls fail.

# In[ ]:


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        correct_answer,
        tool_use,
    ],
    tools = tools,
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# <a name="Inference"></a>
# ### Inference
# Let's try the model we just trained. We write a small agent loop: generate, run any tool calls the model makes, append the results, and generate again until the model answers.

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

# We first save the LoRA, and check it is actually trained:

# In[ ]:


model.save_lora("grpo_saved_lora")

from safetensors import safe_open
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
    # Verify both A and B are non zero
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert(n_zeros.item() != tensor.numel())

# Now we load the LoRA into vLLM and run the agent loop on a new problem:

# In[ ]:


from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.7,
    top_p = 0.8,
    top_k = 20,
    max_tokens = 512,
)
lora_request = model.load_lora("grpo_saved_lora")

def generate(text):
    return model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = lora_request,
    )[0].outputs[0].text

problem = make_problem(random.Random(42))
messages = run_tools(list(problem["prompt"]), tools, generate)
print(messages[-1]["content"])
print("True answer:", problem["answer"])

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
