#!/usr/bin/env python
# coding: utf-8

# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/blob/main/images/Discord button.png?raw=true" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# </div>
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), and how to save it.

# ## Hardware: 2 x 16 GB
# 
# The pinned 4-bit snapshot is 20.83 GiB on disk. On the connected 2 x T4 server,
# this notebook uses two generations and a 1,024-token window.
# 
# `text_only=True` skips the vision/audio towers and loads
# `Qwen3_5ForCausalLM`. The live map placed layers 0-34 on GPU 0 and layers
# 35-63 plus norm/head on GPU 1. Four GRPO steps peaked at 11.94 and 11.58 GiB
# reserved VRAM. No CPU or disk offload was used.
# 
# T4 training uses float32, and `UNSLOTH_COMPILE_DISABLE=1` avoids the
# nested FX/Dynamo RMSNorm failure. The configured 40 GiB path uses four
# generations and 3,072 tokens; it was not run here.

# # Goal: GRPO for competition maths
# 
# We train Qwen3.8 on the numeric-answer subset of
# [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset).
# Rewards check the visible number after `</think>`; no learned judge is used.
# 
# This is text-only, so `text_only=True` omits the vision/audio towers.
# `fast_inference=False` keeps generation on the validated
# Unsloth/Transformers path.

# # Installation
# 
# Qwen3.8 support ships in transformers 5.15.0, so we install that release. Everything else is a normal Unsloth install.

# First we pick up the token. Never paste a token into a cell. Read it from the environment, from
# Kaggle Secrets or from Colab secrets.

# In[14]:


get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\nimport importlib.metadata as metadata\n\n!pip install --upgrade -qqq uv\n\ntry:\n    _stack_ok = all(metadata.version(name).split("+")[0] == version for name, version in {\n        "torch": "2.10.0", "torchvision": "0.25.0", "triton": "3.6.0",\n    }.items())\nexcept metadata.PackageNotFoundError:\n    _stack_ok = False\n\nif not _stack_ok or "COLAB_" in "".join(os.environ):\n    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"\n    except: _numpy = "numpy"; _pil = "pillow"\n    !uv pip install -qqq \\\n        "torch==2.10.0" "torchvision==0.25.0" "triton==3.6.0" {_numpy} {_pil} \\\n        "bitsandbytes==0.50.1" "transformers==5.5.0" \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth"\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq "transformers==5.5.0" \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth"\n\n!uv pip install --upgrade --no-deps -qqq \\\n    "transformers==5.15.1" "tokenizers>=0.22.0,<=0.23.0" "trl==1.9.2" \\\n    "bitsandbytes==0.50.1" "torchao>=0.16.0" \\\n    "unsloth @ git+https://github.com/unslothai/unsloth" \\\n    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo"\n\n!uv pip uninstall -qqq flash-linear-attention fla-core\n!uv pip install --upgrade --no-deps -qqq \\\n    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"\n')


# Qwen3.8 requires Transformers 5.15.1 and the `qwen3_5` architecture.
# After restarting, this cell imports Unsloth first and verifies CUDA 4-bit,
# causal-conv1d, the exact model revision and the absence of external FLA packages.

# In[33]:


import os, importlib.metadata as metadata

os.environ["UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK"] = "128"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

_bnb_version = metadata.version("bitsandbytes")

import unsloth
from unsloth.device_type import ALLOW_BITSANDBYTES
assert ALLOW_BITSANDBYTES, f"Unsloth disabled bitsandbytes {_bnb_version}"

from transformers.utils import is_causal_conv1d_available

assert metadata.version("transformers") == "5.15.1"
assert is_causal_conv1d_available(), "causal-conv1d is unavailable"
print(f"bitsandbytes {_bnb_version} | transformers 5.15.1 | causal-conv1d available")


# ### Which repo to load
# 
# These public repositories provide the same Qwen3.8-27B base model at different
# precisions:
# 
# | repo | precision | published size | use it when |
# |---|---|---:|---|
# | [`unsloth/Qwen3.8-27B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Qwen3.8-27B-unsloth-bnb-4bit) | pre-quantised bitsandbytes NF4 for eligible modules | 22.36 GB (20.83 GiB) | QLoRA fine-tuning; this is what the cells below load. |
# | [`unsloth/Qwen3.8-27B`](https://huggingface.co/unsloth/Qwen3.8-27B) | float16/bfloat16-class weights | about 55.6 GB | inference or training on larger-memory hardware. |
# 
# Ready-made GGUFs for llama.cpp are at
# [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

# ### Load Qwen3.8
# 
# Qwen3.8 is multimodal, but this maths task only needs the language model.
# `text_only=True` leaves out the vision and audio towers and avoids the hybrid
# attention-mask error the full VLM path hits during batched GRPO.
# 
# The loader gets the 4-bit setup and GPU placement from the checkpoint. The cell
# below only supplies the exact repository, revision, sequence length and
# text-only switch.

# The checkpoint is pinned to
# `8aa5f05d26b7205477066e1449e0af13f762a299`.
# `use_exact_model_name=True` ensures that revision is applied to this exact
# 4-bit repository.

# In[ ]:


from unsloth import FastModel
import torch

MODEL_NAME = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_REVISION = "8aa5f05d26b7205477066e1449e0af13f762a299"
lora_rank = 8

assert torch.cuda.is_available(), "CUDA is required."
_n = torch.cuda.device_count()
_gib = min(torch.cuda.get_device_properties(i).total_memory for i in range(_n)) / 2**30
_big = _gib >= 39
if not _big and (_n < 2 or _gib < 14):
    raise RuntimeError("Use one >=40 GiB GPU or two >=14 GiB GPUs.")

NUM_GENERATIONS = 4 if _big else 2
BATCH_SIZE = NUM_GENERATIONS
max_seq_length = 3072 if _big else 1024
max_prompt_length = 384
max_completion_length = max_seq_length - max_prompt_length
print(
    f"{_n} GPU(s), smallest {_gib:.1f} GiB -> "
    f"num_generations={NUM_GENERATIONS}, max_seq_length={max_seq_length}"
)

model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    revision = MODEL_REVISION,
    use_exact_model_name = True,
    max_seq_length = max_seq_length,
    text_only = True,
)
model.config.architectures = model.config.architectures or ["Qwen3_5ForCausalLM"]
model.generation_config.max_length = None

print(model.config.model_type, type(model).__name__)


# Qwen3.8 already exposes the correct chat stop token. The tokenizer's
# `eos_token` is `<|im_end|>`, and the published generation config lists both
# `<|im_end|>` and `<|endoftext|>` as valid stop IDs. Unlike the source
# notebook, no tokenizer mutation is needed. The next cell verifies this contract
# so a changed or mismatched snapshot fails early.

# In[4]:


# FastModel returns a Qwen3VLProcessor; its text tokenizer is one level down.
_text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
print("tokenizer eos:", _text_tokenizer.eos_token, _text_tokenizer.eos_token_id)
assert _text_tokenizer.eos_token == "<|im_end|>"

_eos_ids = model.generation_config.eos_token_id
if isinstance(_eos_ids, int):
    _eos_ids = [_eos_ids]
print("generation_config.eos_token_id:", _eos_ids)
assert _text_tokenizer.eos_token_id in _eos_ids


# Add LoRA to the decoder attention and MLP layers. The text-only loader has
# already omitted the vision/audio towers.

# In[5]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r            = lora_rank,
    lora_alpha   = lora_rank * 2,
    lora_dropout = 0,
    bias         = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)


# <a name="Data"></a>
# ### The Qwen3.8 chat format
# 
# Qwen3.8 uses a ChatML-style turn structure:
# `<|im_start|>ROLE\n...<|im_end|>`. With thinking enabled, the generation prompt
# opens an assistant turn and then `<think>`. The model writes private reasoning,
# closes it with `</think>`, writes the visible answer and finally emits
# `<|im_end|>`.
# 
# `<|im_start|>` and `<|im_end|>` are registered special tokens. `<think>` and
# `</think>` are tokenizer entries but are not marked special, so the closing
# `</think>` remains available to the reward functions after TRL decodes with
# `skip_special_tokens=True`.
# 
# Rather than duplicate the whole chat template, the next cell renders a sentinel
# conversation and derives the exact boundary strings from the pinned tokenizer.

# In[6]:


SENTINEL_REASONING = "SENTINELREASONING"
SENTINEL_ANSWER = "SENTINELANSWER"

_probe = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "probe"},
        {
            "role": "assistant",
            "reasoning_content": SENTINEL_REASONING,
            "content": SENTINEL_ANSWER,
        },
    ],
    tokenize = False,
    reasoning_effort = "low",
)
_before_reasoning, _after_reasoning = _probe.split(SENTINEL_REASONING)
REASONING_TO_ANSWER, ANSWER_SUFFIX = _after_reasoning.split(SENTINEL_ANSWER)

_generation_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "probe"}],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = "low",
)

print("generation prompt ends with:", repr(_generation_prompt[-40:]))
print("REASONING_TO_ANSWER =", repr(REASONING_TO_ANSWER))
print("ANSWER_SUFFIX =", repr(ANSWER_SUFFIX))
assert _generation_prompt.endswith("<think>\n")
assert "</think>" in REASONING_TO_ANSWER


# A generated completion begins immediately after the prompt's opening
# `<think>\n`. A complete example therefore looks like this:
# 
# ```text
# 48 / 2 = 24, and 48 + 24 = 72.
# </think>
# 
# 72<|im_end|>
# ```
# 
# TRL normally decodes completions with `skip_special_tokens=True`.
# `<|im_end|>` is removed, but `</think>` remains because it is not marked as a
# special token:
# 
# ```text
# 48 / 2 = 24, and 48 + 24 = 72.
# </think>
# 
# 72
# ```
# 
# The helper below accepts either form and treats text before the last
# `</think>` as reasoning and text after it as the user-visible answer.

# In[7]:


THINK_END = "</think>"

def split_channels(text):
    """Return (reasoning, answer); the visible answer follows the last </think>."""
    if text is None:
        return "", ""

    idx = text.rfind(THINK_END)
    if idx == -1:
        # The model never closed its reasoning, so it never produced a final answer.
        return text.replace("<think>", "").strip(), ""

    reasoning = text[:idx]
    answer = text[idx + len(THINK_END):]

    for marker in ("<|im_start|>", "assistant", "<think>"):
        reasoning = reasoning.replace(marker, " ")
    for marker in ("<|im_end|>", "<|endoftext|>"):
        answer = answer.split(marker)[0]

    return reasoning.strip(), answer.strip()

# Sanity-check decoding with the end token kept and removed.
_with = "48/2 = 24, 48+24 = 72." + REASONING_TO_ANSWER + "72" + ANSWER_SUFFIX
_ids = tokenizer(text = _with, add_special_tokens = False)["input_ids"]
if len(_ids) and isinstance(_ids[0], (list, tuple)):
    _ids = _ids[0]
_without = _text_tokenizer.decode(_ids, skip_special_tokens = True)

print("markers kept   ->", split_channels(_with))
print("markers dropped ->", split_channels(_without))
assert split_channels(_with)[1] == "72"
assert split_channels(_without)[1] == "72"


# ### Data prep
# 
# Keep only DeepScaleR examples with a bare numeric answer, render them with
# `reasoning_effort='low'`, and remove prompts above 384 tokens.

# In[8]:


import re
from datasets import load_dataset

REASONING_EFFORT = "low"  # Supported values: "low", "medium", "xhigh".

SYSTEM_PROMPT = (
    "You are solving competition mathematics problems.\n"
    "Think through the problem step by step, then after </think> give the final answer as a "
    "bare number and nothing else. No units, no words, and no working in the final answer."
)

IS_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")

raw = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split = "train")
raw = raw.filter(lambda x: IS_NUMERIC.match(str(x["answer"]).strip().replace(",", "")) is not None)
print("problems with a plain numeric answer:", len(raw))

dataset = raw.map(lambda x: {
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": x["problem"]},
    ],
    "answer": str(x["answer"]).strip().replace(",", ""),
}, remove_columns = raw.column_names)

print(dataset[0]["prompt"][1]["content"][:400])
print("gold:", dataset[0]["answer"])


# The first 1,000 rendered prompts measured 145 tokens at the median, 194 at
# the 90th percentile and 760 maximum. Filtering the full split at 384 tokens
# kept 24,467 examples. Completion budgets are 640 tokens on 2 x T4 and 2,688
# tokens on the configured 40 GiB path.

# In[9]:


import numpy as np

def prompt_length(messages):
    """Token count of a rendered Qwen3.8 prompt."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
        reasoning_effort = REASONING_EFFORT,
    )
    ids = tokenizer(text = text, add_special_tokens = False)["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)

lengths = [prompt_length(p) for p in dataset["prompt"][:1000]]
print(
    "prompt tokens: median", int(np.median(lengths)),
    " 90th pct", int(np.percentile(lengths, 90)),
    " max", max(lengths),
)

max_prompt_length = 384
max_completion_length = max_seq_length - max_prompt_length
print("max_prompt_length", max_prompt_length, "max_completion_length", max_completion_length)

dataset = dataset.filter(lambda x: prompt_length(x["prompt"]) <= max_prompt_length)
print("kept", len(dataset), "examples")


# <a name="Inference"></a>
# ### Before training
# 
# This 96-token coherence smoke sample may stop inside `<think>`; it is not
# an accuracy benchmark.

# In[15]:


from transformers import TextStreamer

DEMO_MAX_NEW_TOKENS = min(96, max_completion_length)

text = tokenizer.apply_chat_template(
    dataset[0]["prompt"],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = REASONING_EFFORT,
)
inputs = tokenizer(text = text, return_tensors = "pt", add_special_tokens = False).to("cuda")

_ = model.generate(
    **inputs,
    max_new_tokens = DEMO_MAX_NEW_TOKENS,
    temperature = 1.0,
    do_sample = True,
    streamer = TextStreamer(
        _text_tokenizer,
        skip_prompt = True,
        skip_special_tokens = False,
    ),
)
print("\ngold answer:", dataset[0]["answer"])


# ### Rewards
# 
# `channel_format` checks a closed reasoning channel, `answer_is_a_number`
# checks the visible reply, and `answer_is_correct` gives 4.0 only for an
# exact answer. A small negative proximity score breaks ties among wrong answers
# on the two-generation path.

# In[31]:


import re
from decimal import Decimal, InvalidOperation

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

def _completion_text(completion):
    if not isinstance(completion, list):
        return completion
    if not completion:
        return ""
    content = completion[0].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)

def channel_format(completions, **kwargs):
    scores = []
    for completion in completions:
        reasoning, answer = split_channels(_completion_text(completion))
        scores.append((1.0 if answer else -1.0) + (0.5 if reasoning else 0.0))
    return scores

def answer_is_a_number(completions, **kwargs):
    scores = []
    for completion in completions:
        _, answer = split_channels(_completion_text(completion))
        found = NUMBER.findall(answer.replace(",", ""))
        if len(found) == 1 and answer.replace(",", "").strip() == found[0]:
            scores.append(1.0)
        elif found:
            scores.append(0.25)
        else:
            scores.append(-0.5)
    return scores

def _wrong_answer_score(text, gold):
    reasoning, reply = split_channels(text)
    found = NUMBER.findall((reply or reasoning).replace(",", ""))
    if not found:
        return -0.5
    try:
        relative_error = abs(Decimal(found[-1]) - Decimal(gold)) / (abs(Decimal(gold)) + 1)
        proximity = 1.0 / (1.0 + float(min(relative_error, Decimal("1000000"))))
    except (InvalidOperation, OverflowError):
        return -0.5
    return -0.5 + 0.2 * proximity

def answer_is_correct(prompts, completions, answer, **kwargs):
    scores = []
    for completion, gold in zip(completions, answer):
        text = _completion_text(completion)
        _, reply = split_channels(text)
        found = NUMBER.findall(reply.replace(",", ""))
        try:
            correct = bool(found) and Decimal(found[-1]) == Decimal(gold)
        except InvalidOperation:
            correct = False
        scores.append(4.0 if correct else _wrong_answer_score(text, gold))
    return scores


# In[32]:


# Check reward behavior before spending GPU time.
_good = "48/2=24, 48+24=72." + REASONING_TO_ANSWER + "72" + ANSWER_SUFFIX
_bad = "I think it is about seventy." + REASONING_TO_ANSWER + "about seventy" + ANSWER_SUFFIX
_none = "72"  # Correct number in unfinished reasoning, but no visible answer.

_results = {}
for name, completion in [("good", _good), ("bad", _bad), ("no reply", _none)]:
    scores = (
        channel_format([completion]),
        answer_is_a_number([completion]),
        answer_is_correct(
            prompts = [None],
            completions = [completion],
            answer = ["72"],
        ),
    )
    _results[name] = scores
    print(f"{name:9s}", *scores)

assert _results["good"] == ([1.5], [1.0], [4.0])
assert _results["bad"] == ([1.5], [-0.5], [-0.5])
assert _results["no reply"][0] == [-0.5]
assert _results["no reply"][1] == [-0.5]
assert -0.31 < _results["no reply"][2][0] < -0.29


# <a name="Train"></a>
# ### Train
# 
# The dual-T4 smoke run uses two completions per prompt, a 128-token completion
# cap, sequence-level importance sampling, `loss_type='grpo'` and
# `adamw_8bit`. Capped completions are kept so unfinished answers receive
# their penalty.
# 
# This four-step run validates generation, rewards and backward—not answer
# quality. For training, raise `max_steps` and restore
# `max_completion_length`. T4 uses float32 for this model.

# In[12]:


from trl import GRPOConfig, GRPOTrainer

VALIDATION_STEPS = 4
SMOKE_MAX_COMPLETION_LENGTH = min(128, max_completion_length)

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_steps = 10,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    max_grad_norm = 0.1,
    logging_steps = 1,
    log_completions = False,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 1,
    num_generations = NUM_GENERATIONS,
    chat_template_kwargs = {"reasoning_effort": REASONING_EFFORT},
    max_completion_length = SMOKE_MAX_COMPLETION_LENGTH,
    mask_truncated_completions = False,
    max_steps = VALIDATION_STEPS,
    save_steps = VALIDATION_STEPS,
    report_to = "none",
    output_dir = "outputs_qwen38_validated",
    importance_sampling_level = "sequence",
    loss_type = "grpo",
)


# The connected 2 x T4 run completed four steps in 191.6 seconds:
# 
# | step | loss | grad norm | reward varied in group |
# |---:|---:|---:|:---:|
# | 1 | 5.9604645e-08 | 0.3225 | yes |
# | 2 | 5.2861870e-09 | 0.0000 | no |
# | 3 | 2.2882598e-09 | 0.0000 | no |
# | 4 | 1.3113022e-06 | 0.3770 | yes |
# 
# All four losses were finite and non-zero. Every rollout reached the 128-token
# smoke cap, so this proves the training path—not model-quality improvement.
# For a quality run, restore the 640-token completion budget and evaluate held-out
# exact-answer accuracy.

# In[18]:


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        channel_format,
        answer_is_a_number,
        answer_is_correct,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()


# In[21]:


import torch

for device_index in range(torch.cuda.device_count()):
    peak = torch.cuda.max_memory_reserved(device_index) / 1024 ** 3
    total = torch.cuda.get_device_properties(device_index).total_memory / 1024 ** 3
    print(f"cuda:{device_index} peak reserved VRAM {peak:.2f} GiB out of {total:.2f} GiB")


# <a name="Inference"></a>
# ### After training
# 
# Use a short retained arithmetic example to verify generation, channel parsing
# and exact-answer comparison.

# In[30]:


from transformers import TextStreamer

example = dataset[191]
text = tokenizer.apply_chat_template(
    example["prompt"],
    tokenize = False,
    add_generation_prompt = True,
    reasoning_effort = REASONING_EFFORT,
)
inputs = tokenizer(text = text, return_tensors = "pt", add_special_tokens = False).to("cuda")

out = model.generate(
    **inputs,
    max_new_tokens = min(128, max_completion_length),
    temperature = 1.0,
    do_sample = True,
    streamer = TextStreamer(_text_tokenizer, skip_prompt = True, skip_special_tokens = False),
)
completion = _text_tokenizer.decode(
    out[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens = False,
)
_, reply = split_channels(completion)
print("\nparsed reply:", repr(reply))
print("gold answer :", example["answer"])


# <a name="Save"></a>
# ### Save
# 
# Save the LoRA adapters and tokenizer without duplicating the 27B base model.

# In[23]:


model.save_pretrained("qwen38_27b_lora")
tokenizer.save_pretrained("qwen38_27b_lora")
# model.push_to_hub("your_name/qwen38_27b_lora", private = True)
# tokenizer.push_to_hub("your_name/qwen38_27b_lora", private = True)


# Check that the saved adapter contains non-zero tensors. This validates
# serialization, not model quality.

# In[24]:


from safetensors import safe_open

n_nonzero = 0
with safe_open("qwen38_27b_lora/adapter_model.safetensors", framework = "pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        zero_fraction = (tensor == 0).float().mean().item()
        if zero_fraction < 1.0:
            n_nonzero += 1

assert n_nonzero > 0, "All saved adapter tensors are zero."
print(n_nonzero, "adapter tensors contain non-zero values")


# ### Optional 16-bit merge
# 
# A standalone merged checkpoint is roughly the 55.6 GB size of the public
# full-precision model. The merge cell stays disabled to avoid an accidental
# large write. Official GGUFs are available at
# [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

# In[ ]:


# Merge to 16-bit.
if False:
    model.save_pretrained_merged(
        "qwen38_27b_finetune_16bit",
        tokenizer,
        save_method = "merged_16bit",
    )
if False:
    model.push_to_hub_merged(
        "your_name/qwen38_27b_finetune_16bit",
        tokenizer,
        save_method = "merged_16bit",
        private = True,
    )

# Just LoRA adapters.
if False:
    model.save_pretrained("qwen38_27b_lora")
    tokenizer.save_pretrained("qwen38_27b_lora")


# ### Memory notes
# 
# The 4-bit repository is 20.83 GiB on disk; that is not a CUDA-allocation
# measurement. The final text-only load split decoder layers 0-34 onto GPU 0 and
# 35-63 plus norm/head onto GPU 1. Four-step peak reserved VRAM was 11.94/11.58
# GiB on two 14.56 GiB T4s, with no CPU or disk offload.
# 
# A separate full-VLM planner diagnostic reported 20.487 GiB of weights, 1.510
# GiB activation reserve and 2.441 GiB headroom, but that loader failed before
# GRPO training and is not the path used here. The 40 GiB configuration was not
# executed.

# ### And we're done
# 
# 
# If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. [Unsloth Reinforcement Learning docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) for GSPO, GAPO, Dr GRPO and the rest of the options
# 2. [Memory efficient RL](https://unsloth.ai/docs/basics/memory-efficient-rl)
# 3. [Unsloth notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
# 4. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
# 5. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
# 6. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
# 7. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
# 8. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
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
