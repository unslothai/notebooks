#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i>
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
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install --upgrade -qqq uv\ntry: import numpy, PIL; _numpy = f\'numpy=={numpy.__version__}\'; _pil = f\'pillow=={PIL.__version__}\'\nexcept: _numpy = "numpy"; _pil = "pillow"\n# Pin Kaggle\'s preinstalled torch and torchvision instead of upgrading them.\n# `--upgrade torchvision` pulls a CUDA 13.0 torch on top of Kaggle\'s CUDA 12.8\n# torchaudio, and torchaudio then refuses to import: "PyTorch and TorchAudio were\n# compiled with different CUDA versions". Pin on the base version, not\n# torch.__version__, so the local +cu128 label does not have to exist on the index;\n# PEP 440 treats 2.9.1+cu128 as satisfying ==2.9.1.\ntry:\n    import torch, torchvision\n    _torch = f\'torch=={torch.__version__.split("+")[0]}\'\n    _tv = f\'torchvision=={torchvision.__version__.split("+")[0]}\'\nexcept Exception:\n    _torch, _tv = "torch", "torchvision"\n!uv pip install -qqq {_numpy} {_pil} {_torch} {_tv} bitsandbytes xformers unsloth\n!uv pip install -qqq triton "huggingface_hub>=0.34.0" "datasets==4.3.0"\n!uv pip install -qqq --no-deps --upgrade "torchao>=0.16.0"\n!uv pip install -qqq transformers==5.15.1\n!uv pip install -qqq --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# ### What you are about to load
# 
# Qwen3.8-27B is a `qwen3_5` checkpoint: 27.8B parameters, 64 decoder layers, a vision
# tower, a 248320 token vocabulary and untied embeddings. Three layers out of every four
# are **linear attention** (a gated delta net, the `linear_attn.*` modules) and the fourth
# is ordinary full attention. That ratio is the whole reason a 27B model is tractable
# here - the linear layers keep a fixed size recurrent state instead of a KV cache that
# grows with the sequence.
# 
# The gated delta net runs on `flash-linear-attention`'s Triton kernels. Unsloth ships a
# pruned copy of fla inside `unsloth_zoo`, so there is nothing to install for the fast
# path; the cell below prints which copy is live and confirms it before any weight is
# loaded. Without it, transformers silently falls back to a float32 Python loop over
# chunks, which is several times slower and never says so.
# 
# **Two Tesla T4s, not one.** One T4 cannot hold this model in 4-bit. Two can, and Kaggle
# gives you two. Everything below is written for that, and it deliberately does NOT pass
# `device_map`. Unsloth defaults to `"sequential"`, which fills the first card and then
# the second; that is what the Muse Glimmer notebooks rely on.
# 
# `device_map = "balanced"` looks like the right choice here and is not. It routes through
# `get_balanced_memory`, which caps cuda:0 at 9.12 GiB while giving cuda:1 13.01 GiB. The
# model needs 19.63 GiB once quantised, which fits in 22.13 GiB of budget only if nothing
# large is left over -- and the 2.37 GiB fp16 `lm_head` is. It gets assigned to CPU, and
# bitsandbytes refuses any device map containing CPU or disk entries, so the load dies with
# "Some modules are dispatched on the CPU or the disk". Measured on a real Kaggle T4 x2.
# 
# `offload_embedding = True` keeps the 2.37 GiB input embedding in CPU RAM.

# ### Which repo to load
# 
# `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` is nf4 with the parts that do not survive 4-bit
# left in float16: `lm_head`, `embed_tokens`, the vision tower, and inside every gated
# delta net the `in_proj_qkv`, `in_proj_a` and `in_proj_b` projections.
# 
# That last group is a deliberate choice, and it is worth knowing why, because the obvious
# alternative does not fit. Measured on wikitext-2 against the bfloat16 checkpoint:
# 
# | what stays in float16 inside `linear_attn` | weights | perplexity | KL vs bf16 | spare room on the card holding `lm_head` |
# |---|---|---|---|---|
# | everything | 22.97 GiB | 6.2259 | 0.02435 | **1.33 GiB - does not fit** |
# | `in_proj_qkv`, `in_proj_a`, `in_proj_b` (this repo) | 18.80 GiB | 6.2221 | 0.03063 | 3.24 GiB |
# | `in_proj_a`, `in_proj_b` only | 15.32 GiB | 6.2474 | 0.03390 | 4.90 GiB |
# | nothing | 15.29 GiB | 6.2440 | 0.03449 | 4.92 GiB |
# | *(bfloat16, for reference)* | *51.70 GiB* | *6.1206* | *-* | *-* |
# 
# Keeping the whole of `linear_attn` in float16 costs 7.7 GiB and buys 0.006 nats. Every
# 4-bit layout lands within 0.03 perplexity of every other, while all four sit about 2%
# above bfloat16 - which is to say the 4-bit floor dominates, not where inside the gated
# delta net you spend the 16 bits. What the 7.7 GiB does change is whether the model fits
# at all: two T4s hold 29.2 GiB between them once the CUDA context is out, and the card
# that ends up with `lm_head` needs a couple of GiB for logits.
# 
# `in_proj_a` and `in_proj_b` stay 16-bit for a different reason: they are 5120 -> 48
# projections, 0.025 GiB across all 48 layers. Quantizing them saves nothing measurable
# and still pays a dequantisation on every call - twice per step, because gradient
# checkpointing recomputes the forward.
# 
# The bfloat16 original is at [`unsloth/Qwen3.8-27B`](https://huggingface.co/unsloth/Qwen3.8-27B)
# and GGUFs are at [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

# In[ ]:


MODEL_NAME      = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_NAME_BF16 = "unsloth/Qwen3.8-27B"  # fallback, quantised on the fly


# In[ ]:


# The install cell above runs under %%capture, so a pip failure there leaves no
# trace and surfaces here as a bare ModuleNotFoundError twenty minutes in. Say what
# actually happened instead.
import importlib.util
_missing = [m for m in ("unsloth", "unsloth_zoo", "bitsandbytes", "transformers")
            if importlib.util.find_spec(m) is None]
assert not _missing, (
    f"the install cell did not finish: {_missing} missing. Re-run it with the "
    f"%%capture line deleted to see what pip said."
)

# Unsloth first, before transformers. It injects its vendored copy of
# flash-linear-attention into sys.modules as a real top-level `fla`, and the
# gated-delta modeling module binds whichever kernels exist at *its* import time.
# Import transformers first and that binding is already made, silently, against
# the pure-PyTorch fallback - several times slower, and nothing says so.
from unsloth import FastModel
import fla
print("fla:", fla.__version__, "| vendored:", getattr(fla, "_UNSLOTH_VENDORED_FLA", False))

import transformers
print("transformers:", transformers.__version__)
from transformers.utils import is_flash_linear_attention_available
assert is_flash_linear_attention_available(), (
    "fla is not available - the gated delta net would fall back to a float32 "
    "Python loop. Re-run the install cell."
)

# Decide here rather than 40 minutes into a download. A repo can exist and still
# hold no weights, in which case from_pretrained's error names a missing file
# instead of the missing upload. The prebuilt 4-bit repo is a ~21 GB pull; the
# bf16 source is ~52 GB and is quantised on the way in, so prefer the former when
# it is populated and fall back rather than fail.
from huggingface_hub import HfApi

def _has_weights(repo):
    # A local directory is a legitimate MODEL_NAME (a copy you quantised yourself),
    # and model_info would just 404 on it, so check the filesystem first.
    import glob, os
    if os.path.isdir(repo):
        return bool(glob.glob(os.path.join(repo, "*.safetensors")))
    try:
        info = HfApi().model_info(repo, files_metadata = False)
    except Exception:
        return False
    return any(f.rfilename.endswith(".safetensors") for f in info.siblings)

if not _has_weights(MODEL_NAME):
    print(f"{MODEL_NAME} holds no weights yet - falling back to {MODEL_NAME_BF16}.")
    print("That is a ~52 GB download quantised on the fly, so the load cell is slow.")
    MODEL_NAME = MODEL_NAME_BF16
assert _has_weights(MODEL_NAME), f"neither 4-bit nor bf16 repo has weights: {MODEL_NAME}"
print("loading from:", MODEL_NAME)

from transformers import AutoConfig
print("architecture:", AutoConfig.from_pretrained(MODEL_NAME).model_type)

import torch
print("GPUs:", torch.cuda.device_count(),
      "->", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 2, (
    "Qwen3.8-27B in 4-bit needs more than one 16 GB card. On Kaggle pick the "
    "T4 x2 accelerator (Settings -> Accelerator -> GPU T4 x2)."
)


# In[ ]:


# Workaround for a CUDA illegal memory access when loading across two GPUs.
#
# unsloth_zoo's forced-float32 pass (which qwen3_5 reaches, because it is in
# FORCE_FLOAT32) casts every module and calls torch.cuda.empty_cache() after each one -
# 1297 times for this model. Those casts are asynchronous, and empty_cache() releases
# cached blocks against the CURRENT device without synchronising the others. On a model
# dispatched across two cards it can therefore reclaim memory that in-flight work on the
# other card is still writing into, which corrupts and surfaces later as
# "CUDA error: an illegal memory access was encountered".
#
# Reproduced and fixed on Kaggle T4 x2. Synchronising before each release costs nothing
# at run time (559 s against 587 s for five steps under CUDA_LAUNCH_BLOCKING, which
# merely hid the race) and leaves peak memory unchanged.
#
# Self-disabling: once the fix ships in unsloth_zoo this becomes a no-op.
import inspect
import torch
import unsloth.models.vision as _uv
import unsloth_zoo.patching_utils as _pu

_needs_workaround = "torch.cuda.synchronize(_device_index)" not in inspect.getsource(_pu)
if not _needs_workaround:
    print("unsloth_zoo already synchronises before empty_cache; workaround not needed")
elif torch.cuda.device_count() < 2:
    print("single GPU: workaround not needed")
else:
    _orig_empty = torch.cuda.empty_cache

    def _synchronised_empty_cache():
        for _i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(_i)
        return _orig_empty()

    _orig_patch = _uv.patch_model_and_tokenizer

    def _patched(*args, **kwargs):
        # Scoped to this call and restored in a finally, so nothing else in the
        # process sees a patched empty_cache.
        torch.cuda.empty_cache = _synchronised_empty_cache
        try:
            return _orig_patch(*args, **kwargs)
        finally:
            torch.cuda.empty_cache = _orig_empty

    # vision.py did `from ... import patch_model_and_tokenizer`, so the name has to be
    # replaced there; patching the defining module would silently do nothing.
    _uv.patch_model_and_tokenizer = _patched
    print("multi-GPU load workaround installed")


# In[ ]:


# `is_flash_linear_attention_available()` says fla could be used, not that it was.
# The gated-delta module resolves its kernels at import time and falls back silently
# to a float32 torch loop, so the only honest check is to count calls. Wrap the entry
# points BEFORE the model is built -- the modeling module binds these by value.
import fla.ops.gated_delta_rule as _gdr

FLA_CALLS = {"chunk": 0, "recurrent": 0}

def _count(fn, key):
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        FLA_CALLS[key] += 1
        return fn(*args, **kwargs)
    wrapper._unsloth_counted = True
    return wrapper

for _attr, _key in (("chunk_gated_delta_rule", "chunk"),
                    ("fused_recurrent_gated_delta_rule", "recurrent")):
    _fn = getattr(_gdr, _attr, None)
    if _fn is not None and not getattr(_fn, "_unsloth_counted", False):
        setattr(_gdr, _attr, _count(_fn, _key))
print("fla call counters installed:", list(FLA_CALLS))


# In[ ]:


from unsloth import FastModel
import torch

max_seq_length = 1024  # Qwen3.8 goes to 262144, but 1024 is what two T4s train on

# Keep the gated delta net's projections in 16-bit. There is no from_pretrained
# argument for this - FastModel builds its BitsAndBytesConfig from this list - so
# append to the list itself, in place: unsloth.models.vision imported the name by
# reference, and rebinding it here would not be seen there. Loading a repo that is
# already 4-bit ignores this, since nothing is quantised on the way in.
import unsloth_zoo.peft_utils as _peft_utils
for _keep in ("in_proj_qkv", "in_proj_a", "in_proj_b"):
    if _keep not in _peft_utils.SKIP_QUANTIZATION_MODULES:
        _peft_utils.SKIP_QUANTIZATION_MODULES.append(_keep)

model, tokenizer = FastModel.from_pretrained(
    model_name        = MODEL_NAME,
    max_seq_length    = max_seq_length,
    load_in_4bit      = True,
    full_finetuning   = False,
    offload_embedding = True,        # keeps the 2.37 GiB embedding in CPU RAM
)

print(f"placed over {len(set(model.hf_device_map.values()))} devices")

# Confirm the layout is what was asked for rather than assuming the skip list took.
import bitsandbytes as bnb
_n4 = sum(1 for _, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit))
_qkv = model.get_submodule("model.language_model.layers.0.linear_attn.in_proj_qkv")
print(f"Linear4bit modules: {_n4} | linear_attn.in_proj_qkv: "
      f"{type(_qkv).__name__} {_qkv.weight.dtype}")
for _i in range(torch.cuda.device_count()):
    print(f"  cuda:{_i} {torch.cuda.memory_allocated(_i) / 2**30:5.2f} GiB allocated")


# Do not pass `dtype = torch.float16` here, and do not set `fp16 = True` on the trainer
# further down. Qwen3.5's gated delta net produces NaN gradients in pure float16, so
# Unsloth keeps this architecture on a float32 autocast path and picks the loading dtype
# itself. Asking for float16 explicitly gets you `Unsloth: Model is in bfloat16 precision
# but you want to use float16 precision`, which reads like a configuration mistake rather
# than the architecture constraint it is. The heavy matmuls still run on the T4's float16
# tensor cores - `bnb_4bit_compute_dtype` is float16 either way.

# We now add LoRA adapters so we only need to update a small amount of parameters.
# 
# The adapters land on **every** linear in the language model, the gated delta net
# included: 240 modules inside `linear_attn`, 192 in the MLPs and 64 in the full attention
# layers, 496 in total, 58.4M trainable parameters, 0.21% of the model. Unsloth's target
# matcher reaches the `linear_attn.*` leaves through the `attn` tag, so nothing special
# has to be listed. The vision tower is left frozen because this is a text finetune.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Reaches the linear_attn (gated delta) modules too
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 30% less VRAM, fits 2x larger batch sizes
    random_state = 3407,
)

# Confirm the gated delta net really is being trained, not just the MLPs.
import collections
where = collections.Counter()
for name, _ in model.named_modules():
    if name.endswith("lora_A.default"):
        parent = name.rsplit(".lora_A", 1)[0]
        where["linear_attn" if ".linear_attn." in parent else
              "self_attn"   if ".self_attn."   in parent else
              "mlp"         if ".mlp."         in parent else "other"] += 1
print(dict(where), "->", sum(where.values()), "adapters")


# <a name="Data"></a>
# ### Data Prep
# 
# We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)
# dataset in ShareGPT style, the same multi turn conversational dataset the Llama
# conversational notebook uses.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")


# We now use `standardize_sharegpt` to convert ShareGPT style datasets into HuggingFace's generic format. This changes the dataset from looking like:
# ```
# {"from": "system", "value": "You are an assistant"}
# {"from": "human", "value": "What is 2+2?"}
# {"from": "gpt", "value": "It's 4."}
# ```
# to
# ```
# {"role": "system", "content": "You are an assistant"}
# {"role": "user", "content": "What is 2+2?"}
# {"role": "assistant", "content": "It's 4."}
# ```

# In[ ]:


from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)


# Let's see how row 100 looks like!

# In[ ]:


dataset[100]["conversations"]


# Qwen3.8 ships its own chat template, so we apply the tokenizer's template directly
# rather than calling `get_chat_template`. It renders ChatML: `<|im_start|>role\n ...
# <|im_end|>`, with a `<think>` block opening every assistant turn.

# In[ ]:


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False,
        )
        for convo in convos
    ]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


# And we see how the chat template transformed these conversations.

# In[ ]:


dataset[100]["text"]


# ## Controlling how much the model thinks
# 
# The template renders a reasoning-effort instruction into the system block and it
# defaults to `xhigh`, so if you never set it you are paying for the most expensive
# setting on every prompt. Pass `reasoning_effort` to `apply_chat_template` to change it;
# the accepted values are `xhigh`, `medium` and `low`, and anything else raises from
# inside the template. `high` is accepted as an alias and resolves to `xhigh`.
# 
# Reasoning tokens count against `max_new_tokens`, so a run that hits the ceiling
# mid-thought returns an empty answer, which looks exactly like a wrong one. Raise
# `max_new_tokens` before you raise the effort.

# In[ ]:


reasoning_effort = "low"  # low, medium, xhigh. The template default is xhigh.

messages = [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
    reasoning_effort = reasoning_effort,
)
print(text)


# <a name="Train"></a>
# ### Train the model
# 
# Now let's train our model. We do 30 steps to speed things up, but you can set
# `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
# 
# Two settings are worth understanding rather than copying.
# 
# `per_device_train_batch_size = 1` with `gradient_accumulation_steps = 4` is not
# timidity. A 248320 token vocabulary makes the logits the single largest allocation in
# the step - one sequence of 1024 positions is already 1024 x 248320 floats before the
# cross entropy touches them - and they land on whichever card holds `lm_head`, which is
# the card with the least room left. Raising the batch size is the fastest way to run out
# of memory here; raising gradient accumulation costs time instead, which you have.
# 
# `optim = "adamw_8bit"` matters less than usual, because LoRA r=8 over 496 modules is
# only 58.4M trainable parameters, but it is free to keep.

# In[ ]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = max_seq_length,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        # No fp16 / bf16 here: Unsloth picks the precision for this architecture.
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# We also use Unsloth's `train_on_responses_only` method to only train on the assistant
# outputs and ignore the loss on the user's inputs. This helps increase accuracy of
# finetunes.

# In[ ]:


from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)


# Let's verify masking the instruction part is done. Row 100, then the same row with
# everything the model is not trained on blanked out - you should see only the answer.
# 
# Qwen3.8 is vision capable, so `FastModel` hands back a processor rather than a bare
# tokenizer. Calling it with a single positional argument would be read as an image, so
# reach for the inner text tokenizer when you want to tokenize a plain string.

# In[ ]:


text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
space = text_tokenizer(" ", add_special_tokens = False).input_ids[0]
print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
print("=" * 30)
print(tokenizer.decode([space if x == -100 else x
                        for x in trainer.train_dataset[100]["labels"]]))


# One more memory step. `offload_embedding = True` puts `embed_tokens` in CPU RAM at load
# time, but the trainer moves the model onto the accelerator when `train()` starts, which
# quietly drags the 2.37 GiB embedding back onto a card. The lookup hooks Unsloth installs
# read the weight's device live, so we can simply push it back to CPU once training has
# begun and it stays there.
# 
# The callback returns early when accelerate owns the placement, which is what happens on
# the sharded multi-GPU load: the embedding then carries a dispatch hook and moving it
# by hand would strand the hook's execution device.

# In[ ]:


from transformers import TrainerCallback

class KeepEmbeddingOffloaded(TrainerCallback):
    # The trainer places the model on the accelerator at train() time, which undoes
    # offload_embedding. Put the input embedding back on the CPU once, after that.
    def on_train_begin(self, args, state, control, **kwargs):
        embed_tokens = kwargs["model"].get_input_embeddings()
        # Skip when accelerate owns placement: the embedding then carries a dispatch
        # hook, which is why the loader declined the offload.
        hook = getattr(embed_tokens, "_hf_hook", None)
        if getattr(hook, "execution_device", None) is not None:
            return control
        if embed_tokens.weight.device.type != "cpu":
            embed_tokens.to("cpu")
            torch.cuda.empty_cache()
        return control

trainer.add_callback(KeepEmbeddingOffloaded())


# In[ ]:


# @title Show current memory stats
# Both cards, not just cuda:0 - the model is sharded and the two are not symmetric.
start_gpu_memory = []
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    reserved = torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024
    start_gpu_memory.append(reserved)
    print(f"cuda:{i} = {props.name}. Max memory = "
          f"{props.total_memory / 1024 / 1024 / 1024:.3f} GB. "
          f"{reserved:.3f} GB reserved.")


# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[ ]:


trainer_stats = trainer.train()


# In[ ]:


# Training is done, so the counters have seen the real forward and backward.
print("fla kernel calls during training:", FLA_CALLS)
assert FLA_CALLS["chunk"] > 0, (
    "the vendored fla chunk kernel was never called - the gated delta net ran on the "
    "float32 torch fallback, which is several times slower. Check that unsloth was "
    "imported before transformers."
)
print(f"  {FLA_CALLS['chunk']} chunk calls across the training run")


# In[ ]:


# @title Show final memory and time stats
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    used = torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024
    total = props.total_memory / 1024 / 1024 / 1024
    print(f"cuda:{i}: peak reserved {used:.3f} GB of {total:.3f} GB "
          f"({used / total * 100:.1f} %), {used - start_gpu_memory[i]:.3f} GB for training.")


# ### What this actually costs
# 
# Measured with exactly the settings above - 4-bit, LoRA r=8 on all 496 language linears,
# `use_gradient_checkpointing = "unsloth"`, `max_seq_length = 1024`, batch size 1,
# gradient accumulation 4, `offload_embedding = True` plus the callback:
# 
# | | GiB |
# |---|---|
# | weights resident once training is running | 18.80 |
# | activations, gradients, optimiser at the peak | 1.23 |
# | peak **reserved**, i.e. what the allocator holds | 23.71 |
# 
# Split over two T4s, accelerate puts 7.08 GiB on the first card and 11.36 GiB on the
# second, leaving 7.52 and 3.24 GiB of headroom respectively. The second card is the tight
# one: it carries `lm_head`, so the logits land there.
# 
# If you are close to the limit, the levers that move the number most are lowering
# `max_seq_length` and turning off `finetune_mlp_modules`. Raising
# `per_device_train_batch_size` above 1 is the fastest way to run out.
# 
# Two caveats on these figures. They were measured on one large card with the allocator
# capped to what two T4s add up to, and with the float16 loading path a T4 takes - so the
# memory numbers transfer but **the step time was not measured on a T4** and is not quoted
# here. And `peak reserved` is what the allocator holds including fragmentation, which is
# the number that OOMs you, not the 18.80 GiB of weights.

# <a name="Inference"></a>
# ### Inference
# Let's run the model via Unsloth native inference!

# In[ ]:


messages = [
    {"role": "user", "content": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
    reasoning_effort = "low",
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 256,
    temperature = 0.7, top_p = 0.8, top_k = 20,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an
# online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit,
# scroll down.

# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("HF_ACCOUNT/qwen_lora") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/qwen_lora") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "qwen_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 1024,
        load_in_4bit = True,
        offload_embedding = True,
    )

messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
    reasoning_effort = "low",
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 256,
    temperature = 0.7, top_p = 0.8, top_k = 20,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# ### Saving to float16
# 
# We also support saving to 16-bit directly for deployment. Set `if False` to `if True` to
# let it run.
# 
# A merged Qwen3.8-27B is about 52 GB on disk, which is more than a Kaggle kernel's output
# quota and more than its working disk once the 4-bit weights are already there. The merge
# also needs enough CPU RAM to hold the dequantized model. Do this on a machine with the
# room, or push the adapters and merge elsewhere.

# In[ ]:


if False: # Change to True to save finetune!
    model.save_pretrained_merged("qwen3_8_finetune", tokenizer)

if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/qwen3_8_finetune", tokenizer,
        private = True,
    )


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i>
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
