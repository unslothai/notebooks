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

# ### Installation

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install --upgrade -qqq uv\ntry: import numpy, PIL; _numpy = f\'numpy=={numpy.__version__}\'; _pil = f\'pillow=={PIL.__version__}\'\nexcept: _numpy = "numpy"; _pil = "pillow"\n# Pin Kaggle\'s torch and torchvision: upgrading them pulls a CUDA 13.0 torch onto\n# Kaggle\'s CUDA 12.8 torchaudio, which then refuses to import. Pin the base version,\n# since the local +cu128 label does not exist on the index.\ntry:\n    import torch, torchvision\n    _torch = f\'torch=={torch.__version__.split("+")[0]}\'\n    _tv = f\'torchvision=={torchvision.__version__.split("+")[0]}\'\nexcept Exception:\n    _torch, _tv = "torch", "torchvision"\n!uv pip install -qqq {_numpy} {_pil} {_torch} {_tv} bitsandbytes xformers unsloth\n!uv pip install -qqq triton "huggingface_hub>=0.34.0" "datasets==4.3.0"\n!uv pip install -qqq --no-deps --upgrade "torchao>=0.16.0"\n!uv pip install -qqq transformers==5.15.1\n!uv pip install -qqq --no-deps trl==0.25.1\n')


# GKD needs unslothai/unsloth#11440 and unslothai/unsloth-zoo#1310, which are on
# main but not yet in a PyPI release. Without them the generated trainer downcasts
# `GKDConfig` back to `SFTConfig` and silently drops `lmbda`, `beta` and
# `temperature`, so distillation degrades to plain SFT with no error
# (unslothai/unsloth#1941). Delete the next cell once a release carries both.

# In[ ]:


get_ipython().system('pip install --no-deps --upgrade --force-reinstall      git+https://github.com/unslothai/unsloth-zoo git+https://github.com/unslothai/unsloth')


# ### Knowledge distillation with Unsloth
# 
# Distillation trains a small **student** to match the output distribution of a
# larger **teacher**, rather than to match one-hot labels. It usually beats plain
# finetuning at the same student size, because a full distribution carries far more
# signal per token than a single correct answer.
# 
# This notebook uses TRL's `GKDTrainer`, which Unsloth patches like any other TRL
# trainer.

# Three things are worth knowing before you change the models:
# 
# 1. **Student and teacher must share a vocabulary.** The loss compares two
#    distributions position by position, so a mismatch is not a quality problem, it
#    is a shape error. Same family is the safe rule; even within a family, check.
# 2. **A multimodal checkpoint hands back a processor, not a tokenizer.** Every
#    model here is a `*ForConditionalGeneration`, and passing the processor to a
#    text-only trainer makes it try to read the rendered chat prompt as an image
#    (`Incorrect image source`). Unwrap it, as the helper below does.
# 3. **`lmbda` picks off-policy or on-policy.** `lmbda = 0` scores the fixed
#    dataset completions; `lmbda = 1` makes the student generate and scores its own
#    samples, which costs generation per step but avoids the train/inference
#    mismatch. `beta` interpolates the divergence: 0 is forward KL, 1 is reverse
#    KL, in between is generalized JSD.

# ### Pick a configuration
# 
# A teacher equal to the student is self-distillation: the student learns from its
# own frozen base, which needs only one copy of the weights and so is the only
# option for the 27B and 30B models on two T4s.
# 
# GKD holds full `(batch, seq, vocab)` logits for **both** models, so peak memory is
# driven by the vocabulary rather than by the weights. Gemma-4's 262144-token
# vocabulary is what makes it expensive, not its size.
# 
# | config | weights | measured on a Kaggle 2x T4 |
# | --- | --- | --- |
# | `qwen3` | 1.2 + 3.4 GB | PASS, loss 0.413, 392/392 adapters, 10.9 GB |
# | `gemma-4` | 8.1 + 10.9 GB | PASS, loss 0.079, 410/410 adapters, 14.3 GB |
# | `muse-glimmer` | 22.2 GB | OOM by 208 MiB at seq 384, needs a larger card |
# | `qwen3.8` | 22.3 GB | blocked on a dtype mismatch, see below |
# 
# `unsloth/gemma-4-26B-A4B-it` is 51.6 GB with no prebuilt 4-bit, so it needs an
# A100 or better. Listed for that case, not for T4.

# In[ ]:


CONFIGS = {
    "gemma-4": dict(
        student = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        teacher = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    ),
    "qwen3": dict(
        student = "unsloth/Qwen3-0.6B",
        teacher = "unsloth/Qwen3-1.7B",
    ),
    "qwen3.8": dict(
        student = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
        teacher = None,
    ),
    "muse-glimmer": dict(
        student = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
        teacher = None,
    ),
}

CONFIG = "qwen3"
student_name = CONFIGS[CONFIG]["student"]
teacher_name = CONFIGS[CONFIG]["teacher"]


# Two configurations need a note.
# 
# **gemma-4 needs `torch.compile` off on T4-class cards.** Under compile the
# gradient-checkpoint recompute executes the layer differently from the forward
# pass, and `torch.utils.checkpoint` refuses the mismatch: the recompute disagrees
# about tensor rank, `[1, 474, 512]` saved against `[1, 474, 1, 512]` recomputed.
# It passes on a B200 with compile on, so this is specific to older cards and is
# still being tracked. It costs the other configurations nothing.
# 
# **qwen3.8 is blocked** on something this notebook cannot fix: the first training
# step raises `expected mat1 and mat2 to have the same dtype, but got: BFloat16 !=
# Half` inside the gated delta net. It is not memory, and it does not reproduce on
# a card that supports bfloat16.

# In[ ]:


import os
if CONFIG.startswith("gemma-4"):
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

max_seq_length = 1024
load_in_4bit = True
print(f"student: {student_name}")
print(f"teacher: {teacher_name or '(self-distillation: the student\'s own frozen base)'}")


# ### Load the student

# In[ ]:


from unsloth import FastLanguageModel
import torch

student, processor = FastLanguageModel.from_pretrained(
    student_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)


# A `*ForConditionalGeneration` checkpoint returns a multimodal processor. The
# text-only trainer must be handed the tokenizer it wraps, or the processor tries
# to resolve the chat prompt as an image source and raises.

# In[ ]:


def text_tokenizer(maybe_processor):
    inner = getattr(maybe_processor, "tokenizer", None)
    if inner is not None and type(maybe_processor).__name__.endswith("Processor"):
        print(f"unwrapped {type(maybe_processor).__name__} -> {type(inner).__name__}")
        return inner
    return maybe_processor

tokenizer = text_tokenizer(processor)


# The distillation loss reads the output head directly, so it has to stay a dense
# `[vocab, hidden]` matrix. Unsloth keeps `lm_head` out of quantization for this
# reason; a checkpoint quantized elsewhere may not, so check rather than assume.

# In[ ]:


head = student.get_output_embeddings()
assert head.weight.dim() == 2, f"output head is not dense: {type(head.weight).__name__}"
print(f"output head {type(head).__name__} {tuple(head.weight.shape)} {head.weight.dtype}")


# ### Attach LoRA adapters to the student

# In[ ]:


student = FastLanguageModel.get_peft_model(
    student,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

trainable = [n for n, p in student.named_parameters() if p.requires_grad]
print(f"{len(trainable)} trainable tensors")


# `lora_B` is zero-initialized, so `dL/dA` is zero on the very first step and only
# the `lora_B` halves move. That is expected, not a sign nothing is learning.

# ### Load the teacher, or reuse the student's frozen base

# Under LoRA the frozen base **is** a perfectly good teacher, and it costs no extra
# weights: the adapters are simply disabled for the teacher forward pass. That is
# what `teacher = None` selects above.
# 
# For a separate teacher, two things matter. Release the caching allocator's
# reserved-but-unused blocks first, and give the device-map planner an explicit
# budget. Left alone the planner does `free, _ = torch.cuda.mem_get_info(d)` and
# takes the whole card, leaving nothing for the transient buffers the load itself
# needs. On gemma-4 E2B from E4B it budgeted 8.94 GiB on `cuda:0` when 9.93 GiB was
# free, and by the time weights were being placed only 4.70 GiB remained.

# In[ ]:


if teacher_name is None:
    teacher = student
    print("self-distillation: teacher is the student's base with adapters disabled")
else:
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    TEACHER_HEADROOM_GIB = 2.0
    max_memory = {}
    for _d in range(torch.cuda.device_count()):
        _free, _total = torch.cuda.mem_get_info(_d)
        max_memory[_d] = max(int(_free - TEACHER_HEADROOM_GIB * 2**30), 0)
        print(f"cuda:{_d} free {_free/2**30:.2f} GiB of {_total/2**30:.2f} GiB"
              f" -> teacher budget {max_memory[_d]/2**30:.2f} GiB")

    teacher, teacher_processor = FastLanguageModel.from_pretrained(
        teacher_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
        max_memory = max_memory,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"teacher loaded, {sum(p.requires_grad for p in teacher.parameters())} trainable (must be 0)")


# Same vocabulary, or the loss is a shape error rather than a bad number.

# In[ ]:


if teacher_name is not None:
    s_vocab = student.config.get_text_config().vocab_size
    t_vocab = teacher.config.get_text_config().vocab_size
    assert s_vocab == t_vocab, f"vocab mismatch: student {s_vocab} vs teacher {t_vocab}"
    print(f"vocabularies match: {s_vocab}")


# ### Dataset
# 
# FineTome ships ShareGPT turns keyed `from`/`value`. The chat template expects
# `role`/`content`, and passing the raw rows through renders as
# `'dict object' has no attribute 'role'`.

# In[ ]:


from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:500]")
dataset = standardize_sharegpt(dataset)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])


# Now drop rows that do not fit in `max_seq_length` **as a whole**. This one is
# worth understanding, because the failure it prevents looks like a bug in the loss.
# 
# GKD's collator returns a `prompts` tensor alongside `input_ids`, and the loss
# slices with it:
# 
# ```python
# prompt_lengths = inputs["prompts"].shape[1]
# shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
# ```
# 
# When a row is longer than `max_length` the collator truncates it, and the prompt
# is what gets dropped. `prompts` comes back with width 0, so the slice becomes
# `logits[:, -1:-1, :]`, which is empty, while the labels stay full width. That is
# the `mask [1, 512] does not match ... tensor [1, 0, 262144]` failure.
# 
# Note this is about the **total** length, not the prompt's. A short prompt with a
# long answer still overflows and still loses its prompt, so filtering on the
# prompt alone does not help: on FineTome the median prompt is only 54 tokens while
# completions run into the thousands.

# In[ ]:


def _fits_in_the_budget(example):
    messages = example["messages"]
    if len(messages) < 2: return False
    whole = tokenizer.apply_chat_template(messages, tokenize = True)
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize = True, add_generation_prompt = True,
    )
    return len(whole) <= max_seq_length and len(prompt) < len(whole)

_before = len(dataset)
dataset = dataset.filter(_fits_in_the_budget)
print(f"kept {len(dataset)}/{_before} rows that fit in {max_seq_length} tokens")
assert len(dataset) > 0, (
    f"no row fits in max_seq_length = {max_seq_length}; raise it or use a "
    "dataset with shorter conversations"
)
print(dataset[0]["messages"][:2])


# ### Train
# 
# `lmbda`, `beta` and `temperature` are the distillation knobs; the rest is an
# ordinary TRL config.
# 
# No `fp16` or `bf16` is set here on purpose. Unsloth picks the precision per
# architecture, and overriding it breaks the ones it has already ruled out.

# In[ ]:


from trl import GKDConfig, GKDTrainer

config = GKDConfig(
    output_dir = "outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = 30,
    learning_rate = 2e-4,
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    report_to = "none",
    max_length = max_seq_length,
    lmbda = 0.0,        # 0 off-policy, 1 on-policy
    beta = 0.5,         # 0 forward KL, 1 reverse KL
    temperature = 1.0,
    max_new_tokens = 64,
)


# In[ ]:


trainer = GKDTrainer(
    model = student,
    teacher_model = teacher,
    args = config,
    train_dataset = dataset,
    processing_class = tokenizer,
)

assert type(trainer.args).__name__ == "GKDConfig", type(trainer.args).__name__
assert trainer.args.lmbda == 0.0 and trainer.args.beta == 0.5
print(f"args {type(trainer.args).__name__}: lmbda={trainer.args.lmbda} "
      f"beta={trainer.args.beta} temperature={trainer.args.temperature}")


# A config that subclasses `SFTConfig` used to be silently rebuilt as a plain one,
# dropping every distillation field, so the asserts above check rather than trust.

# In[ ]:


before = {n: p.detach().clone() for n, p in student.named_parameters() if p.requires_grad}
result = trainer.train()

changed = sum(1 for n, p in student.named_parameters()
              if p.requires_grad and not torch.equal(p.detach(), before[n]))
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\ntrain loss      : {result.training_loss}")
print(f"adapters changed: {changed}/{len(before)}")
print(f"peak memory     : {peak:.2f} GB")

history = [h["loss"] for h in trainer.state.log_history if "loss" in h]
if len(history) >= 2:
    print(f"loss first -> last: {history[0]:.4f} -> {history[-1]:.4f}")
assert changed > 0, "no adapter changed: the student did not learn"


# ### What this does and does not cover
# 
# `GKDTrainer` holds the teacher in memory alongside the student, so the teacher
# size is bounded by your GPU. Distilling a Deepseek- or Kimi-class teacher needs a
# served teacher over HTTP or cached top-k teacher logprobs prepared ahead of
# training; neither is wired into Unsloth yet.
# 
# Self-distillation (`teacher = None` above) is the cheapest real configuration:
# under LoRA the frozen base is already a perfectly good teacher, and it costs no
# additional weights.
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
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
