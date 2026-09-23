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


# GKD needs unslothai/unsloth#11440 and unslothai/unsloth-zoo#1310, on main but
# not yet released. Without them `GKDConfig` is downcast to `SFTConfig` and
# distillation silently becomes plain SFT. Delete the next cell once released.

# In[ ]:


get_ipython().system('pip install --no-deps --upgrade --force-reinstall      git+https://github.com/unslothai/unsloth-zoo git+https://github.com/unslothai/unsloth')


# ### Knowledge distillation with Unsloth
# 
# Train a small **student** to match a larger **teacher**'s output distribution
# rather than one-hot labels, which carries far more signal per token. Uses TRL's
# `GKDTrainer`, which Unsloth patches like any other TRL trainer.

# Before changing the models: **student and teacher must share a vocabulary**.
# **`lmbda`** picks off-policy (`0`) or on-policy (`1`, the student scores its own
# samples); **`beta`** picks the divergence, 0 forward KL to 1 reverse KL.

# ### Pick a configuration
# 
# GKD holds full `(batch, seq, vocab)` logits for **both** models, so peak memory
# follows the vocabulary, not the weights.
# 
# | config | weights | measured on a Kaggle 2x T4 |
# | --- | --- | --- |
# | `qwen3` | 1.2 + 3.4 GB | PASS, loss 0.413, 392/392 adapters, 10.9 GB |
# | `gemma-4` | 8.1 + 10.9 GB | PASS, loss 0.079, 410/410 adapters, 14.3 GB |
# | `muse-glimmer` | 22.2 GB | self-distillation, OOM by 208 MiB at seq 384 |
# | `qwen3.8` | 22.3 GB | self-distillation, blocked on a dtype mismatch |
# 
# `unsloth/gemma-4-26B-A4B-it` needs an A100 or better, so add it yourself.

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


# Three notes.
# 
# **`qwen3.8` and `muse-glimmer` self-distil**, having no smaller sibling, which
# only helps from an adapter you already trained: a fresh one equals its own base.
# 
# **gemma-4 needs `torch.compile` off on T4-class cards**, where the
# gradient-checkpoint recompute disagrees with the forward about tensor rank.
# 
# **qwen3.8 is blocked on T4-class cards** by a `BFloat16 != Half` mismatch in
# the gated delta net. It does not reproduce where bfloat16 is native.

# In[ ]:


import os
if CONFIG.startswith("gemma-4"):
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

max_seq_length = 1024
load_in_4bit = True
print(f"student: {student_name}")
# Bound first rather than inlined: an f-string expression cannot contain a
# backslash before Python 3.12, and the apostrophe needs one.
teacher_label = teacher_name or "(self-distillation: the student's own frozen base)"
print(f"teacher: {teacher_label}")


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


# A `*ForConditionalGeneration` checkpoint returns a processor. Hand the trainer
# the tokenizer it wraps, or it reads the chat prompt as an image and raises.

# In[ ]:


def text_tokenizer(maybe_processor):
    inner = getattr(maybe_processor, "tokenizer", None)
    if inner is not None and type(maybe_processor).__name__.endswith("Processor"):
        print(f"unwrapped {type(maybe_processor).__name__} -> {type(inner).__name__}")
        return inner
    return maybe_processor

tokenizer = text_tokenizer(processor)


# The loss reads the output head directly, so it must stay dense. Unsloth keeps
# `lm_head` unquantized; a checkpoint quantized elsewhere may not.

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


# `lora_B` is zero-initialized, so only the `lora_B` halves move on the first
# step. That is expected.

# ### Load the teacher, or reuse the student's frozen base

# `teacher = None` uses the frozen base for free, but the adapters must be
# disabled explicitly: `GKDTrainer` only calls `self.teacher_model(...)`, so the
# student passed directly gives two identical forwards. The wrapper keeps the
# student out of its submodules, since the trainer calls `teacher.eval()` each step.
# 
# A separate teacher needs a device-map budget, or the planner takes the whole card
# and leaves nothing for the load's own buffers.

# In[ ]:


class FrozenBaseTeacher(torch.nn.Module):
    """The student with its adapters switched off for the duration of the call."""
    def __init__(self, peft_student):
        super().__init__()
        object.__setattr__(self, "student", peft_student)

    @property
    def config(self):
        return self.student.config

    def forward(self, *args, **kwargs):
        with self.student.disable_adapter():
            return self.student(*args, **kwargs)


# In[ ]:


teacher_processor = None    # only a separate teacher brings its own

if teacher_name is None:
    teacher = FrozenBaseTeacher(student)
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


# The teacher has to disagree with the student. A gap of zero gives a loss near
# `1e-4`, which reads as converged rather than broken, so measure it.

# In[ ]:


_ids = tokenizer("The capital of France is", return_tensors = "pt").input_ids.to(student.device)
with torch.no_grad():
    _gap = (student(input_ids = _ids).logits - teacher(input_ids = _ids).logits).abs().max().item()
print(f"max teacher-student logit gap: {_gap:.3f}")
if _gap == 0:
    print("WARNING: the teacher matches the student exactly, so the divergence and its\n"
          "gradient are both zero. Expected for self-distillation from a fresh adapter;\n"
          "load an adapter you trained earlier to give this mode something to learn.")


# Equal width is not enough: Qwen2.5-0.5B and 7B are both 151936 wide but assign
# different ids, so compare what the tokenizers produce.

# In[ ]:


if teacher_name is not None:
    s_vocab = student.config.get_text_config().vocab_size
    t_vocab = teacher.config.get_text_config().vocab_size
    assert s_vocab == t_vocab, f"vocab width mismatch: student {s_vocab} vs teacher {t_vocab}"

    probe = "The quick brown fox jumps over 0123456789 lazy dogs, naively."
    teacher_tokenizer = text_tokenizer(teacher_processor)
    assert tokenizer(probe).input_ids == teacher_tokenizer(probe).input_ids, (
        "same vocabulary width but different token ids, so the two models do not "
        "share a vocabulary and GKD cannot compare their distributions")
    print(f"vocabularies match: {s_vocab}, and the ids agree")


# ### Dataset
# 
# FineTome ships ShareGPT turns keyed `from`/`value`; the chat template expects
# `role`/`content` and otherwise raises `'dict object' has no attribute 'role'`.

# In[ ]:


from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:500]")
dataset = standardize_sharegpt(dataset)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])


# Drop rows that do not fit `max_seq_length` **as a whole**. The collator
# truncates an overlong row from the front, so `prompts` comes back empty and the
# loss slices to nothing: `mask [1, 512] does not match ... tensor [1, 0, 262144]`.

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
# ordinary TRL config. No `fp16`/`bf16` on purpose: Unsloth picks precision per
# architecture.

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


# A config subclassing `SFTConfig` used to be silently rebuilt as a plain one,
# dropping every distillation field, hence the asserts.

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
# `GKDTrainer` holds the teacher alongside the student, so your GPU bounds the
# teacher size. A Deepseek- or Kimi-class teacher needs a served teacher or cached
# top-k logprobs, neither wired into Unsloth yet.
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
