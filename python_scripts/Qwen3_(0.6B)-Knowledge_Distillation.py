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


get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==5.15.1\n!pip install --no-deps trl==0.22.2\n')


# In[ ]:


# GKD under Unsloth needs unslothai/unsloth#11440 and unslothai/unsloth-zoo#1310,
# which are on main but not yet in a PyPI release. Without them the generated
# trainer downcasts GKDConfig back to SFTConfig and silently drops lmbda, beta
# and temperature, so distillation degrades to plain SFT with no error
# (unslothai/unsloth#1941). Delete this cell once a release carries both.
get_ipython().system('pip install --no-deps --upgrade --force-reinstall      git+https://github.com/unslothai/unsloth-zoo git+https://github.com/unslothai/unsloth')


# ### Knowledge distillation with Unsloth
# 
# Distillation trains a small **student** to match the output distribution of a
# larger **teacher**, rather than to match one-hot labels. It usually beats plain
# finetuning at the same student size, because a full distribution carries far more
# signal per token than a single correct answer.
# 
# This notebook uses TRL's `GKDTrainer`, which Unsloth patches like any other TRL
# trainer. Three things are worth knowing before you change the models:
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

# In[ ]:


# Each entry is (student, teacher). A teacher equal to the student is
# self-distillation: the student learns from its own frozen base, which needs
# only ONE copy of the weights and so is the only option for the 27B and 30B
# models on two T4s.
#
# Weights on disk, and MEASURED peak memory for 6 steps at the settings below.
# GKD holds full (batch, seq, vocab) logits for BOTH models, so peak is driven by
# the vocabulary, not by the weights: Gemma-4's 262144-token vocabulary is what
# makes it expensive, not its size.
#
#   config        weights           Kaggle 2x T4 result
#   qwen3         1.2 + 3.4 GB      PASS, loss 0.413, 392/392 adapters, 10.9 GB
#   gemma-4       8.1 + 10.9 GB     needs a bigger card, see below
#   qwen3.8       22.3 GB           blocked on a dtype mismatch, see below
#   muse-glimmer  22.2 GB           tight, no sequence length measured to work
#
# gemma-4 loads fine on two T4s once unslothai/unsloth-zoo#1328 is in: both
# models go resident, the teacher is frozen and the trainer builds. It is
# sequence length that has no answer, because the two limits point opposite
# ways and the gap between them is too small to trust:
#
#   256, 384, 512   the prompt fills the budget, leaving GKD nothing to score
#                   ("mask [1, 512] does not match ... tensor [1, 0, 262144]")
#   768             the loss itself does not fit, short by 336 MiB
#
# GKD materialises a full (batch, seq, vocab) tensor for student, teacher and
# mixture, and at a 262144 vocabulary one of those in float32 is 805 MB at 768.
# A value between 512 and 768 might pass on this dataset and fail on a longer
# one, so this config wants a bigger card rather than a lucky number. The real
# fix is a chunked loss (unslothai/unsloth-zoo#1310), which never builds that
# tensor, and is not reachable from GKD yet.
#
# qwen3.8 is blocked on something else entirely: the first training step raises
# "expected mat1 and mat2 to have the same dtype, but got: BFloat16 != Half"
# inside the gated delta net. It is not memory, and it does not reproduce on a
# card that supports bfloat16.
#
# unsloth/gemma-4-26B-A4B-it is 51.6 GB with no prebuilt 4-bit, so it needs an
# A100 or better. Listed for that case, not for T4.
#
# On the version pins: the install cell puts transformers 5.15.1 together with
# trl 0.22.2 installed --no-deps, and both halves are deliberate. 5.15.1 is the
# floor that loads every architecture here (qwen3_5 landed in 5.15.1,
# muse_glimmer in 5.15.0, Gemma-4 in 5.10.1); the canonical 4.56.2 loads none of
# the last three. GKDTrainer is importable from the top level of trl 0.22.2
# through 0.28 and nowhere after, so the old trl is pinned on without its
# dependencies, which would otherwise drag transformers back below that floor.

CONFIGS = {
    "gemma-4": dict(
        student = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
        teacher = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    ),
    "gemma-4-big-teacher": dict(          # A100 / H100, not T4
        student = "unsloth/gemma-4-E4B-it",
        teacher = "unsloth/gemma-4-26B-A4B-it",
    ),
    "qwen3": dict(                        # smallest proven pair, fits 2x T4
        student = "unsloth/Qwen3-0.6B",
        teacher = "unsloth/Qwen3-1.7B",
    ),
    "qwen3.8": dict(
        student = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
        teacher = None,                   # self-distillation
    ),
    "muse-glimmer": dict(
        student = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
        teacher = None,                   # self-distillation
    ),
}

CONFIG = "qwen3"   # the pair proven to fit two T4s; see the table above
student_name = CONFIGS[CONFIG]["student"]
teacher_name = CONFIGS[CONFIG]["teacher"]

# Gemma-4 "E" checkpoints are elastic (MatFormer): execution is data dependent,
# so under torch.compile the gradient-checkpoint recompute can select a different
# compiled graph than the forward pass and training dies with
# "CheckpointError: Recomputed values ... have different metadata". Reproduced on
# E2B <- E4B; disabling compile for that family alone is the fix, and it costs
# nothing for every other config here.
import os
if CONFIG.startswith("gemma-4"):
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    print("gemma-4 is elastic: UNSLOTH_COMPILE_DISABLE=1 set for this run")

# Sequence length has a floor as well as a ceiling here. GKD scores the
# completion, so if the prompt fills the whole budget there are no completion
# tokens left and the loss indexes a zero-length axis: "mask [1, 384] does not
# match the shape of the indexed tensor [1, 0, 262144]". Seen at 256 on Muse
# Glimmer and at 384 on gemma-4 with FineTome. 1024 is comfortably clear of it.
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
    dtype = None,            # None auto-detects: float16 on T4, bfloat16 on Ampere+
    load_in_4bit = load_in_4bit,
)

# A `*ForConditionalGeneration` checkpoint returns a multimodal processor. The
# text-only trainer must be handed the tokenizer it wraps, or the processor tries
# to resolve the chat prompt as an image source and raises.
def text_tokenizer(maybe_processor):
    inner = getattr(maybe_processor, "tokenizer", None)
    if inner is not None and type(maybe_processor).__name__.endswith("Processor"):
        print(f"unwrapped {type(maybe_processor).__name__} -> {type(inner).__name__}")
        return inner
    return maybe_processor

tokenizer = text_tokenizer(processor)

# The distillation loss reads the output head directly, so it has to stay a dense
# [vocab, hidden] matrix. Unsloth keeps lm_head out of quantization for this
# reason; a checkpoint quantized elsewhere may not, so check rather than assume.
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
# Note lora_B is zero-initialized, so dL/dA is zero on the very first step and
# only the lora_B halves move. That is expected, not a sign nothing is learning.


# ### Load the teacher (or reuse the student's frozen base)

# In[ ]:


if teacher_name is None:
    # Self-distillation. Under LoRA the frozen base IS the teacher, so this costs
    # no extra weights: the adapters are disabled for the teacher forward pass.
    teacher = student
    print("self-distillation: teacher is the student's base with adapters disabled")
else:
    # Release the caching allocator's reserved-but-unused blocks before the
    # second model is placed. Unsloth budgets the teacher against what it sees
    # as free, and after the student has loaded and had LoRA attached, PyTorch
    # is still holding reserve it is not using. On gemma-4 E2B <- E4B across two
    # T4s the planner budgeted 8.94 GiB on cuda:0 and tried to put 5.253 GiB
    # there, while the card actually had 4.70 GiB free with 9.86 GiB in use
    # against only 4.445 GiB of student weights. Both cards were in use and the
    # weights fit; the gap was reserve, so hand it back first.
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Give the planner explicit budgets rather than letting it size the teacher
    # against every byte that looks free. Left alone it does
    # `free, _ = torch.cuda.mem_get_info(d); raw_budgets[d] = int(free)`, which
    # leaves nothing for the transient buffers the load itself needs: on
    # gemma-4 E2B <- E4B it budgeted 8.94 GiB on cuda:0 when 9.93 GiB was free,
    # and by the time the weights were being placed only 4.70 GiB remained.
    # Both cards were in use and the weights fit; the budget was simply the
    # whole card. `max_memory` reaches the planner through
    # planner_kwargs_with_max_memory, so hold some back on every card.
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
    n_trainable = sum(p.requires_grad for p in teacher.parameters())
    print(f"teacher loaded, {n_trainable} trainable parameters (must be 0)")

    # Same vocabulary, or the loss is a shape error rather than a bad number.
    s_vocab = student.config.get_text_config().vocab_size
    t_vocab = teacher.config.get_text_config().vocab_size
    assert s_vocab == t_vocab, f"vocab mismatch: student {s_vocab} vs teacher {t_vocab}"
    print(f"vocabularies match: {s_vocab}")


# ### Dataset

# In[ ]:


from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:500]")

# FineTome ships ShareGPT turns keyed `from`/`value`. The chat template expects
# `role`/`content`, and passing the raw rows through renders as
# "'dict object' has no attribute 'role'".
dataset = standardize_sharegpt(dataset)
dataset = dataset.rename_column("conversations", "messages")
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])

# Drop rows whose prompt leaves no room for a completion. GKD scores the
# completion, so if everything up to the last turn already fills max_length the
# completion truncates to nothing and the loss indexes a zero-length axis:
# "mask [1, 512] does not match the shape of the indexed tensor [1, 0, 262144]".
#
# This is a long tail, not a small budget. On FineTome the median prompt is 54
# tokens, but the 99th percentile is 2287 and the longest is 2767, so at 512
# about one row in ten has no room and a single one of them ends the run.
# Filtering is what makes a short sequence length usable at all; raising the
# budget does not fix it, since even 1024 leaves such rows behind.
MIN_COMPLETION_TOKENS = 64

def _leaves_room_for_a_completion(example):
    prompt = tokenizer.apply_chat_template(
        example["messages"][:-1], tokenize = True, add_generation_prompt = True,
    )
    return len(prompt) <= max_seq_length - MIN_COMPLETION_TOKENS

_before = len(dataset)
dataset = dataset.filter(_leaves_room_for_a_completion)
print(f"kept {len(dataset)}/{_before} rows with at least "
      f"{MIN_COMPLETION_TOKENS} tokens left for a completion")
assert len(dataset) > 0, "max_seq_length is too small for every prompt in this dataset"

print(dataset)
print(dataset[0]["messages"][:2])


# ### Train

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

    # Distillation knobs.
    lmbda = 0.0,        # 0 off-policy (score the dataset), 1 on-policy (student samples)
    beta = 0.5,         # 0 forward KL, 1 reverse KL, between is generalized JSD
    temperature = 1.0,  # distillation temperature, NOT the sampling temperature
    max_new_tokens = 64,

    # No fp16 / bf16 here on purpose: Unsloth picks the precision per
    # architecture, and overriding it breaks the ones it has already ruled out.
    # Qwen3.8 is such a case ("Using float16 precision for qwen3_5 won't work!
    # Using float32."). Forcing fp16 = True on a T4 anyway left the dense
    # lm_head at float16 while activations arrived as bfloat16, and training
    # died in a plain nn.Linear with "expected mat1 and mat2 to have the same
    # dtype, but got: c10::BFloat16 != c10::Half".
)

trainer = GKDTrainer(
    model = student,
    teacher_model = teacher,
    args = config,
    train_dataset = dataset,
    processing_class = tokenizer,
)

# Unsloth patches TRL's trainers, and a config that subclasses SFTConfig used to
# be silently rebuilt as a plain one, dropping every distillation field. Assert
# rather than trust it.
assert type(trainer.args).__name__ == "GKDConfig", type(trainer.args).__name__
assert trainer.args.lmbda == 0.0 and trainer.args.beta == 0.5
print(f"args {type(trainer.args).__name__}: lmbda={trainer.args.lmbda} "
      f"beta={trainer.args.beta} temperature={trainer.args.temperature}")


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
print("\nDISTILLATION VERDICT: PASS")


# ### What this does and does not cover
# 
# `GKDTrainer` holds the teacher in memory alongside the student, so the teacher
# size is bounded by your GPU, not by the chunked loss. Distilling a
# Deepseek- or Kimi-class teacher needs a served teacher over HTTP or cached top-k
# teacher logprobs prepared ahead of training; neither is wired into Unsloth yet.
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
