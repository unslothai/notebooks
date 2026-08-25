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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==5.15.1\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# ### What you are about to load
# 
# Qwen3.8-27B is a `qwen3_5` checkpoint: 27.8B parameters, 64 layers, a vision tower and a
# 248320 token vocabulary. Three layers in four are linear attention (a gated delta net),
# which keeps a fixed size recurrent state instead of a growing KV cache, and that is what
# makes a 27B model tractable here.
# 
# **Two Tesla T4s, not one.** One T4 cannot hold this in 4-bit; Kaggle gives you two. Do
# not pass `device_map` - unsloth's `"sequential"` default fills one card then the other.
# `"balanced"` looks right and is not: it caps cuda:0 below cuda:1, leaves the 2.37 GiB
# `lm_head` without a home, and bitsandbytes then refuses the CPU entry.

# In[ ]:


MODEL_NAME = "unsloth/Qwen3.8-27B"  # unsloth resolves this to its 4-bit build


# In[ ]:


import torch
from unsloth.models.mapper import FLOAT_TO_INT_MAPPER

assert torch.cuda.device_count() >= 2, (
    "Qwen3.8-27B in 4-bit needs more than one 16 GB card. On Kaggle pick the "
    "T4 x2 accelerator (Settings -> Accelerator -> GPU T4 x2)."
)

# unslothai/unsloth#9682 relocates MODEL_NAME to the 4-bit repo. On a release that
# predates it you would silently get the ~52 GB bf16 one, so name it ourselves. Only
# the default is rewritten, so pointing MODEL_NAME somewhere else still works.
if MODEL_NAME == "unsloth/Qwen3.8-27B" and MODEL_NAME not in FLOAT_TO_INT_MAPPER:
    MODEL_NAME = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
print("loading from:", MODEL_NAME)


# In[ ]:


# Before unslothai/unsloth-zoo#1100, the forced-float32 pass called empty_cache()
# once per module, which frees blocks on every card while cudaFree only synchronises
# the current one - an illegal memory access on a model split across two GPUs.
# Self-disabling once that fix is installed.
import inspect
import torch
import unsloth.models.vision as _uv
import unsloth_zoo.patching_utils as _pu

if ("torch.cuda.synchronize" in inspect.getsource(_pu.patch_model_and_tokenizer)
        or torch.cuda.device_count() < 2):
    print("multi-GPU load workaround not needed")
else:
    _orig_empty, _orig_patch = torch.cuda.empty_cache, _uv.patch_model_and_tokenizer

    def _synchronised_empty_cache():
        for _i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(_i)
        return _orig_empty()

    def _patched(*args, **kwargs):
        torch.cuda.empty_cache = _synchronised_empty_cache
        try:
            return _orig_patch(*args, **kwargs)
        finally:
            torch.cuda.empty_cache = _orig_empty

    # vision.py imported the name by value, so it has to be replaced there.
    _uv.patch_model_and_tokenizer = _patched
    print("multi-GPU load workaround installed")


# In[ ]:


from unsloth import FastModel

max_seq_length = 1024  # Qwen3.8 goes to 262144, but 1024 is what two T4s train on

model, tokenizer = FastModel.from_pretrained(
    model_name        = MODEL_NAME,
    max_seq_length    = max_seq_length,
    load_in_4bit      = True,
    full_finetuning   = False,
)
print(f"placed over {len(set(model.hf_device_map.values()))} devices")


# Do not pass `dtype = torch.float16`, and do not set `fp16 = True` on the trainer below.
# The gated delta net produces NaN gradients in pure float16, so Unsloth keeps this
# architecture on a float32 autocast path and picks the loading dtype itself. The heavy
# matmuls still use the T4's float16 tensor cores either way.

# We now add LoRA adapters so we only need to update a small amount of parameters.
# 
# They land on every linear in the language model, the gated delta net included: 496
# modules, 58.4M trainable parameters, 0.21% of the model. The vision tower stays frozen
# because this is a text finetune.

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
# We do 30 steps to keep this quick; set `num_train_epochs = 1` and `max_steps = None` for
# a full run.
# 
# `per_device_train_batch_size = 1` with `gradient_accumulation_steps = 4` is deliberate.
# A 248320 token vocabulary makes the logits the largest allocation in the step, and they
# land on whichever card holds `lm_head`, which is the one with least room. Raising the
# batch size is the fastest way to run out of memory here.

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

trainer = train_on_responses_only(trainer)


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
# With the settings above: 18.80 GiB of weights resident, 1.23 GiB of activations,
# gradients and optimiser at the peak, and 23.71 GiB peak **reserved** - the number that
# OOMs you, since it includes fragmentation. Over two T4s that is 7.08 GiB on the first
# card and 11.36 GiB on the second, which is the tight one because it carries `lm_head`.
# 
# If you are close to the limit, lower `max_seq_length` or turn off `finetune_mlp_modules`.
# 
# These were measured on one large card with the allocator capped to what two T4s add up
# to, so the memory transfers but the step time was not measured on a T4.

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
