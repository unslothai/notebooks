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
# get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):\n    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"\n    except: _numpy = "numpy"; _pil = "pillow"\n    !uv pip install -qqq \\\n        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes xformers==0.0.32.post2 \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth"\n    !uv pip install -qqq --no-deps "torchcodec==0.7.0"\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth\n!uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo\n!uv pip install transformers==5.2.0\n# Unsloth bundles the gated delta net kernels; a leftover pip fla would shadow them\n!uv pip uninstall -qqq flash-linear-attention fla-core\n# Prebuilt causal_conv1d when one exists for this torch, otherwise transformers\' torch conv1d (no 10 minute build)\nimport sys, torch; _t = ".".join(torch.__version__.split(".")[:2]); _cu = (torch.version.cuda or "0").split(".")[0]; _py = f"cp{sys.version_info[0]}{sys.version_info[1]}"; _abi = str(torch.compiled_with_cxx11_abi()).upper()\n!uv pip install -qqq "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu{_cu}torch{_t}cxx11abi{_abi}-{_py}-{_py}-linux_x86_64.whl" || echo "No prebuilt causal_conv1d for torch {_t}, using the torch fallback"\n!uv pip install --no-deps --upgrade "torchao>=0.16.0"\n')
# 
# 
# # ### Unsloth
# 
# A decision model doesn't write text. It reads an input, looks at the options you give it, and picks one with a probability. `FastDecisionModel` turns an LLM into one: it adds a small head that reads every question about an input in one pass and scores each option.
# 
# Change `model_name` to `unsloth/Llama-3.2-3B-Instruct` or `unsloth/gemma-4-E4B-it` to try other models.

# In[ ]:


from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
import torch

model, tokenizer = FastDecisionModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-4B",
    max_seq_length = 2048, # Longest input. Long inputs keep the question and options.
    load_in_4bit = True, # 4 bit quantization to reduce memory
)


# We now add LoRA adapters so we only need to update a small amount of parameters! The new decision head always trains.

# In[ ]:


model = FastDecisionModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
)


# <a name="Data"></a>
# ### Data Prep
# We use the [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) dataset. Each row has a `state` (the input), the `questions` to decide about it, and the `gold` answers. There are three kinds of questions:
# 
# * `choice`: pick one option, like which team should handle a ticket.
# * `noul`: yes or no.
# * `score`: pick a level, like how urgent something is.
# 
# Let's see how row 0 looks like!

# In[ ]:


from datasets import load_dataset
import json

dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split = "train")
print(dataset[0]["state"][:500])
print(json.dumps(json.loads(dataset[0]["questions"]), indent = 2)[:1000])
print(dataset[0]["gold"])


# `build_dataset` turns every row into one example holding all of its questions, and tells you how many decisions it skipped and why. To use your own data, give it a list of rows with the same `state`, `questions` and `gold` fields.
# 
# `split_holdout` keeps some rows out of training (80 rows, 400 decisions here), so we can check accuracy and calibrate the model later.

# In[ ]:


items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions")

train_items, eval_items = FastDecisionModel.split_holdout(items, seed = 3407)
print(f"{len(train_items)} training rows, {len(eval_items)} held out")


# Let's check accuracy before training. The head is new, so it's about the same as guessing (about 30% accuracy in our runs).

# In[ ]:


FastDecisionModel.evaluate(model, tokenizer, eval_items)


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps (about 1.7 epochs) to speed things up, which reached 75% to 78% test accuracy in our runs and took about 2 hours on a free T4 GPU (12 minutes on an RTX PRO 6000). The T4 has no bfloat16 and Qwen3.5 gives NaNs in pure float16, so Unsloth trains it in float32 there, which is why the T4 is slow. For the full run, set `num_train_epochs = 2` and remove `max_steps`: that is 70 steps here, and it reached 78% test accuracy in 25 minutes on an A100.

# In[ ]:


from transformers import TrainingArguments

trainer = DecisionTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_items,
    eval_dataset = eval_items,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 10,
        # num_train_epochs = 2, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        lr_scheduler_type = "cosine",
        weight_decay = 0.01,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        eval_strategy = "epoch",
        logging_steps = 10,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
        seed = 3407,
    ),
)


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
# First we calibrate the model on the held-out rows. Calibration adjusts the probabilities, so an answer given with 90% confidence is right about 90% of the time. `ece` is the calibration error, lower is better. After 60 steps we got 80% to 84% held-out accuracy and an `ece` of 0.02 to 0.06.

# In[ ]:


FastDecisionModel.calibrate(model, tokenizer, eval_items)


# Let's check accuracy on the dataset's test split, which the model never saw. We got 75% to 78% after 60 steps:

# In[ ]:


test = load_dataset("LocalLLaMA/typed-decisions", "all", split = "test")
test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
FastDecisionModel.evaluate(model, tokenizer, test_items)


# Now let's make some decisions! Give `predict` an input and your questions. `answer` is the option for `choice`, `True` or `False` for `noul`, and the level number for `score`.

# In[ ]:


FastDecisionModel.for_inference(model)
answers = FastDecisionModel.predict(
    model,
    tokenizer,
    "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, errors",
                "sales": "pricing, new plans",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "today"],
        },
    },
)
for name, result in answers.items():
    print(name, result["answer"], {k: round(v, 3) for k, v in result["probabilities"].items()})


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model, use `save_pretrained` for a local save or `push_to_hub` for an online save. This saves the LoRA adapters, the decision head and its calibration.

# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the model we just saved, set `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastDecisionModel
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name = "qwen_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )


# ### Saving to float16
# 
# We also support saving to `float16` directly with `save_pretrained_merged`. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("qwen_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/qwen_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# To train decision models on your own data, read our [guide](https://unsloth.ai/docs/models/decision-model-training).
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
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
