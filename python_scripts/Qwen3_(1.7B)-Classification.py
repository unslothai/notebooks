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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth\n    !unsloth install-kernels\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==4.57.6\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Classification with many labels
# # 
# # This notebook fine-tunes Qwen3 (1.7B) to sort text into **77 classes**, using the [Banking77](https://huggingface.co/datasets/PolyAI/banking77) customer-support intents.
# # 
# # Some classification recipes keep the language-model head and trim it down to a few tokens such as `"0"` to `"9"`, so each class has to be a single token. That stops working once you have more than about 10 classes, or labels that tokenize into several pieces.
# # 
# # Here we put a **classification head** on the model instead: one output per class, trained with cross-entropy. The class names never go through the tokenizer, so the same code works for 77, 400 or thousands of labels. Change `num_labels` and the label mapping, and nothing else.
# 
# # <a name="Data"></a>
# # ### Data Prep
# # Banking77 has about 10,000 training and 3,000 test questions, each labeled with one of 77 intents such as `card_arrival` or `lost_or_stolen_card`. The `label` column is already an integer, and the names live in the dataset's features.
# 
# # In[ ]:
# 
# 
# from datasets import load_dataset
# 
# dataset = load_dataset("PolyAI/banking77")
# label_names = dataset["train"].features["label"].names
# num_labels = len(label_names)
# 
# id2label = {i: name for i, name in enumerate(label_names)}
# label2id = {name: i for i, name in enumerate(label_names)}
# 
# print(f"{num_labels} classes")
# print(dataset["train"][0])
# 
# 
# # **Using your own data?** Your labels only need to become integers `0 ... num_labels - 1`. If your label column holds strings (for example `"refund_request"` or `"category 317"`), `class_encode_column` builds the mapping for you:
# # 
# # ```python
# # dataset = load_dataset("csv", data_files = {"train": "train.csv", "test": "test.csv"})
# # dataset = dataset.class_encode_column("label")   # strings -> integer ids, names kept in features
# # label_names = dataset["train"].features["label"].names
# # ```
# # 
# # Everything below then works unchanged, whether you have 10 classes or 400.
# 
# # ### Unsloth
# 
# Passing `num_labels` loads the model with a sequence-classification head (`AutoModelForSequenceClassification`) instead of the language-model head. The new head reads the hidden state of the last real token and outputs one score per class. `id2label` and `label2id` are saved in the config, so predictions come back as names.

# In[ ]:


from unsloth import FastLanguageModel
import torch

max_seq_length = 128 # Banking77 questions are short. Raise this for longer texts.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-1.7B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # The classification head itself stays in 16 bit.
    num_labels = num_labels,
    id2label = id2label,
    label2id = label2id,
)

# The head picks the last non-padding token, so the model needs to know the pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


# We now add LoRA adapters so we only need to update a small amount of parameters! Unsloth sees the classification head and makes it fully trainable alongside the adapters, since it starts from random weights.

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
)


# We tokenize the text and rename `label` to `labels`, the name the Hugging Face `Trainer` expects. Padding is done per batch by the data collator.

# In[ ]:


def tokenize(batch):
    return tokenizer(batch["text"], truncation = True, max_length = max_seq_length)

tokenized = dataset.map(tokenize, batched = True, remove_columns = ["text"])
tokenized = tokenized.rename_column("label", "labels")
train_dataset = tokenized["train"]
eval_dataset = tokenized["test"]


# With many classes, plain accuracy can hide classes the model never gets right, so we also track **macro F1**, which weighs every class equally.

# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis = -1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average = "macro"),
    }


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface `Trainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer). We train for 2 epochs over the 10,000 examples, which takes a few minutes on a free Tesla T4.

# In[ ]:


from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from unsloth import is_bfloat16_supported

trainer = Trainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = DataCollatorWithPadding(tokenizer),
    compute_metrics = compute_metrics,
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 64,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.05,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 25,
        eval_strategy = "epoch",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
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
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Eval"></a>
# ### Evaluation
# Accuracy and macro F1 on the 3,080 held-out test questions:

# In[ ]:


metrics = trainer.evaluate()
print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"Macro F1: {metrics['eval_macro_f1']:.4f}")


# <a name="Inference"></a>
# ### Inference
# Let's run the model! Softmax over the head's outputs gives a probability for every class, so we can show the top few guesses for each question.

# In[ ]:


model.eval()

def classify(texts, top_k = 3):
    inputs = tokenizer(texts, padding = True, truncation = True,
                       max_length = max_seq_length, return_tensors = "pt").to(model.device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits.float(), dim = -1)
    top = probs.topk(top_k, dim = -1)
    return [
        [(id2label[i.item()], round(p.item(), 3)) for p, i in zip(ps, ids)]
        for ps, ids in zip(top.values, top.indices)
    ]

questions = [
    "My card still hasn't shown up, it's been two weeks.",
    "I think someone stole my card, please block it!",
    "Why was I charged a fee for withdrawing cash?",
    "Can I top up my account with Apple Pay?",
]
for question, guesses in zip(questions, classify(questions)):
    print(question)
    for name, prob in guesses:
        print(f"    {prob:.3f}  {name}")


# <a name="Save"></a>
# ### Saving finetuned models
# To save the final model, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save. The LoRA adapters are saved together with the trained classification head.
# 
# **[NOTE]** This ONLY saves the LoRA adapters and the head, and not the full model.

# In[ ]:


model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


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
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
