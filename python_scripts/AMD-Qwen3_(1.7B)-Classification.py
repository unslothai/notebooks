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
# get_ipython().run_cell_magic('bash', '', 'python -m pip install -qU uv --root-user-action=ignore\n\nROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F\'ROCm version: \' \'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { [ -r /opt/rocm/.info/version ] && awk -F. \'{print "rocm"$1"."$2; exit}\' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F\': *\' \'/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f=\'${Version}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf \'%{VERSION}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; })"\n[ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }\ncase "$ROCM_TAG" in\n  rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;\n  rocm6.*) T="rocm6.4" ;;\n  *) T="rocm7.1" ;;\nesac\npip install bitsandbytes\nPYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"\nuv pip install --system -U --force-reinstall \\\n    torch torchvision torchaudio triton-rocm \\\n    --index-url "$PYTORCH_INDEX_URL"\nuv pip install --system cut-cross-entropy torchao --no-deps\nuv pip install --system -U --no-deps "unsloth[amd]" "unsloth_zoo[amd]"\nuv pip install --system --no-deps -r "$(python -c \'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))\')" torchao\nuv pip install --system --no-deps -U "tokenizers>=0.22.0,<=0.23.0"\n')
# 
# 
# # In[ ]:
# 
# 
# get_ipython().system('uv pip install --system -qqq sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer "transformers==4.57.6"')
# get_ipython().system('uv pip install --system -qqq --no-deps accelerate peft "trl==0.22.2"')
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
# # ### Unsloth
# We load the data first, because the number of classes sets the size of the model's classification head.

# <a name="Data"></a>
# ### Data Prep
# Banking77 has about 10,000 training and 3,000 test questions, each labeled with one of 77 intents such as `card_arrival` or `lost_or_stolen_card`. The `label` column is already an integer, and the names live in the dataset's features.

# In[ ]:


from datasets import load_dataset

dataset = load_dataset("PolyAI/banking77")
label_names = dataset["train"].features["label"].names
num_labels = len(label_names)

id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

print(f"{num_labels} classes")
print(dataset["train"][0])


# **Using your own data?** Your labels only need to become integers `0 ... num_labels - 1`. If your label column holds strings (for example `"refund_request"` or `"category 317"`), `class_encode_column` builds the mapping for you. Encode the train split and reuse its mapping for the test split, so the ids still match when a class never shows up in the test file:
# 
# ```python
# dataset = load_dataset("csv", data_files = {"train": "train.csv", "test": "test.csv"})
# dataset["train"] = dataset["train"].class_encode_column("label")   # strings -> integer ids
# dataset["test"] = dataset["test"].cast_column("label", dataset["train"].features["label"])
# 
# label_names = dataset["train"].features["label"].names
# num_labels = len(label_names)
# id2label = {i: name for i, name in enumerate(label_names)}
# label2id = {name: i for i, name in enumerate(label_names)}
# ```
# 
# Everything below then works unchanged, whether you have 10 classes or 400.

# ### Load the model
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


# We now add LoRA adapters so we only need to update a small amount of parameters! The classification head starts from random weights, so it is trained in full alongside the adapters.

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


# Let's check the head really is trainable. Some older Unsloth versions left it frozen, and a frozen random head means the model never learns the classes, so we unfreeze it if needed.

# In[ ]:


head = {name: param for name, param in model.named_parameters() if ".modules_to_save." in name}
assert head, "No classification head found. Did you pass num_labels to from_pretrained?"
for name, param in head.items():
    if not param.requires_grad:
        print(f"Unfreezing {name}")
        param.data = param.data.float() # Train the head in float32, like Unsloth does.
        param.requires_grad_(True)
model.print_trainable_parameters()


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
# To save the final model, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save. The LoRA adapters are saved together with the trained classification head. The adapter files do not store the label names, so we save them next to it in `labels.json`. If you push to the Hub, upload `labels.json` too with `upload_file`, so the label names travel with the adapter.
# 
# **[NOTE]** This ONLY saves the LoRA adapters and the head, and not the full model.

# In[ ]:


import json

model.save_pretrained("qwen_lora")  # Local saving
tokenizer.save_pretrained("qwen_lora")
with open("qwen_lora/labels.json", "w") as f:
    json.dump(label_names, f)
# model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
# from huggingface_hub import upload_file
# upload_file(path_or_fileobj = "qwen_lora/labels.json", path_in_repo = "labels.json",
#             repo_id = "your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the classifier we just saved, change `False` to `True`. Passing the saved labels rebuilds the head with the same classes, so `classify` works as before.

# In[ ]:


if False:
    import json
    from unsloth import FastLanguageModel
    with open("qwen_lora/labels.json") as f:
        label_names = json.load(f)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "qwen_lora",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        num_labels = len(label_names),
        id2label = id2label,
        label2id = label2id,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    print(classify(questions))


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
