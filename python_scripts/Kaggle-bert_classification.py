#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth your local device, follow [our guide](https://docs.unsloth.ai/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# Introducing FP8 precision training for faster RL inference. [Read Blog](https://docs.unsloth.ai/new/fp8-reinforcement-learning).
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker).
# 
# [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) is now supported with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which creates kernels!
# 
# Introducing [Vision](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) and [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) for RL! Train Qwen, Gemma etc. VLMs with GSPO - even faster with less VRAM.
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# In[1]:


from unsloth import FastModel
from transformers import AutoModelForSequenceClassification
import torch

# Disable fast generation for bert!
get_ipython().run_line_magic('env', 'UNSLOTH_DISABLE_FAST_GENERATION = 1')

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger",4: "fear",5: "surprise"}

label2id = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}

model, tokenizer = FastModel.from_pretrained(
    model_name = "answerdotai/ModernBERT-large",
    auto_model = AutoModelForSequenceClassification,
    max_seq_length = max_seq_length,
    dtype = dtype,
    num_labels  = 6,
    full_finetuning = True,
    id2label=id2label,
    label2id=label2id,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# We now add LoRA adapters so we only need to update a small amount of parameters!
# 

# In[2]:


model = FastModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
      target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # rsLoRA: For correct scaling, set alpha ~ sqrt(r) if True.
    loftq_config = None,  # We support LoftQ
    task_type="SEQ_CLS",
)


# <a name="Data"></a>
# ### Data Prep  
# We now use the [Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) from `dair-ai`, which contains text labeled by emotion. In this example, we load the **unsplit** version and use only the first 30,000 samples.  
# 
# We then split the dataset into training (80%) and validation (20%), and apply tokenization to prepare the text for training.
# 

# In[3]:


from datasets import load_dataset

# Load the IMDB dataset
dataset = load_dataset("dair-ai/emotion","unsplit",split='train[:30000]')

# Split into training and validation sets
dataset = dataset.train_test_split(test_size=0.2)

def tokenize_function(examples):
    return tokenizer(examples["text"])

# Apply the tokenizer to the dataset
train_dataset = dataset['train'].map(tokenize_function, batched=True)
val_dataset = dataset["test"].map(tokenize_function, batched=True)


# 
# We compute **class weights** using scikit-learn’s ```compute_class_weight```.  
# This is useful when training on datasets where certain classes are underrepresented, ensuring the model does not become biased towards majority labels.

# In[4]:


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels = train_dataset["label"]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)


# In[5]:


# We rename the dataset column from **`label`** to **`labels`**, since this is the expected field name for Hugging Face `Trainer`.
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")


# In[6]:


class_weights


# 
# We define a `compute_metrics` function to evaluate the model during training.  
# Here we use **accuracy** from scikit-learn, which compares predicted labels with the ground truth.  
# 
# **[NOTE]** Accuracy is a good baseline, but for imbalanced datasets you may also want to track metrics like **F1-score**, **precision**, or **recall**.  
# 

# In[7]:


from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface  `Trainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer). We train for one full epoch (num_train_epochs=1) to get a meaningful result.

# In[8]:


from transformers import TrainingArguments,Trainer
from unsloth import is_bfloat16_supported

trainer = Trainer(
    model = model,
    processing_class = tokenizer,
    eval_dataset = val_dataset,
    train_dataset = train_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 1, # bert-style models usually need more than 1 epoch
        # max_steps = 60,
        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        eval_strategy="steps",
        eval_steps=0.10,  # Evaluate every 10% of total training steps
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
    compute_metrics = compute_metrics,
)


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[9]:


trainer_stats = trainer.train()


# <a name="Inference"></a>
# ### Inference
# Let's run the model !

# In[45]:


from transformers import pipeline

sentence1 = "We just finished training ModernBERT with Unsloth and its amazing!"

classifier = pipeline("sentiment-analysis", model=model,tokenizer=tokenizer)

classifier(sentence1)


# <a name="Save"></a>
# ### Saving finetuned models
# To save the final model, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 

# In[15]:


model.save_pretrained("model")  # Local saving
tokenizer.save_pretrained("model")
# model.push_to_hub("your_name/model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/model", token = "...") # Online saving


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
