#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Synthetic Data Generation and Unsloth Tutorial</h1>

# ## Synthetic Data Generation
# 
# In this section, we use the CLI from synthetic-data-kit to generate datasets

# ### Converting to Fine-Tuning Format
# 
# This command uses the **save-as** function to convert curated Q&A pairs to fine-tuning format:
# - Reads curated JSON files from `data/curated/`
# - Converts to format `ft` (fine-tuning format with messages structure)
# - Outputs are saved to `data/final/` with proper conversation format
# - The resulting format is compatible with standard fine-tuning pipelines
# 
# Successfully converted 2 files to fine-tuning format.

# In[1]:


import json
import glob
from pathlib import Path
from datasets import Dataset

# ===== CONFIGURATION =====
data_dir = "./logical_reasoning/data/final"  # Change this to your data directory

# ===== STEP 1: Find all FT files =====
data_path = Path(data_dir)
ft_files = glob.glob(str(data_path / "*.json"))

# ===== STEP 2: Load and convert all files =====
all_data = []

for file_path in ft_files:
    # Load the JSON file
    with open(file_path, 'r') as f:
        ft_data = json.load(f)

    # Convert each item
    for item in ft_data:
        if 'messages' not in item:
            continue

        # Extract only user and assistant messages
        conversation = []
        for msg in item['messages']:
            if msg['role'] == 'user' or msg['role'] == 'assistant':
                conversation.append({
                    "role": msg['role'],
                    "content": msg['content']
                })

        # Add to our data if we have at least one exchange
        if len(conversation) > 0:
            all_data.append({
                "conversations": conversation
            })

print(f"\n🎯 Total conversations: {len(all_data)}")

# ===== STEP 3: Create HuggingFace Dataset =====
dataset = Dataset.from_list(all_data)

# ===== STEP 4: Preview the data =====
print(json.dumps(dataset[0], indent=2))


# ### Loading and Converting Data to HuggingFace Dataset
# 
# This cell performs comprehensive data processing:
# 
# 1. **Finding Files**: Locates all JSON files in `data/final/` directory
# 2. **Loading Data**: Reads each JSON file containing fine-tuning formatted data
# 3. **Format Conversion**: Extracts user and assistant messages from the fine-tuning format
# 4. **Structuring Conversations**: Creates a standardized conversation format with role-content pairs
# 5. **Creating Dataset**: Converts the processed data into a HuggingFace Dataset object
# 
# The output shows 74 total conversations were successfully loaded and formatted. The preview displays a sample conversation showing a knight-and-knave logic puzzle with its solution.

# ## Fine-Tuning
# 
# ### Note: Please remember to shutdown the vLLM instance!
# ### See https://docs.unsloth.ai/new/unsloth-amd-pytorch-synthetic-data-hackathon#how-do-i-free-amd-gpu-memory

# In[2]:


import os
import json
import glob
import torch
import shutil
from pathlib import Path
from datasets import Dataset


# ### Importing Standard Libraries
# 
# Imports essential Python libraries for fine-tuning:
# - `os`, `json`, `glob`: File system operations and JSON handling
# - `torch`: PyTorch deep learning framework
# - `shutil`: File operations
# - `Path`: Path manipulation
# - `Dataset`: HuggingFace datasets library for data handling

# In[3]:


from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq


# ### Importing Unsloth and Training Libraries
# 
# Imports specialized libraries for efficient fine-tuning:
# - `FastLanguageModel` from Unsloth: Optimized model loading and training
# - `get_chat_template`, `standardize_sharegpt`, `train_on_responses_only`: Chat formatting utilities
# - `SFTConfig`, `SFTTrainer`: Supervised fine-tuning configuration and trainer from TRL
# - `DataCollatorForSeq2Seq`: Handles batching and padding for sequence-to-sequence training

# ### Setup Unsloth model and tokenizer for ROCm without bitsandbytes

# In[4]:


max_seq_length = 1024
dtype = torch.bfloat16  # Explicit bfloat16 for ROCm
load_in_4bit = False  

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.3-70B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Explicit for ROCm
    trust_remote_code=True,
)

print(f"✅ Loaded: Llama-3.3-70B-Instruct (bfloat16, ROCm compatible)")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Higher rank for 70B model
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ### Loading Llama-3.3-70B Model with LoRA
# 
# This cell sets up the model for efficient fine-tuning on AMD ROCm hardware:
# 
# **Model Configuration:**
# - Model: Llama-3.3-70B-Instruct (70 billion parameters)
# - Data type: bfloat16 for ROCm compatibility
# - No quantization (load_in_4bit=False) to avoid bitsandbytes dependency
# - Max sequence length: 1024 tokens
# 
# **LoRA (Low-Rank Adaptation) Configuration:**
# - Rank (r): 64 - Higher rank for the large 70B model
# - Target modules: All attention and MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
# - LoRA alpha: 64
# - Dropout: 0 (no dropout)
# - Gradient checkpointing: "unsloth" for memory efficiency
# 
# LoRA enables efficient fine-tuning by only training small adapter layers instead of the entire 70B model, making it feasible to train on a single AMD MI300X GPU with 192GB HBM3 memory.

# In[5]:


"""Prepare dataset with proper chat template and tensor compatibility"""
print("🔧 Preparing dataset for training...")

# Set chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Formatting function that ensures proper tensor conversion
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []

    for convo in convos:
        # Ensure conversation is in correct format
        if isinstance(convo, list) and all(isinstance(msg, dict) for msg in convo):
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        else:
            print(f"⚠️  Skipping malformed conversation: {type(convo)}")
            continue

    return {"text": texts}

dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

print(f"✅ Prepared {len(dataset)} valid examples for training")

# Show sample
if len(dataset) > 0:
    print(f"📝 Sample formatted text:")
    print(dataset["text"][0][:200] + "...")


# ### Preparing Dataset with Chat Template
# 
# This cell formats the dataset for fine-tuning:
# 
# **Steps:**
# 1. **Set Chat Template**: Applies Llama-3.1 chat template formatting
# 2. **Configure Padding**: Sets pad token to eos token if not already set
# 3. **Format Conversations**: The `formatting_prompts_func` function:
#    - Takes raw conversations from the dataset
#    - Applies the chat template to format them properly
#    - Validates conversation structure (list of dicts with role/content)
#    - Filters out malformed conversations
# 4. **Standardize Format**: Uses `standardize_sharegpt` to normalize the data structure
# 5. **Apply Formatting**: Maps the formatting function across all examples
# 6. **Remove Empty**: Filters out any empty or invalid formatted texts
# 
# The output shows 74 valid examples were successfully prepared. A sample of the formatted text is displayed, showing the proper Llama-3.1 chat template structure with system, user, and assistant headers.

# In[6]:


"""Train model with ROCm-optimized settings"""
# Ensure tokenizer has proper padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Setup trainer with ROCm-friendly settings and proper data handling
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=64,  # 🚀 MI300X can handle this with 192GB HBM3!
        gradient_accumulation_steps=1,   # Effective batch size = 8*2 = 16
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_8bit",  # Pure torch optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="logical_reasoning_rocm_outputs",
        report_to="none",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=True,  # Remove unused columns to avoid tensor issues
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # Single worker for ROCm stability
    ),
)

# Train only on responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

FastLanguageModel.for_training(model)
trainer_stats = trainer.train()


trainer_stats = trainer.train()


# ### Training the Model with ROCm-Optimized Settings
# 
# This cell configures and executes the fine-tuning process:
# 
# **Training Configuration (SFTConfig):**
# - **Batch size**: 64 per device - leveraging the AMD MI300X's massive 192GB HBM3 memory
# - **Gradient accumulation**: 1 step
# - **Warmup**: 5 steps
# - **Epochs**: 1 full pass through the dataset
# - **Learning rate**: 1e-4
# - **Optimizer**: adamw_8bit for memory efficiency
# - **Precision**: bf16 (bfloat16) for ROCm
# - **Gradient checkpointing**: Enabled for memory efficiency
# 
# **Special Training Mode:**
# Uses `train_on_responses_only` to compute loss only on the assistant's responses, not on the user's questions. This focuses the model on learning to generate accurate answers rather than memorizing the input format.
# 
# **Key Features:**
# - DataCollatorForSeq2Seq handles variable-length sequences with proper padding
# - No packing to preserve conversation structure
# - Single dataloader worker for ROCm stability
# - Gradient checkpointing via Unsloth for memory optimization
# 
# The model is then trained on the 74 logical reasoning conversations.

# In[ ]:


"""Save the trained model"""
print("\n💾 SAVING ROCM-TRAINED MODEL")

# Save LoRA adapters
lora_path = "logical_reasoning_rocm_lora"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"✅ LoRA adapters saved to: {lora_path}")

# Save merged model
merged_path = "logical_reasoning_rocm_merged"
print("🔄 Saving merged model...")
model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
print(f"✅ Merged model saved to: {merged_path}")

print(f"\n🎉 ROCM MODEL READY!")


# ### Saving the Fine-Tuned Model
# 
# This cell saves the trained model in two formats:
# 
# 1. **LoRA Adapters** (`logical_reasoning_rocm_lora/`):
#    - Saves only the trained LoRA adapter weights (lightweight, ~few hundred MB)
#    - Can be loaded later with the base model
#    - Useful for sharing or deploying with the original base model
# 
# 2. **Merged Model** (`logical_reasoning_rocm_merged/`):
#    - Merges LoRA adapters back into the base model
#    - Creates a standalone model with all weights
#    - Saved in 16-bit precision for better quality
#    - Ready for immediate inference without loading adapters
# 
# Both formats include the tokenizer configuration. The merged model is production-ready and can be used directly for generating answers to logical reasoning questions.And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
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
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
# 
