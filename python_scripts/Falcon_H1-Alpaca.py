#!/usr/bin/env python
# coding: utf-8

# # Unsloth Training for Falcon H1
# 
# This Notebook has been authored by TII Falcon Team.
# For more details on Falcon H1 series of models :
# 1. [Official Page](https://tiiuae.github.io/Falcon-H1/)
# 2. [blogpost](https://falcon-lm.github.io/blog/falcon-h1/)
# 3. [Official github page ](https://github.com/tiiuae/Falcon-H1)
# 4. [hf collection](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df)

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# 
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)

# ### Installation

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Installs Unsloth, Xformers (Flash Attention) and all other packages!\n!pip install unsloth\n# Get latest Unsloth\n!pip uninstall unsloth -y\n')


# In[ ]:


get_ipython().system('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@main"')


# In[ ]:


get_ipython().system('pip install vllm')


# In[ ]:


get_ipython().system('pip install --force-reinstall git+https://github.com/huggingface/transformers.git')


# In[ ]:


get_ipython().system('pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d.git@main')


# In[ ]:


get_ipython().system('pip install --no-build-isolation git+https://github.com/state-spaces/mamba.git@main')


# In[ ]:


get_ipython().system('pip install git+https://github.com/unslothai/unsloth-zoo.git')


# In[ ]:


import unsloth
from unsloth import FastLanguageModel
import torch
import os
os.environ['TRITON_JIT_DISABLE_OPT'] = '1' # Likely the most critical change

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "tiiuae/Falcon-H1-0.5B-Instruct", # Choose any model from https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)


# In[ ]:


# Configure PEFT model
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"], #Mamba out_proj and conv1d layers should not be included here see https://github.com/huggingface/peft/pull/2562
    lora_alpha= 32,
    lora_dropout= 0.1,
    use_gradient_checkpointing=False,
    random_state=3407,
)


# In[ ]:


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# In[ ]:


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# ### Show current memory stats

# In[ ]:


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# # Training

# In[ ]:


import os
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
trainer_stats = trainer.train()


# ### Show final memory and time stats

# In[ ]:


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


# In[ ]:




