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

# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss faster with 32% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# You can now train embedding models 1.8-3.3x faster with 20% less VRAM. [Blog](https://unsloth.ai/docs/new/embedding-finetuning)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# 3x faster LLM training with 30% less VRAM and 500K context. [3x faster](https://unsloth.ai/docs/new/3x-faster-training-packing) • [500K Context](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth

# In[2]:


from unsloth import FastLanguageModel
import torch

# Models supported for Phone Deployment
fourbit_models = [
    "unsloth/Qwen3-4B",              # Any Qwen3 model like 0.6B, 4B, 8B, 32B
    "unsloth/Qwen3-32B",
    "unsloth/Llama-3.1-8B-Instruct", # Llama 3 models work
    "unsloth/Llama-3.3-70B-Instruct",
    "unsloth/gemma-3-270m-it",       # Gemma 3 models work
    "unsloth/gemma-3-27b-it",
    "unsloth/Qwen2.5-7B-Instruct",   # And more models!
    "unsloth/Phi-4-mini-instruct",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
    full_finetuning = True,
    qat_scheme = "int4", # Gemma3 needs int4 due to large vocab (262K)
)


# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:
# 
# ```
# <bos><start_of_turn>user
# Hello!<end_of_turn>
# <start_of_turn>model
# Hey there!<end_of_turn>
# ```
# 
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.

# In[3]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


# In[4]:


from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")


# We sample 10k rows to speed up training for the first successful phone deployment.

# In[5]:


dataset = dataset.shuffle(seed = 3407).select(range(10000))


# We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!

# In[6]:


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


# Let's see how row 100 looks like!

# In[7]:


dataset[100]


# We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one.

# In[8]:


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


# Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one.

# In[9]:


dataset[100]["text"]


# <a name="Train"></a>
# ### Train the model
# Fine-tuning requires careful experimentation. To avoid wasting hours on a broken pipeline, we start with a 5-step sanity check. This ensures the training stabilizes and the model exports correctly to your phone.
# 
# Run this short test first. If the export succeeds, come back and set max_steps = -1 (or num_train_epochs = 1) for the full training run.

# In[10]:


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        max_steps = 60,
        warmup_steps = 5,
        learning_rate = 5e-6,
        optim = "adamw_torch",
        max_grad_norm = 1.0,
        logging_steps = 1,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)


# In[11]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[12]:


trainer_stats = trainer.train()


# In[13]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_training = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
training_percentage = round(used_memory_for_training / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {training_percentage} %.")


# <a name="Save"></a>
# ### Saving, loading finetuned models
# 
# To save the model for phone deployment, we first save the model and tokenizer via `save_pretrained`.

# In[14]:


# Save the model and tokenizer directly
model.save_pretrained("gemma_phone_model")
tokenizer.save_pretrained("gemma_phone_model")


# We then export directly from the local folder using Optimum Executorch as per [the documentation.](https://github.com/huggingface/optimum-executorch/blob/main/optimum/exporters/executorch/README.md)

# In[15]:


get_ipython().run_cell_magic('file', 'export_gemma_model.py', 'from optimum.executorch import ExecuTorchModelForCausalLM\nimport shutil\nimport os\n\nprint("Exporting Gemma3-270M to ExecuTorch format...")\n\n# Export the trained model using Python API\net_model = ExecuTorchModelForCausalLM.from_pretrained(\n    "gemma_phone_model",\n    export = True,\n    recipe = "xnnpack",\n    task = "text-generation",\n)\n\n# Copy .pte file to output directory\ntemp_dir = et_model._temp_dir.name\nos.makedirs("gemma_output", exist_ok = True)\n\nfor f in os.listdir(temp_dir):\n    src = os.path.join(temp_dir, f)\n    dst = os.path.join("gemma_output", f)\n    shutil.copy2(src, dst)\n    size_mb = os.path.getsize(dst) / 1024 / 1024\n    print(f"Exported: {f} ({size_mb:.2f} MB)")\n\nprint("\\nExport complete!")\n')


# In[16]:


get_ipython().system('python export_gemma_model.py')


# And we have the file Gemma3 model.pte of size 306M!

# In[17]:


get_ipython().system('ls -lh gemma_output/model.pte')


# ### Test Inference on Exported Model

# In[18]:


get_ipython().run_cell_magic('file', 'test_executorch.py', 'from transformers import AutoTokenizer\nfrom optimum.executorch import ExecuTorchModelForCausalLM\n\n# Load the exported model for inference\net_model = ExecuTorchModelForCausalLM.from_pretrained("gemma_output", export = False)\ntokenizer = AutoTokenizer.from_pretrained("gemma_phone_model")\n\n# Test generation\nprompt = "<start_of_turn>user\\nWhat is 2 + 2?<end_of_turn>\\n<start_of_turn>model\\n"\noutput = et_model.text_generation(tokenizer, prompt, max_seq_len = 50)\nprint(f"Generated: {output}")\n')


# In[19]:


get_ipython().system('python test_executorch.py')


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
