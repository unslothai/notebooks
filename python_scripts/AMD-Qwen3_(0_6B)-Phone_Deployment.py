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

# Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)
# 
# <table><tr>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
# </tr></table>
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
# # In[1]:
# 
# 
# get_ipython().run_cell_magic('bash', '', 'python -m pip install -qU uv --root-user-action=ignore\n\nROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F\'ROCm version: \' \'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { [ -r /opt/rocm/.info/version ] && awk -F. \'{print "rocm"$1"."$2; exit}\' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F\': *\' \'/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f=\'${Version}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf \'%{VERSION}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; })"\n[ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }\ncase "$ROCM_TAG" in\n  rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;\n  rocm6.*) T="rocm6.4" ;;\n  *) T="rocm7.1" ;;\nesac\npip install bitsandbytes\nPYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"\nuv pip install --system -U --force-reinstall \\\n    torch torchvision torchaudio triton-rocm \\\n    --index-url "$PYTORCH_INDEX_URL"\nuv pip install --system cut-cross-entropy torchao --no-deps\nuv pip install --system -U --no-deps "unsloth[amd]" "unsloth_zoo[amd]"\nuv pip install --system --no-deps -r "$(python -c \'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))\')" torchao\nuv pip install --system --no-deps -U "tokenizers>=0.22.0,<=0.23.0"\n')
# 
# 
# # In[ ]:
# 
# 
# get_ipython().system('uv pip install --system -qqq sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer "transformers==4.57.3" "optimum==1.24.0" pytorch-tokenizers "executorch==1.1.0"')
# get_ipython().system('uv pip install --system -qqq --no-deps accelerate peft "trl==0.25.1" git+https://github.com/huggingface/optimum-executorch.git@v0.1.0')
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
    model_name = "unsloth/Qwen3-0.6B",
    max_seq_length = 1024,
    full_finetuning = True,
    qat_scheme = "phone-deployment", # Flag for phone deployment
)


# <a name="Data"></a>
# ### Data Prep
# Qwen3 has both reasoning and a non reasoning mode. So, we should use 2 datasets:
# 
# 1. We use the [Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) dataset which was used to win the [AIMO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard) (AI Mathematical Olympiad - Progress Prize 2) challenge! We sample 10% of verifiable reasoning traces that used DeepSeek R1, and which got > 95% accuracy.
# 
# 2. We also leverage [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. But we need to convert it to HuggingFace's normal multiturn format as well.

# In[3]:


from datasets import load_dataset
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")


# Let's see the structure of both datasets:

# In[4]:


reasoning_dataset


# In[5]:


non_reasoning_dataset


# We now convert the reasoning dataset into conversational format:

# In[6]:


def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }


# In[7]:


reasoning_conversations = tokenizer.apply_chat_template(
    list(reasoning_dataset.map(generate_conversation, batched = True)["conversations"]),
    tokenize = False,
)


# Let's see the first transformed row:

# In[8]:


reasoning_conversations[0]


# Next we take the non reasoning dataset and convert it to conversational format as well.
# 
# We have to use Unsloth's `standardize_sharegpt` function to fix up the format of the dataset first.

# In[9]:


from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(non_reasoning_dataset)

non_reasoning_conversations = tokenizer.apply_chat_template(
    list(dataset["conversations"]),
    tokenize = False,
)


# Let's see the first row

# In[10]:


non_reasoning_conversations[0]


# Now let's see how long both datasets are:

# In[11]:


print(len(reasoning_conversations))
print(len(non_reasoning_conversations))


# The non reasoning dataset is much longer. Let's assume we want the model to retain some reasoning capabilities, but we specifically want a chat model.
# 
# Let's define a ratio of chat only data. The goal is to define some mixture of both sets of data.
# 
# Let's select 75% reasoning and 25% chat based:

# In[12]:


chat_percentage = 0.25


# Let's sample the reasoning dataset by 75% (or whatever is 100% - chat_percentage)

# In[13]:


import pandas as pd
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
    random_state = 2407,
)
print(len(reasoning_conversations))
print(len(non_reasoning_subset))
print(len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations)))


# Finally combine both datasets:

# In[14]:


data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"

from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[15]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 100,
        learning_rate = 5e-5,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# In[16]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[17]:


trainer_stats = trainer.train()


# In[18]:


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


# <a name="Save"></a>
# ### Saving, loading finetuned models
# 
# To save the model for phone deployment, we first save the model via `save_pretrained_torchao`

# In[19]:


model.save_pretrained_torchao("phone_model", tokenizer = tokenizer)


# We then use Executorch's Qwen3 conversion process

# In[20]:


# Convert the weight checkpoint state dict keys to one that ExecuTorch expects
get_ipython().system('python -m executorch.examples.models.qwen3.convert_weights      "phone_model" pytorch_model_converted.bin')


# In[21]:


# Download model config from ExecuTorch repo
get_ipython().system('curl -L -o 0.6B_config.json https://raw.githubusercontent.com/pytorch/executorch/main/examples/models/qwen3/config/0_6b_config.json')


# And finally we export to a .pte file which can be used for deployment

# In[22]:


# Export to ExecuTorch pte file
get_ipython().system('python -m executorch.examples.models.llama.export_llama      --model "qwen3_0_6b"      --checkpoint pytorch_model_converted.bin      --params 0.6B_config.json      --output_name qwen3_0.6B_model.pte      -kv      --use_sdpa_with_kv_cache      -X      --xnnpack-extended-ops      --max_context_length 1024      --max_seq_length 128      --dtype fp32      --metadata \'{"get_bos_id":199999, "get_eos_ids":[200020,199999]}\'')


# And we have the file `qwen3_0.6B_model.pte` of around 472MB!

# In[23]:


get_ipython().system('ls -l -h qwen3_0.6B_model.pte')


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
