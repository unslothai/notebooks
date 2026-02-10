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

# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
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
# # In[ ]:
# 
# 
# get_ipython().system('pip install -qU llama-index llama-index-packs-raft-dataset')
# 
# 
# # ### Unsloth

# ### Retrieval Augmented Finetuning (RAFT) Cookbook Recipe!
# This cookbook aims to show how to use Unsloth to use retrieval augmented finetuning (RAFT). Supervised finetuning is like a closed-book examination where we encode knowledge from the training dataset into the LLM during finetuning, and then test it on unseen examples in the "exam".
# 
# RAFT differs from this in that it is an open-book exam format of finetuning! We allow the LLM to see not just the question and answer (in chain-of-thought format), but also the contexts. The hope is that the LLM will be able to acquire the domain knowledge, but also an improved ability to synthesize answers from context.
# 
# > Reference: [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131)

# ### Code Setup 

# First, let's setting up the OPENAI API KEY so that we can use the OpenAI LLMs. 

# In[ ]:


import os

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


# Next, we'll set up LlamaIndex. This involves configuring the language model (LLM) and embedding model that LlamaIndex will use. We'll be using OpenAI's `gpt-4o` as our LLM and `text-embedding-ada-002` as our embedding model.

# In[ ]:


from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model = "gpt-4o")
Settings.embed_model = OpenAIEmbedding(model = "text-embedding-ada-002")


# ### Ingest documents 

# We'll use the following code to download a research paper and then load it using `SimpleDirectoryReader`. This will be the data we use for our retrieval augmented finetuning.

# In[ ]:


get_ipython().system('mkdir  -p ../data')
get_ipython().system('wget "https://arxiv.org/pdf/2405.00247.pdf" -O "../data/non_traditional_credentials.pdf"')

docs = SimpleDirectoryReader("../data/").load_data(show_progress = True)


# ### Retrieval Augmented Finetuning

# ### Getting the RAFT dataset
# LlamaIndex has very kindly adapted the source code of the RAFT repository and made it even easier to generate your own RAFT dataset. Just point it to your filepath.t
# > Reference: [RAFTDatasetPack](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb)

# In[14]:


from llama_index.packs.raft_dataset import RAFTDatasetPack

raft_dataset = RAFTDatasetPack(
    file_path = "../data/non_traditional_credentials.pdf",
    llm = Settings.llm,
    embed_model = Settings.embed_model
)


# This cell takes quite long to run! Go have a coffee ☕
# > It took 19 minutes for the cell to finish running

# In[15]:


dataset = raft_dataset.run()


# Let's have a look!

# In[18]:


import pandas as pd
df = pd.DataFrame(dataset)
df.head()


# In[24]:


from IPython.display import display, Markdown

display(Markdown(df.iloc[0]['instruction']))


# In[27]:


display(Markdown(df.iloc[0]['oracle_context']))


# In[16]:


# Save as .jsonl format
dataset.to_json("raft_train.jsonl")


# ### Training the LLM
# Our dataset is a HuggingFace `Dataset` object, so we can leverage the abstraction's advantage to do a train-test split

# In[19]:


splits = dataset.train_test_split(test_size = 0.1)
train_ds = splits["train"]
eval_ds  = splits["test"]


# In[20]:


train_ds, eval_ds


# ### Now let's get the model!

# In[ ]:


from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, 
    full_finetuning = False, 
)


# In[22]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 2025,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# ## Formatting the prompts
# We need to put everything together into a single 'text' field for the LLM to be trained on. According to the [RAFT paper](https://arxiv.org/abs/2403.10131), we add the context along with the question and chain-of-thought answer in a bid to help our LLM learn how to use the context to answer the question. Let's do that!

# In[25]:


def formatting_prompts_func(examples):
    """Define a formatter that injects the retrieved context:"""

    texts = []
    for qn, ctx, oracle, instr, ans in zip(
        examples['question'],
        examples["context"],
        examples["oracle_context"],
        examples["instruction"],
        examples["cot_answer"]
    ):
        # You can choose to use `oracle_context` (gold) vs. `context` (retrieved)
        # Here we show both, but you could just use `context`.
        prompt = (
            "### Question:\n"
            f"{qn}\n\n"
            "### Context:\n"
            f"{ctx}\n\n"
            "### (Oracle Passages):\n"
            f"{oracle}\n\n"
            "### Instruction:\n"
            f"{instr}\n\n"
            "### Answer:\n"
        )
        # Append the gold answer plus EOS
        texts.append(prompt + ans + tokenizer.eos_token)
    return {"text": texts}

# then:
train_ds = train_ds.map(formatting_prompts_func, batched = True)
eval_ds = eval_ds.map(formatting_prompts_func, batched = True)


# Let's take a look at what we just did!

# In[26]:


from IPython.display import display, Markdown

display(Markdown(pd.DataFrame(train_ds).head()['text'].iloc[0]))


# ### And now we finally get to training!

# In[28]:


from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir = "llama32_1bn_raft_v2", #This will also be used as your huggingfacehub model id name
    report_to = "wandb", #Leave this to be blank if you don't want to use wandb
    run_name = "RAFT_SFT_Take7",
    eval_steps = 5,
    eval_strategy = "steps",
    per_device_train_batch_size = 1,    # small batches if quantized
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 2e-5,
    num_train_epochs = 5,
    # max_steps = 60,                    # or set num_train_epochs
    save_strategy = "no",
    gradient_checkpointing = True,
    logging_strategy = "steps",
    logging_steps = 5,
    seed = 42,
    optim = "adamw_torch",
    lr_scheduler_type = "cosine",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = eval_ds, 
    args = training_args,
    dataset_text_field = "text",

)


# Current memory statistics

# In[29]:


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[30]:


trainer_stats = trainer.train()


# Used memory statistics

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


# <a name="Save"></a>
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("llama_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("HF_USERNAME/llama_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("llama_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/llama_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("llama_lora")
    tokenizer.save_pretrained("llama_lora")
if False:
    model.push_to_hub("HF_USERNAME/llama_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/llama_lora", token = "YOUR_HF_TOKEN")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[ ]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("llama_finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("HF_USERNAME/llama_finetune", tokenizer, token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("llama_finetune", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("HF_USERNAME/llama_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("llama_finetune", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("HF_USERNAME/llama_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "HF_USERNAME/llama_finetune", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "YOUR_HF_TOKEN",
    )


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
#   <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
# </div>
