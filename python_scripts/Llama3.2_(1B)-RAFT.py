#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker).
# 
# [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) is now supported with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which creates kernels!
# 
# Introducing [Vision](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) and [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) for RL! Train Qwen, Gemma etc. VLMs with GSPO - even faster with less VRAM.
# 
# Unsloth now supports Text-to-Speech (TTS) models. Read our [guide here](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning).
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth\nelse:\n    # Do this only in Colab notebooks! Otherwise use pip install unsloth\n    import torch; v = re.match(r"[0-9\\.]{3,}", str(torch.__version__)).group(0)\n    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")\n    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo\n    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth\n!pip install transformers==4.55.4\n!pip install --no-deps trl==0.22.2\n')
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

Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")


# ### Ingest documents 

# We'll use the following code to download a research paper and then load it using `SimpleDirectoryReader`. This will be the data we use for our retrieval augmented finetuning.

# In[ ]:


get_ipython().system('mkdir  -p ../data')
get_ipython().system('wget "https://arxiv.org/pdf/2405.00247.pdf" -O "../data/non_traditional_credentials.pdf"')

docs = SimpleDirectoryReader("../data/").load_data(show_progress=True)


# ### Retrieval Augmented Finetuning

# ### Getting the RAFT dataset
# LlamaIndex has very kindly adapted the source code of the RAFT repository and made it even easier to generate your own RAFT dataset. Just point it to your filepath.t
# > Reference: [RAFTDatasetPack](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb)

# In[14]:


from llama_index.packs.raft_dataset import RAFTDatasetPack

raft_dataset = RAFTDatasetPack(
    file_path = "../data/non_traditional_credentials.pdf",
    llm = Settings.llm,
    embed_model=Settings.embed_model
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


splits = dataset.train_test_split(test_size=0.1)
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
train_ds = train_ds.map(formatting_prompts_func, batched=True)
eval_ds = eval_ds.map(formatting_prompts_func, batched=True)


# Let's take a look at what we just did!

# In[26]:


from IPython.display import display, Markdown

display(Markdown(pd.DataFrame(train_ds).head()['text'].iloc[0]))


# ### And now we finally get to training!

# In[28]:


from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="llama32_1bn_raft_v2", #This will also be used as your huggingfacehub model id name
    report_to="wandb", #Leave this to be blank if you don't want to use wandb
    run_name="RAFT_SFT_Take7",
    eval_steps=5,
    eval_strategy="steps",
    per_device_train_batch_size=1,    # small batches if quantized
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=5,
    # max_steps=60,                    # or set num_train_epochs
    save_strategy="no",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=5,
    seed=42,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = eval_ds, 
    args=training_args,
    dataset_text_field="text",

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
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[ ]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",
    )


# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.
# 
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
