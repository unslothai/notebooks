#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# New 3x faster training & 30% less VRAM. New kernels, padding-free & packing. [Blog](https://unsloth.ai/docs/new/3x-faster-training-packing)
# 
# You can now train with 500K context windows on a single 80GB GPU. [Blog](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://unsloth.ai/docs/new/how-to-train-llms-with-unsloth-and-docker).
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) (faster, less VRAM RL) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/all-our-models) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.33.post1")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install transformers==4.56.2 && pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 
# We will be using `FastSentenceTransformer` to fine-tune `ModernBERT`.

# In[ ]:


from unsloth import FastSentenceTransformer

fourbit_models = [
    "unsloth/all-MiniLM-L6-v2",
    "unsloth/embeddinggemma-300m",
    "unsloth/Qwen3-Embedding-4B",
    "unsloth/Qwen3-Embedding-0.6B",
    "unsloth/all-mpnet-base-v2",
    "unsloth/gte-modernbert-base",
    "unsloth/bge-m3"

] # More models at https://huggingface.co/unsloth

model = FastSentenceTransformer.from_pretrained(
    model_name = "unsloth/gte-modernbert-base",
    max_seq_length = 512,   # Choose any for long context!
    full_finetuning = False, # [NEW!] We have full finetuning now!
)


# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[2]:


model = FastSentenceTransformer.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ['Wi', 'Wo', 'Wqkv'],
    lora_alpha = 128,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# <a name="Data"></a>
# ### Data Prep
# We now use the `electroglyph/technical` dataset.

# In[3]:


from datasets import load_dataset
dataset = load_dataset("electroglyph/technical", split = "train")


# Let's take a look at the dataset structure:

# In[4]:


print("Dataset examples:")
for i in range(6):
    print(dataset[i])


# ## Baseline Inference
# Let's test the base model before fine-tuning to see how it performs on our specific domain.

# In[5]:


from sentence_transformers import util

def test_inference(model, run_name = "Run"):
    """Test model with a query and candidate sentences"""
    query = "apexification"
    candidates = [
        "a brick left by Yuki",  # Completely unrelated
        "apples are a tasty treat",  # Unrelated, but shares "ap-" prefix
        "the weed whacker uses an engine that runs on a mixture of gas and oil",  # Unrelated
        "a type of cancer treatment that uses drugs to boost the body's immune response",  # Medical context but wrong procedure
        "a plant hormone for regulating stress responses",  # Scientific but unrelated field
        "induces root tip closure in non-vital teeth"  # CORRECT - this is what apexification actually means
    ]

    query_emb = model.encode(query, convert_to_tensor = True)
    candidate_embs = model.encode(candidates, convert_to_tensor = True)
    scores = util.cos_sim(query_emb, candidate_embs)[0]

    results = []
    for i, score in enumerate(scores):
        results.append((candidates[i], score.item()))
    results.sort(key = lambda x: x[1], reverse = True)

    print(f"\n--- {run_name} Results for query: '{query}' ---")
    for text, score in results:
        print(f"{score:.4f} | {text}")

test_inference(model, run_name = "Pre-Training")


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We use `MultipleNegativesRankingLoss`
# 
#  This loss function uses other positives in the same batch as negative examples, which is efficient for contrastive learning.

# In[6]:


from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from unsloth import is_bf16_supported

# This will use other positives in the same batch as negative examples
loss = losses.MultipleNegativesRankingLoss(model)

trainer = SentenceTransformerTrainer(
    model = model,
    train_dataset = dataset,
    loss = loss,
    args = SentenceTransformerTrainingArguments(
        output_dir = "output",
        num_train_epochs = 2,
        per_device_train_batch_size = 256,
        gradient_accumulation_steps = 1, # Use GA to mimic batch size!
        learning_rate = 3e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 50,
        warmup_ratio = 0.03,
        save_strategy = "no",
        report_to = "none",
        lr_scheduler_type = "constant_with_warmup",
        # Because we have duplicate anchors in the dataset, we don't want
        # to accidentally use them for negative examples
        batch_sampler = BatchSamplers.NO_DUPLICATES,
    ),

)


# In[7]:


# @title Show current memory stats
import torch
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[8]:


trainer_stats = trainer.train()


# In[9]:


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
# Let's run the model after training to see the improvements!

# In[10]:


test_inference(model, run_name = "Post-Training")


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit, scroll down!

# In[11]:


model.save_pretrained("modernbert_lora")  # Local saving
# model.push_to_hub("your_name/modernbert_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[12]:


if False:
    from unsloth import FastSentenceTransformer
    model = FastSentenceTransformer.from_pretrained(
        "model",
        # for_inference = True
    )


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[13]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("modernbert_finetune_16bit", tokenizer = model.tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/modernbert_finetune_16bit", tokenizer = model.tokenizer, save_method = "merged_16bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("modernbert_lora")
if False: # Pushing to HF Hub
    model.push_to_hub("HF_USERNAME/modernbert_lora", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 

# In[ ]:


# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("modernbert_finetune",)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("HF_USERNAME/modernbert_finetune", token = "")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("modernbert_finetune", quantization_method = "f16")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("HF_USERNAME/modernbert_finetune", quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("modernbert_finetune", quantization_method = "q4_k_m")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("HF_USERNAME/modernbert_finetune", quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "HF_USERNAME/modernbert_finetune", # Change hf to your username!
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install-and-update) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
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
# 
