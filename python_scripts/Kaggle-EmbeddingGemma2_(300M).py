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
# %%capture
# import os, re
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth  # Do this in local & cloud setups
# else:
#     !pip install sentencepiece protobuf "datasets==4.3.0" hf_transfer
#     !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth
#     !unsloth install-kernels
#     !pip install --no-deps --upgrade "torchao>=0.16.0"
# !pip install --no-deps "transformers @ git+https://github.com/huggingface/transformers@main" "tokenizers>=0.23.1,<0.24" "safetensors>=0.8.0"
# !pip install "huggingface_hub>=1.31.0,<2.0" "sentence-transformers>=6.1.0" torchcodec
# 
# # ### Unsloth

# In[ ]:


from unsloth import FastSentenceTransformer

model = FastSentenceTransformer.from_pretrained(
    model_name = "unsloth/embeddinggemma-2",
    max_seq_length = 1024,   # The model supports up to 8192 tokens
    full_finetuning = False, # [NEW!] We have full finetuning now!
    config_kwargs = {"vision_config": None, "audio_config": None}, # Text only
)

# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[2]:


model = FastSentenceTransformer.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = "FEATURE_EXTRACTION"
)

# <a name="Data"></a>
# ### Data Prep
# We now use the ``tomaarsen/miriad-4.4M-split`` dataset, a large-scale collection of 4.4 million medical question-answer pairs distilled from peer-reviewed biomedical literature. To maintain efficiency, we use data streaming to ingest a subset of 10,000 training samples and 2,000 evaluation samples.

# In[3]:


from datasets import load_dataset,Dataset

stream_train = list(load_dataset("tomaarsen/miriad-4.4M-split", split = "train",streaming = True).take(10000))
stream_eval = list(load_dataset("tomaarsen/miriad-4.4M-split", split = "eval",streaming = True).take(2000))

train_dataset = Dataset.from_generator(lambda: (yield from stream_train))
eval_dataset = Dataset.from_generator(lambda: (yield from stream_eval))

# Let's take a look at the dataset structure:

# In[4]:


train_dataset[0]

# ## Baseline Performance
# Retrieval quality before finetuning, on the medical eval set and on [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6).

# In[5]:


from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator

queries = dict(enumerate(eval_dataset["question"]))
corpus = dict(enumerate(list(eval_dataset["passage_text"]) + train_dataset["passage_text"][:2000]))
relevant_docs = {idx: [idx] for idx in queries}
evaluator = InformationRetrievalEvaluator(
    queries = queries,
    corpus = corpus,
    relevant_docs = relevant_docs,
    query_prompt = model.prompts["query"],
    corpus_prompt = model.prompts["document"],
    show_progress_bar = False,
    batch_size = 64,
    name = "miriad",
)
nano_beir = NanoBEIREvaluator(
    dataset_names = ["nfcorpus", "scifact", "fiqa2018", "arguana"],
    query_prompts = model.prompts["query"],
    corpus_prompts = model.prompts["document"],
    batch_size = 64,
    show_progress_bar = False,
)
baseline = evaluator(model)
baseline_nano = nano_beir(model)
print(f"Medical retrieval NDCG@10 : {baseline['miriad_cosine_ndcg@10']:.4f}")
print(f"NanoBEIR mean NDCG@10     : {baseline_nano['NanoBEIR_mean_cosine_ndcg@10']:.4f}")

# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We use `MultipleNegativesRankingLoss`
# 
#  This loss function uses other positives in the same batch as negative examples, which is efficient for contrastive learning.
# 
#  We do 30 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[6]:


from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from unsloth import is_bf16_supported

loss = losses.MultipleNegativesRankingLoss(model)

trainer = SentenceTransformerTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    loss = loss,
    args = SentenceTransformerTrainingArguments(
        # num_train_epochs = 1,
        max_steps = 30,
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        gradient_accumulation_steps = 2, # Use GA to mimic batch size!
        learning_rate = 2e-5,
        logging_steps = 5,
        warmup_steps = 0.03, # A float is a ratio of the total steps
        prompts = {  # Map training column names to model prompts
          "question": model.prompts["query"],
          "passage_text": model.prompts["document"],
        },
        report_to = "none", # Use TrackIO/WandB etc
        bf16 = is_bf16_supported(),
        output_dir = "output",
        lr_scheduler_type = "linear",
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

# ### Now after finetuning, let's evaluate the model again!

# In[15]:


after = evaluator(model)
after_nano = nano_beir(model)
print(f"Medical retrieval NDCG@10 : {baseline['miriad_cosine_ndcg@10']:.4f} -> {after['miriad_cosine_ndcg@10']:.4f}")
print(f"NanoBEIR mean NDCG@10     : {baseline_nano['NanoBEIR_mean_cosine_ndcg@10']:.4f} -> {after_nano['NanoBEIR_mean_cosine_ndcg@10']:.4f}")

# <a name="Inference"></a>
# ### Inference
# Let's run the model after training to see the improvements!

# In[22]:


query = "Patient presents with sharp chest pain that improves when leaning forward."

candidates = [
    "Acute Pericarditis often involves pleuritic chest pain relieved by sitting up and leaning forward.",
    "Myocardial Infarction typically presents with crushing substernal pressure and radiation to the left arm.",
    "Pneumothorax is characterized by sudden onset shortness of breath and unilateral chest pain.",
    "Gastroesophageal Reflux Disease (GERD) causes burning retrosternal pain usually after meals."
]

query_emb = model.encode(query, prompt_name = "query", convert_to_tensor = True)
candidate_embs = model.encode(candidates, prompt_name = "document", convert_to_tensor = True)
similarities = model.similarity(query_emb, candidate_embs)

ranking = similarities.argsort(descending = True)[0]

for idx in ranking.tolist():
    score = similarities[0][idx].item()
    text = candidates[idx]
    print(f"{score:.4f} | {text}")

# Matryoshka: keep only the first 512, 256 or 128 dimensions with `truncate_dim`:

# In[ ]:


for dim in [768, 256, 128]:
    q = model.encode(query, prompt_name = "query", truncate_dim = dim, normalize_embeddings = True, convert_to_tensor = True)
    c = model.encode(candidates, prompt_name = "document", truncate_dim = dim, normalize_embeddings = True, convert_to_tensor = True)
    best = model.similarity(q, c)[0].argmax().item()
    print(f"{dim:4d} dims -> top match: {candidates[best][:60]}")

# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit, scroll down!

# In[17]:


model.save_pretrained("embeddinggemma_lora")  # Local saving
model.tokenizer.save_pretrained("embeddinggemma_lora")
# model.push_to_hub("your_name/embeddinggemma_lora", token = "YOUR_HF_TOKEN") # Online saving
# model.tokenizer.push_to_hub("your_name/embeddinggemma_lora", token = "YOUR_HF_TOKEN") # Online saving

# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[18]:


if False:
    from unsloth import FastSentenceTransformer
    model = FastSentenceTransformer.from_pretrained(
        "lora_model",
        config_kwargs = {"vision_config": None, "audio_config": None},
    )

# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[19]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("embeddinggemma_finetune_16bit", tokenizer = model.tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/embeddinggemma_finetune_16bit", tokenizer = model.tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("embeddinggemma_lora")
if False: # Pushing to HF Hub
    model.push_to_hub("HF_USERNAME/embeddinggemma_lora", token = "YOUR_HF_TOKEN")

# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# In[ ]:


# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("embeddinggemma_finetune",)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("HF_USERNAME/embeddinggemma_finetune", token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("embeddinggemma_finetune", quantization_method = "f16")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("HF_USERNAME/embeddinggemma_finetune", quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("embeddinggemma_finetune", quantization_method = "q4_k_m")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("HF_USERNAME/embeddinggemma_finetune", quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "HF_USERNAME/embeddinggemma_finetune", # Change hf to your username!
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "YOUR_HF_TOKEN", # Get a token at https://huggingface.co/settings/tokens
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
