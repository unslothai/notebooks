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
# !pip install --no-deps "transformers @ git+https://github.com/huggingface/transformers@92cd495f2720c064bc78eb2d93e28704c5bce51f" "tokenizers>=0.23.1,<0.24" "safetensors>=0.8.0"
# !pip install "huggingface_hub>=1.31.0,<2.0" "sentence-transformers>=6.1.0" torchcodec
# 
# # ### Unsloth
# 
# Fine-tune **EmbeddingGemma 2** for image to caption retrieval with LoRA and compare Recall@K before and after.

# In[ ]:


import os
try:
    from google.colab import userdata
    os.environ.setdefault("HF_TOKEN", userdata.get("HF_TOKEN") or "")
except Exception:
    pass

# In[ ]:


from unsloth import FastSentenceTransformer
import torch

model = FastSentenceTransformer.from_pretrained(
    model_name = "unsloth/embeddinggemma-2",
    max_seq_length = 512, # an image is 280 tokens
    config_kwargs = {"audio_config": None}, # skip the audio tower: less VRAM
    full_finetuning = False,
)

# In[ ]:


# @title Small display helpers (thumbnails, audio players, video)
import io, base64, html, torch
from IPython.display import display, HTML, Audio, Video
from PIL import Image

def show_images(images, captions = None, size = 160):
    cells = []
    for i, img in enumerate(images):
        buf = io.BytesIO(); img.copy().convert("RGB").resize((size, size)).save(buf, format = "JPEG")
        cap = html.escape(captions[i]) if captions else ""
        cells.append(f'<div style="display:inline-block;margin:4px;width:{size}px;font-size:11px;vertical-align:top">'
                     f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"><br>{cap}</div>')
    display(HTML("".join(cells)))

def recall_at_k(similarity, positives, ks = (1, 5, 10)):
    """similarity: [queries, items]; positives[q] = set of correct item indices."""
    ranking = similarity.argsort(dim = 1, descending = True)
    out = {}
    for k in ks:
        top = ranking[:, :k].tolist()
        out[f"R@{k}"] = round(100 * sum(len(set(t) & positives[q]) > 0 for q, t in enumerate(top)) / len(top), 2)
    return out

# <a name="Data"></a>
# ### Data Prep
# Train on Flickr8k (3,000 images), test on the Flickr30k 1K test split.

# In[ ]:


import json, csv, os, zipfile, random
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from datasets import Dataset

train_file = hf_hub_download("jxie/flickr8k", "data/train-00000-of-00002-2f8f6bfa852eac4b.parquet", repo_type = "dataset")
train_rows = pq.read_table(train_file).to_pylist()
random.seed(3407)
os.makedirs("flickr8k", exist_ok = True)
pairs = [] # image file paths keep the dataset small; sentence-transformers opens them on the fly
for r in train_rows:
    path = f"flickr8k/{r['image']['path']}"
    open(path, "wb").write(r["image"]["bytes"])
    for k in range(5):
        pairs.append({"anchor": r[f"caption_{k}"], "positive": path})
random.shuffle(pairs)
train_dataset = Dataset.from_list(pairs)

repo = "nlphuji/flickr_1k_test_image_text_retrieval"
archive = zipfile.ZipFile(hf_hub_download(repo, "images_flickr_1k_test.zip", repo_type = "dataset"))
rows = list(csv.DictReader(open(hf_hub_download(repo, "test_1k_flickr.csv", repo_type = "dataset"))))
members = {os.path.basename(n): n for n in archive.namelist() if n.endswith(".jpg")}
test_items = [Image.open(archive.open(members[r["filename"]])).convert("RGB") for r in rows]
test_texts, text_to_item = [], []
for i, r in enumerate(rows):
    for c in json.loads(r["raw"]):
        test_texts.append(c); text_to_item.append(i)
print(f"train pairs: {len(train_dataset)}, test: {len(test_items)} images / {len(test_texts)} captions")
show_images([Image.open(p["positive"]) for p in pairs[:4]], [p["anchor"] for p in pairs[:4]])

# ## Baseline Performance

# In[ ]:


def evaluate(model):
    model.eval()
    with torch.no_grad():
        items = model.encode([{"image": x} for x in test_items], batch_size = 16, convert_to_tensor = True)
        texts = model.encode(test_texts, batch_size = 64, convert_to_tensor = True)
    sim = model.similarity(texts, items).float().cpu()
    item_to_texts = [set(j for j, it in enumerate(text_to_item) if it == i) for i in range(len(test_items))]
    return {"text->image": recall_at_k(sim, [{text_to_item[q]} for q in range(len(test_texts))]),
            "image->text": recall_at_k(sim.T, item_to_texts)}

baseline = evaluate(model)
print(baseline)

# We add LoRA adapters to the language model and the vision encoder (`finetune_vision_layers = True`).

# In[ ]:


model = FastSentenceTransformer.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    finetune_vision_layers = True, # also adapt the vision encoder
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # saves VRAM for the vision tower
    random_state = 3407,
    task_type = "FEATURE_EXTRACTION",
)

# <a name="Train"></a>
# ### Train the model
# We train for 60 steps; set `num_train_epochs = 1` and `max_steps = -1` for a full run.

# In[ ]:


from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from unsloth import is_bf16_supported

trainer = SentenceTransformerTrainer(
    model = model,
    train_dataset = train_dataset,
    loss = losses.MultipleNegativesRankingLoss(model),
    args = SentenceTransformerTrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        max_steps = 60,
        learning_rate = 1e-4,
        warmup_ratio = 0.05,
        lr_scheduler_type = "linear",
        logging_steps = 5,
        bf16 = is_bf16_supported(),
        fp16 = not is_bf16_supported(), # Unsloth switches EmbeddingGemma 2 to float32 math on T4
        remove_unused_columns = False,
        report_to = "none",
        output_dir = "outputs",
        save_strategy = "no",
        seed = 3407,
    ),
)

# In[ ]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# In[ ]:


trainer_stats = trainer.train()

# In[ ]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB of {max_memory} GB.")

# ### Evaluate after fine-tuning

# In[ ]:


finetuned = evaluate(model)
for direction in baseline:
    print(f"{direction:12s} before {baseline[direction]}  after {finetuned[direction]}")

# <a name="Inference"></a>
# ### Inference

# In[ ]:


item_embeddings = model.encode(test_items, batch_size = 16, convert_to_tensor = True)
for query in ["two children playing soccer on a field", "a man riding a bike down a mountain trail"]:
    scores = model.similarity(model.encode(query, convert_to_tensor = True), item_embeddings)[0]
    top = scores.topk(4)
    print(query); show_images([test_items[i] for i in top.indices.tolist()], [f"{s:.3f}" for s in top.values.tolist()])

# <a name="Save"></a>
# ### Saving
# `save_pretrained` saves the LoRA adapters, `save_pretrained_merged` the full model.

# In[ ]:


model.save_pretrained("embeddinggemma_lora")  # Local saving
model.tokenizer.save_pretrained("embeddinggemma_lora")
# model.push_to_hub("your_name/embeddinggemma_lora", token = "YOUR_HF_TOKEN") # Online saving

# In[ ]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("embeddinggemma_finetune_16bit", tokenizer = model.tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("HF_USERNAME/embeddinggemma_finetune_16bit", tokenizer = model.tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

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
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
