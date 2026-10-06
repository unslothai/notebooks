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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.35")\n    if str(torch.version.cuda).startswith("13") and xformers.endswith("0.0.35"): xformers = "https://download.pytorch.org/whl/cu130/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"\n    !pip install sentencepiece protobuf "datasets==4.3.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install --no-deps "transformers>=5.18.0" "tokenizers>=0.23.1,<0.24" "safetensors>=0.8.0"\n!pip install "huggingface_hub>=1.31.0,<2.0" "sentence-transformers>=6.1.0" torchcodec\n')
# 
# 
# # ### Unsloth
# 
# Fine-tune **EmbeddingGemma 2** for **sound <-> description retrieval** with LoRA.
# We load only the text and audio towers, add LoRA adapters to the language model **and** the audio
# encoder, train with an in-batch contrastive loss, and measure Recall@K on a standard benchmark before and after.
# 
# Works on a free Tesla T4: Unsloth keeps EmbeddingGemma 2 out of float16 activations automatically.

# In[ ]:


# EmbeddingGemma 2 may need a Hugging Face token (Colab: add HF_TOKEN under Secrets).
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
    max_seq_length = 1024, # audio is 25 tokens per second: a 30 s clip is 750 tokens
    config_kwargs = {"vision_config": None}, # skip the vision tower: less VRAM
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
# * **Train:** Clotho development split (first 10 shards, ~300 clips, 5 captions each).
# * **Test:** Clotho evaluation split as packaged by MTEB (first shard: 209 clips, ~1,000 captions).
# Clips are 15-30 s environmental sounds (rain, traffic, birds, machines...).

# In[ ]:


import random, soundfile as sf, numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
from datasets import Dataset

import os
os.makedirs("clotho", exist_ok = True)
def save_clip(row): # write the wav once; sentence-transformers decodes + resamples audio files itself
    path = f"clotho/{row['index'].rsplit('/', 1)[-1]}.wav"
    open(path, "wb").write(row["audio"]["bytes"])
    return path

def split_captions(text):
    return [c.strip() + "." for c in text.split(". ") if c.strip()]

train_files = sorted(f.rfilename for f in HfApi().dataset_info("CLAPv2/Clotho").siblings if f.rfilename.startswith("data/train/"))[:10]
random.seed(3407)
pairs = []
for f in train_files:
    for r in pq.read_table(hf_hub_download("CLAPv2/Clotho", f, repo_type = "dataset")).to_pylist():
        clip = save_clip(r)
        for c in split_captions(r["text"]):
            pairs.append({"anchor": c, "positive": clip})
random.shuffle(pairs)
train_dataset = Dataset.from_list(pairs)

test_rows = pq.read_table(hf_hub_download("mteb/Clotho", "data/test-00000-of-00005.parquet", repo_type = "dataset")).to_pylist()
test_items = [save_clip(r) for r in test_rows]
test_texts, text_to_item = [], []
for i, r in enumerate(test_rows):
    for c in split_captions(r["text"]):
        test_texts.append(c); text_to_item.append(i)
print(f"train pairs: {len(train_dataset)}, test: {len(test_items)} clips / {len(test_texts)} captions")
print(pairs[0]["anchor"]); display(Audio(filename = pairs[0]["positive"]))


# ## Baseline Performance
# Recall@K on the Clotho test set before training.

# In[ ]:


def evaluate(model):
    model.eval()
    with torch.no_grad():
        items = model.encode([{"audio": x} for x in test_items], batch_size = 8, convert_to_tensor = True)
        texts = model.encode(test_texts, batch_size = 64, convert_to_tensor = True)
    sim = model.similarity(texts, items).float().cpu()
    item_to_texts = [set(j for j, it in enumerate(text_to_item) if it == i) for i in range(len(test_items))]
    return {"text->audio": recall_at_k(sim, [{text_to_item[q]} for q in range(len(test_texts))]),
            "audio->text": recall_at_k(sim.T, item_to_texts)}

baseline = evaluate(model)
print(baseline)


# We now add LoRA adapters to the **language model and the audio encoder**. `finetune_audio_layers = True`
# is what turns on the audio tower: without it Unsloth only adapts the language model.

# In[ ]:


model = FastSentenceTransformer.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    finetune_audio_layers = True, # also adapt the audio encoder
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # saves VRAM for the audio tower
    random_state = 3407,
    task_type = "FEATURE_EXTRACTION",
)


# <a name="Train"></a>
# ### Train the model
# `MultipleNegativesRankingLoss` treats the other clips in the batch as negatives, so a bigger batch
# means a harder task. We train for 60 steps; set `num_train_epochs = 1` and `max_steps = -1` for a full run.

# In[ ]:


from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from unsloth import is_bf16_supported

trainer = SentenceTransformerTrainer(
    model = model,
    train_dataset = train_dataset,
    loss = losses.MultipleNegativesRankingLoss(model),
    args = SentenceTransformerTrainingArguments(
        per_device_train_batch_size = 8,
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


item_embeddings = model.encode([{"audio": x} for x in test_items], batch_size = 8, convert_to_tensor = True)
for query in ["birds chirping in a forest", "a car driving past on a wet road"]:
    scores = model.similarity(model.encode(query, convert_to_tensor = True), item_embeddings)[0]
    best = scores.argmax().item()
    print(f"{query} -> {test_texts[text_to_item.index(best)]} ({scores[best]:.3f})")
    display(Audio(filename = test_items[best]))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# `save_pretrained` saves only the LoRA adapters. `save_pretrained_merged` writes a full model that
# plain `sentence-transformers` can load; it keeps every tower, so the merged model still handles
# text, images, audio and video.

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
