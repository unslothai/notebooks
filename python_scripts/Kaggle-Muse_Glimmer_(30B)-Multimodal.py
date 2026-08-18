#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a Google Colab **A100** or **H100** instance.
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/blob/main/images/Discord button.png?raw=true" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + star us on <a href="https://github.com/unslothai/unsloth">Github</a>
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), and how to save it.
# 
# **Muse Glimmer is a 28B dense model that reads text, images and video.** This notebook fine-tunes both
# modalities in one run: still images through `<|patch|>` tokens and video clips through
# `<|video|>` tokens. There is no audio tower on this model, so text, image and video is the whole
# input surface.
# 
# Muse Glimmer is private for now, so this notebook pulls the model and a matching `transformers` build

# ### News

# Introducing **Unsloth Desktop**, the first desktop app to run and train models. It is free and open source for macOS, Windows, and Linux, and runs on your own local hardware. [Learn more](https://unsloth.ai/docs/desktop) • [Download](https://unsloth.ai/download)
# 
# <p align="center">
# <a href="https://unsloth.ai/docs/desktop"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/unsloth-desktop.jpg" width="900" alt="Introducing Unsloth Desktop"></a>
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
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==5.15.0\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # In[ ]:
# 
# 
# # Muse Glimmer support ships in transformers 5.15.0, so install the release.
# # Pinned to that exact version so the notebook does not change underneath you.
# # Run this BEFORE anything imports transformers.
# get_ipython().system('pip install -q "transformers==5.15.0"')
# 
# import transformers
# print("transformers:", transformers.__version__)
# assert transformers.__version__ == "5.15.0", transformers.__version__
# from transformers import AutoConfig
# _cfg = AutoConfig.from_pretrained("unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit")
# print("architecture available:", _cfg.model_type)
# 
# 
# # ### Which repo to load
# # 
# # Two public repos hold the same weights, and both work with `FastModel.from_pretrained`:
# # 
# # | repo | precision | size | use it when |
# # |---|---|---|---|
# # | [`unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit) | 4-bit (bitsandbytes nf4) | ~21 GB | fine-tuning on one or two consumer GPUs. This is what the cells below use. |
# # | [`unsloth/Muse-Glimmer-30B`](https://huggingface.co/unsloth/Muse-Glimmer-30B) | bf16 | ~56 GB | full-precision inference or full fine-tuning on large-memory hardware. |
# # 
# # GGUFs for llama.cpp are at
# # [`unsloth/Muse-Glimmer-30B-GGUF`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF),
# # from 10 GB up to bf16, plus the vision projector.
# 
# # ### Unsloth
# 
# `FastModel` loads Muse Glimmer through the generic vision path. The 4bit build is
# `unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`, a private repo, so we pass the token explicitly.
# 
# **Memory.** The resident weights split roughly like this:
# 
# | part | dtype | size |
# |---|---|---|
# | 416 quantised text `Linear4bit` modules | nf4 with double quant | 11.72 GiB |
# | `embed_tokens` and `lm_head` (202048 x 6656 each, untied) | 16-bit | 5.38 GiB |
# | vision tower, adapter and projection | 16-bit | 3.44 GiB |
# | total | | **20.68 GiB** |
# 
# The embeddings are 26% of that and they are frozen during LoRA training, so we set
# `offload_embedding = True`. That keeps `embed_tokens` in CPU RAM and moves only the looked-up
# rows to the GPU, taking 2.5 GiB off the resident footprint. `lm_head` has to stay on the GPU.
# 
# Video is the expensive input here. Each frame group costs up to 144 tokens, and the default
# sampler will take up to 96 frames, so an unconstrained clip can be 6912 tokens on its own. We cap
# it at 8 frames, which is 4 groups and at most 576 tokens.
# 
# Measured on one 180 GiB card at `max_seq_length = 2048`, batch size 1, LoRA r=16, mixed
# image and video batches: 18.22 GiB resident after load with the offload on, 20.72 GiB without it.
# Peak reserved through training is reported at the end of this notebook. A 24 GiB card is the
# realistic minimum, and a single 16 GiB card cannot even hold the weights.

# In[ ]:


from unsloth import FastModel
import torch

max_seq_length = 2048

model, processor = FastModel.from_pretrained(
    model_name = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
    load_in_4bit = True,
    max_seq_length = max_seq_length,
    use_gradient_checkpointing = "unsloth",
    offload_embedding = True,   # Keeps the 202048 x 6656 embedding matrix in RAM
)
print(type(model).__name__, type(processor).__name__)


# `MuseGlimmerProcessor` holds a `MuseGlimmerImageProcessor` and a `MuseGlimmerVideoProcessor`. There is no feature
# extractor, because there is no audio tower.
# 
# Two settings on the video processor need attention:
# 
# - `patch_temporal` is read when the prompt is built, but the class defines the same value as
#   `temporal_patch_size`. We mirror it across so video prompts render.
# - `num_frames` and `fps` control how many frames each clip contributes. The defaults, 96 frames
#   at 2 fps, are far too many for a fine-tuning run.

# In[ ]:


print("image token:", processor.image_token, processor.image_token_id)
print("video token:", processor.video_token, processor.video_token_id)
print("feature extractor:", getattr(processor, "feature_extractor", None))

video_processor = processor.video_processor
video_processor.patch_temporal = video_processor.temporal_patch_size
video_processor.num_frames = 8    # At most 8 frames per clip
video_processor.fps = 1.0         # Sampled at 1 frame per second

VIDEO_FPS = video_processor.fps
print("frames per clip:", video_processor.num_frames, "at", video_processor.fps, "fps")


# We now add LoRA adapters. Both the vision tower and the language model are trained, since the
# video path runs through the same tower as still images.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = True,  # The video frames go through this tower too
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
model.print_trainable_parameters()


# <a name="Data"></a>
# ### Data Prep
# 
# We use [XinNUS/Temporal_Caption_Bench](https://huggingface.co/datasets/XinNUS/Temporal_Caption_Bench),
# 2267 short clips of 3 to 12 seconds with human written descriptions of what happens in them. It is
# CC-BY-4.0 and about 670 MB, so it downloads in a minute or so.
# 
# Each row gives us two training examples:
# 
# - the **clip**, asked for a description of what happens over time
# - a **single frame** from the middle of the clip, asked for a description of that still image
# 
# That exercises `<|video|>` and `<|patch|>` in the same run, on the same real data.

# In[ ]:


from datasets import load_dataset, Video

# This dataset stores one mp4 per clip, so it is 2268 small files rather than a few
# shards. It is only 0.62 GiB in total, but fetched one at a time on a free cloud
# runtime the round trips alone can outlast the cell timeout. num_proc pulls them in
# parallel. The slice keeps the download honest: 64 clips is plenty to fine-tune on.
dataset = load_dataset(
    "XinNUS/Temporal_Caption_Bench", split = "train[:64]", num_proc = 8,
)
# decode = False keeps the raw mp4 path, which is what we hand to the video processor.
dataset = dataset.cast_column("video", Video(decode = False))
dataset


# In[ ]:


row = dataset[0]
print("query:", row["query"])
print("clip:", row["video"]["path"])
print("duration:", round(row["duration"], 1), "seconds")
for fact in row["facts"][:3]:
    print(" -", fact["tag"], "|", fact["text"])


# The `shared` facts are the ones every annotator agreed on, so we use those as the caption.

# In[ ]:


from torchcodec.decoders import VideoDecoder
from PIL import Image

VIDEO_INSTRUCTION = "Describe what happens in this video."
IMAGE_INSTRUCTION = "Describe what you see in this image."

def caption_of(row):
    shared = " ".join(f["text"] for f in row["facts"] if f["tag"] == "shared")
    return shared or " ".join(f["text"] for f in row["facts"])

def middle_frame(path):
    decoder = VideoDecoder(path)
    frame = decoder[len(decoder) // 2]           # uint8, channels first
    return Image.fromarray(frame.permute(1, 2, 0).numpy())

def video_example(row):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "video", "video": row["video"]["path"]},
            {"type": "text",  "text": VIDEO_INSTRUCTION},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": caption_of(row)}]},
    ]}

def image_example(row):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": middle_frame(row["video"]["path"])},
            {"type": "text",  "text": IMAGE_INSTRUCTION},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": caption_of(row)}]},
    ]}


# In[ ]:


import random

# Hold one clip back so the inference cells run on something the model has not seen.
test_row = dataset[len(dataset) - 1]
train_rows = [dataset[i] for i in range(len(dataset) - 1)]

converted_dataset = [video_example(r) for r in train_rows]
converted_dataset += [image_example(r) for r in train_rows[:len(train_rows) // 2]]
random.Random(3407).shuffle(converted_dataset)

print(len(converted_dataset), "examples")
print(caption_of(test_row))


# Muse Glimmer uses ATEM, not ChatML and not the Llama format. Turns render as
# `<|start|>{role}<|message|>...<|eot|>`, and the assistant addresses a recipient: `to=self` for its
# private reasoning, closed with `<|eom|>`, then `to=user` for the answer.
# 
# Media is a single placeholder in the template, and `MuseGlimmerProcessor` expands it once it sees the
# real pixels. A video expands into `<|vid_start|>`, then per frame group a `Time: Xs` stamp
# followed by `<|video|>` tokens, separated by `<|vid_frame_separator|>` and closed with
# `<|vid_end|>`. Let us look at both.

# In[ ]:


video_text = processor.apply_chat_template(
    converted_dataset[0]["messages"], tokenize = False, add_generation_prompt = False,
)
print(video_text)


# ### A collator that lets the processor decode the video
# 
# `UnslothVisionDataCollator` normally decodes and re-samples videos itself and hands the
# processor bare frames. Muse Glimmer needs real video metadata: it stamps each frame group with a
# timestamp taken from the clip, and without metadata every stamp is invented. In testing that path
# also collapsed an 8 frame clip down to 2 frames.
# 
# Passing the file path through instead lets the processor decode the clip once, with true
# timestamps and the frame count we asked for. Everything else about the collator, including the
# response-only masking, is unchanged.

# In[ ]:


from unsloth.trainer import UnslothVisionDataCollator
from unsloth_zoo.vision_utils import fetch_image

class MuseGlimmerVisionDataCollator(UnslothVisionDataCollator):
    def _extract_images_videos_for_example(self, example, messages):
        images, videos = [], []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image" and part.get("image") is not None:
                    images.append(fetch_image(part, size_factor = self.patch_size * 2))
                elif part.get("type") == "video" and part.get("video") is not None:
                    videos.append(part["video"])   # decoded by MuseGlimmerVideoProcessor
        return images, videos, {"fps": [VIDEO_FPS] * len(videos)}


# <a name="Train"></a>
# ### Train the model
# 
# `loss_type = "nll"` is deliberate. TRL's newer default, `chunked_nll`, normalises the loss by the
# accumulated token count itself, and on model classes that declare `accepts_loss_kwargs = False`
# (Muse Glimmer is one) the trainer then divides by `gradient_accumulation_steps` a second time. The loss
# and the gradients both come out `1/gradient_accumulation_steps` too small and nothing warns you.
# Plain `nll` is invariant to gradient accumulation and re-enables Unsloth's own fused
# cross-entropy, which uses less VRAM here.
# 
# The masking strings are the real ATEM markers, read out of the rendered sample above.

# ## Controlling how much the model thinks
# 
# This model reasons in a private `to=self` channel before it answers. The chat template
# renders `Reasoning strength: <level>.` into the system block, and it defaults to `high`,
# so if you never set it you are paying for the most expensive setting on every prompt.
# Pass `reasoning_strength` to `apply_chat_template` to change it.
# 
# Measured on the 4-bit checkpoint, one multi-step word problem, greedy decoding:
# 
# | reasoning_strength | private reasoning tokens | total completion tokens | answer |
# |---|---|---|---|
# | minimal | 248 | 255 | correct |
# | low | 247 | 254 | correct |
# | medium | 401 | 408 | correct |
# | high (default) | 502 | 509 | correct |
# 
# All four reached the same right answer, and `high` spent roughly twice the tokens of
# `minimal` to get there. Reasoning tokens count against `max_new_tokens`, so a run that
# hits the ceiling mid-thought returns an empty answer, which looks exactly like a wrong
# one. Raise `max_new_tokens` before you raise the effort. One sample per setting is a
# smoke test rather than a measurement, so sweep your own task before locking a value in.

# In[ ]:


reasoning_strength = "low"  # minimal, low, medium, high. The template default is high.

messages = [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}]
text = processor.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
    reasoning_strength = reasoning_strength,
)
print([line for line in text.splitlines() if "Reasoning strength" in line])


# In[ ]:


from trl import SFTTrainer, SFTConfig

instruction_part = "<|start|>user<|message|>"
response_part    = "<|start|>assistant to=user<|message|>"

collator = MuseGlimmerVisionDataCollator(
    model, processor,
    max_seq_length = max_seq_length,
    train_on_responses_only = True,
    instruction_part = instruction_part,
    response_part    = response_part,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = converted_dataset,
    processing_class = processor.tokenizer,
    data_collator = collator,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        # num_train_epochs = 1, # Set this instead of max_steps for a full run
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",

        # Keeps the loss invariant to gradient accumulation on this model class.
        loss_type = "nll",

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = max_seq_length,
    ),
)


# Check that the video expands with real timestamps and that only the answer is trained on:

# In[ ]:


import re

video_batch = collator([video_example(train_rows[0])])
image_batch = collator([image_example(train_rows[0])])

for name, batch in (("video", video_batch), ("image", image_batch)):
    ids = batch["input_ids"][0].tolist()
    print(name, "sequence length", len(ids),
          "| video tokens", ids.count(processor.video_token_id),
          "| image tokens", ids.count(processor.image_token_id))
    print("  grid:", batch.get("video_grid_thw", batch.get("image_grid_thw")).tolist())
    print("  timestamps:", re.findall(r"Time: [\d.]+s", processor.tokenizer.decode(ids)))
    labels = batch["labels"][0]
    print("  trained tokens:", int((labels != -100).sum()), "of", labels.numel())
    print("  trained text:",
          processor.tokenizer.decode([t for t in labels.tolist() if t != -100])[:160])


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
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference on a video
# 
# `add_generation_prompt = True` ends the prompt at `<|start|>assistant`, which leaves the recipient
# open, so the model usually opens its private `to=self` reasoning channel first. Appending
# ` to=user<|message|>` pins it to the answer channel, which is what we trained.

# In[ ]:


from transformers import TextStreamer

def muse_glimmer_prompt(messages, answer_directly = True):
    text = processor.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
    return text + " to=user<|message|>" if answer_directly else text

messages = [
    {"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": VIDEO_INSTRUCTION},
    ]},
]
inputs = processor(
    text = [muse_glimmer_prompt(messages)],
    videos = [[test_row["video"]["path"]]],
    fps = VIDEO_FPS,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

streamer = TextStreamer(processor.tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens = 256,
                   use_cache = True, do_sample = False)

print("\nReference caption:", caption_of(test_row))


# ### Inference on a still image

# In[ ]:


from IPython.display import display

test_frame = middle_frame(test_row["video"]["path"])
display(test_frame)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": IMAGE_INSTRUCTION},
    ]},
]
inputs = processor(
    text = [muse_glimmer_prompt(messages)],
    images = [[test_frame]],
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

streamer = TextStreamer(processor.tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = streamer, max_new_tokens = 256,
                   use_cache = True, do_sample = False)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the adapters, use `save_pretrained` for a local save or `push_to_hub` for an online save.
# 
# **Note:** this saves only the LoRA adapters, not the full model. For the merged 16-bit model,
# scroll down.

# In[ ]:


model.save_pretrained("muse_glimmer_lora")      # Local saving
processor.save_pretrained("muse_glimmer_lora")
# model.push_to_hub("your_name/muse_glimmer_lora", private = True)
# processor.push_to_hub("your_name/muse_glimmer_lora", private = True)


# To load the adapters back for inference, change `False` to `True`:

# In[ ]:


if False:
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name = "muse_glimmer_multimodal_lora",
        load_in_4bit = True,
        max_seq_length = 2048,
        offload_embedding = True,
    )
    video_processor = processor.video_processor
    video_processor.patch_temporal = video_processor.temporal_patch_size
    video_processor.num_frames = 8
    video_processor.fps = 1.0


# ### Saving to 16-bit
# 
# `save_pretrained_merged` folds the adapters back into the base weights and writes a normal
# 16-bit checkpoint you can serve.
# 
# Two things to know before you run it. The merge reads the **16-bit** base, not the 4bit one we
# trained on, and Muse Glimmer has no entry in Unsloth's 4bit-to-16bit name map yet, so we point
# `_name_or_path` at `unsloth/Muse-Glimmer-30B` by hand. And the output is about 56 GB, so you need that much
# free disk plus room to stage the download.
# 
# GGUF export is not available for Muse Glimmer yet, since the architecture is not in upstream
# `llama.cpp`. Use the merged 16-bit save below instead.

# In[ ]:


# Select ONLY 1 to save! (Both not needed!)

# The merge needs the 16bit base weights; Muse Glimmer is not in the 4bit-to-16bit name map yet.
model.config._name_or_path = "unsloth/Muse-Glimmer-30B"

# Save locally to 16bit
if False: model.save_pretrained_merged("muse_glimmer_multimodal_finetune", processor)

# Push to your own private Hugging Face repo
if False:
    model.push_to_hub_merged(
        "your_name/muse_glimmer_multimodal_finetune", processor, private = True,
    )


# And we are done. If you have any questions on Unsloth, we have a
# [Discord](https://discord.gg/unsloth) channel.
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
#   Join Discord if you need help + star us on <a href="https://github.com/unslothai/unsloth">Github</a>
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
