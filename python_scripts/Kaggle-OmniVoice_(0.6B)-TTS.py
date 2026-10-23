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
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==5.15.0\n!pip install --no-deps trl==0.22.2\n!pip install --no-deps omnivoice==0.2.1\nimport importlib.metadata as _md\ntry: _tv = tuple(int(x) for x in _md.version("torch").split("+")[0].split(".")[:2])\nexcept Exception: _tv = (2, 11)\n_torchcodec = {(2, 11): "torchcodec>=0.11.0", (2, 10): "torchcodec==0.10.*", (2, 9): "torchcodec==0.9.*", (2, 8): "torchcodec==0.7.*", (2, 7): "torchcodec==0.5.*"}.get(_tv, "torchcodec>=0.12.0" if _tv >= (2, 12) else "torchcodec")\n!pip install pydub webdataset soundfile librosa "{_torchcodec}" "datasets>=3.4.1,<4.0.0"\n')
# 
# 
# # ### Unsloth
# 
# [OmniVoice](https://github.com/k2-fsa/OmniVoice) is a zero-shot TTS model for 600+ languages. It is a `Qwen3-0.6B` backbone with its own audio embeddings and audio heads over 8 audio codebooks, and unlike most TTS models it is a **diffusion language model**: attention is bidirectional and it fills in masked audio tokens over a few steps instead of generating them left to right.
# 
# We first save the Qwen3 backbone as a standalone checkpoint, so `FastModel` can load it and add LoRA adapters to it.

# In[ ]:


from unsloth import FastModel
import torch
from omnivoice import OmniVoice
from transformers import AutoTokenizer, Qwen3ForCausalLM

# Save OmniVoice's Qwen3 backbone as its own checkpoint
omnivoice = OmniVoice.from_pretrained("k2-fsa/OmniVoice", dtype = torch.float32, train = True)
backbone = Qwen3ForCausalLM(omnivoice.config.llm_config)
backbone.model.load_state_dict(omnivoice.llm.state_dict())
backbone.tie_weights()
backbone.save_pretrained("omnivoice_backbone")
AutoTokenizer.from_pretrained("k2-fsa/OmniVoice").save_pretrained("omnivoice_backbone")
del omnivoice, backbone

model, tokenizer = FastModel.from_pretrained(
    model_name = "omnivoice_backbone",
    max_seq_length = 2048,
    dtype = None, # None for auto detection. bfloat16 on Ampere+, float16 on T4
    load_in_4bit = False,
    full_finetuning = False, # We support full finetuning as well!
    attn_implementation = "sdpa", # OmniVoice's bidirectional mask needs SDPA, not Flash Attention
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
# 
# Use `FastModel` and not `FastLanguageModel` here: OmniVoice needs its bidirectional attention mask, and `FastModel` keeps it.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Now we put the LoRA backbone back into OmniVoice. The audio embeddings and audio heads are OmniVoice's own layers, so we train them fully, in float32. Loading OmniVoice without `train = True` also loads its audio tokenizer, which we need for the data and for inference.

# In[ ]:


dtype = model.get_input_embeddings().weight.dtype
omnivoice = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", dtype = dtype, attn_implementation = "sdpa", device_map = "cuda",
)
omnivoice.llm = model.base_model.model.model
omnivoice.audio_embeddings.float()
omnivoice.audio_heads.float()
for name, param in omnivoice.named_parameters():
    param.requires_grad_("lora_" in name or name.startswith(("audio_embeddings", "audio_heads")))
torch.cuda.empty_cache()

params = [p for n, p in omnivoice.named_parameters() if not n.startswith("audio_tokenizer.")]
trainable = sum(p.numel() for p in params if p.requires_grad)
total = sum(p.numel() for p in params)
print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")


# <a name="Data"></a>
# ### Data Prep
# 
# We will use the `Etherll/kaira`, which is designed for training TTS models. Ensure that your dataset follows the required format: **text, audio**. You can modify this section to accommodate your own dataset, but maintaining the correct structure is essential for optimal training.
# 
# OmniVoice works on audio tokens from the Higgs audio tokenizer at 24kHz, so we resample the audio and turn each clip into 8 rows of audio tokens (25 per second).

# In[ ]:


from datasets import load_dataset, Audio
dataset = load_dataset("Etherll/kaira", split = "train")
dataset = dataset.cast_column("audio", Audio(sampling_rate = 24000))
reference = dataset[0] # One clip of the speaker, to clone the voice from later


# In[ ]:


#@title Tokenize the audio
audio_tokenizer = omnivoice.audio_tokenizer
feature_extractor = omnivoice.feature_extractor

def tokenize_audio(example):
    inputs = feature_extractor(
        raw_audio = example["audio"]["array"],
        sampling_rate = 24000,
        return_tensors = "pt",
    ).to(audio_tokenizer.device)
    with torch.inference_mode():
        audio_codes = audio_tokenizer.encode(inputs["input_values"]).audio_codes
    return {"audio_tokens": audio_codes.squeeze(0).to(torch.int16).cpu().numpy()}

dataset = dataset.map(tokenize_audio, remove_columns = ["audio"])


# OmniVoice is trained with its own recipe: each step masks a random share of the audio tokens and learns to fill them in, keeps a random prefix of the clip as a voice prompt, and sometimes drops the text so classifier-free guidance works at inference. We reuse OmniVoice's own sample processor and collator with the settings from its fine-tuning config.

# In[ ]:


from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.data.collator import PaddingDataCollator

text_tokenizer = omnivoice.text_tokenizer
processor = OmniVoiceSampleProcessor(
    text_tokenizer = text_tokenizer,
    num_channels = omnivoice.config.num_audio_codebook,
    audio_mask_id = omnivoice.config.audio_mask_id,
    prompt_ratio_range = (0.0, 0.3), # Share of each clip kept as the voice prompt
    mask_ratio_range = (0.0, 1.0),   # Share of the rest that is masked
    drop_cond_ratio = 0.1,           # Drop text and prompt for classifier-free guidance
    language_ratio = 0.8,            # Tell the model the language most of the time
    use_pinyin_ratio = 0.0,
    instruct_ratio = 0.0,
    only_instruct_ratio = 0.0,
)

class OmniVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, language = "en"):
        self.dataset = dataset
        self.language = language
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        row = self.dataset[idx]
        return processor({
            "audio_tokens": torch.tensor(row["audio_tokens"]),
            "label": {"text": row["text"], "language_id": self.language},
        })

train_dataset = OmniVoiceDataset(dataset)
data_collator = PaddingDataCollator(processor, batch_tokens = None)


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface `Trainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and remove `max_steps` (or set it to `-1`).

# In[ ]:


from transformers import Trainer, TrainingArguments
from unsloth import is_bfloat16_supported

trainer = Trainer(
    model = omnivoice,
    train_dataset = train_dataset,
    data_collator = data_collator,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        max_grad_norm = 1.0,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        remove_unused_columns = False, # Keep OmniVoice's audio_mask and position_ids
        report_to = "none", # Use TrackIO/WandB etc
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
# Let's run the model! You can change the text. OmniVoice can clone a voice from a short clip and its transcript, or pick a voice by itself when you give it no reference. `num_step` is the number of diffusion steps (32 is the default, 16 is faster).

# In[ ]:


from IPython.display import Audio, display

input_text = "Hey there my name is Kaira, and I'm a speech generation model that can sound like a person."

reference_audio = (torch.tensor(reference["audio"]["array"], dtype = torch.float32).unsqueeze(0), 24000)
reference_text = reference["text"]

# Training is done, so the audio layers can go back to the backbone's 16bit dtype
omnivoice.audio_embeddings.to(dtype)
omnivoice.audio_heads.to(dtype)
omnivoice.eval()

cloned = omnivoice.generate(
    text = input_text,
    language = "en",
    ref_audio = reference_audio,
    ref_text = reference_text,
    num_step = 32,
)[0]
auto_voice = omnivoice.generate(text = input_text, language = "en", num_step = 32)[0]

print("Cloned from a Kaira clip:")
display(Audio(cloned, rate = omnivoice.sampling_rate))
print("No reference clip:")
display(Audio(auto_voice, rate = omnivoice.sampling_rate))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, use `save_pretrained` for a local save. The audio embeddings and audio heads were trained fully, so we save them next to the adapters.
# 
# **[NOTE]** This ONLY saves the LoRA adapters and the audio layers, and not the full model. To save a full OmniVoice checkpoint, scroll down!

# In[ ]:


model.save_pretrained("omnivoice_lora")  # Local saving of the backbone LoRA adapters
text_tokenizer.save_pretrained("omnivoice_lora")
audio_layers = {k: v for k, v in omnivoice.state_dict().items() if k.startswith(("audio_embeddings", "audio_heads"))}
torch.save(audio_layers, "omnivoice_lora/audio_layers.pt")
# model.push_to_hub("your_name/omnivoice_lora", token = "...") # Online saving


# ### Saving a full OmniVoice checkpoint
# 
# We merge the LoRA adapters into the backbone and save a complete OmniVoice checkpoint, with the text tokenizer and the audio tokenizer, in 16bit. It loads with `OmniVoice.from_pretrained` like the original model. Use `push_to_hub` to upload it to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[ ]:


import os, shutil
from huggingface_hub import snapshot_download

save_directory = "omnivoice_finetuned"
omnivoice.llm = model.merge_and_unload().model
# The audio tokenizer is attached for inference only; it ships as its own folder below
audio_tokenizer, omnivoice.audio_tokenizer = omnivoice.audio_tokenizer, None
omnivoice.save_pretrained(save_directory)
omnivoice.audio_tokenizer = audio_tokenizer
text_tokenizer.save_pretrained(save_directory)
shutil.copytree(
    os.path.join(snapshot_download("k2-fsa/OmniVoice"), "audio_tokenizer"),
    os.path.join(save_directory, "audio_tokenizer"),
    dirs_exist_ok = True,
)
# omnivoice.push_to_hub("your_name/omnivoice_finetuned", token = "...") # Then upload the folder's
# tokenizer files and audio_tokenizer/ too, e.g. with huggingface_hub's upload_folder.


# Now load the saved checkpoint like any OmniVoice model and generate:

# In[ ]:


del omnivoice, model, trainer
torch.cuda.empty_cache()

finetuned = OmniVoice.from_pretrained(save_directory, dtype = dtype, device_map = "cuda")
audio = finetuned.generate(text = input_text, language = "en", ref_audio = reference_audio, ref_text = reference_text)[0]
display(Audio(audio, rate = finetuned.sampling_rate))


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
