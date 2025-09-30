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
# Unsloth now supports [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which automatically creates kernels!
# 
# [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) is now supported! Train Qwen2.5-VL, Gemma 3 etc. with GSPO or GRPO.
# 
# Introducing Unsloth [Standby for RL](https://docs.unsloth.ai/basics/memory-efficient-rl): GRPO is now faster, uses 30% less memory with 2x longer context.
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
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.55.4\n!pip uninstall -y unsloth unsloth_zoo\n!pip install "git+https://github.com/Erland366/unsloth.git@feat/distributed_with_lock"\n!pip install "git+https://github.com/Erland366/unsloth-zoo.git@feat/distributed_with_lock"\n!pip install git+https://github.com/Erland366/nbdistributed.git\n!pip install --no-deps trl==0.22.2\n!git clone https://github.com/SparkAudio/Spark-TTS\n!pip install omegaconf einx\n')
# 
# 
# # In[ ]:
# 
# 
# # Let's start our distributed runtime
# get_ipython().run_line_magic('load_ext', 'nbdistributed')
# get_ipython().run_line_magic('dist_init', '-n 2')
# import time
# 
# time.sleep(5)
# get_ipython().run_line_magic('dist_status', '')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!
# 
# Thank you to [Etherl](https://huggingface.co/Etherll) for creating this notebook!

# In[ ]:


from unsloth import FastModel
import torch
from huggingface_hub import snapshot_download

max_seq_length = 2048 # Choose any for long context!

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Qwen3 new models
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # Other very popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

# Download model and code
snapshot_download("unsloth/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B")

model, tokenizer = FastModel.from_pretrained(
    model_name = f"Spark-TTS-0.5B/LLM",
    use_exact_model_name = True,
    max_seq_length = max_seq_length,
    dtype = torch.float32, # Spark seems to only work on float32 for now
    full_finetuning = True, # We support full finetuning now!
    load_in_4bit = False,
    #token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


#LoRA does not work with float32 only works with bfloat16 !!!
model = FastModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
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
# 
# We will use the `MrDragonFox/Elise`, which is designed for training TTS models. Ensure that your dataset follows the required format: **text, audio** for single-speaker models or **source, text, audio** for multi-speaker models. You can modify this section to accommodate your own dataset, but maintaining the correct structure is essential for optimal training.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("MrDragonFox/Elise", split = "train")


# In[ ]:


#@title Tokenization Function

import locale
import torchaudio.transforms as T
import os
import torch
import sys
import numpy as np
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")
def extract_wav2vec2_features( wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""

        if wavs.shape[0] != 1:

             raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
        wav_np = wavs.squeeze(0).cpu().numpy()

        processed = audio_tokenizer.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values

        input_values = input_values.to(audio_tokenizer.feature_extractor.device)

        model_output = audio_tokenizer.feature_extractor(
            input_values,
        )


        if model_output.hidden_states is None:
             raise ValueError("Wav2Vec2Model did not return hidden states. Ensure config `output_hidden_states=True`.")

        num_layers = len(model_output.hidden_states)
        required_layers = [11, 14, 16]
        if any(l >= num_layers for l in required_layers):
             raise IndexError(f"Requested hidden state indices {required_layers} out of range for model with {num_layers} layers.")

        feats_mix = (
            model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]
        ) / 3

        return feats_mix
def formatting_audio_func(example):
    text = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
    audio_array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    target_sr = audio_tokenizer.config['sample_rate']

    if sampling_rate != target_sr:
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
        audio_tensor_temp = torch.from_numpy(audio_array).float()
        audio_array = resampler(audio_tensor_temp).numpy()

    if audio_tokenizer.config["volume_normalize"]:
        audio_array = audio_volume_normalize(audio_array)

    ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)

    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(audio_tokenizer.device)
    ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(audio_tokenizer.device)


    feat = extract_wav2vec2_features(audio_tensor)

    batch = {

        "wav": audio_tensor,
        "ref_wav": ref_wav_tensor,
        "feat": feat.to(audio_tokenizer.device),
    }


    semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)

    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()] # Squeeze batch dim
    )
    semantic_tokens = "".join(
        [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()] # Squeeze batch dim
    )

    inputs = [
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|end_semantic_token|>",
        "<|im_end|>"
    ]
    inputs = "".join(inputs)
    return {"text": inputs}


dataset = dataset.map(formatting_audio_func, remove_columns=["audio"])
print("Moving Bicodec model and Wav2Vec2Model to cpu.")
audio_tokenizer.model.cpu()
audio_tokenizer.feature_extractor.cpu()
torch.cuda.empty_cache()


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!

# In[ ]:


from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = False, # We're doing full float32 s disable mixed precision
        bf16 = False, # We're doing full float32 s disable mixed precision
        logging_steps = 1,
        optim = "adamw_torch_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
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
# Let's run the model! You can change the prompts
# 

# In[ ]:


input_text = "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person."

chosen_voice = None # None for single-speaker


# In[ ]:


#@title Run Inference

import torch
import re
import numpy as np
from typing import Dict, Any
import torchaudio.transforms as T

FastModel.for_inference(model) # Enable native 2x faster inference

@torch.inference_mode()
def generate_speech_from_text(
    text: str,
    temperature: float = 0.8,   # Generation temperature
    top_k: int = 50,            # Generation top_k
    top_p: float = 1,        # Generation top_p
    max_new_audio_tokens: int = 2048, # Max tokens for audio part
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> np.ndarray:
    """
    Generates speech audio from text using default voice control parameters.

    Args:
        text (str): The text input to be converted to speech.
        temperature (float): Sampling temperature for generation.
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
        max_new_audio_tokens (int): Max number of new tokens to generate (limits audio length).
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Generated waveform as a NumPy array.
    """

    torch.compiler.reset()

    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

    print("Generating token sequence...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_audio_tokens, # Limit generation length
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id, # Stop token
        pad_token_id=tokenizer.pad_token_id # Use models pad token id
    )
    print("Token sequence generated.")


    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]


    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    # print(f"\nGenerated Text (for parsing):\n{predicts_text}\n") # Debugging

    # Extract semantic token IDs using regex
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        print("Warning: No semantic tokens found in the generated output.")
        # Handle appropriately - perhaps return silence or raise error
        return np.array([], dtype=np.float32)

    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0) # Add batch dim

    # Extract global token IDs using regex (assuming controllable mode also generates these)
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
    if not global_matches:
         print("Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail.")
         pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
    else:
         pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0) # Add batch dim

    pred_global_ids = pred_global_ids.unsqueeze(0) # Shape becomes (1, 1, N_global)

    print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
    print(f"Found {pred_global_ids.shape[2]} global tokens.")


    # 5. Detokenize using BiCodecTokenizer
    print("Detokenizing audio tokens...")
    # Ensure audio_tokenizer and its internal model are on the correct device
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    # Squeeze the extra dimension from global tokens as seen in SparkTTS example
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(device).squeeze(0), # Shape (1, N_global)
        pred_semantic_ids.to(device)           # Shape (1, N_semantic)
    )
    print("Detokenization complete.")

    return wav_np

if __name__ == "__main__":
    print(f"Generating speech for: '{input_text}'")
    text = f"{chosen_voice}: " + input_text if chosen_voice else input_text
    generated_waveform = generate_speech_from_text(input_text)

    if generated_waveform.size > 0:
        import soundfile as sf
        output_filename = "generated_speech_controllable.wav"
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(output_filename, generated_waveform, sample_rate)
        print(f"Audio saved to {output_filename}")

        # Optional: Play in notebook
        from IPython.display import Audio, display
        display(Audio(generated_waveform, rate=sample_rate))
    else:
        print("Audio generation failed (no tokens found?).")


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# ### Saving to float16
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
