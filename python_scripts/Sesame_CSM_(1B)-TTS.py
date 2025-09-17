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
# [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) is now supported! Train Qwen2.5-VL, Gemma 3 etc. with GSPO or GRPO.
# 
# Introducing Unsloth [Standby for RL](https://docs.unsloth.ai/basics/memory-efficient-rl): GRPO is now faster, uses 30% less memory with 2x longer context.
# 
# Gpt-oss fine-tuning now supports 8× longer context with 0 accuracy loss. [Read more](https://docs.unsloth.ai/basics/long-context-gpt-oss-training)
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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth\nelse:\n    # Do this only in Colab notebooks! Otherwise use pip install unsloth\n    import torch; v = re.match(r"[0-9\\.]{3,}", str(torch.__version__)).group(0)\n    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")\n    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo\n    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth\n!pip install transformers==4.52.3\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# In[21]:


from unsloth import FastModel
from transformers import CsmForConditionalGeneration
import torch

model, processor = FastModel.from_pretrained(
    model_name = "unsloth/csm-1b",
    max_seq_length= 2048, # Choose any for long context!
    dtype = None, # Leave as None for auto-detection
    auto_model = CsmForConditionalGeneration,
    load_in_4bit = False, # Select True for 4bit - reduces memory usage
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[4]:


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


# <a name="Data"></a>
# ### Data Prep  
# 
# We will use the `MrDragonFox/Elise`, which is designed for training TTS models. Ensure that your dataset follows the required format: **text, audio** for single-speaker models or **source, text, audio** for multi-speaker models. You can modify this section to accommodate your own dataset, but maintaining the correct structure is essential for optimal training.

# In[ ]:


#@title Dataset Prep functions
from datasets import load_dataset, Audio, Dataset
import os
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("unsloth/csm-1b")

raw_ds = load_dataset("MrDragonFox/Elise", split="train")

# Getting the speaker id is important for multi-speaker models and speaker consistency
speaker_key = "source"
if "source" not in raw_ds.column_names and "speaker_id" not in raw_ds.column_names:
    print("Unsloth: No speaker found, adding default \"source\" of 0 for all examples")
    new_column = ["0"] * len(raw_ds)
    raw_ds = raw_ds.add_column("source", new_column)
elif "source" not in raw_ds.column_names and "speaker_id" in raw_ds.column_names:
    speaker_key = "speaker_id"

target_sampling_rate = 24000
raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

def preprocess_example(example):
    conversation = [
        {
            "role": str(example[speaker_key]),
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }
    ]

    try:
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs = {
                "padding": "max_length", # pad to the max_length
                "max_length": 256, # this should be the max length of audio
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs = {
                "sampling_rate": 24_000,
                "max_length": 240001, # max input_values length of the whole dataset
                "padding": "max_length",
            },
            common_kwargs = {"return_tensors": "pt"},
        )
    except Exception as e:
        print(f"Error processing example with text '{example['text'][:50]}...': {e}")
        return None

    required_keys = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
    processed_example = {}
    # print(model_inputs.keys())
    for key in required_keys:
        if key not in model_inputs:
            print(f"Warning: Required key '{key}' not found in processor output for example.")
            return None

        value = model_inputs[key][0]
        processed_example[key] = value


    # Final check (optional but good)
    if not all(isinstance(processed_example[key], torch.Tensor) for key in processed_example):
         print(f"Error: Not all required keys are tensors in final processed example. Keys: {list(processed_example.keys())}")
         return None

    return processed_example

processed_ds = raw_ds.map(
    preprocess_example,
    remove_columns=raw_ds.column_names,
    desc="Preprocessing dataset",
)


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface  `Trainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[10]:


from transformers import TrainingArguments, Trainer
from unsloth import is_bfloat16_supported

trainer = Trainer(
    model = model,
    train_dataset = processed_ds,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01, # Turn this on if overfitting
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


# In[11]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[12]:


trainer_stats = trainer.train()


# In[13]:


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

# In[ ]:


from IPython.display import Audio, display
import soundfile as sf

text = "We just finished fine tuning a text to speech model... and it's pretty good!"
speaker_id = 0
inputs = processor(f"[{speaker_id}]{text}", add_special_tokens=True).to("cuda")
audio_values = model.generate(
    **inputs,
    max_new_tokens=125, # 125 tokens is 10 seconds of audio, for longer speech increase this
    # play with these parameters to tweak results
    # depth_decoder_top_k=0,
    # depth_decoder_top_p=0.9,
    # depth_decoder_do_sample=True,
    # depth_decoder_temperature=0.9,
    # top_k=0,
    # top_p=1.0,
    # temperature=0.9,
    # do_sample=True,
    #########################################################
    output_audio=True
)
audio = audio_values[0].to(torch.float32).cpu().numpy()
sf.write("example_without_context.wav", audio, 24000)
display(Audio(audio, rate=24000))


# In[ ]:


text = "Sesame is a super cool TTS model which can be fine tuned with Unsloth."

speaker_id = 0
# Another equivalent way to prepare the inputs
conversation = [
    {"role": str(speaker_id), "content": [{"type": "text", "text": text}]},
]
audio_values = model.generate(
    **processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    ).to("cuda"),
    max_new_tokens=125, # 125 tokens is 10 seconds of audio, for longer speech increase this
    # play with these parameters to tweak results
    # depth_decoder_top_k=0,
    # depth_decoder_top_p=0.9,
    # depth_decoder_do_sample=True,
    # depth_decoder_temperature=0.9,
    # top_k=0,
    # top_p=1.0,
    # temperature=0.9,
    # do_sample=True,
    #########################################################
    output_audio=True
)
audio = audio_values[0].to(torch.float32).cpu().numpy()
sf.write("example_without_context.wav", audio, 24000)
display(Audio(audio, rate=24000))


# #### Voice and style consistency
# 
# Sesame CSM's power comes from providing audio context for each speaker. Let's pass a sample utterance from our dataset to ground speaker identity and style.

# In[ ]:


speaker_id = 0

utterance = raw_ds[3]["audio"]["array"]
utterance_text = raw_ds[3]["text"]
text = "Sesame is a super cool TTS model which can be fine tuned with Unsloth."

# CSM will fill in the audio for the last text.
# You can even provide a conversation history back in as you generate new audio

conversation = [
    {"role": str(speaker_id), "content": [{"type": "text", "text": utterance_text},{"type": "audio", "path": utterance}]},
    {"role": str(speaker_id), "content": [{"type": "text", "text": text}]},
]

inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    )
audio_values = model.generate(
    **inputs.to("cuda"),
    max_new_tokens=125, # 125 tokens is 10 seconds of audio, for longer text increase this
    # play with these parameters to tweak results
    # depth_decoder_top_k=0,
    # depth_decoder_top_p=0.9,
    # depth_decoder_do_sample=True,
    # depth_decoder_temperature=0.9,
    # top_k=0,
    # top_p=1.0,
    # temperature=0.9,
    # do_sample=True,
    #########################################################
    output_audio=True
)
audio = audio_values[0].to(torch.float32).cpu().numpy()
sf.write("example_with_context.wav", audio, 24000)
display(Audio(audio, rate=24000))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[18]:


model.save_pretrained("lora_model")  # Local saving
processor.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# processor.push_to_hub("your_name/lora_model", token = "...") # Online saving


# ### Saving to float16
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("model", processor, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", processor, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", processor, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", processor, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    processor.save_pretrained("model")
if False:
    model.push_to_hub("hf/model", token = "")
    processor.push_to_hub("hf/model", token = "")


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
