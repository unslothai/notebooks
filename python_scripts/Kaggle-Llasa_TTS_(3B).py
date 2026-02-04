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
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss faster with 32% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
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
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.48\n!pip install --no-deps trl==0.22.2\n!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps\n!pip install omegaconf torchcodec "datasets>=3.4.1,<4.0.0"\n%env UNSLOTH_DISABLE_FAST_GENERATION = 1\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!
# 
# Thank you to [Etherl](https://huggingface.co/Etherll) for creating this notebook!

# In[ ]:


from unsloth import FastLanguageModel
import torch
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

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llasa-3B",
    max_seq_length = max_seq_length,
    dtype = None, # Select None for auto detection
    load_in_4bit = False, # Choose True for 4bit which reduces memory
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
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
# We will use the `MrDragonFox/Elise`, which is designed for training TTS models. Ensure that your dataset follows the required format:
# **text, audio**. You can modify this section to accommodate your own dataset, but maintaining the correct structure is essential for optimal training.

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("MrDragonFox/Elise", split = "train")
OUTPUT_DIR = 'processed_data_memmap'


# In[ ]:


#@title Tokenization Function

import os
import numpy as np
from datasets import load_dataset
import torch
import torchaudio
from transformers import AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model
from tqdm import tqdm
from torch.utils.data import Dataset

XCODEC2_MODEL_NAME = "HKUST-Audio/xcodec2"
SAMPLE_RATE = 16000
DEVICE = "cuda"

def preprocess_and_save(
    dataset,
    output_dir: str,
    tokenizer: AutoTokenizer,
    codec_model: XCodec2Model,
    sample_rate: int = 16000,
    max_length: int = 2048,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    codec_model = codec_model.to(device).eval()
    os.makedirs(output_dir, exist_ok = True)
    memmap_path = os.path.join(output_dir, f"input_ids.memmap")
    shape_path = os.path.join(output_dir, f"input_ids_shape.npy")
    num_samples = len(dataset)
    shape = (num_samples, max_length)
    try:
        arr = np.memmap(memmap_path, dtype = np.int32, mode = 'w+', shape = shape)
    except Exception as e:
        raise e
    valid_sequences_count = 0
    skipped_count = 0
    for idx, example in tqdm(enumerate(dataset), total = num_samples, desc = f"Processing"):
        try:
            if 'text' not in example or example['text'] is None:
                skipped_count += 1
                continue
            text = f"<|TEXT_UNDERSTANDING_START|>{example['text']}<|TEXT_UNDERSTANDING_END|>"
            text_ids = tokenizer.encode(text, add_special_tokens = False)
            if "audio" not in example or "array" not in example["audio"] or "sampling_rate" not in example["audio"] or example["audio"]["array"] is None:
                skipped_count += 1
                continue
            waveform = torch.tensor(example["audio"]["array"]).float()
            original_sr = example["audio"]["sampling_rate"]
            if original_sr != sample_rate:
              waveform = torchaudio.functional.resample(waveform, original_sr, sample_rate)

            original_shape_after_resample = waveform.shape
            waveform = waveform.squeeze()
            if waveform.dim() == 0:
                skipped_count += 1
                continue
            elif waveform.dim() > 1:
                waveform = waveform[0]
                if waveform.dim() != 1:
                    skipped_count += 1
                    continue
            final_waveform = waveform.unsqueeze(0).to(device)
            speech_codes = None
            with torch.inference_mode():
                speech_codes_raw = codec_model.encode_code(final_waveform)
                speech_codes = speech_codes_raw[0][0]

            codes_np = speech_codes.cpu().numpy()
            speech_token_ids = [f"<|s_{code}|>" for code in codes_np]
            speech_token_ids = tokenizer.convert_tokens_to_ids(speech_token_ids)
            speech_ids = (
                [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
                + speech_token_ids
                + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
            )
            max_text_space = max_length - len(speech_ids)
            if max_text_space < 0:
                skipped_count += 1
                continue
            truncated_text_ids = text_ids[:max_text_space]
            combined_sequence = truncated_text_ids + speech_ids
            padding_length = max_length - len(combined_sequence)
            final_sequence = combined_sequence + [tokenizer.pad_token_id] * padding_length
            final_sequence = final_sequence[:max_length]
            arr[valid_sequences_count] = np.array(final_sequence, dtype = np.int32)
            valid_sequences_count += 1
        except Exception as e:
            skipped_count += 1
            continue
        arr.flush()
        actual_shape = (valid_sequences_count, max_length)
        np.save(shape_path, np.array(actual_shape))


class MemmapTTSDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token_id is None:
             if self.tokenizer.eos_token_id is not None:
                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
             else:
                 raise ValueError("Tokenizer passed to Dataset must have pad_token_id set.")

        self.pad_token_id = self.tokenizer.pad_token_id
        self.ignore_index = -100

        memmap_file = os.path.join(data_path, f'input_ids.memmap')
        shape_file = os.path.join(data_path, f'input_ids_shape.npy')

        if not os.path.exists(memmap_file) or not os.path.exists(shape_file):
             raise FileNotFoundError(f"Required files not found  in {data_path}")

        self.shape = tuple(np.load(shape_file))
        if not self.shape or len(self.shape) != 2 or self.shape[0] == 0:
             self.length = 0
             self.memmap_data = None
        else:
             self.memmap_data = np.memmap(memmap_file, dtype = 'int32', mode = 'r', shape = self.shape)
             self.length = self.shape[0]

        try:
            self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
            self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
            self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
            self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
            assert isinstance(self.pad_token_id, int)
        except Exception as token_err:
            raise ValueError(f"Tokenizer is missing required special tokens or pad_token_id. Error: {token_err}")

    def __len__(self):
        return self.length

    def replace_tagged_token(self, token_list, target_token_id, new_sequence_ids):
        if isinstance(new_sequence_ids, torch.Tensor):
            new_sequence_ids = new_sequence_ids.tolist()
        try:
            idx = token_list.index(target_token_id)
            return token_list[:idx] + new_sequence_ids + token_list[idx+1:]
        except ValueError:
            return token_list

    def pad_sequence_torch(self, sequence, max_length, value = 0):
        current_len = len(sequence)
        if current_len >= max_length:
            return sequence[:max_length]
        else:
            padding_size = max_length - current_len
            padding = torch.full((padding_size,), value, dtype = sequence.dtype, device = sequence.device)
            return torch.cat([sequence, padding], dim = 0)

    def __getitem__(self, idx):
        if self.memmap_data is None or idx >= self.length:
            raise IndexError(f"Index out of bounds (length={self.length}).")

        raw_input_ids_np = self.memmap_data[idx]
        input_ids_tensor_raw = torch.tensor(raw_input_ids_np, dtype = torch.long)
        input_ids = None
        speech_gen_idx_in_final = -1

        try:
            text_start_idx = (input_ids_tensor_raw == self.text_understanding_start_id).nonzero(as_tuple = True)[0][0].item()
            text_end_idx = (input_ids_tensor_raw == self.text_understanding_end_id).nonzero(as_tuple = True)[0][0].item()
            speech_start_idx = (input_ids_tensor_raw == self.speech_generation_start_id).nonzero(as_tuple = True)[0][0].item()

            speech_end_marker_indices = (input_ids_tensor_raw == self.speech_generation_end_id).nonzero(as_tuple = True)[0]
            pad_start_indices = (input_ids_tensor_raw == self.pad_token_id).nonzero(as_tuple = True)[0]

            if len(speech_end_marker_indices) > 0:
                 speech_end_idx = speech_end_marker_indices[0].item()
            elif len(pad_start_indices) > 0:
                 speech_end_idx = pad_start_indices[0].item() - 1
            else:
                 speech_end_idx = len(input_ids_tensor_raw) - 1

            if not (text_start_idx < text_end_idx < speech_start_idx < speech_end_idx < len(input_ids_tensor_raw)):
                 raise ValueError("Parsed indices are invalid or out of order")

            original_text_sequence = input_ids_tensor_raw[:speech_start_idx]
            original_speech_sequence_with_markers = input_ids_tensor_raw[speech_start_idx : speech_end_idx +1]
            chat = [
                {"role": "user", "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
            ]

            try:
                 import inspect
                 sig = inspect.signature(self.tokenizer.apply_chat_template)
                 if 'add_generation_prompt' in sig.parameters:
                     templated_ids_list = self.tokenizer.apply_chat_template(chat, tokenize = True, add_generation_prompt = False)
                 else:
                     templated_ids_list = self.tokenizer.apply_chat_template(chat, tokenize = True)

                 final_ids_list = self.replace_tagged_token(templated_ids_list, self.text_understanding_start_id, original_text_sequence.tolist())
                 final_ids_list = self.replace_tagged_token(final_ids_list, self.speech_generation_start_id, original_speech_sequence_with_markers.tolist())
                 input_ids = torch.tensor(final_ids_list, dtype = torch.long)

                 try:
                     speech_gen_idx_in_final = (input_ids == self.speech_generation_start_id).nonzero(as_tuple = True)[0][0].item()
                 except IndexError:
                      speech_gen_idx_in_final = -1
            except Exception:
                 input_ids = input_ids_tensor_raw
                 try:
                     speech_gen_idx_in_final = (input_ids == self.speech_generation_start_id).nonzero(as_tuple = True)[0][0].item()
                 except IndexError:
                     speech_gen_idx_in_final = -1

        except Exception:
            input_ids = input_ids_tensor_raw
            try:
                speech_gen_idx_in_final = (input_ids == self.speech_generation_start_id).nonzero(as_tuple = True)[0][0].item()
            except IndexError:
                speech_gen_idx_in_final = -1

        if input_ids is None:
            input_ids = input_ids_tensor_raw
            try:
                speech_gen_idx_in_final = (input_ids == self.speech_generation_start_id).nonzero(as_tuple = True)[0][0].item()
            except IndexError:
                speech_gen_idx_in_final = -1

        labels = torch.full_like(input_ids, self.ignore_index)
        if speech_gen_idx_in_final != -1 and speech_gen_idx_in_final < len(input_ids):
             labels[speech_gen_idx_in_final:] = input_ids[speech_gen_idx_in_final:].clone()

        attention_mask = (input_ids != self.pad_token_id).long()
        labels[input_ids == self.pad_token_id] = self.ignore_index

        input_ids = self.pad_sequence_torch(input_ids, self.max_length, value = self.pad_token_id)
        attention_mask = self.pad_sequence_torch(attention_mask, self.max_length, value = 0)
        labels = self.pad_sequence_torch(labels, self.max_length, value = self.ignore_index)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }



try:
    codec_model = XCodec2Model.from_pretrained(XCODEC2_MODEL_NAME)

except Exception as e:
    raise f"ERROR loading XCodec2 model: {e}."

preprocess_and_save(
        dataset = dataset,
        output_dir = OUTPUT_DIR,
        tokenizer = tokenizer,
        codec_model = codec_model,
        sample_rate = SAMPLE_RATE,
        max_length = max_seq_length,
        device = DEVICE
    )
try:
    train_dataset = MemmapTTSDataset(
        data_path = OUTPUT_DIR,
        tokenizer = tokenizer,
        max_length = max_seq_length
     )
    print(f"Dataset loaded for split 'train'. Number of samples: {len(train_dataset)}")
except Exception as e:
    print(f"Error initializing dataset: {e}")
print("Moving XCodec2 model to cpu")
codec_model.to('cpu')
torch.cuda.empty_cache()


# <a name="Train"></a>
# ### Train the model
# Now let's use Hugging Face `Trainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[ ]:


from transformers import TrainingArguments, Trainer
trainer = Trainer(
    model = model,
    train_dataset = train_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 5e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
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
# Let's run the model! You can change the prompts
# 

# In[ ]:


input_text = "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person."


# In[ ]:


#@title Run Inference
import soundfile as sf

from IPython.display import display, Audio
FastLanguageModel.for_inference(model)

def ids_to_speech_tokens(speech_ids):

    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

#TTS start!
with torch.inference_mode():
    with torch.amp.autocast('cuda',dtype = model.dtype):
        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize = True,
            return_tensors = 'pt',
            continue_final_message = True
        )
        input_ids = input_ids.to('cuda')

        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length = 2048,  # We trained our model with a max length of 2048
            eos_token_id = speech_end_id ,
            do_sample = True,
            top_p = 1.2,           #  Adjusts the diversity of generated content
            temperature = 1.2,   #  Controls randomness in output
        )
    # Extract the speech tokens
    generated_ids = outputs[0][input_ids.shape[1]:-1]

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

    # Convert  token <|s_23456|> to int 23456
    speech_tokens = extract_speech_ids(speech_tokens)

    speech_tokens = torch.tensor(speech_tokens).cpu().unsqueeze(0).unsqueeze(0)

    # Decode the speech tokens to speech waveform
    gen_wav = codec_model.decode_code(speech_tokens)

sf.write("output.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)

display(Audio(gen_wav[0, 0, :].cpu().numpy(), rate = 16000))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("llasa_tts_lora")  # Local saving
tokenizer.save_pretrained("llasa_tts_lora")
# model.push_to_hub("your_name/llasa_tts_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/llasa_tts_lora", token = "YOUR_HF_TOKEN") # Online saving


# ### Saving to float16
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("llasa_tts_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("HF_USERNAME/llasa_tts_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("llasa_tts_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/llasa_tts_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("llasa_tts_lora")
    tokenizer.save_pretrained("llasa_tts_lora")
if False:
    model.push_to_hub("HF_USERNAME/llasa_tts_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/llasa_tts_lora", token = "YOUR_HF_TOKEN")


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
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
# 
