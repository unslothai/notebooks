#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Oute_TTS_(1B).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
# # In[1]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth\nelse:\n    # Do this only in Colab notebooks! Otherwise use pip install unsloth\n    import torch; v = re.match(r"[0-9\\.]{3,}", str(torch.__version__)).group(0)\n    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")\n    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo\n    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n!pip install omegaconf einx\n!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS\nimport os\nos.remove("/content/OuteTTS/outetts/models/gguf_model.py")\nos.remove("/content/OuteTTS/outetts/interface.py")\nos.remove("/content/OuteTTS/outetts/__init__.py")\n!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy\n!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps\n%env UNSLOTH_DISABLE_FAST_GENERATION = 1\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!
# 
# Thank you to [Etherl](https://huggingface.co/Etherll) for creating this notebook!

# In[2]:


from unsloth import FastModel
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

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Llama-OuteTTS-1.0-1B",
    max_seq_length = max_seq_length,
    dtype = None, # Set to None for auto detection
    load_in_4bit = False, # Set to True for 4bit which reduces memory
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[3]:


model = FastModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj",],
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
# We will use the `MrDragonFox/Elise`, which is designed for training TTS models. Ensure that your dataset follows the required format: **text, audio**, but maintaining the correct structure is essential for optimal training.

# In[4]:


from datasets import load_dataset,Audio,Dataset
dataset = load_dataset("MrDragonFox/Elise", split = "train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))


# In[5]:


#@title Tokenization Function

import torch
from tqdm import tqdm
import io
import tempfile
from datasets import Dataset
import sys
sys.path.append('OuteTTS')
import os
import dac
# V3 Imports
from outetts.version.v3.audio_processor import AudioProcessor
from outetts.version.v3.prompt_processor import PromptProcessor
from outetts.dac.interface import DacInterface
from outetts.models.config import ModelConfig # Need a dummy config for AudioProcessor
import whisper
from outetts.utils.preprocessing import text_normalizations
import soundfile as sf
import numpy as np

class DataCreationV3:
    def __init__(
            self,
            model_tokenizer_path: str,
            whisper_model_name: str = "turbo",
            device: str = None
        ):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create a dummy ModelConfig mainly for device and paths needed by AudioProcessor/DacInterface
        dummy_config = ModelConfig(
            tokenizer_path=model_tokenizer_path,
            device=self.device,
            audio_codec_path=None # Let AudioProcessor use default DAC path
        )
        self.audio_processor = AudioProcessor(config=dummy_config)
        self.prompt_processor = PromptProcessor(model_tokenizer_path)

        print(f"Loading Whisper model: {whisper_model_name} on {self.device}")
        self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)
        print("Whisper model loaded.")

    # Renamed and adapted from the previous version
    def create_speaker_representation(self, audio_bytes: bytes, transcript: str):
        """
        Creates a v3-compatible speaker dictionary using Whisper and AudioProcessor.
        """
        if not audio_bytes or not transcript:
             print("Missing audio bytes or transcript in create_speaker_representation.")
             return None

        # Whisper needs a file path, so save bytes to a temporary file
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio_file:
                tmp_audio_file.write(audio_bytes)
                tmp_audio_file.flush() # Ensure data is written

                # 1. Get word timings using Whisper
                whisper_result = self.whisper_model.transcribe(tmp_audio_file.name, word_timestamps=True)
                # Use the provided transcript for consistency, but Whisper timings
                normalized_transcript = text_normalizations(transcript)

                words_with_timings = []
                if whisper_result and 'segments' in whisper_result:
                    for segment in whisper_result['segments']:
                        if 'words' in segment:
                            for word_info in segment['words']:
                                # Use original word casing/punctuation from Whisper's output if needed,
                                # but strip excess whitespace for consistency.
                                cleaned_word = word_info['word'].strip()
                                if cleaned_word: # Ignore empty strings
                                    words_with_timings.append({
                                        'word': cleaned_word,
                                        'start': float(word_info['start']),
                                        'end': float(word_info['end'])
                                    })
                else:
                    print(f"Whisper did not return segments/words for: {transcript[:50]}...")
                    return None # Indicate failure

                if not words_with_timings:
                    print(f"No word timings extracted by Whisper for: {transcript[:50]}...")
                    return None

                # Prepare data dict for AudioProcessor
                speaker_data_dict = {
                    "audio": {"bytes": audio_bytes},
                    "text": normalized_transcript, # Use the potentially normalized transcript
                    "words": words_with_timings
                }

                # 2. Use AudioProcessor to create the speaker representation
                v3_speaker = self.audio_processor.create_speaker_from_dict(speaker_data_dict)
                return v3_speaker

        except Exception as e:
            print(f"Error during speaker creation (Whisper/AudioProcessor): {e}")
            return None # Indicate failure


    # --- V3 Changes: run method is now a generator ---
    def process_dataset(self, dataset: Dataset):
        """
        Processes a Hugging Face Dataset object in memory and yields training prompts.

        Args:
            dataset (Dataset): The Hugging Face dataset to process.
                               Expected columns: 'text' (str) and 'audio' (dict with 'bytes').

        Yields:
            str: The processed training prompt string for each valid row.
        """
        processed_count = 0
        skipped_count = 0

        # Iterate directly over the dataset
        for i, item in enumerate(tqdm(dataset, desc="Processing Dataset")):
            try:
                # --- Adapt to your dataset's column names ---
                transcript = item.get('text')
                audio_info = item.get('audio')
                # --- End Adapt ---

                if not transcript or not isinstance(transcript, str):
                    print(f"Row {i}: Skipping due to missing or invalid 'text' column.")
                    skipped_count += 1
                    continue

                audio_array = audio_info['array']
                buffer = io.BytesIO()
                # Ensure array is float32 for common compatibility, adjust subtype if needed
                sf.write(buffer, audio_array.astype(np.float32), audio_info['sampling_rate'], format='WAV', subtype='FLOAT')
                buffer.seek(0)
                audio_bytes = buffer.getvalue()

                # Create speaker representation
                speaker = self.create_speaker_representation(audio_bytes, transcript)

                if speaker is None:
                    print(f"Row {i}: Failed to create speaker representation for text: {transcript[:50]}... Skipping.")
                    skipped_count += 1
                    continue

                # Get the V3 training prompt
                prompt = self.prompt_processor.get_training_prompt(speaker)

                processed_count += 1
                yield prompt # Yield the processed prompt string

            except KeyboardInterrupt:
                 print("Processing interrupted by user.")
                 break
            except Exception as e:
                print(f"Row {i}: Unhandled error processing item: {e}", exc_info=True)
                skipped_count += 1
                # Decide if you want to stop on errors or just skip
                continue

        print(f"Dataset processing finished. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":

    _MODEL_TOKENIZER_PATH = "OuteAI/Llama-OuteTTS-1.0-1B"
    _WHISPER_MODEL = "turbo" # Or "small.en", "medium.en", "large-v2", etc.


    data_processor = DataCreationV3(
        model_tokenizer_path=_MODEL_TOKENIZER_PATH,
        whisper_model_name=_WHISPER_MODEL
    )

    # Process the dataset and collect prompts (or process iteratively)
    all_prompts = []
    print("Starting dataset processing...")
    procced_dataset = data_processor.process_dataset(dataset)
    for prompt in procced_dataset:
        if prompt:
             all_prompts.append({'text': prompt})
    dataset = Dataset.from_list(all_prompts)
    print("Moving Whisper model to CPU")
    data_processor.whisper_model.to('cpu')
    torch.cuda.empty_cache()


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!

# In[6]:


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
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# In[7]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


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
# Let's run the model! You can change the prompts
# 

# In[10]:


input_text = "Hey there my name is Elise, and I'm a speech generation model that can sound like a person."


# In[11]:


#@title Run Inference

import torch
import re
import numpy as np
from typing import Dict, Any
import torchaudio.transforms as T
from transformers import LogitsProcessor
import transformers.generation.utils as generation_utils
from transformers import AutoModelForCausalLM
import re
FastModel.for_inference(model)

def get_audio(tokens):
        decoded_output = tokenizer.batch_decode(tokens, skip_special_tokens=False)[0]
        c1 = list(map(int,re.findall(r"<\|c1_(\d+)\|>", decoded_output)))
        c2 = list(map(int,re.findall(r"<\|c2_(\d+)\|>", decoded_output)))

        t = min(len(c1), len(c2))
        c1 = c1[:t]
        c2 = c2[:t]
        output = [c1,c2]
        if not output:
            print("No audio tokens found in the output")
            return None

        return data_processor.audio_processor.audio_codec.decode(
            torch.tensor([output], dtype=torch.int64).to(data_processor.audio_processor.audio_codec.device)
        )

class RepetitionPenaltyLogitsProcessorPatch(LogitsProcessor):
    def __init__(self, penalty: float):
        penalty_last_n = 64
        print("🔄 Using patched RepetitionPenaltyLogitsProcessor -> RepetitionPenaltyLogitsProcessorPatch | penalty_last_n: {penalty_last_n}")
        if penalty_last_n is not None:
            if not isinstance(penalty_last_n, int) or penalty_last_n < 0:
                raise ValueError(f"`penalty_last_n` has to be a non-negative integer, but is {penalty_last_n}")
        if not isinstance(penalty, float) or penalty <= 0:
            raise ValueError(f"`penalty` has to be a positive float, but is {penalty}")

        self.penalty_last_n = penalty_last_n
        self.penalty = penalty

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor`):
                Indices of input sequence tokens in the vocabulary (shape `(batch_size, sequence_length)`).
            scores (`torch.FloatTensor`):
                Prediction scores of a language modeling head (shape `(batch_size, vocab_size)`).

        Returns:
            `torch.FloatTensor`: The modified prediction scores.
        """
        # Check if penalties should be applied
        if self.penalty_last_n == 0 or self.penalty == 1.0:
            return scores

        batch_size, seq_len = input_ids.shape
        vocab_size = scores.shape[-1]

        # Process each batch item independently
        for b in range(batch_size):
            # 1. Determine the penalty window
            start_index = max(0, seq_len - self.penalty_last_n)
            window_indices = input_ids[b, start_index:] # Shape: (window_len,)

            if window_indices.numel() == 0: # Skip if window is empty
                continue

            # 2. Find unique tokens within the window
            tokens_in_window = set(window_indices.tolist())

            # 3. Apply repetition penalty to the scores for this batch item
            for token_id in tokens_in_window:
                if token_id >= vocab_size:
                    continue

                logit = scores[b, token_id]

                if logit <= 0:
                    logit *= self.penalty
                else:
                    logit /= self.penalty

                # Update the score
                scores[b, token_id] = logit

        return scores

generation_utils.RepetitionPenaltyLogitsProcessor = RepetitionPenaltyLogitsProcessorPatch
AutoModelForCausalLM.generate = generation_utils.GenerationMixin.generate

if __name__ == "__main__":
    formated_text = "<|text_start|>"+input_text+"<|text_end|>"
    prompt = "\n".join([
        "<|im_start|>",
        formated_text,
        "<|audio_start|><|global_features_start|>",
    ])
    with torch.inference_mode():
        with torch.amp.autocast('cuda',dtype=model.dtype):
          model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

          print("Generating token sequence...")
          generated_ids = model.generate(
              **model_inputs,
              temperature=0.4,
              top_k=40,
              top_p=0.9,
              repetition_penalty=1.1,
              min_p=0.05,
              max_new_tokens=2048, # Limit generation length
          )
          print("Token sequence generated.")


    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    audio = get_audio(generated_ids)
    audio = audio.cpu()
    from IPython.display import Audio, display
    display(Audio(audio.squeeze(0), rate=24000))


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[12]:


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# ### Saving to float16
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[13]:


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
