#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Run*" and press "*Run All*" on **AMD Dev Cloud**!
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

# Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)
# 
# <table><tr>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
# </tr></table>
# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# ### Installation

# In[1]:


get_ipython().run_cell_magic('bash', '', 'python -m pip install -qU uv --root-user-action=ignore\n\nROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F\'ROCm version: \' \'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { [ -r /opt/rocm/.info/version ] && awk -F. \'{print "rocm"$1"."$2; exit}\' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F\': *\' \'/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f=\'${Version}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf \'%{VERSION}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; })"\n[ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }\ncase "$ROCM_TAG" in\n  rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;\n  rocm6.*) T="rocm6.4" ;;\n  *) T="rocm7.1" ;;\nesac\npip install bitsandbytes\nPYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"\nuv pip install --system -U --force-reinstall \\\n    torch torchvision torchaudio triton-rocm \\\n    --index-url "$PYTORCH_INDEX_URL"\nuv pip install --system cut-cross-entropy torchao --no-deps\nuv pip install --system -U --no-deps "unsloth[amd]" "unsloth_zoo[amd]"\nuv pip install --system --no-deps -r "$(python -c \'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))\')" torchao\nuv pip install --system --no-deps -U "tokenizers>=0.22.0,<=0.23.0"\n')


# In[ ]:


get_ipython().system('uv pip install --system -qqq sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer "transformers==4.56.2" librosa soundfile evaluate jiwer torchcodec')
get_ipython().system('uv pip install --system -qqq --no-deps accelerate peft "trl==0.22.2"')


# In[2]:


from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
import torch

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
    model_name = "unsloth/whisper-large-v3",
    dtype = None, # Leave as None for auto detection
    load_in_4bit = False, # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "English",
    whisper_task = "transcribe",
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[3]:


model = FastModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = None, # ** MUST set this for Whisper **
)


# <a name="Data"></a>
# ### Data Prep  
# 
# We default to `vysakh25/laion-nonverbal-filtered` (single voice `shimmer`, GPT-4o TTS clips re-licensed CC-BY-4.0 by LAION, ~24 kHz, three emotion buckets sampled by default). Public-domain `keithito/lj_speech` (Linda Johnson via LibriVox) is the safety net. The original `MrDragonFox/Elise` and all known mirrors (`Jinsaryko/Elise`, `BarryFutureman/elise_v2`, `mrfakename/Elise`, etc.) were DMCA-disabled on 2026-04-24 (Moon Silk Audios); re-uploads under different usernames carry the same legal risk regardless of the licence tag the uploader sets. Set the `TTS_DATASET` env var to override with any HF dataset that exposes `audio` and `text` columns.
# 
# **Licensing notes.** The LAION default is GPT-4o TTS audio re-licensed CC-BY-4.0 by LAION. (a) OpenAI's Terms of Use restrict using OpenAI service Output to develop AI models that compete with OpenAI's own services; whether a fine-tune triggers that clause depends on your end use, and you should review the terms yourself. For unrestricted commercial use, switch to LJSpeech (public domain) or your own recordings. (b) CC-BY-4.0 requires attribution: if you redistribute checkpoints fine-tuned on this data, credit LAION (`vysakh25/laion-nonverbal-filtered`) in the model card.

# In[4]:


import numpy as np
import tqdm

#Set this to the language you want to train on
model.generation_config.language = "<|en|>"
model.generation_config.task = "transcribe"
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None

def formatting_prompts_func(example):
    audio_arrays = example['audio']['array']
    sampling_rate = example["audio"]["sampling_rate"]
    features = tokenizer.feature_extractor(
        audio_arrays, sampling_rate = sampling_rate
    )
    tokenized_text = tokenizer.tokenizer(example["text"])
    return {
        "input_features": features.input_features[0],
        "labels": tokenized_text.input_ids,
    }
# `MrDragonFox/Elise` and all known mirrors (`Jinsaryko/Elise`,
# `BarryFutureman/elise_v2`, `mrfakename/Elise`, etc.) were
# DMCA-disabled by Moon Silk Audios on 2026-04-24; re-uploads under
# different usernames carry the same legal risk regardless of the
# licence string set by the re-uploader. Default to LAION's `shimmer`
# voice slices of `vysakh25/laion-nonverbal-filtered` (CC-BY-4.0
# GPT-4o TTS clips, ~24 kHz, three emotion buckets, single voice).
# Public-domain `keithito/lj_speech` is the safety net. Override via
# the `TTS_DATASET` env var with any HF dataset that exposes `audio`
# and `text` columns.
#
# Licensing notes: (a) The LAION default is GPT-4o TTS audio re-licensed
# CC-BY-4.0 by LAION; OpenAI's Terms of Use restrict using OpenAI
# service Output to develop AI models that compete with OpenAI's own
# services. Whether a fine-tune triggers that clause depends on your
# end use; review the terms yourself. For unrestricted commercial use,
# switch to LJSpeech (public domain) or your own recordings.
# (b) CC-BY-4.0 requires attribution: credit LAION
# (`vysakh25/laion-nonverbal-filtered`) in any redistributed model card.
import io
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

_LAION_REPO = "vysakh25/laion-nonverbal-filtered"
_LAION_SHARDS = [
    "parquets/english_shimmer_intense_contentment_relaxation_peacefulness_calmness_satisfaction_and_serenity.parquet",
    "parquets/english_shimmer_intense_teasing_bantering_and_playful_mocking.parquet",
    "parquets/english_shimmer_intense_happiness_excitement_joy_exhilaration_delight_jubilation_and_bliss.parquet",
]

def _row_to_audio_text(example):
    """Decode LAION's `audio_bytes` and pair with `text_clean`.
    Returns `{text, audio}` matching the schema the trainer expects.
    """
    import soundfile as sf
    text = (example.get("text_clean") or example.get("text_original") or "").strip()
    raw = example.get("audio_bytes")
    if isinstance(raw, dict) and "bytes" in raw:
        raw = raw["bytes"]
    audio = None
    if raw:
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype = "float32", always_2d = False)
            if hasattr(arr, "ndim") and arr.ndim > 1:
                arr = arr.mean(axis = -1)
            audio = {"array": arr, "sampling_rate": int(sr)}
        except Exception:
            audio = None
    return {"text": text, "audio": audio}

def _load_laion_shimmer():
    locals_ = []
    for shard in _LAION_SHARDS:
        locals_.append(hf_hub_download(_LAION_REPO, shard, repo_type = "dataset"))
    tables = [pq.read_table(p) for p in locals_]
    table = pa.concat_tables(tables, promote_options = "default") if len(tables) > 1 else tables[0]
    ds = Dataset(table)
    ds = ds.map(_row_to_audio_text, remove_columns = ds.column_names)
    ds = ds.cast_column("audio", Audio())
    return ds

def _load_via_hf(name):
    ds = load_dataset(name, split = "train")
    cols = ds.column_names
    if "text" not in cols:
        for _alt in ("transcription", "normalized_text", "sentence"):
            if _alt in cols:
                ds = ds.rename_column(_alt, "text")
                break
    return ds

_user = os.environ.get("TTS_DATASET")
DATASET_CANDIDATES = [_user] if _user else ["__laion_shimmer__", "keithito/lj_speech"]

dataset = None
_last_err = None
for _name in DATASET_CANDIDATES:
    try:
        dataset = _load_laion_shimmer() if _name == "__laion_shimmer__" else _load_via_hf(_name)
        print(f"Loaded dataset: {_name} ({len(dataset)} rows)")
        break
    except (FileNotFoundError, HfHubHTTPError, ValueError, OSError, ConnectionError) as e:
        _last_err = e
        print(f"Could not load {_name}: {type(e).__name__}: {e}")

if dataset is None:
    raise RuntimeError(
        "Could not load any TTS dataset. Set TTS_DATASET to any HF "
        "dataset that exposes `audio` and `text` columns (any sample "
        "rate; notebooks resample as needed). Avoid Elise mirrors: "
        "`MrDragonFox/Elise` and all known mirrors (`Jinsaryko/Elise`, "
        "`BarryFutureman/elise_v2`, `mrfakename/Elise`, etc.) were "
        "DMCA-disabled by Moon Silk Audios on 2026-04-24."
    ) from _last_err
dataset = dataset.cast_column("audio", Audio(sampling_rate = 16000))
dataset = dataset.train_test_split(test_size = 0.06)
train_dataset = [formatting_prompts_func(example) for example in tqdm.tqdm(dataset['train'], desc = 'Train split')]
test_dataset = [formatting_prompts_func(example) for example in tqdm.tqdm(dataset['test'], desc = 'Test split')]


# In[6]:


# @title Create compute_metrics and datacollator
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pdb

metric = evaluate.load("wer")
def compute_metrics(pred):

    pred_logits = pred.predictions[0]
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id


    pred_ids = np.argmax(pred_logits, axis = -1)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens = True)

    wer = 100 * metric.compute(predictions = pred_str, references = label_str)

    return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors = "pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors = "pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# <a name="Train"></a>
# ### Train the model
# Now let's use Hugging Face `Seq2SeqTrainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[8]:


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bf16_supported
trainer = Seq2SeqTrainer(
    model = model,
    train_dataset = train_dataset,
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor = tokenizer),
    eval_dataset = test_dataset,
    tokenizer = tokenizer.feature_extractor,
    compute_metrics = compute_metrics,
    args = Seq2SeqTrainingArguments(
        # predict_with_generate = True,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        weight_decay = 0.001,
        remove_unused_columns = False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        eval_steps = 5 ,
        eval_strategy = "steps",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc

    ),
)


# In[9]:


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
# Let's run the model! Because we finetuned Whisper for speech recognition, we need to have a audio file.
# 
# For example we use the Harvard Sentences audio dataset https://en.wikipedia.org/wiki/Harvard_sentences

# In[14]:


get_ipython().system('wget https://upload.wikimedia.org/wikipedia/commons/5/5b/Speech_12dB_s16.flac')

from IPython.display import Audio, display
display(Audio("Speech_12dB_s16.flac", rate = 24000))


# In[15]:


from transformers import pipeline
import torch
FastModel.for_inference(model)
model.eval()
# Create pipeline without specifying the device
whisper = pipeline(
    "automatic-speech-recognition",
    model = model,
    tokenizer = tokenizer.tokenizer,
    feature_extractor = tokenizer.feature_extractor,
    processor = tokenizer,
    return_language = True,
    torch_dtype = torch.float16  # Remove the device parameter
)
# Example usage
audio_file = "Speech_12dB_s16.flac"
transcribed_text = whisper(audio_file)
print(transcribed_text["text"])


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[16]:


model.save_pretrained("whisper_lora")  # Local saving
tokenizer.save_pretrained("whisper_lora")
# model.push_to_hub("your_name/whisper_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/whisper_lora", token = "YOUR_HF_TOKEN") # Online saving


# ### Saving to float16
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("whisper_finetune_16bit", tokenizer, save_method = None,)
if False: model.push_to_hub_merged("HF_USERNAME/whisper_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("whisper_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/whisper_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("whisper_lora")
    tokenizer.save_pretrained("whisper_lora")
if False:
    model.push_to_hub("HF_USERNAME/whisper_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/whisper_lora", token = "YOUR_HF_TOKEN")


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
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
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
