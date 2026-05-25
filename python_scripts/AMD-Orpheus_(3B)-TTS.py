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

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('bash', '', 'python -m pip install -qU uv --root-user-action=ignore\n\nROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F\'ROCm version: \' \'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { [ -r /opt/rocm/.info/version ] && awk -F. \'{print "rocm"$1"."$2; exit}\' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F\': *\' \'/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f=\'${Version}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf \'%{VERSION}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; })"\n[ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }\ncase "$ROCM_TAG" in\n  rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;\n  rocm6.*) T="rocm6.4" ;;\n  *) T="rocm7.1" ;;\nesac\npip install bitsandbytes\nPYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"\nuv pip install --system -U --force-reinstall \\\n    torch torchvision torchaudio triton-rocm \\\n    --index-url "$PYTORCH_INDEX_URL"\nuv pip install --system cut-cross-entropy torchao --no-deps\nuv pip install --system -U --no-deps "unsloth[amd]" "unsloth_zoo[amd]"\nuv pip install --system --no-deps -r "$(python -c \'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))\')" torchao\nuv pip install --system --no-deps -U "tokenizers>=0.22.0,<=0.23.0"\n')
# 
# 
# # In[ ]:
# 
# 
# get_ipython().system('uv pip install --system -qqq sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer "transformers==4.56.2" snac torchcodec')
# get_ipython().system('uv pip install --system -qqq --no-deps accelerate peft "trl==0.22.2"')
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
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048, # Choose any for long context!
    dtype = None, # Select None for auto detection
    load_in_4bit = False, # Select True for 4bit which reduces memory usage
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
)


# <a name="Data"></a>
# ### Data Prep  
# 
# We default to `vysakh25/laion-nonverbal-filtered` (single voice `shimmer`, GPT-4o TTS clips re-licensed CC-BY-4.0 by LAION, ~24 kHz, three emotion buckets sampled by default). Each clip carries `parentheticals` like "Light Laugh, a gentle sound that barely breaks the silence" that the loader maps onto Orpheus's pretrained `<laughs>` / `<sighs>` / `<giggles>` etc. tokens, and inserts the cue token at the position in the sentence where the audio performs the cue (preserving acoustic alignment). Public-domain `keithito/lj_speech` (Linda Johnson via LibriVox) is the cue-less safety net. The original `MrDragonFox/Elise` and all known mirrors (`Jinsaryko/Elise`, `BarryFutureman/elise_v2`, `mrfakename/Elise`, etc.) were DMCA-disabled on 2026-04-24 (Moon Silk Audios); re-uploads under different usernames carry the same legal risk regardless of the licence tag the uploader sets. Set the `ORPHEUS_DATASET` env var to override with any HF dataset that exposes `audio` and `text` columns.
# 
# **Licensing notes.** The LAION default is GPT-4o TTS audio re-licensed CC-BY-4.0 by LAION. (a) OpenAI's Terms of Use restrict using OpenAI service Output to develop AI models that compete with OpenAI's own services; whether a fine-tune of Orpheus on LAION shimmer triggers that clause depends on your end use, and you should review the terms yourself. For unrestricted commercial use, switch to LJSpeech (public domain) or your own recordings. (b) CC-BY-4.0 requires attribution: if you redistribute checkpoints fine-tuned on this data, credit LAION (`vysakh25/laion-nonverbal-filtered`) in the model card.

# In[ ]:


import io
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Dataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# `MrDragonFox/Elise` and `Jinsaryko/Elise` were DMCA-disabled by Moon
# Silk Audios on 2026-04-24. Byte-identical re-uploads under any other
# username carry the same risk regardless of the licence string set
# by the re-uploader. We default to LAION's `shimmer` voice slices of
# `vysakh25/laion-nonverbal-filtered` (CC-BY-4.0, GPT-4o TTS clips
# Whisper-filtered so the non-verbal cues are actually performed in
# the audio, not just typed). The transcripts have parentheticals
# stripped, so we map them back to Orpheus's pretrained `<laughs>` /
# `<sighs>` / etc. tokens before training. Override with the
# `ORPHEUS_DATASET` env var to point at any HF dataset that exposes
# `audio` and `text` columns.

# Three shimmer emotion buckets spanning calm -> playful -> charged.
# Add or remove buckets to broaden / narrow the emotional range.
_LAION_REPO = "vysakh25/laion-nonverbal-filtered"
_LAION_SHARDS = [
    "parquets/english_shimmer_intense_contentment_relaxation_peacefulness_calmness_satisfaction_and_serenity.parquet",
    "parquets/english_shimmer_intense_teasing_bantering_and_playful_mocking.parquet",
    "parquets/english_shimmer_intense_happiness_excitement_joy_exhilaration_delight_jubilation_and_bliss.parquet",
]

# Map LAION parenthetical description keywords -> Orpheus cue tokens.
_CUE_FORMS = (
    # <gasps>: specific intake of breath, MUST come before generic
    # <sighs> entries so "quiet gasp, a restrained intake of breath"
    # picks <gasps>, not <sighs>.
    (("gasp", "gasps", "gasped", "gasping"),
     "<gasps>"),
    # <whispers>: specific quiet voice; precedes <sighs>.
    (("whisper", "whispers", "whispered", "whispering"),
     "<whispers>"),
    (("scream", "screams", "screamed", "screaming"),
     "<screams>"),
    (("cough", "coughs", "coughed", "coughing"),
     "<coughs>"),
    (("sniff", "sniffs", "sniffed", "sniffing",
      "sniffle", "sniffles", "sniffled", "sniffling"),
     "<sniffles>"),
    (("yawn", "yawns", "yawned", "yawning"),
     "<yawns>"),
    (("groan", "groans", "groaned", "groaning"),
     "<groans>"),
    (("moan", "moans", "moaned", "moaning"),
     "<moans>"),
    (("cry", "cries", "cried", "crying",
      "sob", "sobs", "sobbed", "sobbing"),
     "<cries>"),
    # <laughs>: full laughter forms incl. noun "laughter" and the
    # drop-e past/gerund of cackle, chuckle etc.
    (("laugh", "laughs", "laughed", "laughing", "laughter",
      "chuckle", "chuckles", "chuckled", "chuckling",
      "cackle", "cackles", "cackled", "cackling",
      "chortle", "chortles", "chortled", "chortling",
      "guffaw", "guffaws", "guffawed", "guffawing",
      "snort", "snorts", "snorted", "snorting",
      "squeal", "squeals", "squealed", "squealing"),
     "<laughs>"),
    (("giggle", "giggles", "giggled", "giggling",
      "snicker", "snickers", "snickered", "snickering",
      "snigger", "sniggers", "sniggered", "sniggering",
      "titter", "titters", "tittered", "tittering"),
     "<giggles>"),
    # <sighs>: generic exhale / hum. LAST so specific cues win.
    # Includes consonant-doubled "humming" / "hummed" inflections
    # explicitly; a suffix-list regex misses them.
    (("sigh", "sighs", "sighed", "sighing",
      "exhale", "exhales", "exhaled", "exhaling",
      "breath", "breaths", "breathe", "breathes", "breathed", "breathing",
      "hum", "hums", "hummed", "humming", "hummer"),
     "<sighs>"),
)
_CUE_FORM_TO_TOKEN = {f.lower(): t for forms, t in _CUE_FORMS for f in forms}
# Explicit \b(form1|form2|...)\b alternation. No \w* expansion, no
# suffix-list expansion: previous round caught either false positives
# (sober -> sob+er, human -> hum+an, cryptic -> cry+ptic) or false
# negatives (laughter / humming / cackled silently skipped). Listing
# inflected forms verbatim avoids both.
_CUE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _CUE_FORM_TO_TOKEN) + r")\b",
    re.IGNORECASE,
)

def _parentheticals_to_cues(pars_field):
    """Map LAION natural-language descriptions onto Orpheus cue tokens.

    Earliest-position match wins, so specific cues like <gasps> override
    generic ones like <sighs> when both keyword stems appear in the
    same description. Returns a de-duplicated list preserving order.
    """
    if pars_field is None:
        return []
    if isinstance(pars_field, str):
        try:
            pars = json.loads(pars_field)
        except Exception:
            return []
    else:
        try:
            pars = list(pars_field)
        except Exception:
            return []
    out, seen = [], set()
    for desc in pars:
        d = str(desc)
        m = _CUE_PATTERN.search(d)
        if m:
            tok = _CUE_FORM_TO_TOKEN[m.group(1).lower()]
            if tok not in seen:
                out.append(tok)
                seen.add(tok)
    return out

# Strip parentheticals (fallback path) and inline-replace them with cue
# tokens at their original position (primary path). Inline replacement
# preserves temporal alignment between the cue token and the SNAC codes
# of the corresponding clip moment, instead of jamming every cue at the
# start of the transcript.
_PAREN_RE = re.compile(r"\([^)]*\)")
def _strip_parentheticals(txt):
    return _PAREN_RE.sub("", txt).strip() if txt else txt

def _inline_parens_to_cues(txt):
    if not txt:
        return txt
    def _sub(m):
        cues = _parentheticals_to_cues([m.group(0).strip("()")])
        return " " + " ".join(cues) + " " if cues else " "
    return _PAREN_RE.sub(_sub, txt).strip()

def _row_to_orpheus(example):
    """Re-inject Orpheus cue tokens INLINE into the transcript and decode `audio_bytes`."""
    import soundfile as sf
    # Prefer `text_original` so cue tokens land at the position where
    # the audio actually performs the cue. Fall back to `text_clean`
    # (cues prepended) when text_original is empty.
    text_original = (example.get("text_original") or "").strip()
    if text_original:
        text = _inline_parens_to_cues(text_original)
    else:
        text = (example.get("text_clean") or "").strip()
        cues = _parentheticals_to_cues(example.get("parentheticals"))
        if cues:
            text = (" ".join(cues) + " " + text).strip()
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
        except Exception as _e:
            audio = None
    return {"text": text, "audio": audio}

def _load_laion_shimmer():
    locals_ = []
    for shard in _LAION_SHARDS:
        locals_.append(hf_hub_download(_LAION_REPO, shard, repo_type = "dataset"))
    tables = [pq.read_table(p) for p in locals_]
    table = pa.concat_tables(tables, promote_options = "default") if len(tables) > 1 else tables[0]
    ds = Dataset(table)
    ds = ds.map(_row_to_orpheus, remove_columns = ds.column_names)
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

_user = os.environ.get("ORPHEUS_DATASET")
DATASET_CANDIDATES = [_user] if _user else ["__laion_shimmer__", "keithito/lj_speech"]

dataset = None
last_err = None
for _name in DATASET_CANDIDATES:
    try:
        dataset = _load_laion_shimmer() if _name == "__laion_shimmer__" else _load_via_hf(_name)
        print(f"Loaded dataset: {_name} ({len(dataset)} rows)")
        break
    except (FileNotFoundError, HfHubHTTPError, ValueError, OSError, ConnectionError) as e:
        last_err = e
        print(f"Could not load {_name}: {type(e).__name__}: {e}")

if dataset is None:
    raise RuntimeError(
        "Could not load any TTS dataset. Set ORPHEUS_DATASET to any HF "
        "dataset that exposes `audio` and `text` columns (any sample "
        "rate; the notebook resamples to 24 kHz for SNAC). Avoid Elise "
        "mirrors: the original `MrDragonFox/Elise` and `Jinsaryko/Elise` "
        "were DMCA-disabled by Moon Silk Audios on 2026-04-24."
    ) from last_err


# In[ ]:


#@title Tokenization Function

import locale
import torchaudio.transforms as T
import os
import torch
from snac import SNAC
locale.getpreferredencoding = lambda: "UTF-8"
ds_sample_rate = dataset[0]["audio"]["sampling_rate"]

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")
def tokenise_audio(waveform):
  waveform = torch.from_numpy(waveform).unsqueeze(0)
  waveform = waveform.to(dtype = torch.float32)
  resample_transform = T.Resample(orig_freq = ds_sample_rate, new_freq = 24000)
  waveform = resample_transform(waveform)

  waveform = waveform.unsqueeze(0).to("cuda")

  #generate the codes from snac
  with torch.inference_mode():
    codes = snac_model.encode(waveform)

  all_codes = []
  for i in range(codes[0].shape[1]):
    all_codes.append(codes[0][0][i].item()+128266)
    all_codes.append(codes[1][0][2*i].item()+128266+4096)
    all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
    all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
    all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
    all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
    all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))


  return all_codes

def add_codes(example):
    # Always initialize codes_list to None
    codes_list = None

    try:
        answer_audio = example.get("audio")
        # If there's a valid audio array, tokenise it
        if answer_audio and "array" in answer_audio:
            audio_array = answer_audio["array"]
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
        # Keep codes_list as None if we fail
    example["codes_list"] = codes_list

    return example

dataset = dataset.map(add_codes, remove_columns = ["audio"])

tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai =  tokeniser_length + 6
pad_token = tokeniser_length + 7

audio_tokens_start = tokeniser_length + 10

dataset = dataset.filter(lambda x: x["codes_list"] is not None)
dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)

def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]

    removed_frames = 0

    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
        else:
            removed_frames += 1

    example["codes_list"] = result

    return example

dataset = dataset.map(remove_duplicate_frames)

tok_info = '''*** HERE you can modify the text prompt
If you are training a multi-speaker model (e.g., canopylabs/orpheus-3b-0.1-ft),
ensure that the dataset includes a "source" field and format the input accordingly:
- Single-speaker: f"{example['text']}"
- Multi-speaker: f"{example['source']}: {example['text']}"
'''
print(tok_info)

def create_input_ids(example):
    # Determine whether to include the source field
    text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]

    text_ids = tokenizer.encode(text_prompt, add_special_tokens = True)
    text_ids.append(end_of_text)

    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + example["text_tokens"]
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example


dataset = dataset.map(create_input_ids, remove_columns = ["text", "codes_list"])
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

dataset = dataset.remove_columns(columns_to_remove)


# <a name="Train"></a>
# ### Train the model
# Now let's use Hugging Face `Trainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
# 
# **Note:** Using a per_device_train_batch_size >1 may lead to errors if multi-GPU setup to avoid issues, ensure CUDA_VISIBLE_DEVICES is set to a single GPU (e.g., CUDA_VISIBLE_DEVICES=0).

# In[ ]:


from transformers import TrainingArguments,Trainer,DataCollatorForSeq2Seq
trainer = Trainer(
    model = model,
    train_dataset = dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
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

# In[ ]:


prompts = [
    "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person.",
]

chosen_voice = None # None for single-speaker


# In[ ]:


#@title Run Inference


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# Moving snac_model cuda to cpu
snac_model.to("cpu")

prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

all_input_ids = []

for prompt in prompts_:
  input_ids = tokenizer(prompt, return_tensors = "pt").input_ids
  all_input_ids.append(input_ids)

start_token = torch.tensor([[ 128259]], dtype = torch.int64) # Start of human
end_tokens = torch.tensor([[128009, 128260]], dtype = torch.int64) # End of text, End of human

all_modified_input_ids = []
for input_ids in all_input_ids:
  modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim = 1) # SOH SOT Text EOT EOH
  all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
for modified_input_ids in all_modified_input_ids:
  padding = max_length - modified_input_ids.shape[1]
  padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype = torch.int64), modified_input_ids], dim = 1)
  attention_mask = torch.cat([torch.zeros((1, padding), dtype = torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype = torch.int64)], dim = 1)
  all_padded_tensors.append(padded_tensor)
  all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim = 0)
all_attention_masks = torch.cat(all_attention_masks, dim = 0)

input_ids = all_padded_tensors.to("cuda")
attention_mask = all_attention_masks.to("cuda")
generated_ids = model.generate(
      input_ids = input_ids,
      attention_mask = attention_mask,
      max_new_tokens = 1200,
      do_sample = True,
      temperature = 0.6,
      top_p = 0.95,
      repetition_penalty = 1.1,
      num_return_sequences = 1,
      eos_token_id = 128258,
     use_cache = True
  )
token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple = True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove

processed_rows = []

for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []

for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)


def redistribute_codes(code_list):
  layer_1 = []
  layer_2 = []
  layer_3 = []
  for i in range((len(code_list)+1)//7):
    layer_1.append(code_list[7*i])
    layer_2.append(code_list[7*i+1]-4096)
    layer_3.append(code_list[7*i+2]-(2*4096))
    layer_3.append(code_list[7*i+3]-(3*4096))
    layer_2.append(code_list[7*i+4]-(4*4096))
    layer_3.append(code_list[7*i+5]-(5*4096))
    layer_3.append(code_list[7*i+6]-(6*4096))
  codes = [torch.tensor(layer_1).unsqueeze(0),
         torch.tensor(layer_2).unsqueeze(0),
         torch.tensor(layer_3).unsqueeze(0)]

  # codes = [c.to("cuda") for c in codes]
  audio_hat = snac_model.decode(codes)
  return audio_hat

my_samples = []
for code_list in code_lists:
  samples = redistribute_codes(code_list)
  my_samples.append(samples)
from IPython.display import display, Audio
if len(prompts) != len(my_samples):
  raise Exception("Number of prompts and samples do not match")
else:
  for i in range(len(my_samples)):
    print(prompts[i])
    samples = my_samples[i]
    display(Audio(samples.detach().squeeze().to("cpu").numpy(), rate = 24000))
# Clean up to save RAM
del my_samples,samples


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("orpheus_lora")  # Local saving
tokenizer.save_pretrained("orpheus_lora")
# model.push_to_hub("your_name/orpheus_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("your_name/orpheus_lora", token = "YOUR_HF_TOKEN") # Online saving


# ### Saving to float16
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("orpheus_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("HF_USERNAME/orpheus_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("orpheus_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/orpheus_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("orpheus_lora")
    tokenizer.save_pretrained("orpheus_lora")
if False:
    model.push_to_hub("HF_USERNAME/orpheus_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/orpheus_lora", token = "YOUR_HF_TOKEN")


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
