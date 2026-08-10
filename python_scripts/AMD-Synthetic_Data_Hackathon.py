#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Synthetic Data Generation and Unsloth Tutorial</h1>

# ### Installation

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'python -m pip install -qU uv --root-user-action=ignore\n\nROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F\'ROCm version: \' \'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { [ -r /opt/rocm/.info/version ] && awk -F. \'{print "rocm"$1"."$2; exit}\' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F\': *\' \'/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}\'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f=\'${Version}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf \'%{VERSION}\\n\' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F\'[.-]\' \'{print "rocm"$1"."$2; exit}\' <<<"$ver"; })"\n[ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }\ncase "$ROCM_TAG" in\n  rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;\n  rocm6.*) T="rocm6.4" ;;\n  *) T="rocm7.1" ;;\nesac\npip install bitsandbytes\nPYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"\nuv pip install --system -U --force-reinstall \\\n    torch torchvision torchaudio triton-rocm \\\n    --index-url "$PYTORCH_INDEX_URL"\nuv pip install --system cut-cross-entropy torchao --no-deps\nuv pip install --system -U --no-deps "unsloth[amd]" "unsloth_zoo[amd]"\nuv pip install --system --no-deps -r "$(python -c \'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))\')" torchao\nuv pip install --system --no-deps -U "tokenizers>=0.22.0,<=0.23.0"\n')


# In[ ]:


get_ipython().system('uv pip install --system -qqq sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer "transformers==4.56.2" vllm "synthetic-data-kit==0.0.3"')
get_ipython().system('uv pip install --system -qqq --no-deps accelerate peft "trl==0.22.2"')


# ## Synthetic Data Generation
# 
# In this section, we use the CLI from synthetic-data-kit to generate datasets

# ### Converting to Fine-Tuning Format
# 
# The **save-as** function of the synthetic-data-kit CLI converts curated Q&A
# pairs to fine-tuning format:
# 
# ```
# synthetic-data-kit save-as ./logical_reasoning/data/curated/ --format ft
# ```
# 
# - Reads curated JSON files from `data/curated/`
# - Converts to format `ft` (fine-tuning format with messages structure)
# - Outputs are saved to `data/final/` with proper conversation format
# - The resulting format is compatible with standard fine-tuning pipelines
# 
# That command needs a served model behind it, so the next cell synthesises an
# equivalent `data/final/` set in-notebook instead. It writes the same `ft`
# schema, and it skips itself if you already produced those files with the CLI.

# In[ ]:


import itertools
import json
import os
import random
from pathlib import Path

# ===== CONFIGURATION =====
# The same data/final/ the CLI writes with `save-as --format ft`.
data_dir = "./logical_reasoning/data/final"
num_examples = 74          # how many puzzles to synthesise
seed = 3407

# Files already produced by the CLI are left untouched.
final_dir = Path(data_dir)
final_dir.mkdir(parents = True, exist_ok = True)

# A leftover *.json.tmp means an earlier publish died, so the shards are partial.
interrupted = sorted(final_dir.glob("*.json.tmp"))
existing = [] if interrupted else sorted(final_dir.glob("*.json"))

NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Isabel", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
]

SYSTEM_PROMPT = (
    "You are a careful logical reasoning assistant. Solve knight and knave "
    "puzzles by checking every possible assignment and explain your reasoning."
)

PUZZLE_INTRO = (
    "On this island every inhabitant is either a knight, who always tells the "
    "truth, or a knave, who always lies."
)


def make_statement(rng, speaker, people):
    """Return (english, predicate) for one random claim made by `speaker`."""
    others = [p for p in people if p != speaker]
    kind = rng.choice(
        ["is_knight", "is_knave", "both", "either", "same", "different", "implies"]
    )
    if kind in ("is_knight", "is_knave"):
        target = rng.choice(people)
        if kind == "is_knight":
            text = f"{target} is a knight."
            return text, lambda a, t = target: a[t]
        text = f"{target} is a knave."
        return text, lambda a, t = target: not a[t]

    if not others:
        text = f"{speaker} is a knight."
        return text, lambda a, s = speaker: a[s]

    a_name, b_name = rng.sample(people, 2) if len(people) >= 2 else (speaker, speaker)
    if kind == "both":
        text = f"{a_name} and {b_name} are both knights."
        return text, lambda a, x = a_name, y = b_name: a[x] and a[y]
    if kind == "either":
        text = f"At least one of {a_name} and {b_name} is a knight."
        return text, lambda a, x = a_name, y = b_name: a[x] or a[y]
    if kind == "same":
        text = f"{a_name} and {b_name} are the same kind."
        return text, lambda a, x = a_name, y = b_name: a[x] == a[y]
    if kind == "different":
        text = f"{a_name} and {b_name} are of different kinds."
        return text, lambda a, x = a_name, y = b_name: a[x] != a[y]
    text = f"If {a_name} is a knight, then {b_name} is a knight."
    return text, lambda a, x = a_name, y = b_name: (not a[x]) or a[y]


def solve(people, predicates):
    """Return every assignment where each speaker's claim matches their kind."""
    solutions = []
    for combo in itertools.product([True, False], repeat = len(people)):
        assignment = dict(zip(people, combo))
        if all(predicates[p](assignment) == assignment[p] for p in people):
            solutions.append(assignment)
    return solutions


def build_puzzle(rng):
    """Synthesise one puzzle that has exactly one consistent solution."""
    people = rng.sample(NAMES, rng.choice([2, 3, 4]))
    statements, predicates = {}, {}
    for person in people:
        text, predicate = make_statement(rng, person, people)
        statements[person] = text
        predicates[person] = predicate
    solutions = solve(people, predicates)
    if len(solutions) != 1:
        return None

    answer = solutions[0]
    lines = [PUZZLE_INTRO, "", "You meet the following inhabitants:", ""]
    lines += [f"- {p} says: \"{statements[p]}\"" for p in people]
    lines += ["", "Who is a knight and who is a knave?"]
    question = "\n".join(lines)

    reasoning = [
        f"There are {len(people)} inhabitants, so there are "
        f"{2 ** len(people)} possible assignments. Exactly one of them is "
        "consistent with every statement.",
        "",
    ]
    for person in people:
        kind = "knight" if answer[person] else "knave"
        verdict = "true" if answer[person] else "false"
        reasoning.append(
            f"- {person} is a {kind}, so the claim \"{statements[person]}\" "
            f"must be {verdict}, and it is."
        )
    reasoning += ["", "Final answer:"]
    reasoning += [
        f"- {p}: {'knight' if answer[p] else 'knave'}" for p in people
    ]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "\n".join(reasoning)},
        ]
    }


if existing:
    print(f"Found {len(existing)} existing ft file(s) in {final_dir}, skipping generation.")
else:
    rng = random.Random(seed)
    records, seen = [], set()
    attempts = 0
    while len(records) < num_examples and attempts < num_examples * 500:
        attempts += 1
        puzzle = build_puzzle(rng)
        if puzzle is None:
            continue
        key = puzzle["messages"][1]["content"]
        if key in seen:
            continue
        seen.add(key)
        records.append(puzzle)

    if len(records) < num_examples:
        raise RuntimeError(
            f"Only synthesised {len(records)} of {num_examples} puzzles. "
            "Raise the attempt budget or lower num_examples."
        )

    # Two shards, mirroring the two curated files the CLI produces.
    midpoint = len(records) // 2
    shards = {
        "knights_and_knaves_easy_ft.json": records[:midpoint],
        "knights_and_knaves_hard_ft.json": records[midpoint:],
    }
    # Stage as *.json.tmp, which the skip glob above and the loader both ignore.
    staged = []
    for filename, shard in shards.items():
        tmp_path = final_dir / f"{filename}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(shard, f, indent = 2)
        staged.append((tmp_path, final_dir / filename, len(shard)))

    for tmp_path, final_path, count in staged:
        os.replace(tmp_path, final_path)
        print(f"Wrote {count} records to {final_path}")

    # Only now is the set whole, so only now may the marker go; clearing it
    # earlier would hide a recovery run that is itself interrupted.
    for leftover in interrupted:
        leftover.unlink(missing_ok = True)

print(f"Total ft files now in {final_dir}: {len(sorted(final_dir.glob('*.json')))}")


# ### Building the Fine-Tuning Dataset
# 
# The cell above materialises the `data/final/` directory that the loader below
# reads. It synthesises knight-and-knave logic puzzles directly in the notebook:
# 
# - Each puzzle picks two to four inhabitants and gives every one of them a
#   random claim about the others.
# - All `2^n` knight/knave assignments are enumerated and only puzzles with
#   exactly one consistent assignment are kept, so every answer is verifiable.
# - Records are written in the same `ft` schema that
#   `synthetic-data-kit save-as --format ft` emits, that is a JSON list of
#   `{"messages": [system, user, assistant]}` objects, split over two shards.
# 
# Generation is seeded, so the same 74 conversations come out on every run. If
# you already produced `data/final/*.json` with the synthetic-data-kit CLI and a
# served model, the cell detects those files and leaves them alone.

# In[1]:


import json
import glob
from pathlib import Path
from datasets import Dataset

# ===== CONFIGURATION =====
data_dir = "./logical_reasoning/data/final"  # Change this to your data directory

# ===== STEP 1: Find all FT files =====
data_path = Path(data_dir)
ft_files = sorted(glob.glob(str(data_path / "*.json")))

if not ft_files:
    raise FileNotFoundError(
        f"No .json files found in {data_path.resolve()}. Run the synthetic "
        "data generation cell above, or produce the files yourself with "
        "`synthetic-data-kit save-as ./logical_reasoning/data/curated/ "
        "--format ft`, before running this cell."
    )

# ===== STEP 2: Load and convert all files =====
all_data = []

for file_path in ft_files:
    # Load the JSON file
    with open(file_path, 'r') as f:
        ft_data = json.load(f)

    # Convert each item
    for item in ft_data:
        if 'messages' not in item:
            continue

        # Extract only user and assistant messages
        conversation = []
        for msg in item['messages']:
            if msg['role'] == 'user' or msg['role'] == 'assistant':
                conversation.append({
                    "role": msg['role'],
                    "content": msg['content']
                })

        # Add to our data if we have at least one exchange
        if len(conversation) > 0:
            all_data.append({
                "conversations": conversation
            })

print(f"\nTotal conversations: {len(all_data)} from {len(ft_files)} file(s)")

if not all_data:
    raise ValueError(
        f"Found {len(ft_files)} file(s) in {data_path.resolve()} but none of "
        "them contained a usable record. Every record needs a `messages` list "
        "with at least one user or assistant turn, which is what the `ft` "
        "format produces. Check the files before training on an empty set."
    )

# ===== STEP 3: Create HuggingFace Dataset =====
dataset = Dataset.from_list(all_data)

# ===== STEP 4: Preview the data =====
print(json.dumps(dataset[0], indent = 2))


# ### Loading and Converting Data to HuggingFace Dataset
# 
# This cell performs comprehensive data processing:
# 
# 1. **Finding Files**: Locates all JSON files in `data/final/` directory
# 2. **Loading Data**: Reads each JSON file containing fine-tuning formatted data
# 3. **Format Conversion**: Extracts user and assistant messages from the fine-tuning format
# 4. **Structuring Conversations**: Creates a standardized conversation format with role-content pairs
# 5. **Creating Dataset**: Converts the processed data into a HuggingFace Dataset object
# 
# The output shows 74 total conversations were successfully loaded and formatted. The preview displays a sample conversation showing a knight-and-knave logic puzzle with its solution.

# ## Fine-Tuning
# 
# ### Note: Please remember to shutdown the vLLM instance!
# ### See https://unsloth.ai/docs/new/unsloth-amd-pytorch-synthetic-data-hackathon#how-do-i-free-amd-gpu-memory

# In[2]:


import os
import json
import glob
import torch
import shutil
from pathlib import Path
from datasets import Dataset


# ### Importing Standard Libraries
# 
# Imports essential Python libraries for fine-tuning:
# - `os`, `json`, `glob`: File system operations and JSON handling
# - `torch`: PyTorch deep learning framework
# - `shutil`: File operations
# - `Path`: Path manipulation
# - `Dataset`: HuggingFace datasets library for data handling

# In[3]:


from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq


# ### Importing Unsloth and Training Libraries
# 
# Imports specialized libraries for efficient fine-tuning:
# - `FastLanguageModel` from Unsloth: Optimized model loading and training
# - `get_chat_template`, `standardize_sharegpt`, `train_on_responses_only`: Chat formatting utilities
# - `SFTConfig`, `SFTTrainer`: Supervised fine-tuning configuration and trainer from TRL
# - `DataCollatorForSeq2Seq`: Handles batching and padding for sequence-to-sequence training

# ### Setup Unsloth model and tokenizer for ROCm without bitsandbytes

# In[4]:


max_seq_length = 1024

# ===== HARDWARE AWARE CONFIGURATION =====
# Llama-3.3-70B needs about 140GB for its weights, so it is only picked on an
# MI300X class GPU; smaller cards, free Colab and Kaggle included, get
# Llama-3.1-8B. Override these values if you know what your GPU can hold.
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU visible. Switch the runtime to a GPU accelerator before "
        "running this cell."
    )

gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
supports_bf16 = torch.cuda.is_bf16_supported()

if gpu_memory_gb >= 150:
    model_name = "unsloth/Llama-3.3-70B-Instruct"
    load_in_4bit = False
    lora_rank = 64
    per_device_train_batch_size = 64
    gradient_accumulation_steps = 1
    optim = "adamw_torch_fused"  # bitsandbytes is not available on ROCm
elif gpu_memory_gb >= 40:
    model_name = "unsloth/Llama-3.1-8B-Instruct"
    load_in_4bit = False
    lora_rank = 32
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    optim = "adamw_8bit"
else:
    model_name = "unsloth/Llama-3.1-8B-Instruct"
    load_in_4bit = True
    lora_rank = 16
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    optim = "adamw_8bit"

# T4 and other pre-Ampere cards have no bfloat16, so ask for float16 there.
dtype = torch.bfloat16 if supports_bf16 else torch.float16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = "auto",
    trust_remote_code = True,
)

print(f"Loaded: {model_name} on a {gpu_memory_gb:.0f}GB GPU")
print(f"dtype={dtype}, load_in_4bit={load_in_4bit}, lora_rank={lora_rank}")

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha = lora_rank,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


# ### Loading the Model with LoRA
# 
# This cell sizes the run to the GPU it finds:
# 
# **On a 150GB or larger GPU (the MI300X the hackathon used):**
# - Model: Llama-3.3-70B-Instruct (70 billion parameters)
# - Data type: bfloat16, no quantization, so bitsandbytes is not needed
# - LoRA rank 64, batch size 64, `adamw_torch_fused`
# 
# **On a 40GB to 150GB GPU:**
# - Model: Llama-3.1-8B-Instruct in bfloat16, LoRA rank 32
# 
# **On anything smaller (free Colab and Kaggle tiers):**
# - Model: Llama-3.1-8B-Instruct in 4-bit, LoRA rank 16, batch size 2 with 8
#   gradient accumulation steps
# - float16 instead of bfloat16, because T4 class cards do not support bfloat16
# 
# Llama-3.3-70B needs roughly 140GB for the weights alone, so it is only picked
# when the GPU can actually hold it. Edit the configuration block directly if you
# want to force a specific checkpoint.
# 
# **Shared LoRA configuration:**
# - Target modules: all attention and MLP layers (q_proj, k_proj, v_proj, o_proj,
#   gate_proj, up_proj, down_proj)
# - LoRA alpha equal to the rank, dropout 0
# - Gradient checkpointing: "unsloth" for memory efficiency
# 
# LoRA enables efficient fine-tuning by only training small adapter layers
# instead of every weight in the base model.

# In[5]:


"""Prepare dataset with proper chat template and tensor compatibility"""
print("🔧 Preparing dataset for training...")

# Set chat template
tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Formatting function that ensures proper tensor conversion
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = []

    for convo in convos:
        # Ensure conversation is in correct format
        if isinstance(convo, list) and all(isinstance(msg, dict) for msg in convo):
            text = tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False)
            texts.append(text)
        else:
            print(f"⚠️  Skipping malformed conversation: {type(convo)}")
            continue

    return {"text": texts}

dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func, batched = True, remove_columns = dataset.column_names)

dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

print(f"✅ Prepared {len(dataset)} valid examples for training")

# Show sample
if len(dataset) > 0:
    print(f"📝 Sample formatted text:")
    print(dataset["text"][0][:200] + "...")


# ### Preparing Dataset with Chat Template
# 
# This cell formats the dataset for fine-tuning:
# 
# **Steps:**
# 1. **Set Chat Template**: Applies Llama-3.1 chat template formatting
# 2. **Configure Padding**: Sets pad token to eos token if not already set
# 3. **Format Conversations**: The `formatting_prompts_func` function:
#    - Takes raw conversations from the dataset
#    - Applies the chat template to format them properly
#    - Validates conversation structure (list of dicts with role/content)
#    - Filters out malformed conversations
# 4. **Standardize Format**: Uses `standardize_sharegpt` to normalize the data structure
# 5. **Apply Formatting**: Maps the formatting function across all examples
# 6. **Remove Empty**: Filters out any empty or invalid formatted texts
# 
# The output shows 74 valid examples were successfully prepared. A sample of the formatted text is displayed, showing the proper Llama-3.1 chat template structure with system, user, and assistant headers.

# In[6]:


"""Train the model with the settings chosen for this GPU"""
# Ensure tokenizer has proper padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, padding = True),
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = optim,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "logical_reasoning_rocm_outputs",
        report_to = "none",
        bf16 = supports_bf16,
        fp16 = not supports_bf16,
        dataloader_pin_memory = False,
        remove_unused_columns = True,  # Remove unused columns to avoid tensor issues
        gradient_checkpointing = True,
        dataloader_num_workers = 0,  # Single worker for stability
    ),
)

# Train only on responses
trainer = train_on_responses_only(trainer)

FastLanguageModel.for_training(model)
trainer_stats = trainer.train()


# ### Training the Model
# 
# This cell configures and executes the fine-tuning process:
# 
# **Training Configuration (SFTConfig):**
# - **Batch size and gradient accumulation**: taken from the block that sized the
#   model, so 64 x 1 on an MI300X and 2 x 8 on a small GPU
# - **Warmup**: 5 steps
# - **Epochs**: 1 full pass through the dataset
# - **Learning rate**: 1e-4
# - **Optimizer**: `adamw_torch_fused` on the no-bitsandbytes ROCm path,
#   `adamw_8bit` elsewhere
# - **Precision**: bfloat16 where the GPU supports it, float16 otherwise
# - **Gradient checkpointing**: enabled for memory efficiency
# 
# **Special Training Mode:**
# Uses `train_on_responses_only` to compute loss only on the assistant's
# responses, not on the user's questions. This focuses the model on learning to
# generate accurate answers rather than memorizing the input format.
# 
# **Key Features:**
# - DataCollatorForSeq2Seq handles variable-length sequences with proper padding
# - No packing, to preserve conversation structure
# - Single dataloader worker for stability
# 
# The model is then trained on the 74 logical reasoning conversations.

# In[ ]:


"""Save the trained model"""
print("\n💾 SAVING ROCM-TRAINED MODEL")

# Save LoRA adapters
lora_path = "logical_reasoning_rocm_lora"
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"✅ LoRA adapters saved to: {lora_path}")

# Save merged model
merged_path = "logical_reasoning_rocm_merged"
print("🔄 Saving merged model...")
model.save_pretrained_merged(merged_path, tokenizer, save_method = "merged_16bit")
print(f"✅ Merged model saved to: {merged_path}")

print(f"\n🎉 ROCM MODEL READY!")


# ### Saving the Fine-Tuned Model
# 
# This cell saves the trained model in two formats:
# 
# 1. **LoRA Adapters** (`logical_reasoning_rocm_lora/`):
#    - Saves only the trained LoRA adapter weights (lightweight, ~few hundred MB)
#    - Can be loaded later with the base model
#    - Useful for sharing or deploying with the original base model
# 
# 2. **Merged Model** (`logical_reasoning_rocm_merged/`):
#    - Merges LoRA adapters back into the base model
#    - Creates a standalone model with all weights
#    - Saved in 16-bit precision for better quality
#    - Ready for immediate inference without loading adapters
# 
# Both formats include the tokenizer configuration. The merged model is production-ready and can be used directly for generating answers to logical reasoning questions.And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
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
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
