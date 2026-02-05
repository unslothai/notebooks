import argparse
import ast
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from glob import glob
from nbconvert import PythonExporter
import nbformat
from spellchecker import SpellChecker


SPELL_IGNORE_WORDS = {
    "unsloth", "qwen", "llama", "gemma", "lora", "gguf", "vllm", "grpo",
    "kaggle", "colab", "alpaca", "qlora", "peft", "sft", "dpo", "orpo",
    "bnb", "bitsandbytes", "xformers", "triton", "cuda", "pytorch",
    "tokenizer", "huggingface", "finetune", "finetuning", "bf16", "fp16",
    "fp8", "int4", "int8", "eos", "vram", "gpu", "cpu", "trl", "sdpa",
    "ipynb", "ggml", "ollama", "mistral", "deepseek", "pixtral", "qat",
    "nemotron", "magistral", "ministral", "granite", "ernie", "bert",
    "roberta", "xlm", "matmul", "autocast", "dtype", "warmup",
    "pretrained", "instruct", "mergekit", "wandb", "tensorboard", "lmstudio",
    "venv", "conda", "repo", "param",
    "numpy", "scipy", "sklearn", "tokenizers", "datasets",
    "checkpointing", "logits", "softmax", "quantized", "quantize",
    "quantization", "backprop", "embeddings", "hyperparameters", "trainable",
    "nemo", "nvidia", "multimodal", "env", "linux", "macos", "runpod",
    "eval", "cot", "codeforces", "completions",
    # HTML/markdown tags and attributes commonly found in notebooks
    "img", "src", "href", "div", "png", "svg", "alt", "https", "http",
    "html", "css", "url", "readme", "github", "runtime", "cpp", "natively",
    "pretraining", "finetunes", "tts", "llms", "vlm", "vlms", "gpt", "oss",
    "dataset", "nli", "finetuned", "tutoring", "tutored",
    "unslothai", "nbsp", "executorch", "regex",
    "prequantized", "prepend", "prepended", "hugging", "submodule",
    "repo", "repos", "txt", "csv", "json", "yaml", "toml",
    "subfolder", "subdirectory", "gradio", "chatbot", "natively",
    # Common words in notebooks that are valid but not in dictionary
    "etc", "pre", "multi", "chatml", "vicuna", "labonne", "maxime",
    "maths", "tokenized", "workflow", "functiongemma", "templating",
    "tomaarsen", "miriad", "langid", "bahasa",
    "electroglyph", "runpod",
    # GitHub usernames, package names, tech terms
    "willccbb", "sglang", "thytu", "vicgalle", "kadirnar", "saibo",
    "etherl", "mithex", "pydantic", "scikit", "jsonl", "docstrings",
    "tokenization", "tokenize", "prepending", "customizable", "chatbots",
    "modelfile", "subprocess", "app", "bot", "dict", "globals", "configs",
    "shouldn", "backticks", "analyse", "filepath", "pclass", "skp",
    "pte", "uncomment", "entrypoint", "pid", "resize",
    "alibaba", "moby", "ebooks", "pdf", "ppt", "docx", "num",
    "doesn", "removeprefix", "multiturn", "rechne", "direkt", "ich",
}

SPELL_KNOWN_FIXES = {
    "Optinal": "Optional",
    "trainig": "training",
    "competive": "competitive",
    "whicht": "which",
    "simpilicity": "simplicity",
    "managable": "manageable",
    "randomnly": "randomly",
    "enclused": "enclosed",
    "effecient": "efficient",
    "fibonnaci": "fibonacci",
    "Fibonnaci": "Fibonacci",
    "SHould": "Should",
    "GTP-OSS": "GPT-OSS",
    "stratgegy": "strategy",
    "verifer": "verifier",
    "verisons": "versions",
    "datases": "datasets",
}


def check_spelling(notebook_content, notebook_name):
    """Check spelling in markdown cells and code comments. Auto-fix known misspellings."""
    spell = SpellChecker()
    spell.word_frequency.load_words(SPELL_IGNORE_WORDS)
    issues = []
    fixed = False
    for i, cell in enumerate(notebook_content.get("cells", [])):
        source = cell.get("source", [])
        if isinstance(source, str):
            source = [source]
        text = "".join(source)

        # Apply known fixes
        new_text = text
        for wrong, right in SPELL_KNOWN_FIXES.items():
            if wrong in new_text:
                new_text = new_text.replace(wrong, right)
        if new_text != text:
            cell["source"] = new_text.splitlines(True)
            fixed = True

        # Check for unknown misspellings in markdown cells (use new_text which has known fixes applied)
        if cell.get("cell_type") == "markdown":
            # Strip HTML tags and URLs before extracting words
            clean_text = re.sub(r'<[^>]+>', ' ', new_text)
            clean_text = re.sub(r'https?://\S+', ' ', clean_text)
            clean_text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', clean_text)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text)
            # Filter out code identifiers (camelCase, snake_case, ALL_CAPS)
            english_words = [
                w for w in words
                if w == w.lower() or w == w.capitalize()
            ]
            lower_words = [w.lower() for w in english_words]
            misspelled = spell.unknown(lower_words)
            misspelled -= SPELL_IGNORE_WORDS
            if misspelled:
                issues.append((i, misspelled))
    return fixed, issues


def validate_notebook_syntax(notebook_path):
    """Validate Python syntax of all code cells in a notebook."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        return []

    errors = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        # Remove IPython magics and shell commands for AST parsing
        # Replace with 'pass' to avoid empty blocks (e.g., if COLAB: !pip install)
        clean_lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%", "%%")):
                indent = line[:len(line) - len(stripped)]
                clean_lines.append(indent + "pass")
            else:
                clean_lines.append(line)
        clean_source = "\n".join(clean_lines)

        if not clean_source.strip():
            continue

        try:
            ast.parse(clean_source)
        except SyntaxError as e:
            errors.append((i, e.lineno, str(e)))

    return errors


def _get_base_name_from_filename(filename):
    """Extract a base name from the notebook filename for dynamic model naming."""
    name = os.path.splitext(os.path.basename(filename))[0]
    for prefix in ("Kaggle-", "HuggingFace Course-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    lower = name.lower()
    if re.match(r"gemma[-_]?3n", lower):
        return "gemma_3n"
    if re.match(r"gemma[-_]?3", lower):
        return "gemma_3"

    stop_match = re.search(r"[\(\[\{]", name)
    trimmed = name[:stop_match.start()] if stop_match else name
    trimmed = trimmed.strip(" _-") or name

    segments = re.split(r"[^A-Za-z0-9]+", trimmed)
    segments = [s for s in segments if s]
    if not segments:
        base = trimmed.lower()
        base = base.replace("-", "_")
        base = re.sub(r"__+", "_", base)
        return base.strip("_")

    max_len = 24
    parts = []
    for seg in segments:
        if re.fullmatch(r"[A-Za-z]+", seg):
            token = seg.lower()
        elif re.fullmatch(r"[A-Za-z][0-9]", seg):
            token = seg.lower()
        else:
            if not parts:
                lead = re.match(r"[A-Za-z]+", seg)
                if lead:
                    token = lead.group(0).lower()
                    parts.append(token)
            break
        candidate = "_".join(parts + [token]) if parts else token
        if len(candidate) <= max_len:
            parts.append(token)
        else:
            break

    base = "_".join(parts) if parts else segments[0].lower()
    return base


def _strip_extra_trailing_blank_lines(lines):
    """Remove consecutive trailing blank lines, keeping at most one."""
    while len(lines) > 1 and lines[-1].strip() == "" and lines[-2].strip() == "":
        lines.pop()
    return lines


def _space_equals_in_code(text):
    """Add spaces around = in code, but preserve compound operators (+=, -=, etc.)."""
    # Characters that form compound assignment operators when followed by =
    # e.g., +=, -=, *=, /=, //=, **=, %=, |=, &=, ^=, :=, @=
    COMPOUND_OP_CHARS = ("+", "-", "*", "/", "%", "|", "&", "^", ":", "@")

    new_lines = []
    in_shell_command = False
    for line in text.splitlines(True):
        stripped = line.lstrip()
        # Track multi-line shell commands (lines starting with ! or continuations)
        if stripped.startswith("!"):
            in_shell_command = True
        # Skip shell commands - they have their own syntax (pip URLs, version specs, etc.)
        # Also skip lines containing URL fragments like #subdirectory= or #egg=
        if in_shell_command or "#subdirectory=" in line or "#egg=" in line:
            new_lines.append(line)
            # Check if this line continues (ends with backslash)
            if in_shell_command and not line.rstrip().endswith("\\"):
                in_shell_command = False
            continue
        in_quote = None
        escaped = False
        out = []
        for i, ch in enumerate(line):
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if in_quote:
                out.append(ch)
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ("\"", "'"):
                out.append(ch)
                in_quote = ch
                continue

            if ch == "=":
                prev_char = line[i - 1] if i > 0 else ""
                next_char = line[i + 1] if i + 1 < len(line) else ""
                # Don't add space before = if it's part of ==, <=, >=, !=
                # or a compound operator like +=, -=, *=, /=, etc.
                if prev_char not in ("=", "<", ">", "!") and prev_char not in COMPOUND_OP_CHARS and next_char != "=":
                    if out and out[-1] not in (" ", "\t"):
                        out.append(" ")
                    out.append("=")
                    if next_char not in (" ", "\t", "\n", ""):
                        out.append(" ")
                    continue
            out.append(ch)
        new_lines.append("".join(out))
    return "".join(new_lines)


def update_old_unsloth(filename):
    """Update notebook with various fixes using JSON-based cell manipulation."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)
    except Exception:
        return

    base = _get_base_name_from_filename(filename)
    if base.endswith("_finetune"):
        base_gguf = base
        base_lora = f"{base}_lora"
        base_16 = f"{base}_16bit"
        base_4 = f"{base}_4bit"
    else:
        base_gguf = f"{base}_finetune"
        base_lora = f"{base}_lora"
        base_16 = f"{base}_finetune_16bit"
        base_4 = f"{base}_finetune_4bit"

    def replace_hf_prefix(name, new_name):
        if "/" in name:
            prefix = name.split("/", 1)[0]
            if prefix == "hf":
                prefix = "HF_USERNAME"
            return f"{prefix}/{new_name}"
        return new_name

    def replace_common(text):
        """Apply common text replacements for both code and markdown cells."""
        text = text.replace("</a></a>", "</a>")
        text = text.replace(
            "To install Unsloth your local device",
            "To install Unsloth on your local device",
        )
        text = re.sub(r"!{2,}", "!", text)
        text = text.replace("ee notice", "we notice")

        # Convert versions like X.X.X to 2026.2.1
        text = re.sub(r"[\d]{4}\.[\d]{1,2}\.[\d]{1,2}([^\d])", r"2026.2.1\1", text)

        # Change gguf-quantization-options link
        text = text.replace(
            "https://github.com/unslothai/unsloth/wiki#gguf-quantization-options",
            "https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf",
        )
        text = text.replace("https://docs.unsloth.ai/", "https://unsloth.ai/docs/")

        # Redirect Alpaca dataset
        text = text.replace(
            "https://huggingface.co/datasets/yahma/alpaca-cleaned",
            "https://huggingface.co/datasets/unsloth/alpaca-cleaned",
        )
        text = text.replace("yahma/alpaca-cleaned", "unsloth/alpaca-cleaned")
        text = text.replace("Alpaca dataset from [yahma]", "[Alpaca dataset]")

        # Train on completions
        text = text.replace(
            "TRL's docs [here](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).",
            "our docs [here](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#training-on-completions-only-masking-out-inputs)",
        )

        # Fix incorrect conversational link pointing to Alpaca notebook
        text = text.replace(
            "conversational [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Alpaca.ipynb)",
            "conversational [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Conversational.ipynb)",
        )

        # Fix Meta-Llama
        text = text.replace("unsloth/Meta-Llama", "unsloth/Llama")

        # TRL's `DPOTrainer`
        text = text.replace("TRL's `DPOTrainer`", "`DPOTrainer` and `GRPOTrainer` for reinforcement learning!")

        # Move packing = ...
        text = re.sub(
            r"(\n[ \t]*)packing\s*=\s*(True|False).*?\n(\1args\s*=\s*SFTConfig\(\n)",
            r"\3\1    packing = \2, # Makes training 2-5x faster for short sequences,\n",
            text,
        )

        # Ensure GGUF usage line matches base name used in code
        text = re.sub(
            r"Now, use the `[^`]+\.Q8_0\.gguf` file or `[^`]+\.Q4_K_M\.gguf` file in llama\.cpp\.",
            f"Now, use the `{base_gguf}.Q8_0.gguf` file or `{base_gguf}.Q4_K_M.gguf` file in llama.cpp.",
            text,
        )

        # Fix concatenated markdown line if it slipped in
        text = text.replace("Unsloth!Now, use the", "Unsloth!\nNow, use the")

        # Update docs domain
        text = text.replace("docs.unsloth.ai", "unsloth.ai/docs")
        text = text.replace("[Wiki page]", "[docs page]")
        text = text.replace("[wiki page]", "[docs page]")

        text = text.replace(
            "You can go to https://huggingface.co/settings/tokens for your personal tokens.",
            "You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.",
        )

        # GGUF filename references
        text = text.replace("model-unsloth-Q4_K_M.gguf", f"{base_gguf}.Q4_K_M.gguf")
        text = text.replace("model-unsloth.Q4_K_M.gguf", f"{base_gguf}.Q4_K_M.gguf")
        text = text.replace("model-unsloth.Q8_0.gguf", f"{base_gguf}.Q8_0.gguf")
        text = text.replace("model-unsloth.gguf", f"{base_gguf}.Q8_0.gguf")

        # Fix "Huggingface" -> "Hugging Face" (only capitalized, not in URLs/packages)
        text = text.replace("Huggingface's", "Hugging Face's")
        text = re.sub(r"Huggingface  (`[^`]+`)", r"Hugging Face \1", text)
        text = text.replace("Huggingface TRL's", "Hugging Face TRL's")

        # Fix instruction_part missing < before |end_of_role|>
        text = text.replace(
            '<|start_of_role|>user|end_of_role|>',
            '<|start_of_role|>user<|end_of_role|>',
        )

        # Fix typos in specific phrases
        text = text.replace("Prime and Prejudice", "Pride and Prejudice")
        text = text.replace("2x Telsa T4s", "2x Tesla T4s")
        text = text.replace("float32 s disable", "float32 so disable")
        text = text.replace("and its amazing", "and it's amazing")
        text = text.replace("look like this:", "looks like this:")
        text = text.replace("Replace with out specific", "Replace without specific")
        text = text.replace("AutoModelForPeftCausalLM", "AutoPeftModelForCausalLM")

        # Remove @nocommit placeholders
        text = re.sub(r'\[@nocommit[^\]]*\]\([^\)]*\)\.?', '', text)

        # Fix empty Open Math Reasoning URL
        text = text.replace(
            "[Open Math Reasoning]()",
            "[Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)"
        )

        # Fix footer heading
        text = text.replace("Some other links:", "Some other resources:")

        # Fix old installation URL paths (both variants)
        text = text.replace(
            "unsloth.ai/docs/get-started/installing-+-updating",
            "unsloth.ai/docs/get-started/install"
        )
        text = text.replace(
            "unsloth.ai/docs/get-started/install-and-update",
            "unsloth.ai/docs/get-started/install"
        )

        # Fix footer numbering (6. → 4.)
        text = re.sub(r'\n6\. See notebooks for DPO', r'\n4. See notebooks for DPO', text)

        # Fix duplicate "See our docs" sentences (same line duplicates)
        text = re.sub(
            r'(See \[our docs\]\([^)]+\) for more deployment options\.)\s*\1',
            r'\1',
            text
        )

        # Fix Nemo → NeMo capitalization (but not Mistral-Nemo model names)
        text = re.sub(r'\bNemo Gym\b', 'NeMo Gym', text)

        return text

    def replace_code(text):
        """Apply code-specific replacements."""
        # Update gguf save/push names
        text = re.sub(
            r"(save_pretrained_gguf\(\s*)([\"\'])([^\"\']*)([\"\'])",
            rf"\1\2{base_gguf}\4",
            text,
            flags=re.DOTALL,
        )

        def _replace_push_gguf(match):
            new_name = replace_hf_prefix(match.group(3), base_gguf)
            return f"{match.group(1)}{match.group(2)}{new_name}{match.group(4)}"

        text = re.sub(
            r"(push_to_hub_gguf\(\s*)([\"\'])([^\"\']*)([\"\'])",
            _replace_push_gguf,
            text,
            flags=re.DOTALL,
        )

        # Update merged save/push names
        def _replace_save_merged(match):
            method = match.group(6)
            new_name = base_16 if method == "merged_16bit" else base_4
            return f"{match.group(1)}{match.group(2)}{new_name}{match.group(4)}{match.group(5)}{method}{match.group(7)}"

        text = re.sub(
            r"(save_pretrained_merged\(\s*)([\"\'])([^\"\']*)([\"\'])(.*?save_method\s*=\s*[\"\'])(merged_16bit|merged_4bit|mxfp4)([\"\'])",
            _replace_save_merged,
            text,
            flags=re.DOTALL,
        )

        def _replace_push_merged(match):
            method = match.group(6)
            new_name = base_16 if method == "merged_16bit" else base_4
            replaced = replace_hf_prefix(match.group(3), new_name)
            return f"{match.group(1)}{match.group(2)}{replaced}{match.group(4)}{match.group(5)}{method}{match.group(7)}"

        text = re.sub(
            r"(push_to_hub_merged\(\s*)([\"\'])([^\"\']*)([\"\'])(.*?save_method\s*=\s*[\"\'])(merged_16bit|merged_4bit|mxfp4)([\"\'])",
            _replace_push_merged,
            text,
            flags=re.DOTALL,
        )

        # Update LoRA save/push names
        text = re.sub(
            r"(\b(?:model|tokenizer|processor)\.save_pretrained\(\s*)([\"\'])([^\"\']*)([\"\'])",
            rf"\1\2{base_lora}\4",
            text,
        )

        def _replace_push_lora(match):
            new_name = replace_hf_prefix(match.group(3), base_lora)
            return f"{match.group(1)}{match.group(2)}{new_name}{match.group(4)}"

        text = re.sub(
            r"(\b(?:model|tokenizer|processor)\.push_to_hub\(\s*)([\"\'])([^\"\']*)([\"\'])",
            _replace_push_lora,
            text,
        )

        # LoRA load snippets
        text = re.sub(
            r"(model_name\s*=\s*)([\"\'])([^\"\']*)([\"\'])([^\n]*YOUR MODEL YOU USED FOR TRAINING)",
            rf"\1\2{base_lora}\4\5",
            text,
        )
        text = re.sub(
            r"([\"\'])([^\"\']*)([\"\'])([^\n]*YOUR MODEL YOU USED FOR TRAINING)",
            rf"\1{base_lora}\3\4",
            text,
        )
        text = re.sub(r"([\"\'])lora_model([\"\'])", rf"\1{base_lora}\2", text)
        text = re.sub(r"([\"\'])finetuned_model([\"\'])", rf"\1{base_lora}\2", text)

        # Also handle AutoPeftModelForCausalLM.from_pretrained("xxx_lora")
        # and AutoTokenizer.from_pretrained("xxx_lora") for load-back consistency
        text = re.sub(
            r"(Auto(?:PeftModel\w*|Tokenizer|Model\w*)\.from_pretrained\(\s*)([\"\'])([^\"\']*_lora[^\"\']*)([\"\'])",
            rf"\1\2{base_lora}\4",
            text,
        )

        # Update hf/ to HF_USERNAME/ in quoted strings
        text = text.replace('"hf/', '"HF_USERNAME/')
        text = text.replace("'hf/", "'HF_USERNAME/")

        # Update tokens - only match string literals to avoid breaking token = get_token()
        text = re.sub(
            r'(\btoken\s*=\s*)([\"\'])([^\"\']*)([\"\'])',
            r'\1"YOUR_HF_TOKEN"',
            text,
        )

        # Preserve special tokens that should not be replaced by HF token
        text = re.sub(
            r"unsloth_eos_token\s*=\s*[\"\']YOUR_HF_TOKEN[\"\']",
            'unsloth_eos_token = "eos_token"',
            text,
        )
        text = re.sub(
            r"patch_token\s*=\s*[\"\']YOUR_HF_TOKEN[\"\']",
            'patch_token = "<|IMAGE_PLACEHOLDER|>"',
            text,
        )

        # If dtype=None helper line is directly before from_pretrained and dtype=dtype is used,
        # drop the helper line and inline dtype=None with the standard comment.
        dtype_comment = "None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+"
        dtype_line_re = re.compile(r"^[ \t]*dtype\s*=\s*None\s*#.*$")
        dtype_param_re = re.compile(r"(\bdtype\s*=\s*)dtype\b\s*,?")

        lines = text.splitlines(True)
        updated_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if dtype_line_re.match(line) and i + 1 < len(lines) and ".from_pretrained" in lines[i + 1]:
                # Try to update dtype within the from_pretrained call
                replaced = False
                depth = 0
                j = i + 1
                while j < len(lines):
                    current = lines[j]
                    if j == i + 1 and ".from_pretrained" not in current:
                        break
                    new_current, count = dtype_param_re.subn(
                        r"\1None, # " + dtype_comment,
                        current,
                    )
                    if count:
                        replaced = True
                    lines[j] = new_current
                    depth += current.count("(") - current.count(")")
                    if depth <= 0 and ".from_pretrained" in lines[i + 1]:
                        break
                    j += 1
                if replaced:
                    # Drop the dtype helper line and continue from the call
                    i += 1
                    continue
            updated_lines.append(line)
            i += 1
        text = "".join(updated_lines)

        # Normalize vLLM naming in code where it is used as a package/path
        # Use word boundary to preserve UNSLOTH_VLLM_STANDBY env var
        text = re.sub(r'\bvLLM\b', 'vllm', text)
        text = re.sub(r'\bVLLM\b(?!_)', 'vllm', text)

        # Simplify gated models comment
        text = re.sub(
            r"# use one if using gated models.*",
            "# HF Token for gated models",
            text,
        )

        # Fix A=A to A = A in code
        text = _space_equals_in_code(text)

        return text

    updated = False
    for cell in notebook_content.get("cells", []):
        if not isinstance(cell.get("source"), list):
            continue
        is_code = cell.get("cell_type") == "code"
        text = "".join(cell["source"])
        new_text = replace_common(text)
        if is_code:
            new_text = replace_code(new_text)
        if new_text != text:
            updated = True
        cell["source"] = _strip_extra_trailing_blank_lines(new_text.splitlines(True))

    if updated:
        with open(filename, "w", encoding="utf-8") as w:
            json.dump(notebook_content, w, indent=1)
        os.chmod(filename, 0o644)
pass


DONT_UPDATE_EXCEPTIONS = [
    "Falcon_H1-Alpaca.ipynb",
    "Liquid_LFM2-Conversational.ipynb",
    "Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb", # Daniel's?
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb",
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb",
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb",
    "Qwen3_VL_(8B)-Vision-GRPO.ipynb",
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb",
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb",
    "Synthetic_Data_Hackathon.ipynb",
    "Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb"
]


FIRST_MAPPING_NAME = {
    "gpt-oss-(20B)-Fine-tuning.ipynb" : "gpt_oss_(20B)-Fine-tuning.ipynb",
    "Qwen2_5_7B_VL_GRPO.ipynb" : "Qwen2.5_VL_(7B)-Vision-GRPO.ipynb",
    "Qwen3_(4B)-Instruct.ipynb" : "Qwen3_(4B)-Conversational.ipynb",
    "Qwen3_(4B)_Instruct-QAT.ipynb" : "Qwen3_(4B)-QAT.ipynb",

    # GPT OSS 
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb" : "(DGX Spark)-gpt-oss-(20B)-GRPO-2048.ipynb",
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb" : "gpt-oss-(20B)-GRPO-2048.ipynb",
    "Deepseek_OCR_(3B).ipynb" : "Deepseek_OCR_(3B)-Fine-Tuning.ipynb",
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb" : "(OpenEnv)-gpt-oss-BF16-(20B)-GRPO-2048.ipynb",
    "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb" : "gpt-oss-BF16-(20B)-GRPO-2048.ipynb",
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb" : "(OpenEnv)-gpt-oss-(20B)-GRPO-2048.ipynb",
    "GPT_OSS_BNB_(20B)-Inference.ipynb" : "gpt-oss-BNB-(20B)-Inference.ipynb",
    "GPT_OSS_MXFP4_(20B)-Inference.ipynb" : "gpt-oss-MXFP4-(20B)-Inference.ipynb",
    "gpt_oss_(20B)_500K_Context_Fine_tuning" : "gpt_oss_(20B)-500K-Context.ipynb",

    # Gemma
    "Gemma3_(4B).ipynb" : "Gemma3_(4B)-Conversational.ipynb",
    "Gemma3_(270M).ipynb" : "Gemma3_(270M)-Conversational.ipynb",

    # Granite
    "Granite4.0_350M.ipynb" : "Granite4.0_(350M)-Conversational.ipynb",
    "Granite4.0.ipynb" : "Granite4.0_(3B)-Conversational.ipynb",

    # Bert
    "bert_classification.ipynb" : "ModernBERT_(Large)-Classification.ipynb",

    # Whisper
    "Whisper.ipynb" : "Whisper_(Large)-Fine-Tuning.ipynb",

    # Spark
    "Spark_TTS_(0_5B).ipynb" : "Spark_TTS_(0.5B)-TTS.ipynb",

    # FP8
    "Qwen3_8B_FP8_GRPO.ipynb" : "Qwen3_(8B)-FP8-GRPO.ipynb",
    "Llama_FP8_GRPO.ipynb" : "Llama3.2_(1B)-FP8-GRPO.ipynb",

    # Ministral
    "Ministral_3_VL_(3B)_Vision.ipynb" : "Ministral3_VL_(3B)-Vision.ipynb",
    "Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb" : "Ministral3_(3B)-GRPO-Sudoku.ipynb"
}

def get_current_git_branch():
    try:
        # Run the git command to get the current branch name
        # '--abbrev-ref HEAD' gives the branch name (e.g., 'main', 'feature/new-feature')
        # 'STDOUT' captures standard output, 'STDERR' redirects error output
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')
        return branch_name
    except subprocess.CalledProcessError as e:
        print(f"Error getting Git branch: {e}")
        print(f"Command output: {e.output.decode('utf-8')}")
        return None
    except FileNotFoundError:
        print("Error: 'git' command not found. Make sure Git is installed and in your PATH.")
        return None


def update_or_append_pip_install(base_content, package_name, new_install_line):
    pattern = re.compile(rf"^!(uv )?pip install .*?{package_name}.*$", re.MULTILINE)

    updated_content, substitutions_count = pattern.subn(new_install_line, base_content)

    if substitutions_count == 0:
        output = base_content.strip() + "\n" + new_install_line
    else:
        output = updated_content
    return output

current_branch = get_current_git_branch()
# =======================================================
# GENERAL ANNOUNCEMENTS (THE VERY TOP)
# =======================================================

general_announcement_content = """To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)"""

general_announcement_content_a100 = general_announcement_content.replace("on a **free** Tesla T4 Google Colab instance!", "on your A100 Google Colab Pro instance!")
general_announcement_content_fp8 = general_announcement_content.replace("on a **free** Tesla T4 Google Colab instance!", "on your L4 Google Colab Pro instance!")

announcement_separation = '<div class="align-center">'

general_announcement_content_hf_course = general_announcement_content.split(announcement_separation)
general_announcement_content_hf_course = general_announcement_content_hf_course[0] + announcement_separation + '<a href="https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt"><img src="https://github.com/unslothai/notebooks/raw/main/assets/hf%20course.png" width="165"></a>' + general_announcement_content_hf_course[1]
general_announcement_content_hf_course = general_announcement_content_hf_course.split("To install Unsloth")
hf_additional_string_announcement = "In this [Hugging Face](https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt) and Unsloth notebook, you will learn to transform {full_model_name} into a Reasoning model using GRPO."
general_announcement_content_hf_course = (
    general_announcement_content_hf_course[0] + 
    hf_additional_string_announcement + 
    "\n\n" +
    "To install Unsloth" + general_announcement_content_hf_course[1]
)

general_announcement_content_meta = general_announcement_content.split(announcement_separation)
general_announcement_content_meta = general_announcement_content_meta[0] + "\n\n" + '<a href="https://github.com/meta-llama/synthetic-data-kit"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/meta%20round%20logo.png" width="137"></a>' + general_announcement_content_meta[1]

# CONSTANT
PIN_TRANSFORMERS = "!pip install transformers==4.56.2"
UV_PIN_TRANSFORMERS = PIN_TRANSFORMERS.replace("pip", "uv pip")

PIN_TRL = "!pip install --no-deps trl==0.22.2"
UV_PIN_TRL = PIN_TRL.replace("pip", "uv pip")
SPACES = " " * 4

# =======================================================
# INSTALLATION (MANY OF THIS IS SPECIFIC TO ONE OF THE NOTEBOOKS)
# =======================================================

XFORMERS_INSTALL = """xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")"""

installation_content = """%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # Do this in local & cloud setups
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    __XFORMERS_INSTALL__
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
""".replace("__XFORMERS_INSTALL__", XFORMERS_INSTALL)
installation_content = update_or_append_pip_install(
    installation_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_content = update_or_append_pip_install(
    installation_content,
    "trl",
    PIN_TRL,
)

installation_kaggle_content = """%%capture
import os

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install --upgrade transformers "huggingface_hub>=0.34.0" "datasets==4.3.0"
"""

installation_kaggle_content = update_or_append_pip_install(
    installation_kaggle_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_kaggle_content = update_or_append_pip_install(
    installation_kaggle_content,
    "trl",
    PIN_TRL,
)

# =======================================================
# GRPO Notebook
# =======================================================

installation_grpo_content = """%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # [NEW] Extra 30% context lengths!
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install or uv pip install
    !pip install unsloth vllm
else:
    pass # For Colab / Kaggle, we need extra instructions hidden below \\/"""

installation_extra_grpo_content = r"""#@title Colab Extra Install { display-mode: "form" }
%%capture
import os
!pip install --upgrade -qqq uv
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install!
    !pip install unsloth vllm
else:
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.10.2', 'triton')
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}"""

installation_extra_grpo_content = update_or_append_pip_install(
    installation_extra_grpo_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)
installation_extra_grpo_content = update_or_append_pip_install(
    installation_extra_grpo_content,
    "trl",
    UV_PIN_TRL,
)


installation_grpo_kaggle_content = """%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # [NEW] Extra 30% context lengths!
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.10.2', 'triton')
!uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
!uv pip install -qqq {_triton} "huggingface_hub>=0.34.0" "datasets==4.3.0"
"""

installation_grpo_kaggle_content = update_or_append_pip_install(
    installation_grpo_kaggle_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)

installation_grpo_kaggle_content = update_or_append_pip_install(
    installation_grpo_kaggle_content,
    "trl",
    UV_PIN_TRL,
)

# =======================================================
# Meta Synthetic Data Kit Notebook
# =======================================================

installation_synthetic_data_content = """%%capture
import os
!pip install --upgrade -qqq uv
if "COLAB_" not in "".join(os.environ.keys()):
    # If you're not in Colab, just use pip install!
    !pip install unsloth vllm synthetic-data-kit==0.0.3
else:
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.10.2', 'triton')
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}
    !uv pip install synthetic-data-kit==0.0.3"""

installation_synthetic_data_content = update_or_append_pip_install(
    installation_synthetic_data_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)

installation_synthetic_data_content = update_or_append_pip_install(
    installation_synthetic_data_content,
    "trl",
    UV_PIN_TRL,
)

installation_grpo_synthetic_data_content = """%%capture
!pip install --upgrade -qqq uv
try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
except: _numpy = "numpy"; _pil = "pillow"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
_vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.10.2', 'triton')
!uv pip install -qqq --upgrade unsloth {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers
!uv pip install -qqq {_triton}
!uv pip install "huggingface_hub>=0.34.0" "datasets==4.3.0"
!uv pip install synthetic-data-kit==0.0.3"""
installation_grpo_synthetic_data_content = update_or_append_pip_install(
    installation_grpo_synthetic_data_content,
    "transformers",
    UV_PIN_TRANSFORMERS,
)
installation_grpo_synthetic_data_content = update_or_append_pip_install(
    installation_grpo_synthetic_data_content,
    "trl",
    UV_PIN_TRL,
)

# =======================================================
# Orpheus Notebook
# =======================================================

# Add install snac under install unsloth
installation_orpheus_content = installation_content + """\n!pip install snac torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_orpheus_kaggle_content = installation_kaggle_content + """\n!pip install snac torchcodec \"datasets>=3.4.1,<4.0.0\""""

# =======================================================
# Whisper Notebook
# =======================================================

installation_whisper_content = installation_content + """\n!pip install librosa soundfile evaluate jiwer torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_whisper_kaggle_content = installation_kaggle_content + """\n!pip install librosa soundfile evaluate jiwer torchcodec \"datasets>=3.4.1,<4.0.0\""""

# =======================================================
# Spark Notebook
# =======================================================

installation_spark_content = installation_content + """\n!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_spark_kaggle_content = installation_kaggle_content + """\n!git clone https://github.com/SparkAudio/Spark-TTS
!pip install omegaconf einx torchcodec \"datasets>=3.4.1,<4.0.0\""""

# =======================================================
# GPT OSS Notebook
# =======================================================
installation_gpt_oss_content = r"""%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):    
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \
        "torch>=2.8.0" "triton>=3.4.0" {_numpy} {_pil} torchvision bitsandbytes "transformers==4.56.2" \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \
        git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo"""

# installation_gpt_oss_content = update_or_append_pip_install(
#     installation_gpt_oss_content,
#     "transformers",
#     "!uv pip install transformers==4.56.2",
# )
# installation_gpt_oss_content = update_or_append_pip_install(
#     installation_gpt_oss_content,
#     "trl",
#     UV_PIN_TRL,
# )

installation_gpt_oss_kaggle_content = installation_gpt_oss_content

# =======================================================
# Oute Notebook
# =======================================================

installation_oute_content = installation_content + """\n!pip install omegaconf einx
!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS
import os
os.remove("/content/OuteTTS/outetts/models/gguf_model.py")
os.remove("/content/OuteTTS/outetts/interface.py")
os.remove("/content/OuteTTS/outetts/__init__.py")
!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy torchcodec \"datasets>=3.4.1,<4.0.0\"
!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""

installation_oute_kaggle_content = installation_kaggle_content + """\n!pip install omegaconf einx
!rm -rf OuteTTS && git clone https://github.com/edwko/OuteTTS
import os
os.remove("/content/OuteTTS/outetts/models/gguf_model.py")
os.remove("/content/OuteTTS/outetts/interface.py")
os.remove("/content/OuteTTS/outetts/__init__.py")
!pip install pyloudnorm openai-whisper uroman MeCab loguru flatten_dict ffmpy randomname argbind tiktoken ftfy torchcodec \"datasets>=3.4.1,<4.0.0\"
!pip install descript-audio-codec descript-audiotools julius openai-whisper --no-deps
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""

# =======================================================
# Llasa Notebook
# =======================================================

# Llasa Need Unsloth==2025.4.1, Transformers==4.48 to running stable, and trl ==0.15.2
# installation_llasa_content = re.sub(r'\bunsloth\b(==[\d\.]*)?', 'unsloth==2025.4.1', installation_content)
installation_llasa_content = installation_content
installation_llasa_content = re.sub(r'\btrl\b(==[\d\.]*)?', 'trl==0.15.2', installation_llasa_content)

installation_llasa_content += """\

!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps
!pip install omegaconf torchcodec \"datasets>=3.4.1,<4.0.0\"
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""
installation_llasa_content = update_or_append_pip_install(
    installation_llasa_content,
    "transformers",
    "!pip install transformers==4.56.1",
)

installation_llasa_kaggle_content = installation_kaggle_content + """\n!pip install torchtune torchao vector_quantize_pytorch einx tiktoken xcodec2==0.1.5 --no-deps
!pip install omegaconf torchcodec \"datasets>=3.4.1,<4.0.0\"
%env UNSLOTH_DISABLE_FAST_GENERATION = 1"""
installation_llasa_kaggle_content = update_or_append_pip_install(
    installation_llasa_kaggle_content,
    "transformers",
    "!pip install transformers==4.48",
)
installation_llasa_kaggle_content = update_or_append_pip_install(
    installation_llasa_kaggle_content,
    "trl",
    PIN_TRL,
)

# =======================================================
# Tool Calling Notebook
# =======================================================

installation_tool_calling_content = installation_content + """\n!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg"""
installation_tool_calling_kaggle_content = installation_kaggle_content + """\n!pip install protobuf==3.20.3 # required
!pip install --no-deps transformers-cfg"""

# =======================================================
# Sesame CSM Notebook
# =======================================================
installation_sesame_csm_content = installation_content + """\n!pip install torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_sesame_csm_content = update_or_append_pip_install(
    installation_sesame_csm_content,
    "transformers",
    "!pip install transformers==4.52.3",
)
installation_sesame_csm_content = update_or_append_pip_install(
    installation_sesame_csm_content,
    "trl",
    PIN_TRL
)

installation_sesame_csm_kaggle_content = installation_kaggle_content + """\n!pip install torchcodec \"datasets>=3.4.1,<4.0.0\""""
installation_sesame_csm_kaggle_content = update_or_append_pip_install(
    installation_sesame_csm_kaggle_content,
    "transformers",
    "!pip install transformers==4.52.3 torchcodec",
)
installation_sesame_csm_kaggle_content = update_or_append_pip_install(
    installation_sesame_csm_kaggle_content,
    "trl",
    PIN_TRL
)

# =======================================================
# Llama Vision Notebook
# =======================================================
installation_llama_vision_content = installation_content
installation_llama_vision_content = update_or_append_pip_install(
    installation_llama_vision_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_llama_vision_content = update_or_append_pip_install(
    installation_llama_vision_content,
    "trl",
    PIN_TRL
)


installation_llama_vision_kaggle_content = installation_kaggle_content
installation_llama_vision_kaggle_content = update_or_append_pip_install(
    installation_llama_vision_kaggle_content,
    "transformers",
    PIN_TRANSFORMERS,
)
installation_llama_vision_kaggle_content = update_or_append_pip_install(
    installation_llama_vision_kaggle_content,
    "trl",
    PIN_TRL
)

# =======================================================
# Gemma3N Notebook
# =======================================================
gemma3n_extra_content = """\

!pip install torchcodec
import torch; torch._dynamo.config.recompile_limit = 64;"""
installation_gemma3n_content = installation_content 
installation_gemma3n_content += gemma3n_extra_content

installation_gemma3n_kaggle_content = installation_kaggle_content
installation_gemma3n_kaggle_content += gemma3n_extra_content

# =======================================================
# Qwen3VL Notebook
# =======================================================
gemma3n_extra_content = """\

!pip install torchcodec
import torch; torch._dynamo.config.recompile_limit = 64;"""
installation_qwen3_vl_content = installation_content 
installation_qwen3_vl_content = update_or_append_pip_install(
    installation_qwen3_vl_content,
    "transformers",
    "!pip install transformers==4.57.1",
)

installation_qwen3_vl_kaggle_content  = installation_kaggle_content
installation_qwen3_vl_kaggle_content  = update_or_append_pip_install(
    installation_qwen3_vl_kaggle_content,
    "transformers",
    "!pip install transformers==4.57.1",
)



# =======================================================
# SGLang Notebook
# =======================================================
installation_sglang_content = """%%capture
import sys
import os
!git clone https://github.com/sgl-project/sglang.git && cd sglang && pip install -e "python[all]"
!pip install -U transformers==4.53.0
sys.path.append(f'{os.getcwd()}/sglang/')
sys.path.append(f'{os.getcwd()}/sglang/python')"""
installation_sglang_kaggle_content = installation_sglang_content

# =======================================================
# Deepseek OCR Notebook
# =======================================================
installation_deepseek_ocr_content = installation_content
installation_deepseek_ocr_content += """\n!pip install jiwer
!pip install einops addict easydict"""

installation_deepseek_ocr_kaggle_content = installation_kaggle_content
installation_deepseek_ocr_kaggle_content += """\n!pip install jiwer
!pip install einops addict easydict"""

# =======================================================
# ERNIE_4_5_VL Notebook
# =======================================================
installation_ernie_4_5_vl_content = installation_content
installation_ernie_4_5_vl_content += """\n!pip install decord"""

installation_ernie_4_5_vl_kaggle_content = installation_kaggle_content
installation_ernie_4_5_vl_kaggle_content += """\n!pip install decord"""

# =======================================================
# Nemotron 3 Nano  Notebook
# =======================================================
installation_nemotron_nano_content = """%%capture
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):    
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \\
        "torch==2.7.1" "triton>=3.3.0" {_numpy} {_pil} torchvision bitsandbytes "transformers==4.56.2" \\
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo

# Mamba is supported only on torch==2.7.1. If you have newer torch versions, please wait 30 minutes!
!uv pip install --no-build-isolation mamba_ssm==2.2.5 causal_conv1d==1.5.2"""

installation_nemotron_nano_kaggle_content = installation_nemotron_nano_content


# =======================================================
# QAT Notebook
# =======================================================
installation_qat_content = """%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9]{1,}\.[0-9]{1,}", str(torch.__version__)).group(0)
    __XFORMERS_INSTALL__
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
!pip install torchao==0.14.0 fbgemm-gpu-genai==1.4.2
!pip install transformers==4.55.4 && pip install --no-deps trl==0.22.2""".replace("__XFORMERS_INSTALL__", XFORMERS_INSTALL)
installation_qat_kaggle_content = installation_qat_content

# =======================================================
# Ministral Notebook
# =======================================================
installation_ministral_content = installation_content
installation_ministral_content = update_or_append_pip_install(
    installation_ministral_content,
    "transformers",
    "!pip install git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce"
)

installation_ministral_kaggle_content = installation_kaggle_content
installation_ministral_kaggle_content = update_or_append_pip_install(
    installation_ministral_kaggle_content,
    "transformers",
    "!pip install git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce"
)

# =======================================================
# NEWS (WILL KEEP CHANGING THIS)
# =======================================================

new_announcement = """
Train MoEs - DeepSeek, GLM, Qwen and gpt-oss faster with 32% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)

You can now train embedding models 1.8-3.3x faster with 20% less VRAM. [Blog](https://unsloth.ai/docs/new/embedding-finetuning)

Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)

3x faster LLM training with 30% less VRAM and 500K context. [3x faster](https://unsloth.ai/docs/new/3x-faster-training-packing) • [500K Context](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)

New in Reinforcement Learning: [FP8 RL](https://docs.unsloth.ai/new/fp8-reinforcement-learning) • [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) • [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning)

Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)."""

# =======================================================
# LAST BLOCK CLOSE STATEMENT
# =======================================================

OTHER_RESOURCES = """Some other resources:
1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install-and-update) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
"""

text_for_last_cell_gguf = """And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

__OTHER_RESOURCES__
<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️

  <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
</div>""".replace("__OTHER_RESOURCES__", OTHER_RESOURCES)

text_for_last_cell_ollama = text_for_last_cell_gguf.replace("Now, ", "You can also ", 1)

text_for_last_cell_gemma3 = text_for_last_cell_gguf.replace("model-unsloth", "gemma-3-finetune")

text_for_last_cell_non_gguf = """And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

__OTHER_RESOURCES__
<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️

  This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
</div>""".replace("__OTHER_RESOURCES__", OTHER_RESOURCES)

hf_course_name = "HuggingFace Course"

ARCHITECTURE_MAPPING = {
    # Gemma Family
    "gemma": "Gemma",
    "codegemma": "Gemma", # Explicitly map specific models if needed

    # Llama Family
    "llama": "Llama",
    "tinylama": "Llama",

    # Qwen Family
    "qwen": "Qwen",

    # Phi Family
    "phi": "Phi",

    # Mistral Family
    "mistral": "Mistral",
    "pixtral": "Mistral",
    "zephyr": "Mistral",
    "Magistral" : "Mistral",
    "Ministral" : "Mistral",

    # Whisper
    "whisper": "Whisper",

    # Text-to-Speech Models (Group or keep separate?)
    "oute": "TTS", 
    "llasa": "TTS",
    "spark": "TTS",
    "orpheus": "TTS",
    "sesame": "TTS",

    # gpt oss
    "gpt oss": "GPT-OSS",

    # Linear Attention
    "falcon": "Linear Attention",
    "liquid": "Linear Attention",

    # Deepseek
    "deepseek": "Deepseek",

    # Granite
    "granite": "Granite",
    
    # Bert
    "bert": "BERT",
    "modernbert": "BERT",

    # Other Models (Assign architecture or keep specific)
    # 'codeforces': 'CodeForces Model', # Example
    # 'unsloth': 'Unsloth Model',     # Example
    "meta synthetic data": "Llama",
}

TYPE_MAPPING = {
    "Gemma3N" : {
        "Conversational" : "Multimodal"
    },
    "Meta Synthetic Data" : {
        "Synthetic Data" : "GRPO",
        "GRPO LoRA" : "GRPO"
    },
}

KNOWN_TYPES_ORDERED = [
    "Tool Calling",          
    "Text Completion",       
    "Synthetic Data",        
    "Reasoning Conversational",
    "Vision GRPO",
    "Fine Tuning",
    "500K Context",
    "QAT",
    
    "Conversational",
    "Alpaca",
    "Vision",
    "Reasoning",
    "Completion",
    "Finetune",             
    "Studio",               
    "Coder",                
    "Inference",            
    "Ollama",               
    "Audio",                
    "Thinking",

    # FP8 GRPO
    "FP8 GRPO",

    # GPT OSS
    "GRPO 2048",
    "GRPO Sudoku",
    
    "ORPO",
    "GRPO",
    "DPO",
    "CPT",
    "TTS",                  
    "LoRA",
    "VL",                   
    "RAFT",

    # Deepseek OCR
    "Evaluation",
    "Eval",

    # BERT, ModernBERT,
    "Classification",
]

def extract_model_info_refined(filename, architecture_mapping, known_types_ordered):
    if not filename.endswith(".ipynb"):
        return {'name': filename, 'size': None, 'type': None, 'architecture': None}
    stem = filename[:-len(".ipynb")]

    requires_a100 = False
    if 'A100' in stem:
        requires_a100 = True
        stem = stem.replace('_A100', '')

    original_stem_parts = stem.replace('+', '_').split('_') 
    type_ = None
    stem_searchable = stem.lower().replace('_', ' ').replace('+', ' ')
    found_type_indices = [] 

    for type_keyword in known_types_ordered:
        kw_lower = type_keyword.lower()
        pattern = r'\b' + re.escape(kw_lower) + r'\b'
        match = re.search(pattern, stem_searchable)
        if match:
            type_ = type_keyword 
            try:
                 
                 kw_parts = type_keyword.split(' ')
                 for i in range(len(original_stem_parts) - len(kw_parts) + 1):
                     match_parts = True
                     for j in range(len(kw_parts)):
                         if original_stem_parts[i+j].lower() != kw_parts[j].lower():
                             match_parts = False
                             break
                     if match_parts:
                         found_type_indices = list(range(i, i + len(kw_parts)))
                         break
            except Exception:
                pass 
            break 
    size = None
    size_match = re.search(r'_\((.*?)\)', stem)
    size_start_index = -1
    if size_match:
        size = size_match.group(1)
        size_start_index = size_match.start() 
    name = None
    if size_start_index != -1:
        name_part = stem[:size_start_index]
        name = name_part.replace('_', ' ').strip()
        if not name:
             post_size_part = stem[size_match.end():]
             if post_size_part.startswith('_'): post_size_part = post_size_part[1:]
             if post_size_part.startswith('+'): post_size_part = post_size_part[1:]
             name = post_size_part.replace('_', ' ').replace('+', ' ').strip()
    else:
        name = stem.replace('_', ' ').strip()
        if type_ and name.lower().endswith(type_.lower()):
            name = name[:-len(type_)].strip()

    if not name:
        name_parts_filtered = [p for i, p in enumerate(original_stem_parts) if i not in found_type_indices]
        name = ' '.join(name_parts_filtered).strip()
        if not name: 
             name = stem.replace('_',' ').strip()

    architecture = None
    if name: 
        name_lower_for_mapping = name.lower()
        sorted_keys = sorted(architecture_mapping.keys(), key=len, reverse=True)
        for key in sorted_keys:
            
            pattern = r'\b' + re.escape(key.lower()) + r'\b'
            if re.search(pattern, name_lower_for_mapping):
                architecture = architecture_mapping[key]
                break
            elif key.lower() in name_lower_for_mapping and architecture is None:
               architecture = architecture_mapping[key]

    for key in TYPE_MAPPING:
        if key.lower() in name.lower():
            type_ = TYPE_MAPPING[key].get(type_, type_)
            break
    for key in TYPE_MAPPING:
        kaggle_key = f"Kaggle {key}"
        if kaggle_key.lower() in name.lower():
            type_ = TYPE_MAPPING.get(kaggle_key, {}).get(type_, type_)
            break

    if "kaggle" in name.lower():
        # Remove "kaggle" from the name
        name = name.replace("Kaggle", "").strip()

    return {'name': name,
            'size': size,
            'type': type_,
            'architecture': architecture,
            'requires_a100': requires_a100}

extracted_info_refined = {}
original_template_path = os.path.abspath("original_template")
list_files = [f for f in os.listdir(original_template_path) if f.endswith(".ipynb")]
standardized_name = [f.replace("-", "_") for f in list_files]

standard_to_original_name = {
    k : v for k, v in zip(standardized_name, list_files)
}
original_to_standard_name = {
    v : k for k, v in zip(standardized_name, list_files)
}
list_files = [f for f in os.listdir(original_template_path) if f.endswith(".ipynb")]
for std_name in standard_to_original_name:
    extracted_info_refined[std_name] = extract_model_info_refined(
        std_name,
        ARCHITECTURE_MAPPING,
        KNOWN_TYPES_ORDERED  
    )

badge_section = '<a href="{link_colab}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'




def copy_folder(source_path, new_name, destination_path=None, replace=False):
    if destination_path is None:
        destination_path = os.path.dirname(source_path)

    new_path = os.path.join(destination_path, new_name)

    try:
        if replace and os.path.exists(new_path):
            shutil.rmtree(new_path)
            print(f"Removed existing folder: '{new_path}'")

        shutil.copytree(source_path, new_path)
        print(f"Successfully copied '{source_path}' to '{new_path}'")
    except FileNotFoundError:
        print(f"Error: Source folder '{source_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def is_path_contains_any(file_path, words):
    return any(re.search(word, file_path, re.IGNORECASE) for word in words)

def extract_version_from_row(row):
    """Extracts the version number from a row string for sorting."""
    match = re.search(r"\| (.*?) \|", row)  # Match content between first "|" and " |"
    if match:
        model_name = match.group(1)
        return extract_version(model_name)
    else:
        return (0, 0)

def extract_version(model_name):
    """Extracts the version number for sorting.

    Handles cases like:
        - Phi 3 Medium
        - Phi 3.5 Mini
        - Phi 4
    Returns a tuple of (major version, minor version) for proper sorting.
    Returns (0, 0) if no version is found.
    """
    match = re.search(r"(\d+(\.\d+)?)", model_name)
    if match:
        version_str = match.group(1)
        if "." in version_str:
            major, minor = version_str.split(".")
            return (int(major), int(minor))
        else:
            return (int(version_str), 0)
    else:
        return (0, 0)


def update_notebook_sections(
    notebook_path,
    general_announcement,
    installation_steps,
    installation_steps_kaggle,
    new_announcement,
):
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)

        updated = False

        first_markdown_index = -1
        news_markdown_index = -1

        for i, cell in enumerate(notebook_content["cells"]):
            if cell["cell_type"] == "markdown":
                if first_markdown_index == -1:
                    first_markdown_index = i

                source_str = "".join(cell["source"]).strip()

                if "###" in source_str:
                    news_markdown_index = i
                    break

        if f"{hf_course_name}-" in notebook_path: 
            full_model_name = notebook_path.split("/")[-1].replace(".ipynb", "")
            full_model_name = full_model_name.split("-")
            full_model_name = " ".join(full_model_name[1:]).replace("_", " ")
            general_announcement = general_announcement_content_hf_course.format(full_model_name=full_model_name)
        elif "Meta" in notebook_path:
            general_announcement = general_announcement_content_meta
        elif "A100" in notebook_path:
            general_announcement = general_announcement_content_a100

        # Update the general announcement section
        if first_markdown_index != -1:
            if news_markdown_index == first_markdown_index:
                # "# News" is the first markdown, insert above it
                if first_markdown_index >= 0:
                    notebook_content["cells"].insert(
                        first_markdown_index,
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                f"{line}\n"
                                for line in general_announcement.splitlines()
                            ],
                        },
                    )
                    updated = True
                    news_markdown_index += 1  # Adjust index since a new cell is added
                else:
                    notebook_content["cells"][first_markdown_index]["source"] = [
                        f"{line}\n" for line in general_announcement.splitlines()
                    ]
                    updated = True
            elif not "".join(
                notebook_content["cells"][first_markdown_index]["source"]
            ).strip():
                # First markdown is empty, replace it
                notebook_content["cells"][first_markdown_index]["source"] = [
                    f"{line}\n" for line in general_announcement.splitlines()
                ]
                updated = True

        i = 0 if news_markdown_index == -1 else news_markdown_index

        is_gguf = False
        is_ollama = False
        is_gemma3 = is_path_contains_any(notebook_path.lower(), ["gemma3"])
        is_llama = is_path_contains_any(notebook_path.lower(), ["llama"])
        is_vision = is_path_contains_any(notebook_path.lower(), ["vision"])
        is_qwen3 = is_path_contains_any(notebook_path.lower(), ["qwen3"])

        while i < len(notebook_content["cells"]):
            cell = notebook_content["cells"][i]

            if cell["cell_type"] == "markdown":
                source_str = "".join(cell["source"]).strip()

                if "### Ollama Support" in source_str:
                    is_ollama = True
                elif "gguf" in source_str and not is_gemma3:
                    is_gguf = True

                if source_str == "### News":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
                    ):
                        announcement = new_announcement
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in announcement.splitlines()
                        ]
                        updated = True
                        i += 1
                elif source_str == "### Installation":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "code"
                    ):
                        if is_path_contains_any(notebook_path, ["kaggle"]):
                            installation = installation_steps_kaggle
                        else:
                            installation = installation_steps

                        # GRPO INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["grpo"]) and not is_path_contains_any(notebook_path.lower(), ["gpt_oss", "gpt-oss"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_grpo_kaggle_content
                                # Kaggle will delete the second cell instead -> Need to check
                                del notebook_content["cells"][i + 2]
                            else:
                                installation = installation_grpo_content
                                # TODO: Remove after GRPO numpy bug fixed!
                                # Error : ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                                notebook_content["cells"][i + 2]["source"] = installation_extra_grpo_content

                        # META INSTALLATION
                        elif is_path_contains_any(notebook_path.lower(), ["Meta"]): 
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_grpo_synthetic_data_content
                                # Kaggle will delete the second cell instead -> Need to check
                                del notebook_content["cells"][i + 2]
                            else:
                                installation = installation_synthetic_data_content
                                # TODO: Remove after GRPO numpy bug fixed!
                                # Error : ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                                notebook_content["cells"][i + 2]["source"] = installation_extra_grpo_content
                        
                        # ORPHEUS INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["orpheus"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_orpheus_kaggle_content
                            else:
                                installation = installation_orpheus_content

                        # WHISPER INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["whisper"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_whisper_kaggle_content
                            else:
                                installation = installation_whisper_content

                        # SPARK INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["spark"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_spark_kaggle_content
                            else:
                                installation = installation_spark_content

                        # OUTE INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["oute"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_oute_kaggle_content
                            else:
                                installation = installation_oute_content

                        # LLASA INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["llasa"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_llasa_kaggle_content
                            else:
                                installation = installation_llasa_content

                        # TOOL CALLING INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["tool_calling"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_tool_calling_kaggle_content
                            else:
                                installation = installation_tool_calling_content

                        # SESAME CSM INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["sesame_csm"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_sesame_csm_kaggle_content
                            else:
                                installation = installation_sesame_csm_content

                        # SGLANG INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["sglang"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_sglang_kaggle_content
                            else:
                                installation = installation_sglang_content

                        # QAT INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["qat"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_qat_kaggle_content
                            else:
                                installation = installation_qat_content
                                
                        # GPT OSS INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["gpt_oss", "gpt-oss"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_gpt_oss_kaggle_content
                            else:
                                installation = installation_gpt_oss_content

                        # Llama Vision INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["llama"]) and is_vision:
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_llama_vision_kaggle_content
                            else:
                                installation = installation_llama_vision_content

                        # Gemma3N INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["gemma3n"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_gemma3n_kaggle_content
                            else:
                                installation = installation_gemma3n_content

                        # ERNIE VL INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["ernie_4_5_vl"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_ernie_4_5_vl_kaggle_content
                            else:
                                installation = installation_ernie_4_5_vl_content
                                
                        # Deepseek OCR INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["deepseek_ocr"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_deepseek_ocr_kaggle_content
                            else:
                                installation = installation_deepseek_ocr_content
                                
                        # Qwen3VL INSTALLATION
                        if is_path_contains_any(notebook_path.lower(), ["qwen3"]) and is_vision:
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_qwen3_vl_kaggle_content
                            else:
                                installation = installation_qwen3_vl_content
                                
                        # Nemotron Nano 3 INSTALLATION also Granite has mamba
                        if is_path_contains_any(notebook_path.lower(), ["nemotron-3-nano","nemotron-nano-3", "granite4"]):
                            if is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                                installation = installation_nemotron_nano_kaggle_content
                            else:
                                installation = installation_nemotron_nano_content
                                
                        notebook_content["cells"][i + 1]["source"] = installation
                        updated = True
                        # TODO: Remove after GRPO numpy bug fixed! 
                        # Error: ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
                        if is_path_contains_any(notebook_path.lower(), ["grpo"]) and not is_path_contains_any(notebook_path.lower(), ["kaggle"]):
                            i += 2
                        else:
                            i += 1

            i += 1

        # Add text to the last cell
        if notebook_content["cells"]:
            last_cell = notebook_content["cells"][-1]
            if is_ollama:
                text_for_last_cell = text_for_last_cell_ollama
            elif is_gguf:
                text_for_last_cell = text_for_last_cell_gguf
            elif is_gemma3 and not is_vision and is_gguf: # Vision cannot be transformed to GGUF yet
                text_for_last_cell = text_for_last_cell_gemma3
            else:
                text_for_last_cell = text_for_last_cell_non_gguf

            if last_cell["cell_type"] == "markdown":
                # Check if the last cell already contains footer content using key markers
                existing_text = "".join(last_cell["source"])
                # Key markers that indicate footer content already exists
                footer_markers = [
                    "And we're done! If you have any questions on Unsloth",
                    "Train your own reasoning model - Llama GRPO notebook",
                    "This notebook and all Unsloth notebooks are licensed"
                ]
                # Specific check for LGPL license line
                lgpl_marker = "This notebook and all Unsloth notebooks are licensed [LGPL-3.0]"

                # Check if notebook has partial footer content but missing LGPL line
                has_partial_footer = any(marker in existing_text for marker in footer_markers[:2])  # First two markers only
                has_lgpl = lgpl_marker in existing_text

                # Add content if:
                # 1. No footer markers at all, OR
                # 2. Has partial footer but missing LGPL license line
                if not any(marker in existing_text for marker in footer_markers) or (has_partial_footer and not has_lgpl):
                    # If there's partial footer but missing LGPL, only add the LGPL line
                    if has_partial_footer and not has_lgpl:
                        # Add just the LGPL license line
                        last_cell["source"].append("\n  This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).\n")
                    else:
                        # Add complete footer
                        last_cell["source"].extend(
                            [f"{line}\n" for line in text_for_last_cell.splitlines()]
                        )
                    updated = True  # Mark as updated only if content was added
            else:
                notebook_content["cells"].append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f"{line}\n" for line in text_for_last_cell.splitlines()
                        ],
                    }
                )
                updated = True

        # Ensure GPU metadata is set for Colab
        if "metadata" not in notebook_content:
            notebook_content["metadata"] = {}
        if "accelerator" not in notebook_content["metadata"]:
            notebook_content["metadata"]["accelerator"] = "GPU"
            updated = True
        if "colab" not in notebook_content["metadata"]:
            notebook_content["metadata"]["colab"] = {"provenance": [], "gpuType" : "T4", "include_colab_link": True}
            updated = True
        if "kernelspec" not in notebook_content["metadata"]:
            notebook_content["metadata"]["kernelspec"] = {
                "display_name": "Python 3",
                "name": "python3",
            }
            updated = True
        # Fix rendering in github
        if "widgets" not in notebook_content["metadata"]:
            notebook_content["metadata"]["widgets"] = {
                "application/vnd.jupyter.widget-state+json" : {
                    "state" : {}
                }
            }
            updated = True
        if notebook_content["metadata"]["widgets"].get("application/vnd.jupyter.widget-state+json", None) is not None:
            notebook_content["metadata"]["widgets"]["application/vnd.jupyter.widget-state+json"]["state"] = {}
            updated = True

        if updated:
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=1)
            os.chmod(notebook_path, 0o644)
            print(f"Updated: {notebook_path}")
        else:
            print(f"No sections found to update in: {notebook_path}")

    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in notebook at {notebook_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {notebook_path}: {e}")


import re

def replace(text, g, f):
    text = text.replace("(", r"\(")
    text = text.replace(")", r"\)")
    if g == "":
        g = g + "\n"
    else:
        g = "\1" + g + "\2"
    f = re.sub(
        r"([\s]{1,})([\"\'][ ]{0,})" + text + r"(\\n[\"\']\,\n)",
        g,
        f,
        flags = re.MULTILINE,
    )
    if " = " not in text:
        # Also replace x=x and x = x
        text = text.replace("=", " = ")
        f = re.sub(
            r"([\s]{1,})([\"\'][ ]{0,})" + text + r"(\\n[\"\']\,\n)",
            g,
            f,
            flags = re.MULTILINE,
        )
    return f
pass

def update_unsloth_config(filename):
    with open(filename, "r", encoding = "utf-8") as f: f = f.read()
    if "from transformers import TrainingArguments\\n" not in f: return
    if "from trl import SFTTrainer\\n" not in f: return
    if "SFTConfig" in f: return
    if "UnslothTrainingArguments" in f: return

    f = replace("from unsloth import is_bfloat16_supported", "", f)
    f = replace("from transformers import TrainingArguments", "", f)
    f = f.replace("from trl import SFTTrainer", "from trl import SFTTrainer, SFTConfig")
    f = f.replace("TrainingArguments(\\n", "SFTConfig(\\n")
    f = replace("fp16=not is_bfloat16_supported(),", "", f)
    f = replace("bf16=is_bfloat16_supported(),", "", f)
    f = replace("fp16 = not is_bfloat16_supported(),", "", f)
    f = replace("bf16 = is_bfloat16_supported(),", "", f)
    f = replace("logging_steps=1,", "", f)
    f = replace("logging_steps = 1,", "", f)
    f = replace("dataset_num_proc=2,", "", f)
    f = replace("dataset_num_proc=4,", "", f)
    f = replace("dataset_num_proc = 2,", "", f)
    f = replace("dataset_num_proc = 4,", "", f)

    # Fix all spacings x=x to x = x
    spaces = r'(\"[ ]{4,}[^\<\n]{1,}[^ \=\'\"])\=([^ \=\'\"].*?\,\n)'
    f = re.sub(spaces, r"\1 = \2", f)

    with open(filename, "w", encoding = "utf-8") as w: w.write(f)
pass


def main():
    notebook_directory = "nb"
    notebook_pattern = "*.ipynb"

    notebook_files = glob(os.path.join(notebook_directory, notebook_pattern))
    print(f"Found {len(notebook_files)} notebooks")
    # filter out the DONT_UPDATE_EXCEPTIONS
    notebook_files = [x for x in notebook_files if os.path.basename(x) not in DONT_UPDATE_EXCEPTIONS]
    print(f"Filtered out {len(DONT_UPDATE_EXCEPTIONS)} notebooks")
    print(f"Remaining {len(notebook_files)} notebooks")

    if not notebook_files:
        print(
            f"No notebooks found in the directory: {notebook_directory} with pattern: {notebook_pattern}"
        )
        return

    for notebook_file in notebook_files:
        update_notebook_sections(
            notebook_file,
            general_announcement_content,
            installation_content,
            installation_kaggle_content,
            new_announcement,
        )
        # update_unsloth_config(notebook_file)
        update_old_unsloth(notebook_file)

    # Spelling check
    print("\n=== Spelling Check ===")
    spell_issues_found = False
    for notebook_file in notebook_files:
        try:
            with open(notebook_file, "r", encoding="utf-8") as f:
                nb_content = json.load(f)
            fixed, issues = check_spelling(nb_content, os.path.basename(notebook_file))
            if fixed:
                with open(notebook_file, "w", encoding="utf-8") as f:
                    json.dump(nb_content, f, indent=1)
                os.chmod(notebook_file, 0o644)
                print(f"  AUTO-FIXED spelling in {os.path.basename(notebook_file)}")
            if issues:
                spell_issues_found = True
                for cell_idx, words in issues:
                    print(f"  SPELLING: {os.path.basename(notebook_file)} cell {cell_idx}: {words}")
        except Exception:
            pass
    if not spell_issues_found:
        print("  No spelling issues found.")

    # AST syntax check
    print("\n=== AST Syntax Check ===")
    syntax_issues_found = False
    for notebook_file in notebook_files:
        errors = validate_notebook_syntax(notebook_file)
        if errors:
            syntax_issues_found = True
            for cell_idx, lineno, msg in errors:
                print(f"  SYNTAX: {os.path.basename(notebook_file)} cell {cell_idx} line {lineno}: {msg}")
    if not syntax_issues_found:
        print("  No syntax issues found.")


def add_colab_badge(notebooks_dir):
    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    for path in paths:
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"])
        is_colab = not is_kaggle
        if is_colab:
            with open(path, "r", encoding="utf-8") as f:
                notebook_content = json.load(f)

            badge = badge_section.format(link_colab=(f"https://colab.research.google.com/github/unslothai/notebooks/blob/main/"+path).replace(" ", "%20"))
            notebook_content["cells"].insert(
                0,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"{line}\n"
                        for line in badge.splitlines()
                    ],
                },
            )

            with open(path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=1)
            os.chmod(path, 0o644)


def update_readme(
    args,
    readme_path,
    notebooks_dir,
    architecture_mapping, 
    known_types_ordered,  
    type_order=None,      
    kaggle_accelerator="nvidiaTeslaT4",
):
    base_url_colab = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/"
    base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/unslothai/notebooks/blob/main/"

    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))
    paths = [x.replace("\\", "/") for x in paths]

    list_models = ['GRPO'] 
    unique_architectures = sorted(list(set(architecture_mapping.values())))
    for arch in unique_architectures:
        if arch not in list_models:
            list_models.append(arch)
    list_models.append('Other') 

    sections = {}
    for section in list_models:
        sections[section] = {
            "Colab": {"header": f"### {section} Notebooks\n", "rows": []},
            "Kaggle": {"header": f"### {section} Notebooks\n", "rows": []},
        }

    colab_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"
    kaggle_table_header = "| Model | Type | Notebook Link |\n| --- | --- | --- |\n"

    notebook_data = []

    print(f"Processing {len(paths)} notebooks...")
    for path in paths:
        # Ignore HF course and Advanced notebooks
        if is_path_contains_any(path.lower(), [hf_course_name.lower(), "Advanced".lower()]):
            continue

        notebook_name = os.path.basename(path)
        old_notebook_name = notebook_name
        check = False
        if notebook_name in FIRST_MAPPING_NAME:
            notebook_name = FIRST_MAPPING_NAME[notebook_name]
            check = True
        
        # For Kaggle
        if notebook_name.lstrip("Kaggle-") in FIRST_MAPPING_NAME:
            notebook_name = FIRST_MAPPING_NAME[notebook_name.lstrip("Kaggle-")]
            notebook_name = "Kaggle-" + notebook_name

        std_notebook_name = notebook_name.replace("-", "_")
        is_kaggle = is_path_contains_any(path.lower(), ["kaggle"]) 

        try:
            info = extract_model_info_refined(
                std_notebook_name,
                architecture_mapping,
                known_types_ordered
            )
        except Exception as e:
            print(f"Error processing {notebook_name}: {e}")
            info = {'name': notebook_name.replace('.ipynb',''), 'size': None, 'type': 'Error', 'architecture': None, 'requires_a100': False} # Fallback

        model_name = info['name'] if info and info['name'] else notebook_name.replace('.ipynb','') 
        model_type = info['type'] if info and info['type'] else "" 
        architecture = info['architecture'] if info else None
        size = info['size'] 
        size = size.replace(r"_", " ") if size else None 
        size = f"**({size})**" if size else ""

        requires_a100 = info.get('requires_a100', False)

        section_name = "Other" 
        if model_type == 'GRPO':
            section_name = 'GRPO'
        elif architecture and architecture in list_models:
             section_name = architecture
        link_base = base_url_kaggle if is_kaggle else base_url_colab
        link_url = f"{link_base}{path}"

        if is_kaggle:
            image_src = "https://kaggle.com/static/images/open-in-kaggle.svg"
            image_alt = "Open in Kaggle"
            if kaggle_accelerator:
                link_url += f"&accelerator={kaggle_accelerator}"
        else:
            image_src = "https://colab.research.google.com/assets/colab-badge.svg"
            image_alt = "Open In Colab"
        link = f'<a href="{link_url}" target="_blank" rel="noopener noreferrer"><img src="{image_src}" alt="{image_alt}"></a>'

        notebook_data.append(
            {
                "model": model_name,
                "type": model_type,
                "link": link,
                "section": section_name,
                "path": path, 
                "architecture" : architecture, 
                "size" : size, 
                "requires_a100": requires_a100,
            }
        )

    def get_sort_key(x):
        section_index = list_models.index(x["section"])
        version_key = extract_version(x["model"]) 

        type_sort_val = float("inf") 
        current_type = x["type"].strip('*') 
        if type_order and current_type in type_order:
            type_sort_val = type_order.index(current_type)
        elif current_type: 
             type_sort_val = current_type

        return version_key

    notebook_data.sort(key=get_sort_key)

    for data in notebook_data:
        model_prefix = "(A100) " if data.get('requires_a100', False) else ""
        row = f"| **{model_prefix}{data['model']}** {data['size']} | {data['type']} | {data['link']} |\n"
        platform = "Kaggle" if "kaggle" in data['link'].lower() else "Colab"
        sections[data["section"]][platform]["rows"].append(row)

    for section in sections:
        try:
            sections[section]["Colab"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Colab rows for section '{section}' by version: {e}")
        try:
            sections[section]["Kaggle"]["rows"].sort(key=lambda x: extract_version_from_row(x), reverse=True)
        except Exception as e:
            print(f"Warning: Could not sort Kaggle rows for section '{section}' by version: {e}")

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        start_marker = "<!-- START OF EDITING -->"
        start_index = readme_content.find(start_marker)
        if start_index == -1:
            raise ValueError(f"Start marker '{start_marker}' not found in README.")
        start_index += len(start_marker)

        end_marker_alt = None
        end_marker = "<!-- End of Notebook Links -->"
        end_index = readme_content.find(end_marker)
        if end_index == -1:
            end_marker_alt = "# 📒 Kaggle Notebooks"
            end_index = readme_content.find(end_marker_alt)
            if end_index == -1:
                raise ValueError(f"End marker '{end_marker}' or '{end_marker_alt}' not found in README.")
        content_before = readme_content[:start_index]
        content_after = readme_content[end_index:] 

        temp = (
            "(https://github.com/unslothai/notebooks/#-kaggle-notebooks).\n\n"
            if args.to_main_repo
            else "(https://github.com/unslothai/notebooks/#-kaggle-notebooks).\n\n"
        )

        colab_updated_notebooks_links = "\n"

        kaggle_updated_notebooks_links = (
            "# 📒 Kaggle Notebooks\n"
            "<details>\n  <summary>\n" 
            "    Click for all our Kaggle notebooks categorized by model:\n  "
            "</summary>\n\n"
        )

        for section in list_models:
            if sections[section]["Colab"]["rows"]:
                colab_updated_notebooks_links += sections[section]["Colab"]["header"]
                colab_updated_notebooks_links += colab_table_header
                colab_updated_notebooks_links += "".join(sections[section]["Colab"]["rows"]) + "\n"

            if sections[section]["Kaggle"]["rows"]:
                kaggle_updated_notebooks_links += sections[section]["Kaggle"]["header"]
                kaggle_updated_notebooks_links += kaggle_table_header
                kaggle_updated_notebooks_links += "".join(sections[section]["Kaggle"]["rows"]) + "\n"

        kaggle_updated_notebooks_links += "</details>\n\n"

        now = datetime.now() 
        timestamp = f"\n"

        updated_readme_content = (
            content_before
            + colab_updated_notebooks_links
            + kaggle_updated_notebooks_links 
            + timestamp
            + content_after 
        )

        if end_marker_alt and end_index != -1:
             content_after = readme_content[end_index:]
             next_section_index = content_after.find("\n#")
             if next_section_index != -1:
                 content_after = content_after[next_section_index:] 
             else:
                  
                  explicit_end_marker_index = content_after.find("")
                  if explicit_end_marker_index != -1:
                      content_after = content_after[explicit_end_marker_index:]
                  else:
                      content_after = "" 

             updated_readme_content = ( 
                content_before
                + colab_updated_notebooks_links
                + kaggle_updated_notebooks_links 
                + timestamp
                + content_after
             )


        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_readme_content)

        print(f"Successfully updated {readme_path}")

    except FileNotFoundError:
        print(f"Error: README file '{readme_path}' not found.")
    except ValueError as ve:
        print(f"Error processing README: {ve}")
    except Exception as e:
        print(f"An error occurred while updating {readme_path}: {e}")
        import traceback
        traceback.print_exc()


def copy_and_update_notebooks(
    template_dir,
    destination_dir,
    general_announcement,
    installation,
    installation_kaggle,
    new_announcement,
):
    """Copies notebooks from template_dir to destination_dir, updates them, and renames them."""
    template_notebooks = glob(os.path.join(template_dir, "*.ipynb"))

    temp_location = os.path.join(destination_dir, ".temp_backup")
    if os.path.exists(destination_dir):
        if os.path.exists(temp_location):
            shutil.rmtree(temp_location)
        os.makedirs(temp_location, exist_ok=True)
        # Move everything currently in destination_dir into .temp_backup
        for entry in os.listdir(destination_dir):
            if entry == ".temp_backup":
                continue
            if entry not in DONT_UPDATE_EXCEPTIONS:
                continue
            src_path = os.path.join(destination_dir, entry)
            shutil.move(src_path, temp_location)
    else:
        os.makedirs(destination_dir, exist_ok=True)

    def _preserve_outputs(dest_path, template_path):
        """Copy template to dest, preserving output cells from existing dest if cell count matches."""
        existing_outputs = {}
        existing_nb = None
        if os.path.exists(dest_path):
            try:
                with open(dest_path, "r", encoding="utf-8") as f:
                    existing_nb = json.load(f)
                for idx, cell in enumerate(existing_nb.get("cells", [])):
                    if cell.get("outputs"):
                        existing_outputs[idx] = cell["outputs"]
            except Exception:
                existing_outputs = {}
                existing_nb = None

        shutil.copyfile(template_path, dest_path)
        os.chmod(dest_path, 0o644)

        if existing_outputs and existing_nb is not None:
            try:
                with open(dest_path, "r", encoding="utf-8") as f:
                    new_nb = json.load(f)
                if len(new_nb.get("cells", [])) == len(existing_nb.get("cells", [])):
                    for idx, outputs in existing_outputs.items():
                        if idx < len(new_nb["cells"]):
                            new_nb["cells"][idx]["outputs"] = outputs
                    with open(dest_path, "w", encoding="utf-8") as f:
                        json.dump(new_nb, f, indent=1)
                    os.chmod(dest_path, 0o644)
            except Exception:
                pass

    for template_notebook_path in template_notebooks:
        notebook_name = os.path.basename(template_notebook_path)

        colab_notebook_name = notebook_name
        destination_notebook_path = os.path.join(destination_dir, colab_notebook_name)

        _preserve_outputs(destination_notebook_path, template_notebook_path)
        print(f"Copied '{colab_notebook_name}' to '{destination_dir}'")

        kaggle_notebook_name = "Kaggle-" + notebook_name
        destination_notebook_path = os.path.join(destination_dir, kaggle_notebook_name)

        _preserve_outputs(destination_notebook_path, template_notebook_path)

        print(f"Copied '{kaggle_notebook_name}' to '{destination_dir}'")

        if "GRPO" in template_notebook_path:
            hf_course_notebook_name = f"{hf_course_name}-" + notebook_name
            destination_notebook_path = os.path.join(destination_dir, hf_course_notebook_name)
            _preserve_outputs(destination_notebook_path, template_notebook_path)
            print(f"Copied f'{hf_course_name}-{notebook_name}' to '{destination_notebook_path}'")

        update_notebook_sections(
            os.path.join(destination_dir, colab_notebook_name),
            general_announcement,
            installation,
            installation_kaggle,
            new_announcement,
        )

        update_notebook_sections(
            destination_notebook_path,
            general_announcement,
            installation_kaggle,
            installation_kaggle,
            new_announcement,
        )

    # Move Exceptions back to destination_dir from temp_location
    for entry in DONT_UPDATE_EXCEPTIONS:
        src_path = os.path.join(temp_location, entry)
        dst_path = os.path.join(destination_dir, entry)
        if os.path.exists(src_path):
            # shutil.rmtree(dst_path)
            shutil.move(src_path, dst_path)
            print(f"Moved '{entry}' back to '{dst_path}'")
        else:
            print(f"Warning: '{entry}' not found in '{temp_location}'")
    
    # finally remove the temp_location
    shutil.rmtree(temp_location)

def missing_files(nb: str | os.PathLike, original_template: str | os.PathLike) -> list[str]:
    nb_abs = os.path.abspath(nb)
    original_template_abs = os.path.abspath(original_template)

    files_in_nb = {f for f in os.listdir(nb_abs) if os.path.isfile(os.path.join(nb_abs, f))}
    files_in_original_template = {f for f in os.listdir(original_template_abs) if os.path.isfile(os.path.join(original_template_abs, f))}

    files_in_nb = {f for f in files_in_nb if not (f.startswith("Kaggle") or f.startswith("HuggingFace Course"))}
    files_in_original_template = {f for f in files_in_original_template if not f.startswith("Kaggle")}

    only_in_nb = files_in_nb - files_in_original_template
    return sorted(list(only_in_nb))


def remove_unwanted_section(script_content):
    start_marker = "# ### Installation"
    end_marker = "# ### Unsloth"

    start_index = script_content.find(start_marker)
    end_index = script_content.find(end_marker)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        before_section = script_content[:start_index]
        section_to_comment = script_content[start_index:end_index]
        after_section = script_content[end_index:]

        lines = section_to_comment.split('\n')
        commented_lines = [f"# {line}" for line in lines]
        commented_section = '\n'.join(commented_lines)
        return before_section + commented_section + after_section
    else:
        return script_content

def convert_notebook_to_script(notebook_path: str, output_path: str):
    exporter = PythonExporter()
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    (body, resources) = exporter.from_notebook_node(notebook_content)

    body = remove_unwanted_section(body)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)

    print(f"Converted {notebook_path} to {output_path}")

def convert_folder(input_folder: str, output_folder: str):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join(input_folder, filename)
            script_filename = filename.replace('.ipynb', '.py')
            output_path = os.path.join(output_folder, script_filename)
            convert_notebook_to_script(notebook_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--to_main_repo",
        action="store_true",
        help="Whether update notebooks and README.md for Unsloth main repository or not. Default is False.",
    )
    parser.add_argument(
        "--check_missing_files",
        action="store_true",
        help="Check for missing files in the destination directory compared to the original template.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="If true, instead of checking from original_template to nb, it will check nb to original_template instead"
    )
    parser.add_argument(
        "--disable_convert_to_script",
        action="store_true",
        help="If true, it will not convert the notebooks to scripts",
    )
    args = parser.parse_args()

    if args.check_missing_files:
        original_template = "original_template"
        nb = "nb"
        if args.reverse:
            missing_files_list = missing_files(original_template, nb)
        else:
            missing_files_list = missing_files(nb, original_template)
        if not missing_files_list:
            print("No missing files.")
        else:
            print(f"Missing files in {nb} compared to {original_template}:")
            for file in missing_files_list:
                if file not in DONT_UPDATE_EXCEPTIONS:
                    print(file)
        exit(0)
    copy_and_update_notebooks(
        "original_template",
        "nb",
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        new_announcement,
    )
    main()

    notebook_directory = "nb"
    readme_path = "README.md"
    type_order = [
        "Alpaca",
        "Conversational",
        "CPT",
        "DPO",
        "ORPO",
        "Text_Completion",
        "CSV",
        "Inference",
        "Unsloth_Studio",
        "GRPO"
    ]  # Define your desired order here
    update_readme(
        args, 
        readme_path, 
        notebook_directory, 
        ARCHITECTURE_MAPPING,
        KNOWN_TYPES_ORDERED,
        type_order
    )

    # Apply targeted fixes to ALL notebooks (including DONT_UPDATE_EXCEPTIONS)
    # These are safe fixes that should apply everywhere.
    _ALL_NB_FIXES = {
        "fibonnaci": "fibonacci",
        "Fibonnaci": "Fibonacci",
        "SHould": "Should",
        "GTP-OSS": "GPT-OSS",
        "stratgegy": "strategy",
        "verifer": "verifier",
        "verisons": "versions",
        "datases": "datasets",
        "Huggingface's": "Hugging Face's",
        "Huggingface TRL's": "Hugging Face TRL's",
        "Prime and Prejudice": "Pride and Prejudice",
        "2x Telsa T4s": "2x Tesla T4s",
        "float32 s disable": "float32 so disable",
        "and its amazing": "and it's amazing",
        "look like this:": "looks like this:",
        "Replace with out specific": "Replace without specific",
        "AutoModelForPeftCausalLM": "AutoPeftModelForCausalLM",
        "<|start_of_role|>user|end_of_role|>": "<|start_of_role|>user<|end_of_role|>",
        # New fixes
        "[Open Math Reasoning]()": "[Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)",
        "Some other links:": "Some other resources:",
        "unsloth.ai/docs/get-started/installing-+-updating": "unsloth.ai/docs/get-started/install",
        "unsloth.ai/docs/get-started/install-and-update": "unsloth.ai/docs/get-started/install",
        # Also handle old domain format that may be in exception files
        "docs.unsloth.ai/get-started/installing-+-updating": "unsloth.ai/docs/get-started/install",
        "docs.unsloth.ai/get-started/install-and-update": "unsloth.ai/docs/get-started/install",
        # Handle intermediate format (domain changed but path not)
        "unsloth.ai/get-started/installing-+-updating": "unsloth.ai/docs/get-started/install",
        "unsloth.ai/get-started/install-and-update": "unsloth.ai/docs/get-started/install",
        "Nemo Gym": "NeMo Gym",
        # Fix old domain for exception files
        "https://docs.unsloth.ai/": "https://unsloth.ai/docs/",
    }
    for nb_path in glob(os.path.join("nb", "*.ipynb")):
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                raw = f.read()
            new_raw = raw
            new_raw = re.sub(
                r"# use one if using gated models[^\n]*",
                "# HF Token for gated models",
                new_raw,
            )
            new_raw = re.sub(
                r"Huggingface  (`[^`]+`)",
                r"Hugging Face \1",
                new_raw,
            )
            new_raw = re.sub(
                r'\[@nocommit[^\]]*\]\([^\)]*\)\.?',
                '',
                new_raw,
            )
            for wrong, right in _ALL_NB_FIXES.items():
                new_raw = new_raw.replace(wrong, right)
            # Fix footer numbering (various formats)
            new_raw = re.sub(r'\n6\. See notebooks for DPO', r'\n4. See notebooks for DPO', new_raw)
            new_raw = re.sub(r'"6\. See notebooks for DPO', r'"4. See notebooks for DPO', new_raw)
            # Fix duplicate "See our docs" sentences
            new_raw = re.sub(
                r'(See \[our docs\]\([^)]+\) for more deployment options\.)\s*\1',
                r'\1',
                new_raw
            )
            # Fix broken #Save link ONLY if file has NO <a name="Save"> anchor
            if '[how to save it](#Save)' in new_raw and '<a name="Save">' not in new_raw:
                new_raw = new_raw.replace('[how to save it](#Save)', 'how to save it')
            if new_raw != raw:
                with open(nb_path, "w", encoding="utf-8") as f:
                    f.write(new_raw)
        except Exception:
            pass
        os.chmod(nb_path, 0o644)

    if not args.disable_convert_to_script:
        convert_folder("nb", "python_scripts")
