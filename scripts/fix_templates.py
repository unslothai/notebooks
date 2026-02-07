"""Fix template notebooks for PR #183 issues."""
import json
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "original_template")


def _detect_indent(path):
    """Detect JSON indent level from a notebook file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped and stripped != line:
                return len(line) - len(stripped)
    return 1


def load_nb(name):
    path = os.path.join(TEMPLATE_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def save_nb(path, data):
    indent = _detect_indent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def fix_bert_model_name():
    """Issue 1b: answerdotai/ModernBERT-large -> unsloth/ModernBERT-large"""
    path, nb = load_nb("bert_classification.ipynb")
    cell = nb["cells"][5]
    src = cell["source"]
    for i, line in enumerate(src):
        if "answerdotai/ModernBERT-large" in line:
            src[i] = line.replace("answerdotai/ModernBERT-large", "unsloth/ModernBERT-large")
            print(f"  Fixed bert_classification.ipynb cell 5 line {i}: {src[i].strip()}")
    save_nb(path, nb)


def fix_embedding_gemma_import():
    """Issue 2: Add 'import torch' to EmbeddingGemma cell 13."""
    path, nb = load_nb("EmbeddingGemma_(300M).ipynb")
    cell = nb["cells"][13]
    src = cell["source"]
    # Check if import torch already exists
    has_torch = any("import torch" in line for line in src)
    if not has_torch:
        src.insert(0, "import torch\n")
        print(f"  Added 'import torch' to EmbeddingGemma_(300M).ipynb cell 13")
    else:
        print(f"  EmbeddingGemma_(300M).ipynb cell 13 already has 'import torch'")
    save_nb(path, nb)


def fix_grpo_version_check(notebook_name, cell_idx):
    """Issue 4: Wrap apply_chat_template with TRL version check."""
    path, nb = load_nb(notebook_name)
    cell = nb["cells"][cell_idx]
    src = cell["source"]
    text = "".join(src)

    # Check if already has Version check
    if "Version" in text:
        print(f"  {notebook_name} cell {cell_idx} already has Version check")
        return

    # Build new source with version check wrapping the existing code
    new_lines = []
    new_lines.append("from unsloth_zoo.utils import Version\n")
    new_lines.append("\n")
    new_lines.append("# Only apply chat template for TRL < 0.24.0, otherwise TRL handles it\n")
    new_lines.append('if Version("trl") < Version("0.24.0"):\n')

    # Indent all existing lines by 4 spaces
    for line in src:
        if line.strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append("    " + line)

    cell["source"] = new_lines
    save_nb(path, nb)
    print(f"  Added Version check to {notebook_name} cell {cell_idx}")


def fix_push_to_hub(notebook_name, cell_idx, remove_line_containing, replace_with=None):
    """Issue 5: Fix tokenizer/processor push_to_hub lines."""
    path, nb = load_nb(notebook_name)
    cell = nb["cells"][cell_idx]
    src = cell["source"]

    new_src = []
    for line in src:
        if remove_line_containing in line:
            if replace_with is not None:
                # Replace with the new line
                new_src.append(line.replace(remove_line_containing, replace_with))
                print(f"  {notebook_name} cell {cell_idx}: replaced '{remove_line_containing.strip()}' with '{replace_with.strip()}'")
            else:
                # Remove the line entirely
                print(f"  {notebook_name} cell {cell_idx}: removed '{line.strip()}'")
                continue
        else:
            new_src.append(line)

    cell["source"] = new_src
    save_nb(path, nb)


def main():
    print("=== Fixing template notebooks ===")

    # Issue 1b: bert_classification model name
    print("\n1b: Fix bert_classification model name")
    fix_bert_model_name()

    # Issue 2: EmbeddingGemma import torch
    print("\n2: Fix EmbeddingGemma import torch")
    fix_embedding_gemma_import()

    # Issue 4: TRL Version check for GRPO vision templates
    print("\n4: Add TRL Version checks")
    fix_grpo_version_check("Gemma3_(4B)-Vision-GRPO.ipynb", 21)
    fix_grpo_version_check("Qwen2_5_7B_VL_GRPO.ipynb", 19)
    fix_grpo_version_check("Qwen3_VL_(8B)-Vision-GRPO.ipynb", 20)

    # Issue 5: tokenizer vs processor push_to_hub
    print("\n5: Fix tokenizer/processor push_to_hub")

    # Gemma3_(4B)-Vision-GRPO.ipynb cell 33: uses model,tokenizer -> replace processor with tokenizer
    fix_push_to_hub(
        "Gemma3_(4B)-Vision-GRPO.ipynb", 33,
        remove_line_containing="# processor.push_to_hub(",
        replace_with="# tokenizer.push_to_hub(",
    )

    # Qwen3_VL_(8B)-Vision-GRPO.ipynb cell 36: uses model,tokenizer -> replace processor with tokenizer
    fix_push_to_hub(
        "Qwen3_VL_(8B)-Vision-GRPO.ipynb", 36,
        remove_line_containing="# processor.push_to_hub(",
        replace_with="# tokenizer.push_to_hub(",
    )

    # Uses model, processor -> remove tokenizer line
    for notebook_name, cell_idx in [
        ("Gemma3_(4B)-Vision.ipynb", 35),
        ("Gemma3N_(4B)-Audio.ipynb", 31),
        ("Gemma3N_(4B)-Vision.ipynb", 36),
        ("Sesame_CSM_(1B)-TTS.ipynb", 21),
    ]:
        fix_push_to_hub(
            notebook_name, cell_idx,
            remove_line_containing="# tokenizer.push_to_hub(",
        )

    # Fix 6: A100 metadata for all A100 templates
    print("\n6: Fix A100 gpuType metadata")
    fix_a100_metadata()

    print("\n=== All template fixes applied ===")


def fix_a100_metadata():
    """Set gpuType to A100 and fix T4 announcement text in all A100 template notebooks."""
    old_text = "a **free** Tesla T4 Google Colab instance"
    new_text = "your A100 Google Colab Pro instance"
    for name in os.listdir(TEMPLATE_DIR):
        if "A100" in name and name.endswith(".ipynb"):
            path, nb = load_nb(name)
            changed = False
            # Fix gpuType metadata
            colab = nb.get("metadata", {}).get("colab", {})
            if colab.get("gpuType") != "A100":
                nb.setdefault("metadata", {}).setdefault("colab", {})["gpuType"] = "A100"
                changed = True
                print(f"  Fixed {name}: gpuType -> A100")
            else:
                print(f"  {name}: gpuType already A100")
            # Fix announcement cell text (T4 -> A100)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "markdown":
                    continue
                for i, line in enumerate(cell.get("source", [])):
                    if old_text in line:
                        cell["source"][i] = line.replace(old_text, new_text)
                        changed = True
                        print(f"  Fixed {name}: T4 announcement -> A100")
            if changed:
                save_nb(path, nb)


if __name__ == "__main__":
    main()
