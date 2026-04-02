"""Fix angle-bracket tags hidden by GitHub's notebook renderer.

GitHub interprets raw angle-bracket tags like <start_working_out>, <SOLUTION>,
<think> as HTML and silently hides them. This script fixes:
1. Stored cell outputs: adds text/html with HTML-escaped content
2. Code comments: replaces angle-bracket content with safe text
String literals are left unchanged (unfixable without breaking functionality).

For stream outputs specifically, the fix converts them to display_data outputs
(since streams don't support text/html), preserving all original content.
"""
import json
import os
import re
import sys
from html import escape as html_escape

SCRIPT_DIR = os.path.dirname(__file__)
NB_DIR = os.path.join(SCRIPT_DIR, "..", "nb")
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "..", "original_template")

# Angle-bracket pattern that looks like an HTML tag (not special tokens like <|im_end|>)
HTML_LIKE_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9_]*>")

# Comment replacements: (old_substring, new_substring)
COMMENT_REPLACEMENTS = [
    ("# Acts as <think>",                                        "# Acts as think-open tag"),
    ("# Acts as </think>",                                       "# Acts as think-close tag"),
    ("# Remove generated <think> and </think>",                  "# Remove generated think tags"),
    ("# No need to reward <start_working_out> since",            "# No need to reward the opening tag since"),
    ("# No need to reward <think> since",                        "# No need to reward the think tag since"),
    ("# Match the answer after </think>, extracting",            "# Match the answer after the think-close tag, extracting"),
]


def _detect_indent(path):
    """Detect JSON indent level from a notebook file."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped and stripped != line:
                return len(line) - len(stripped)
    return 1


def load_nb(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return json.load(f)


def save_nb(path, data):
    indent = _detect_indent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def _text_has_html_tags(text):
    """Check if text contains angle-bracket sequences that look like HTML tags."""
    return bool(HTML_LIKE_TAG.search(text))


def _join_text(text_field):
    """Join a text field that may be a string or list of strings."""
    if isinstance(text_field, list):
        return "".join(text_field)
    return text_field



def _stream_to_display_data(output, text_str, original_text):
    """Convert a stream output to a display_data output preserving content.

    Stream outputs only support a ``text`` field -- no ``text/html``.
    By converting to ``display_data`` we can supply both ``text/plain``
    (for local Jupyter) and ``text/html`` (for GitHub rendering).
    """
    escaped = html_escape(text_str)
    html_content = "<pre>" + escaped + "</pre>"

    # Keep the same list/str format as the original text field
    if isinstance(original_text, list):
        html_lines = html_content.split("\n")
        html_field = []
        for i, line in enumerate(html_lines):
            if i < len(html_lines) - 1:
                html_field.append(line + "\n")
            else:
                if line:
                    html_field.append(line)
    else:
        html_field = html_content

    return {
        "output_type": "display_data",
        "data": {
            "text/plain": original_text,
            "text/html": html_field,
        },
        "metadata": {},
    }


def fix_outputs(nb, dry_run=False):
    """Fix stored cell outputs with angle-bracket tags.

    For execute_result/display_data: add text/html with escaped content.
    For stream outputs: convert to display_data preserving all content.
    """
    fixes = 0
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        outputs = cell.get("outputs", [])
        new_outputs = []
        for out_idx, output in enumerate(outputs):
            output_type = output.get("output_type", "")

            if output_type in ("execute_result", "display_data"):
                data = output.get("data", {})
                text_plain = data.get("text/plain")
                if text_plain is None:
                    new_outputs.append(output)
                    continue

                plain_str = _join_text(text_plain)
                if not _text_has_html_tags(plain_str):
                    new_outputs.append(output)
                    continue

                # Already has text/html -- skip
                if "text/html" in data:
                    new_outputs.append(output)
                    continue

                # Add text/html with escaped content wrapped in <pre>
                if not dry_run:
                    escaped = html_escape(plain_str)
                    html_content = "<pre>" + escaped + "</pre>"
                    if isinstance(text_plain, list):
                        html_lines = html_content.split("\n")
                        html_field = []
                        for i, line in enumerate(html_lines):
                            if i < len(html_lines) - 1:
                                html_field.append(line + "\n")
                            else:
                                if line:
                                    html_field.append(line)
                        data["text/html"] = html_field
                    else:
                        data["text/html"] = html_content
                fixes += 1
                action = "[DRY RUN] Would add" if dry_run else "Added"
                print(f"  {action} text/html to cell {cell_idx} output {out_idx} ({output_type})")
                new_outputs.append(output)

            elif output_type == "stream":
                text = output.get("text")
                if text is None:
                    new_outputs.append(output)
                    continue
                text_str = _join_text(text)
                if not _text_has_html_tags(text_str):
                    new_outputs.append(output)
                    continue

                # Convert stream to display_data preserving all content
                fixes += 1
                action = "[DRY RUN] Would convert" if dry_run else "Converted"
                print(f"  {action} stream to display_data in cell {cell_idx} output {out_idx}")
                if not dry_run:
                    new_output = _stream_to_display_data(output, text_str, text)
                    new_outputs.append(new_output)
                else:
                    new_outputs.append(output)
                continue
            else:
                new_outputs.append(output)

        if "outputs" in cell:
            cell["outputs"] = new_outputs
    return fixes


def fix_comments(nb, dry_run=False):
    """Fix code comments that contain angle-bracket tags."""
    fixes = 0
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        new_source = []
        cell_changed = False
        for line in source:
            new_line = line
            for old, new in COMMENT_REPLACEMENTS:
                if old in new_line:
                    new_line = new_line.replace(old, new)
                    cell_changed = True
            new_source.append(new_line)
        if cell_changed:
            fixes += 1
            if not dry_run:
                cell["source"] = new_source
            action = "[DRY RUN] Would fix" if dry_run else "Fixed"
            print(f"  {action} comments in cell {cell_idx}")
    return fixes


def process_notebook(path, dry_run=False):
    """Process a single notebook file."""
    nb = load_nb(path)

    output_fixes = fix_outputs(nb, dry_run=dry_run)
    comment_fixes = fix_comments(nb, dry_run=dry_run)

    total = output_fixes + comment_fixes
    if total > 0 and not dry_run:
        save_nb(path, nb)

    return output_fixes, comment_fixes


def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=== DRY RUN: No files will be modified ===\n")
    else:
        print("=== Fixing HTML tags in notebooks ===\n")

    total_output_fixes = 0
    total_comment_fixes = 0
    total_files = 0

    for dir_label, dir_path in [("nb", NB_DIR), ("original_template", TEMPLATE_DIR)]:
        print(f"--- Processing {dir_label}/ ---")
        if not os.path.isdir(dir_path):
            print(f"  Directory not found: {dir_path}")
            continue

        for name in sorted(os.listdir(dir_path)):
            if not name.endswith(".ipynb"):
                continue
            path = os.path.join(dir_path, name)
            output_fixes, comment_fixes = process_notebook(path, dry_run=dry_run)
            if output_fixes + comment_fixes > 0:
                total_output_fixes += output_fixes
                total_comment_fixes += comment_fixes
                total_files += 1
                print(f"  -> {name}: {output_fixes} output fix(es), {comment_fixes} comment fix(es)")
        print()

    print(f"=== Summary: {total_files} file(s), {total_output_fixes} output fix(es), {total_comment_fixes} comment fix(es) ===")


if __name__ == "__main__":
    main()
