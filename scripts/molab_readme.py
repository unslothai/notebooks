# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""molab_readme — renders the molab README section.

Import like every other ``scripts/`` helper::

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import molab_readme

``render_molab_readme_section(notebooks)`` returns the markdown block:
a heading, a one-line intro, and a badge table (one row per non-skipped
entry, in the order given).  The caller drops it between the
``<!-- MOLAB:START -->`` and ``<!-- MOLAB:END -->`` markers in
``README.md``.  Those markers are NOT emitted here.

Rules:

- Pure function.  No file I/O, no network, deterministic for byte-level
  parity tests.
- Notebooks with ``nb.skip == True`` are dropped from the table.
- Badge image is exactly ``https://marimo.io/molab-shield.svg``.
- Badge URL is
  ``https://molab.marimo.io/github/unslothai/notebooks/blob/main/molab/<file>.py``
  where ``<file>.py`` is ``nb.output.name``.
- No ``/wasm`` suffix.  None of the active notebooks are WASM-compatible.
- Wording: "Open in molab".  Do NOT hedge with "upcoming" or
  "coming soon".
"""
from __future__ import annotations

import re
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid a hard import cycle at type-check time; at runtime the caller
    # always imports molab_manifest before calling this module.
    from molab_manifest import MolabNotebook

# ---------------------------------------------------------------------------
# Badge constants (AC5 — must match exactly)
# ---------------------------------------------------------------------------
_BADGE_IMAGE = "https://marimo.io/molab-shield.svg"
_MOLAB_BASE_URL = "https://molab.marimo.io/github/unslothai/notebooks/blob/main/molab/"


def _badge_markdown(nb: "MolabNotebook") -> str:
    """Return the markdown badge string for a single notebook.

    Output format (matches existing Colab/Kaggle badge style in the README):
        [![Open in molab](<image>)](<url>)
    """
    filename = nb.output.name  # e.g. "TinyLlama_(1.1B)-Alpaca.py"
    url = f"{_MOLAB_BASE_URL}{filename}"
    return f"[![Open in molab]({_BADGE_IMAGE})]({url})"


# Task-type keywords and remaps, mirrored from update_all_notebooks.py
# (KNOWN_TYPES_ORDERED / TYPE_MAPPING) so the molab Type column matches the
# Colab and AMD tables. Longer phrases first so they win over substrings.
_KNOWN_TYPES_ORDERED = [
    "Tool Calling", "Text Completion", "Synthetic Data",
    "Reasoning Conversational", "Vision GRPO", "Fine Tuning", "500K Context",
    "QAT", "Conversational", "Alpaca", "Vision", "Reasoning", "Completion",
    "Finetune", "Studio", "Coder", "Inference", "Ollama", "Audio", "Thinking",
    "FP8 GRPO", "GRPO 2048", "GRPO Sudoku", "ORPO", "GRPO", "DPO", "CPT",
    "TTS", "LoRA", "VL", "RAFT", "Evaluation", "Eval", "Classification",
    "Mobile Actions",
]
_TYPE_MAPPING = {
    "Gemma3N": {"Conversational": "Multimodal"},
    "Meta Synthetic Data": {"Synthetic Data": "GRPO", "GRPO LoRA": "GRPO"},
}


def _model_type_size(stem: str) -> "tuple[str, str, str]":
    """Split a notebook stem into (model, type, size), like the Colab/AMD
    tables: type is the first matching keyword, size is the ``_(...)`` group,
    model is the remaining name."""
    s = stem.replace("-", "_")  # standardize separators before parsing
    if "A100" in s:
        s = s.replace("_A100", "")
    searchable = s.lower().replace("_", " ").replace("+", " ")
    type_ = ""
    for kw in _KNOWN_TYPES_ORDERED:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", searchable):
            type_ = kw
            break
    size = ""
    size_match = re.search(r"_\((.*?)\)", s)
    if size_match:
        size = size_match.group(1)
        model = s[: size_match.start()].replace("_", " ").strip()
        if not model:  # size leads the name, e.g. "_(300M)-..."
            model = s[size_match.end():].lstrip("_+").replace("_", " ").strip()
    else:
        model = s.replace("_", " ").strip()
        if type_ and model.lower().endswith(type_.lower()):
            model = model[: -len(type_)].strip()
    for key in _TYPE_MAPPING:  # per-model type remaps
        if key.lower() in model.lower():
            type_ = _TYPE_MAPPING[key].get(type_, type_)
            break
    return model, type_, size


def _row(nb: "MolabNotebook") -> str:
    """Return one ``| Model | Type | Notebook |`` row, like the AMD table."""
    model, type_, size = _model_type_size(nb.output.stem)
    model_cell = f"**{model}** **({size})**" if size else f"**{model}**"
    return f"| {model_cell} | {type_} | {_badge_markdown(nb)} |"


_INTRO = (
    "# 🍃 Open in molab\n"
    "\n"
    "Run any of these on [molab](https://molab.marimo.io), Marimo's "
    "hosted GPU notebooks. They're reactive: change a value in one "
    "cell, the cells below recompute on their own.\n"
    "\n"
)
_TABLE_HEADER = "| Model | Type | Notebook |\n| --- | --- | --- |\n"


def render_molab_readme_section(
    notebooks: Iterable["MolabNotebook"],
    popular_stems: "list[str] | None" = None,
) -> str:
    """Render the molab README section as a markdown string.

    Parameters
    ----------
    notebooks:
        Iterable of MolabNotebook entries (typically
        ``molab_manifest.MOLAB_NOTEBOOKS`` or
        ``molab_manifest.get_p0_notebooks()``).  Entries with ``skip=True``
        are silently excluded.  The iteration order determines row order in the
        rendered table — pass a sorted/stable list for deterministic output.
    popular_stems:
        Optional list of ``output.stem`` values for the "most popular"
        notebooks. When given, they render in a short top table (in
        ``popular_stems`` order) and the rest fold into a collapsible
        ``<details>`` block, like the AMD Notebooks section. When ``None`` or
        empty (default), all active notebooks render in a single flat table.

    Returns
    -------
    str
        Markdown string for the molab section body.  Does NOT include the
        ``<!-- MOLAB:START -->`` / ``<!-- MOLAB:END -->`` markers; the caller
        inserts those around this string.
    """
    active = [nb for nb in notebooks if not nb.skip]

    popular: list["MolabNotebook"] = []
    if popular_stems:
        by_stem = {nb.output.stem: nb for nb in active}
        seen: set[str] = set()
        for stem in popular_stems:
            nb = by_stem.get(stem)
            if nb is not None and stem not in seen:
                popular.append(nb)
                seen.add(stem)

    popular_set = {nb.output.stem for nb in popular}
    rest = [nb for nb in active if nb.output.stem not in popular_set]

    # Nothing to split: render one flat table (avoids an empty table/details).
    if not popular or not rest:
        body = "\n".join(_row(nb) for nb in active)
        return f"{_INTRO}{_TABLE_HEADER}{body}\n"

    popular_body = "\n".join(_row(nb) for nb in popular)
    rest_body = "\n".join(_row(nb) for nb in rest)
    return (
        f"{_INTRO}{_TABLE_HEADER}{popular_body}\n"
        "\n"
        "<details>\n"
        "  <summary>\n"
        "    Click for all our molab notebooks:\n"
        "  </summary>\n"
        "\n"
        f"{_TABLE_HEADER}{rest_body}\n"
        "</details>\n"
    )
