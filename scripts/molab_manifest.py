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

"""molab_manifest — the molab catalog, built at import time.

Import like every other ``scripts/`` helper::

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import molab_manifest

What it does
============

Scans ``original_template/*.ipynb`` (and ``nb/*.ipynb`` for the few
notebooks that live only there) and produces one ``MolabNotebook``
record per included notebook.  Add a new template and it shows up in
the catalog the next time the generator runs.  The AMD, Kaggle, and
python_scripts generators work the same way.

Four families are filtered out:

- ``AMD-`` prefixed.  Different ROCm install stack, has its own generator.
- ``Kaggle-`` prefixed.  Environment-specific, relies on Kaggle's
  secret/credential model.
- ``hf_course``.  Hugging Face course duplicates, not Unsloth-authored.
- ``vllm``-pulling notebooks (content-based, not stem-prefix).  vllm's
  dep tree has many version-specific torch hard-pins and historically
  cp313-only abi3 wheels; uv on hosted molab silently aborts the
  resolution when vllm is constrained alongside our other floor pins.
  Until a workable wheel/resolver combination is found, GRPO notebooks
  that pull vllm are excluded from the molab catalog.  Source notebooks
  are unaffected — they continue to ship via the Colab/Kaggle/AMD
  generators.

No per-notebook skip list
=========================

Every entry has to generate.  When a source notebook hits an edge case
(duplicate imports, runtime clones, etc.) the fix goes into
``scripts/molab_generate.py``'s post-pass, not into a skip list here.
``test_committed_molab_file_exists`` hard-fails if an active entry has
no committed file.

Tiers
=====

``p0``       — the four original pilots: TinyLlama, Llama 3.1 8B,
               Qwen3 Embedding 0.6B, Embedding Gemma 300M.
``catalog``  — everything else.

Install text
============

``original_template/<name>.ipynb`` carries placeholder install cells.
The real pinned install commands are injected by the Colab/Kaggle
generator into ``nb/<name>.ipynb``.  ``molab_dependencies.py`` reads
from ``nb/`` for that reason — never from the template.

Runtime proof
=============

Every entry is ``runtime_proof = "pending"``.  No molab GPU smoke run
has happened yet; this stays "pending" until a human runs an end-to-end
test on a hosted GPU backend.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Repo root — resolved relative to THIS file (scripts/molab_manifest.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Exclusion reasons (test-visible curated record).
# Tests assert MOLAB_NOTEBOOKS contains no entry whose source stem starts
# with an excluded prefix.
# ---------------------------------------------------------------------------
EXCLUSION_REASONS: dict[str, str] = {
    "amd": (
        "AMD- prefixed notebooks use a different ROCm install stack and are "
        "generated separately; they are never fed through the molab generator."
    ),
    "kaggle": (
        "Kaggle- prefixed notebooks are environment-specific variants that rely "
        "on the Kaggle secret/credential model."
    ),
    "hf_course": (
        "Hugging Face course duplicate notebooks (named 'HuggingFace Course-*', "
        "with a space) are not Unsloth-authored content; their spaced filenames "
        "also produce broken molab links."
    ),
    "vllm": (
        "Notebooks whose install cells pull in vllm are excluded from the molab "
        "catalog: vllm's dep tree (torch==, xformers==, flashinfer-python==) "
        "interacts badly with uv on hosted molab's Python 3.13 runtime and "
        "silently aborts the resolution. Until a workable combination is found, "
        "GRPO notebooks that need vllm fast inference are not generated for "
        "molab. The same source notebooks still ship via Colab / Kaggle / AMD "
        "generators."
    ),
}

# ---------------------------------------------------------------------------
# Support tier and runtime-proof status types
# ---------------------------------------------------------------------------
SupportTier = Literal["p0", "catalog"]
RuntimeProofStatus = Literal["pending", "verified", "failed"]


# ---------------------------------------------------------------------------
# Core contract dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MolabNotebook:
    """Manifest entry for a single molab-generated notebook.

    Fields
    ------
    source : Path
        Path to the *original_template/<name>.ipynb* canonical template.
        The dependency planner (``molab_dependencies.py``) must resolve the
        actual install commands from the corresponding *nb/<name>.ipynb*
        post-injection file, NOT from this template.
    output : Path
        Path to the generated ``molab/<name>.py`` marimo application file.
    display_name : str
        Human-readable label used in README badge tables.
    tier : SupportTier
        ``"p0"`` for the original 4 pilot notebooks; ``"catalog"`` for all
        others.
    runtime_proof : RuntimeProofStatus
        Whether a live molab smoke test has been completed.
    wasm_compatible : bool
        ``False`` for all entries (torch/CUDA dependency present).
    skip : bool
        If ``True``, the generator skips this entry without removing it
        from the manifest.
    notes : str
        Free-form notes (e.g. known issues, skip reason).
    """

    source: Path
    output: Path
    display_name: str
    tier: SupportTier
    runtime_proof: RuntimeProofStatus = "pending"
    wasm_compatible: bool = False
    skip: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# P0 pilot set + display-name overrides
# ---------------------------------------------------------------------------
_P0_STEMS: frozenset[str] = frozenset(
    {
        "TinyLlama_(1.1B)-Alpaca",
        "Llama3.1_(8B)-Alpaca",
        "Qwen3_Embedding_(0_6B)",
        "EmbeddingGemma_(300M)",
    }
)

# Per-stem display-name overrides for the pilot tier.  Every other entry
# gets a derived name (underscores -> spaces, hyphens preserved as
# model/task separators).
_DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "TinyLlama_(1.1B)-Alpaca": "TinyLlama (1.1B) — Alpaca",
    "Llama3.1_(8B)-Alpaca": "Llama 3.1 (8B) — Alpaca",
    "Qwen3_Embedding_(0_6B)": "Qwen3 Embedding (0.6B)",
    "EmbeddingGemma_(300M)": "Embedding Gemma (300M)",
}


def _derive_display_name(stem: str) -> str:
    """Build a human-readable display name from a notebook filename stem."""
    if stem in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[stem]
    name = stem.replace("_", " ")
    name = re.sub(r"\s*-\s*", " - ", name)
    return name


def _is_excluded(stem: str) -> bool:
    """True for filename stems that match an excluded family.

    Excluded: ``AMD-`` prefix, ``Kaggle-`` prefix, and HF-course duplicates.
    The course notebooks are named ``HuggingFace Course-*`` (with a space) and
    also appear as ``hf_course`` / ``hf-course`` / ``huggingface_course``, so
    match any separator. The space variant ships broken molab links (spaces in
    the URL), so it must be caught here."""
    if stem.startswith("AMD-") or stem.startswith("Kaggle-"):
        return True
    if re.search(r"hf[\s_-]course|huggingface[\s_-]course", stem, re.IGNORECASE):
        return True
    return False


# A pip install line whose argv list contains ``vllm`` as a distribution
# token.  Matches both ``!pip install ... vllm ...`` and ``%pip install vllm``;
# also accepts ``vllm[extras]`` and ``vllm==X.Y.Z`` / ``vllm>=X.Y.Z``.  The
# negative lookbehind (?<![\w.-]) prevents matching tokens like ``my-vllm-fork``;
# the negative lookahead (?![\w.-]) prevents ``vllm-tools``.
_RE_INSTALL_LINE = re.compile(r"\s*[!%]?\s*(?:uv\s+)?pip\s+install\b")
_RE_VLLM_TOKEN = re.compile(r"(?<![\w.-])vllm(?:\[[^\]]+\])?(?:[<>=!~]|$|\s)")


def _install_cell_pulls_vllm(nb_path: Path) -> bool:
    """Return True iff any code cell in ``nb_path`` runs ``pip install`` with
    ``vllm`` in its argument list.

    Used for the ``vllm`` content-based exclusion family.  Reads the .ipynb
    JSON directly (no notebook_inventory import) so manifest scan stays
    cheap and free of cycles with ``molab_dependencies``.

    Returns False if the file is missing or unparseable — the rest of the
    pipeline will surface that separately via ``resolve_nb_source``.
    """
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        # Only look at lines that are pip install commands; vllm can show up
        # in markdown / prose without being installed.
        for raw_line in source.splitlines():
            if not _RE_INSTALL_LINE.match(raw_line):
                continue
            if _RE_VLLM_TOKEN.search(raw_line):
                return True
    return False


def _scan_manifest() -> list[MolabNotebook]:
    """Scan ``original_template/`` AND ``nb/`` and materialise the catalog.

    Runs at import time so any new notebook added to either directory
    automatically appears in the molab catalog the next time the generator
    runs — exactly like the AMD/Kaggle/python_scripts generators auto-
    discover new templates.

    Sources:
      * ``original_template/<name>.ipynb`` — canonical templates (the
        primary source).  These are the notebooks the Colab/Kaggle/AMD
        generators consume.
      * ``nb/<name>.ipynb`` — notebooks present in ``nb/`` but with no
        matching entry in ``original_template/``.  These are curriculum /
        experimental notebooks (Gemma4, Qwen3.5, OpenEnv, NeMo-Gym
        families) that are shipped to users via ``nb/`` directly.

    Filtered out: ``AMD-`` and ``Kaggle-`` prefixed variants in either
    directory (they have their own generators), and HF-course duplicates.

    EVERY non-excluded notebook is included with ``skip=False``.  The molab
    generator is expected to handle every catalog entry — when a source
    notebook hits an edge case (duplicate imports, runtime clones, etc.)
    the fix belongs in ``scripts/molab_generate.py``'s post-pass, not in a
    per-notebook skip list.  ``test_committed_molab_file_exists`` will
    hard-fail on any active entry that does not produce a ``molab/*.py``.

    Output is sorted alphabetically by ``output.stem`` so manifest order
    and downstream README badge ordering are stable across runs.
    """
    template_dir = _REPO_ROOT / "original_template"
    nb_dir = _REPO_ROOT / "nb"

    seen_stems: set[str] = set()
    out: list[MolabNotebook] = []

    def _make(source: Path, stem: str, tier: SupportTier) -> MolabNotebook:
        return MolabNotebook(
            source=source,
            output=_REPO_ROOT / "molab" / f"{stem}.py",
            display_name=_derive_display_name(stem),
            tier=tier,
        )

    def _content_excluded(stem: str) -> bool:
        """Content-based exclusion: check the post-injection nb/<stem>.ipynb
        for any pip install line that pulls in vllm.  The molab install
        harness silently aborts uv resolution when vllm is constrained
        alongside our other floor pins, so these are excluded until a
        workable wheel/resolver combination is found."""
        nb_file = nb_dir / f"{stem}.ipynb"
        if not nb_file.exists():
            return False
        return _install_cell_pulls_vllm(nb_file)

    # Pass 1 — canonical templates.
    for ipynb in sorted(template_dir.glob("*.ipynb")):
        stem = ipynb.stem
        if _is_excluded(stem) or stem in seen_stems:
            continue
        if _content_excluded(stem):
            continue
        seen_stems.add(stem)
        tier: SupportTier = "p0" if stem in _P0_STEMS else "catalog"
        out.append(_make(ipynb, stem, tier))

    # Pass 2 — nb-only notebooks (no original_template/ counterpart).
    if nb_dir.exists():
        for ipynb in sorted(nb_dir.glob("*.ipynb")):
            stem = ipynb.stem
            if _is_excluded(stem) or stem in seen_stems:
                continue
            if _install_cell_pulls_vllm(ipynb):
                continue
            seen_stems.add(stem)
            out.append(_make(ipynb, stem, "catalog"))

    out.sort(key=lambda n: n.output.stem)
    return out


# ---------------------------------------------------------------------------
# The catalog — materialised at import time from original_template/.
# ---------------------------------------------------------------------------
MOLAB_NOTEBOOKS: tuple[MolabNotebook, ...] = tuple(_scan_manifest())


# ---------------------------------------------------------------------------
# Convenience helpers.
# ---------------------------------------------------------------------------

def get_p0_notebooks() -> list[MolabNotebook]:
    """Return only the P0-tier, non-skipped pilot notebooks."""
    return [nb for nb in MOLAB_NOTEBOOKS if nb.tier == "p0" and not nb.skip]


def get_catalog_notebooks() -> list[MolabNotebook]:
    """Return only the catalog-tier, non-skipped notebooks."""
    return [nb for nb in MOLAB_NOTEBOOKS if nb.tier == "catalog" and not nb.skip]


def get_active_notebooks() -> list[MolabNotebook]:
    """Return all non-skipped notebooks regardless of tier."""
    return [nb for nb in MOLAB_NOTEBOOKS if not nb.skip]
