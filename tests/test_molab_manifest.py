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

"""Manifest integrity gate (AC1 — updated for broad-catalog scope).

Verifies that MOLAB_NOTEBOOKS is a materialized, explicit allowlist:
- Covers every ``original_template/*.ipynb`` EXCEPT the three excluded families:
  AMD- prefixed, Kaggle- prefixed, and HF-course duplicates.  GRPO/vLLM, A100/DGX,
  vision/audio, GGUF-export and all other "heavy" families are now IN scope.
- EXCLUSION_REASONS exists and covers at minimum the three truly excluded families.
- Every entry carries all required fields with sensible values.
- No entry carries ``wasm_compatible=True`` without an explicit proof in notes.
- Every entry's ``source`` template actually exists on disk.
- No entry whose source stem matches AMD-, Kaggle-, or HF-course appears in the
  manifest (unless explicitly ``skip=True`` with notes — which is still wrong here,
  those should not be in the list at all).
"""
from __future__ import annotations

import re
import sys
import typing
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_manifest as mm  # noqa: E402


# ---------------------------------------------------------------------------
# Truly excluded families (SCOPE CHANGE 2026-05-22: only these three)
# ---------------------------------------------------------------------------

# Patterns that flag a notebook stem as belonging to a TRULY excluded family.
# GRPO/vLLM/A100/DGX/vision/audio/GGUF etc. are now IN scope (best-effort).
_EXCLUDED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("amd", re.compile(r"^amd[-_]", re.IGNORECASE)),
    ("kaggle", re.compile(r"^kaggle[-_]", re.IGNORECASE)),
    # HF-course duplicates carry "HF_course" or "HuggingFace Course" in the
    # name; the real files use a space ("HuggingFace Course-*"), so allow any
    # separator between the two words.
    ("hf_course", re.compile(r"hf[\s_-]course|huggingface[\s_-]course", re.IGNORECASE)),
]

# Minimum set of family tags that MUST appear in EXCLUSION_REASONS.
_REQUIRED_EXCLUSION_KEYS = {"amd", "kaggle", "hf_course"}


# ---------------------------------------------------------------------------
# Module-level assertions (single-call tests)
# ---------------------------------------------------------------------------


def test_exclusion_reasons_is_non_empty() -> None:
    """EXCLUSION_REASONS must exist and be non-empty (AC1)."""
    assert mm.EXCLUSION_REASONS, (
        "molab_manifest.EXCLUSION_REASONS is empty. "
        "The manifest must record why each excluded family is absent."
    )


def test_exclusion_reasons_covers_required_families() -> None:
    """EXCLUSION_REASONS must cover the three truly excluded families (AC1).

    GRPO, vLLM, A100/DGX, vision/audio, and GGUF-export are now IN scope;
    they no longer need to appear in EXCLUSION_REASONS."""
    missing = _REQUIRED_EXCLUSION_KEYS - set(mm.EXCLUSION_REASONS)
    assert not missing, (
        f"EXCLUSION_REASONS is missing required family tags: {sorted(missing)}. "
        "Add entries for 'amd', 'kaggle', 'hf_course' to scripts/molab_manifest.py."
    )


def test_molab_notebooks_is_non_empty() -> None:
    """MOLAB_NOTEBOOKS must contain at least one entry."""
    assert mm.MOLAB_NOTEBOOKS, (
        "MOLAB_NOTEBOOKS is empty. The allowlist must have at least one entry."
    )


def test_tier_partition_is_consistent() -> None:
    """get_p0_notebooks() + get_catalog_notebooks() must partition get_active_notebooks() (AC1).

    The helpers must be exhaustive and non-overlapping: every active notebook
    is either p0 or catalog, and no notebook appears in both."""
    p0 = mm.get_p0_notebooks()
    catalog = mm.get_catalog_notebooks()
    active = mm.get_active_notebooks()

    p0_sources = {nb.source for nb in p0}
    catalog_sources = {nb.source for nb in catalog}
    active_sources = {nb.source for nb in active}

    overlap = p0_sources & catalog_sources
    assert not overlap, (
        f"TIER OVERLAP: {len(overlap)} notebook(s) appear in both get_p0_notebooks() "
        "and get_catalog_notebooks():\n"
        + "\n".join(f"  {p.name}" for p in sorted(overlap))
        + "\nEvery notebook must have exactly one tier."
    )

    union = p0_sources | catalog_sources
    missing_from_partition = active_sources - union
    assert not missing_from_partition, (
        f"TIER GAP: {len(missing_from_partition)} active notebook(s) are in neither "
        "get_p0_notebooks() nor get_catalog_notebooks():\n"
        + "\n".join(f"  {p.name}" for p in sorted(missing_from_partition))
        + "\nEvery active notebook must be reachable via a tier helper."
    )

    extra_in_partition = union - active_sources
    assert not extra_in_partition, (
        f"TIER PHANTOM: {len(extra_in_partition)} notebook(s) appear in tier helpers "
        "but not in get_active_notebooks():\n"
        + "\n".join(f"  {p.name}" for p in sorted(extra_in_partition))
    )


def test_molab_notebooks_covers_broad_catalog() -> None:
    """MOLAB_NOTEBOOKS should cover the broad original_template/ catalog.

    The manifest must include entries for substantially all original_template/*.ipynb
    notebooks except AMD-/Kaggle-/HF-course families.  This test is a sanity
    check: if the manifest has fewer entries than (template_count - excluded_count)
    it is likely still using the old 4-notebook pilot list."""
    template_dir = REPO_ROOT / "original_template"
    if not template_dir.exists():
        pytest.skip("original_template/ not found; cannot check catalog coverage.")

    all_templates = list(template_dir.glob("*.ipynb"))
    excluded_templates = [
        p for p in all_templates
        if any(pat.search(p.stem) for _, pat in _EXCLUDED_PATTERNS)
    ]
    expected_min = len(all_templates) - len(excluded_templates)

    # Count all manifest entries (including skip=True — those are still manifest
    # entries; they just don't get generated).  The manifest must cover every
    # non-excluded template: actual_count >= expected_min.  Using >= (not ==)
    # so that adding new templates to original_template/ before updating the
    # manifest does not produce a false pass — the failure direction is correct:
    # too few manifest entries is the regression we want to catch.
    manifest_stems = {nb.source.stem for nb in mm.MOLAB_NOTEBOOKS}
    actual_count = len(manifest_stems)

    assert actual_count >= expected_min, (
        f"MANIFEST COVERAGE: only {actual_count} entries found in MOLAB_NOTEBOOKS, "
        f"but original_template/ has {len(all_templates)} notebooks "
        f"({len(excluded_templates)} excluded => {expected_min} expected). "
        "Every non-excluded template must have a manifest entry (skip=True is "
        "acceptable for un-convertible notebooks; the entry must still exist). "
        "Update MOLAB_NOTEBOOKS in scripts/molab_manifest.py."
    )


def test_molab_notebooks_is_explicit_not_glob() -> None:
    """Every entry must have required fields set to non-empty/valid values (AC1).

    An explicit materialized list always has a display_name and correct suffixes;
    a glob-derived list typically does not."""
    for nb in mm.MOLAB_NOTEBOOKS:
        assert nb.display_name.strip(), (
            f"Entry {nb.source.name} has an empty display_name. "
            "Manifest must be explicit and curated."
        )
        assert nb.source.suffix in {".ipynb"}, (
            f"Entry {nb.source.name}: source must be an .ipynb template, "
            f"got suffix '{nb.source.suffix}'."
        )
        assert nb.output.suffix == ".py", (
            f"Entry {nb.source.name}: output must be a .py file, "
            f"got suffix '{nb.output.suffix}'."
        )


def test_molab_notebooks_have_valid_tier() -> None:
    """Every entry must have a tier value that matches the manifest's own
    SupportTier Literal definition (AC1).

    The valid set is derived from ``molab_manifest.SupportTier`` at runtime
    via ``typing.get_args`` so this test can never drift from the manifest."""
    # molab_manifest.SupportTier is Literal["p0", "catalog"] (or similar).
    # get_args returns the tuple of allowed string values.
    valid_tiers = set(typing.get_args(mm.SupportTier))
    if not valid_tiers:
        # Fallback: SupportTier may not be a plain Literal if the manifest
        # evolves; treat any non-empty string as acceptable and skip.
        pytest.skip(
            "Could not extract tier values from molab_manifest.SupportTier "
            "via typing.get_args; update this test if the type changed."
        )
    for nb in mm.MOLAB_NOTEBOOKS:
        assert nb.tier in valid_tiers, (
            f"Entry {nb.source.name} has unknown tier '{nb.tier}'. "
            f"Expected one of {sorted(valid_tiers)} "
            f"(derived from molab_manifest.SupportTier)."
        )


def test_molab_notebooks_have_valid_runtime_proof() -> None:
    """Every entry must have a recognised runtime_proof value."""
    valid_proofs = {"pending", "verified", "failed"}
    for nb in mm.MOLAB_NOTEBOOKS:
        assert nb.runtime_proof in valid_proofs, (
            f"Entry {nb.source.name} has unknown runtime_proof "
            f"'{nb.runtime_proof}'. Expected one of {valid_proofs}."
        )


def test_source_templates_exist_on_disk() -> None:
    """Every manifest entry's source template must be present (AC1)."""
    missing = [nb.source for nb in mm.MOLAB_NOTEBOOKS if not nb.source.exists()]
    assert not missing, (
        "Manifest entries reference templates that do not exist:\n"
        + "\n".join(f"  {p}" for p in missing)
        + "\nUpdate MOLAB_NOTEBOOKS in scripts/molab_manifest.py."
    )


def test_output_paths_are_under_molab_dir() -> None:
    """All output paths must be directly under <repo_root>/molab/ (AC1 / architecture)."""
    molab_root = REPO_ROOT / "molab"
    for nb in mm.MOLAB_NOTEBOOKS:
        assert nb.output.parent == molab_root, (
            f"Entry {nb.source.name}: output path {nb.output} is not directly "
            f"under {molab_root}. All generated marimo notebooks must live in molab/."
        )


def test_no_wasm_compatible_without_proof() -> None:
    """No entry may have wasm_compatible=True without a proof note (AC1)."""
    for nb in mm.MOLAB_NOTEBOOKS:
        if nb.wasm_compatible:
            assert nb.notes.strip(), (
                f"Entry {nb.source.name} has wasm_compatible=True but no notes "
                "explaining the proof. WASM-compatible entries must include a "
                "documented rationale (heavy deps like torch are typically absent)."
            )


def test_no_duplicate_output_paths() -> None:
    """No two manifest entries may share the same output path."""
    seen: dict[Path, str] = {}
    for nb in mm.MOLAB_NOTEBOOKS:
        if nb.output in seen:
            pytest.fail(
                f"DUPLICATE OUTPUT: {nb.output} is claimed by both "
                f"'{seen[nb.output]}' and '{nb.source.stem}'. "
                "Every manifest entry must have a unique output path."
            )
        seen[nb.output] = nb.source.stem


def test_no_duplicate_source_paths() -> None:
    """No two manifest entries may share the same source template path."""
    seen: dict[Path, str] = {}
    for nb in mm.MOLAB_NOTEBOOKS:
        if nb.source in seen:
            pytest.fail(
                f"DUPLICATE SOURCE: {nb.source} is listed by both "
                f"'{seen[nb.source]}' and '{nb.display_name}'. "
                "Remove the duplicate from MOLAB_NOTEBOOKS."
            )
        seen[nb.source] = nb.display_name


# ---------------------------------------------------------------------------
# Parametrised: entries with AMD-/Kaggle-/HF-course stems must not appear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nb", mm.MOLAB_NOTEBOOKS, ids=lambda nb: nb.source.stem)
def test_entry_not_in_truly_excluded_family(nb: mm.MolabNotebook) -> None:
    """Manifest entries whose source stem matches AMD-, Kaggle-, or HF-course
    must NOT appear at all.  These are never generated; skip/notes does not
    make them acceptable in this list — remove them entirely (AC1)."""
    stem = nb.source.stem
    for family_tag, pattern in _EXCLUDED_PATTERNS:
        if pattern.search(stem):
            pytest.fail(
                f"MANIFEST VIOLATION: {stem} matches the truly-excluded family "
                f"'{family_tag}' (pattern: {pattern.pattern}). "
                "AMD-, Kaggle-, and HF-course notebooks are never generated for molab. "
                "Remove this entry from MOLAB_NOTEBOOKS entirely. "
                f"Reason: EXCLUSION_REASONS['{family_tag}'] = "
                f"{mm.EXCLUSION_REASONS.get(family_tag, 'no entry found')!r}"
            )
