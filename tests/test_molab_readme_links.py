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

"""README molab-links gate (AC5 / R-readme — broad-catalog scope).

Two test axes:

AXIS 1 — renderer unit tests (call molab_readme.render_molab_readme_section
directly): verify the renderer's output invariants against the broadcast API
contract from readme-implementer (T3). These tests skip gracefully if
molab_readme.py is not yet on disk.

AXIS 2 — committed README tests (read README.md between markers): verify the
committed README.md molab section has the right structure once the generator
and renderer have run. These tests skip gracefully when the section is absent.

Both axes check:
  - Badge image URL is exactly ``https://marimo.io/molab-shield.svg``.
  - Link URL matches the canonical shape:
    ``https://molab.marimo.io/github/unslothai/notebooks/blob/main/molab/<file>.py``
  - No link contains ``/wasm``.
  - Wording contains "Open in molab" (live, not hedged).
  - Skipped (nb.skip=True) notebooks do NOT appear.
  - Every linked file target exists on disk under ``molab/``.
  - Section is non-empty once the renderer lands.
  - No <!-- MOLAB:START/END --> markers in the renderer return value.
  - Output is deterministic (same input → identical output).
"""
from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_manifest as mm  # noqa: E402


# ---------------------------------------------------------------------------
# Optional: import molab_readme renderer
# ---------------------------------------------------------------------------

#: Sentinel indicating molab_readme.py is not yet on disk.
_README_MOD_ABSENT = object()


def _import_molab_readme():
    """Import molab_readme from scripts/; return sentinel if not yet present."""
    readme_path = REPO_ROOT / "scripts" / "molab_readme.py"
    if not readme_path.exists():
        return _README_MOD_ABSENT
    spec = importlib.util.spec_from_file_location("molab_readme", str(readme_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise ImportError(
            f"molab_readme.py exists but failed to import: {exc}"
        ) from exc
    if not hasattr(mod, "render_molab_readme_section"):
        raise ImportError(
            "molab_readme.py does not expose 'render_molab_readme_section'. "
            "This function is required by the renderer contract (T3)."
        )
    return mod


_README_MOD = _import_molab_readme()

_README = REPO_ROOT / "README.md"

# Expected URL shapes.
_BADGE_IMG_URL = "https://marimo.io/molab-shield.svg"
_MOLAB_LINK_PREFIX = "https://molab.marimo.io/github/unslothai/notebooks/blob/main/molab/"

_MOLAB_URL_RE = re.compile(
    # Anchor on the ``.py`` suffix so the URL matches in full even when
    # the filename contains parens (e.g. ``TinyLlama_(1.1B)-Alpaca.py``).
    # The previous ``[^\s)\]\"']+`` form stopped at the first ``)`` and
    # truncated ~85/103 molab URLs before validation (review finding #10).
    r"https://molab\.marimo\.io/[^\s\"'`]+?\.py",
    re.IGNORECASE,
)
_SHIELD_URL_RE = re.compile(
    # Anchor on ``molab-shield.svg`` so the URL is captured in full
    # regardless of surrounding markdown parens.
    r"https?://[^\s\"'`]*?molab-shield\.svg",
    re.IGNORECASE,
)

_OPEN_IN_MOLAB_RE = re.compile(r"open\s+in\s+molab", re.IGNORECASE)
_MOLAB_MARKER_RE = re.compile(r"<!--\s*MOLAB:(START|END)\s*-->", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper: extract the molab section from README.md
# ---------------------------------------------------------------------------


def _read_molab_section() -> str | None:
    """Return the content between <!-- MOLAB:START --> and <!-- MOLAB:END -->.
    Returns None if either marker is absent or README.md does not exist."""
    if not _README.exists():
        return None
    text = _README.read_text(encoding="utf-8")
    start_match = re.search(r"<!--\s*MOLAB:START\s*-->", text, re.IGNORECASE)
    end_match = re.search(r"<!--\s*MOLAB:END\s*-->", text, re.IGNORECASE)
    if start_match is None or end_match is None:
        return None
    return text[start_match.end():end_match.start()]


# ---------------------------------------------------------------------------
# AXIS 1 — renderer unit tests (call render_molab_readme_section directly)
# ---------------------------------------------------------------------------


def _renderer_skip_if_absent() -> None:
    """Skip the calling test if molab_readme.py is not yet on disk."""
    if _README_MOD is _README_MOD_ABSENT:
        pytest.skip(
            "scripts/molab_readme.py not yet committed; "
            "renderer unit tests deferred until readme-implementer lands."
        )


def test_renderer_badge_image_url() -> None:
    """render_molab_readme_section output must use the exact badge image URL (AC5)."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    shield_urls = _SHIELD_URL_RE.findall(result)
    assert shield_urls, (
        "render_molab_readme_section output contains no molab-shield.svg URLs. "
        f"Expected: {_BADGE_IMG_URL}"
    )
    for url in shield_urls:
        assert url == _BADGE_IMG_URL, (
            f"Badge image URL mismatch in renderer output.\n"
            f"  Found:    {url}\n"
            f"  Expected: {_BADGE_IMG_URL}"
        )


def test_renderer_link_url_shape() -> None:
    """render_molab_readme_section output must use the canonical link URL shape (AC5)."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    molab_urls = _MOLAB_URL_RE.findall(result)
    assert molab_urls, (
        "render_molab_readme_section output contains no molab.marimo.io links. "
        f"Expected links starting with: {_MOLAB_LINK_PREFIX}"
    )
    bad = [u for u in molab_urls if not u.startswith(_MOLAB_LINK_PREFIX)]
    assert not bad, (
        "Renderer output contains molab links with wrong URL shape.\n"
        f"  Expected prefix: {_MOLAB_LINK_PREFIX}\n"
        "  Bad URLs:\n" + "\n".join(f"    {u}" for u in bad)
    )


def test_renderer_no_wasm_links() -> None:
    """render_molab_readme_section output must not contain /wasm links (AC5)."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    wasm_links = [u for u in _MOLAB_URL_RE.findall(result) if "/wasm" in u.lower()]
    assert not wasm_links, (
        "WASM LINK VIOLATION: renderer output contains /wasm links.\n"
        + "\n".join(f"  {u}" for u in wasm_links)
    )


def test_renderer_open_in_molab_wording() -> None:
    """render_molab_readme_section output must use 'Open in molab' wording (AC5)."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    assert _OPEN_IN_MOLAB_RE.search(result), (
        "render_molab_readme_section output does not contain 'Open in molab' wording.\n"
        "The renderer must use live 'Open in molab' badge text (not hedged wording). "
        "See plan Q5 / Lead decision."
    )


def test_renderer_no_molab_markers_in_output() -> None:
    """render_molab_readme_section return value must NOT include MOLAB:START/END markers (AC5).

    Those markers are inserted by the writer (generator-implementer); the
    renderer emits only the section body."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    assert not _MOLAB_MARKER_RE.search(result), (
        "render_molab_readme_section return value contains <!-- MOLAB:START/END --> "
        "markers. The renderer must return only the section BODY — markers are added "
        "by the writer that inserts the section into README.md."
    )


def test_renderer_skipped_notebooks_not_in_output() -> None:
    """Notebooks with skip=True must not appear in renderer output (AC5)."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    skipped = [nb for nb in mm.MOLAB_NOTEBOOKS if nb.skip]
    violations: list[str] = []
    for nb in skipped:
        if nb.output.name in result:
            violations.append(nb.output.name)
    assert not violations, (
        "render_molab_readme_section includes skipped notebooks:\n"
        + "\n".join(f"  {name}" for name in violations)
        + "\nNotebooks with skip=True must be omitted from the rendered section."
    )


def test_renderer_is_deterministic() -> None:
    """render_molab_readme_section must produce identical output on repeated calls (AC5)."""
    _renderer_skip_if_absent()
    result_a = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    result_b = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    assert result_a == result_b, (
        "render_molab_readme_section is non-deterministic: two consecutive calls "
        "with identical input produced different output. "
        "The renderer must be a pure function of its inputs."
    )


def test_renderer_link_targets_exist_for_generated_files() -> None:
    """When the renderer is called with only the GENERATED notebooks (those whose
    output file exists on disk), every URL in the output must link to a real file
    (AC5 / R-readme).

    The renderer takes an Iterable[MolabNotebook] — in production the generator
    passes only the notebooks it successfully generated.  This test exercises that
    contract: pass only the on-disk entries and assert all linked targets exist."""
    _renderer_skip_if_absent()
    molab_dir = REPO_ROOT / "molab"
    if not molab_dir.exists():
        pytest.skip(
            "molab/ directory not yet present; "
            "link-target check deferred until generator lands."
        )

    # Only pass notebooks that are already on disk — as the generator would.
    generated_notebooks = [
        nb for nb in mm.MOLAB_NOTEBOOKS
        if not nb.skip and nb.output.exists()
    ]
    if not generated_notebooks:
        pytest.skip(
            "No generated molab/*.py files found; "
            "link-target check deferred until generator runs."
        )

    result = _README_MOD.render_molab_readme_section(generated_notebooks)
    missing: list[tuple[str, Path]] = []
    for url in _MOLAB_URL_RE.findall(result):
        path_match = re.search(r"/molab/([^/\s?#]+\.py)", url)
        if path_match is None:
            continue
        filename = path_match.group(1)
        target = molab_dir / filename
        if not target.exists():
            missing.append((url, target))

    if missing:
        pytest.fail(
            "Renderer output (called with on-disk notebooks) links to files "
            "that do not exist:\n"
            + "\n".join(f"  URL: {url}\n  Missing: {target}" for url, target in missing)
            + "\nThe renderer must produce valid links for the notebooks it is given."
        )


def test_renderer_popular_split() -> None:
    """When ``popular_stems`` is provided, the renderer surfaces those notebooks
    in a top table and folds the rest into a collapsible ``<details>`` block,
    mirroring the AMD Notebooks section (R-readme).

    Invariants:
      - a single ``<details>`` collapsible is emitted,
      - every popular badge appears BEFORE the ``<details>`` opens,
      - every input notebook still appears exactly once (none dropped/duplicated),
      - passing ``popular_stems=None`` reproduces the flat single-table output.
    """
    _renderer_skip_if_absent()
    active = [nb for nb in mm.MOLAB_NOTEBOOKS if not nb.skip]
    if len(active) < 4:
        pytest.skip("Not enough active notebooks to exercise the popular split.")

    popular = active[:3]
    popular_stems = [nb.output.stem for nb in popular]
    result = _README_MOD.render_molab_readme_section(
        mm.MOLAB_NOTEBOOKS, popular_stems=popular_stems
    )

    # Exactly one collapsible, and it opens after the popular rows.
    assert result.count("<details>") == 1 and result.count("</details>") == 1, (
        "Expected exactly one <details> collapsible in the split rendering."
    )
    details_at = result.index("<details>")
    for nb in popular:
        url_frag = f"/molab/{nb.output.name}"
        first = result.find(url_frag)
        assert first != -1, f"Popular notebook {nb.output.name} missing from output."
        assert first < details_at, (
            f"Popular notebook {nb.output.name} must render before <details>, "
            "not inside the collapsible."
        )

    # Every active notebook appears exactly once across both tables.
    for nb in active:
        url_frag = f"/molab/{nb.output.name}"
        assert result.count(url_frag) == 1, (
            f"{nb.output.name} should appear exactly once; "
            f"found {result.count(url_frag)}."
        )

    # popular_stems=None is the legacy flat table (no collapsible).
    flat = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    assert "<details>" not in flat, (
        "Default (popular_stems=None) must render a single flat table."
    )


def test_renderer_model_type_columns() -> None:
    """The renderer emits AMD-style ``| Model | Type | Notebook |`` tables and
    derives the Type from the stem the same way the Colab/AMD tables do."""
    _renderer_skip_if_absent()
    result = _README_MOD.render_molab_readme_section(mm.MOLAB_NOTEBOOKS)
    assert "| Model | Type | Notebook |" in result, (
        "Expected the AMD-style three-column header."
    )
    if not hasattr(_README_MOD, "_model_type_size"):
        pytest.skip("renderer does not expose _model_type_size.")
    cases = {
        "Gemma4_(E2B)-Vision": ("Gemma4", "Vision", "E2B"),
        "gpt-oss-(20B)-Fine-tuning": ("gpt oss", "Fine Tuning", "20B"),
        "Qwen3_(14B)-Reasoning-Conversational": ("Qwen3", "Reasoning Conversational", "14B"),
        "Gemma3N_(4B)-Conversational": ("Gemma3N", "Multimodal", "4B"),  # TYPE_MAPPING remap
        "Unsloth_Studio": ("Unsloth Studio", "Chat UI", ""),
        # FIRST_MAPPING remaps Whisper.ipynb -> Whisper_(Large)-Fine-Tuning, so the
        # molab row matches the Colab/AMD tables (Fine Tuning, Large) not a blank.
        "Whisper": ("Whisper", "Fine Tuning", "Large"),
        # Verbose label is shortened and Type blanked (matches update_all_notebooks.py).
        "CodeForces-cot-Finetune_for_Reasoning_on_CodeForces": ("CodeForces CoT Reasoning", "", ""),
    }
    for stem, expected in cases.items():
        assert _README_MOD._model_type_size(stem) == expected, (
            f"_model_type_size({stem!r}) = {_README_MOD._model_type_size(stem)!r}, "
            f"expected {expected!r}."
        )


# ---------------------------------------------------------------------------
# AXIS 2 — committed README tests (read README.md between markers)
# ---------------------------------------------------------------------------


def test_readme_molab_section_exists() -> None:
    """README.md must contain <!-- MOLAB:START --> and <!-- MOLAB:END --> markers (AC5).

    Skipped if README.md does not yet exist or section not yet rendered."""
    if not _README.exists():
        pytest.skip("README.md not found.")
    section = _read_molab_section()
    if section is None:
        pytest.skip(
            "README.md has no MOLAB:START/END markers; "
            "section not yet rendered by readme-implementer / generator."
        )
    assert section.strip(), (
        "README.md molab section (between <!-- MOLAB:START --> and "
        "<!-- MOLAB:END -->) is empty. The renderer must emit at least one entry."
    )


def test_readme_molab_badge_image_url() -> None:
    """Badge image URL must be exactly ``https://marimo.io/molab-shield.svg`` (AC5)."""
    section = _read_molab_section()
    if section is None:
        pytest.skip("Molab section not yet rendered.")

    shield_urls = _SHIELD_URL_RE.findall(section)
    assert shield_urls, (
        "No molab-shield.svg image URLs found in the molab README section. "
        f"Expected: {_BADGE_IMG_URL}"
    )
    for url in shield_urls:
        assert url == _BADGE_IMG_URL, (
            f"Badge image URL mismatch.\n"
            f"  Found:    {url}\n"
            f"  Expected: {_BADGE_IMG_URL}\n"
            "Update the README renderer to use the official shield URL."
        )


def test_readme_molab_link_url_shape() -> None:
    """Each molab badge link must match the canonical URL shape (AC5)."""
    section = _read_molab_section()
    if section is None:
        pytest.skip("Molab section not yet rendered.")

    molab_urls = _MOLAB_URL_RE.findall(section)
    assert molab_urls, (
        "No molab.marimo.io links found in the molab README section. "
        f"Expected links starting with: {_MOLAB_LINK_PREFIX}"
    )
    bad: list[str] = []
    for url in molab_urls:
        if not url.startswith(_MOLAB_LINK_PREFIX):
            bad.append(url)
    if bad:
        pytest.fail(
            "Molab README links do not match the canonical URL shape.\n"
            f"  Expected prefix: {_MOLAB_LINK_PREFIX}\n"
            "  Bad URLs:\n"
            + "\n".join(f"    {u}" for u in bad)
        )


def test_readme_molab_no_wasm_links() -> None:
    """No molab link may contain ``/wasm`` (training notebooks are never WASM-compatible, AC5)."""
    section = _read_molab_section()
    if section is None:
        pytest.skip("Molab section not yet rendered.")

    wasm_links = [
        url for url in _MOLAB_URL_RE.findall(section)
        if "/wasm" in url.lower()
    ]
    if wasm_links:
        pytest.fail(
            "WASM LINK VIOLATION: molab README section contains /wasm links.\n"
            "Training notebooks must not use /wasm (requires a separate WASM allowlist).\n"
            "Remove /wasm from:\n"
            + "\n".join(f"  {u}" for u in wasm_links)
        )


def test_readme_molab_link_targets_exist() -> None:
    """Every molab badge URL must reference a file that exists under molab/ (AC5 / R-readme)."""
    section = _read_molab_section()
    if section is None:
        pytest.skip("Molab section not yet rendered.")

    molab_dir = REPO_ROOT / "molab"
    if not molab_dir.exists():
        pytest.skip(
            "molab/ directory does not yet exist; "
            "link-target check deferred until generator lands."
        )

    missing_targets: list[tuple[str, Path]] = []
    for url in _MOLAB_URL_RE.findall(section):
        path_match = re.search(r"/molab/([^/\s?#]+\.py)", url)
        if path_match is None:
            continue
        filename = path_match.group(1)
        target = molab_dir / filename
        if not target.exists():
            missing_targets.append((url, target))

    if missing_targets:
        pytest.fail(
            "README molab badges link to files that do not exist:\n"
            + "\n".join(
                f"  URL: {url}\n  Missing: {target}"
                for url, target in missing_targets
            )
            + "\nRun the generator to create the missing molab/*.py files, "
            "or remove the badge from the README."
        )


def test_readme_molab_generated_notebooks_have_badges() -> None:
    """Every non-skipped active manifest entry that has a generated file on disk
    must appear in the molab README section (AC5).

    Best-effort: we only check notebooks that have actually been generated.
    Notebooks whose molab/*.py does not exist yet are not checked (generation
    is best-effort for the broad catalog)."""
    section = _read_molab_section()
    if section is None:
        pytest.skip("Molab section not yet rendered.")

    molab_dir = REPO_ROOT / "molab"
    if not molab_dir.exists():
        pytest.skip("molab/ directory not yet present.")

    active_notebooks = mm.get_active_notebooks()
    missing_entries: list[str] = []

    for nb in active_notebooks:
        if not nb.output.exists():
            # Not yet generated — skip this entry.
            continue
        filename = nb.output.name
        expected_url_fragment = f"/molab/{filename}"
        if expected_url_fragment not in section:
            missing_entries.append(filename)

    if missing_entries:
        pytest.fail(
            "README molab section is missing badge entries for generated notebooks:\n"
            + "\n".join(f"  {f}" for f in missing_entries)
            + "\nThe README renderer must include all generated manifest entries."
        )
