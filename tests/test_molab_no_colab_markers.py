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

"""Colab / Kaggle / platform-specific marker exclusion gate (AC4 — broad-catalog scope).

Parametrised over molab/*.py files that ACTUALLY EXIST ON DISK so that a
partial generation run (best-effort) does not red the whole suite.

Generated ``molab/*.py`` files must NOT contain:
  - ``COLAB_`` environment variable references
  - ``google.colab`` imports or usage
  - Kaggle accelerator setup text
  - ``%%capture`` magic
  - ``%%bash`` magic
  - Raw ``!pip`` install invocations (use PEP 723 metadata instead)
  - AMD / ROCm-specific installation text
    (``rocm``, ``amdgpu``, ``hip``, ``rocsolver``, ``rocblas``)

Each forbidden marker has its own parametrised test so failures are
individually identifiable in CI output.  No torch / GPU / marimo is
imported at module load time.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_manifest as mm  # noqa: E402

# ---------------------------------------------------------------------------
# Parametrise over files actually on disk — tolerates partial generation.
# ---------------------------------------------------------------------------

_MOLAB_DIR = REPO_ROOT / "molab"
_GENERATED_FILES: list[Path] = (
    sorted(_MOLAB_DIR.glob("*.py")) if _MOLAB_DIR.exists() else []
)

if not _GENERATED_FILES:
    # Fall back to manifest output paths so parametrize list is non-empty
    # (every test will skip individually until generation lands).
    _GENERATED_FILES = [nb.output for nb in mm.get_active_notebooks()]


# ---------------------------------------------------------------------------
# generation_status.json skip helper (FIX 3 / DA-03 / consolidated in P2 / ARCH-03)
# ---------------------------------------------------------------------------

from tests._molab_test_utils import (  # noqa: E402
    GENERATION_STATUS as _GENERATION_STATUS,
    skip_if_generation_failed as _skip_if_generation_failed,
)


# ---------------------------------------------------------------------------
# Forbidden marker definitions
# ---------------------------------------------------------------------------


class ForbiddenMarker(NamedTuple):
    """A pattern that must never appear in generated molab files."""

    label: str
    pattern: re.Pattern[str]
    reason: str


_FORBIDDEN: list[ForbiddenMarker] = [
    ForbiddenMarker(
        label="COLAB_ env var",
        pattern=re.compile(r"\bCOLAB_", re.MULTILINE),
        reason=(
            "COLAB_ is a Colab-specific environment variable. "
            "molab notebooks must not reference Colab platform env vars."
        ),
    ),
    ForbiddenMarker(
        label="google.colab import",
        pattern=re.compile(r"\bgoogle\.colab\b", re.MULTILINE),
        reason=(
            "google.colab is unavailable outside Google Colab. "
            "Remove all google.colab usages from molab notebooks."
        ),
    ),
    ForbiddenMarker(
        label="Kaggle accelerator",
        pattern=re.compile(
            r"kaggle[_\-]?accelerator|accelerator\s*=\s*['\"]gpu",
            re.IGNORECASE | re.MULTILINE,
        ),
        reason=(
            "Kaggle accelerator configuration must not appear in molab notebooks."
        ),
    ),
    ForbiddenMarker(
        label="%%capture magic",
        pattern=re.compile(r"^%%capture\b", re.MULTILINE),
        reason=(
            "%%capture is a Jupyter magic command and is invalid in marimo. "
            "Remove all %%capture cells from generated molab notebooks."
        ),
    ),
    ForbiddenMarker(
        label="%%bash magic",
        pattern=re.compile(r"^%%bash\b", re.MULTILINE),
        reason=(
            "%%bash is a Jupyter magic command and is invalid in marimo. "
            "Use subprocess or marimo-native shell support instead."
        ),
    ),
    ForbiddenMarker(
        label="raw !pip install",
        pattern=re.compile(r"^\s*!pip\s+install\b", re.MULTILINE),
        reason=(
            "Raw !pip install is a Jupyter-style shell invocation. "
            "molab notebooks must declare dependencies via PEP 723 inline "
            "metadata or a marimo setup cell (subprocess.run / mo.ui approach), "
            "not raw !pip."
        ),
    ),
    # NOTE: AMD/ROCm is checked separately in test_no_amd_rocm_in_code below
    # (not via this list) because it requires stripping mo.md() prose strings
    # before matching — prose documentation may legitimately list AMD as a
    # hardware option, but executable code (imports, pip install, hip.* calls)
    # must never reference ROCm/AMD-specific symbols.
]

# AMD/ROCm patterns that must not appear in EXECUTABLE CODE (outside mo.md strings).
_AMD_CODE_PATTERN = re.compile(
    r"\bamdgpu\b|\bhipcc?\b|\brocsolver\b|\brocblas\b"
    r"|\brocrand\b|\bmiopen\b|\bhip\.is_available\b"
    r"|import\s+rocm|pip\s+install\s+[^\n]*\brocm\b",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Parametrised test: each marker x each generated file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "py_file,marker",
    [
        pytest.param(py_file, m, id=f"{py_file.stem}::{m.label}")
        for py_file in _GENERATED_FILES
        for m in _FORBIDDEN
    ],
)
def test_no_forbidden_marker(py_file: Path, marker: ForbiddenMarker) -> None:
    """Generated molab file must not contain the specified forbidden marker (AC4)."""
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)

    text = py_file.read_text(encoding="utf-8")
    matches = list(marker.pattern.finditer(text))
    if not matches:
        return

    excerpts: list[str] = []
    for m in matches[:5]:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        excerpts.append(repr(text[start:end]))

    pytest.fail(
        f"FORBIDDEN MARKER: '{marker.label}' found in {py_file.name}.\n"
        f"Reason: {marker.reason}\n"
        f"Occurrences ({len(matches)} total, first {len(excerpts)} shown):\n"
        + "\n".join(f"  {e}" for e in excerpts)
    )


# ---------------------------------------------------------------------------
# Additional check: no Jupyter cell magic in any form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_no_amd_rocm_in_code(py_file: Path) -> None:
    """Generated molab file must not contain AMD/ROCm code symbols (AC4).

    Prose documentation inside ``mo.md(...)`` strings may legitimately list AMD
    as a hardware option and is excluded from this check.  Only executable code
    (imports, pip install calls, hip.* function calls, ROCm-specific library
    names) is forbidden.
    """
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)

    text = py_file.read_text(encoding="utf-8")

    # Strip mo.md(...) string content so prose hardware lists don't trigger.
    # Replace each mo.md(r"""...""") or mo.md("...") argument with a placeholder
    # of the same length so line numbers stay accurate for excerpts.
    _MO_MD_STRING_RE = re.compile(
        r'mo\.md\s*\(\s*(?:r?""".*?"""|r?\'\'\'.*?\'\'\'|r?"[^"]*"|r?\'[^\']*\')\s*\)',
        re.DOTALL,
    )
    scrubbed = _MO_MD_STRING_RE.sub(
        lambda m: "mo.md(" + " " * (len(m.group()) - 6) + ")",
        text,
    )

    matches = list(_AMD_CODE_PATTERN.finditer(scrubbed))
    if not matches:
        return

    excerpts: list[str] = []
    for m in matches[:5]:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)  # use original text for readability
        excerpts.append(repr(text[start:end]))

    pytest.fail(
        f"AMD/ROCm CODE FOUND: {py_file.name} contains AMD/ROCm-specific "
        "code symbols outside of documentation strings.\n"
        "AMD variants are a separate exclusion and use a different install stack. "
        "Remove AMD/ROCm imports, pip install calls, and library references.\n"
        f"Occurrences ({len(matches)} total, first {len(excerpts)} shown):\n"
        + "\n".join(f"  {e}" for e in excerpts)
    )


_FORBIDDEN_DOMAINS: tuple[str, ...] = (
    "colab.research.google.com",
    "colab-badge.svg",
)


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_no_colab_domains(py_file: Path) -> None:
    """Generated molab file must not reference any Colab-only domain.

    Catches ``colab.research.google.com`` and the ``colab-badge.svg`` image
    URL that survive the generator's rewrite logic.  The generator's
    ``_COLAB_URL_REWRITES`` table converts known Colab URLs to their molab
    / GitHub / molab-shield equivalents; any URL that slips through is a
    real leak (review finding #13).
    """
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)

    text = py_file.read_text(encoding="utf-8")
    found = sorted({d for d in _FORBIDDEN_DOMAINS if d in text})
    if found:
        excerpt_lines = [
            f"  line {i + 1}: {line.strip()[:160]}"
            for i, line in enumerate(text.splitlines())
            if any(d in line for d in found)
        ][:5]
        pytest.fail(
            f"FORBIDDEN DOMAIN: {py_file.name} contains {found!r}.\n"
            "Colab-specific URLs must be rewritten to molab / GitHub / "
            "molab-shield equivalents in scripts/molab_generate.py "
            "_COLAB_URL_REWRITES.\n"
            + "\n".join(excerpt_lines)
        )


@pytest.mark.parametrize("py_file", _GENERATED_FILES, ids=lambda p: p.stem)
def test_no_jupyter_cell_magic(py_file: Path) -> None:
    """Generated molab file must not contain any Jupyter cell magic (``%%``).

    Catches any ``%%<anything>`` at the start of a logical line.
    """
    if not py_file.exists():
        pytest.skip(f"{py_file.name} not yet generated.")
    _skip_if_generation_failed(py_file)

    text = py_file.read_text(encoding="utf-8")
    magic_re = re.compile(r"^%%[a-zA-Z]", re.MULTILINE)
    bad_lines = [
        (i + 1, line.rstrip())
        for i, line in enumerate(text.splitlines())
        if magic_re.match(line)
    ]
    if bad_lines:
        excerpt = "\n".join(
            f"  line {lineno}: {line}" for lineno, line in bad_lines[:10]
        )
        pytest.fail(
            f"JUPYTER MAGIC FOUND: {py_file.name} contains Jupyter cell magic.\n"
            "Remove all %% magic commands; marimo does not support them.\n"
            + excerpt
        )
