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

"""Shared helpers for ``tests/test_molab_*.py`` (review P2 / ARCH-03).

Three test modules previously copy-pasted the same generation_status.json
loader and skip helper with subtly different bodies. Consolidated here so the
contract is single-source-of-truth.

Kept as a regular module (not ``conftest.py``) because these are plain
helpers, not pytest fixtures — ``conftest.py`` already hosts an unrelated
GPU-free import harness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STATUS_FILE = _REPO_ROOT / "molab" / "generation_status.json"


def _load_generation_status() -> dict[str, str]:
    """Return the generation status dict, or empty dict if file is absent.

    Values are either ``"ok"`` or a string beginning with ``"failed:"``.
    """
    if not _STATUS_FILE.exists():
        return {}
    try:
        return json.loads(_STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


GENERATION_STATUS: dict[str, str] = _load_generation_status()


def skip_if_generation_failed(py_file: Path) -> None:
    """Skip the calling test if the generator recorded a failure for ``py_file``.

    Does NOT skip when the status file is absent — that keeps tests hard-red
    until the generator actually runs, which is the correct pre-generation
    baseline signal.
    """
    status = GENERATION_STATUS.get(py_file.stem)
    if status is not None and status != "ok":
        reason = status if status.startswith("failed:") else f"failed: {status}"
        pytest.skip(
            f"{py_file.name}: generator recorded failure — {reason}. "
            "Fix the generator or update the status file once the notebook "
            "is convertible."
        )
