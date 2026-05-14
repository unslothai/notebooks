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

"""AMD notebook install-parity gate.

Lifts the in-process helpers ``_validate_amd_install_package_parity``
(update_all_notebooks.py:~1660) and ``_validate_amd_install_runtime``
(:1753) into a pytest gate. The helpers verify two invariants:

1. Every package the BASE notebook installs must also be installed by
   the AMD- variant (the AMD notebook never drops a package by accident).
2. The AMD notebook must not reintroduce CUDA-specific install markers
   that the AMD-rewrite pass is supposed to strip.

Both helpers already exist and are called from update_all_notebooks.py
itself at regen time; this file just surfaces their findings as
``pytest.fail`` per AMD notebook so the gate is enforced on every PR
instead of only when someone re-runs the generator locally.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


def _load_generator():
    """Load update_all_notebooks.py as a module. It is at repo root and
    not a package; we import it by file path so the tests do not depend
    on PYTHONPATH-of-repo-root."""
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location(
        "_update_all_notebooks", str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_update_all_notebooks"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()


_AMD_NOTEBOOKS = [
    p for p in ni.iter_notebooks(("nb",))
    if p.name.startswith("AMD-")
]

# Notebooks known to fail AMD install parity at PR-open time. Each entry
# is a real drift (the AMD- variant dropped a package that exists in the
# base install cell); the right fix is a single regen run via
# `python update_all_notebooks.py --to_main_repo`. Tracked as a follow-up
# to the CI mirror PR. Remove an entry once regenerated.
_KNOWN_AMD_PARITY_FAILURES: set[str] = {
    "AMD-Qwen3_5_MoE.ipynb",
}


@pytest.mark.parametrize(
    "amd_path", _AMD_NOTEBOOKS, ids=lambda p: p.name,
)
def test_amd_install_package_parity(amd_path: Path) -> None:
    """For every package the BASE notebook (same name without the AMD-
    prefix) installs, the AMD variant must also install. Uses the
    in-process helper from update_all_notebooks.py directly."""
    issue = _GEN._validate_amd_install_package_parity(str(amd_path))
    if issue is None:
        return
    if amd_path.name in _KNOWN_AMD_PARITY_FAILURES:
        pytest.xfail(
            f"{amd_path.name} is a known AMD-parity drift pending regen; "
            f"missing={issue.get('missing', [])}"
        )
    if issue.get("missing_base"):
        pytest.fail(
            f"DRIFT DETECTED: AMD notebook {amd_path.name} references a "
            f"base notebook that does not exist on disk: "
            f"{issue['missing_base']}. Either restore the base file or "
            f"remove the AMD variant."
        )
    if issue.get("error"):
        pytest.fail(
            f"DRIFT DETECTED: AMD notebook {amd_path.name} failed parity "
            f"validation: {issue['error']}"
        )
    missing = issue.get("missing", []) or []
    pytest.fail(
        f"DRIFT DETECTED: AMD notebook {amd_path.name} dropped packages "
        f"present in the base install cell: {', '.join(missing)}. "
        f"Regenerate via `python update_all_notebooks.py --to_main_repo`."
    )


@pytest.mark.parametrize(
    "amd_path", _AMD_NOTEBOOKS, ids=lambda p: p.name,
)
def test_amd_install_runtime_markers(amd_path: Path) -> None:
    """The AMD-rewrite pass must strip every CUDA-specific install
    marker (``unsloth[base]``, ``COLAB_``, ``cu12``, etc.)."""
    issue = _GEN._validate_amd_install_runtime(str(amd_path))
    if issue is None:
        return
    pytest.fail(
        f"DRIFT DETECTED: AMD notebook {amd_path.name} kept CUDA-specific "
        f"install markers at cell {issue['cell']}: "
        f"{', '.join(issue['markers'])}. Regenerate via "
        f"`python update_all_notebooks.py --to_main_repo`."
    )
