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

"""The Colab/Kaggle install cells pick a vLLM per GPU with

    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')

Leaving either side unpinned resolves to whatever is newest on PyPI. vLLM's
default PyPI wheel is the CUDA 13 build from 0.20.0 on ("CUDA 13.0 default",
v0.20.0 release notes), while Colab ships a CUDA 12 torch, so the wheel loads
against libcudart.so.13, Unsloth disables vLLM, and `fast_inference = True`
then dies with "Please install vLLM before enabling `fast_inference`!".

Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb shipped unpinned on the non-T4 branch and
broke on L4. This gate keeps every branch of every such line pinned.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


# `_vllm, _triton = (...) if is_t4 else (...)` / the older `get_vllm, get_triton` spelling.
_SELECTOR_RE = re.compile(
    r"^\s*(?:_vllm|get_vllm)\s*,\s*(?:_triton|get_triton)\s*=\s*(?P<rhs>.+)$",
    re.MULTILINE,
)
# A vLLM requirement inside that line: "vllm" or "vllm==0.15.1", quotes included.
_VLLM_SPEC_RE = re.compile(r"""['"](vllm(?:\[[^'"]*\])?[^'"]*)['"]""")


def _selector_lines():
    """(notebook, line, spec) for every vLLM requirement in a selector line."""
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            for match in _SELECTOR_RE.finditer(source):
                line = match.group(0).strip()
                for spec in _VLLM_SPEC_RE.findall(match.group("rhs")):
                    yield path, line, spec


_CASES = list(_selector_lines())


def test_selector_lines_are_actually_found():
    """Guards the parametrised test below against silently matching nothing."""
    assert len(_CASES) >= 80, (
        f"only {len(_CASES)} vLLM install specs found; the selector regex has "
        f"drifted away from the notebooks' install cells"
    )


@pytest.mark.parametrize(
    "path,line,spec",
    _CASES,
    ids = [f"{p.parent.name}/{p.stem}:{s}" for p, _l, s in _CASES],
)
def test_vllm_is_pinned_in_install_selector(path, line, spec):
    assert "==" in spec, (
        f"{path.relative_to(REPO_ROOT)} installs vLLM unpinned ({spec!r}) in:\n"
        f"  {line}\n"
        f"An unpinned vLLM resolves to the latest release, whose default PyPI "
        f"wheel is built for CUDA 13 and cannot load against Colab's CUDA 12 "
        f"torch. Pin the same version the generator uses."
    )
