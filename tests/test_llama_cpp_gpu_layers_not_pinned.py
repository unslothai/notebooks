# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A notebook's llama.cpp run must not pin a GPU layer count.

On a CUDA machine Unsloth installs a CUDA llama.cpp build, which fits the layers to the free GPU memory and runs
the rest on the CPU. A pinned `-ngl 99` turns that off: the 15.5 GB Inkling 2 Mini Q4_K_M then fails to load on a
T4 (`cudaMalloc failed: out of memory`, `unable to create context`) where the unpinned run decodes at 27 tok/s.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# `"-ngl", "99"` in an argv list, or `-ngl 99` / `--n-gpu-layers=99` on a shell line.
_PINNED = re.compile(r"""(?:["'](?:-ngl|--n-gpu-layers|--gpu-layers)["']\s*,\s*["']?\d+)"""
                     r"""|(?:(?:^|\s)(?:-ngl|--n-gpu-layers|--gpu-layers)(?:\s+|=)\d+)""")


def _strip_comment(line):
    # A comment may say "-ngl"; only code counts. Good enough for these cells: no '#' inside the argv strings.
    return line.split("#", 1)[0]


def _code_sources():
    for folder in ("original_template", "nb", "kaggle"):
        for path in sorted((REPO_ROOT / folder).glob("*.ipynb")):
            cells = json.loads(path.read_text(encoding = "utf-8")).get("cells", [])
            for index, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", "")
                    yield f"{path.relative_to(REPO_ROOT)} cell {index}", "".join(source) if isinstance(source, list) else source
    for path in sorted((REPO_ROOT / "python_scripts").glob("*.py")):
        yield str(path.relative_to(REPO_ROOT)), path.read_text(encoding = "utf-8")


def _pinned_lines(source):
    return [line.strip() for line in source.splitlines() if _PINNED.search(_strip_comment(line))]


def test_detector_flags_the_pinned_forms():
    assert _pinned_lines('[llama, "-m", f, "-ngl", "99", "--temp", "1.0"]')
    assert _pinned_lines("!./llama.cpp/llama-cli -m model.gguf -ngl 99 -p hi")
    assert _pinned_lines("!llama-server -m m.gguf --n-gpu-layers=35")
    assert not _pinned_lines('[llama, "-m", f, "--temp", "1.0"],  # no -ngl: llama.cpp fits the layers to the GPU')
    assert not _pinned_lines('[llama, "-m", f, "-ngl", "auto"]')


def test_no_notebook_pins_llama_cpp_gpu_layers():
    sources = list(_code_sources())
    assert any("Inkling_2_Mini" in where for where, _ in sources), "the Inkling 2 Mini notebook is gone: update this test"
    offenders = {where: lines for where, source in sources if (lines := _pinned_lines(source))}
    if offenders:
        pytest.fail("llama.cpp runs that pin the GPU layer count (drop -ngl so llama.cpp fits them):\n"
                    + "\n".join(f"  {where}: {lines}" for where, lines in offenders.items()))
