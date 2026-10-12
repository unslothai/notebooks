# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
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
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""What a cell that launches `sglang.launch_server` owes the reader.

Two ways `import sgl_kernel` kills the server subprocess, both of which
surface only as `RuntimeError: Server process exited with code 1`.

1. The GPU. sgl-kernel dropped its compute_75 gencode flag in
   sgl-project/sglang#9207, so the sglang-kernel 0.4.5 that sglang 0.5.16
   pins ships SASS for sm_80, sm_89, sm_90, sm_90a, sm_100a, sm_103a and
   sm_120a and no PTX at all:

       $ cuobjdump --list-elf sgl_kernel/sm100/common_ops.abi3.so | \\
             grep -oE 'sm_[0-9]+[a-z]*' | sort -u
       sm_100a sm_103a sm_120a sm_80 sm_89 sm_90 sm_90a
       $ cuobjdump --list-ptx sgl_kernel/sm100/common_ops.abi3.so
       (nothing)

   A cubin runs on a device of the same major generation with a minor at
   least as high, so sm_80 covers 8.0 through 8.9 and nothing covers
   Turing. Check the compute capability in the cell, where the message can
   name the cause and the fix.

2. libnvrtc. That same library links `libnvrtc.so.13` and carries no
   RUNPATH. torch preloads one libnvrtc by absolute path
   (`_preload_cuda_deps`), globbing `nvidia/cuda_nvrtc/lib` before
   `nvidia/cu13/lib`, so a session that still has the CUDA 12
   `nvidia-cuda-nvrtc-cu12` wheel installed beside the CUDA 13 one -- Colab,
   after `pip install sglang[all]` swaps torch and leaves the old torch's
   NVIDIA wheels behind -- preloads `libnvrtc.so.12` and never `.so.13`.
   Reproduced against sglang-kernel 0.4.5 + torch 2.11.0+cu130:

       $ pip install nvidia-cuda-nvrtc-cu12          # the Colab leftover
       $ ld.so --inhibit-cache python -c 'import sgl_kernel'
       ImportError: libnvrtc.so.13: cannot open shared object file
       $ LD_LIBRARY_PATH=.../nvidia/cu13/lib \\
             ld.so --inhibit-cache python -c 'import sgl_kernel'
       (clean)

   So the launch has to hand the child an environment pointing at the
   libnvrtc that matches its torch.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIRS = ("original_template", "nb", "kaggle")

# The floor sglang-kernel's wheels are built for.
MIN_CAPABILITY = (8, 0)


def _launch_cells():
    """Every (id, source) code cell that spawns `sglang.launch_server`."""
    found = []
    for directory in NOTEBOOK_DIRS:
        for path in sorted((REPO_ROOT / directory).glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if "sglang.launch_server" not in source:
                    continue
                found.append((f"{directory}/{path.name}:cell{index}", source))
    return found


_CELLS = _launch_cells()


def _linenos(tree, predicate):
    return sorted(node.lineno for node in ast.walk(tree) if predicate(node))


def _capability_linenos(tree):
    """Lines reading `torch.cuda.get_device_capability`."""
    return _linenos(
        tree,
        lambda node: isinstance(node, ast.Attribute)
        and node.attr == "get_device_capability",
    )


def _server_launch_calls(tree):
    """Every `subprocess.Popen(...)` call node."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
    ]


def _server_launch_linenos(tree):
    """Lines spawning the server subprocess."""
    return sorted(node.lineno for node in _server_launch_calls(tree))


def _capability_floor_linenos(tree):
    """Lines comparing something against the `(8, 0)` capability floor."""
    def is_floor(node):
        if not isinstance(node, ast.Compare):
            return False
        for operand in [node.left, *node.comparators]:
            if not isinstance(operand, ast.Tuple):
                continue
            values = [
                element.value
                for element in operand.elts
                if isinstance(element, ast.Constant)
            ]
            if tuple(values) == MIN_CAPABILITY:
                return True
        return False

    return _linenos(tree, is_floor)


def _raise_linenos(tree):
    return _linenos(tree, lambda node: isinstance(node, ast.Raise))


def test_some_notebook_launches_sglang():
    """Guard against the parametrisation below silently collecting nothing."""
    assert _CELLS, (
        "no notebook launches sglang.launch_server any more; this file is "
        "measuring nothing, so retire it or repoint it."
    )


@pytest.mark.parametrize("cell_id,source", _CELLS, ids=[c for c, _ in _CELLS])
def test_launch_cell_rejects_unsupported_gpu(cell_id, source):
    tree = ast.parse(source)

    launches = _server_launch_linenos(tree)
    assert launches, (
        f"{cell_id} names sglang.launch_server but never spawns it; this "
        f"test can no longer tell the guard from the launch."
    )
    first_launch = launches[0]

    capability = [line for line in _capability_linenos(tree) if line < first_launch]
    assert capability, (
        f"{cell_id} spawns sglang.launch_server without reading "
        f"torch.cuda.get_device_capability() first. sglang-kernel ships no "
        f"kernels below compute capability {MIN_CAPABILITY[0]}."
        f"{MIN_CAPABILITY[1]}, so on a T4 the server exits inside "
        f"`import sgl_kernel` and the cell reports only 'Server process "
        f"exited with code 1'."
    )

    floor = [line for line in _capability_floor_linenos(tree) if line < first_launch]
    assert floor, (
        f"{cell_id} reads the compute capability before launching but never "
        f"compares it against {MIN_CAPABILITY}, so an sm_75 session still "
        f"reaches the server launch."
    )

    raises = [line for line in _raise_linenos(tree) if line < first_launch]
    assert raises, (
        f"{cell_id} checks the compute capability but raises nothing before "
        f"the launch, so an unsupported GPU still dies in the subprocess "
        f"instead of in this cell."
    )


@pytest.mark.parametrize("cell_id,source", _CELLS, ids=[c for c, _ in _CELLS])
def test_launch_cell_points_the_server_at_a_matching_nvrtc(cell_id, source):
    tree = ast.parse(source)

    launches = _server_launch_calls(tree)
    assert launches, (
        f"{cell_id} names sglang.launch_server but never spawns it; this "
        f"test can no longer tell what environment the server gets."
    )

    for call in launches:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "env" in keywords, (
            f"{cell_id} spawns the server without an explicit `env`, so it "
            f"inherits a LD_LIBRARY_PATH with no libnvrtc directory on it. "
            f"When a stale nvidia-cuda-nvrtc-cu12 wheel shadows the CUDA 13 "
            f"one -- Colab, after installing sglang -- torch preloads "
            f"libnvrtc.so.12, sgl_kernel needs libnvrtc.so.13, and the "
            f"server exits with an ImportError this cell reports only as "
            f"'Server process exited with code 1'."
        )

    assert "LD_LIBRARY_PATH" in source, (
        f"{cell_id} passes an `env` to the server but never sets "
        f"LD_LIBRARY_PATH, which is the only thing that makes an unloaded "
        f"libnvrtc.so.N reachable from sgl_kernel's RUNPATH-less "
        f"common_ops library."
    )
    assert "libnvrtc" in source, (
        f"{cell_id} sets LD_LIBRARY_PATH without resolving a libnvrtc "
        f"directory, so it cannot know the path it is adding holds the "
        f"libnvrtc this torch needs."
    )
