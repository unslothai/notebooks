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

"""A cell that pins torch must not run after a cell that imported it.

`Gemma4_(E2B)_GRPO` died at the first Unsloth cell with

    RuntimeError: operator torchvision::nms does not exist

Its first install cell imported torch to pick an xformers pin, which loads
Colab's libtorch 2.11.0 into the live kernel. The next cell installed
`vllm==0.15.1`, which pins `torch==2.9.1` / `torchvision==0.24.1`, so both were
downgraded ON DISK. The kernel kept the 2.11.0 libtorch it had already loaded,
torchvision 0.24.1's `_C.so` could not register its operators against it, and
every torchvision entry point failed from there.

Reinstalling torchvision does not repair this: on disk torch 2.9.1 and
torchvision 0.24.1 are already a matched pair, so the install is a no-op. The
split is live-kernel versus disk, and only ordering fixes it -- pin torch
before anything imports it. The 44 GRPO notebooks built on
`installation_grpo_content` were always fine because that block never imports
torch.

So the rule is positional, not version-specific: in any notebook, no code cell
may import torch ahead of a later cell that pins torch or a distribution that
hard-pins torch. That catches the next notebook to grow a torch import above
its vLLM cell, whatever the pins have moved to by then.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# `original_template/` is the source most notebooks are generated from, so a
# hazard reintroduced there would reach `nb/` on the next generator run.
NOTEBOOK_DIRS = ("nb", "kaggle", "original_template")

# torch itself, the companions that ship operators linked against libtorch, and
# vllm, which hard-pins all three (`vllm==0.15.1` -> `torch==2.9.1`,
# `torchvision==0.24.1`, `torchaudio==2.9.1`). Matched anywhere in the cell:
# the GRPO extra install block picks its pin into a variable
# (`_vllm = 'vllm==0.15.1'`) and only interpolates it into the shell line.
_TORCH_PIN = re.compile(r"\b(?:torch|torchvision|torchaudio|vllm)\s*==\s*\d[\w.+]*")

# `import torch` in statement position: at the start of a line, after a `;`, or
# as the body of a one-line `else:`. Anchoring this way keeps `# import torch`
# and `pip install torch` from counting, since neither sits at a statement
# boundary.
_TORCH_IMPORT = re.compile(
    r"(?:^|(?<=[;:]))[ \t]*(?:import[ \t]+torch\b|from[ \t]+torch[\s.])",
    re.MULTILINE,
)


def _code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        cells.append((index, source))
    return cells


def _notebooks_pinning_torch():
    """(label, path) for every notebook that pins torch somewhere."""
    found = []
    for directory in NOTEBOOK_DIRS:
        for path in sorted((REPO_ROOT / directory).glob("*.ipynb")):
            if any(_TORCH_PIN.search(source) for _index, source in _code_cells(path)):
                found.append((f"{directory}/{path.name}", path))
    return found


_PINNING = _notebooks_pinning_torch()


def test_some_notebook_pins_torch():
    """Guard against the parametrisation below silently collecting nothing."""
    assert _PINNING, (
        "no notebook pins torch, torchvision, torchaudio or vllm any more; "
        "this file is measuring nothing, so retire it or repoint it."
    )


@pytest.mark.parametrize(
    "label,path", _PINNING, ids=[label for label, _ in _PINNING]
)
def test_no_cell_imports_torch_before_a_cell_pins_it(label, path):
    importing = [index for index, source in _code_cells(path)
                 if _TORCH_IMPORT.search(source)]
    for index, source in _code_cells(path):
        pins = sorted(set(_TORCH_PIN.findall(source)))
        if not pins:
            continue
        earlier = [other for other in importing if other < index]
        assert not earlier, (
            f"{label}: cell {index} pins {pins}, but cell(s) {earlier} already "
            f"imported torch. On Colab the pin downgrades torch on disk while "
            f"the kernel keeps the libtorch it loaded, so torchvision's "
            f"operators no longer register: `RuntimeError: operator "
            f"torchvision::nms does not exist`. Reinstalling torchvision does "
            f"not help, because on disk the pair already matches. Keep the "
            f"install cells above the pin free of torch imports."
        )


@pytest.mark.parametrize("source", [
    "import torch",
    "import torch, re",
    "import torch; torch._dynamo.config.recompile_limit = 64;",
    "else: import torch; v = str(torch.__version__)",
    "    import torch\n",
    "from torch import nn",
    "from torch.nn import functional as F",
], ids=["bare", "multi", "semicolon", "one-line-else", "indented", "from",
        "from-submodule"])
def test_the_torch_imports_we_really_write_are_detected(source):
    assert _TORCH_IMPORT.search(source)


@pytest.mark.parametrize("source", [
    "# import torch",
    "!pip install torch",
    "!uv pip install -qqq torchvision",
    'xformers = "0.0.34"  # import torch to pick this',
    "import torchvision",
    "import torchao",
    'print("import torch")',
], ids=["comment", "pip", "uv-pip", "trailing-comment", "torchvision",
        "torchao", "string"])
def test_things_that_only_look_like_a_torch_import_are_not(source):
    assert not _TORCH_IMPORT.search(source)


@pytest.mark.parametrize("source", [
    "!pip install torch==2.9.1",
    "!uv pip install torchvision==0.24.1",
    "!pip install --force-reinstall torchaudio==2.9.1",
    "_vllm = 'vllm==0.15.1'",
    "!uv pip install -qqq --upgrade vllm==0.9.2",
    "!pip install torch == 2.9.1",
], ids=["torch", "torchvision", "torchaudio", "vllm-variable", "vllm-inline",
        "spaced"])
def test_the_pins_that_swap_libtorch_are_detected(source):
    assert _TORCH_PIN.search(source)


@pytest.mark.parametrize("source", [
    "!pip install torch torchvision torchaudio",
    '!pip install --no-deps --upgrade "torchao>=0.16.0"',
    "!pip install unsloth vllm",
    "!pip install transformers==5.5.0",
    "!pip install triton==3.2.0",
], ids=["unpinned", "torchao-floor", "vllm-unpinned", "transformers",
        "triton"])
def test_installs_that_do_not_swap_libtorch_are_not_flagged(source):
    """An unpinned install resolves against the session's torch, and a pin on
    an unrelated distribution never moves it."""
    assert not _TORCH_PIN.search(source)
