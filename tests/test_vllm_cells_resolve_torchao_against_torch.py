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

"""An install cell that pins vLLM does not get to install torchao unbounded.

A cell that installs a pinned vLLM has, by the next line, replaced torch with
whatever that vLLM requires, and both pins in these notebooks go backwards from
the machine's own torch:

    vllm==0.9.2  (the T4 branch)      -> torch 2.7.0
    vllm==0.15.1 (everything else)    -> torch 2.9.1

torchao 0.18.0, published 2026-08-03, begins with

    from torch.nn.functional import ScalingType, scaled_grouped_mm

and neither name exists before torch 2.10. So the

    !uv pip install -qqq --no-deps --upgrade "torchao>=0.16.0"

these cells used to end with started resolving to 0.18.0 the day it was
published, and the next `import transformers` stopped in modeling_utils.py,
which imports torchao whenever torchao is installed:

    ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'

`--no-deps` is precisely the flag that stops the resolver from noticing, which
is why the version has to be chosen in the cell instead.

Meta-Synthetic-Data-Llama3.1 (8B) is where this was caught. It launches vLLM
as a subprocess, so the traceback went to the server's own log and the notebook
could only say `RuntimeError: vllm exited with 1` over a TensorFlow banner.

Two gates: no cell may carry the unbounded spec next to a vLLM pin, and the
resolver those cells now carry is executed here against several torch versions
so its arithmetic is checked rather than assumed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIRS = (REPO_ROOT / "nb", REPO_ROOT / "kaggle")

# First torch carrying torch.nn.functional.ScalingType, and the first torchao
# that imports it unconditionally. Kept beside update_all_notebooks.py's
# TORCHAO_SCALING_TYPE_TORCH / TORCHAO_SCALING_TYPE_RELEASE.
SCALING_TYPE_TORCH = (2, 10)
SCALING_TYPE_TORCHAO = "0.18.0"

_VLLM_PIN = re.compile(r"vllm\s*==\s*\d")
# An install line naming torchao with no upper bound.
_UNBOUNDED_TORCHAO = re.compile(
    r"pip\s+install\b[^\n]*[\"']torchao>=[0-9][^\"',<]*[\"']")
# The generated cells call it `_torchao`; the hand-maintained GRPO notebook
# names its install variables `get_*`, so match the binding rather than one
# spelling of it.
_BINDS_TORCHAO = re.compile(
    r"^\s*(\w*torchao)\s*=\s*[\"']torchao>=", re.MULTILINE | re.IGNORECASE)


def _code_cells(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"]


def _notebooks():
    found = []
    for directory in NB_DIRS:
        found += sorted(directory.glob("*.ipynb"))
    return found


def _vllm_cells(path: Path):
    return [cell for cell in _code_cells(path) if _VLLM_PIN.search(cell)]


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_cell_that_pins_vllm_does_not_install_torchao_unbounded(path):
    for cell in _vllm_cells(path):
        if "torchao" not in cell:
            continue
        assert not _UNBOUNDED_TORCHAO.search(cell), (
            f"{path.relative_to(REPO_ROOT)} pins vLLM and then installs "
            f"torchao with no upper bound. vLLM's pin takes torch below "
            f"{'.'.join(str(p) for p in SCALING_TYPE_TORCH)}, so this resolves "
            f"to torchao {SCALING_TYPE_TORCHAO} and the next import of "
            "transformers fails on ScalingType."
        )
        bound = _BINDS_TORCHAO.search(cell)
        assert bound, (
            f"{path.relative_to(REPO_ROOT)} installs torchao beside a vLLM pin "
            "without resolving it against the installed torch first."
        )
        assert re.search(
            r"pip\s+install\b[^\n]*\{" + re.escape(bound.group(1)) + r"\}",
            cell), (
            f"{path.relative_to(REPO_ROOT)} resolves `{bound.group(1)}` but "
            "does not install it."
        )


def test_some_notebooks_really_pin_vllm():
    """A regex that stopped matching would leave every case above vacuous."""
    with_pin = [p for p in _notebooks() if _vllm_cells(p)]
    assert len(with_pin) >= 40, f"only {len(with_pin)} notebooks pin vLLM"


# --------------------------------------------------------------------------
# Run the resolver the notebooks actually ship, against fabricated torches.
# --------------------------------------------------------------------------

def _shipped_resolver() -> str:
    """The `_torchao = ...` block lifted out of a real notebook cell.

    Taken from the tree rather than rebuilt here, so this executes what ships.
    `!` lines are dropped; everything else in the block is plain Python.
    """
    for path in _notebooks():
        for cell in _vllm_cells(path):
            if not _BINDS_TORCHAO.search(cell):
                continue
            lines = cell.splitlines()
            start = next(i for i, line in enumerate(lines)
                         if line.strip().startswith("try:")
                         and "importlib.metadata" in lines[i + 1])
            end = next(i for i, line in enumerate(lines)
                       if _BINDS_TORCHAO.match(line))
            block = lines[start:end + 1]
            indent = len(block[0]) - len(block[0].lstrip())
            return "\n".join(line[indent:] for line in block)
    raise AssertionError("no notebook carries the torchao resolver")


def _resolve(torch_version):
    """`_torchao` as the shipped block computes it for this installed torch.

    `torch_version = None` stands for a torch whose metadata cannot be read.
    """
    import importlib.metadata as real_md

    class _Meta:
        PackageNotFoundError = real_md.PackageNotFoundError

        @staticmethod
        def version(name):
            assert name == "torch"
            if torch_version is None:
                raise real_md.PackageNotFoundError(name)
            return torch_version

    source = _shipped_resolver()
    namespace = {"__builtins__": __builtins__, "_FAKE_MD": _Meta}
    exec(compile(
        source.replace("import importlib.metadata as _md", "_md = _FAKE_MD"),
        "<resolver>", "exec"), namespace)
    name = _BINDS_TORCHAO.search(source).group(1)
    return namespace[name]


CAPPED = f"torchao>=0.16.0,<{SCALING_TYPE_TORCHAO}"
OPEN = "torchao>=0.16.0"


@pytest.mark.parametrize("torch_version", [
    "2.7.0+cu126",      # what vllm==0.9.2 installs, the T4 branch
    "2.9.1+cu128",      # what vllm==0.15.1 installs, everywhere else
    "2.6.0",
    "2.8.0+cu126",
])
def test_a_torch_without_scaling_type_gets_the_capped_torchao(torch_version):
    assert _resolve(torch_version) == CAPPED


@pytest.mark.parametrize("torch_version", [
    "2.10.0+cu128",     # the release ScalingType arrived in
    "2.11.0+cu128",     # Colab's own torch at the time of writing
    "3.0.0",
])
def test_a_torch_with_scaling_type_is_left_unbounded(torch_version):
    assert _resolve(torch_version) == OPEN


def test_an_unreadable_torch_fails_safe():
    """The capped spec imports on every torch these notebooks can meet, so the
    unknown case has to land there and not on 0.18.0."""
    assert _resolve(None) == CAPPED


@pytest.mark.parametrize("torch_version", ["", "unknown", "2", "2.x.0"])
def test_a_torch_version_that_does_not_parse_fails_safe(torch_version):
    assert _resolve(torch_version) == CAPPED


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_resolved_spec_survives_the_shell(path):
    """`>` and `<` are shell redirects.

    IPython substitutes `{name}` into the command line before the shell sees
    it, so an unquoted spec makes the cell create a file called `0.18.0`
    instead of installing anything, and torchao is left wherever it was.
    """
    import shlex

    checked = 0
    for cell in _vllm_cells(path):
        bound = _BINDS_TORCHAO.search(cell)
        if not bound:
            continue
        placeholder = "{" + bound.group(1) + "}"
        for line in cell.splitlines():
            if placeholder not in line or "install" not in line:
                continue
            command = line.strip().lstrip("!").replace(placeholder, CAPPED)
            assert shlex.split(command)[-1] == CAPPED, (
                f"unquoted torchao spec in {path.name}: {line.strip()}")
            checked += 1
    if not any(_BINDS_TORCHAO.search(c) for c in _vllm_cells(path)):
        return
    assert checked, f"{path.name} resolves a torchao spec but never installs it"
