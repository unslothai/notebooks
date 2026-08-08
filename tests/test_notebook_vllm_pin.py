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


# A quoted pip requirement for vLLM: 'vllm', "vllm==0.15.1", 'vllm[audio]==0.15.1'.
# Strict on purpose. "everything up to the closing quote" also swallows
# "vllm.entrypoints.openai.api_server", "vllm_requirements.txt" and
# f"vllm server is running.", which are not requirements and cannot be pinned.
_VLLM_SPEC_RE = re.compile(
    r"""['"](vllm(?:\[[^\[\]'"]*\])?\s*(?:[=<>!~]=?[^'"]*)?)['"]"""
)


def _selector_lines():
    """(notebook, line, spec) for every quoted vLLM requirement in a code cell.

    Keyed on the requirement itself, not on the names the install cell unpacks
    it into. Matching `_vllm, _triton = ...` / `get_vllm, get_triton = ...`
    literally meant a rename to any other valid spelling dropped those
    notebooks from the gate silently, while the count guard below still
    passed, so an unpinned vLLM could reach CI green.
    """
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            for line in source.splitlines():
                for spec in _VLLM_SPEC_RE.findall(line):
                    yield path, line.strip(), spec


_CASES = list(_selector_lines())


# Pinning the selector only helps if the install command still reads it. The
# Colab branch interpolates it -- `!uv pip install -qqq --upgrade {_vllm} ...`
# -- and swapping that `{_vllm}` for a bare `vllm` leaves every case above
# intact and green while Colab resolves the latest wheel again. So each
# assignment is checked against the commands in its own cell.
#
# Only the interpolation is required, not the absence of a bare `vllm`
# elsewhere: the non-Colab branch of these cells is `!pip install unsloth
# vllm` on purpose, unpinned for people on their own CUDA, in 83 cells.
_ASSIGNMENT_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*,\s*[A-Za-z_]\w*\s*=")


def _install_commands(source):
    """The `!...pip install...` lines of a cell, backslash continuations
    joined, because the command that consumes the selector is often wrapped."""
    commands, pending = [], ""
    for line in source.splitlines():
        pending = f"{pending} {line.strip()}" if pending else line
        if line.rstrip().endswith("\\"):
            pending = pending.rstrip()[:-1]
            continue
        if pending.lstrip().startswith("!") and "pip install" in pending:
            commands.append(pending)
        pending = ""
    return commands


def _selector_bindings():
    """(notebook, line, name, commands) per vLLM selector assignment."""
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            commands = _install_commands(source)
            for line in source.splitlines():
                if not _VLLM_SPEC_RE.search(line):
                    continue
                match = _ASSIGNMENT_RE.match(line)
                if match is None:
                    continue
                yield path, line.strip(), match.group("name"), commands


_BINDINGS = list(_selector_bindings())


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


def test_every_selector_assignment_is_bound_to_a_command():
    """Guards the parametrised test below against silently matching nothing."""
    assert len(_BINDINGS) >= 40, (
        f"only {len(_BINDINGS)} vLLM selector assignments found; the assignment "
        f"pattern has drifted away from the notebooks' install cells"
    )


@pytest.mark.parametrize(
    "path,line,name,commands",
    _BINDINGS,
    ids = [f"{p.parent.name}/{p.stem}:{n}" for p, _l, n, _c in _BINDINGS],
)
def test_the_install_command_still_reads_the_pinned_selector(path, line, name, commands):
    assert any(f"{{{name}}}" in command for command in commands), (
        f"{path.relative_to(REPO_ROOT)} pins vLLM in:\n"
        f"  {line}\n"
        f"but no pip install command in that cell interpolates {{{name}}}, so "
        f"the pin installs nothing. Commands found: {commands or 'none'}"
    )


def test_a_command_that_stopped_reading_the_selector_is_caught():
    """Discriminating case: the pin is still there, the command ignores it."""
    cell = (
        "    _vllm, _triton = ('vllm==0.9.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')\n"
        "    !uv pip install -qqq --upgrade vllm {_numpy} unsloth\n"
    )
    commands = _install_commands(cell)
    assert commands, cell
    assert not any("{_vllm}" in command for command in commands)


def test_a_wrapped_command_still_counts_as_reading_the_selector():
    """The real cells wrap the command, so a line-at-a-time scan would call a
    correct notebook broken."""
    cell = (
        "    !uv pip install -qqq --upgrade \\\n"
        "        unsloth {get_vllm} {get_numpy} torchvision\n"
    )
    assert any("{get_vllm}" in command for command in _install_commands(cell))


def test_the_requirement_is_found_whatever_the_selector_is_called():
    """Discriminating case for the detection above, held here so it keeps
    meaning something no matter how the notebooks spell the assignment. The
    names are not part of the contract; the quoted requirement is."""
    for line in (
        "    _vllm, _triton = ('vllm', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')",
        "    get_vllm, get_triton = ('vllm', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')",
        "    vllm_spec, triton_spec = ('vllm', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')",
        "    _vllm = 'vllm' if is_t4 else 'vllm==0.15.1'",
    ):
        assert _VLLM_SPEC_RE.findall(line) == ["vllm", "vllm==0.15.1"]


def test_a_vllm_string_that_is_not_a_requirement_is_left_alone():
    """These all appear in the notebooks. Treating them as requirements would
    fail the gate on lines that install nothing."""
    for line in (
        '!pkill -f "vllm.entrypoints.openai.api_server"',
        'with open("vllm_requirements.txt", "wb") as file:',
        'print(f"vllm server is running.")',
    ):
        assert _VLLM_SPEC_RE.findall(line) == []


def test_extras_and_other_operators_still_read_as_requirements():
    assert _VLLM_SPEC_RE.findall("('vllm[audio]==0.15.1',)") == ["vllm[audio]==0.15.1"]
    assert _VLLM_SPEC_RE.findall('("vllm>=0.15.1",)') == ["vllm>=0.15.1"]
    # An unpinned extra is still unpinned, and must reach the assertion above.
    assert "==" not in _VLLM_SPEC_RE.findall("('vllm[audio]',)")[0]
