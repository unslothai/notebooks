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

Leaving either side unpinned resolves to whatever is newest on PyPI, and vLLM's
default wheel is the CUDA 13 build from 0.20.0 on while Colab ships a CUDA 12
torch, so Unsloth disables vLLM and `fast_inference = True` dies with "Please
install vLLM before enabling `fast_inference`!".

Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb shipped unpinned on the non-T4 branch and
broke on L4. This gate keeps every branch of every such line pinned.
"""
from __future__ import annotations

import ast
import re
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


# A quoted pip requirement for vLLM: 'vllm', "vllm==0.15.1", 'vllm[audio]==0.15.1'.
# Strict on purpose: "everything up to the closing quote" also swallows
# "vllm.entrypoints.openai.api_server" and "vllm_requirements.txt", which are not
# requirements. Case-insensitive because PEP 503 lowercases the project name, so
# `VLLM` resolves the same CUDA 13 wheel; separators are left out since `vllm`
# has none and matching them invites the `VLLM_USE_V1` false positive.
_VLLM_SPEC_RE = re.compile(
    r"""['"](vllm(?:\[[^\[\]'"]*\])?\s*(?:[=<>!~]=?[^'"]*)?)['"]""",
    re.IGNORECASE,
)


def _selector_lines():
    """(notebook, line, spec) for every quoted vLLM requirement in a code cell.

    Keyed on the requirement, not on the names the cell unpacks it into:
    matching `_vllm, _triton = ...` literally let a rename drop notebooks from
    the gate silently while the count guard below still passed.
    """
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            for line in source.splitlines():
                for spec in _VLLM_SPEC_RE.findall(line):
                    yield path, line.strip(), spec


_CASES = list(_selector_lines())


# Pinning the selector only helps if the install command still reads it, so each
# assignment is checked against the commands in its own cell: swapping the
# interpolated `{_vllm}` for a bare `vllm` leaves every case above green while
# Colab resolves the latest wheel again. Only the interpolation is required, not
# the absence of a bare `vllm` elsewhere -- the non-Colab branch is `!pip install
# unsloth vllm`, unpinned on purpose in 83 cells.
#
# Any arity, not just today's two-name unpack: requiring a tuple dropped
# `_vllm = 'vllm==...'` out of the bindings while the count guard stayed happy.
_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:,\s*[A-Za-z_]\w*\s*)*=(?!=)"
)


def _bound_name(line):
    """The name that actually holds the vLLM spec on an assignment line.

    Taking the first target is wrong once vLLM is not the first value:
    `_triton, _vllm = ("triton", "vllm==0.15.1")` records `_triton`, so a
    command interpolating only `{_triton}` satisfies the binding check while
    the pin is never installed. Today's notebooks all put vLLM first, so it is
    a hole rather than a live break, but a future edit falls into it.

    Parsed rather than pattern-matched, so target and value pair by position.
    Falls back to the first target when the line does not parse on its own (a
    `{...}` placeholder, an f-string, a ternary), keeping prior behaviour.
    """
    match = _ASSIGNMENT_RE.match(line)
    if match is None:
        return None
    fallback = match.group("name")
    try:
        node = ast.parse(textwrap.dedent(line).strip()).body[0]
    except (SyntaxError, ValueError, IndexError):
        return fallback
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return fallback
    target, value = node.targets[0], node.value
    if not isinstance(target, ast.Tuple) or not isinstance(value, ast.Tuple):
        return fallback
    if len(target.elts) != len(value.elts):
        return fallback
    for name_node, value_node in zip(target.elts, value.elts):
        if not isinstance(name_node, ast.Name): continue
        if not isinstance(value_node, ast.Constant): continue
        if not isinstance(value_node.value, str): continue
        if _VLLM_SPEC_RE.search(f'"{value_node.value}"'):
            return name_node.id
    return fallback


def _install_commands(source):
    """The `pip install` commands of a cell, backslash continuations joined,
    because the command consuming the selector is often wrapped.

    A `%%bash` cell's lines run as shell and carry no `!`: 152 AMD cells hold
    1064 installs a `!`-only scan never sees. The predicate is
    `notebook_inventory.is_shell_cell`, shared with the Pillow pin gate.
    """
    shell_cell = ni.is_shell_cell(source)
    commands, pending = [], ""
    for line in source.splitlines():
        pending = f"{pending} {line.strip()}" if pending else line
        if line.rstrip().endswith("\\"):
            pending = pending.rstrip()[:-1]
            continue
        prefixed = pending.lstrip().startswith("!")
        if (prefixed or shell_cell) and "pip install" in pending:
            commands.append(pending)
        pending = ""
    return commands


def _upgrading(commands):
    """The commands that can replace an already-installed vLLM.

    Only `-U`/`--upgrade` resolves a fresh vLLM over one already present. Both
    spellings count, the short one inside clusters such as `-qU`; the flag test
    is `notebook_inventory.UPGRADE_FLAG_RE`, shared with the Pillow pin gate so
    the two cannot drift on what an upgrade is.

    Narrowing to upgrades keeps the non-Colab branch, `!pip install unsloth
    vllm`, out of scope: unpinned on purpose, and it upgrades nothing.
    """
    return [command for command in commands if ni.is_upgrading(command)]


# A vLLM requirement written straight into a shell command with no exact
# version: `vllm`, `vllm[audio]`, `vllm>=0.15.1`. Extras are part of the name,
# so hopping over them separates `vllm[audio]` from `vllm[audio]==0.15.1`. Not
# `{_vllm}`, `vllm==0.15.1`, `vllm_requirements.txt` or a `vllm-project/vllm`
# URL. Case-insensitive for the PEP 503 reason above; the lookarounds keep
# `UNSLOTH_VLLM_STANDBY` and `VLLM_USE_V1=1` out.
_BARE_VLLM_RE = re.compile(
    r"(?<![\w{=./\[-])vllm(?:\[[^\]\s]*\])?(?![\w}=./\[-])", re.IGNORECASE
)


def _selector_bindings():
    """(notebook, line, name, upgrading commands) per vLLM selector."""
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            commands = _upgrading(_install_commands(source))
            for line in source.splitlines():
                if not _VLLM_SPEC_RE.search(line):
                    continue
                name = _bound_name(line)
                if name is None:
                    continue
                yield path, line.strip(), name, commands


_BINDINGS = list(_selector_bindings())


def _upgrading_commands():
    """Every upgrading pip install in the tree, for the bare-vLLM check."""
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            for command in _upgrading(_install_commands(source)):
                yield path, command


_UPGRADES = list(_upgrading_commands())


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


def test_no_detected_selector_line_escapes_the_binding_check():
    """A count floor cannot see a shape that stopped being recognised: 47 of 48
    bindings still clears it. Every detected line must produce a binding."""
    detected = {(path, line) for path, line, _spec in _CASES}
    bound = {(path, line) for path, line, _name, _commands in _BINDINGS}
    assert detected == bound, (
        f"{len(detected - bound)} line(s) carry a vLLM requirement but are not "
        f"recognised as an assignment, so no command is checked for them: "
        f"{sorted(str(p.name) + ': ' + l for p, l in detected - bound)[:3]}"
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
        f"but no upgrading pip install in that cell interpolates {{{name}}}, "
        f"so the pin installs nothing. Commands found: {commands or 'none'}"
    )


def test_upgrading_commands_are_actually_found():
    """Guards the check below against an empty scan."""
    assert len(_UPGRADES) >= 40, (
        f"only {len(_UPGRADES)} upgrading pip installs found; the command scan "
        f"has drifted away from the notebooks' install cells"
    )


@pytest.mark.parametrize(
    "path,command",
    _UPGRADES,
    ids = [f"{p.parent.name}/{p.stem}" for p, _c in _UPGRADES],
)
def test_no_upgrading_command_names_vllm_bare(path, command):
    """One command satisfying the binding does not stop a second in the same
    cell upgrading vLLM unpinned; every upgrade goes through the selector."""
    assert not _BARE_VLLM_RE.search(command), (
        f"{path.relative_to(REPO_ROOT)} upgrades an unpinned vLLM in:\n"
        f"  {command}\n"
        f"--upgrade resolves the latest release over whatever is installed, "
        f"and its default PyPI wheel is the CUDA 13 build. Pass the pinned "
        f"selector instead of a bare `vllm`."
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
    """The names are not part of the contract; the quoted requirement is."""
    for line in (
        "    _vllm, _triton = ('vllm', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')",
        "    get_vllm, get_triton = ('vllm', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')",
        "    vllm_spec, triton_spec = ('vllm', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')",
        "    _vllm = 'vllm' if is_t4 else 'vllm==0.15.1'",
    ):
        assert _VLLM_SPEC_RE.findall(line) == ["vllm", "vllm==0.15.1"]


def test_a_case_variant_requirement_is_still_a_requirement():
    """PEP 503 lowercases project names, so `VLLM` and `vLLM` are the same
    project as `vllm` and resolve the same CUDA 13 wheel. No notebook spells it
    that way today; the gate exists for the one that will.
    """
    assert _VLLM_SPEC_RE.findall("    _v, _t = ('VLLM', 't') if is_t4 else ('vLLM==0.15.1', 't')") == [
        "VLLM", "vLLM==0.15.1",
    ]
    assert "==" not in _VLLM_SPEC_RE.findall("('VLLM[audio]',)")[0]
    for command in (
        "!uv pip install -qqq --upgrade unsloth VLLM torchvision",
        "!pip install --upgrade vLLM",
        "!uv pip install --upgrade VLLM[audio]",
        "!uv pip install --upgrade VLLM>=0.15.1",
    ):
        assert _BARE_VLLM_RE.search(command), command
    for command in (
        '!uv pip install --upgrade "VLLM==0.15.1"',
        "!uv pip install --upgrade unsloth {_vllm} torchvision",
        "!pip install --upgrade git+https://github.com/vllm-project/VLLM",
    ):
        assert not _BARE_VLLM_RE.search(command), command


def test_a_case_variant_requirement_in_a_shell_magic_cell_is_caught():
    """Both halves must line up: no `!`, so it is scanned only because the cell
    is `%%bash`, and flagged only because the name matches any case."""
    cell = (
        "%%bash\n"
        "set -e\n"
        "uv pip install --system -U VLLM\n"
    )
    upgrading = _upgrading(_install_commands(cell))
    assert upgrading == ["uv pip install --system -U VLLM"]
    assert _BARE_VLLM_RE.search(upgrading[0])


def test_an_upper_case_vllm_env_var_is_not_read_as_a_requirement():
    """69 cells set `UNSLOTH_VLLM_STANDBY`. Matching an env var or a path as a
    requirement would fail the gate on lines that install nothing."""
    for line in (
        'os.environ["UNSLOTH_VLLM_STANDBY"] = "1"',
        'os.environ["VLLM_USE_V1"] = "1"',
        'os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"',
    ):
        assert _VLLM_SPEC_RE.findall(line) == [], line
    for command in (
        "!VLLM_USE_V1=1 uv pip install --upgrade {_vllm}",
        "!uv pip install --upgrade --cache-dir /opt/VLLM_CACHE/ {_vllm}",
        "!VLLM=1 pip install --upgrade unsloth",
    ):
        assert not _BARE_VLLM_RE.search(command), command


def test_a_vllm_string_that_is_not_a_requirement_is_left_alone():
    """All appear in the notebooks and install nothing."""
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


def test_a_single_target_selector_is_recognised_as_an_assignment():
    """`_vllm = 'vllm==...'` is a spelling the detection accepts, so the
    binding check has to accept it too or that notebook goes unchecked."""
    for line, name in (
        ("    _vllm = 'vllm==0.15.1'", "_vllm"),
        ("    _vllm, _triton = ('vllm==0.9.2', 'triton')", "_vllm"),
        ("    _vllm, _triton, _torch = ('vllm==0.9.2', 'triton', 'torch')", "_vllm"),
    ):
        match = _ASSIGNMENT_RE.match(line)
        assert match is not None and match.group("name") == name, line


def test_a_comparison_is_not_mistaken_for_an_assignment():
    assert _ASSIGNMENT_RE.match("    if _vllm == 'vllm==0.15.1':") is None


def test_the_name_bound_is_the_one_holding_the_spec():
    """The binding check only means something if the command installs the
    *pinned* selector, which needs pairing by position once vLLM is not the
    first value. Reverting `_bound_name` to the first target reddens the second
    case here and nothing else in the file."""
    for line, name in (
        # The case the first-target reading gets wrong.
        ("    _triton, _vllm = ('triton', 'vllm==0.15.1')", "_vllm"),
        ("    _t, _u, _vllm = ('t', 'u', 'vllm==0.15.1')", "_vllm"),
        # ...without disturbing the spelling every notebook uses today.
        ("    _vllm, _triton = ('vllm==0.9.2', 'triton')", "_vllm"),
        ("    _vllm = 'vllm==0.15.1'", "_vllm"),
    ):
        assert _bound_name(line) == name, line


def test_a_line_that_does_not_parse_keeps_the_previous_behaviour():
    """Selector lines are extracted one physical line at a time, so many are
    not valid Python alone. The first-target fallback keeps them as strict as
    before rather than dropping them from the binding check."""
    for line, name in (
        # A placeholder the generator fills in later.
        ("    _vllm, _t = ({SPEC}, 'triton')", "_vllm"),
        # A ternary whose branches are not plain constants.
        ("    _t, _vllm = ('t', 'vllm==0.15.1') if is_t4 else (_a, _b)", "_t"),
        # Mismatched arity: nothing to pair by position.
        ("    _t, _vllm = _pair", "_t"),
    ):
        assert _bound_name(line) == name, line


def test_a_line_that_is_not_an_assignment_binds_nothing():
    """`None` is what makes `_selector_bindings` skip a line rather than record
    a binding named `None`."""
    assert _bound_name("    if _vllm == 'vllm==0.15.1':") is None
    assert _bound_name("    !pip install vllm==0.15.1") is None


def test_a_bare_vllm_is_told_apart_from_a_pinned_or_interpolated_one():
    """Discriminating cases, so the bare check keeps meaning something once the
    tree is clean."""
    for command in (
        "!uv pip install -qqq --upgrade unsloth vllm torchvision",
        "!pip install --upgrade vllm",
        # Extras are part of the name: without the hop, `[` ended the match and
        # an unpinned extras install read as clean.
        "!uv pip install --upgrade vllm[audio]",
        "!uv pip install --upgrade unsloth vllm[audio,video] torchvision",
        # A range resolves the latest that satisfies it, so it is not a pin.
        "!uv pip install --upgrade vllm>=0.15.1",
    ):
        assert _BARE_VLLM_RE.search(command), command
    for command in (
        "!uv pip install -qqq --upgrade unsloth {_vllm} torchvision",
        "!uv pip install -qqq --upgrade unsloth {get_vllm} torchvision",
        '!uv pip install --upgrade "vllm==0.15.1"',
        "!uv pip install --upgrade vllm==0.15.1",
        '!uv pip install --upgrade "vllm[audio]==0.15.1"',
        "!uv pip install --upgrade vllm[audio]==0.15.1",
        "!pip install --upgrade -r vllm_requirements.txt",
        "!pip install --upgrade git+https://github.com/vllm-project/vllm",
        "!pip install --upgrade unsloth[vllm]==2026.8.9",
    ):
        assert not _BARE_VLLM_RE.search(command), command


def test_a_non_upgrading_local_install_is_out_of_scope():
    """The non-Colab branch installs vLLM unpinned on purpose; it upgrades
    nothing, so it must not be collected."""
    cell = (
        'if "COLAB_" not in "".join(os.environ.keys()):\n'
        "    !pip install unsloth vllm\n"
        "else:\n"
        "    _vllm, _triton = ('vllm==0.9.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')\n"
        "    !uv pip install -qqq --upgrade unsloth {_vllm} torchvision\n"
    )
    commands = _install_commands(cell)
    assert len(commands) == 2, commands
    upgrading = _upgrading(commands)
    assert len(upgrading) == 1 and "{_vllm}" in upgrading[0], upgrading


def test_a_second_upgrading_command_cannot_hide_behind_the_first():
    """The failure this pair of checks exists for: one command satisfies the
    binding while a later one upgrades vLLM unpinned."""
    cell = (
        "    _vllm, _triton = ('vllm==0.9.2', 'triton') if is_t4 else ('vllm==0.15.1', 'triton')\n"
        "    !uv pip install -qqq --upgrade {_vllm}\n"
        "    !uv pip install -qqq --upgrade unsloth vllm torchvision\n"
    )
    upgrading = _upgrading(_install_commands(cell))
    assert any("{_vllm}" in command for command in upgrading)
    assert [c for c in upgrading if _BARE_VLLM_RE.search(c)], upgrading


def test_the_short_upgrade_flag_counts_as_an_upgrade():
    """Matching only the long spelling let `!uv pip install -U vllm` past the
    check entirely."""
    for command in (
        "!uv pip install -U vllm",
        "!uv pip install -qU vllm",
        "!uv pip install --system -U --force-reinstall vllm",
        "!uv pip install --upgrade vllm",
    ):
        assert _upgrading([command]) == [command], command


def test_a_flag_that_merely_contains_u_is_not_an_upgrade():
    """Reading `--force-reinstall` or `--index-url` as an upgrade would pull
    unrelated commands into scope."""
    for command in (
        '!uv pip install --system --force-reinstall torch --index-url "$URL"',
        "!uv pip install -qqq unsloth vllm",
        "!pip install --no-deps unsloth vllm",
    ):
        assert _upgrading([command]) == [], command


def test_a_shell_magic_cell_has_its_installs_scanned():
    """`%%bash` cells run their lines as shell, so an install there needs no
    `!`. A `!`-only scan skipped 1064 commands across the AMD notebooks."""
    cell = (
        "%%bash\n"
        "set -e\n"
        "uv pip install --system -U vllm\n"
    )
    assert _install_commands(cell) == ["uv pip install --system -U vllm"]


def test_a_python_cell_still_needs_the_bang():
    """`%%capture` cells are Python, so a bare `pip install` there is a string
    or a comment rather than a command."""
    cell = (
        "%%capture\n"
        "print('run pip install vllm yourself')\n"
        "!uv pip install --upgrade {_vllm}\n"
    )
    assert _install_commands(cell) == ["!uv pip install --upgrade {_vllm}"]
