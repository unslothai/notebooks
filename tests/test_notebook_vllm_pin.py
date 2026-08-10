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

An unpinned side resolves to the newest release, whose default wheel is the CUDA
13 build from 0.20.0 on; Colab ships a CUDA 12 torch, so Unsloth disables vLLM
and `fast_inference = True` dies with "Please install vLLM before enabling
`fast_inference`!". Advanced_Llama3_1_(3B)_GRPO_LoRA.ipynb shipped unpinned on
the non-T4 branch and broke on L4. This gate keeps every branch pinned.
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


# A quoted pip requirement: 'vllm', "vllm==0.15.1", 'vllm[audio]==0.15.1'. Strict
# on purpose, since "everything up to the closing quote" also swallows
# "vllm.entrypoints.openai.api_server". Case-insensitive because PEP 503
# lowercases the name, so `VLLM` resolves the same CUDA 13 wheel; no separators,
# which is what keeps `VLLM_USE_V1` out.
_VLLM_SPEC_RE = re.compile(
    r"""['"](vllm(?:\[[^\[\]'"]*\])?\s*(?:[=<>!~]=?[^'"]*)?)['"]""",
    re.IGNORECASE,
)


def _logical_lines(source):
    """The cell's lines with backslash continuations joined into one.

    A wrapped command is one statement: read a physical line at a time, its tail
    (`"vllm==0.15.1" unsloth`) reads as a bare requirement rather than as the
    install carrying it, and the notebooks wrap nearly every install. Same fold
    as `scripts/molab_dependencies._logical_lines`.

    Comments go first, and for both scans at once so `_CASES` and `_BINDINGS`
    keep comparing the same text. A line such as `# Pin "vllm==0.15.1" for CUDA
    12` is prose, but it is neither a direct install nor a bindable assignment,
    so leaving it in reddens the gate on a correct notebook; 145 install
    commands in the tree already carry a trailing comment.
    """
    lines, pending = [], ""
    for line in ni.strip_comments(source).splitlines():
        pending = f"{pending} {line.strip()}" if pending else line
        if line.rstrip().endswith("\\"):
            pending = pending.rstrip()[:-1]
            continue
        lines.append(pending)
        pending = ""
    # A cell ending mid-continuation still carries its requirement.
    if pending:
        lines.append(pending)
    return lines


def _selector_cases(source):
    """(line, spec) per quoted vLLM requirement, so a synthetic cell can drive
    the scan the gate actually runs on rather than a copy of it."""
    for line in _logical_lines(source):
        for spec in _VLLM_SPEC_RE.findall(line):
            yield line.strip(), spec


def _selector_lines():
    """(notebook, line, spec) for every quoted vLLM requirement in a code cell.

    Keyed on the requirement, not the names it unpacks into: matching
    `_vllm, _triton = ...` literally let a rename drop notebooks silently while
    the count guard below still passed.
    """
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            for line, spec in _selector_cases(source):
                yield path, line, spec


_CASES = list(_selector_lines())


# A pinned selector only helps if the install still reads it, so each assignment
# is checked against its own cell's commands: swapping `{_vllm}` for a bare
# `vllm` leaves every case above green while Colab resolves the latest wheel.
# Only the interpolation is required, not the absence of a bare `vllm` elsewhere:
# the non-Colab `!pip install unsloth vllm` is unpinned on purpose in 83 cells.
# Any arity, since requiring a tuple dropped `_vllm = 'vllm==...'` out entirely.
_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:,\s*[A-Za-z_]\w*\s*)*=(?!=)"
)


def _result_tuples(value):
    """The tuples an unpacked right-hand side can evaluate to.

    Every selector is a ternary, so pairing only a plain `ast.Tuple` pairs
    nothing at all: both branches have to be walked.
    """
    if isinstance(value, ast.IfExp):
        return [value.body, value.orelse]
    return [value]


def _bound_name(line):
    """The name that actually holds the vLLM spec on an assignment line.

    The first target is wrong once vLLM is not the first value:
    `_triton, _vllm = ("triton", "vllm==0.15.1")` records `_triton`, so a
    command interpolating only `{_triton}` satisfies the binding check while the
    pin never installs. A hole rather than a live break today, but all 48
    selectors are written in the shape it applies to.

    Parsed, so target and value pair by position through either branch, falling
    back to the first target when nothing pairs: a line that does not parse
    alone, a non-tuple right-hand side, or branches that disagree.
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
    target = node.targets[0]
    if not isinstance(target, ast.Tuple):
        return fallback
    holders = set()
    for result in _result_tuples(node.value):
        if not isinstance(result, ast.Tuple): continue
        if len(result.elts) != len(target.elts): continue
        for name_node, value_node in zip(target.elts, result.elts):
            if not isinstance(name_node, ast.Name): continue
            if not isinstance(value_node, ast.Constant): continue
            if not isinstance(value_node.value, str): continue
            if _VLLM_SPEC_RE.search(f'"{value_node.value}"'):
                holders.add(name_node.id)
    # Branches naming different holders leave no single name to demand, so the
    # strict first-target reading stands rather than a guess between them.
    if len(holders) == 1:
        return holders.pop()
    return fallback


# A line that is itself the pip install rather than a selector feeding one.
# Anchored on the invocation, so a selector line that merely mentions pip
# installing in a comment stays in the binding check.
#
# `python -m pip install` is the same command by pip's own documented syntax,
# and it is how this repo already spells it: 156 install lines under nb/ use
# it, all of them in the AMD cells. Reading only the bare `pip` left
# `!python -m pip install -U "vllm==0.15.1"` looking like a selector line that
# binds no name, so a correctly pinned install would have failed
# `test_no_detected_selector_line_escapes_the_binding_check` -- the gate
# rejecting the very spelling it asks for.
_INSTALL_COMMAND_RE = re.compile(
    r"^\s*[!%]?\s*(?:uv\s+|python[0-9.]*\s+-m\s+)?pip\s+install\b"
)


def _installs_the_requirement_directly(line):
    """Whether the requirement is written straight into the install command.

    `!uv pip install --upgrade "vllm==0.15.1"` binds no name, so the binding
    check has nothing to demand. Not unchecked: the pin assertion still requires
    the `==` and the bare-vLLM check still rejects the unpinned spelling.
    """
    return _INSTALL_COMMAND_RE.match(line) is not None


def _lines_needing_a_binding(cases):
    """The detected lines the binding check may demand a name for.

    Split out of the check so a synthetic case can drive it: no notebook uses
    the direct form today, so a test calling only the helper would stay green
    with the exemption removed from the check.
    """
    return {
        (path, line)
        for path, line, _spec in cases
        if not _installs_the_requirement_directly(line)
    }


def _install_commands(source):
    """The `pip install` commands of a cell, continuations folded.

    `%%bash` lines run as shell and carry no `!`: 152 AMD cells hold 1064
    installs a `!`-only scan never sees, so the test is
    `notebook_inventory.is_shell_cell`, shared with the Pillow pin gate. `%`
    counts alongside `!` because `%pip install` is IPython's own recommended
    spelling, and a `!`-only scan would let it install an unpinned vLLM.
    """
    shell_cell = ni.is_shell_cell(source)
    commands = []
    for command in _logical_lines(source):
        prefixed = command.lstrip().startswith(("!", "%"))
        if (prefixed or shell_cell) and "pip install" in command:
            commands.append(command)
    return commands


def _upgrading(commands):
    """The commands that can replace an already-installed vLLM.

    Only `-U`/`--upgrade` does, the short spelling inside clusters such as
    `-qU`; the flag test is `notebook_inventory.UPGRADE_FLAG_RE`, shared with
    the Pillow pin gate so the two cannot drift on what an upgrade is. This also
    keeps the non-Colab `!pip install unsloth vllm` out of scope: unpinned on
    purpose, and it upgrades nothing.
    """
    return [command for command in commands if ni.is_upgrading(command)]


# A vLLM requirement in a shell command with no exact version: `vllm`,
# `vllm[audio]`, `vllm>=0.15.1`. Extras are part of the name, so hopping them
# separates `vllm[audio]` from `vllm[audio]==0.15.1`. The lookarounds keep
# `{_vllm}`, `vllm_requirements.txt`, a `vllm-project/vllm` URL and
# `UNSLOTH_VLLM_STANDBY` out; case-insensitive for the PEP 503 reason above.
_BARE_VLLM_RE = re.compile(
    r"(?<![\w{=./\[-])vllm(?:\[[^\]\s]*\])?(?![\w}=./\[-])", re.IGNORECASE
)


def _selector_bindings():
    """(notebook, line, name, upgrading commands) per vLLM selector."""
    for path in ni.iter_notebooks():
        for _index, source in ni.iter_code_cells(path):
            commands = _upgrading(_install_commands(source))
            for line in _logical_lines(source):
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
    still clears it. Every detected line must bind, bar those installing the
    pin themselves."""
    detected = _lines_needing_a_binding(_CASES)
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
    """PEP 503 lowercases project names, so `VLLM` resolves the same CUDA 13
    wheel as `vllm`. No notebook spells it that way today; the gate exists for
    the one that will."""
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
    """The check only means something if the command installs the *pinned*
    selector, which needs pairing by position once vLLM is not the first value.
    Reverting `_bound_name` to the first target reddens the second case here
    and nothing else in the file."""
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
    """Nothing pairs on these, either because the line is not valid Python alone
    or because no branch lines a name up with the spec. The first-target
    fallback keeps them in the binding check rather than dropping them."""
    for line, name in (
        # A placeholder the generator fills in later.
        ("    _vllm, _t = ({SPEC}, 'triton')", "_vllm"),
        # A right-hand side that is not a tuple: nothing to pair by position.
        ("    _t, _vllm = _pair", "_t"),
        # Mismatched arity, in both branches.
        ("    _t, _u, _vllm = ('t', 'vllm==0.15.1') if is_t4 else ('t', 'u')", "_t"),
        # Branches disagreeing: guessing one would demand the wrong `{...}`.
        (
            "    _a, _b = ('vllm==0.9.2', 't') if is_t4 else ('t', 'vllm==0.15.1')",
            "_a",
        ),
    ):
        assert _bound_name(line) == name, line


def test_the_conditional_selector_pairs_by_position_in_both_branches():
    """Every notebook uses a ternary, so pairing only a plain tuple pairs none
    of the 48 selectors and the whole binding check runs on the first-target
    fallback. Reverting `_result_tuples` reddens the first three cases."""
    for line, name in (
        # vLLM second in both branches: the case the fallback gets wrong.
        (
            "    _triton, _vllm = ('triton==3.2.0', 'vllm==0.9.2') if is_t4 "
            "else ('triton', 'vllm==0.15.1')",
            "_vllm",
        ),
        # Pinned on one branch only: still the same holder, still bound.
        ("    _t, _vllm = ('t', 'vllm') if is_t4 else ('t', 'vllm==0.15.1')", "_vllm"),
        # A branch that is not a tuple of constants does not veto the other.
        ("    _t, _vllm = ('t', 'vllm==0.15.1') if is_t4 else (_a, _b)", "_vllm"),
        # ...without disturbing the spelling the notebooks use today.
        (
            "    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 "
            "else ('vllm==0.15.1', 'triton')",
            "_vllm",
        ),
    ):
        assert _bound_name(line) == name, line


def test_a_directly_pinned_install_needs_no_binding():
    """The command carries the requirement with no selector to interpolate, so
    demanding a binding would fail a correct notebook. The other two checks
    still cover it, which is what makes the exemption safe: dropping
    `_installs_the_requirement_directly` reddens this test."""
    path = REPO_ROOT / "nb" / "Synthetic.ipynb"
    command = '!uv pip install --upgrade "vllm==0.15.1"'
    # The gate on a case list of one: detected must come out empty.
    assert _bound_name(command) is None
    assert _lines_needing_a_binding([(path, command, "vllm==0.15.1")]) == set()
    # Neither a command nor an assignment, so the exemption is not a blanket.
    stray = "    _specs.append('vllm==0.15.1')"
    assert _lines_needing_a_binding([(path, stray, "vllm==0.15.1")]) == {(path, stray)}
    # Still pin-checked, and the unpinned spelling still rejected twice over.
    assert _VLLM_SPEC_RE.findall(command) == ["vllm==0.15.1"]
    unpinned = '!uv pip install --upgrade "vllm"'
    assert "==" not in _VLLM_SPEC_RE.findall(unpinned)[0]
    assert _BARE_VLLM_RE.search(unpinned)
    assert not _BARE_VLLM_RE.search(command)


def test_a_wrapped_direct_install_is_exempt_like_an_unwrapped_one():
    """The notebooks wrap nearly every install, so an exemption reading only
    unwrapped commands fails a correctly pinned notebook: a physical line at a
    time the tail is `"vllm==0.15.1"`, neither command nor assignment, and the
    gate demands a binding no selector can supply. Reverting `_selector_cases`
    to `source.splitlines()` reddens this."""
    path = REPO_ROOT / "nb" / "Synthetic.ipynb"
    cell = '    !uv pip install -qqq --upgrade \\\n        "vllm==0.15.1" unsloth\n'
    cases = [(path, line, spec) for line, spec in _selector_cases(cell)]
    assert [spec for _p, _l, spec in cases] == ["vllm==0.15.1"], cases
    # The gate: a command binds nothing, so both sides must come out empty.
    bound = {(path, line) for line, _spec in _selector_cases(cell)
             if _bound_name(line) is not None}
    assert _lines_needing_a_binding(cases) == bound == set()
    # The fold must not cost the check that makes the exemption safe.
    unpinned = '    !uv pip install -qqq --upgrade \\\n        vllm unsloth\n'
    upgrading = _upgrading(_install_commands(unpinned))
    assert upgrading and _BARE_VLLM_RE.search(upgrading[0]), upgrading


def test_a_requirement_named_only_in_a_comment_is_not_a_selector():
    """Prose is not a selector: a commented requirement is neither a direct
    install nor a bindable assignment, so leaving it in `_CASES` reddens the
    hard gate on a correct notebook. 145 install commands already carry a
    trailing comment, so this is a spelling away."""
    path = REPO_ROOT / "nb" / "Synthetic.ipynb"
    cell = (
        '    # Pin "vllm==0.15.1"; the 0.20 wheel is CUDA 13.\n'
        "    _vllm = 'vllm==0.15.1'  # keep in step with the generator\n"
        "    !uv pip install -qqq --upgrade unsloth {_vllm}\n"
    )
    cases = [(path, line, spec) for line, spec in _selector_cases(cell)]
    assert [spec for _p, _l, spec in cases] == ["vllm==0.15.1"], cases
    bound = {(path, line) for line, _spec in _selector_cases(cell)
             if _bound_name(line) is not None}
    assert _lines_needing_a_binding(cases) == bound
    # A trailing comment must not feed the bare check either.
    assert not any(
        _BARE_VLLM_RE.search(command)
        for command in _upgrading(_install_commands(
            "    !uv pip install -qqq --upgrade unsloth {_vllm}  # not vllm\n"))
    )


def test_the_exemption_does_not_swallow_a_selector_assignment():
    """Exempting every line that mentions pip installing would hand the
    binding check a way to be satisfied by a comment."""
    for line in (
        "    _vllm, _t = ('vllm==0.15.1', 't')  # run pip install after this",
        "    _vllm = 'vllm==0.15.1'",
    ):
        assert not _installs_the_requirement_directly(line), line
    for command in (
        '!uv pip install --upgrade "vllm==0.15.1"',
        '!pip install "vllm==0.15.1"',
        '%pip install "vllm==0.15.1"',
        # No `!`, because a `%%bash` cell runs its lines as shell.
        'uv pip install --system -U "vllm==0.15.1"',
    ):
        assert _installs_the_requirement_directly(command), command


def test_the_module_invocation_is_the_same_install():
    """`python -m pip install` is pip's own documented syntax and the spelling
    this repo already uses, 156 install lines of it under nb/. Reading only the
    bare `pip` made a correctly pinned `!python -m pip install -U
    "vllm==0.15.1"` look like a selector line binding no name, so the gate
    would have rejected the spelling it asks for."""
    for command in (
        '!python -m pip install -U "vllm==0.15.1"',
        '!python3 -m pip install "vllm==0.15.1"',
        '%python -m pip install "vllm==0.15.1"',
        # `%%bash` cells carry no `!`, which is where the repo's own uses live.
        'python3.12 -m pip install --root-user-action=ignore "vllm==0.15.1"',
    ):
        assert _installs_the_requirement_directly(command), command
    # Still anchored on a real invocation: prose about the module form, and a
    # module that is not pip, both stay in the binding check.
    for line in (
        "    _vllm = 'vllm==0.15.1'  # then python -m pip install it",
        '    !python -m build install "vllm==0.15.1"',
    ):
        assert not _installs_the_requirement_directly(line), line


def test_a_line_that_is_not_an_assignment_binds_nothing():
    """`None` is what makes `_selector_bindings` skip a line rather than record
    a binding named `None`."""
    assert _bound_name("    if _vllm == 'vllm==0.15.1':") is None
    assert _bound_name("    !pip install vllm==0.15.1") is None


def test_a_percent_pip_install_is_collected():
    """`%pip install` is IPython's own recommended spelling, so a `!`-only scan
    is a hole: the notebook that adds it would ship an unpinned vLLM past a
    green gate. Asserted through `_install_commands`, the collector every later
    check reads from, rather than through the prefix test."""
    assert _install_commands("%pip install --upgrade vllm") == [
        "%pip install --upgrade vllm"]
    assert _BARE_VLLM_RE.search(_upgrading(
        _install_commands("%pip install --upgrade vllm"))[0])


def test_a_commented_percent_line_is_not_an_install():
    """The prefix widened to `%`; it must not widen to anything that merely
    starts a line."""
    assert _install_commands("# %pip install --upgrade vllm") == []
    assert _install_commands("    pip install vllm") == []


def test_a_bare_vllm_is_told_apart_from_a_pinned_or_interpolated_one():
    """Discriminating cases, so the bare check keeps meaning something once the
    tree is clean."""
    for command in (
        "!uv pip install -qqq --upgrade unsloth vllm torchvision",
        "!pip install --upgrade vllm",
        # Without the hop over extras, `[` ended the match and these read clean.
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
    """Reading `--index-url` as an upgrade would pull unrelated commands into
    scope."""
    for command in (
        '!uv pip install --system torch --index-url "$URL"',
        "!uv pip install -qqq unsloth vllm",
        "!pip install --no-deps unsloth vllm",
    ):
        assert _upgrading([command]) == [], command


def test_a_reinstall_counts_as_an_upgrade():
    """`--force-reinstall` with no version resolves vLLM from the index, so it
    replaces the installed one exactly as `-U` does. pip: "Reinstall all
    packages even if they are already up-to-date"; uv spells it `--reinstall`.
    Skipping it left an unpinned vLLM install outside both gates."""
    for command in (
        "!pip install --force-reinstall vllm",
        "!uv pip install --reinstall vllm",
        "!uv pip install --system --force-reinstall vllm --index-url $URL",
    ):
        assert _upgrading([command]) == [command], command
    # `--reinstall-package` names one package, so it is not the whole-environment
    # reinstall the flag above is, and the option boundary has to hold.
    assert _upgrading(["!uv pip install --reinstall-package torch vllm"]) == []


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
