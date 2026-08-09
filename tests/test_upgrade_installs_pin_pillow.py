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

"""An `--upgrade` install that names torchvision must pin Pillow first.

torchvision drags in a newer Pillow and swaps it under a kernel that already
imported PIL: the Python half is new, the compiled `_imaging` extension is old
(`Image.py:116: RuntimeWarning`), torchvision's `import PIL` fails, and
`unsloth_zoo` turns that into a hard error, so the notebook dies on its first
real cell. Found on Colab in `Advanced_Llama3_1_(3B)_GRPO_LoRA`; 45 of the 46
notebooks with such an install already carried the pin.

The scan folds backslash continuations first, since the broken install splits
`--upgrade` from `torchvision` across lines, and counts both flag spellings via
`notebook_inventory.UPGRADE_FLAG_RE` -- `-U` alone is 152 installs.

SCOPE: Python cells only. See `_torchvision_upgrades_by_cell`.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


def _pip_commands(source):
    """`pip install` commands with backslash continuations folded into one."""
    commands, pending = [], None
    for line in source.splitlines():
        if pending is not None:
            # Strip the `\` from EVERY continuation: one left in wedges a stray
            # token into the folded command.
            pending += " " + line.strip()
            if not line.rstrip().endswith("\\"):
                commands.append(pending)
                pending = None
            else:
                pending = pending.rstrip()[:-1].rstrip()
            continue
        if "pip install" in line:
            if line.rstrip().endswith("\\"):
                pending = line.rstrip()[:-1].strip()
            else:
                commands.append(line.strip())
    if pending is not None:
        commands.append(pending)
    return commands


def _all_code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def _code_cells(path):
    return [cell for cell in _all_code_cells(path) if not ni.is_shell_cell(cell)]


def _code(path):
    return "\n".join(_code_cells(path))


def _upgrades_torchvision(source):
    return [
        command for command in _pip_commands(source)
        if ni.is_upgrading(command) and "torchvision" in command
    ]


def _torchvision_upgrades_by_cell(path):
    """(command, cell) pairs. A placeholder resolves in its cell, not in the
    notebook: two notebooks hold two such installs in separate cells, and
    notebook scope lets one cell's exact `get_pil` answer for the other's.

    `%%bash` cells are deliberately not scanned, which excludes the 152 AMD
    ROCm installs

        uv pip install --system -U --force-reinstall \\
            torch torchvision torchaudio triton-rocm \\
            --index-url "$PYTORCH_INDEX_URL"

    that recognising `-U` makes visible. Not because they leave Pillow alone --
    a `--dry-run` there plans `pillow==11.0.0 -> 12.2.0` just like Colab -- but
    because the mismatch also needs `PIL` in `sys.modules` without `PIL.Image`,
    the state Colab's kernel is in before cell 1 (`google/colab/_reprs.py` line
    14 is a bare `import PIL as pil`). The AMD notebooks run on bare JupyterLab
    with no PIL resident, keep the `%%bash` install at code cell 0 where a
    subprocess shell cannot import PIL into the kernel, and make every later
    upgrade `--no-deps`. The generator draws the same line: update_all_notebooks
    `_AMD_INSTALL_PACKAGE_IGNORE` keeps the Colab Pillow pin out of the AMD
    variant, since the ROCm cell owns that half of the stack.

    The exclusion is a cell-type predicate rather than a notebook list, so a
    rename cannot widen it and an AMD install moved into a Python cell is back
    in scope. The tests below hold it to the cell-0 ROCm install with no PIL
    above it.
    """
    return [
        (command, cell)
        for cell in _code_cells(path)
        for command in _upgrades_torchvision(cell)
    ]


def _excluded_shell_upgrades():
    """(path, command) for every torchvision upgrade the scope above drops."""
    return [
        (path, command)
        for path in _NOTEBOOKS
        for cell in _all_code_cells(path)
        if ni.is_shell_cell(cell)
        for command in _upgrades_torchvision(cell)
    ]


# `(?<!\$)` so a shell `${VAR}` is not read as a Python placeholder: a bash
# assignment must not satisfy a probe written for an f-string pin.
_RE_PLACEHOLDER = re.compile(r"(?<!\$)\{(\w+)\}")


def _strip_comment(line):
    """`line` up to its first executable-code `#`, quotes respected.

    A pip requirement can carry a `#` of its own, as in the `#egg=` fragment of
    a VCS URL, so splitting on the first one would discard half a live command.
    """
    quote, escaped = None, False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _executable(text):
    """`text` with comments dropped, because a commented pin pins nothing."""
    return "\n".join(_strip_comment(line) for line in text.splitlines())


def _pins_pillow(command, source):
    """Whether THIS command pins Pillow, not whether the notebook mentions it.

    Dropping `{get_pil}` from the install while leaving `get_pil = ...` a few
    lines above still satisfies a whole-notebook search, and the upgrade
    resolves a fresh Pillow exactly as before. So a placeholder counts only
    when the notebook assigns that name an EXACT pin: the `get_pil = "pillow"`
    fallback, or a range like `pillow>=11`, is the failure rather than a pin.

    Comments are stripped from both first. Commenting a pin out is how one gets
    dropped, and the pattern is anchored on the name rather than the start of a
    line, so `# get_pil = "pillow==11.3.0"` read as a live assignment and left
    this gate green over an install reaching an undefined placeholder.
    """
    command, source = _executable(command), _executable(source)
    if re.search(r"pillow\s*==", command, re.I):
        return True
    for name in _RE_PLACEHOLDER.findall(command):
        # `[^\n;]` stops at the end of the statement. These notebooks chain both
        # pins on one line, so allowing `;` let `{get_numpy}` borrow the
        # `pillow` belonging to `get_pil` and every unpinned command read as
        # pinned.
        assignment = re.search(
            rf"\b{re.escape(name)}\s*=\s*[^\n;]*pillow\s*==", source, re.I
        )
        if assignment:
            return True
    return False


_NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb")) if NB_DIR.is_dir() else []


def _write_notebook(tmp_path, cells):
    """A throwaway .ipynb holding `cells` as code cells."""
    path = tmp_path / "sample.ipynb"
    path.write_text(
        json.dumps(
            {"cells": [
                {"cell_type": "code", "source": [source]} for source in cells
            ]}
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_an_upgrade_install_naming_torchvision_pins_pillow(path):
    upgrades = _torchvision_upgrades_by_cell(path)
    if not upgrades:
        pytest.skip("no --upgrade install names torchvision")
    unpinned = [c for c, cell in upgrades if not _pins_pillow(c, cell)]
    assert not unpinned, (
        f"{path.name} runs {unpinned[0][:160]!r} without pinning Pillow in that "
        f"command. That resolves a newer Pillow, leaves `_imaging` behind, and "
        f"the next `from unsloth import ...` dies on a PIL/torchvision mismatch. "
        f"Pin it the way the other notebooks do: "
        f"`try: import PIL; get_pil = f'pillow=={{PIL.__version__}}'`, then pass "
        f"`{{get_pil}}` to this install."
    )


def test_the_scan_folds_continuations_before_matching():
    """The broken install splits `--upgrade` from `torchvision` across a
    continuation, so a per-line match finds nothing."""
    text = (
        "!uv pip install -qqq --upgrade \\\n"
        "    unsloth vllm torchvision bitsandbytes\n"
    )
    assert _upgrades_torchvision(text), "continuation was not folded"
    per_line = [
        line for line in text.splitlines()
        if "--upgrade" in line and "torchvision" in line
    ]
    assert per_line == [], "the sample no longer exercises the continuation"


def test_an_upgrade_without_torchvision_is_not_flagged():
    """Only torchvision drags Pillow in; flagging every `--upgrade` is noise."""
    assert not _upgrades_torchvision("!uv pip install -qqq --upgrade unsloth vllm")


def test_a_pinned_torchvision_install_without_upgrade_is_not_flagged():
    assert not _upgrades_torchvision('!uv pip install -qqq "torchvision==0.24.0"')


_DEFINES_PIL = "try: import PIL; get_pil = f'pillow=={PIL.__version__}'"


@pytest.mark.parametrize("pin", ["pillow==11.3.0", "Pillow==11.3.0"])
def test_a_literal_pin_in_the_command_counts(pin):
    command = f"!uv pip install --upgrade {pin} torchvision"
    assert _pins_pillow(command, command)


def test_an_interpolated_pin_counts_when_the_notebook_defines_it():
    command = "!uv pip install --upgrade {get_pil} torchvision"
    assert _pins_pillow(command, _DEFINES_PIL + "\n" + command)


def test_an_unpinned_command_is_reported():
    """The discriminating case: without this the whole check is vacuous."""
    command = "!uv pip install -qqq --upgrade unsloth torchvision"
    assert not _pins_pillow(command, command)


def test_a_definition_elsewhere_does_not_excuse_an_unpinned_command():
    """The regression this gate exists for: the assignment stays above the
    install, so a whole-notebook search still calls it pinned."""
    command = "!uv pip install -qqq --upgrade unsloth torchvision"
    source = _DEFINES_PIL + "\n" + command
    assert re.search(r"pillow\s*==", source, re.I), "the decoy must look convincing"
    assert not _pins_pillow(command, source)


def test_a_placeholder_naming_something_else_does_not_count():
    """`{get_numpy}` shares the command and must not read as a Pillow pin."""
    command = "!uv pip install --upgrade {get_numpy} torchvision"
    source = 'get_numpy = f"numpy=={numpy.__version__}"\n' + command
    assert not _pins_pillow(command, source)


@pytest.mark.parametrize(
    "definition",
    ['get_pil = "pillow"', "get_pil = 'pillow>=11.3.0'", 'get_pil = "Pillow"'],
    ids=["unversioned", "range", "unversioned-capitalised"],
)
def test_a_placeholder_that_is_not_an_exact_pin_does_not_count(definition):
    """These notebooks already carry `except: get_pil = "pillow"`, so the
    unversioned spelling is one deleted line away -- and passing it to an
    `--upgrade` install is the failure, not a pin against it."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    assert not _pins_pillow(command, definition + "\n" + command)


def test_the_real_fallback_line_does_not_hide_the_pinned_one():
    """The notebook defines the name twice, pinned then unversioned:

        try: import PIL; get_pil = f'pillow=={PIL.__version__}'
        except: get_pil = "pillow"

    The pinned branch runs when PIL is loaded, which is exactly when the
    mismatch can happen, so it must count; requiring the FIRST assignment to be
    the pinned one would be a coin flip on source order."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    source = (
        'except: get_pil = "pillow"\n'
        + _DEFINES_PIL
        + "\n"
        + command
    )
    assert _pins_pillow(command, source)


def test_one_unpinned_command_is_caught_beside_a_pinned_one():
    """A notebook may run several upgrades; checking only the first lets a
    later unpinned one through."""
    pinned = "!uv pip install --upgrade {get_pil} torchvision"
    unpinned = "!uv pip install --upgrade torchvision"
    source = _DEFINES_PIL + "\n" + pinned + "\n" + unpinned
    commands = _upgrades_torchvision(source)
    assert len(commands) == 2
    assert [c for c in commands if not _pins_pillow(c, source)] == [unpinned]


def test_at_least_one_notebook_actually_exercises_the_check():
    """A broken glob or fold would leave every parametrised case skipped and
    the suite green."""
    exercised = [p.name for p in _NOTEBOOKS if _upgrades_torchvision(_code(p))]
    assert len(exercised) >= 40, (
        f"only {len(exercised)} notebooks reached the assertion; the scan is "
        f"probably broken rather than the repo suddenly clean"
    )


def test_every_torchvision_upgrade_is_still_reached_per_cell(tmp_path):
    """Splitting the scan per cell must not lose commands: two notebooks hold
    two such installs each."""
    per_cell = sum(len(_torchvision_upgrades_by_cell(p)) for p in _NOTEBOOKS)
    whole = sum(len(_upgrades_torchvision(_code(p))) for p in _NOTEBOOKS)
    assert per_cell == whole >= 40, (per_cell, whole)


def test_one_cells_pin_does_not_answer_for_another_cells_command():
    """Two install cells, one pinned and one not. Notebook scope lets the
    pinned cell's `get_pil` satisfy the unpinned one."""
    command = "!uv pip install --upgrade unsloth {get_pil} torchvision"
    pinned_cell = "try: import PIL; get_pil = f'pillow=={PIL.__version__}'\n" + command
    # The fallback when PIL is not importable; an `--upgrade` install resolves
    # a fresh Pillow from it.
    unpinned_cell = 'get_pil = "pillow"\n' + command
    notebook = pinned_cell + "\n" + unpinned_cell
    # Whole-notebook scope: the pinned cell answers for the unpinned one.
    assert _pins_pillow(command, notebook)
    # Cell scope: each cell answers for itself.
    assert _pins_pillow(command, pinned_cell)
    assert not _pins_pillow(command, unpinned_cell)


def test_a_sibling_pin_on_the_same_line_does_not_count():
    """These notebooks chain both pins in one statement, so a pattern running to
    end of LINE lets `{get_numpy}` borrow the `pillow` belonging to `get_pil`
    and every unpinned command reads as pinned."""
    definition = (
        "try: import numpy, PIL; "
        "get_numpy = f'numpy=={numpy.__version__}'; "
        "get_pil = f'pillow=={PIL.__version__}'"
    )
    unpinned = "!uv pip install --upgrade unsloth {get_numpy} torchvision"
    source = definition + "\n" + unpinned
    assert not _pins_pillow(unpinned, source)
    pinned = "!uv pip install --upgrade unsloth {get_numpy} {get_pil} torchvision"
    assert _pins_pillow(pinned, definition + "\n" + pinned)


def test_the_short_upgrade_flag_counts_as_an_upgrade():
    """`-U` and `--upgrade` are the same flag; reading only the long spelling
    exempted 152 installs."""
    for command in (
        "!uv pip install -U torchvision",
        "!uv pip install -qU unsloth torchvision",
        "uv pip install --system -U --force-reinstall torch torchvision",
    ):
        assert _upgrades_torchvision(command) == [command], command


def test_a_flag_that_merely_contains_u_is_not_an_upgrade():
    """Reading `--force-reinstall` or `--index-url` as an upgrade would drag
    unrelated installs into scope."""
    for command in (
        '!uv pip install --force-reinstall torchvision --index-url "$URL"',
        "!uv pip install -qqq torchvision",
    ):
        assert _upgrades_torchvision(command) == [], command


def test_a_short_flag_upgrade_in_a_python_cell_is_still_caught(tmp_path):
    """The other half of dropping `%%bash` cells: a `-U torchvision` install in
    an ordinary Python cell must still be collected and required to pin,
    otherwise the `-U` fix is scoped away to nothing."""
    unpinned = "!uv pip install -qU unsloth torchvision"
    path = _write_notebook(tmp_path, [unpinned])
    found = _torchvision_upgrades_by_cell(path)
    assert [c for c, _cell in found] == [unpinned]
    assert not _pins_pillow(*found[0])


def test_a_shell_magic_install_is_out_of_scope(tmp_path):
    """A `%%bash` cell runs in a subprocess shell, so it cannot leave `PIL`
    resident, which is the state the mismatch needs. See
    `_torchvision_upgrades_by_cell`."""
    path = _write_notebook(
        tmp_path,
        [
            "%%bash\nuv pip install --system -U --force-reinstall torch torchvision\n",
            "!uv pip install -qU unsloth torchvision",
        ],
    )
    found = [c for c, _cell in _torchvision_upgrades_by_cell(path)]
    assert found == ["!uv pip install -qU unsloth torchvision"], found


def test_the_shell_magic_exclusion_covers_only_the_rocm_stack_install():
    """The exclusion covers one command shape, the AMD ROCm stack install at
    code cell 0, so it cannot grow into an amnesty for anything in a `%%bash`
    cell: a different upgrading torchvision install there goes red."""
    excluded = _excluded_shell_upgrades()
    assert len(excluded) >= 100, (
        f"only {len(excluded)} shell-magic torchvision upgrades found; either "
        f"the AMD template changed or the scan stopped folding continuations"
    )
    unexpected = [
        (path.name, command)
        for path, command in excluded
        if "triton-rocm" not in command or "$PYTORCH_INDEX_URL" not in command
    ]
    assert not unexpected, (
        f"{len(unexpected)} shell-magic upgrade(s) are not the ROCm stack "
        f"install the exclusion was reasoned about: {unexpected[:2]}. Either "
        f"pin Pillow in them or extend the note in "
        f"_torchvision_upgrades_by_cell to cover them."
    )


def test_the_excluded_installs_are_the_notebooks_first_cell():
    """Half of why the exclusion holds: nothing has run yet, so there is no
    imported PIL for the upgrade to swap under."""
    late = []
    for path in _NOTEBOOKS:
        cells = _all_code_cells(path)
        for index, cell in enumerate(cells):
            if ni.is_shell_cell(cell) and _upgrades_torchvision(cell) and index:
                late.append((path.name, index))
    assert not late, (
        f"{len(late)} shell-magic torchvision upgrade(s) no longer run first, "
        f"so Python cells above them may have imported PIL: {late[:3]}"
    )


def test_no_excluded_notebook_imports_pil_before_the_install():
    """The other half: an `import PIL` above the install puts the kernel in the
    vulnerable state and the exclusion stops holding."""
    offenders = []
    for path, _command in _excluded_shell_upgrades():
        cells = _all_code_cells(path)
        index = next(
            i for i, cell in enumerate(cells)
            if ni.is_shell_cell(cell) and _upgrades_torchvision(cell)
        )
        earlier = "\n".join(cells[:index])
        if re.search(r"\bimport +PIL\b|\bfrom +PIL\b", earlier):
            offenders.append(path.name)
    assert not offenders, (
        f"{len(offenders)} notebook(s) import PIL before an excluded install, "
        f"so the upgrade can break the kernel there: {offenders[:3]}"
    )


def test_a_commented_out_assignment_is_not_a_pin():
    """How a pin actually gets dropped. The assignment pattern is anchored on
    the name, not on the start of a line, so a commented-out `get_pil` read as
    live and the install still reached `{get_pil}` undefined."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    for definition in (
        '# get_pil = "pillow==11.3.0"',
        "#get_pil = f'pillow=={PIL.__version__}'",
        '    # get_pil = "pillow==11.3.0"',
    ):
        assert not _pins_pillow(command, definition + "\n" + command), definition
    # A pin surviving only in a trailing comment beside the unversioned
    # fallback is the same hole, and the one an edit leaves behind.
    trailing = 'get_pil = "pillow"  # was pillow==11.3.0'
    assert not _pins_pillow(command, trailing + "\n" + command)
    # A literal pin commented out of the command itself, likewise.
    assert not _pins_pillow(
        "!uv pip install --upgrade torchvision  # pillow==11.3.0",
        "!uv pip install --upgrade torchvision  # pillow==11.3.0",
    )


def test_stripping_comments_leaves_live_code_alone():
    """A quoted `#`, such as a VCS URL's `#egg=` fragment, is part of the
    requirement; splitting on the first one would truncate a live command."""
    assert _strip_comment(_DEFINES_PIL) == _DEFINES_PIL
    assert _strip_comment('pip install "x @ git+https://h/r#egg=x"  # tail') == (
        'pip install "x @ git+https://h/r#egg=x"  '
    )
    command = "!uv pip install --upgrade {get_pil} torchvision"
    assert _pins_pillow(command, _DEFINES_PIL + "  # keep PIL in step\n" + command)


def test_a_shell_variable_is_not_read_as_a_python_placeholder():
    """`${PIL_PIN}` is a shell expansion, and reading it as `{PIL_PIN}` let a
    bash assignment satisfy a probe written for an f-string pin."""
    command = "uv pip install -U torchvision ${PIL_PIN}"
    source = 'PIL_PIN="pillow==11.0.0"\n' + command
    assert not _pins_pillow(command, source)
    assert _pins_pillow("!uv pip install -U torchvision {get_pil}", _DEFINES_PIL)


def test_a_fold_leaves_no_stray_backslash():
    """Each continuation drops its own `\\`; one left in wedges a stray token
    into the folded command."""
    text = (
        "uv pip install --system -U --force-reinstall \\\n"
        "    torch torchvision torchaudio triton-rocm \\\n"
        '    --index-url "$PYTORCH_INDEX_URL"\n'
    )
    assert _pip_commands(text) == [
        "uv pip install --system -U --force-reinstall torch torchvision "
        'torchaudio triton-rocm --index-url "$PYTORCH_INDEX_URL"'
    ]
