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
`notebook_inventory.UPGRADE_FLAG_RE`: `-U` alone is 152 installs.

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


def _runs_as_shell(command, shell_cell):
    """Whether a folded line is a command rather than prose.

    A Python cell needs `!`/`%` for a shell command, so `# !pip install -U
    torchvision` and `print("pip install -U torchvision")` never execute: 87
    such lines sit in the tree, and collecting them reports an unpinned upgrade
    in a notebook that runs none. A `%%bash` cell runs every line as shell.

    Applied to the FOLDED command, since a continuation line carries no prefix
    of its own and filtering physical lines would drop the tail of every
    wrapped install.
    """
    return shell_cell or command.lstrip().startswith(("!", "%"))


def _pip_commands(source):
    """`pip install` commands with backslash continuations folded into one."""
    shell_cell = ni.is_shell_cell(source)
    folded, pending = [], None
    for line in ni.strip_comments(source).splitlines():
        if pending is not None:
            # EVERY continuation drops its `\`; one left in wedges a stray token.
            pending += " " + line.strip()
            if not line.rstrip().endswith("\\"):
                folded.append(pending)
                pending = None
            else:
                pending = pending.rstrip()[:-1].rstrip()
            continue
        if "pip install" in line:
            if line.rstrip().endswith("\\"):
                pending = line.rstrip()[:-1].strip()
            else:
                folded.append(line.strip())
    if pending is not None:
        folded.append(pending)
    return [c for c in folded if _runs_as_shell(c, shell_cell)]


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


# PEP 503 lowercases project names, so `TorchVision` installs the same wheel.
# A case-sensitive test dropped such a command out of the gate entirely, which
# is the one direction that fails silently: Pillow upgrades, nothing complains.
_TORCHVISION_RE = re.compile(r"torchvision", re.I)


# Pillow reaches the kernel as torchvision's dependency, so `--no-deps` is the
# one flag that makes a replacing install harmless: nothing but the named
# requirements is touched. Two notebooks reinstall a pinned torchvision that
# way. It stops protecting the moment the command names Pillow itself, which is
# then an ordinary unpinned requirement.
_NO_DEPS_RE = re.compile(r"(?<![\w-])--no-deps(?![\w-])")
_PILLOW_REQ_RE = re.compile(r"(?<![\w{./-])pillow(?:\[[^\]\s]*\])?(?![\w}./-])", re.I)


def _resolves_dependencies(command):
    """Whether this install can land a Pillow it was not asked for."""
    return not _NO_DEPS_RE.search(command) or bool(_PILLOW_REQ_RE.search(command))


def _upgrades_torchvision(source):
    return [
        command for command in _pip_commands(source)
        if ni.is_upgrading(command)
        and _TORCHVISION_RE.search(command)
        and _resolves_dependencies(command)
    ]


def _torchvision_upgrades_by_cell(path):
    """(command, cell) pairs. A placeholder resolves in its cell, not in the
    notebook: two notebooks hold two such installs in separate cells, and
    notebook scope lets one cell's exact `get_pil` answer for the other's.

    `%%bash` cells are deliberately not scanned, which excludes the 152 AMD
    ROCm installs that recognising `-U` makes visible. Not because they leave
    Pillow alone (a `--dry-run` there plans `pillow==11.0.0 -> 12.2.0` just like
    Colab) but because the mismatch also needs `PIL` in `sys.modules` without
    `PIL.Image`, the state Colab's kernel is in before cell 1
    (`google/colab/_reprs.py` line 14 is a bare `import PIL as pil`). The AMD
    notebooks run on bare JupyterLab with no PIL resident, keep the `%%bash`
    install at code cell 0 where a subprocess shell cannot import PIL into the
    kernel, and make every later upgrade `--no-deps`. The generator draws the
    same line via `update_all_notebooks._AMD_INSTALL_PACKAGE_IGNORE`.

    A cell-type predicate rather than a notebook list, so a rename cannot widen
    it and an AMD install moved into a Python cell is back in scope. The tests
    below hold it to the cell-0 ROCm install with no PIL above it.
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


# The pin has to name the version ALREADY IMPORTED into the kernel, which only
# a read off the live module can promise. An exact literal is not the same
# thing: against a kernel holding Pillow 12, `pillow==11.3.0` is still a pin,
# and pip still swaps the on-disk Pillow beneath the cached `PIL` -- downwards
# this time, but the same replacement, and the same broken `_imaging`. So a
# literal satisfies the letter of "pinned" while reproducing the exact failure
# this gate exists to stop. Only `pillow=={PIL.__version__}` leaves pip with
# the requirement already satisfied and nothing to install.
#
# This is what the gate's own message has always prescribed, and what all 84
# Pillow pins under nb/ are already written as, so the narrower rule rejects
# nothing the repo does today.
#
# `(?i:pillow)` because pip folds distribution names per PEP 503, while
# `PIL.__version__` stays case-sensitive: it is a Python attribute, and
# `pil.__version__` is a NameError rather than a quieter spelling.
_PILLOW_PIN = r"(?i:pillow)\s*==\s*\{\s*PIL\.__version__\s*\}"
_PILLOW_PIN_RE = re.compile(_PILLOW_PIN)


def _pins_pillow(command, source):
    """Whether THIS command pins Pillow, not whether the notebook mentions it.

    Dropping `{get_pil}` from the install while leaving `get_pil = ...` a few
    lines above still satisfies a whole-notebook search, and the upgrade
    resolves a fresh Pillow exactly as before. So a placeholder counts only
    when the notebook assigns that name a pin READ OFF THE LOADED MODULE: the
    `get_pil = "pillow"` fallback, a range like `pillow>=11`, and a literal
    `pillow==11.3.0` are each the failure rather than a pin against it. See
    `_PILLOW_PIN` on why an exact literal is not enough.

    Comments go first, because commenting a pin out is how one gets dropped and
    the pattern is anchored on the name, not the start of a line: so
    `# get_pil = f'pillow=={PIL.__version__}'` read as live and left the gate
    green over an install reaching an undefined placeholder.
    """
    command, source = ni.strip_comments(command), ni.strip_comments(source)
    if _PILLOW_PIN_RE.search(command):
        return True
    for name in _RE_PLACEHOLDER.findall(command):
        # `[^\n;]` stops at the end of the statement. These notebooks chain both
        # pins on one line, so allowing `;` let `{get_numpy}` borrow the `pillow`
        # belonging to `get_pil` and every unpinned command read as pinned.
        # The name must open a statement and take a single `=`, or a comparison
        # (`if get_pil == "pillow==11.3.0":`) and a keyword argument both read as
        # live pins. `_ASSIGNMENT_RE` in the vLLM gate draws the same line.
        assignment = re.search(
            r"(?:^|[;:])\s*" + re.escape(name) + r"\s*=(?!=)\s*[^\n;]*" + _PILLOW_PIN,
            source, re.M,
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


@pytest.mark.parametrize("name", ["pillow", "Pillow", "PILLOW"])
def test_a_pin_written_into_the_command_counts(name):
    """Spelled inline rather than through a placeholder, and in any case pip
    accepts, since PEP 503 folds the distribution name."""
    command = f"!uv pip install --upgrade {name}=={{PIL.__version__}} torchvision"
    assert _pins_pillow(command, command)


@pytest.mark.parametrize("pin", ["pillow==11.3.0", "Pillow==11.3.0"])
def test_an_exact_literal_is_not_a_pin_against_the_loaded_pillow(pin):
    """The version that matters is the one already in `sys.modules`, and a
    literal only equals it by luck. Against a kernel holding Pillow 12,
    `pillow==11.3.0` is exact and still makes pip replace Pillow underneath the
    cached `PIL`, which is the mismatch verbatim. Accepting it let a command
    reproduce the failure while reading as pinned."""
    command = f"!uv pip install --upgrade {pin} torchvision"
    assert _upgrades_torchvision(command) == [command]
    assert not _pins_pillow(command, command)


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
    [
        'get_pil = "pillow"',
        "get_pil = 'pillow>=11.3.0'",
        'get_pil = "Pillow"',
        'get_pil = "pillow==11.3.0"',
    ],
    ids=["unversioned", "range", "unversioned-capitalised", "literal"],
)
def test_a_placeholder_that_is_not_read_off_the_loaded_pillow_does_not_count(
    definition,
):
    """These notebooks already carry `except: get_pil = "pillow"`, so the
    unversioned spelling is one deleted line away -- and passing it to an
    `--upgrade` install is the failure, not a pin against it. A literal exact
    version is the same failure wearing a pin's clothes: it only matches the
    loaded Pillow by luck, and pip replaces the package whenever it does not."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    assert not _pins_pillow(command, definition + "\n" + command)


def test_the_real_fallback_line_does_not_hide_the_pinned_one():
    """The notebook defines the name twice, pinned in the `try` and unversioned
    in the `except`. The pinned branch runs when PIL is loaded, which is exactly
    when the mismatch can happen, so it must count; requiring the FIRST
    assignment to be the pinned one would be a coin flip on source order."""
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
    ):
        assert _upgrades_torchvision(command) == [command], command
    # The AMD spelling carries no `!`, because it only ever runs in a `%%bash`
    # cell; scanned as a bare line it is prose, not a command.
    rocm = "uv pip install --system -U --force-reinstall torch torchvision"
    assert _upgrades_torchvision(f"%%bash\n{rocm}") == [rocm]
    assert _upgrades_torchvision(rocm) == []


def test_a_flag_that_merely_contains_u_is_not_an_upgrade():
    """Reading `--index-url` as an upgrade would drag unrelated installs into
    scope."""
    for command in (
        '!uv pip install torchvision --index-url "$URL"',
        "!uv pip install -qqq torchvision",
    ):
        assert _upgrades_torchvision(command) == [], command


def test_a_no_deps_install_cannot_pull_a_new_pillow():
    """Pillow only arrives as torchvision's dependency, so `--no-deps` leaves it
    alone however hard the command replaces torchvision. Two notebooks reinstall
    a pinned torchvision that way, and demanding a Pillow pin there asks for a
    requirement the install would ignore."""
    for command in (
        '!pip install --force-reinstall --no-deps "torchvision==0.26.0"',
        "!uv pip install -U --no-deps torchvision",
    ):
        assert ni.is_upgrading(command), command
        assert _upgrades_torchvision(command) == [], command
    # Naming Pillow spends the exemption: it is a requirement of its own now,
    # and an unpinned one resolves from the index like any other.
    named = "!uv pip install -U --no-deps pillow torchvision"
    assert _upgrades_torchvision(named) == [named]
    assert not _pins_pillow(named, named)
    pinned = "!uv pip install -U --no-deps pillow=={PIL.__version__} torchvision"
    assert _upgrades_torchvision(pinned) == [pinned]
    assert _pins_pillow(pinned, pinned)


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
    """How a pin actually gets dropped. The pattern is anchored on the name, not
    the start of a line, so a commented-out `get_pil` read as live while the
    install still reached `{get_pil}` undefined."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    for definition in (
        '# get_pil = "pillow==11.3.0"',
        "#get_pil = f'pillow=={PIL.__version__}'",
        '    # get_pil = "pillow==11.3.0"',
    ):
        assert not _pins_pillow(command, definition + "\n" + command), definition
    # The spelling an edit leaves behind: the live assignment is the
    # unversioned fallback and only the trailing comment carries the pin.
    trailing = 'get_pil = "pillow"  # was pillow==11.3.0'
    assert not _pins_pillow(command, trailing + "\n" + command)
    # A literal pin commented out of the command itself, likewise.
    assert not _pins_pillow(
        "!uv pip install --upgrade torchvision  # pillow==11.3.0",
        "!uv pip install --upgrade torchvision  # pillow==11.3.0",
    )


def test_non_executable_text_is_not_collected_as_a_command():
    """87 lines in the tree mention `pip install` in prose. A Python cell needs
    `!`/`%` for a shell command, so collecting the rest reports an unpinned
    upgrade in a notebook that never runs one."""
    for source in (
        "# !uv pip install --upgrade torchvision",
        'print("uv pip install --upgrade torchvision")',
        "    pip install --upgrade torchvision",
        "!uv pip install --upgrade unsloth  # not torchvision",
    ):
        assert _upgrades_torchvision(source) == [], source
    # ...while the executable spellings are still collected, `%%bash` included.
    assert _upgrades_torchvision("!uv pip install --upgrade torchvision")
    assert _upgrades_torchvision("%pip install --upgrade torchvision")
    assert _upgrades_torchvision("%%bash\nuv pip install --system -U torchvision")


def test_a_case_variant_torchvision_is_still_torchvision():
    """PEP 503 lowercases project names, so `TorchVision` resolves the same
    wheel and drags the same Pillow in. Excluding it drops the command out of
    the gate, the one direction that fails quietly. The vLLM gate already reads
    its own requirement case-insensitively for this reason."""
    for command in (
        "!uv pip install --upgrade TorchVision",
        "!uv pip install --upgrade unsloth TORCHVISION",
    ):
        assert _upgrades_torchvision(command) == [command], command
        assert not _pins_pillow(command, command)
    # The pin still counts whatever case either name is written in.
    pinned = "!uv pip install --upgrade Pillow=={PIL.__version__} TorchVision"
    assert _upgrades_torchvision(pinned) == [pinned]
    assert _pins_pillow(pinned, pinned)


def test_upgrade_strategy_is_not_an_upgrade():
    """`--upgrade-strategy` only picks how dependencies resolve; it upgrades
    nothing on its own, so reading it as one drags ordinary installs into both
    gates. See pip's install docs on `-U` versus `--upgrade-strategy`."""
    assert not ni.is_upgrading("!pip install --upgrade-strategy eager torchvision")
    assert _upgrades_torchvision(
        "!pip install --upgrade-strategy only-if-needed torchvision") == []
    # The real flag, alone or paired with the strategy, still counts.
    assert ni.is_upgrading("!pip install --upgrade torchvision")
    assert ni.is_upgrading(
        "!pip install --upgrade --upgrade-strategy eager torchvision")


def test_a_reinstall_is_a_replacing_install():
    """`--force-reinstall torchvision` carries no version, so pip resolves it
    from the index and reinstalls Pillow underneath it: the same swap `-U`
    makes, and the same broken `_imaging`. uv spells the flag `--reinstall`."""
    for command in (
        "!pip install --force-reinstall torchvision",
        "!uv pip install --reinstall torchvision",
    ):
        assert ni.is_upgrading(command), command
        assert _upgrades_torchvision(command) == [command], command
        assert not _pins_pillow(command, command)
    # Pinned in the same command, it is a pin rather than a resolve.
    pinned = "!pip install --force-reinstall pillow=={PIL.__version__} torchvision"
    assert _upgrades_torchvision(pinned) == [pinned]
    assert _pins_pillow(pinned, pinned)
    # `--reinstall-package` replaces one named package, not the environment.
    assert not ni.is_upgrading("!uv pip install --reinstall-package vllm torchvision")


def test_only_a_real_assignment_defines_the_pin():
    """A single unanchored `=` reads a comparison or a keyword argument as a
    live pin, so the gate stays green over a name nothing ever assigns. The
    vLLM gate's `_ASSIGNMENT_RE` already requires both halves."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    for source in (
        'if get_pil == "pillow==11.3.0":',
        'assert get_pil == "pillow==11.3.0"',
        'download(get_pil="pillow==11.3.0")',
        'd = {"get_pil": "pillow==11.3.0"}',
    ):
        assert not _pins_pillow(command, source + "\n" + command), source
    # ...without disturbing the spellings the notebooks actually use, where the
    # assignment opens the line or follows the `try:`/`;` of a chained one.
    for source in (
        _DEFINES_PIL,
        "get_pil = f'pillow=={PIL.__version__}'",
        "    get_pil = f'pillow=={PIL.__version__}'",
        "try: get_pil = f'pillow=={PIL.__version__}'",
    ):
        assert _pins_pillow(command, source + "\n" + command), source


def test_stripping_comments_leaves_live_code_alone():
    """A quoted `#`, such as a VCS URL's `#egg=` fragment, is part of the
    requirement; splitting on the first one would truncate a live command."""
    assert ni.strip_comment(_DEFINES_PIL) == _DEFINES_PIL
    assert ni.strip_comment('pip install "x @ git+https://h/r#egg=x"  # tail') == (
        'pip install "x @ git+https://h/r#egg=x"  '
    )
    command = "!uv pip install --upgrade {get_pil} torchvision"
    assert _pins_pillow(command, _DEFINES_PIL + "  # keep PIL in step\n" + command)


def test_a_shell_variable_is_not_read_as_a_python_placeholder():
    """`${PIL_PIN}` is a shell expansion, and reading it as `{PIL_PIN}` let a
    bash assignment satisfy a probe written for an f-string pin."""
    command = "uv pip install -U torchvision ${PIL_PIN}"
    source = 'PIL_PIN="pillow=={PIL.__version__}"\n' + command
    assert not _pins_pillow(command, source)
    assert _pins_pillow("!uv pip install -U torchvision {get_pil}", _DEFINES_PIL)


def test_a_fold_leaves_no_stray_backslash():
    """Each continuation drops its own `\\`; one left in wedges a stray token
    into the folded command."""
    text = (
        "%%bash\n"
        "uv pip install --system -U --force-reinstall \\\n"
        "    torch torchvision torchaudio triton-rocm \\\n"
        '    --index-url "$PYTORCH_INDEX_URL"\n'
    )
    assert _pip_commands(text) == [
        "uv pip install --system -U --force-reinstall torch torchvision "
        'torchaudio triton-rocm --index-url "$PYTORCH_INDEX_URL"'
    ]
