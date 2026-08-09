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

torchvision depends on Pillow, so `uv pip install --upgrade ... torchvision`
resolves a newer Pillow and swaps it in underneath a kernel that has already
imported PIL. The Python half of the package is then the new version while the
compiled `_imaging` extension on the path is the old one, and PIL says so:

    /usr/local/lib/python3.12/dist-packages/PIL/Image.py:116: RuntimeWarning:
    The _imaging extension was built for another version of Pillow or PIL

torchvision's own `import PIL` then fails, and because `unsloth_zoo`
translates that into a hard error telling the user to reinstall Pillow and
restart, `from unsloth import FastLanguageModel` never returns. The notebook
dies on its first real cell with nothing trained.

Found on a Colab L4: `Advanced_Llama3_1_(3B)_GRPO_LoRA` failed exactly that
way while `Advanced_Llama3_2_(3B)_GRPO_LoRA`, which already pinned Pillow,
passed on the same worker minutes later. 45 of the 46 notebooks with such an
install already carried the pin; this test stops the 46th recurring.

The scan folds backslash continuations before matching. The install that broke
puts `--upgrade` and `torchvision` on different physical lines, so a per-line
match reports zero offenders and the check silently passes.

Both spellings of the flag count, via the shared
`notebook_inventory.UPGRADE_FLAG_RE`. Matching only the long `--upgrade`
exempted every install written `-U`, which is 152 of them.

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
            # Strip the `\` from EVERY continuation, not just the first. Left
            # in, a folded command reads `... triton-rocm \ --index-url ...`
            # and the stray token sits between two things a pattern may want
            # to match across.
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
    """(command, cell) pairs. The cell, not the notebook, is the scope a
    placeholder resolves in.

    `Meta-Synthetic-Data-Llama3.1_(8B).ipynb` and
    `Meta_Synthetic_Data_Llama3_2_(3B).ipynb` each hold two such installs, in
    separate cells. Resolving against the concatenated notebook lets one
    cell's exact `get_pil` answer for the other cell's unpinned one, so
    dropping the pin from either passes.

    `%%bash` cells are deliberately not scanned, which excludes the 152 AMD
    ROCm installs

        uv pip install --system -U --force-reinstall \\
            torch torchvision torchaudio triton-rocm \\
            --index-url "$PYTORCH_INDEX_URL"

    that recognising `-U` above makes visible for the first time. Not because
    the upgrade is harmless in itself: the ROCm index does carry Pillow (to
    12.2.0), and a `uv pip install --dry-run` of that exact command in an
    environment holding Pillow 11.0.0 plans `- pillow==11.0.0 /
    + pillow==12.2.0`. It moves Pillow just like the Colab one.

    It is out of scope because the breakage needs a second ingredient the AMD
    cells cannot supply. Pillow's `Image.py` compares `__version__`, read from
    an ALREADY-CACHED `PIL/__init__.py`, against the freshly loaded
    `_imaging.PILLOW_VERSION`, so the mismatch fires only when `PIL` is in
    `sys.modules` and `PIL.Image` is not. Reproduced: with nothing imported the
    upgrade is clean, with `PIL` and `PIL.Image` both loaded the kernel stays
    coherently on the old version, and only a bare `import PIL` before the
    upgrade produces the `Image.py:116` RuntimeWarning above.

    Colab's kernel is in exactly that state before the first cell runs. Its
    kernel class is `google.colab._kernel.Kernel`, loading it runs
    `google/colab/__init__.py`, which imports `_reprs`, whose line 14 is a bare
    `import PIL as pil` -- `pil.Image` is touched only inside `_image_repr`.
    Colab reports it too, in the "previously imported in this runtime: [PIL]"
    banner `google/colab/_pip.py` builds from `set(installed) & sys.modules`.

    The AMD notebooks run on a bare JupyterLab in a ROCm container instead,
    where a first-cell `sys.modules` dump holds no PIL: neither ipykernel nor
    IPython imports it, and `%matplotlib inline` is not on by default (and
    would load `PIL.Image` too, the safe state). No AMD notebook imports PIL,
    the `%%bash` cell is code cell 0 and runs in a subprocess shell so it
    cannot put PIL into the kernel on its way past, and every later AMD
    upgrade is `--no-deps`. Every notebook the gate does police creates the
    state itself, one line above the install:
    `try: import PIL; get_pil = f'pillow=={PIL.__version__}'`.

    The generator already draws the same line: `_AMD_INSTALL_PACKAGE_IGNORE` in
    update_all_notebooks.py lists "pillow" and "pil", so a Colab pin is
    deliberately not propagated into the AMD variant -- the ROCm cell owns that
    half of the stack, as it does for torch and torchao.

    The exclusion is a cell-type predicate rather than a list of notebooks, so
    a rename cannot widen it and an AMD install that moves into a Python cell
    is back in scope automatically. The tests below hold it to the ROCm install
    at cell 0 with no PIL above it, so it cannot quietly grow.
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


# `(?<!\$)` so a shell `${VAR}` is not read as a Python placeholder. The pin is
# interpolated by Python, and letting `${PIL_PIN}` match meant a bash variable
# could satisfy the gate through an assignment probe written for f-strings.
_RE_PLACEHOLDER = re.compile(r"(?<!\$)\{(\w+)\}")


def _pins_pillow(command, source):
    """Whether THIS command pins Pillow, not whether the notebook mentions it.

    Asking the whole notebook is the failure this gate exists to prevent, one
    level up: drop `{get_pil}` from the install but leave `get_pil = ...`
    defined a few lines above, and a whole-notebook search still says yes while
    the upgrade resolves a fresh Pillow exactly as before.

    The pin is usually interpolated, so a placeholder counts only when the
    notebook assigns that name an EXACT pin. `get_pil = "pillow"` is what these
    notebooks fall back to when PIL is not importable, and passing that to an
    `--upgrade` install resolves a fresh Pillow -- the failure itself, not a pin
    of it. A range such as `pillow>=11` is no better.
    """
    if re.search(r"pillow\s*==", command, re.I):
        return True
    for name in _RE_PLACEHOLDER.findall(command):
        # `[^\n;]`, so the match cannot run past the end of the statement. These
        # notebooks put both pins on one line:
        #   try: import numpy, PIL; get_numpy = f'numpy=={numpy.__version__}'; get_pil = f'pillow=={PIL.__version__}'
        # and allowing `;` lets `get_numpy` match the `pillow` belonging to
        # `get_pil`. Every unpinned command then looks pinned through
        # `{get_numpy}`, which is the whole bug this function was rewritten to
        # catch. Caught by sabotage: dropping `{get_pil}` from the real notebook
        # left the suite green.
        assignment = re.search(
            rf"\b{re.escape(name)}\s*=\s*[^\n;]*pillow\s*==", source, re.I
        )
        if assignment:
            return True
    return False


_NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb")) if NB_DIR.is_dir() else []


def _write_notebook(tmp_path, cells):
    """A throwaway .ipynb holding `cells` as code cells, so the scope rules can
    be exercised on a notebook instead of on a bare string."""
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
    """The install that broke splits `--upgrade` from `torchvision` across a
    backslash continuation. Matching per physical line finds nothing."""
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
    """Only torchvision drags Pillow in. Flagging every `--upgrade` would make
    the check noise, and noise gets skipped."""
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
    """The exact regression this gate exists for. Someone drops `{get_pil}` from
    the install and leaves the assignment above it; a whole-notebook search says
    Pillow is pinned while the upgrade resolves a fresh one just as before."""
    command = "!uv pip install -qqq --upgrade unsloth torchvision"
    source = _DEFINES_PIL + "\n" + command
    assert re.search(r"pillow\s*==", source, re.I), "the decoy must look convincing"
    assert not _pins_pillow(command, source)


def test_a_placeholder_naming_something_else_does_not_count():
    """`{get_numpy}` is interpolated into the same command and must not be read
    as a Pillow pin."""
    command = "!uv pip install --upgrade {get_numpy} torchvision"
    source = 'get_numpy = f"numpy=={numpy.__version__}"\n' + command
    assert not _pins_pillow(command, source)


@pytest.mark.parametrize(
    "definition",
    ['get_pil = "pillow"', "get_pil = 'pillow>=11.3.0'", 'get_pil = "Pillow"'],
    ids=["unversioned", "range", "unversioned-capitalised"],
)
def test_a_placeholder_that_is_not_an_exact_pin_does_not_count(definition):
    """`{get_pil}` in the command is only half of it. These notebooks already
    carry `except: get_pil = "pillow"` for the case where PIL is not importable,
    so the unversioned spelling is one deleted line away -- and passing it to an
    `--upgrade` install beside torchvision resolves a newer Pillow, which is the
    failure this gate exists to stop rather than a pin against it."""
    command = "!uv pip install --upgrade {get_pil} torchvision"
    assert not _pins_pillow(command, definition + "\n" + command)


def test_the_real_fallback_line_does_not_hide_the_pinned_one():
    """The notebook defines the name twice, pinned then unversioned:

        try: import PIL; get_pil = f'pillow=={PIL.__version__}'
        except: get_pil = "pillow"

    The pinned assignment is what runs when PIL is loaded, which is exactly when
    the mismatch can happen, so this must still count. Requiring the FIRST
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
    """A notebook may run several upgrades. Checking only the first would let a
    later unpinned one through."""
    pinned = "!uv pip install --upgrade {get_pil} torchvision"
    unpinned = "!uv pip install --upgrade torchvision"
    source = _DEFINES_PIL + "\n" + pinned + "\n" + unpinned
    commands = _upgrades_torchvision(source)
    assert len(commands) == 2
    assert [c for c in commands if not _pins_pillow(c, source)] == [unpinned]


def test_at_least_one_notebook_actually_exercises_the_check():
    """A glob that matched nothing, or a fold that silently stopped working,
    would leave every parametrised case skipped and the suite green."""
    exercised = [p.name for p in _NOTEBOOKS if _upgrades_torchvision(_code(p))]
    assert len(exercised) >= 40, (
        f"only {len(exercised)} notebooks reached the assertion; the scan is "
        f"probably broken rather than the repo suddenly clean"
    )


def test_every_torchvision_upgrade_is_still_reached_per_cell(tmp_path):
    """Splitting the scan per cell must not lose commands. Two notebooks hold
    two such installs each, so the per-cell pass has to find more commands
    than there are notebooks carrying them."""
    per_cell = sum(len(_torchvision_upgrades_by_cell(p)) for p in _NOTEBOOKS)
    whole = sum(len(_upgrades_torchvision(_code(p))) for p in _NOTEBOOKS)
    assert per_cell == whole >= 40, (per_cell, whole)


def test_one_cells_pin_does_not_answer_for_another_cells_command():
    """Two install cells, one pinned and one not. Resolving placeholders
    against the whole notebook lets the pinned cell's `get_pil` satisfy the
    unpinned cell, which is how a real notebook could drop the pin from one
    of its two installs and stay green."""
    command = "!uv pip install --upgrade unsloth {get_pil} torchvision"
    pinned_cell = "try: import PIL; get_pil = f'pillow=={PIL.__version__}'\n" + command
    # The fallback these notebooks use when PIL is not importable. Passed to an
    # `--upgrade` install it resolves a fresh Pillow, which is the failure.
    unpinned_cell = 'get_pil = "pillow"\n' + command
    notebook = pinned_cell + "\n" + unpinned_cell
    # Whole-notebook scope: the pinned cell answers for the unpinned one.
    assert _pins_pillow(command, notebook)
    # Cell scope: each cell answers for itself.
    assert _pins_pillow(command, pinned_cell)
    assert not _pins_pillow(command, unpinned_cell)


def test_a_sibling_pin_on_the_same_line_does_not_count():
    """These notebooks define both pins in one statement chain:

        try: import numpy, PIL; get_numpy = f'numpy=={numpy.__version__}'; get_pil = f'pillow=={PIL.__version__}'

    so a pattern that runs to end of LINE lets `{get_numpy}` borrow the `pillow`
    that belongs to `get_pil`, and every unpinned command reads as pinned."""
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
    """`uv pip install --help`: `-U, --upgrade` is "Allow package upgrades".
    A plain `"--upgrade" in command` test read only the long spelling, so every
    install written `-U` was exempt -- 152 of them in the tree."""
    for command in (
        "!uv pip install -U torchvision",
        "!uv pip install -qU unsloth torchvision",
        "uv pip install --system -U --force-reinstall torch torchvision",
    ):
        assert _upgrades_torchvision(command) == [command], command


def test_a_flag_that_merely_contains_u_is_not_an_upgrade():
    """`--force-reinstall` and `--index-url` reinstall from a named index.
    Reading either as an upgrade would drag unrelated installs into scope."""
    for command in (
        '!uv pip install --force-reinstall torchvision --index-url "$URL"',
        "!uv pip install -qqq torchvision",
    ):
        assert _upgrades_torchvision(command) == [], command


def test_a_short_flag_upgrade_in_a_python_cell_is_still_caught(tmp_path):
    """The scope below drops `%%bash` cells. This is the other half of that
    decision: a `-U torchvision` install in an ordinary Python cell, the kind
    the gate exists for, must still be collected and must still be required to
    pin. Without this the `-U` fix would be scoped away to nothing."""
    unpinned = "!uv pip install -qU unsloth torchvision"
    path = _write_notebook(tmp_path, [unpinned])
    found = _torchvision_upgrades_by_cell(path)
    assert [c for c, _cell in found] == [unpinned]
    assert not _pins_pillow(*found[0])


def test_a_shell_magic_install_is_out_of_scope(tmp_path):
    """A `%%bash` cell runs in a subprocess shell, so it cannot leave `PIL`
    resident in the kernel, which is the state the mismatch needs. See
    `_torchvision_upgrades_by_cell` for the full reasoning."""
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
    """The exclusion is justified for one command shape: the AMD ROCm torch
    stack install at code cell 0. Holding it to that shape is what stops it
    growing into a general amnesty for anything anyone puts in a `%%bash`
    cell -- add a different upgrading torchvision install there and this goes
    red rather than silently exempting it."""
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
    """Half of why the exclusion holds: nothing has run in the kernel yet, so
    there is no already-imported PIL for the upgrade to swap under."""
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
    """The other half. An `import PIL` anywhere above the install puts the
    kernel in exactly the vulnerable state, and the exclusion stops holding."""
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


def test_a_shell_variable_is_not_read_as_a_python_placeholder():
    """The pin is a Python f-string interpolation. `${PIL_PIN}` is a shell
    expansion, and reading it as `{PIL_PIN}` let a bash assignment satisfy a
    probe written for `get_pil = f'pillow=={PIL.__version__}'`."""
    command = "uv pip install -U torchvision ${PIL_PIN}"
    source = 'PIL_PIN="pillow==11.0.0"\n' + command
    assert not _pins_pillow(command, source)
    assert _pins_pillow("!uv pip install -U torchvision {get_pil}", _DEFINES_PIL)


def test_a_fold_leaves_no_stray_backslash():
    """Each continuation drops its own `\\`. Left in, the folded command reads
    `... triton-rocm \\ --index-url ...` and the stray token sits between two
    things the next pattern may want to match across."""
    text = (
        "uv pip install --system -U --force-reinstall \\\n"
        "    torch torchvision torchaudio triton-rocm \\\n"
        '    --index-url "$PYTORCH_INDEX_URL"\n'
    )
    assert _pip_commands(text) == [
        "uv pip install --system -U --force-reinstall torch torchvision "
        'torchaudio triton-rocm --index-url "$PYTORCH_INDEX_URL"'
    ]
