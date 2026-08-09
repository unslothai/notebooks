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

The scan folds backslash continuations before matching, since the broken install
splits `--upgrade` from `torchvision` across physical lines.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"


def _pip_commands(source):
    """`pip install` commands with backslash continuations folded into one."""
    commands, pending = [], None
    for line in source.splitlines():
        if pending is not None:
            pending += " " + line.strip()
            if not line.rstrip().endswith("\\"):
                commands.append(pending)
                pending = None
            continue
        if "pip install" in line:
            if line.rstrip().endswith("\\"):
                pending = line.rstrip()[:-1].strip()
            else:
                commands.append(line.strip())
    if pending is not None:
        commands.append(pending)
    return commands


def _code(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _upgrades_torchvision(source):
    return [
        command for command in _pip_commands(source)
        if "--upgrade" in command and "torchvision" in command
    ]


_RE_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _pins_pillow(command, source):
    """Whether THIS command pins Pillow, not whether the notebook mentions it.

    Dropping `{get_pil}` from the install while leaving `get_pil = ...` a few
    lines above still satisfies a whole-notebook search, and the upgrade
    resolves a fresh Pillow exactly as before. So a placeholder counts only
    when the notebook assigns that name an EXACT pin: the `get_pil = "pillow"`
    fallback, or a range like `pillow>=11`, is the failure rather than a pin.
    """
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


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_an_upgrade_install_naming_torchvision_pins_pillow(path):
    source = _code(path)
    upgrades = _upgrades_torchvision(source)
    if not upgrades:
        pytest.skip("no --upgrade install names torchvision")
    unpinned = [c for c in upgrades if not _pins_pillow(c, source)]
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
