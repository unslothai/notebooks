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


def _pins_pillow(source):
    """Either an explicit `pillow==` or the `PIL.__version__` idiom the other
    45 notebooks use, which pins whatever the image already shipped."""
    return bool(re.search(r"pillow\s*==", source, re.I)) or "PIL.__version__" in source


_NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb")) if NB_DIR.is_dir() else []


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_an_upgrade_install_naming_torchvision_pins_pillow(path):
    source = _code(path)
    upgrades = _upgrades_torchvision(source)
    if not upgrades:
        pytest.skip("no --upgrade install names torchvision")
    assert _pins_pillow(source), (
        f"{path.name} runs {upgrades[0][:160]!r} without pinning Pillow first. "
        f"That resolves a newer Pillow, leaves `_imaging` behind, and the next "
        f"`from unsloth import ...` dies on a PIL/torchvision mismatch. Pin it "
        f"the way the other notebooks do: "
        f"`try: import PIL; get_pil = f'pillow=={{PIL.__version__}}'`"
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


@pytest.mark.parametrize(
    "pin",
    ["pillow==11.3.0", "Pillow==11.3.0", "f'pillow=={PIL.__version__}'"],
)
def test_both_pinning_spellings_count(pin):
    assert _pins_pillow(f"!uv pip install --upgrade {pin} torchvision")


def test_an_unpinned_source_is_reported():
    """The discriminating case: without this the whole check is vacuous."""
    assert not _pins_pillow("!uv pip install -qqq --upgrade unsloth torchvision")


def test_at_least_one_notebook_actually_exercises_the_check():
    """A glob that matched nothing, or a fold that silently stopped working,
    would leave every parametrised case skipped and the suite green."""
    exercised = [p.name for p in _NOTEBOOKS if _upgrades_torchvision(_code(p))]
    assert len(exercised) >= 40, (
        f"only {len(exercised)} notebooks reached the assertion; the scan is "
        f"probably broken rather than the repo suddenly clean"
    )
