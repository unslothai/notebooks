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

"""A `!` command must not end in `&`. Only Colab's kernel tolerates it.

`ipykernel`'s `ZMQInteractiveShell.system_piped` refuses a backgrounded
command outright:

    if cmd.rstrip().endswith("&"):
        raise OSError("Background processes not supported.")

so on Kaggle, plain Jupyter or papermill the cell cannot run at all. Found
twice: `Gemma3N_(2B)-Inference` starting sglang, and
`Meta-Synthetic-Data-Llama3.1_(8B)` starting vLLM, whose own comment says
"we prepend nohup and postpend & to make the Colab cell run in background".
`subprocess.Popen` does the same thing on every kernel.

The `&` is usually several backslash continuations below the `!`, so the
scan folds continuations first. Matching per physical line finds the sglang
family and misses the vLLM one entirely, which is how the second one survived.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

# Known offenders, with the PR that fixes each. A new one must fail rather
# than be added here; this list only exists so the check can land before the
# other fix does, and it should be emptied when that PR merges.
KNOWN = {
    "Gemma3N_(2B)-Inference.ipynb": "unslothai/notebooks#317",
    "Kaggle-Gemma3N_(2B)-Inference.ipynb": "unslothai/notebooks#317",
    "AMD-Gemma3N_(2B)-Inference.ipynb": "unslothai/notebooks#317",
}


def _shell_commands(source):
    """`!` commands with backslash continuations folded into one line."""
    commands, pending = [], None
    for line in source.splitlines():
        if pending is not None:
            pending += " " + line.strip()
            if not line.rstrip().endswith("\\"):
                commands.append(pending)
                pending = None
            continue
        if re.match(r"^\s*!", line):
            if line.rstrip().endswith("\\"):
                pending = line.rstrip()[:-1].strip()
            else:
                commands.append(line.strip())
    if pending is not None:
        commands.append(pending)
    return commands


def _backgrounded(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for command in _shell_commands("".join(cell.get("source", []))):
            # `[^&]&` so `a && b`, which is not backgrounding, does not match.
            if re.search(r"[^&]&\s*$", command):
                found.append(command[:120])
    return found


_NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb")) if NB_DIR.is_dir() else []


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_no_notebook_backgrounds_a_shell_command(path):
    found = _backgrounded(path)
    if path.name in KNOWN:
        pytest.skip(f"known, fixed by {KNOWN[path.name]}")
    assert not found, (
        f"{path.name} ends a `!` command with `&`: {found}. "
        f"ipykernel raises OSError on that, so the cell cannot run outside "
        f"Colab. Use subprocess.Popen"
    )


def test_the_known_list_still_describes_reality():
    """A name that no longer offends must leave the list, or it hides the next
    regression in that notebook."""
    stale = [name for name in KNOWN
             if (NB_DIR / name).is_file() and not _backgrounded(NB_DIR / name)]
    assert not stale, f"fixed, so remove from KNOWN: {stale}"


def test_a_double_ampersand_is_not_backgrounding():
    assert not _backgrounded_text("!make && make install")
    assert _backgrounded_text("!python server.py &")


def _backgrounded_text(text):
    return [c for c in _shell_commands(text) if re.search(r"[^&]&\s*$", c)]


def test_a_continuation_is_folded_before_matching():
    """The vLLM launch put its `&` eleven lines below the `!`."""
    text = "! nohup python -m vllm.entrypoints.openai.api_server \\\n" \
           "    --model x \\\n" \
           "    --port 8000 \\\n" \
           "    > vllm.log &"
    assert _backgrounded_text(text)
