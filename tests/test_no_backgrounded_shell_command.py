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

`ipykernel`'s `ZMQInteractiveShell.system_piped` raises `OSError("Background
processes not supported.")`, so on Kaggle, plain Jupyter or papermill the cell
cannot run at all. `subprocess.Popen` does the same job on every kernel.

The `&` usually sits several continuations below the `!`, so the scan folds them
first; per-line matching misses the vLLM launch. Every notebook root CI watches
is scanned, since `original_template/` feeds `nb/` on the next regeneration.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "notebooks-tests-ci.yml"

# Every notebook root the workflow watches. `molab/` and `python_scripts/` hold
# no notebook today, but are listed so one appearing there is not exempt.
ROOTS = ("nb", "original_template", "kaggle", "molab", "python_scripts")

# Known offenders, keyed by repo-relative path (a template and its copies share
# a basename) and pinned to the exact command (a name or marker match would hide
# a second backgrounded command in the same notebook). A new offender must fail
# rather than join the list, which should be emptied when #317 merges.
_SGLANG = (
    "unslothai/notebooks#317",
    "!nohup python -m sglang.launch_server --model-path unsloth/gemma-3n-E2B-it"
    " --attention-backend fa3 --port 8000 > sglang.log &",
)
KNOWN = {
    "original_template/Gemma3N_(2B)-Inference.ipynb": _SGLANG,
    "nb/Gemma3N_(2B)-Inference.ipynb": _SGLANG,
    "nb/Kaggle-Gemma3N_(2B)-Inference.ipynb": _SGLANG,
    "nb/AMD-Gemma3N_(2B)-Inference.ipynb": _SGLANG,
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
            # `[^&]&` so `a && b` does not match.
            if re.search(r"[^&]&\s*$", command):
                # Whole, not truncated: KNOWN compares exactly and the sglang
                # line is 125 characters.
                found.append(command)
    return found


def _for_message(commands):
    """Shortened only for the failure text, never for comparison."""
    return [c if len(c) <= 120 else c[:117] + "..." for c in commands]


def _collect(repo_root):
    """Every notebook under every watched root."""
    found = []
    for root in ROOTS:
        directory = repo_root / root
        if directory.is_dir():
            found += sorted(directory.rglob("*.ipynb"))
    return found


def _rel(path):
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


_NOTEBOOKS = _collect(REPO_ROOT)


def _unaccounted(name, commands):
    """The commands KNOWN does not account for, so anything else in a
    grandfathered notebook still fails. The tests guarding the filter call it
    rather than restating it, or they stay green against any rewrite of it.
    """
    if name not in KNOWN:
        return list(commands)
    _pr, allowed = KNOWN[name]
    return [command for command in commands if command != allowed]


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=_rel)
def test_no_notebook_backgrounds_a_shell_command(path):
    name = _rel(path)
    found = _unaccounted(name, _backgrounded(path))
    if name in KNOWN and not found:
        pytest.skip(f"known, fixed by {KNOWN[name][0]}")
    assert not found, (
        f"{name} ends a `!` command with `&`: {_for_message(found)}. "
        f"ipykernel raises OSError on that, so the cell cannot run outside "
        f"Colab. Use subprocess.Popen"
    )


def _workflow_notebook_dirs():
    """Top-level directories the CI workflow watches, from its own `paths:`."""
    text = WORKFLOW.read_text(encoding="utf-8")
    return sorted({m for m in re.findall(r"^\s+- '([^'/]+)/\*\*'", text, re.M)})


def test_the_scan_covers_every_watched_directory_that_holds_notebooks():
    """A root left out is scanned by nothing, and an empty collection makes the
    check above assert about no files."""
    watched = _workflow_notebook_dirs()
    assert len(watched) >= 5, watched
    missing = [
        directory
        for directory in watched
        if directory not in ROOTS and any((REPO_ROOT / directory).rglob("*.ipynb"))
    ]
    assert not missing, f"the workflow watches these notebook dirs, add to ROOTS: {missing}"
    scanned = {_rel(path).split("/")[0] for path in _NOTEBOOKS}
    assert {"nb", "original_template", "kaggle"} <= scanned, scanned
    assert len(_NOTEBOOKS) >= 500, len(_NOTEBOOKS)


def _stale(known, repo_root):
    """Entries the repo has outgrown: the notebook stopped offending or is gone.

    #317 deletes `AMD-Gemma3N_(2B)-Inference` rather than fixing it, and an
    `is_file()` guard alone reads that as "nothing to check", keeping the entry
    forever.
    """
    stale = []
    for name, (_pr, allowed) in known.items():
        path = repo_root / name
        if not path.is_file():
            stale.append(name)
        elif allowed not in _backgrounded(path):
            stale.append(name)
    return stale


def test_the_known_list_still_describes_reality():
    """A stale entry hides the next regression in that notebook."""
    stale = _stale(KNOWN, REPO_ROOT)
    assert not stale, f"fixed, edited or deleted, so update KNOWN: {stale}"


def test_a_deleted_notebook_does_not_stay_grandfathered(tmp_path):
    """An entry whose notebook is gone must be reported."""
    assert _stale(KNOWN, tmp_path) == list(KNOWN)


def test_a_present_and_still_offending_notebook_is_not_reported():
    """So the check cannot pass by calling everything stale."""
    name = next(iter(KNOWN))
    assert name not in _stale({name: KNOWN[name]}, REPO_ROOT)


def test_a_second_offender_in_a_known_notebook_is_not_hidden():
    """A name match skipped the notebook, hiding anything beside the sglang command."""
    name, (_pr, _allowed) = next(iter(KNOWN.items()))
    commands = _backgrounded(REPO_ROOT / name) + ["!python other_server.py &"]
    assert _unaccounted(name, commands) == ["!python other_server.py &"]


def test_a_second_sglang_command_is_not_hidden():
    """A marker like `sglang.launch_server` would filter this one out too."""
    name, (_pr, _allowed) = next(iter(KNOWN.items()))
    second = "!nohup python -m sglang.launch_server --port 8001 > two.log &"
    commands = _backgrounded(REPO_ROOT / name) + [second]
    assert _unaccounted(name, commands) == [second]


def test_editing_the_grandfathered_command_is_reported():
    """The entry pins one command, so changing even a flag has to fail."""
    name, (_pr, allowed) = next(iter(KNOWN.items()))
    edited = allowed.replace("--port 8000", "--port 8001")
    assert edited != allowed
    assert _unaccounted(name, [edited]) == [edited]


def test_the_known_command_survives_extraction_whole():
    """It is 125 characters and the scan used to clip at 120, so every exact
    comparison failed open."""
    _name, (_pr, allowed) = next(iter(KNOWN.items()))
    assert len(allowed) > 120
    for name in KNOWN:
        if (REPO_ROOT / name).is_file():
            assert allowed in _backgrounded(REPO_ROOT / name)


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
