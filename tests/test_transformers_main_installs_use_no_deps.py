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

"""Installing transformers main without `--no-deps` re-resolves numpy.

`Falcon_H1-Alpaca.ipynb` installed transformers main as

    !pip install --force-reinstall git+https://github.com/huggingface/transformers.git

and died two cells later on a free Colab T4:

    ***** numpy was upgraded mid-session (loaded: 2.0.2, installed: 2.5.2) but
    the kernel still has the old version in memory ... Please restart your
    runtime/kernel *****

`--force-reinstall` tells pip to ignore what is installed and resolve the whole
dependency tree from scratch, so numpy comes back at the newest release even
though the numpy already present satisfied every requirement. Measured with
`pip install --dry-run --report` against a Colab-shaped environment
(numpy 2.0.2, transformers 4.56.2):

    --force-reinstall  ->  27 distributions, including numpy 2.5.2
    --no-deps          ->  1 distribution, transformers alone

Colab has imported numpy before cell 1 executes, and numpy is a C extension
that cannot be swapped under a live kernel, so no later cell can undo it. The
only fix available to a notebook is not to change numpy on disk at all. That is
the same reasoning `tests/test_qat_numpy_pin.py` already applies to the QAT
install cell, which pins numpy beside `fbgemm-gpu-genai` for this exact error.

`tests/test_transformers_main_no_deps_floors.py` covers what a `--no-deps`
install then owes: it matches on `--no-deps` and so cannot see a cell that
never passed the flag. This file is the other half, the flag itself.

Every other transformers-main install in the repo already uses `--no-deps`, for
the related reason recorded there: without it pip also replaces the CUDA torch
build the runtime already has.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIRS = ("nb", "kaggle", "original_template")
SCRIPT_DIRS = ("python_scripts", "molab")

_TRANSFORMERS_MAIN = re.compile(
    r"git\+(?:https://|ssh://git@)github\.com/huggingface/transformers"
    r"(?:\.git)?(?![\w./-])")

# `-U` is pip's spelling of `--upgrade` and does not imply `--no-deps`.
_NO_DEPS = re.compile(r"(?<![\w-])--no-deps(?![\w-])")


def _logical_lines(text):
    """Source lines with backslash continuations joined, comments dropped.

    An install command wrapped over several lines is one command to the shell,
    so `--no-deps` on any of its physical lines counts.
    """
    joined = []
    buffer = ""
    for line in text.splitlines():
        stripped = line.strip()
        if buffer == "" and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def _offending_lines(text):
    """Install lines that fetch transformers main without `--no-deps`."""
    bad = []
    for line in _logical_lines(text):
        if not _TRANSFORMERS_MAIN.search(line):
            continue
        if "install" not in line:
            continue
        if _NO_DEPS.search(line):
            continue
        bad.append(line)
    return bad


def _sources(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding = "utf-8"))
        return ["".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"]
    return [path.read_text(encoding = "utf-8")]


def _candidates():
    for directory in NOTEBOOK_DIRS:
        for path in sorted((REPO_ROOT / directory).glob("*.ipynb")):
            yield path
    for directory in SCRIPT_DIRS:
        root = REPO_ROOT / directory
        if root.is_dir():
            for path in sorted(root.glob("*.py")):
                yield path


_CASES = [(str(path.relative_to(REPO_ROOT)), path) for path in _candidates()]


def test_there_are_files_to_check():
    assert len(_CASES) > 100


def test_some_file_still_installs_transformers_main():
    """Or every check below passes by matching nothing at all."""
    hits = [name for name, path in _CASES
            if any(_TRANSFORMERS_MAIN.search(source) for source in _sources(path))]
    assert hits


@pytest.mark.parametrize("name, path", _CASES, ids = [name for name, _ in _CASES])
def test_transformers_main_is_installed_with_no_deps(name, path):
    for source in _sources(path):
        bad = _offending_lines(source)
        assert not bad, (
            f"{name} installs transformers main without --no-deps:\n"
            f"    {bad[0]}\n"
            f"pip then re-resolves the whole dependency tree and replaces "
            f"numpy, which Colab and Kaggle have already imported before the "
            f"first cell runs. The kernel keeps the old C extension, Unsloth "
            f"refuses to continue with \"numpy was upgraded mid-session\", and "
            f"no later cell can repair it. Add --no-deps and pin the floors "
            f"the base images fall short of, the way every other "
            f"transformers-main cell in this repo does")


# --- detector self-tests -------------------------------------------------

def test_detector_flags_the_line_that_broke_falcon_h1():
    line = ("!pip install --force-reinstall "
            "git+https://github.com/huggingface/transformers.git ")
    assert _offending_lines(line) == [line.strip()]


def test_detector_flags_the_uv_spelling_too():
    line = ("!uv pip install --system -qqq --upgrade --force-reinstall "
            "git+https://github.com/huggingface/transformers.git")
    assert _offending_lines(line)


def test_detector_accepts_the_liquid_lfm2_line():
    line = ("!pip install --no-deps git+https://github.com/huggingface/"
            "transformers.git # Need main branch for Liquid LFM2 models")
    assert _offending_lines(line) == []


def test_no_deps_anywhere_in_a_continued_command_counts():
    text = ("!uv pip install --system -qqq --no-deps accelerate peft \\\n"
            "    git+https://github.com/huggingface/transformers.git\n")
    assert _offending_lines(text) == []


def test_a_continued_command_without_the_flag_is_still_flagged():
    text = ("!uv pip install --system -qqq accelerate peft \\\n"
            "    git+https://github.com/huggingface/transformers.git\n")
    assert _offending_lines(text)


def test_a_commented_out_line_is_not_an_install():
    text = ("# !pip install --force-reinstall "
            "git+https://github.com/huggingface/transformers.git\n")
    assert _offending_lines(text) == []


def test_another_repository_under_huggingface_is_not_transformers():
    line = ("!pip install --force-reinstall "
            "git+https://github.com/huggingface/transformers-neuronx.git")
    assert _offending_lines(line) == []


def test_a_released_pin_is_not_a_main_install():
    assert _offending_lines('!pip install --force-reinstall transformers==4.56.2') == []


def test_no_deps_must_be_the_whole_flag():
    line = ("!pip install --no-deps-please "
            "git+https://github.com/huggingface/transformers.git")
    assert _offending_lines(line)
