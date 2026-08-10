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

"""A notebook that imports the stack has to install it first.

Colab and Kaggle ship neither unsloth nor trl, and they ship whatever
transformers their base image happens to carry that week. So a notebook under
`nb/` that says `from unsloth import FastLanguageModel` and never runs a
`pip install` cannot run anywhere except a machine that was already set up by
hand.

`Synthetic_Data_Hackathon.ipynb` shipped in that state and died on

    ModuleNotFoundError: No module named 'trl'

raised from inside `unsloth/models/_utils.py`, on a Colab whose base image had
resolved transformers to 5.13.1. It is one of the notebooks generation skips
(`DONT_UPDATE_EXCEPTIONS`), so the install cell every other notebook receives
from `update_all_notebooks.py` was never written into it, and its own AMD
variant did have one, because the AMD generator injects the ROCm install cell
itself. Nothing compared the two.

The check is deliberately coarse -- SOME pip install line, BEFORE the first
import of the stack -- because the pinning of individual packages is already
covered by `test_notebook_pin_consistency.py`. What was missing was a check
that there is anything there at all.

Scoped to `nb/` and `kaggle/`, the trees a reader opens. `original_template/`
carries `# Placeholder` cells that the generator turns into the install cells,
so it has nothing to assert against.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIRS = (REPO_ROOT / "nb", REPO_ROOT / "kaggle")

# Everything a notebook must not assume is already present. torch is left out
# on purpose: Colab and Kaggle really do ship it, and every install cell here
# reads its version rather than installing it.
STACK = ("unsloth", "unsloth_zoo", "trl", "transformers", "peft")

_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(" + "|".join(STACK) + r")\b", re.MULTILINE)
# `!pip install`, `%pip install`, `!uv pip install --system`, and the same
# lines inside a `%%bash` cell.
_INSTALL = re.compile(r"(?:uv\s+)?pip\s+install\b")


def _first_code_cell(path: Path, pattern: re.Pattern):
    """Index of the first code cell matching `pattern`, or None."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if pattern.search("".join(cell.get("source", []))):
            return index
    return None


def _imported_names(path: Path) -> set[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code")
    return set(_IMPORT.findall(code))


def _notebooks():
    found = []
    for directory in NB_DIRS:
        found += sorted(directory.glob("*.ipynb"))
    return found


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_notebook_that_imports_the_stack_installs_it(path):
    imported = _imported_names(path)
    if not imported:
        return          # a notebook that imports none of it needs no cell
    install_at = _first_code_cell(path, _INSTALL)
    assert install_at is not None, (
        f"{path.relative_to(REPO_ROOT)} imports "
        + ", ".join(sorted(imported))
        + " but has no pip install cell anywhere, so on Colab or Kaggle it "
          "stops on ModuleNotFoundError."
    )
    import_at = _first_code_cell(path, _IMPORT)
    assert install_at < import_at, (
        f"{path.relative_to(REPO_ROOT)} imports the stack in cell {import_at} "
        f"but does not install it until cell {install_at}; the import runs "
        "first and fails."
    )


def test_the_suite_is_looking_at_the_notebooks():
    """A glob that stopped matching would leave every case above vacuous."""
    assert len(_notebooks()) > 400


def test_the_gate_would_have_caught_the_notebook_that_shipped_without_a_cell(
        tmp_path):
    """The exact shape of the bug: imports, no install line."""
    notebook = tmp_path / "x.ipynb"
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# title"]},
        {"cell_type": "code", "metadata": {}, "outputs": [],
         "execution_count": None,
         "source": ["from unsloth import FastLanguageModel\n",
                    "from trl import SFTTrainer\n"]},
    ]}), encoding="utf-8")
    assert _imported_names(notebook) == {"unsloth", "trl"}
    assert _first_code_cell(notebook, _INSTALL) is None


def test_an_install_cell_ahead_of_the_import_satisfies_the_gate(tmp_path):
    notebook = tmp_path / "x.ipynb"
    notebook.write_text(json.dumps({"cells": [
        {"cell_type": "code", "metadata": {}, "outputs": [],
         "execution_count": None,
         "source": ["%%capture\n", "!pip install unsloth\n"]},
        {"cell_type": "code", "metadata": {}, "outputs": [],
         "execution_count": None,
         "source": ["from unsloth import FastLanguageModel\n"]},
    ]}), encoding="utf-8")
    assert _first_code_cell(notebook, _INSTALL) == 0
    assert _first_code_cell(notebook, _IMPORT) == 1


@pytest.mark.parametrize("line", [
    "!pip install unsloth",
    "%pip install unsloth",
    "!uv pip install -qqq unsloth",
    "uv pip install --system -U unsloth",
    "    pip install bitsandbytes",
], ids=["bang", "magic", "uv", "uv-system-bash", "bare-in-bash"])
def test_the_ways_these_notebooks_really_install_all_count(line):
    assert _INSTALL.search(line)


@pytest.mark.parametrize("line", [
    "# pip installation is described at unsloth.ai/docs",
    "print('run pip freeze to see versions')",
], ids=["prose", "unrelated"])
def test_prose_about_pip_is_not_an_install_line(line):
    assert not _INSTALL.search(line)
