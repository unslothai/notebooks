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

"""No python_scripts mirror may keep its installation section executable.

nbconvert turns `%%capture` / `!pip` into `get_ipython()` calls, which raise
NameError under plain `python`. remove_unwanted_section comments the section
out, but used to do so only when a `# Unsloth` heading closed it, so about 27
notebooks (Whisper, NeMo-Gym, the OpenEnv and gpt-oss RL notebooks, ...) and
their AMD / Kaggle / HuggingFace Course copies shipped scripts that died on
line one of the install cell.

The installation section here is read independently of the generator: from
the Installation heading, commented or not, to the next live markdown heading
of any level or the commented `# Unsloth` heading the old terminator leaves.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = sorted((REPO_ROOT / "python_scripts").glob("*.py"))

# Scripts allowed one live call in the installation section, and why. Each is
# a `%%capture` cell that both installs and does setup the rest of the script
# needs, so commenting it would drop that setup silently. Left live, the script
# fails loudly at `get_ipython` instead, and must be run as a notebook.
_ALLOWED_LIVE = {
    # Clones OpenEnv, `%cd`s into it, puts it on sys.path and binds
    # `working_directory`, which the environment launcher reads later.
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.py": 1,
    "OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.py": 1,
    "AMD-OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.py": 1,
    "AMD-OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.py": 1,
    # Same clone + `%cd` + sys.path + `working_directory` cell for textarena.
    "Openenv_wordle_grpo.py": 1,
    "AMD-Openenv_wordle_grpo.py": 1,
}

_START = re.compile(r"^(?:# )?# #+ Install(?:ation\b| Unsloth\b)")
_HEADING = re.compile(r"^# #+ \S")
_DEAD_UNSLOTH = re.compile(r"^# # #+ Unsloth\b")


def _live_heading(lines, start, i):
    # A commented-out section prefixes every line with "# ", blank ones
    # included, so only a live markdown cell is set off by a blank line.
    j = i - 1
    while j > start and lines[j].startswith("#"):
        j -= 1
    return j > start and lines[j] == ""


def _live_install_calls(src):
    lines = src.split("\n")
    start = next((i for i, line in enumerate(lines) if _START.match(line)), None)
    if start is None:
        return []
    end = next(
        (
            i for i in range(start + 1, len(lines))
            if _DEAD_UNSLOTH.match(lines[i]) or (_HEADING.match(lines[i]) and _live_heading(lines, start, i))
        ),
        len(lines),
    )
    calls = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "get_ipython"
            and start < node.lineno - 1 < end
        ):
            calls.append(f"line {node.lineno}: get_ipython().{node.func.attr}(...)")
    return calls


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_install_section_has_no_live_ipython_call(script):
    calls = _live_install_calls(script.read_text(encoding="utf-8"))
    allowed = _ALLOWED_LIVE.get(script.name, 0)
    assert len(calls) <= allowed, (
        f"{script.name} keeps its install section executable, so `python {script.name}` "
        f"dies with NameError: get_ipython: {calls}. Fix remove_unwanted_section in "
        f"update_all_notebooks.py, or add the script to _ALLOWED_LIVE with the reason."
    )


def test_allowlist_has_no_stale_entries():
    """An entry that no longer needs its allowance hides the next regression."""
    stale = []
    for name, allowed in _ALLOWED_LIVE.items():
        path = REPO_ROOT / "python_scripts" / name
        if not path.is_file() or len(_live_install_calls(path.read_text(encoding="utf-8"))) != allowed:
            stale.append(name)
    assert not stale, f"_ALLOWED_LIVE entries that no longer match: {stale}"



@pytest.fixture(scope = "module")
def gen():
    spec = importlib.util.spec_from_file_location("update_all_notebooks", REPO_ROOT / "update_all_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CAPTURE = "get_ipython().run_cell_magic('capture', '', 'import os\\n!pip install unsloth\\n')"


def _cells(*parts):
    return "#!/usr/bin/env python\n# coding: utf-8\n\n# ### Installation\n\n" + "".join(parts)


def _code(src):
    return f"# In[ ]:\n\n\n{src}\n\n\n"


def _live(out):
    return [line for line in out.splitlines() if line.startswith("get_ipython")]


def test_any_level_heading_ends_the_section(gen):
    out = gen.remove_unwanted_section(_cells(_code(_CAPTURE), "# ## Load the model\n\n", _code("x = 1")))
    assert _live(out) == []
    assert "\n# ## Load the model\n" in out, "the terminating heading must stay as it was"
    assert out.endswith("\nx = 1\n\n\n")


def test_first_non_install_cell_ends_the_section(gen):
    src = _cells(_code(_CAPTURE), _code("get_ipython().system('pip install trl')"), _code("from unsloth import FastModel"))
    out = gen.remove_unwanted_section(src)
    assert _live(out) == []
    assert "\nfrom unsloth import FastModel\n" in out


def test_setup_cell_stays_live(gen):
    clone = (
        "get_ipython().run_cell_magic('capture', '', '!git clone https://x/OpenEnv.git\\n%cd OpenEnv\\n"
        "import sys\\nsys.path.insert(0, \"./src\")\\n')"
    )
    out = gen.remove_unwanted_section(_cells(_code(_CAPTURE), _code(clone), _code("from unsloth import FastModel")))
    assert _live(out) == [clone]


def test_capture_cell_binding_a_later_name_stays_live(gen):
    cell = "get_ipython().run_cell_magic('capture', '', '!pip install foo\\nwork_dir = \"/tmp/x\"\\n')"
    out = gen.remove_unwanted_section(_cells(_code(_CAPTURE), _code(cell), _code("print(work_dir)")))
    assert _live(out) == [cell]


def test_mixed_cell_only_loses_its_pip_line(gen):
    mixed = "get_ipython().system('pip install -q \"transformers==5.15.0\"')\nimport transformers\nassert transformers.__version__"
    out = gen.remove_unwanted_section(_cells(_code(_CAPTURE), _code(mixed)))
    assert _live(out) == []
    assert "\n# get_ipython().system('pip install -q \"transformers==5.15.0\"')\nimport transformers\n" in out


def test_standby_env_is_lifted_without_a_unsloth_heading(gen):
    cell = "get_ipython().run_cell_magic('capture', '', 'import os\\nos.environ[\"UNSLOTH_VLLM_STANDBY\"] = \"1\"\\n!pip install vllm\\n')"
    out = gen.remove_unwanted_section(_cells(_code(cell), "# ## Load\n\n", _code("x = 1")))
    assert '\nos.environ["UNSLOTH_VLLM_STANDBY"] = "1"\n' in out
    assert _live(out) == []


def test_unsloth_heading_path_is_unchanged(gen):
    """Pinned output of the original terminator, which every templated notebook takes."""
    src = "a\n# ### Installation\n\n# In[1]:\n\n\nget_ipython().system('pip x')\n\n\n# ### Unsloth\n\nimport x\n"
    assert gen.remove_unwanted_section(src) == (
        "a\n# # ### Installation\n# \n# # In[1]:\n# \n# \n# get_ipython().system('pip x')\n# \n# \n# # ### Unsloth\n\nimport x\n"
    )
