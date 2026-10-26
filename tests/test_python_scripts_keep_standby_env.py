# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""python_scripts/ comment out the install cells, which is where the GRPO notebooks
set UNSLOTH_VLLM_STANDBY. The scripts must still set it, once, before unsloth loads."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope = "module")
def gen():
    spec = importlib.util.spec_from_file_location("update_all_notebooks", REPO_ROOT / "update_all_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CAPTURE = (
    "get_ipython().run_cell_magic('capture', '', 'import os\\n"
    "os.environ[\"UNSLOTH_VLLM_STANDBY\"] = \"1\" # [NEW] Extra 30% context lengths!\\n"
    "if \"COLAB_\" not in \"\".join(os.environ.keys()):\\n    !pip install unsloth vllm\\n')"
)


def _script(install_lines, after = "from unsloth import FastLanguageModel\n"):
    return (
        "#!/usr/bin/env python\n\n# ### Installation\n\n# In[ ]:\n\n\n"
        + "\n".join(install_lines)
        + "\n\n\n# ### Unsloth\n\n# In[ ]:\n\n\n" + after
    )


def _live(text):
    return [l for l in text.splitlines() if not l.startswith("#") and "UNSLOTH_VLLM_STANDBY" in l]


def test_capture_cell_keeps_standby_live(gen):
    out = gen.remove_unwanted_section(_script([_CAPTURE]))
    assert _live(out) == ['os.environ["UNSLOTH_VLLM_STANDBY"] = "1"']
    ast.parse(out.replace("get_ipython()", "x"))
    # The env line runs before the first unsloth import, and the heading line is unchanged.
    assert out.index('os.environ["UNSLOTH_VLLM_STANDBY"]') < out.index("from unsloth import")
    assert "\n# # ### Unsloth\n" in out


def test_amd_one_liner_is_kept(gen):
    out = gen.remove_unwanted_section(_script(['import os; os.environ["UNSLOTH_VLLM_STANDBY"] = "1"', "get_ipython().system('pip install vllm')"]))
    assert _live(out) == ['os.environ["UNSLOTH_VLLM_STANDBY"] = "1"']


def test_not_duplicated_when_the_script_sets_it_later(gen):
    later = "import os\nos.environ['UNSLOTH_VLLM_STANDBY'] = \"1\" # Standby\nfrom unsloth import FastLanguageModel\n"
    out = gen.remove_unwanted_section(_script([_CAPTURE], after = later))
    assert len(_live(out)) == 1


def test_install_cell_without_standby_is_unchanged(gen):
    src = _script(["get_ipython().system('pip install unsloth')", "import os", 'os.environ["HF_TOKEN"] = "x"'])
    out = gen.remove_unwanted_section(src)
    assert "Kept from the installation cell" not in out
    assert not [l for l in out.splitlines()[:16] if l and not l.startswith("#")]


def test_indented_standby_is_not_lifted(gen):
    cell = "get_ipython().run_cell_magic('capture', '', 'import os\\nif True:\\n    os.environ[\"UNSLOTH_VLLM_STANDBY\"] = \"1\"\\n')"
    out = gen.remove_unwanted_section(_script([cell]))
    assert _live(out) == []
