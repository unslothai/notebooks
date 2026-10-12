# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DPO completions must not already end in EOS (unslothai/unsloth#3228).

TRL's DPOTrainer appends EOS to string completions (unconditionally up to
0.28; from 0.29 unless the text ends with it). Zephyr's chat template ends
every turn with `</s>\\n`, so an unstripped completion trained on
`</s> \\n</s>` with every TRL version.
"""
from __future__ import annotations

import ast
import re
import sys
import typing
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402

EOS = "</s>"


class _ZephyrTokenizer:
    eos_token = EOS

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|{m['role']}|>\n{m['content']}{EOS}\n" for m in messages)
        return text + ("<|assistant|>\n" if add_generation_prompt else "")


def _dpo_formatters():
    for path in ni.iter_notebooks():
        for _, source in ni.iter_code_cells(path):
            if "def apply_chat_template" in source and 'task == "dpo"' in source:
                yield path, source


_CASES = list(_dpo_formatters())


def test_dpo_notebooks_found():
    assert _CASES, "no notebook defines the DPO apply_chat_template helper"


@pytest.mark.parametrize("path,source", _CASES, ids=[f"{p.parent.name}/{p.name}" for p, _ in _CASES])
def test_dpo_completions_leave_eos_to_trainer(path, source):
    tree = ast.parse(ni.strip_jupyter_magics(source))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_chat_template")
    ns = {"re": re, "List": typing.List, "Literal": typing.Literal, "Optional": typing.Optional}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)

    example = {
        "chosen": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Good answer."}],
        "rejected": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bad answer."}],
    }
    out = ns["apply_chat_template"](example, _ZephyrTokenizer(), task="dpo")

    assert out["text_prompt"] == f"<|system|>\n{EOS}\n<|user|>\nHi{EOS}\n<|assistant|>\n"
    assert out["text_chosen"] == "Good answer."
    assert out["text_rejected"] == "Bad answer."
