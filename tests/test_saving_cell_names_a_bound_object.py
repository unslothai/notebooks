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

"""Every saving call names an object the notebook actually binds.

The saving cells are instructions: local lines run, upload lines sit commented
for the reader to uncomment, so a wrong receiver is invisible to every executing
check and surfaces as a `NameError` on the reader's machine.

The Qwen3 vision GRPO notebooks load with `model, tokenizer =
FastVisionModel.from_pretrained(...)` and never bind `processor`, yet their
online-saving line read `processor.push_to_hub(...)`. The Gemma and Sesame
notebooks beside them do unpack into `model, processor`, which is how the
mismatch survived review and rode the AMD generator into a second copy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"

# `model.push_to_hub(`, `tokenizer.save_pretrained_gguf(`, ... The receiver is
# a bare name, so `api.model.push_to_hub(` is left to whatever binds `api`.
_SAVING_CALL = re.compile(
    r"(?<![\w.])([A-Za-z_]\w*)\s*\.\s*(?:push_to_hub|save_pretrained)\w*\s*\(")

_LEADING_COMMENT = re.compile(r"^(\s*)(?:#\s?)+")
_NAME_RUN = re.compile(r"[A-Za-z_]\w*")

_ASSIGN = re.compile(r"^\s*([^=<>!+\-*/%&|^]+?)\s*=(?!=)")
_AUGASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*[+\-*/%&|^@]?=")
_FOR = re.compile(r"(?:^|\W)for\s+(.+?)\s+in\s")
_AS = re.compile(r"(?:^|\W)as\s+([A-Za-z_]\w*)")
_IMPORT = re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+(.+)$")
_DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
_PARAMS = re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\((.*)")
_WALRUS = re.compile(r"([A-Za-z_]\w*)\s*:=")


def _code(path: Path) -> list[str]:
    """Code-cell lines with any comment marker stripped.

    Commented lines count as code on purpose: an instruction to uncomment has
    to name something real, and its binding is often commented beside it.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for raw in "".join(cell.get("source", [])).splitlines():
            lines.append(_LEADING_COMMENT.sub(r"\1", raw))
    return lines


def _bound_names(lines) -> set[str]:
    names = set()
    for line in lines:
        for pattern in (_DEF, _AUGASSIGN, _AS, _IMPORT, _PARAMS):
            match = pattern.search(line)
            if match:
                names.update(_NAME_RUN.findall(match.group(1)))
        for pattern in (_ASSIGN, _FOR):
            match = pattern.search(line)
            if match and "(" not in match.group(1):
                names.update(_NAME_RUN.findall(match.group(1)))
        names.update(_WALRUS.findall(line))
    return names


def _unbound_receivers(path: Path) -> set[str]:
    lines = _code(path)
    bound = _bound_names(lines)
    used = {name for line in lines for name in _SAVING_CALL.findall(line)}
    return used - bound


@pytest.mark.parametrize(
    "path", sorted(NB_DIR.glob("*.ipynb")), ids=lambda p: p.name)
def test_every_saving_call_has_a_receiver(path):
    unbound = _unbound_receivers(path)
    assert not unbound, (
        f"{path.name} saves through {sorted(unbound)}, which the notebook "
        f"never binds. Uncommenting that line raises NameError."
    )


def test_the_qwen3_vision_grpo_family_saves_through_its_tokenizer():
    """The four that carried the bug unpack into `model, tokenizer`, so the
    upload line has to say `tokenizer`."""
    for name in (
        "Qwen3_VL_(8B)-Vision-GRPO.ipynb",
        "AMD-Qwen3_VL_(8B)-Vision-GRPO.ipynb",
        "Qwen3_5_(4B)_Vision_GRPO.ipynb",
        "AMD-Qwen3_5_(4B)_Vision_GRPO.ipynb",
    ):
        text = (NB_DIR / name).read_text(encoding="utf-8")
        # Kept out of the assert so a failure reports the name instead of
        # diffing megabytes of JSON.
        pushes_processor = "processor.push_to_hub" in text
        pushes_tokenizer = "tokenizer.push_to_hub" in text
        assert not pushes_processor, (
            f"{name} is back to pushing through an unbound `processor`")
        assert pushes_tokenizer, (
            f"{name} no longer uploads the tokenizer at all")


def test_a_processor_receiver_is_fine_where_the_notebook_binds_one():
    """Not a blanket ban: the Gemma vision notebooks do unpack into
    `model, processor`."""
    gemma = NB_DIR / "Gemma3_(4B)-Vision.ipynb"
    pushes_processor = "processor.push_to_hub" in gemma.read_text(encoding="utf-8")
    assert pushes_processor, "the sample notebook no longer exercises the case"
    assert _unbound_receivers(gemma) == set()


def test_an_unbound_receiver_is_caught(tmp_path):
    """The discriminating case: same cell, one bound receiver and one not."""
    notebook = {"cells": [{"cell_type": "code", "source": [
        "model, tokenizer = FastVisionModel.from_pretrained('m')\n",
        "tokenizer.save_pretrained('lora')\n",
        "# processor.push_to_hub('me/lora')\n",
    ]}]}
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert _unbound_receivers(path) == {"processor"}


def test_a_markdown_mention_is_not_a_call(tmp_path):
    """Prose names these objects constantly. Only code cells are read."""
    notebook = {"cells": [
        {"cell_type": "markdown", "source": ["Call `processor.push_to_hub(...)`\n"]},
        {"cell_type": "code", "source": ["model = 1\n", "model.push_to_hub('x')\n"]},
    ]}
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert _unbound_receivers(path) == set()
