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

"""`.generate()` is the transformers surface, so it takes transformers names.

vLLM's sampling knobs and the transformers ones overlap enough to be confusing
(`temperature`, `top_k` and `top_p` are spelled the same in both) that the two
that are NOT spelled the same get mixed up. `max_tokens` is the vLLM name;
transformers calls it `max_new_tokens`, and rejects anything it does not know:

    ValueError: The following `model_kwargs` are not used by the model:
    ['max_tokens'] (note: typos in the generate arguments will also show up in
    this list)

The LFM2.5 GRPO notebook shipped exactly that. Its own earlier cell already
said `max_new_tokens = 1024`, and the post-training copy of the same cell said
`max_tokens = 1024`, so the notebook ran for nineteen minutes and then died on
the second-to-last cell. Nothing static caught it: the file is valid JSON, the
pins are consistent, and a name that generate rejects at runtime looks exactly
like a name it accepts.

A vLLM-only name only reaches `SamplingParams` through `fast_generate` or
`GRPOConfig(vllm_sampling_params = SamplingParams(...))`, neither of which is
this call, so a hit here is always a bug.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"
TEMPLATE_DIR = REPO_ROOT / "original_template"

# Named on `SamplingParams`, absent from `GenerationConfig`, and asserted to
# stay that way by `test_the_denied_names_are_really_not_transformers_names`
# below -- so this list cannot quietly rot into false positives when
# transformers adds one of them.
VLLM_ONLY_KWARGS = {
    "max_tokens": "max_new_tokens",
    "min_tokens": "min_new_tokens",
    "stop": "stop_strings",
    "stop_token_ids": "eos_token_id",
    "ignore_eos": "min_new_tokens",
    "include_stop_str_in_output": None,
    "presence_penalty": "repetition_penalty",
    "frequency_penalty": "repetition_penalty",
    "prompt_logprobs": None,
    "detokenize": None,
    "best_of": "num_return_sequences",
}

# Any receiver: `model.generate(`, `base_model.generate(`, `self.m.generate(`.
# `fast_generate` has no `.` in front of `generate`, so the vLLM entry point is
# left alone, which is the point -- there `max_tokens` is correct.
_GENERATE_CALL = re.compile(r"\.\s*generate\s*\(")
_OPEN = "([{"
_CLOSE = ")]}"
# `name =` / `name=` at the start of an argument.
_KEYWORD = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=(?!=)")


def _call_body(source: str, open_paren: int) -> str:
    """The text between `(` at `open_paren` and its partner.

    Returns "" for an unbalanced call, which happens when a cell was split
    mid-expression; an unparseable call is not evidence of a bug.
    """
    depth = 0
    in_string = ""
    index = open_paren
    while index < len(source):
        char = source[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if source.startswith(in_string, index):
                index += len(in_string)
                in_string = ""
                continue
        elif char in "\"'":
            for quote in ('"""', "'''", '"', "'"):
                if source.startswith(quote, index):
                    in_string = quote
                    index += len(quote)
                    break
            continue
        elif char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
            if depth == 0:
                return source[open_paren + 1:index]
        index += 1
    return ""


def _top_level_keywords(body: str) -> set[str]:
    """Argument names of THIS call, not of the calls nested in it.

    `streamer = TextStreamer(tokenizer, skip_prompt = False)` is one argument;
    `skip_prompt` belongs to `TextStreamer` and must not be attributed to
    `generate`.
    """
    names = set()
    depth = 0
    in_string = ""
    index = 0
    segment_start = 0
    segments = []
    while index < len(body):
        char = body[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if body.startswith(in_string, index):
                index += len(in_string)
                in_string = ""
                continue
        elif char in "\"'":
            for quote in ('"""', "'''", '"', "'"):
                if body.startswith(quote, index):
                    in_string = quote
                    index += len(quote)
                    break
            continue
        elif char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        elif char == "," and depth == 0:
            segments.append(body[segment_start:index])
            segment_start = index + 1
        index += 1
    segments.append(body[segment_start:])
    for segment in segments:
        match = _KEYWORD.match(segment.strip())
        if match:
            names.add(match.group(1))
    return names


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _offending_kwargs(source: str) -> set[str]:
    found = set()
    for match in _GENERATE_CALL.finditer(source):
        body = _call_body(source, match.end() - 1)
        if not body:
            continue
        keywords = _top_level_keywords(body)
        # `llm.generate(prompts, sampling_params = ...)` is the vLLM offline
        # entry point, where these names are carried inside SamplingParams and
        # never appear as siblings anyway.
        if "sampling_params" in keywords:
            continue
        found |= keywords & set(VLLM_ONLY_KWARGS)
    return found


def _notebooks():
    return sorted(NB_DIR.glob("*.ipynb")) + sorted(TEMPLATE_DIR.glob("*.ipynb"))


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_generate_is_not_given_vllm_sampling_names(path):
    offending = _offending_kwargs(_code(path))
    assert not offending, (
        f"{path.relative_to(REPO_ROOT)} passes vLLM sampling argument(s) "
        + ", ".join(
            f"`{name}`"
            + (f" (transformers spells it `{VLLM_ONLY_KWARGS[name]}`)"
               if VLLM_ONLY_KWARGS[name] else " (no transformers equivalent)")
            for name in sorted(offending))
        + " to `.generate()`. transformers raises ValueError on unknown "
          "`model_kwargs`, so this cell cannot run."
    )


# --------------------------------------------------------------------------
# The extractor itself, so a hit and a miss are both demonstrated.
# --------------------------------------------------------------------------

def test_the_bug_this_file_exists_to_catch_is_detected():
    assert _offending_kwargs(
        "_ = model.generate(**inputs, temperature = 0.1, max_tokens = 1024)"
    ) == {"max_tokens"}


def test_the_fixed_call_is_clean():
    assert _offending_kwargs(
        "_ = model.generate(**inputs, temperature = 0.1, max_new_tokens = 1024)"
    ) == set()


def test_a_nested_call_argument_is_not_attributed_to_generate():
    """`TextStreamer(...)` sits inside the argument list of every one of these
    cells; reading its keywords as generate's would be a false positive."""
    assert _offending_kwargs(
        "_ = model.generate(**inputs, max_new_tokens = 8,\n"
        "                   streamer = TextStreamer(tok, stop = ['x']))"
    ) == set()


def test_fast_generate_keeps_the_vllm_spelling():
    """`max_tokens` is correct on the vLLM entry point, so it must not fire."""
    assert _offending_kwargs(
        "out = model.fast_generate(prompt, max_tokens = 1024)") == set()


def test_the_vllm_offline_entry_point_is_left_alone():
    assert _offending_kwargs(
        "out = llm.generate(prompts, sampling_params = params, max_tokens = 8)"
    ) == set()


def test_a_name_inside_a_string_is_not_an_argument():
    assert _offending_kwargs(
        "_ = model.generate(**inputs, stop_strings = ['max_tokens = 1'])"
    ) == set()


def test_an_unbalanced_call_is_not_reported():
    """A cell that ends mid-expression is unparseable, not buggy."""
    assert _offending_kwargs("_ = model.generate(**inputs, max_tokens = 1024") == set()


def test_a_multiline_call_is_read_whole():
    assert _offending_kwargs(
        "_ = model.generate(\n"
        "    **inputs,\n"
        "    repetition_penalty = 1.05,\n"
        "    max_tokens = 1024,\n"
        ")"
    ) == {"max_tokens"}
