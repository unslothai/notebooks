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

"""Only the fla cleanup survives the CUDA source into an AMD setup block.

`_extract_preserved_setup_lines` reads stripped lines and the AMD composer
re-emits what it returns unconditionally, after the ROCm bootstrap. So a rule
broad enough to catch any `!uv pip uninstall` would take a line that the source
runs only inside an `if` and run it always, against the ROCm stack, which is the
one place a stray uninstall of torch cannot be recovered from. The cleanup line
is matched by its exact package list at column zero instead, and it is rewritten
to carry the `--system` flag every other AMD pip call has, exactly once.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FLA_UNINSTALL = "!uv pip uninstall -qqq flash-linear-attention fla-core"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "update_all_notebooks", REPO_ROOT / "update_all_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


def test_fla_uninstall_is_preserved_with_system():
    preserved = GEN._extract_preserved_setup_lines(
        "!uv pip install -qqq transformers==5.2.0\n" + FLA_UNINSTALL + "\n"
    )
    assert preserved == ["!uv pip uninstall --system -qqq flash-linear-attention fla-core"]


def test_reversed_package_order_is_still_the_cleanup_line():
    preserved = GEN._extract_preserved_setup_lines(
        "!uv pip uninstall -qqq fla-core flash-linear-attention\n"
    )
    assert preserved == ["!uv pip uninstall --system -qqq fla-core flash-linear-attention"]


def test_an_existing_system_flag_is_not_doubled():
    line = "!uv pip uninstall --system -qqq flash-linear-attention fla-core"
    preserved = GEN._extract_preserved_setup_lines(line + "\n")
    assert preserved == [line]
    assert "--system --system" not in preserved[0]


def test_a_conditional_uninstall_is_not_preserved():
    """Indented in the source means guarded; the AMD cell would run it always."""
    preserved = GEN._extract_preserved_setup_lines(
        'if os.environ.get("KEEP") != "1":\n    ' + FLA_UNINSTALL + "\n"
    )
    assert preserved == []


def test_an_unrelated_uninstall_is_not_preserved():
    preserved = GEN._extract_preserved_setup_lines(
        "!uv pip uninstall torch\n!uv pip uninstall -qqq flash-linear-attention\n"
    )
    assert preserved == []


def test_the_fla_tilelang_env_line_is_dropped():
    preserved = GEN._extract_preserved_setup_lines('os.environ["FLA_TILELANG"] = "0"\n')
    assert preserved == []


def test_other_environment_lines_are_still_preserved():
    preserved = GEN._extract_preserved_setup_lines(
        'os.environ["UNSLOTH_MOE_DISABLE_AUTOTUNE"] = "1"\n'
        'os.environ["FLA_TILELANG"] = "0"\n'
        'os.environ["UNSLOTH_VLLM_STANDBY"] = "1"\n'
    )
    assert preserved == [
        'os.environ["UNSLOTH_MOE_DISABLE_AUTOTUNE"] = "1"',
        'os.environ["UNSLOTH_VLLM_STANDBY"] = "1"',
    ]
