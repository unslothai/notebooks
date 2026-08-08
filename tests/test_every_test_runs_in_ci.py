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

"""A test nothing runs is not a gate.

`notebooks-tests-ci.yml` names each test file in its own step rather than
discovering `tests/`, which is fine until someone adds a file and forgets.
Three had already drifted off the list: the two AMD generator tests here and
`test_transformers5_hub_floor.py`, added with the hub floor it guards.

This is the cheapest possible check, and it fails on the file you just wrote
rather than months later when the regression it was meant to catch ships.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "notebooks-tests-ci.yml"
TESTS = REPO_ROOT / "tests"


def _named_in_workflow():
    text = WORKFLOW.read_text(encoding="utf-8")
    return set(re.findall(r"tests/([A-Za-z0-9_]+\.py)", text))


def test_the_workflow_is_where_we_think_it_is():
    assert WORKFLOW.is_file(), f"no workflow at {WORKFLOW}"


@pytest.mark.parametrize(
    "path", sorted(TESTS.glob("test_*.py")), ids=lambda p: p.name)
def test_every_test_file_has_a_ci_step(path):
    assert path.name in _named_in_workflow(), (
        f"{path.name} is never run by notebooks-tests-ci.yml. Add a step for "
        f"it, or the tests in it are decoration."
    )
