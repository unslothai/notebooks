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
discovering `tests/`, which is fine until someone adds a file and forgets --
three had already drifted off the list. This fails on the file you just wrote
rather than months later when the regression ships.

It reads the `run:` commands, not the file text: every step names its test in
`name:` too, so a text scan stayed green when a command was replaced and its
label left behind.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "notebooks-tests-ci.yml"
TESTS = REPO_ROOT / "tests"

_TEST_FILE = re.compile(r"tests/([A-Za-z0-9_]+\.py)")


def _run_commands(node):
    """Every `run:` string in the workflow, at any nesting depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                yield value
            else:
                yield from _run_commands(value)
    elif isinstance(node, list):
        for item in node:
            yield from _run_commands(item)


def _named_in_workflow():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    named = set()
    for command in _run_commands(workflow):
        if "pytest" not in command:
            continue
        named.update(_TEST_FILE.findall(command))
    return named


def test_the_workflow_is_where_we_think_it_is():
    assert WORKFLOW.is_file(), f"no workflow at {WORKFLOW}"


@pytest.mark.parametrize(
    "path", sorted(TESTS.glob("test_*.py")), ids=lambda p: p.name)
def test_every_test_file_has_a_ci_step(path):
    assert path.name in _named_in_workflow(), (
        f"{path.name} is never run by notebooks-tests-ci.yml. Add a step for "
        f"it, or the tests in it are decoration."
    )


def test_a_label_without_a_command_does_not_count(tmp_path, monkeypatch):
    """The failure this file exists to catch: a step keeps its `name:` while
    its command is replaced, and nothing says so."""
    workflow = tmp_path / "ci.yml"
    workflow.write_text(
        "# tests/test_ghost.py is mentioned here too\n"
        "jobs:\n"
        "  lint:\n"
        "    steps:\n"
        "      - name: tests/test_ghost.py\n"
        "        run: echo skipped\n"
        "      - name: tests/test_real.py\n"
        "        run: python -m pytest tests/test_real.py -q\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("tests.test_every_test_runs_in_ci.WORKFLOW", workflow)
    assert _named_in_workflow() == {"test_real.py"}
