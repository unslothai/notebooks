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
import shlex
from pathlib import Path, PurePosixPath

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


# `#` only starts a comment at a word boundary, leaving `...#subdirectory=x`
# in a URL alone.
_COMMENT = re.compile(r"(?<!\S)#.*$")
# A pytest call must not vouch for what runs beside it.
_SEPARATORS = re.compile(r";|&&|\|\|")


def _shell_commands(block):
    """The individual commands in one `run:` block, comments stripped and
    continuations folded, so a filename is collected only from a line that
    executes it."""
    folded, pending = [], ""
    for raw in block.splitlines():
        line = _COMMENT.sub("", raw).rstrip()
        if not line.strip():
            continue
        if line.endswith("\\"):
            pending += " " + line[:-1].strip()
            continue
        pending += " " + line.strip()
        folded.append(pending.strip())
        pending = ""
    if pending.strip():
        folded.append(pending.strip())
    return [part.strip() for line in folded
            for part in _SEPARATORS.split(line) if part.strip()]


# Leading `FOO=bar` assignments belong to the command, not to a different one.
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PYTHON = re.compile(r"^python[0-9.]*$")


def _invokes_pytest(command):
    """Whether pytest is what this command RUNS, not just a word inside it.

    `pip install pytest ...` contains the word, and a step disabled as
    `echo pytest tests/test_x.py` keeps it while running nothing.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:      # an unbalanced quote, e.g. left by comment stripping
        tokens = command.split()
    while tokens and _ASSIGNMENT.match(tokens[0]):
        tokens.pop(0)
    if not tokens:
        return False
    head = PurePosixPath(tokens[0]).name    # `.venv/bin/pytest` is still pytest
    if head in ("pytest", "py.test"):
        return True
    if not _PYTHON.match(head) or "-m" not in tokens:
        return False
    return tokens[tokens.index("-m") + 1:][:1] == ["pytest"]


def _named_in_workflow():
    """Test files a `pytest` command in the workflow actually runs.

    Asking whether the whole `run:` block contains "pytest" and harvesting every
    filename in it reads a comment naming another test as coverage, so that test
    can be absent from CI while the gate passes. Hence per command, and per
    command that really invokes pytest.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    named = set()
    for block in _run_commands(workflow):
        for command in _shell_commands(block):
            if not _invokes_pytest(command):
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


def _named_in(text):
    """The gate's collection rule applied to one `run:` block."""
    named = set()
    for command in _shell_commands(text):
        if not _invokes_pytest(command):
            continue
        named.update(_TEST_FILE.findall(command))
    return named


@pytest.mark.parametrize("command", [
    "echo pytest tests/test_ghost.py",
    'echo "skipping python -m pytest tests/test_ghost.py for now"',
    "pip install pytest -r tests/test_requirements.py",
    "grep -n pytest tests/test_ghost.py",
], ids=["echo", "echo-quoted", "pip-install", "grep"])
def test_a_command_that_only_mentions_pytest_is_not_coverage(command):
    """Matching the word alone leaves the test absent from CI, gate still green."""
    assert _named_in(command) == set()


@pytest.mark.parametrize("command", [
    "pytest tests/test_real.py -q",
    "python -m pytest tests/test_real.py -q",
    "python3.12 -m pytest tests/test_real.py -q",
    "PYTHONPATH=. python -m pytest tests/test_real.py -q",
    ".venv/bin/pytest tests/test_real.py -q",
], ids=["bare", "module", "versioned", "env-prefix", "path"])
def test_the_ways_we_really_invoke_pytest_still_count(command):
    """The other direction: a covered test must not be reported missing."""
    assert _named_in(command) == {"test_real.py"}


def test_a_comment_beside_a_real_pytest_call_is_not_coverage():
    """Harvesting every filename in a block that contains "pytest" counts the
    commented one."""
    block = (
        "set -euxo pipefail\n"
        "# tests/test_ghost.py is coming in a follow-up\n"
        "python -m pytest tests/test_real.py -q\n"
    )
    assert _named_in(block) == {"test_real.py"}


def test_a_non_pytest_command_beside_a_pytest_one_is_not_coverage():
    """`echo` and `ls` name files too, so splitting on the separators keeps a
    pytest call from vouching for its neighbours."""
    block = (
        "echo tests/test_echoed.py\n"
        "python -m pytest tests/test_real.py -q\n"
        "ls tests/test_listed.py && cat tests/test_catted.py\n"
    )
    assert _named_in(block) == {"test_real.py"}


def test_a_pytest_call_split_over_a_continuation_still_counts():
    """Without folding, a wrapped command reads as a missing test."""
    block = (
        "python -m pytest \\\n"
        "    tests/test_wrapped.py \\\n"
        "    -q --tb=short\n"
    )
    assert _named_in(block) == {"test_wrapped.py"}


def test_a_url_fragment_is_not_read_as_a_comment():
    """`#subdirectory=` must survive comment stripping, or the command is
    truncated and its filenames lost."""
    block = (
        "pip install 'x @ git+https://example.com/x.git@abc#subdirectory=y'\n"
        "python -m pytest tests/test_after_url.py -q\n"
    )
    assert _named_in(block) == {"test_after_url.py"}
    assert any("subdirectory=y" in c for c in _shell_commands(block))


def test_the_real_workflow_still_reports_its_pytest_files():
    """A fold or split that stopped matching would leave the gate asserting
    against an empty set."""
    assert len(_named_in_workflow()) >= 10
