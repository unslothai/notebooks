# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
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
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The NeMo Gym venv must ask for an interpreter by specifier, not a version.

These notebooks clone `NVIDIA-NeMo/Gym` at unpinned HEAD and build a venv for
it. They asked for `--python 3.12`. On 2026-08-04 upstream commit ea4c6c6
raised the project's own floor from `>=3.12` to `>=3.13.14`, so `uv sync`
started refusing to run:

    error: The requested interpreter resolved to Python 3.12.3, which is
    incompatible with the project's Python requirement: `>=3.13.14`
    (from `project.requires-python`)

The notebook wraps that in `subprocess.run(..., check = True)`, so the cell
raises `CalledProcessError ... returned non-zero exit status 2` and the run
ends before any training. Reproduced locally against upstream HEAD, and seen
on a Colab L4 during a sweep pinned to unsloth main and unsloth_zoo main.

A fixed version is the wrong shape of fix twice over. It goes stale the next
time upstream moves, and `--python 3.13` does not even fix it today: uv
resolves that to the newest 3.13 it happens to have, which was 3.13.8 here,
still under the floor. A specifier lets uv pick, and provision, something that
satisfies whatever upstream currently declares.

Deliberately no upper bound on the accepted specifier: pinning one here would
recreate the same staleness in the test.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

GYM_REPO = "NVIDIA-NeMo/Gym"

# `"--python", <request>` as it appears in a `uv venv` argument list. The
# request is either a literal or a name, because the notebook has to make the
# same request twice, once to create and once to rebuild with `--clear`, and
# two literals could drift apart. A name is resolved from its assignment.
_RE_PYTHON_REQUEST = re.compile(r'"--python",\s*("[^"]+"|[A-Za-z_]\w*)')


def _python_requests(source):
    requests = []
    for raw in _RE_PYTHON_REQUEST.findall(source):
        if raw.startswith('"'):
            requests.append(raw.strip('"'))
            continue
        assignment = re.search(rf'^\s*{re.escape(raw)}\s*=\s*"([^"]+)"', source, re.M)
        # Unresolvable is reported as-is, so it fails the checks below rather
        # than vanishing from the list and leaving them vacuously true.
        requests.append(assignment.group(1) if assignment else raw)
    return requests


def _code(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _gym_notebooks():
    if not NB_DIR.is_dir():
        return []
    return sorted(p for p in NB_DIR.glob("*.ipynb") if GYM_REPO in _code(p))


_GYM = _gym_notebooks()


def test_the_gym_notebooks_are_still_discoverable():
    """If the clone URL changes, every parametrised case below silently stops
    running and this file becomes decoration."""
    assert len(_GYM) >= 4, f"found {len(_GYM)} notebooks cloning {GYM_REPO}"


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_the_gym_venv_asks_for_an_interpreter_by_specifier(path):
    requests = _python_requests(_code(path))
    assert requests, f"{path.name} builds no venv with an explicit --python"
    for request in requests:
        assert request.startswith(">="), (
            f"{path.name} asks uv for Python {request!r}. NeMo Gym raises its own "
            f"`requires-python` floor over time, so a bare version goes stale and "
            f"`uv sync` exits 2. Ask by specifier, for example '>=3.13.14'."
        )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_the_request_clears_the_floor_upstream_declares_today(path):
    """3.13.14 is what upstream required as of 2026-08-04. A specifier that
    admits anything older is no better than the 3.12 pin it replaced."""
    for request in _python_requests(_code(path)):
        floor = tuple(int(n) for n in re.findall(r"\d+", request)[:3])
        assert floor >= (3, 13, 14), (
            f"{path.name} asks for {request!r}, which admits interpreters below "
            f"the 3.13.14 upstream requires"
        )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_no_stale_prose_promises_the_old_floor(path):
    """The markdown said 'requires Python 3.12+'. Leaving that in place sends a
    reader who hits the failure looking in the wrong direction."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    prose = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )
    assert "Python 3.12+" not in prose, (
        f"{path.name} still tells the reader NeMo Gym requires Python 3.12+"
    )


def test_a_bare_version_is_rejected_by_the_specifier_check():
    """Discriminating case, held here rather than only against the tree, so the
    check keeps meaning something once every notebook is compliant."""
    assert not "3.12".startswith(">=")
    assert not "3.13".startswith(">=")
    assert ">=3.13.14".startswith(">=")


def test_the_floor_check_rejects_a_specifier_that_is_too_low():
    for request in (">=3.12", ">=3.13", ">=3.13.8"):
        floor = tuple(int(n) for n in re.findall(r"\d+", request)[:3])
        assert not floor >= (3, 13, 14), request


# The setup must be able to repair a venv that a previous failed run left behind.
# `uv venv` succeeds and `uv sync` is the half that fails, so everyone who hit
# the floor error has a complete-looking `.venv/bin/python` holding the wrong
# interpreter. Guarding the sync on that path existing skips exactly them, and
# they hit a missing `ng_run` much later instead. uv also refuses to overwrite a
# venv without `--clear`:
#
#   error: A virtual environment already exists at `.venv`. Use `--clear` to
#   replace it
#
# so the rebuild has to say so or it silently keeps the broken one.


def _strip_comments(source):
    """Comments out, because this file's own comments name every token it
    searches for. Checking the raw text let a sabotage that deleted `--clear`
    from the argument list still pass, on the strength of a comment three lines
    above saying uv refuses to overwrite a venv without --clear. That is the
    same vacuous-grep failure this suite exists to prevent."""
    kept = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _setup_source(path):
    """The cell that clones Gym and builds its venv, comments removed."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if GYM_REPO in source and "uv" in source:
            return _strip_comments(source)
    return ""


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_uv_sync_is_not_guarded_on_the_venv_already_existing(path):
    source = _setup_source(path)
    assert source, f"{path.name}: setup cell not found"
    guard = 'if not os.path.exists(os.path.join(GYM_DIR, ".venv", "bin", "python")):'
    assert guard in source, f"{path.name}: the existence guard moved, re-check this test"
    body = source.split(guard, 1)[1]
    # Everything the guard covers is indented; the sync must come after it.
    guarded, rest = [], []
    seen_dedent = False
    for line in body.splitlines()[1:]:
        if line.strip() and not line.startswith((" ", "\t")):
            seen_dedent = True
        (rest if seen_dedent else guarded).append(line)
    # `_uv_sync()` as well as the literal, or moving the call into the helper
    # and calling THAT from inside the guard would restore the bug unnoticed.
    guarded_text, rest_text = "\n".join(guarded), "\n".join(rest)
    assert "_uv_sync()" not in guarded_text and "uv sync" not in guarded_text, (
        f"{path.name} runs the sync inside the venv-exists guard, so a user "
        f"whose earlier run failed at sync never gets it retried"
    )
    assert "_uv_sync()" in rest_text or "uv sync" in rest_text, (
        f"{path.name} never runs `uv sync` outside the guard"
    )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_rejected_venv_is_rebuilt_with_clear(path):
    source = _setup_source(path)
    assert '"--clear"' in source, (
        f"{path.name} never rebuilds the venv with --clear, and uv refuses to "
        f"replace an existing one without it, so a stale interpreter survives"
    )
    rebuild = source.split("--clear", 1)[0].rsplit("subprocess.run", 1)[-1]
    assert "_GYM_PYTHON" in rebuild or ">=" in rebuild, (
        f"{path.name} rebuilds without asking for a satisfying interpreter"
    )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_failed_sync_surfaces_its_output(path):
    """`capture_output` with no print leaves the user the same bare
    CalledProcessError this PR exists to explain."""
    source = _setup_source(path)
    assert "_sync.stdout" in source and "_sync.stderr" in source, (
        f"{path.name} captures the sync output but never prints it on failure"
    )
    assert "check_returncode" in source, f"{path.name} never fails on a bad sync"


def test_an_unresolvable_name_is_not_silently_dropped():
    """A name with no assignment must survive into the list, so the specifier
    check fails on it. Dropping it would leave every check vacuously true."""
    source = 'subprocess.run(["uv", "venv", "--python", _MISSING])'
    assert _python_requests(source) == ["_MISSING"]
    assert not _python_requests(source)[0].startswith(">=")


def test_a_name_is_resolved_to_its_assignment():
    source = '_GYM_PYTHON = ">=3.13.14"\nsubprocess.run(["uv", "venv", "--python", _GYM_PYTHON])'
    assert _python_requests(source) == [">=3.13.14"]


def test_both_the_create_and_the_rebuild_requests_are_seen():
    """The rebuild must ask for the same thing as the create. Seeing only one
    would let the other drift to a stale pin unnoticed."""
    source = (
        '_GYM_PYTHON = ">=3.13.14"\n'
        'subprocess.run(["uv", "venv", "--python", _GYM_PYTHON])\n'
        'subprocess.run(["uv", "venv", "--python", _GYM_PYTHON, "--clear"])\n'
    )
    assert _python_requests(source) == [">=3.13.14", ">=3.13.14"]
