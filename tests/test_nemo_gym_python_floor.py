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

These notebooks clone `NVIDIA-NeMo/Gym` at unpinned HEAD and asked for
`--python 3.12`. On 2026-08-04 upstream raised its own floor to `>=3.13.14`, so
`uv sync` refused the interpreter, and `subprocess.run(..., check = True)` turned
that into a `CalledProcessError` before any training.

A fixed version is the wrong shape of fix twice over: it goes stale the next
time upstream moves, and `--python 3.13` resolves to the newest 3.13 to hand
(3.13.8 here), still under the floor. A specifier lets uv pick and provision
something that satisfies whatever upstream currently declares. Deliberately no
upper bound, which would recreate the same staleness in the test.
"""

import ast
import json
import os
import re
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

GYM_REPO = "NVIDIA-NeMo/Gym"

# `"--python", <request>` in a `uv venv` argument list. A name as well as a
# literal, because the notebook requests twice -- create and `--clear` rebuild --
# and two literals could drift apart. Names resolve from their assignment.
_RE_PYTHON_REQUEST = re.compile(r'"--python",\s*("[^"]+"|[A-Za-z_]\w*)')


def _python_requests(source):
    requests = []
    for raw in _RE_PYTHON_REQUEST.findall(source):
        if raw.startswith('"'):
            requests.append(raw.strip('"'))
            continue
        assignment = re.search(rf'^\s*{re.escape(raw)}\s*=\s*"([^"]+)"', source, re.M)
        # Unresolvable is reported as-is, so it fails the checks below rather
        # than vanishing and leaving them vacuously true.
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
    """If the clone URL changes, every case below silently stops running."""
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
    """Upstream required 3.13.14 as of 2026-08-04; a specifier admitting
    anything older is no better than the 3.12 pin it replaced."""
    for request in _python_requests(_code(path)):
        floor = tuple(int(n) for n in re.findall(r"\d+", request)[:3])
        assert floor >= (3, 13, 14), (
            f"{path.name} asks for {request!r}, which admits interpreters below "
            f"the 3.13.14 upstream requires"
        )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_no_stale_prose_promises_the_old_floor(path):
    """The markdown said 'requires Python 3.12+', which sends a reader who
    hits the failure in the wrong direction."""
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
    """Held here, not only against the tree, so the check keeps meaning
    something once every notebook is compliant."""
    assert not "3.12".startswith(">=")
    assert not "3.13".startswith(">=")
    assert ">=3.13.14".startswith(">=")


def test_the_floor_check_rejects_a_specifier_that_is_too_low():
    for request in (">=3.12", ">=3.13", ">=3.13.8"):
        floor = tuple(int(n) for n in re.findall(r"\d+", request)[:3])
        assert not floor >= (3, 13, 14), request


# The setup must repair a venv a previous failed run left behind. `uv venv`
# succeeds and `uv sync` is the half that fails, so everyone who hit the floor
# error has a complete-looking `.venv/bin/python` holding the wrong interpreter;
# guarding the sync on that path existing skips exactly them. uv also refuses to
# overwrite a venv without `--clear`, so the rebuild has to pass it.


def _strip_comments(source):
    """Comments out, because this file's own comments name every token it
    searches for: deleting `--clear` from the argument list still passed on the
    strength of a comment mentioning it."""
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


# The rebuild is destructive, so it belongs behind the failure it repairs. A
# sync also fails for reasons that have nothing to do with the interpreter -- an
# index outage, a resolution conflict, a dropped download -- and clearing the
# venv there discards a working environment and retries the same losing command,
# leaving the user with neither. The condition is the venv's own interpreter
# against the floor, because uv's error text is free to be reworded.

_FLOOR_CHECK = "_venv_python_satisfies_floor"
_PROJECT_FLOOR = "_gym_requires_python"


def _rebuild_guard(source):
    """The `if` statement that guards the `uv venv --clear` rebuild."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if "'--clear'" in body:
            return node
    return None


def _function_source(source, name):
    function = next(
        (
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    assert function is not None, f"{name} is gone"
    return ast.get_source_segment(source, function)


def _run_floor_check(source, reported, exists=True, returncode=0, gym_dir="/gym"):
    """Run the notebook's own interpreter check against a stubbed venv.

    `gym_dir` is a real directory, so the check reads the `pyproject.toml` it
    holds -- that file is the floor being tested.
    """
    probe = types.SimpleNamespace(returncode=returncode, stdout=reported + "\n")
    namespace = {
        "os": types.SimpleNamespace(
            path=types.SimpleNamespace(join=os.path.join, exists=lambda p: exists),
        ),
        "re": re,
        "subprocess": types.SimpleNamespace(run=lambda *a, **k: probe),
        "GYM_DIR": gym_dir,
        "_GYM_PYTHON": _python_requests(source)[0],
    }
    exec(_function_source(source, _PROJECT_FLOOR), namespace)
    exec(_function_source(source, _FLOOR_CHECK), namespace)
    return namespace[_FLOOR_CHECK]()


def _checkout(tmp_path, requires_python):
    """A Gym checkout on disk declaring `requires-python`."""
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "nemo-gym"\nrequires-python = "{requires_python}"\n',
        encoding="utf-8",
    )
    return str(tmp_path)


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_the_rebuild_is_gated_on_the_interpreter_being_below_the_floor(path):
    """Without the gate, any failed sync deletes a working venv and retries
    the command that just failed."""
    source = _setup_source(path)
    guard = _rebuild_guard(source)
    assert guard is not None, f"{path.name}: no `if` guards the --clear rebuild"
    negated = [
        node.operand.func.id
        for node in ast.walk(guard.test)
        if isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Call)
        and isinstance(node.operand.func, ast.Name)
    ]
    assert _FLOOR_CHECK in negated, (
        f"{path.name} clears the venv on any failed sync. Rebuild only when "
        f"`not {_FLOOR_CHECK}()`, or an unrelated failure costs the user a "
        f"working environment."
    )
    assert "returncode" in ast.dump(guard.test), (
        f"{path.name} no longer keys the rebuild on the sync having failed"
    )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
@pytest.mark.parametrize(
    "reported, satisfied",
    [
        ("3.12.11", False),   # what the original failure leaves behind
        ("3.13.8", False),    # what a bare `--python 3.13` resolved to
        ("3.13.14", True),    # exactly the floor
        ("3.14.3", True),     # what the rebuild produced here
    ],
)
def test_the_interpreter_check_answers_the_floor_correctly(path, reported, satisfied):
    """Run the check itself: a comparison the wrong way round either never
    repairs the 3.12 venv or clears a healthy one."""
    assert _run_floor_check(_setup_source(path), reported) is satisfied


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_an_unreadable_venv_counts_as_not_satisfying_the_floor(path):
    """A half-finished setup leaves a missing or broken interpreter, which
    must fall through to the rebuild rather than raise."""
    source = _setup_source(path)
    assert _run_floor_check(source, "3.14.3", exists=False) is False
    assert _run_floor_check(source, "", returncode=1) is False


# Which floor the venv is measured against matters as much as the comparison.
# Step 1 never updates an existing checkout, so one made before 2026-08-04 still
# declares `>=3.12` and its 3.12 venv is correct for it. Held to the notebook's
# newer constant, that venv reads as stale and the first unrelated sync failure
# deletes it -- the loss the gate above exists to prevent.


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_venv_valid_for_the_checkout_survives_an_unrelated_sync_failure(path, tmp_path):
    gym = _checkout(tmp_path, ">=3.12")
    assert _run_floor_check(_setup_source(path), "3.12.11", gym_dir=gym) is True


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_venv_below_the_checkouts_own_floor_is_still_rebuilt(path, tmp_path):
    """The repair this guard exists for: a fresh checkout declares 3.13.14 and
    a leftover 3.12 venv has to go."""
    gym = _checkout(tmp_path, ">=3.13.14")
    assert _run_floor_check(_setup_source(path), "3.12.11", gym_dir=gym) is False
    assert _run_floor_check(_setup_source(path), "3.13.14", gym_dir=gym) is True


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_an_unreadable_checkout_falls_back_to_the_notebook_floor(path, tmp_path):
    """No pyproject to read, so the notebook's constant is the only floor left
    and must still condemn the original failure's 3.12 venv."""
    assert _run_floor_check(_setup_source(path), "3.12.11", gym_dir=str(tmp_path)) is False


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_requirement_with_no_lower_bound_keeps_the_venv(path, tmp_path):
    """`<4` bounds nothing from below, so nothing is stale against it and
    clearing is the destructive default again."""
    gym = _checkout(tmp_path, "<4")
    assert _run_floor_check(_setup_source(path), "3.9.1", gym_dir=gym) is True


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_compound_requirement_reads_only_its_lower_bound(path, tmp_path):
    """`>=3.13.14,<3.15` means 3.13.14; reading it as 3.15 condemns a valid
    3.13 venv."""
    gym = _checkout(tmp_path, ">=3.13.14,<3.15")
    assert _run_floor_check(_setup_source(path), "3.13.14", gym_dir=gym) is True
    assert _run_floor_check(_setup_source(path), "3.13.8", gym_dir=gym) is False


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_a_failed_sync_surfaces_its_output(path):
    """`capture_output` with no print leaves the same bare CalledProcessError
    this change exists to explain."""
    source = _setup_source(path)
    assert "_sync.stdout" in source and "_sync.stderr" in source, (
        f"{path.name} captures the sync output but never prints it on failure"
    )
    assert "check_returncode" in source, f"{path.name} never fails on a bad sync"


def test_an_unresolvable_name_is_not_silently_dropped():
    """A name with no assignment must survive so the specifier check fails on
    it rather than passing vacuously."""
    source = 'subprocess.run(["uv", "venv", "--python", _MISSING])'
    assert _python_requests(source) == ["_MISSING"]
    assert not _python_requests(source)[0].startswith(">=")


def test_a_name_is_resolved_to_its_assignment():
    source = '_GYM_PYTHON = ">=3.13.14"\nsubprocess.run(["uv", "venv", "--python", _GYM_PYTHON])'
    assert _python_requests(source) == [">=3.13.14"]


def test_both_the_create_and_the_rebuild_requests_are_seen():
    """Seeing only one request would let the other drift to a stale pin."""
    source = (
        '_GYM_PYTHON = ">=3.13.14"\n'
        'subprocess.run(["uv", "venv", "--python", _GYM_PYTHON])\n'
        'subprocess.run(["uv", "venv", "--python", _GYM_PYTHON, "--clear"])\n'
    )
    assert _python_requests(source) == [">=3.13.14", ">=3.13.14"]


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_uv_is_refreshed_before_the_interpreter_is_requested(path):
    """Colab ships a uv whose embedded download list predates every 3.13.14
    build, so `>=3.13.14` matches no interpreter and the venv step dies. Seen on
    a Colab L4 after the specifier landed; uv's own hint is to update it."""
    source = _setup_source(path)
    assert "_refresh_uv()" in source, f"{path.name} never refreshes uv"
    before, _, after = source.partition("_refresh_uv()\n")
    # It must precede every `uv venv`, or the stale uv answers.
    assert '"uv", "venv"' not in before, (
        f"{path.name} asks for an interpreter before refreshing uv"
    )
    assert '"uv", "venv"' in after, f"{path.name} never creates the venv"


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_the_uv_refresh_has_a_pip_fallback(path):
    """`uv self update` is refused by a pip-installed uv, which is how some
    images ship it."""
    source = _setup_source(path)
    assert '"self", "update"' in source, f"{path.name}: no standalone update path"
    assert "pip" in source.split("_refresh_uv", 1)[1][:800], (
        f"{path.name}: no pip fallback beside the self update"
    )


@pytest.mark.parametrize("path", _GYM, ids=lambda p: p.name)
def test_the_refresh_cannot_abort_the_cell(path):
    """A raising refresh would break images where uv is current and `self
    update` legitimately fails; the venv step reports the real problem."""
    source = _setup_source(path)
    body = source.split("def _refresh_uv():", 1)[1].split("_refresh_uv()", 1)[0]
    assert "check = True" not in body and "check=True" not in body, (
        f"{path.name}: the uv refresh can abort the cell"
    )
