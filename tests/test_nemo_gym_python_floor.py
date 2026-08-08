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

# `"--python", "<request>"` as it appears in a `uv venv` argument list.
_RE_PYTHON_REQUEST = re.compile(r'"--python",\s*"([^"]+)"')


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
    requests = _RE_PYTHON_REQUEST.findall(_code(path))
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
    for request in _RE_PYTHON_REQUEST.findall(_code(path)):
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
