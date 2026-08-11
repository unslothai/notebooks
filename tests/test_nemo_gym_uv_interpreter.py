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

"""The NeMo Gym notebooks must let uv fetch the interpreter the checkout pins.

NeMo Gym ships `.python-version` holding a single patch release (3.13.14 as of
upstream ea4c6c6) and `uv sync` honours that file, so it accepts no other
interpreter. The uv preinstalled on Colab predates that release, so `uv sync`
exits 2 with "No interpreter found for Python 3.13.14 in managed installations
or search path" and the setup cell takes the whole notebook down with it.

Two near-misses this guards against: naming an interpreter in the notebook
(`uv venv --python 3.12`) rather than letting the checkout name it, which is
wrong the moment upstream moves; and satisfying only the
`requires-python = ">=3.13.14"` floor, where uv hands back the newest release
it knows, a 3.14, and `uv sync` dies compiling yappi (no cp314 wheel).

Ordering matters too: uv has to be refreshed before it is asked for the
interpreter, and called through the binary the fresh wheel ships, since the
name `uv` still resolves to the stale copy earlier on PATH.

The uv install is left unpinned on purpose. uv embeds the list of Python
builds it can fetch, so pinning it below MIN_UV brings the failure above
straight back: pin at or above MIN_UV, or not at all.

nb/, python_scripts/ and molab/ are covered together: the last two are
generated mirrors, so a one-layer fix disappears on the next regeneration.
"""

import ast
import json
import re
from pathlib import Path

import pytest
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every artifact layer that can carry the NeMo Gym bootstrap.
SEARCH_DIRS = ["nb", "kaggle", "python_scripts", "molab", "original_template"]

GYM_CLONE_URL = "https://github.com/NVIDIA-NeMo/Gym.git"

# Oldest uv carrying cpython-3.13.14, the interpreter Gym/.python-version
# names. Measured per release in a throwaway venv against
# `uv python list --all-versions`: 0.11.20 lacks it, 0.11.21 has it. Below
# this, uv sync exits 2 with "No interpreter found for Python 3.13.14 in
# managed installations or search path". Bump alongside Gym/.python-version.
MIN_UV = Version("0.11.21")

# The checks below run over whatever discovery finds; this list only catches
# discovery silently finding nothing, which would leave them with no cases.
EXPECTED_FILES = {
    "molab/NeMo-Gym-Multi-Environment.py",
    "molab/NeMo-Gym-Sudoku.py",
    "nb/AMD-NeMo-Gym-Multi-Environment.ipynb",
    "nb/AMD-NeMo-Gym-Sudoku.ipynb",
    "nb/NeMo-Gym-Multi-Environment.ipynb",
    "nb/NeMo-Gym-Sudoku.ipynb",
    "python_scripts/AMD-NeMo-Gym-Multi-Environment.py",
    "python_scripts/AMD-NeMo-Gym-Sudoku.py",
    "python_scripts/NeMo-Gym-Multi-Environment.py",
    "python_scripts/NeMo-Gym-Sudoku.py",
}


def _notebook_text(path: Path) -> str:
    """Concatenated code-cell source for a notebook, raw text otherwise."""
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8")
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        chunks.append(source if isinstance(source, str) else "".join(source))
    return "\n".join(chunks)


def _discover():
    found = {}
    for folder in SEARCH_DIRS:
        directory = REPO_ROOT / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix not in (".ipynb", ".py"):
                continue
            text = _notebook_text(path)
            if GYM_CLONE_URL in text:
                found[str(path.relative_to(REPO_ROOT))] = text
    return found


GYM_FILES = _discover()


def _holds_uv_below_min(spec):
    """Does `spec` rule out every uv that can fetch the pinned interpreter?

    Answered through packaging rather than by hand so PEP 440 ordering is the
    real one: `<=0.11.21rc1` reads as below `0.11.21`, not equal to it.

    A spec is fine as long as SOME version it admits is at or above MIN_UV,
    since pip is being run with `--upgrade` and takes the newest match. Three
    probes settle that: MIN_UV covers ranges spanning it, a sentinel covers
    specs open at the top, and the spec's own versions cover exact pins.
    """
    try:
        specifier = SpecifierSet(spec)
    except InvalidSpecifier:
        return True

    probes = [MIN_UV, Version("9999")]
    for clause in specifier:
        try:
            probes.append(Version(clause.version.rstrip(".*")))
        except InvalidVersion:
            continue
    return not any(
        probe >= MIN_UV and specifier.contains(probe, prereleases = True)
        for probe in probes
    )


def test_gym_bootstrap_files_are_all_present():
    """Discovery has to keep finding the notebooks this file is about."""
    missing = sorted(EXPECTED_FILES - set(GYM_FILES))
    assert not missing, (
        "DRIFT DETECTED: these NeMo Gym artifacts no longer clone "
        f"{GYM_CLONE_URL}, so nothing below is checking them: {missing}"
    )


def test_every_behavioural_check_runs_over_what_discovery_found():
    """Parameterizing over EXPECTED_FILES reads as equivalent while the two
    sets match, then silently stops covering the next notebook that clones Gym.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    over = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call)
                    and getattr(dec.func, "attr", "") == "parametrize"):
                continue
            if getattr(dec.args[0], "value", None) != "relpath":
                continue
            over[node.name] = ast.unparse(dec.args[1])
    assert over, "no relpath-parameterized checks found; this test is vacuous"
    wrong = {name: src for name, src in over.items() if "GYM_FILES" not in src}
    assert not wrong, f"these run over a fixed list, not discovery: {wrong}"


@pytest.mark.parametrize("relpath", sorted(GYM_FILES))
def test_no_hardcoded_interpreter_request(relpath):
    """The notebook must not name a Python version for the Gym venv."""
    text = GYM_FILES[relpath]

    hardcoded = re.findall(r"--python[\"'\s,]+[0-9][0-9.]*", text)
    assert not hardcoded, (
        f"DRIFT DETECTED: {relpath} pins the NeMo Gym interpreter itself "
        f"({hardcoded}). Gym/.python-version already names the exact patch "
        "release uv sync will accept, and anything hardcoded here is wrong as "
        "soon as upstream moves it."
    )

    floor = re.findall(r"--python[\"'\s,]+[<>=!~]+", text)
    assert not floor, (
        f"DRIFT DETECTED: {relpath} asks uv for a version range "
        f"({floor}). uv answers a range with the newest release it knows, "
        "which is a 3.14, and uv sync then fails building yappi from source "
        "because there is no cp314 wheel."
    )

    assert "uv\", \"venv\"" not in text and "uv', 'venv'" not in text, (
        f"DRIFT DETECTED: {relpath} still runs `uv venv` separately. `uv sync` "
        "creates Gym/.venv itself and rebuilds it when the interpreter inside "
        "does not match the pin, which is what repairs the venv a failed "
        "earlier attempt leaves behind."
    )


@pytest.mark.parametrize("relpath", sorted(GYM_FILES))
def test_uv_is_refreshed_then_asked_for_the_interpreter(relpath):
    """uv must be upgraded, addressed by path, and told to fetch the pin."""
    text = GYM_FILES[relpath]

    upgrade = re.search(
        r"pip[\"'\s,]+.*install.*--upgrade[\"'\s,]+[\"']uv([^\"']*)[\"']", text
    )
    assert upgrade, (
        f"DRIFT DETECTED: {relpath} does not upgrade uv before using it. The "
        "uv preinstalled on Colab has no 3.13.14 in its embedded download "
        "list, so uv sync exits 2 with 'No interpreter found for Python "
        "3.13.14 in managed installations or search path'."
    )

    # A constraint is allowed only if it can still reach the release that
    # first carries the interpreter. No constraint, which is what ships,
    # gets the newest uv.
    spec = upgrade.group(1)
    assert not _holds_uv_below_min(spec), (
        f"DRIFT DETECTED: {relpath} holds uv at `uv{spec}`, entirely below "
        f"{MIN_UV}. uv ships its list of "
        "installable Python builds inside the release, so a uv this old has "
        "no cpython-3.13.14 to fetch and uv sync exits 2 with 'No interpreter "
        "found for Python 3.13.14 in managed installations or search path' "
        "before the notebook gets anywhere. Pinning uv against a compromised "
        "release is a reasonable thing to want, but it has to be a floor at "
        "or above MIN_UV rather than a pin at a release that predates the "
        "interpreter, and MIN_UV moves when Gym/.python-version moves."
    )

    assert "find_uv_bin(" in text, (
        f"DRIFT DETECTED: {relpath} does not resolve uv with "
        "uv.find_uv_bin(). Calling plain `uv` after the upgrade can still "
        "reach the stale copy that is already earlier on PATH."
    )

    py_install = re.search(
        r"\[\s*_UV\s*,\s*[\"']python[\"']\s*,\s*[\"']install[\"']", text
    )
    assert py_install, (
        f"DRIFT DETECTED: {relpath} no longer runs `uv python install` from "
        "the Gym checkout, which is what provisions the interpreter named in "
        "Gym/.python-version."
    )

    sync = re.search(r"\[\s*_UV\s*,\s*[\"']sync[\"']", text)
    assert sync, (
        f"DRIFT DETECTED: {relpath} no longer runs `uv sync` through the "
        "refreshed uv binary."
    )

    assert upgrade.start() < py_install.start() < sync.start(), (
        f"DRIFT DETECTED: {relpath} runs uv before refreshing it. The order "
        "has to be upgrade uv, install the pinned interpreter, then sync."
    )


@pytest.mark.parametrize(
    "spec, capped",
    [
        ("", False),
        (">=0.11.21", False),
        # A bare floor still resolves to the newest uv under --upgrade.
        (">=0.10.7", False),
        ("==0.10.7", True),
        ("~=0.10.7", True),
        # Same prefix as MIN_UV, so 0.11.21 is still in range.
        ("~=0.11.5", False),
        ("<0.11.21", True),
        ("<0.11.22", False),
        (">=0.10.7,<0.11", True),
        (">=0.11.21,<0.12", False),
        ("==0.12.3", False),
        ("==0.11.*", False),
        ("==0.10.*", True),
        # PEP 440 sorts a release candidate below its own release, so this
        # cap stops at 0.11.20 even though it names 0.11.21.
        ("<=0.11.21rc1", True),
        (">=0.11.21rc1", False),
        ("==0.11.21rc1", True),
        ("==0.11.21.post1", False),
        ("not-a-specifier", True),
    ],
)
def test_the_stale_uv_guard_is_not_vacuous(spec, capped):
    """The guard above only helps if it can tell these specs apart."""
    assert _holds_uv_below_min(spec) is capped, (
        f"uv{spec} was read as {'capped' if not capped else 'open'}, expected "
        f"{'capped' if capped else 'open'}"
    )


@pytest.mark.parametrize("relpath", sorted(GYM_FILES))
def test_setup_is_not_skipped_by_a_venv_existence_guard(relpath):
    """A half-built .venv from a failed run must not skip the repair."""
    text = GYM_FILES[relpath]

    guard = re.search(
        r"if\s+not\s+os\.path\.exists\(\s*os\.path\.join\(\s*GYM_DIR", text
    )
    assert not guard, (
        f"DRIFT DETECTED: {relpath} guards the NeMo Gym setup on Gym/.venv "
        "not existing. `uv venv` is the half that succeeds and `uv sync` is "
        "the half that fails, so that guard skips the rebuild for exactly the "
        "people whose venv holds the wrong interpreter."
    )


@pytest.mark.parametrize("relpath", sorted(GYM_FILES))
def test_venv_location_is_verified_after_sync(relpath):
    """uv must not be trusted to have put the venv where the cell looks."""
    text = GYM_FILES[relpath]

    assert "_gym_venv_python" in text and "RuntimeError" in text, (
        f"DRIFT DETECTED: {relpath} does not check that Gym/.venv/bin/python "
        "exists after uv sync. Everything below the setup drives that path "
        "directly, so an environment built somewhere else fails several "
        "subprocesses later with an unrelated-looking error."
    )
