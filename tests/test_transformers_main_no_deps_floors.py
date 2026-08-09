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

"""Installing transformers main with `--no-deps` means pinning its floors here.

`--no-deps` is deliberate on those cells: without it pip re-resolves torch and
replaces the CUDA build the runtime already has. The cost is that pip never
enforces a single thing transformers declares. It prints a WARNING and exits 0,
so the install looks fine and the notebook dies at the next import instead:

    safetensors>=0.8.0 is required for a normal functioning of this module,
    but found safetensors==0.7.0.

That is one requirement of nine. Fixing whichever one surfaced first only moves
the error to the next, which is how this same cell has now broken twice, on
`huggingface_hub` and then on `safetensors`. So this gate is written against
the whole requirement set rather than against the two names that have bitten.

`TRANSFORMERS_MAIN_REQUIREMENTS` is `install_requires` from transformers'
`setup.py`, resolved through `src/transformers/dependency_versions_table.py`,
at 5.15.0.dev0 (`e8ea728a`). `BASE_IMAGE_VERSIONS` is what Colab and Kaggle
actually shipped at the point the notebook reaches that cell, read off executed
notebooks rather than assumed. A requirement that either image fails is a
requirement the notebook has to pin; the rest are satisfied already and pinning
them would be noise.

There is no network here on purpose: a test that resolves transformers main at
run time fails on an unrelated upstream commit. When transformers moves a
floor, move it in the table below and the notebooks go red together.
"""

import json
import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"

# `install_requires` in transformers' setup.py at 5.15.0.dev0 (`e8ea728a`),
# with each name expanded through dependency_versions_table.py. `typer` is
# declared with no bound, so any installed version satisfies it.
TRANSFORMERS_MAIN_REQUIREMENTS = {
    "huggingface-hub": ">=1.5.0,<2.0",
    "numpy": ">=1.17",
    "packaging": ">=20.0",
    "pyyaml": ">=5.1",
    "regex": ">=2025.10.22",
    "tokenizers": ">=0.22.0,<=0.23.0",
    "typer": "",
    "safetensors": ">=0.8.0",
    "tqdm": ">=4.60",
}

# What each image has installed by the time the git-main install runs, which is
# not the same as what the image boots with: the Liquid LFM2 cells install
# `transformers==4.56.2` first, and that pulls `huggingface_hub` back under 1.0
# on both. Measured from executed notebooks, Colab A100 and Kaggle T4.
BASE_IMAGE_VERSIONS = {
    "colab": {
        "huggingface-hub": "0.36.2",
        "numpy": "2.0.2",
        "packaging": "26.2",
        "pyyaml": "6.0.3",
        "regex": "2025.11.3",
        "tokenizers": "0.22.2",
        "typer": "0.24.2",
        "safetensors": "0.8.0",
        "tqdm": "4.67.3",
    },
    "kaggle": {
        "huggingface-hub": "0.36.2",
        "numpy": "2.0.2",
        "packaging": "26.3",
        "pyyaml": "6.0.3",
        "regex": "2025.11.3",
        "tokenizers": "0.22.2",
        "typer": "0.24.2",
        # The one that broke the notebook. Colab ships 0.8.0, Kaggle 0.7.0, so
        # a Colab-only check reports this cell as healthy.
        "safetensors": "0.7.0",
        "tqdm": "4.67.3",
    },
}

# A version each requirement's own upper bound excludes. Under `--no-deps` an
# open-ended floor resolves whatever is newest, so `huggingface_hub>=1.5.0`
# alone installs hub 2.x the day it ships and transformers rejects it. A floor
# without the cap is half a pin, so the notebook's specifier has to reject
# these too. Requirements with no upper bound have nothing to sample.
ABOVE_THE_CAP = {
    "huggingface-hub": "2.0.0",
    "tokenizers": "0.24.0",
}

# The install this gate is about. `transformers==5.x` pins are a different
# shape and already covered by test_transformers5_hub_floor.py.
_GIT_MAIN_TRANSFORMERS = re.compile(
    r"--no-deps[^\n]*git\+https://github\.com/huggingface/transformers")


def _requirements_needing_a_pin():
    """Requirements at least one base image fails, so the cell must pin them."""
    needed = {}
    for name, requirement in TRANSFORMERS_MAIN_REQUIREMENTS.items():
        if not requirement:
            continue
        specifier = SpecifierSet(requirement)
        short = [image for image, versions in BASE_IMAGE_VERSIONS.items()
                 if not specifier.contains(Version(versions[name]),
                                           prereleases = True)]
        if short:
            needed[name] = (requirement, short)
    return needed


def _code_cells(path):
    notebook = json.loads(path.read_text(encoding = "utf-8"))
    return ["".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"]


def _install_cell_index(cells):
    """Index of the cell installing transformers main, or None."""
    for index, source in enumerate(cells):
        for line in source.splitlines():
            if line.lstrip().startswith("#"):
                continue
            if _GIT_MAIN_TRANSFORMERS.search(line):
                return index
    return None


# `huggingface_hub>=1.5.0,<2.0`, `"safetensors>=0.8.0"`, `-U tokenizers>=0.22`.
# Underscore and hyphen spellings are the same distribution to pip.
def _specifier_for(text, name):
    pattern = re.compile(
        re.escape(name).replace(r"\-", "[-_]")
        + r"\s*((?:[<>=!~]=?\s*[0-9][\w.*+!-]*\s*,?\s*)+)", re.IGNORECASE)
    clauses = []
    for match in pattern.finditer(text):
        clauses.append(match.group(1).strip().rstrip(","))
    if not clauses:
        return None
    return SpecifierSet(",".join(clauses))


def _notebooks_installing_transformers_main():
    for path in sorted(NB_DIR.glob("*.ipynb")):
        cells = _code_cells(path)
        index = _install_cell_index(cells)
        if index is not None:
            yield path.name, cells, index


_CASES = [(name, cells, index)
          for name, cells, index in _notebooks_installing_transformers_main()]


def test_some_notebook_installs_transformers_main_without_deps():
    """Or every check below is vacuously green forever."""
    assert _CASES


def test_the_requirement_set_still_asks_for_something():
    """Guard the guard. If a table edit ever leaves nothing unsatisfied, the
    parametrised check would pass while pinning nothing at all."""
    needed = _requirements_needing_a_pin()
    assert "safetensors" in needed, (
        "safetensors>=0.8.0 against Kaggle's 0.7.0 is the case this file was "
        "written for; if it stops being reported the comparison is broken")
    assert "huggingface-hub" in needed


def test_every_requirement_has_a_measurement_for_every_image():
    """A name in one table and not the other reads as satisfied."""
    for image, versions in BASE_IMAGE_VERSIONS.items():
        missing = set(TRANSFORMERS_MAIN_REQUIREMENTS) - set(versions)
        assert not missing, f"{image} has no measured version for {sorted(missing)}"


@pytest.mark.parametrize(
    "name, cells, install_index", _CASES,
    ids = lambda v: v if isinstance(v, str) and len(v) < 80 else "")
def test_unsatisfied_requirements_are_floored(name, cells, install_index):
    """Every requirement the base images fall short of is pinned in the cell."""
    text = "\n".join(cells)
    for package, (requirement, short_on) in _requirements_needing_a_pin().items():
        specifier = _specifier_for(text, package)
        assert specifier is not None, (
            f"{name} installs transformers main with --no-deps, which declares "
            f"{package}{requirement}, and never pins {package}. "
            f"{', '.join(short_on)} ship a version below that floor, so pip "
            f"reports a WARNING, exits 0, and the next import raises "
            f"\"{package}...is required for a normal functioning of this "
            f"module\" before a line of training")
        for image in short_on:
            stale = BASE_IMAGE_VERSIONS[image][package]
            assert not specifier.contains(Version(stale), prereleases = True), (
                f"{name} pins {package}{specifier} but {image}'s preinstalled "
                f"{stale} satisfies it, so pip leaves it in place and the "
                f"import fails anyway. Pin it to {requirement}")


@pytest.mark.parametrize(
    "name, cells, install_index", _CASES,
    ids = lambda v: v if isinstance(v, str) and len(v) < 80 else "")
def test_pins_respect_the_upper_bound(name, cells, install_index):
    """A floor with no cap resolves the next major the day it ships."""
    text = "\n".join(cells)
    for package, too_new in ABOVE_THE_CAP.items():
        specifier = _specifier_for(text, package)
        if specifier is None:
            continue
        assert not specifier.contains(Version(too_new), prereleases = True), (
            f"{name} pins {package}{specifier}, which admits {too_new}. "
            f"transformers main declares "
            f"{package}{TRANSFORMERS_MAIN_REQUIREMENTS[package]} and rejects "
            f"it at import, and under --no-deps nothing else holds it down")


@pytest.mark.parametrize(
    "name, cells, install_index", _CASES,
    ids = lambda v: v if isinstance(v, str) and len(v) < 80 else "")
def test_the_pins_are_not_left_behind_an_earlier_install(name, cells, install_index):
    """Order matters: these notebooks install `transformers==4.56.2` first, and
    that resolves `huggingface_hub` back under 1.0. A floor placed before it is
    undone by it, silently, and the cell looks correct in a text scan."""
    for package in _requirements_needing_a_pin():
        pinned_in = [index for index, source in enumerate(cells)
                     if _specifier_for(source, package) is not None]
        assert any(index >= install_index for index in pinned_in), (
            f"{name} pins {package} only in cell(s) {pinned_in}, all before the "
            f"transformers main install in cell {install_index}. An earlier "
            f"pinned transformers re-resolves it downward and the floor is lost")


def test_a_floor_below_the_shipped_version_is_not_a_floor():
    """The discriminating case for the comparison itself: `safetensors>=0.4.3`
    reads as a pin and changes nothing, because Kaggle's 0.7.0 satisfies it."""
    specifier = _specifier_for('!pip install "safetensors>=0.4.3"', "safetensors")
    assert specifier is not None
    assert specifier.contains(Version("0.7.0"))
    assert _specifier_for('!pip install "safetensors>=0.8.0"',
                          "safetensors").contains(Version("0.7.0")) is False


def test_the_underscore_spelling_is_read_as_the_same_package():
    """Notebooks write `huggingface_hub`; transformers declares
    `huggingface-hub`. Missing that reports a pinned notebook as unpinned."""
    text = '!pip install "huggingface_hub>=1.5.0,<2.0"'
    specifier = _specifier_for(text, "huggingface-hub")
    assert specifier is not None
    assert not specifier.contains(Version("0.36.2"))
    assert not specifier.contains(Version("2.0.0"))
    assert specifier.contains(Version("1.27.0"))


def test_a_commented_out_install_is_not_an_install():
    """A cell that only mentions the command in prose must not be gated."""
    assert _install_cell_index([
        "# !pip install --no-deps git+https://github.com/huggingface/transformers.git"
    ]) is None
    assert _install_cell_index([
        "!pip install --no-deps git+https://github.com/huggingface/transformers.git"
    ]) == 0
