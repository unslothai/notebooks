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

"""A trackio TrainingArguments field a pinned transformers has never heard of.

`Openenv_wordle_grpo.ipynb` pinned `transformers==4.56.2` and then built its
`GRPOConfig` with

    trackio_space_id = 'outputs',

which stops the notebook dead ten cells past the install:

    TypeError: GRPOConfig.__init__() got an unexpected keyword argument
    'trackio_space_id'

`GRPOConfig` is a dataclass over `transformers.TrainingArguments`, so its
accepted keywords are whatever the *installed* transformers declares. 4.56.2
configured the Space through the `TRACKIO_SPACE_ID` environment variable and
had no such field; 4.57.0 added it. Unsloth reflects the installed
`TrainingArguments` when it generates `UnslothGRPOConfig`, so it forwards the
keyword faithfully and TRL is right to reject it.

The floors below were read out of the wheels, not from release notes:

    field                     first transformers with it
    project                   4.57.0
    trackio_space_id          4.57.0
    trackio_static_space_id   5.6.0

(absent in 4.56.2; present in 4.57.0, 4.57.6, 5.0.0, 5.5.4, 5.6.0, 5.14.1 for
the first two, and in 5.6.0 and 5.14.1 for the third).

There is deliberately no assertion that some notebook still uses one of these:
the fix for the wordle notebook was to stop passing the field, so the correct
steady state is zero live uses. The detectors are exercised against synthetic
sources instead, both ways, so a regex that stopped matching cannot leave this
gate quietly green.
"""

import json
import re
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIRS = ("nb", "kaggle", "original_template")
SCRIPT_DIRS = ("python_scripts", "molab")

# Measured from the published wheels, one row per field.
TRACKIO_ARGUMENT_FLOORS = {
    "project": "4.57.0",
    "trackio_space_id": "4.57.0",
    "trackio_static_space_id": "5.6.0",
}

# Only exact pins are decided here. `transformers>=4.51` admits a version that
# has the field, so nothing can be concluded from it.
_TRANSFORMERS_PIN = re.compile(
    r"transformers\s*==\s*v?([0-9]+(?:\.[0-9]+)*(?:[.\-]?(?:dev|a|b|rc)[0-9]*)?)",
    re.IGNORECASE)

_KWARG = {name: re.compile(r"(?<![\w.])" + name + r"\s*=(?!=)")
          for name in TRACKIO_ARGUMENT_FLOORS}


def _strip_comment(line):
    """Drop a trailing `#` comment, ignoring `#` inside a string literal."""
    out = []
    quote = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is None:
            if char == "#":
                break
            if char in "\"'":
                quote = char
        else:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        out.append(char)
        index += 1
    return "".join(out)


def _code_text(source):
    """`source` with comment tails removed, so prose cannot trip a detector."""
    return "\n".join(_strip_comment(line) for line in source.splitlines())


def _pinned_transformers(text):
    """The lowest exact `transformers==` pin in `text`, or None."""
    found = [Version(match.group(1)) for match in _TRANSFORMERS_PIN.finditer(text)]
    return min(found) if found else None


def _trackio_kwargs_used(text):
    return sorted(name for name, pattern in _KWARG.items() if pattern.search(text))


def _sources(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding = "utf-8"))
        return ["".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"]
    return [path.read_text(encoding = "utf-8")]


def _candidates():
    for directory in NOTEBOOK_DIRS:
        for path in sorted((REPO_ROOT / directory).glob("*.ipynb")):
            yield path
    for directory in SCRIPT_DIRS:
        root = REPO_ROOT / directory
        if root.is_dir():
            for path in sorted(root.glob("*.py")):
                yield path


_CASES = [(str(path.relative_to(REPO_ROOT)), path) for path in _candidates()]


def test_there_are_files_to_check():
    """A glob that stopped matching would make every check below vacuous."""
    assert len(_CASES) > 100


@pytest.mark.parametrize("name, path", _CASES, ids = [name for name, _ in _CASES])
def test_trackio_arguments_exist_in_the_pinned_transformers(name, path):
    text = _code_text("\n".join(_sources(path)))
    used = _trackio_kwargs_used(text)
    if not used:
        return
    pinned = _pinned_transformers(text)
    if pinned is None:
        return
    for field in used:
        floor = Version(TRACKIO_ARGUMENT_FLOORS[field])
        assert pinned >= floor, (
            f"{name} pins transformers=={pinned} and passes {field} = ... . "
            f"That TrainingArguments field arrived in transformers {floor}, so "
            f"the dataclass built on the pinned version has no such keyword and "
            f"the config raises TypeError: __init__() got an unexpected keyword "
            f"argument '{field}' before a line of training. Either raise the pin "
            f"to {floor} or stop passing the field")


# --- detector self-tests -------------------------------------------------

def test_detector_flags_the_shape_that_broke_the_wordle_notebook():
    text = _code_text(
        '!uv pip install --no-deps transformers==4.56.2 trl==0.29.1\n'
        'grpo_config = GRPOConfig(\n'
        '    report_to = "trackio",\n'
        "    trackio_space_id = 'outputs',\n"
        ")\n")
    assert _trackio_kwargs_used(text) == ["trackio_space_id"]
    assert _pinned_transformers(text) == Version("4.56.2")


def test_detector_accepts_the_field_on_a_transformers_that_has_it():
    text = _code_text(
        '!pip install transformers==4.57.6\n'
        '    trackio_space_id = "unsloth/runs",\n')
    assert _trackio_kwargs_used(text) == ["trackio_space_id"]
    assert _pinned_transformers(text) >= Version("4.57.0")


def test_a_commented_out_mention_is_not_a_use():
    text = _code_text(
        '!pip install transformers==4.56.2\n'
        '    # or trackio_space_id = "user/space" here on 4.57 and later.\n')
    assert _trackio_kwargs_used(text) == []


def test_reading_the_attribute_is_not_passing_the_keyword():
    text = _code_text("print(args.trackio_space_id)\nx == trackio_space_id\n")
    assert _trackio_kwargs_used(text) == []


def test_the_lowest_pin_in_a_file_is_the_one_that_decides():
    text = _code_text('!pip install transformers==4.57.6\n'
                      '!pip install --no-deps transformers==4.56.2\n')
    assert _pinned_transformers(text) == Version("4.56.2")


def test_an_open_ended_requirement_is_not_an_exact_pin():
    assert _pinned_transformers('!pip install "transformers>=4.51.0"') is None


def test_every_floor_is_a_parseable_version():
    for field, floor in TRACKIO_ARGUMENT_FLOORS.items():
        assert Version(floor) >= Version("4.57.0"), field
