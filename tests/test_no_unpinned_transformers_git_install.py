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

"""Installing transformers from git means naming the commit.

`git+https://github.com/huggingface/transformers` with no `@ref` is whatever
upstream's default branch holds the minute a reader runs the notebook. Two
things follow. The notebook stops being reproducible, which is the ordinary
cost. And in the molab variant the dependency is not a cell at all: it is
declared in the PEP 723 header, which uv resolves and *builds* before the first
line of the notebook runs, so the build backend of an unreviewed upstream
revision executes in a runtime holding the reader's Hugging Face token.

The Liquid LFM2 and Falcon H1 cells did this for as long as `lfm2` and
`falcon_h1` had no tagged release. They have one now and pin it. A pinned
`@<sha>` is fine and stays fine: it names one immutable tree.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# The notebook roots `scripts/notebook_inventory.py` reads, plus the two
# generated trees. `kaggle/` holds hand-committed Kaggle exports rather than
# generated ones, which is exactly why it needs its own check.
SEARCH_ROOTS = ("nb", "original_template", "kaggle", "python_scripts", "molab")

# `.git` is optional, and the ref runs to whitespace, a quote or the `#` of a
# `#subdirectory=`. Slashes are inside the ref so that `@refs/heads/main` is
# read whole; capturing only `@refs` would read a branch as a pin. A backslash
# ends it too: in `python_scripts/*.py` these lines sit inside a quoted string,
# where the newline after the ref is the two characters `\` and `n`.
_TRANSFORMERS_GIT = re.compile(
    r"git\+https://github\.com/huggingface/transformers(?:\.git)?"
    r"(@[^\s\\\"'#,]+)?")
# What is allowed, not what is forbidden: a branch moves whatever it is called,
# so `@release-5.x` and `@refs/heads/main` have to fail the same way `@main`
# does. That leaves a commit (a short sha is still one tree, so length is not
# the test) or a release tag, optionally with an `-rc1` style suffix.
_IMMUTABLE = re.compile(r"[0-9a-f]{7,40}|v?\d+(?:\.\d+)*(?:-[\w.]+)?",
                        re.IGNORECASE)

# Still unpinned, tracked rather than silently tolerated. These predate the
# Liquid LFM2 / Falcon H1 pin and are a separate change: their install lines
# are hand-maintained and untested against a release.
KNOWN_UNPINNED = {
    "AMD-GPT_OSS_BNB_(20B)-Inference",
    "AMD-GPT_OSS_MXFP4_(20B)-Inference",
    "GPT_OSS_BNB_(20B)-Inference",
    "GPT_OSS_MXFP4_(20B)-Inference",
}


def _text(path):
    """Notebook code cells joined, or the file itself for a script."""
    if path.suffix != ".ipynb":
        return path.read_text(encoding = "utf-8")
    notebook = json.loads(path.read_text(encoding = "utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _files():
    for root in SEARCH_ROOTS:
        directory = REPO_ROOT / root
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix in (".ipynb", ".py"):
                yield path


_CASES = [(f"{path.parent.name}/{path.name}", _text(path))
          for path in _files()
          if _TRANSFORMERS_GIT.search(_text(path))]


def test_some_file_installs_transformers_from_git():
    """Or the check below is vacuously green forever."""
    assert _CASES


@pytest.mark.parametrize("name, text", _CASES,
                         ids = lambda v: v if isinstance(v, str) and len(v) < 80 else "")
def test_transformers_git_installs_name_a_commit(name, text):
    if Path(name).stem in KNOWN_UNPINNED:
        pytest.skip(f"{name} is a known unpinned install, tracked in KNOWN_UNPINNED")
    for ref in _TRANSFORMERS_GIT.findall(text):
        assert ref, (
            f"{name} installs transformers from git with no @ref, so it builds "
            "whatever upstream's default branch holds at run time. In molab that "
            "happens in the PEP 723 header, before the notebook body runs. Pin a "
            "release with == or name a commit with @<sha>"
        )
        assert _IMMUTABLE.fullmatch(ref.lstrip("@")), (
            f"{name} pins transformers to {ref}, which is a branch and moves "
            "like the default one does. Name a commit or a release tag"
        )


def test_the_pattern_reads_a_pinned_ref_as_pinned():
    """A sha spelling that read as unpinned would fail every notebook."""
    pinned = ("!pip install git+https://github.com/huggingface/transformers.git"
              "@bf3f0ae70d0e902efab4b8517fce88f6697636ce")
    assert _TRANSFORMERS_GIT.findall(pinned) == [
        "@bf3f0ae70d0e902efab4b8517fce88f6697636ce"]


def test_the_pattern_reads_a_bare_url_as_unpinned():
    """And the shape this file exists for still reports as unpinned."""
    bare = "!pip install --no-deps git+https://github.com/huggingface/transformers.git"
    assert _TRANSFORMERS_GIT.findall(bare) == [""]


@pytest.mark.parametrize("ref", [
    "bf3f0ae70d0e902efab4b8517fce88f6697636ce",
    "bf3f0ae",
    "v5.15.0",
    "5.15.0",
    "v5.15.0-rc1",
])
def test_a_commit_or_a_release_tag_is_immutable(ref):
    assert _IMMUTABLE.fullmatch(ref)


@pytest.mark.parametrize("ref", [
    "main",
    "master",
    "release-5.x",
    "5.x",
    "refs/heads/main",
    "my-branch",
])
def test_a_branch_is_not_immutable_whatever_it_is_called(ref):
    """A name blacklist passed `@release-5.x` and read `@refs/heads/main` as
    `@refs`, so the gate admitted the mutable installs it exists to stop."""
    url = f"git+https://github.com/huggingface/transformers.git@{ref}"
    assert _TRANSFORMERS_GIT.findall(url) == [f"@{ref}"]
    assert not _IMMUTABLE.fullmatch(ref)
