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
upstream's default branch holds the minute a reader runs the notebook. Beyond
the reproducibility cost, in molab that dependency is not a cell at all: the
PEP 723 header has uv *build* it before the notebook's first line, so an
unreviewed revision's build backend runs in a runtime holding the reader's
Hugging Face token.

The Liquid LFM2 and Falcon H1 cells did this until `lfm2` and `falcon_h1` had
a release to pin. A pinned `@<sha>` is fine: it names one immutable tree.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# `scripts/notebook_inventory.py`'s roots plus the generated trees. `kaggle/`
# is hand-committed rather than generated, so nothing else would catch it.
SEARCH_ROOTS = ("nb", "original_template", "kaggle", "python_scripts", "molab")

# The ref runs to whitespace, a quote, or the `#` of a `#subdirectory=`.
# Slashes stay inside it, or `@refs/heads/main` would be read as `@refs` and
# pass as a pin. So does a backslash: in `python_scripts/*.py` the line sits in
# a quoted string, where the trailing newline is the characters `\` and `n`.
_TRANSFORMERS_GIT = re.compile(
    r"git\+https://github\.com/huggingface/transformers(?:\.git)?"
    r"(@[^\s\\\"'#,]+)?")
# What is allowed, not what is forbidden: a branch moves whatever it is called,
# so `@release-5.x` has to fail the way `@main` does. That leaves a commit (a
# short sha is one tree too) or a release tag, `-rc1` suffix included.
_IMMUTABLE = re.compile(r"[0-9a-f]{7,40}|v?\d+(?:\.\d+)*(?:-[\w.]+)?",
                        re.IGNORECASE)

# Still unpinned, recorded rather than silently tolerated. They predate the
# Liquid LFM2 / Falcon H1 pin and are a separate change: hand-maintained
# install lines, untested against a release.
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
