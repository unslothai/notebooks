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

"""Sample images a notebook downloads have to still be there, and say so.

`Gemma3N_(2B)-Inference` fetched its first sample image from
`sgl-project/sglang` at `refs/heads/main/test/lang/example_image.png`.
sgl-project/sglang#13448 moved that file to `examples/assets/`, so the branch
URL started answering 404 with a text body, and the notebook reported

    UnidentifiedImageError: cannot identify image file <_io.BytesIO ...>

from inside PIL, naming neither the URL nor the status, and then
`NameError: name 'image' is not defined` two cells later. Two guards:

  1. every URL the notebooks hand to `load_image_from_url` still resolves to
     an image, so a link that rots upstream goes red here first;
  2. `load_image_from_url` itself rejects a non-image response, so the next
     rotted link reports the URL and the status instead of a PIL internal.
"""

from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("original_template", "nb", "kaggle", "python_scripts", "molab")

_LOAD_CALL = re.compile(r"""load_image_from_url\(\s*["']([^"']+)["']""")
_HELPER = "load_image_from_url"
_TIMEOUT = 60


def _code_chunks(path):
    """A file's Python source, one independently parseable chunk at a time.

    Notebooks go cell by cell: a whole notebook concatenated is not valid
    Python, because the install cells are `!pip` / `%%capture` magics.
    """
    if path.suffix != ".ipynb":
        yield path.read_text(encoding="utf-8")
        return
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            yield "".join(cell.get("source", []))


def _sources():
    for directory in SEARCH_DIRS:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if path.suffix in (".ipynb", ".py"):
                for chunk in _code_chunks(path):
                    yield path, chunk


def _image_urls():
    """Every distinct URL a notebook downloads through the helper."""
    urls = {}
    for path, text in _sources():
        for url in _LOAD_CALL.findall(text):
            urls.setdefault(url, path.relative_to(REPO_ROOT).as_posix())
    return sorted(urls.items())


def _helper_definitions():
    """Every (id, source) definition of `load_image_from_url`."""
    found = []
    for path, text in _sources():
        if f"def {_HELPER}" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == _HELPER:
                segment = ast.get_source_segment(text, node)
                found.append((path.relative_to(REPO_ROOT).as_posix(), segment))
    return found


_URLS = _image_urls()
_HELPERS = _helper_definitions()


class _FakeResponse:
    """A 404 that answers with an HTML error page, as raw.githubusercontent does."""

    status_code = 404
    headers = {"Content-Type": "text/html; charset=utf-8"}
    content = b"<!DOCTYPE html><html><body>404: Not Found</body></html>"

    def raise_for_status(self):
        raise RuntimeError(f"{self.status_code} Client Error for url")


class _FakeRequests:
    def get(self, url, **kwargs):
        return _FakeResponse()


def test_some_notebook_downloads_a_sample_image():
    """Guard against the parametrisations below silently collecting nothing."""
    assert _URLS, (
        f"no notebook calls {_HELPER} any more; this file is measuring "
        f"nothing, so retire it or repoint it."
    )
    assert _HELPERS, (
        f"no notebook defines {_HELPER} any more; this file is measuring "
        f"nothing, so retire it or repoint it."
    )


@pytest.mark.parametrize("url,origin", _URLS, ids=[u for u, _ in _URLS])
def test_sample_image_url_still_serves_an_image(url, origin):
    request = urllib.request.Request(
        url, headers={"User-Agent": "unsloth-notebooks-tests"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        pytest.fail(
            f"{origin} downloads {url}, which answers HTTP {error.code}. "
            f"`requests.get` hands that error page to `Image.open`, which "
            f"raises UnidentifiedImageError and takes every later cell with "
            f"it. Repoint the notebook at a live image."
        )
    except Exception as error:  # noqa: BLE001 - see below
        # Anything short of an HTTP answer is the runner, not the notebook:
        # no route out, DNS off, or the whole-suite run, where
        # tests/security/conftest.py installs a session-wide socket blocker.
        # CI runs this file as its own pytest invocation, where the blocker
        # is not collected and the check really executes.
        pytest.skip(f"cannot reach {url}: {type(error).__name__}: {error}")

    assert status == 200, f"{origin} downloads {url}, which answers HTTP {status}."
    assert content_type.startswith("image/"), (
        f"{origin} downloads {url}, which answers with Content-Type "
        f"{content_type!r} rather than an image."
    )


@pytest.mark.parametrize(
    "origin,definition", _HELPERS, ids=[o for o, _ in _HELPERS]
)
def test_helper_rejects_a_non_image_response(origin, definition):
    namespace = {"requests": _FakeRequests(), "Image": None, "BytesIO": None}
    exec(definition, namespace)

    with pytest.raises(Exception) as caught:  # noqa: PT011 - the type is the point
        namespace[_HELPER]("https://example.invalid/gone.png")

    # `Image` and `BytesIO` are None above, so reaching PIL at all is an
    # AttributeError -- which is exactly the "failed deep inside the decoder"
    # shape this guard exists to prevent.
    assert not isinstance(caught.value, AttributeError), (
        f"{origin}: {_HELPER} passed a 404 HTML body straight to the image "
        f"decoder. Check `response.raise_for_status()` and the Content-Type "
        f"before decoding, so a dead link reports the URL and the status."
    )
