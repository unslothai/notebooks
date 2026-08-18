# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Keep the Unsloth Desktop promo consistent in notebook News sections."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import notebook_inventory as ni  # noqa: E402


OLD_TOP_BANNER = "<!-- unsloth-desktop-banner -->"
OLD_STUDIO_INTRO = "Introducing **Unsloth Studio**"
DESKTOP_INTRO = "Introducing **Unsloth Desktop**"
DESKTOP_DOCS = "https://unsloth.ai/docs/desktop"
DESKTOP_DOWNLOAD = "https://unsloth.ai/download"
DESKTOP_IMAGE = (
    "https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/"
    "assets/unsloth-desktop.jpg"
)
DESKTOP_COPY = (
    "Introducing **Unsloth Desktop**, the first desktop app to run and train "
    "models. It is free and open source for macOS, Windows, and Linux, and runs "
    "on your own local hardware."
)
NOTEBOOKS = [REPO_ROOT / "Template_Notebook.ipynb", *ni.iter_notebooks()]


def _source(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def _news_announcement(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as handle:
        cells = json.load(handle).get("cells", [])
    for index, cell in enumerate(cells[:-1]):
        if _source(cell).strip() == "### News":
            return _source(cells[index + 1])
    return None


PUBLISHED_NEWS = [
    (path, announcement)
    for path in ni.iter_notebooks(("nb", "kaggle"))
    if (announcement := _news_announcement(path)) is not None
]
DESKTOP_NEWS = [
    (path, announcement)
    for path in NOTEBOOKS
    if (announcement := _news_announcement(path)) is not None
    and announcement.startswith(DESKTOP_INTRO)
]


def test_desktop_screenshot_asset_exists() -> None:
    asset = REPO_ROOT / "assets" / "unsloth-desktop.jpg"
    assert asset.is_file(), "The Desktop banner screenshot is missing."
    assert asset.read_bytes().startswith(b"\xff\xd8\xff"), (
        "assets/unsloth-desktop.jpg is not a JPEG file."
    )


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_old_desktop_top_banner_and_studio_intro_are_gone(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    text = "\n".join(_source(cell) for cell in notebook.get("cells", []))
    assert OLD_TOP_BANNER not in text
    assert OLD_STUDIO_INTRO not in text


@pytest.mark.parametrize(
    ("path", "announcement"),
    PUBLISHED_NEWS,
    ids=lambda item: str(item.relative_to(REPO_ROOT)) if isinstance(item, Path) else None,
)
def test_every_published_news_section_starts_with_desktop(
    path: Path, announcement: str
) -> None:
    assert announcement.startswith(DESKTOP_INTRO), path.relative_to(REPO_ROOT)


@pytest.mark.parametrize(
    ("path", "announcement"),
    DESKTOP_NEWS,
    ids=lambda item: str(item.relative_to(REPO_ROOT)) if isinstance(item, Path) else None,
)
def test_desktop_news_promo_has_copy_screenshot_and_links(
    path: Path, announcement: str
) -> None:
    promo = announcement.split("\n\nTrain MoEs -", 1)[0]
    assert DESKTOP_COPY in promo, path.relative_to(REPO_ROOT)
    assert DESKTOP_DOCS in promo
    assert DESKTOP_DOWNLOAD in promo
    assert DESKTOP_IMAGE in promo
    assert "\u2014" not in promo and "\u2013" not in promo


def test_notebook_generator_owns_the_desktop_news_promo() -> None:
    generator = (REPO_ROOT / "update_all_notebooks.py").read_text(encoding="utf-8")
    assert 'new_announcement = """' + DESKTOP_INTRO in generator
    assert 'new_announcement = """' + OLD_STUDIO_INTRO not in generator
    assert DESKTOP_IMAGE in generator
    assert DESKTOP_DOCS in generator
    assert DESKTOP_DOWNLOAD in generator
