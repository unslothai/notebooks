# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""Install cells pick the torchcodec build for the torch they leave installed (#304).

torchcodec declares no torch requirement, so a bare `pip install torchcodec` takes the newest
release; before 0.12 each release loads against a single torch minor, and on torch 2.10 the
newest one fails to import with `undefined symbol`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PIP_INSTALL = re.compile(r"^\s*!(?:uv )?pip install\b.*$", re.MULTILINE)
BARE = re.compile(r"(?<![\w\"'{./-])torchcodec(?![\w.=<>~!\[-])")

# pytorch/torchcodec README compatibility table; 0.12+ targets torch >= 2.11.
COMPATIBLE = {
    "2.7": {"0.3", "0.4", "0.5"},
    "2.8": {"0.6", "0.7"},
    "2.9": {"0.8", "0.9"},
    "2.10": {"0.10"},
    "2.11": {"0.11", "0.12", "0.13", "0.17"},
    "2.14": {"0.12", "0.13", "0.17"},
}
RELEASES = ["0.5.0", "0.7.0", "0.9.1", "0.10.0", "0.11.1", "0.12.0", "0.13.0", "0.17.0"]


def _install_cells():
    for path in sorted((REPO_ROOT / "nb").glob("*.ipynb")):
        if path.name.startswith("AMD-"):
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            src = cell["source"]
            src = "".join(src) if isinstance(src, list) else src
            if any("torchcodec" in line for line in PIP_INSTALL.findall(src)):
                yield path.name, src


CELLS = list(_install_cells())


def test_some_notebooks_install_torchcodec():
    assert len(CELLS) >= 10


@pytest.mark.parametrize("name,src", CELLS, ids=[name for name, _ in CELLS])
def test_no_bare_torchcodec_install(name, src):
    bare = [line.strip() for line in PIP_INSTALL.findall(src) if BARE.search(line)]
    assert not bare, f"{name}: unpinned torchcodec ignores the installed torch: {bare}"


@pytest.mark.parametrize("name,src", [c for c in CELLS if "{_torchcodec}" in c[1]],
                         ids=[name for name, src in CELLS if "{_torchcodec}" in src])
@pytest.mark.parametrize("torch_version", ["2.7.1", "2.8.0+cu126", "2.9.1", "2.10.0+cu128", "2.11.0", "2.14.1"])
def test_selected_torchcodec_loads_on_installed_torch(name, src, torch_version, monkeypatch):
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if "import importlib.metadata as _md" in line)
    end = next(i for i, line in enumerate(lines) if line.lstrip().startswith("_torchcodec ="))
    snippet = "\n".join(line.strip() for line in lines[start:end + 1])

    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda dist: torch_version)
    scope = {}
    exec(snippet, scope)

    spec = SpecifierSet(scope["_torchcodec"].removeprefix("torchcodec"))
    allowed = COMPATIBLE[".".join(torch_version.split("+")[0].split(".")[:2])]
    picked = [v for v in RELEASES if Version(v) in spec]
    assert picked, f"{name}: {scope['_torchcodec']} admits no release"
    wrong = [v for v in picked if ".".join(v.split(".")[:2]) not in allowed]
    assert not wrong, f"{name}: torch {torch_version} -> {scope['_torchcodec']} admits {wrong}"
