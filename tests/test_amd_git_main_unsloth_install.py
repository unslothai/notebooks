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

"""An AMD variant keeps the git-main unsloth upgrade its CUDA source uses.

A source installs unsloth / unsloth_zoo from git main only for code no release
carries yet (`FastDiffusionModel`). Both names sit in
`_AMD_INSTALL_PACKAGE_IGNORE` and the parity validator subtracts the same set,
so the upgrade vanished from the AMD variant unseen and the notebook died at
`FastModel.from_pretrained` on the PyPI build.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"

UNSLOTH_GIT = "git+https://github.com/unslothai/unsloth.git"
ZOO_GIT = "git+https://github.com/unslothai/unsloth-zoo.git"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "update_all_notebooks", REPO_ROOT / "update_all_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()


def _code_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _amd_pairs():
    """(amd_path, source_path) for every AMD variant with a CUDA counterpart."""
    for amd in sorted(NB_DIR.glob("AMD-*.ipynb")):
        source = amd.with_name(amd.name[len("AMD-"):])
        if source.exists():
            yield amd, source


def test_extractor_finds_only_unsloth_git_specs():
    text = (
        "!pip install --no-deps --upgrade "
        f"{ZOO_GIT} {UNSLOTH_GIT}\n"
        "!pip install git+https://github.com/huggingface/transformers.git\n"
        "!pip install unsloth unsloth_zoo\n"
    )
    assert GEN._extract_git_main_unsloth_specs([text]) == [ZOO_GIT, UNSLOTH_GIT]


def test_extractor_is_empty_for_released_installs():
    assert GEN._extract_git_main_unsloth_specs(['!pip install unsloth "unsloth_zoo[amd]"']) == []


def test_compose_reemits_the_git_main_upgrade():
    # Synthetic source: no shipped template installs from git main today, but the
    # re-emit path still has to work for the next one that needs unreleased code.
    source = GEN.installation_diffusiongemma_content.replace(
        '--force-reinstall "unsloth_zoo>=2026.6.5" "unsloth>=2026.6.5"',
        f"--force-reinstall {ZOO_GIT} {UNSLOTH_GIT}",
    )
    assert UNSLOTH_GIT in source, "synthetic source lost its git main upgrade"
    install, extras = GEN._compose_amd_installation(
        "nb/AMD-DiffusionGemma_(26B-A4B)-Sudoku.ipynb",
        [source],
    )
    assert UNSLOTH_GIT not in install, "the shared %%bash base cell must stay literal"
    assert extras is not None
    line = next((text for text in extras.splitlines() if UNSLOTH_GIT in text), None)
    assert line is not None, f"git main upgrade dropped from extras cell:\n{extras}"
    assert ZOO_GIT in line
    # --no-deps protects the ROCm stack installed above.
    assert "--no-deps" in line
    assert "--force-reinstall" in line


def test_compose_reemits_a_forced_release_reinstall():
    """A source that force-reinstalls released unsloth must keep doing so on AMD.

    The base cell installs the same two packages with a plain -U, which leaves an
    equal-or-newer build (a leftover git one) in place, so the AMD variant needs
    its own reinstall or a rerun keeps executing the old build.
    """
    install, extras = GEN._compose_amd_installation(
        "nb/AMD-DiffusionGemma_(26B-A4B)-Sudoku.ipynb",
        [GEN.installation_diffusiongemma_content],
    )
    assert "unsloth>=" not in install, "the shared %%bash base cell must stay literal"
    assert extras is not None
    line = next(
        (text for text in extras.splitlines() if "--force-reinstall" in text), None
    )
    assert line is not None, f"release reinstall dropped from extras cell:\n{extras}"
    assert "unsloth>=2026.6.5" in line and "unsloth_zoo>=2026.6.5" in line
    # --no-deps protects the ROCm stack installed above.
    assert "--no-deps" in line


def test_forced_release_extractor_ignores_plain_installs():
    """Only a --force-reinstall line counts; a plain install needs no re-emit."""
    assert GEN._extract_forced_unsloth_release_specs(
        ['!pip install --no-deps --upgrade "unsloth>=2026.6.5"']
    ) == []
    assert GEN._extract_forced_unsloth_release_specs(
        ['!pip install --no-deps --force-reinstall "unsloth>=2026.6.5" trl==1.9.2']
    ) == ["unsloth>=2026.6.5"]


def test_compose_adds_nothing_for_a_released_source():
    _install, extras = GEN._compose_amd_installation(
        "nb/AMD-Some_Model.ipynb", [GEN.installation_content]
    )
    assert extras is None or "github.com/unslothai" not in extras


@pytest.mark.parametrize(
    "amd,source", list(_amd_pairs()), ids=lambda p: getattr(p, "name", p)
)
def test_generated_amd_notebooks_match_their_source(amd, source):
    specs = GEN._extract_git_main_unsloth_specs([_code_text(source)])
    if not specs:
        pytest.skip("source installs released unsloth")
    amd_text = _code_text(amd)
    for spec in specs:
        assert spec in amd_text, f"{amd.name} lost {spec}"
