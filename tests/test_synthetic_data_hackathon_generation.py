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

"""Gate on the Synthetic_Data_Hackathon in-notebook dataset generator.

The generator skips itself whenever ``data/final/`` already holds a
``*.json`` file, so a run that dies part-way through writing the shards
would otherwise leave a half dataset that every later run accepts and
trains on silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "nb" / "Synthetic_Data_Hackathon.ipynb"
FINAL_DIR = Path("logical_reasoning") / "data" / "final"
EXPECTED_RECORDS = 74


def _generator_cell() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    matches = [src for src in cells if "knights_and_knaves_easy_ft.json" in src]
    assert len(matches) == 1, f"expected one generator cell, found {len(matches)}"
    return matches[0]


def _run_generator() -> None:
    exec(compile(_generator_cell(), str(NOTEBOOK), "exec"), {"__name__": "__main__"})


def _published_records() -> int:
    return sum(
        len(json.loads(path.read_text())) for path in sorted(FINAL_DIR.glob("*.json"))
    )


@pytest.fixture(autouse=True)
def _in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_generator_writes_the_full_dataset() -> None:
    _run_generator()
    assert _published_records() == EXPECTED_RECORDS


def test_generator_skips_an_already_complete_dataset(capsys) -> None:
    _run_generator()
    before = {path.name: path.read_text() for path in sorted(FINAL_DIR.glob("*.json"))}
    _run_generator()
    after = {path.name: path.read_text() for path in sorted(FINAL_DIR.glob("*.json"))}
    assert after == before
    assert "skipping generation" in capsys.readouterr().out


def test_interrupted_run_leaves_nothing_for_the_next_run_to_accept() -> None:
    real_dump = json.dump
    calls = {"n": 0}

    def dump_then_die(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt("kernel interrupted mid-publish")
        return real_dump(*args, **kwargs)

    # Restored by hand rather than through monkeypatch, whose undo() would
    # also drop the autouse chdir into the temporary directory.
    json.dump = dump_then_die
    try:
        with pytest.raises(KeyboardInterrupt):
            _run_generator()
    finally:
        json.dump = real_dump

    _run_generator()
    assert _published_records() == EXPECTED_RECORDS
