"""Intel- notebooks match `--intel`, and the AMD code path behaviour is unchanged.

Intel shares the AMD branches of `update_notebook_sections` through
`is_vendor_gpu_notebook`. Every AMD notebook in nb/ must come out of another pass
unchanged without reaching any Intel code, and every Intel- notebook in nb/ must
be in the allowlist, `INTEL_NOTEBOOKS`, pass the validator and match a fresh `--intel` generation.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "nb"
AMD_NOTEBOOKS = sorted(p.name for p in NB_DIR.glob("AMD-*.ipynb"))


@pytest.fixture(scope = "module")
def gen():
    spec = importlib.util.spec_from_file_location("update_all_notebooks", REPO_ROOT / "update_all_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _intel_names(gen):
    return [f"Intel-{name}" for name in gen.INTEL_NOTEBOOKS]


def _sources(path):
    """Cell types and code only: outputs come from whoever last ran the notebook."""
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]


def _run_sections(gen, path):
    with contextlib.redirect_stdout(io.StringIO()):
        gen.update_notebook_sections(
            str(path), gen.general_announcement_content, gen.installation_content,
            gen.installation_kaggle_content, gen.new_announcement,
        )


def test_every_allowlisted_notebook_exists_and_is_valid(gen):
    for name in _intel_names(gen):
        path = NB_DIR / name
        assert path.is_file(), f"{name} is in INTEL_NOTEBOOKS but missing from nb/; run update_all_notebooks.py --intel"
        assert gen._validate_intel_notebook(str(path)) == [], name


def test_no_intel_notebook_outside_the_allowlist(gen):
    """--intel only regenerates INTEL_NOTEBOOKS; any other Intel- notebook would go stale silently."""
    assert sorted(p.name for p in NB_DIR.glob("Intel-*.ipynb")) == sorted(_intel_names(gen))


def test_intel_notebooks_match_a_fresh_generation(gen, tmp_path, monkeypatch):
    """The same steps as `update_all_notebooks.py --intel`, into a scratch nb/ holding only the
    nb/-only sources, so no existing Intel- file is there to start from."""
    monkeypatch.setattr(gen, "_unsloth_model_exists", lambda name: False)  # no network
    scratch = tmp_path / "nb"
    scratch.mkdir()
    template_dir = REPO_ROOT / "original_template"
    for name in gen.INTEL_NOTEBOOKS:
        if name in gen.DONT_UPDATE_EXCEPTIONS or not (template_dir / name).is_file():
            shutil.copy(NB_DIR / name, scratch / name)
    with contextlib.redirect_stdout(io.StringIO()):
        paths = gen.copy_and_update_intel_notebooks(
            str(template_dir), str(scratch), gen.general_announcement_content,
            gen.installation_content, gen.installation_kaggle_content, gen.new_announcement,
        )
        for path in paths:
            gen._process_single_notebook(path)
        gen._postprocess_generated_notebooks(paths)
    assert len(paths) == len(gen.INTEL_NOTEBOOKS)
    for path in paths:
        name = Path(path).name
        assert _sources(path) == _sources(NB_DIR / name), f"{name} differs from a fresh --intel generation; rerun it"


@pytest.mark.parametrize("name", AMD_NOTEBOOKS)
def test_amd_notebook_is_unchanged_and_never_reaches_intel_code(gen, tmp_path, monkeypatch, name):
    """The AMD notebooks in nb/ come from the generator before Intel shared its branches."""
    def _intel_called(*args, **kwargs):
        raise AssertionError(f"Intel code ran for {name}")
    for intel_only in ("_compose_intel_installation", "_apply_intel_cell_rules", "_intel_stale_followup_cells"):
        monkeypatch.setattr(gen, intel_only, _intel_called)
    path = tmp_path / name
    shutil.copy(NB_DIR / name, path)
    _run_sections(gen, path)
    assert _sources(path) == _sources(NB_DIR / name)
