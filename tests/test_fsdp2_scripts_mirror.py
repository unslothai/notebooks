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

"""``fsdp2_scripts/`` must stay an exact mirror of ``python_scripts/``.

Every ``python_scripts/<name>.py`` gets a launcher ``fsdp2_scripts/<name>.py``
that runs it on every visible GPU through ``unsloth_fsdp2_launch.py``, a copy
of ``scripts/fsdp2_launch.py``. All of it comes from
``update_all_notebooks.py:generate_fsdp2_scripts``, so this asserts:

  1. The two folders hold the same set of names, in BOTH directions. A
     launcher whose script was deleted or renamed would ship a command that
     fails on a missing file; a script with no launcher silently has no
     multi-GPU entry point.

  2. Each committed launcher is byte-identical to the generator's render for
     its name, and the helper copy is byte-identical to its source. Catches a
     hand edit, or a generator change committed without regenerating.

  3. Every launcher ast-parses and the name it passes to ``main`` is its own
     file name, which exists in ``python_scripts/``. Names carry spaces and
     parentheses, so this is where a quoting slip would show.

  4. The generator itself: a second run is a no-op, a deleted or renamed
     script loses its launcher, the AMD-only pass touches only ``AMD-*``
     launchers, a render failure leaves the folder as it was, read-only files
     are replaced and removed, unchanged launchers get their permissions back.

  5. Repository wiring: the staging folder is git-ignored, the git diff
     summary counts only the root README as a README change, and the launcher
     helper tests run in their own CI job rather than in every api-drift cell.

No torch, no network: only the generator module and the files on disk.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "python_scripts"
LAUNCHERS = REPO_ROOT / "fsdp2_scripts"
HELPER_SOURCE = REPO_ROOT / "scripts" / "fsdp2_launch.py"


def _load_generator():
    """Load update_all_notebooks.py by file path so the test does not
    depend on the repo root being on sys.path."""
    path = REPO_ROOT / "update_all_notebooks.py"
    spec = importlib.util.spec_from_file_location("_update_all_notebooks", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_update_all_notebooks"] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()
HELPER_NAME = _GEN.FSDP2_HELPER_NAME


def _script_names():
    return {p.name for p in SCRIPTS.glob("*.py")}


def _launcher_names(folder=LAUNCHERS):
    return {p.name for p in Path(folder).glob("*.py") if p.name != HELPER_NAME}


def _snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(Path(folder).iterdir()) if p.is_file()}


# ---------------------------------------------------------------------------
# The committed tree
# ---------------------------------------------------------------------------

def test_there_are_scripts_to_mirror():
    assert len(_script_names()) > 100, "python_scripts/ looks empty; the mirror check would be vacuous"


def test_every_script_has_a_launcher():
    missing = sorted(_script_names() - _launcher_names())
    assert not missing, (
        f"DRIFT DETECTED: {len(missing)} python_scripts/ files have no fsdp2_scripts/ "
        f"launcher: {missing[:10]}. Regenerate with update_all_notebooks.py."
    )


def test_every_launcher_has_a_script():
    orphaned = sorted(_launcher_names() - _script_names())
    assert not orphaned, (
        f"DRIFT DETECTED: {len(orphaned)} fsdp2_scripts/ launchers point at a "
        f"python_scripts/ file that does not exist: {orphaned[:10]}. Regenerate "
        f"with update_all_notebooks.py."
    )


def test_assert_fsdp2_mirror_accepts_the_committed_tree():
    _GEN._assert_fsdp2_mirror(str(SCRIPTS), str(LAUNCHERS))
    _GEN._assert_fsdp2_mirror(str(SCRIPTS), str(LAUNCHERS), include_prefix="AMD-")


def test_each_launcher_matches_the_generator():
    stale = sorted(
        name for name in _launcher_names() & _script_names()
        if (LAUNCHERS / name).read_text(encoding="utf-8") != _GEN.render_fsdp2_launcher(name)
    )
    assert not stale, (
        f"{len(stale)} fsdp2_scripts/ launchers differ from render_fsdp2_launcher: "
        f"{stale[:10]}. They are generated; regenerate instead of editing them."
    )


def test_readme_matches_the_generator():
    readme = LAUNCHERS / _GEN.FSDP2_README_NAME
    assert readme.is_file()
    assert readme.read_text(encoding="utf-8") == _GEN.render_fsdp2_readme()


def test_helper_copy_is_the_source_byte_for_byte():
    copy = LAUNCHERS / HELPER_NAME
    assert HELPER_SOURCE.is_file(), f"missing helper source {HELPER_SOURCE}"
    assert copy.is_file(), f"missing helper copy {copy}; regenerate fsdp2_scripts/"
    assert copy.read_bytes() == HELPER_SOURCE.read_bytes(), (
        f"fsdp2_scripts/{HELPER_NAME} differs from scripts/fsdp2_launch.py. Edit the "
        f"source and regenerate; the copy is generated."
    )


def _main_target(tree):
    """The constant script name passed to ``main(__file__, <name>)``."""
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "main"
    ]
    assert len(calls) == 1, f"expected one main(...) call, found {len(calls)}"
    args = calls[0].args
    assert len(args) == 2 and isinstance(args[0], ast.Name) and args[0].id == "__file__"
    assert isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)
    return args[1].value


def test_every_launcher_parses_and_names_its_own_existing_script():
    problems = []
    scripts = _script_names()
    for path in sorted(LAUNCHERS.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        if path.name == HELPER_NAME:
            continue
        try:
            target = _main_target(tree)
        except AssertionError as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        if target != path.name:
            problems.append(f"{path.name}: targets {target!r}")
        elif target not in scripts:
            problems.append(f"{path.name}: python_scripts/{target} does not exist")
    assert not problems, "\n".join(problems[:20])


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Qwen3_(4B)-GRPO.py",
    "HuggingFace Course-Gemma3_(4B)-Vision-GRPO.py",
    "it's a \"quoted\" name.py",
    "dollar $HOME `tick` !bang.py",
    "unicode été.py",
])
def test_render_survives_awkward_names(name):
    text = _GEN.render_fsdp2_launcher(name)
    tree = ast.parse(text)
    assert _main_target(tree) == name
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env python"
    # The name only ever appears in comments or in the repr'd constant: every
    # line that is not code must stay a comment, whatever the name holds.
    code = {
        "import os",
        "import sys",
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
        "from unsloth_fsdp2_launch import main",
        'if __name__ == "__main__":',
        f"    main(__file__, {name!r})",
    }
    for line in lines:
        assert line.startswith("#") or line in code, f"unexpected line {line!r}"
    assert len(lines) <= 16


@pytest.mark.parametrize("name", [
    "no_suffix",
    "sub/dir.py",
    "back\\slash.py",
    "new\nline.py",
    "null\x00.py",
    "unsloth_fsdp2_launch.py",
])
def test_render_refuses_names_it_cannot_carry(name):
    with pytest.raises(ValueError):
        _GEN.render_fsdp2_launcher(name)


def test_readme_has_no_em_dash_or_emoji():
    text = _GEN.render_fsdp2_readme()
    assert "—" not in text and "–" not in text
    assert all(ord(ch) < 0x2000 for ch in text), "non-ASCII symbol in README"


# ---------------------------------------------------------------------------
# The generator, in throwaway trees
# ---------------------------------------------------------------------------

def _tree(tmp_path, names, helper=b"def main(launcher_path, script_name, argv=None):\n    pass\n"):
    scripts = tmp_path / "python_scripts"
    scripts.mkdir()
    for name in names:
        (scripts / name).write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fsdp2_launch.py").write_bytes(helper)
    return scripts, tmp_path / "fsdp2_scripts"


def _gen(scripts, out, **kw):
    return _GEN.generate_fsdp2_scripts(str(scripts), str(out), **kw)


def test_second_generation_of_the_real_tree_is_a_no_op(tmp_path):
    shutil.copytree(SCRIPTS, tmp_path / "python_scripts")
    shutil.copytree(LAUNCHERS, tmp_path / "fsdp2_scripts")
    (tmp_path / "scripts").mkdir()
    shutil.copy2(HELPER_SOURCE, tmp_path / "scripts" / "fsdp2_launch.py")
    before = _snapshot(tmp_path / "fsdp2_scripts")
    result = _gen(tmp_path / "python_scripts", tmp_path / "fsdp2_scripts")
    assert result == {"written": [], "removed": []}
    assert _snapshot(tmp_path / "fsdp2_scripts") == before


def test_generation_writes_everything_then_nothing(tmp_path):
    scripts, out = _tree(tmp_path, ["A (1B).py", "AMD-B.py", "Kaggle-C.py"])
    first = _gen(scripts, out)
    assert first["removed"] == []
    assert sorted(first["written"]) == sorted(
        ["A (1B).py", "AMD-B.py", "Kaggle-C.py", HELPER_NAME, "README.md"])
    assert (out / HELPER_NAME).read_bytes() == (tmp_path / "scripts" / "fsdp2_launch.py").read_bytes()
    assert _gen(scripts, out) == {"written": [], "removed": []}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".fsdp2_scripts.staging-")]


def test_helper_change_rewrites_only_the_helper(tmp_path):
    scripts, out = _tree(tmp_path, ["A.py", "B.py"])
    _gen(scripts, out)
    (tmp_path / "scripts" / "fsdp2_launch.py").write_bytes(b"def main(*a, **k):\n    return 1\n")
    assert _gen(scripts, out) == {"written": [HELPER_NAME], "removed": []}


def test_deleted_and_renamed_scripts_lose_their_launchers(tmp_path):
    scripts, out = _tree(tmp_path, ["A.py", "B.py", "C.py"])
    _gen(scripts, out)
    (scripts / "A.py").unlink()
    (scripts / "C.py").rename(scripts / "C (renamed).py")
    (out / "notes.txt").write_text("not a launcher\n", encoding="utf-8")
    result = _gen(scripts, out)
    assert sorted(result["removed"]) == ["A.py", "C.py"]
    assert result["written"] == ["C (renamed).py"]
    assert _launcher_names(out) == {"B.py", "C (renamed).py"}
    assert (out / HELPER_NAME).is_file() and (out / "notes.txt").is_file()
    _GEN._assert_fsdp2_mirror(str(scripts), str(out))


def test_amd_only_pass_stays_in_its_prefix(tmp_path):
    scripts, out = _tree(tmp_path, ["A.py", "AMD-A.py", "AMD-B.py"])
    _gen(scripts, out)
    # Simulate a full-pass change the AMD-only pass must not touch: a
    # non-AMD launcher whose script is gone.
    (scripts / "A.py").unlink()
    (scripts / "AMD-B.py").unlink()
    (scripts / "AMD-C.py").write_text("print('c')\n", encoding="utf-8")
    result = _gen(scripts, out, include_prefix="AMD-")
    assert result["removed"] == ["AMD-B.py"]
    assert "AMD-C.py" in result["written"] and "A.py" not in result["written"]
    assert _launcher_names(out) == {"A.py", "AMD-A.py", "AMD-C.py"}
    _GEN._assert_fsdp2_mirror(str(scripts), str(out), include_prefix="AMD-")
    with pytest.raises(RuntimeError, match="no script"):
        _GEN._assert_fsdp2_mirror(str(scripts), str(out))


def test_assert_mirror_reports_both_directions(tmp_path):
    scripts, out = _tree(tmp_path, ["A.py", "B.py"])
    _gen(scripts, out)
    (out / "A.py").unlink()
    (out / "Z.py").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        _GEN._assert_fsdp2_mirror(str(scripts), str(out))
    assert "A.py" in str(exc.value) and "Z.py" in str(exc.value)


def test_a_render_failure_leaves_the_folder_untouched(tmp_path, monkeypatch):
    scripts, out = _tree(tmp_path, ["A.py", "B.py"])
    _gen(scripts, out)
    (scripts / "A.py").unlink()
    (scripts / "C.py").write_text("print('c')\n", encoding="utf-8")
    before = _snapshot(out)
    real = _GEN.render_fsdp2_launcher

    def boom(name):
        if name == "C.py":
            raise ValueError("render failed")
        return real(name)

    monkeypatch.setattr(_GEN, "render_fsdp2_launcher", boom)
    with pytest.raises(ValueError, match="render failed"):
        _gen(scripts, out)
    assert _snapshot(out) == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".fsdp2_scripts.staging-")]


def test_missing_helper_is_an_error_not_a_skip(tmp_path):
    scripts, out = _tree(tmp_path, ["A.py"])
    (tmp_path / "scripts" / "fsdp2_launch.py").unlink()
    with pytest.raises(FileNotFoundError):
        _gen(scripts, out)
    assert not out.exists()


def test_the_main_pass_calls_the_generator_after_both_conversions():
    """Both convert_folder(..., "python_scripts", ...) calls in the CLI are
    followed by the fsdp2 generation and its mirror assert, inside the same
    ``not args.disable_convert_to_script`` block."""
    source = (REPO_ROOT / "update_all_notebooks.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    blocks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "disable_convert_to_script" in ast.unparse(node.test)
    ]
    found = []
    for block in blocks:
        calls = [
            ast.unparse(stmt.value) for stmt in block.body
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        ]
        converts = [i for i, c in enumerate(calls)
                    if c.startswith("convert_folder(") and "'python_scripts'" in c]
        if not converts:
            continue
        after = calls[converts[0] + 1:]
        gen = [c for c in after if c.startswith("generate_fsdp2_scripts(")]
        check = [c for c in after if c.startswith("_assert_fsdp2_mirror(")]
        assert gen and check, f"no fsdp2 generation after {calls[converts[0]][:60]}"
        prefixed = "include_prefix='AMD-'" in calls[converts[0]]
        assert prefixed == ("include_prefix='AMD-'" in gen[0]) == ("include_prefix='AMD-'" in check[0])
        found.append(prefixed)
    assert sorted(found) == [False, True], f"expected the full and the AMD-only pass, got {found}"


def test_unchanged_launchers_get_their_permissions_reset(tmp_path):
    if os.name == "nt":
        pytest.skip("_set_file_permissions is a no-op on Windows")
    scripts, out = _tree(tmp_path, ["A.py"])
    _gen(scripts, out)
    (out / "A.py").chmod(0o600)
    assert _gen(scripts, out) == {"written": [], "removed": []}
    assert stat.S_IMODE((out / "A.py").stat().st_mode) == 0o644


def _read_only_refusal(monkeypatch, name):
    """Make ``os.<name>`` refuse a read-only path the way Windows does."""
    real = getattr(_GEN.os, name)
    refused = []

    def guarded(*paths):
        for path in paths:
            if os.path.exists(path) and not os.stat(path).st_mode & stat.S_IWRITE:
                refused.append(path)
                raise PermissionError(13, "Access is denied", path)
        return real(*paths)

    monkeypatch.setattr(_GEN.os, name, guarded)
    return refused


def test_read_only_launchers_are_replaced(tmp_path, monkeypatch):
    scripts, out = _tree(tmp_path, ["A.py"])
    _gen(scripts, out)
    (out / "A.py").write_text("stale\n", encoding="utf-8")
    (out / "A.py").chmod(stat.S_IREAD)
    refused = _read_only_refusal(monkeypatch, "replace")
    assert _gen(scripts, out)["written"] == ["A.py"]
    assert refused and (out / "A.py").read_text(encoding="utf-8") == _GEN.render_fsdp2_launcher("A.py")


def test_read_only_orphans_are_removed(tmp_path, monkeypatch):
    scripts, out = _tree(tmp_path, ["A.py", "B.py"])
    _gen(scripts, out)
    (scripts / "B.py").unlink()
    (out / "B.py").chmod(stat.S_IREAD)
    refused = _read_only_refusal(monkeypatch, "remove")
    assert _gen(scripts, out)["removed"] == ["B.py"]
    assert refused and not (out / "B.py").exists()


def test_staging_folder_is_git_ignored():
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.fsdp2_scripts.staging-*" in lines


def test_git_diff_summary_counts_only_the_root_readme(monkeypatch, capsys):
    changed = ["README.md", "fsdp2_scripts/README.md", "nb/README.md"]

    def fake_run(argv, **kwargs):
        if argv[:3] == ["git", "diff", "--name-only"]:
            return subprocess.CompletedProcess(argv, 0, stdout="\n".join(changed) + "\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(_GEN.subprocess, "run", fake_run)
    _GEN._summarize_git_diff()
    out = capsys.readouterr().out
    assert "README changes: 1" in out
    assert "Other changes: 2" in out


def test_launcher_helper_tests_run_in_their_own_ci_job():
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "notebooks-tests-ci.yml").read_text(encoding="utf-8"))
    jobs = {
        name: job for name, job in workflow["jobs"].items()
        if any("tests/test_fsdp2_launch.py" in step.get("run", "") for step in job.get("steps", []))
    }
    assert list(jobs) == ["fsdp2-launch"], f"tests/test_fsdp2_launch.py runs in {sorted(jobs)}"
    job = jobs["fsdp2-launch"]
    assert "strategy" not in job, "the launcher tests need one CPU torch, not a matrix"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "https://download.pytorch.org/whl/cpu" in runs and "pytest" in runs


# ---------------------------------------------------------------------------
# Scripts that must run single process
# ---------------------------------------------------------------------------

import re  # noqa: E402

# Python-side process launches, servers and file removes / moves the adapter cannot
# coordinate: get_ipython shell steps run once per node, these run on every rank.
_UNSAFE = re.compile(
    r"subprocess\.Popen|os\.system\(|vllm serve|[\"']--port[\"']|--port\b"
    r"|os\.remove\(|os\.unlink\(|shutil\.rmtree\(|shutil\.move\(|os\.rename\("
)


def _capture_bodies(text):
    """(line, body) of every live get_ipython().run_cell_magic('capture', ..., body)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(text)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_cell_magic" and len(node.args) == 3
                and all(isinstance(a, ast.Constant) for a in node.args)
                and node.args[0].value == "capture"):
            yield node.lineno, node.args[2].value


def _unsafe_lines(path):
    text = path.read_text(encoding="utf-8")
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        # Comments are not run; get_ipython() lines are shell steps the adapter
        # coordinates (a %%capture body is checked separately below).
        if line.lstrip().startswith("#") or "get_ipython()" in line:
            continue
        if _UNSAFE.search(line):
            hits.append(f"{number}: {line.strip()[:100]}")
    for number, body in _capture_bodies(text):
        for line in body.splitlines():
            if line.lstrip().startswith(("!", "%", "#")):
                continue
            if _UNSAFE.search(line):
                hits.append(f"{number} (%%capture): {line.strip()[:100]}")
    return hits


def test_scan_reads_live_capture_bodies_and_skips_comments(tmp_path):
    live = tmp_path / "live.py"
    live.write_text(
        "import os\n"
        "get_ipython().run_cell_magic('capture', '', "
        "'!git clone https://example.com/x X\\nos.remove(\"X/a.py\")\\n')\n",
        encoding="utf-8",
    )
    assert any("%%capture" in hit and "os.remove" in hit for hit in _unsafe_lines(live))
    commented = tmp_path / "commented.py"
    commented.write_text("# os.remove('X/a.py')\n# shutil.rmtree('X')\nprint(1)\n", encoding="utf-8")
    assert _unsafe_lines(commented) == []
    top = tmp_path / "top.py"
    top.write_text("import shutil\nshutil.rmtree('outputs')\n", encoding="utf-8")
    assert _unsafe_lines(top)


def test_scripts_that_start_servers_or_processes_are_single_process():
    missing = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        hits = _unsafe_lines(path)
        if hits and _GEN.fsdp2_single_process_reason(path.name) is None:
            missing[path.name] = hits[:3]
    assert not missing, (
        "These python_scripts start servers or processes from Python on every rank. Add "
        "their base name to FSDP2_SINGLE_PROCESS in update_all_notebooks.py with a reason "
        f"(or make them multi-rank safe): {missing}"
    )


def test_single_process_list_names_real_scripts():
    bases = {_GEN._fsdp2_base_name(name) for name in _script_names()}
    stale = sorted(set(_GEN.FSDP2_SINGLE_PROCESS) - bases)
    assert not stale, f"FSDP2_SINGLE_PROCESS names scripts that no longer exist: {stale}"
    readme = (LAUNCHERS / "README.md").read_text(encoding="utf-8")
    for base in _GEN.FSDP2_SINGLE_PROCESS:
        assert base[:-3] in readme, f"{base} is single process but not listed in the README"


@pytest.mark.parametrize("name", [
    "Meta-Synthetic-Data-Llama3.1_(8B).py",
    "AMD-Meta-Synthetic-Data-Llama3.1_(8B).py",
    "Kaggle-Meta-Synthetic-Data-Llama3.1_(8B).py",
    "NeMo-Gym-Sudoku.py",
    "AMD-NeMo-Gym-Multi-Environment.py",
    "Synthetic_Data_Hackathon.py",
    "Deepseek_OCR_(3B)-Evaluation.py",
    "Oute_TTS_(1B).py",
    "AMD-Oute_TTS_(1B).py",
    "Kaggle-Oute_TTS_(1B).py",
])
def test_single_process_launchers_say_so(name):
    assert (SCRIPTS / name).is_file()
    tree = ast.parse((LAUNCHERS / name).read_text(encoding="utf-8"))
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "main")
    keywords = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    assert keywords["single_process"] is True
    assert keywords["reason"] == _GEN.fsdp2_single_process_reason(name)


def test_ordinary_launchers_pass_no_single_process_flag():
    tree = ast.parse(_GEN.render_fsdp2_launcher("Qwen3_(4B)-GRPO.py"))
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "main")
    assert call.keywords == []


def test_launchers_import_one_module_per_line():
    text = _GEN.render_fsdp2_launcher("A.py")
    assert "import os\nimport sys\n" in text and "import os, sys" not in text


def test_assert_mirror_requires_readme_and_helper(tmp_path):
    scripts, out = _tree(tmp_path, ["A.py"])
    _gen(scripts, out)
    (out / "README.md").unlink()
    with pytest.raises(RuntimeError, match="README.md is missing"):
        _GEN._assert_fsdp2_mirror(str(scripts), str(out))
    _gen(scripts, out)
    (out / HELPER_NAME).unlink()
    with pytest.raises(RuntimeError, match=HELPER_NAME):
        _GEN._assert_fsdp2_mirror(str(scripts), str(out))
