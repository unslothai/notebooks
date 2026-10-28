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

"""Behaviour of ``scripts/fsdp2_launch.py``, the runtime behind ``fsdp2_scripts/``.

Two layers:

  1. In-process checks with ``os.execv`` and the GPU count stubbed out: the
     torchrun argv (names with spaces and parentheses, forwarded args), the
     ``--nproc-per-node`` > ``UNSLOTH_NPROC_PER_NODE`` > visible-GPU order,
     refusal above the visible count, worker mode never launching again,
     n == 0 / 1 running in-process, ``UNSLOTH_FSDP2`` set by default but an
     explicit 0 kept, and ``--dry-run``.

  2. A real two-process ``torch.distributed.run --standalone`` on CPU over a
     fake notebook mirror that uses every ``get_ipython()`` form the adapter
     supports. It proves shell steps run once per node while the other rank
     waits, that both ranks see the files / env / cwd afterwards, that return
     codes reach every rank, that installs are skipped unless
     ``UNSLOTH_FSDP2_RUN_INSTALLS=1``, and that a failing rank fails the
     launch. Skipped when torch is not importable.

No GPU is used: the subprocesses run with ``CUDA_VISIBLE_DEVICES`` empty.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_PATH = REPO_ROOT / "scripts" / "fsdp2_launch.py"

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _load_launch():
    spec = importlib.util.spec_from_file_location("_fsdp2_launch_under_test", str(LAUNCH_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fl(monkeypatch):
    for name in ("LOCAL_RANK", "TORCHELASTIC_RUN_ID", "UNSLOTH_FSDP2", "UNSLOTH_NPROC_PER_NODE",
                 "UNSLOTH_FSDP2_RUN_INSTALLS"):
        monkeypatch.delenv(name, raising = False)
    mod = _load_launch()
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))
    return mod


SCRIPT_NAME = "My Model (1.5B)-GRPO.py"


@pytest.fixture
def layout(tmp_path):
    """tmp/fsdp2_scripts/<launcher> + tmp/python_scripts/<target>, the generated shape."""
    (tmp_path / "python_scripts").mkdir()
    (tmp_path / "fsdp2_scripts").mkdir()
    target = tmp_path / "python_scripts" / SCRIPT_NAME
    target.write_text(textwrap.dedent("""\
        import json, os, sys
        with open(os.environ["FAKE_RAN"], "w") as f:
            json.dump({"argv": sys.argv, "fsdp2": os.environ.get("UNSLOTH_FSDP2"),
                       "file": __file__, "path0": sys.path[0],
                       "has_ipython": "get_ipython" in globals()}, f)
    """))
    launcher = tmp_path / "fsdp2_scripts" / SCRIPT_NAME
    launcher.write_text("# launcher stub; main() is called directly in these tests\n")
    return tmp_path, launcher, target


class _Exec(Exception):
    pass


def _stub(fl, monkeypatch, visible):
    calls = []

    def fake_execv(path, argv):
        calls.append((path, list(argv)))
        raise _Exec()

    monkeypatch.setattr(fl, "_count_visible_gpus", lambda: visible)
    monkeypatch.setattr(fl.os, "execv", fake_execv)
    return calls


def _expected(launcher, nproc, *rest):
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={nproc}", str(launcher), *rest]


def test_argv_spaces_parentheses_and_forwarded_args(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 4)
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node", "2", "a b (c)", "--lr=1e-4"])
    path, argv = calls[0]
    assert path == sys.executable
    assert argv == _expected(launcher, 2, "a b (c)", "--lr=1e-4")
    assert os.environ["UNSLOTH_FSDP2"] == "1"


def test_visible_gpus_are_the_default_count(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 3)
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert calls[0][1] == _expected(launcher, 3)


def test_flag_beats_env_beats_visible(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 8)
    monkeypatch.setenv("UNSLOTH_NPROC_PER_NODE", "4")
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node=2"])
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert calls[0][1] == _expected(launcher, 2)
    assert calls[1][1] == _expected(launcher, 4)


def test_refuses_more_than_visible(fl, layout, monkeypatch, capsys):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node", "4"])
    assert raised.value.code == 2
    assert "only 2 GPU(s) are visible" in capsys.readouterr().err
    monkeypatch.setenv("UNSLOTH_NPROC_PER_NODE", "3")
    with pytest.raises(SystemExit):
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert calls == []


def test_forced_count_with_no_gpu_visible_launches(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 0)
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node", "2"])
    assert calls[0][1] == _expected(launcher, 2)


@pytest.mark.parametrize("bad", ["two", "-1"])
def test_bad_count_is_an_error(fl, layout, monkeypatch, bad):
    _, launcher, _ = layout
    _stub(fl, monkeypatch, visible = 2)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node", bad])
    assert raised.value.code == 2


def test_missing_target_is_a_clear_error(fl, layout, monkeypatch, capsys):
    _, launcher, _ = layout
    _stub(fl, monkeypatch, visible = 2)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), "Nope.py", [])
    assert raised.value.code == 2
    assert "target script not found" in capsys.readouterr().err


def test_explicit_fsdp2_zero_is_kept(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    monkeypatch.setenv("UNSLOTH_FSDP2", "0")
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert calls[0][1] == _expected(launcher, 2)
    assert os.environ["UNSLOTH_FSDP2"] == "0"


def test_script_args_that_look_like_launcher_flags_ride_behind_a_separator(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, ["x", "--", "--help", "--dry-run"])
    assert calls[0][1] == _expected(launcher, 2, "--", "x", "--help", "--dry-run")
    # And the worker side reads them back as script args.
    assert fl._parse(["--", "x", "--help", "--dry-run"]) == (
        {"nproc": None, "dry_run": False, "help": False}, ["x", "--help", "--dry-run"],
    )


@pytest.mark.parametrize("visible", [0, 1])
def test_zero_or_one_gpu_runs_in_process(fl, layout, monkeypatch, visible):
    tmp, launcher, target = layout
    calls = _stub(fl, monkeypatch, visible = visible)
    ran = tmp / "ran.json"
    monkeypatch.setenv("FAKE_RAN", str(ran))
    assert fl.main(str(launcher), SCRIPT_NAME, ["arg 1"]) == 0
    assert calls == []
    seen = json.loads(ran.read_text())
    assert seen["argv"] == [str(target), "arg 1"]
    assert seen["fsdp2"] is None  # single process: FSDP2 is not switched on
    assert seen["file"] == str(target)
    assert seen["path0"] == str(target.parent)
    assert seen["has_ipython"] is True


@pytest.mark.parametrize("env", [{"LOCAL_RANK": "0"}, {"TORCHELASTIC_RUN_ID": "abc"}])
def test_worker_mode_never_relaunches(fl, layout, monkeypatch, env):
    tmp, launcher, target = layout
    calls = _stub(fl, monkeypatch, visible = 8)
    counted = []
    monkeypatch.setattr(fl, "_count_visible_gpus", lambda: counted.append(1) or 8)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    ran = tmp / "ran.json"
    monkeypatch.setenv("FAKE_RAN", str(ran))
    assert fl.main(str(launcher), SCRIPT_NAME, ["x"]) == 0
    assert calls == [] and counted == []
    seen = json.loads(ran.read_text())
    assert seen["argv"] == [str(target), "x"]
    assert seen["fsdp2"] == "1"


def test_dry_run_prints_command_and_env(fl, layout, monkeypatch, capsys):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    assert fl.main(str(launcher), SCRIPT_NAME, ["--dry-run", "a b"]) == 0
    out = capsys.readouterr().out
    assert calls == []
    assert "UNSLOTH_FSDP2=1" in out
    assert shlex.join(_expected(launcher, 2, "a b")) in out
    assert "UNSLOTH_FSDP2" not in os.environ  # a dry run changes nothing


def test_generic_cli_dry_run(tmp_path):
    target = tmp_path / "some dir" / "x (y).py"
    target.parent.mkdir()
    target.write_text("print('hi')\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_NPROC_PER_NODE = "2")
    env.pop("LOCAL_RANK", None)
    env.pop("TORCHELASTIC_RUN_ID", None)
    env.pop("UNSLOTH_FSDP2", None)
    result = subprocess.run(
        [sys.executable, str(LAUNCH_PATH), "--dry-run", str(target), "--epochs", "1"],
        env = env, capture_output = True, text = True, timeout = 300,
    )
    assert result.returncode == 0, result.stderr
    expected = ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                str(LAUNCH_PATH), str(target), "--epochs", "1"]
    command = result.stdout.splitlines()[-1]
    assert shlex.split(command)[1:] == expected
    assert os.path.samefile(shlex.split(command)[0], sys.executable)


@pytest.mark.parametrize("command, install", [
    ("pip install unsloth", True),
    ("pip install --upgrade -qqq uv", True),
    ("uv pip install --system -qqq \"trl==0.22.2\"", True),
    ("python -m pip install -qU uv --root-user-action=ignore", True),
    ("uv pip uninstall --system -qqq fla-core", True),
    ("command -v zstd >/dev/null 2>&1 || (apt-get -qq update && apt-get -qq install -y zstd)", True),
    ("curl -fsSL https://ollama.com/install.sh | sh", True),
    ("unsloth install-kernels mamba_ssm", True),
    ("git clone https://github.com/meta-pytorch/OpenEnv.git", False),
    ("wget -qqq https://example.com/a.mp3 -O audio.mp3", False),
    ("mkdir -p ../data", False),
    ("uv venv -q --allow-existing --python 3.12 litert_env", False),
    ("echo pip is great", False),
])
def test_install_detection(fl, command, install):
    assert fl.is_install_command(command) is install


def test_expansion_matches_ipython(fl):
    from types import SimpleNamespace

    def expand(cmd, **local):
        return fl._expand(cmd, SimpleNamespace(f_globals = {}, f_locals = local))

    assert expand("echo {x} $x '$x'", x = 5) == "echo 5 5 '$x'"
    assert expand("echo {x*2} $$HOME", x = 5) == "echo 10 $HOME"
    # One unknown name leaves the whole command alone, like IPython.
    assert expand("echo {x} {nope}", x = 5) == "echo {x} {nope}"
    assert expand("mkdir -p data/{pdf,html}") == "mkdir -p data/{pdf,html}"
    assert expand('ROCM="$(awk \'{print $1}\')"') == 'ROCM="$(awk \'{print $1}\')"'


def test_minimal_transform(fl):
    body = textwrap.dedent("""\
        import os
        if x:
            !uv pip install -qqq \\
                "torch>=2.8.0" {_numpy}
        y = !ls
        %cd OpenEnv
        %env A=1
    """)
    out = fl._minimal_transform(body)
    assert "get_ipython().system('uv pip install -qqq          \"torch>=2.8.0\" {_numpy}')" in out
    assert "y = get_ipython().getoutput('ls')" in out
    assert "get_ipython().run_line_magic('cd', 'OpenEnv')" in out
    assert "get_ipython().run_line_magic('env', 'A=1')" in out
    compile(out, "<cell>", "exec")


def test_env_magic_forms(fl, monkeypatch, capsys):
    shell = fl.FSDP2Shell()
    monkeypatch.delenv("FSDP2_T", raising = False)
    shell.run_line_magic("env", "FSDP2_T=1")
    assert os.environ["FSDP2_T"] == "1"
    shell.run_line_magic("env", "FSDP2_T 2")
    assert os.environ["FSDP2_T"] == "2"
    # IPython keeps a trailing comment in the value; so does the adapter.
    shell.run_line_magic("env", "FSDP2_T = 1 # note")
    assert os.environ["FSDP2_T"] == "1 # note"
    assert shell.run_line_magic("env", "FSDP2_T") == "1 # note"
    assert shell.run_line_magic("env", "")["FSDP2_T"] == "1 # note"
    monkeypatch.delenv("FSDP2_T")


def test_unknown_magic_raises(fl):
    shell = fl.FSDP2Shell()
    with pytest.raises(NotImplementedError, match = "matplotlib"):
        shell.run_line_magic("matplotlib", "inline")
    with pytest.raises(NotImplementedError, match = "html"):
        shell.run_cell_magic("html", "", "<b>x</b>")


# ---------------------------------------------------------------------------------------
# Real two-rank torchrun on CPU.
# ---------------------------------------------------------------------------------------
_FAKE_MIRROR = textwrap.dedent('''\
    import json, os, subprocess, sys
    rank = os.environ["RANK"]
    log = os.environ["FAKE_LOG"]
    word = "sys"
    get_ipython().system("echo system-{word} >> " + log)
    get_ipython().run_cell_magic("bash", "", 'echo bash-cell >> "$FAKE_LOG"\\n')
    get_ipython().run_cell_magic("file", "written.txt", "hello\\n")
    written = open("written.txt").read()
    get_ipython().run_line_magic("env", "A_VAR=1")
    os.makedirs("sub", exist_ok = True)
    get_ipython().run_line_magic("cd", "sub")
    cwd = os.getcwd()
    get_ipython().run_cell_magic("capture", "", "!echo capture-bang >> " + log + "\\nx = 1\\nprint('hidden-output')\\n")
    get_ipython().system("false")
    false_code = _exit_code
    try:
        get_ipython().run_cell_magic("bash", "", "exit 3\\n")
        bash_code = 0
    except subprocess.CalledProcessError as error:
        bash_code = error.returncode
    get_ipython().system("pip install fake-pkg-xyz")
    get_ipython().run_cell_magic("capture", "", "!uv pip install --system fake-pkg-abc\\n!echo after-install >> " + log + "\\n")
    try:
        get_ipython().run_line_magic("matplotlib", "inline")
        unknown = ""
    except NotImplementedError as error:
        unknown = str(error)
    lines = get_ipython().getoutput("echo a; echo b")
    with open(os.path.join(os.environ["FAKE_OUT"], f"rank{rank}.json"), "w") as f:
        json.dump({
            "written": written, "a_var": os.environ.get("A_VAR"), "cwd": cwd, "x": x,
            "false_code": false_code, "bash_code": bash_code, "unknown": unknown,
            "lines": lines, "fsdp2": os.environ.get("UNSLOTH_FSDP2"), "argv": sys.argv,
            "file": __file__,
        }, f)
    if os.environ.get("FAKE_FAIL_RANK") == rank:
        sys.exit(7)
''')

_LAUNCHER_STUB = textwrap.dedent('''\
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unsloth_fsdp2_launch import main
    if __name__ == "__main__":
        main(__file__, {name!r})
''')


def _torchrun_case(tmp_path, extra_env):
    root = tmp_path / "repo"
    (root / "python_scripts").mkdir(parents = True)
    (root / "fsdp2_scripts").mkdir()
    (root / "python_scripts" / SCRIPT_NAME).write_text(_FAKE_MIRROR)
    (root / "fsdp2_scripts" / "unsloth_fsdp2_launch.py").write_text(LAUNCH_PATH.read_text())
    launcher = root / "fsdp2_scripts" / SCRIPT_NAME
    launcher.write_text(_LAUNCHER_STUB.format(name = SCRIPT_NAME))
    # A `pip` / `uv` on PATH that only records it was called, so "installs" are
    # observable without installing anything.
    shim = tmp_path / "shim"
    shim.mkdir()
    for tool in ("pip", "uv"):
        (shim / tool).write_text(f'#!/bin/bash\necho "{tool}-ran $*" >> "$FAKE_LOG"\n')
        (shim / tool).chmod(0o755)
    work, out, tmpdir = tmp_path / "work", tmp_path / "out", tmp_path / "tmp"
    for d in (work, out, tmpdir):
        d.mkdir()
    log = tmp_path / "shell.log"
    log.touch()
    env = {k: v for k, v in os.environ.items()
           if k not in ("LOCAL_RANK", "TORCHELASTIC_RUN_ID", "UNSLOTH_FSDP2",
                        "UNSLOTH_NPROC_PER_NODE", "UNSLOTH_FSDP2_RUN_INSTALLS")}
    env.update(
        CUDA_VISIBLE_DEVICES = "", PATH = f"{shim}{os.pathsep}{env.get('PATH', '')}",
        FAKE_LOG = str(log), FAKE_OUT = str(out), TMPDIR = str(tmpdir), OMP_NUM_THREADS = "1",
    )
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, str(launcher), "--nproc-per-node", "2", "arg one", "--lr=1"],
        cwd = work, env = env, capture_output = True, text = True, timeout = 600,
    )
    ranks = {p.stem: json.loads(p.read_text()) for p in out.glob("rank*.json")}
    return result, log.read_text().splitlines(), ranks, work, root


@pytest.mark.skipif(not _HAS_TORCH, reason = "torch is not importable")
def test_real_torchrun_two_ranks_on_cpu(tmp_path):
    result, log, ranks, work, root = _torchrun_case(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(ranks) == ["rank0", "rank1"], result.stdout + result.stderr
    # Shell steps ran once for the node, not once per rank.
    assert log.count("system-sys") == 1
    assert log.count("bash-cell") == 1
    assert log.count("capture-bang") == 1
    assert log.count("after-install") == 1
    # Installs are skipped by default, with a note naming the command.
    assert not any(line.startswith(("pip-ran", "uv-ran")) for line in log)
    assert "skipped install step: pip install fake-pkg-xyz" in result.stderr
    assert "UNSLOTH_FSDP2_RUN_INSTALLS=1" in result.stderr
    assert "hidden-output" not in result.stdout + result.stderr
    assert (work / "written.txt").read_text() == "hello\n"
    target = str(root / "python_scripts" / SCRIPT_NAME)
    for seen in ranks.values():
        assert seen["written"] == "hello\n"
        assert seen["a_var"] == "1"
        assert seen["cwd"] == str(work / "sub")
        assert seen["x"] == 1
        assert seen["false_code"] == 1
        assert seen["bash_code"] == 3
        assert "matplotlib" in seen["unknown"]
        assert seen["lines"] == ["a", "b"]
        assert seen["fsdp2"] == "1"
        assert seen["argv"] == [target, "arg one", "--lr=1"]
        assert seen["file"] == target


@pytest.mark.skipif(not _HAS_TORCH, reason = "torch is not importable")
def test_real_torchrun_installs_opt_in_and_failing_rank(tmp_path):
    result, log, ranks, _, _ = _torchrun_case(
        tmp_path, {"UNSLOTH_FSDP2_RUN_INSTALLS": "1", "FAKE_FAIL_RANK": "1"},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert log.count("pip-ran install fake-pkg-xyz") == 1
    assert log.count("uv-ran pip install --system fake-pkg-abc") == 1
    assert log.count("system-sys") == 1
    assert "skipped install step" not in result.stderr
    assert "rank1" in ranks
