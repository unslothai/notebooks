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
     codes reach every rank, that an install runs once when its package is
     missing and is skipped when it is installed (``UNSLOTH_FSDP2_RUN_INSTALLS``
     forces either way), and that a failing rank fails the launch. Skipped when
     torch is not importable.

  3. Subprocess checks of the launch environment: marker folders that are
     never reused across launches, a ``LOCAL_RANK`` left in the shell,
     multiprocessing spawn / forkserver children of the script.

  4. The marker protocol played one rank at a time in-process: a private
     folder, writes that do not follow planted symlinks, rejected malformed
     markers, the arrival + command-hash handshake, and local rank 0's
     install decision, output and errors reaching the other rank. Plus log
     redaction, bash on Windows, and single-process scripts.

No GPU is used: the subprocesses run with ``CUDA_VISIBLE_DEVICES`` empty.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_PATH = REPO_ROOT / "scripts" / "fsdp2_launch.py"

_HAS_TORCH = importlib.util.find_spec("torch") is not None

# Every variable the helper reads to decide what kind of process it is in.
_LAUNCH_ENV = (
    "LOCAL_RANK", "TORCHELASTIC_RUN_ID", "UNSLOTH_FSDP2", "UNSLOTH_NPROC_PER_NODE",
    "UNSLOTH_FSDP2_RUN_INSTALLS", "UNSLOTH_FSDP2_RUN_ID", "UNSLOTH_FSDP2_MP_TARGET",
    "UNSLOTH_FSDP2_WAIT_TIMEOUT", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "MASTER_PORT",
    "SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
)


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _LAUNCH_ENV}
    env.update(CUDA_VISIBLE_DEVICES = "", OMP_NUM_THREADS = "1")
    env.update(extra)
    return env


def _load_launch():
    spec = importlib.util.spec_from_file_location("_fsdp2_launch_under_test", str(LAUNCH_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fl(monkeypatch):
    for name in _LAUNCH_ENV:
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


@pytest.mark.parametrize("env", [
    {"LOCAL_RANK": "0", "WORLD_SIZE": "2"},
    {"LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "4"},
    {"LOCAL_RANK": "0", "WORLD_SIZE": "1"},
    {"TORCHELASTIC_RUN_ID": "abc"},
])
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
    env = _clean_env(UNSLOTH_NPROC_PER_NODE = "2")
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
    get_ipython().system("pip install -q --upgrade pytest")
    get_ipython().run_cell_magic("capture", "", "!uv pip install --system fake-pkg-abc\\n!echo after-install >> " + log + "\\n")
    try:
        get_ipython().run_line_magic("matplotlib", "inline")
        unknown = ""
    except NotImplementedError as error:
        unknown = str(error)
    lines = get_ipython().getoutput("echo a; echo b")
    once_lines = get_ipython().getoutput("echo getoutput-ran >> " + log + "; echo $RANK")
    with open(os.path.join(os.environ["FAKE_OUT"], f"rank{rank}.json"), "w") as f:
        json.dump({
            "written": written, "a_var": os.environ.get("A_VAR"), "cwd": cwd, "x": x,
            "false_code": false_code, "bash_code": bash_code, "unknown": unknown,
            "lines": lines, "once_lines": once_lines, "fsdp2": os.environ.get("UNSLOTH_FSDP2"), "argv": sys.argv,
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


def _fake_pip_env(tmp_path, log):
    """Env for observing installs without installing anything: a ``pip`` package and an
    ``unsloth`` console script on PYTHONPATH (what this interpreter runs) that record
    their arguments, and a ``pip`` / ``python`` / ``unsloth`` first on PATH that belong to
    no environment and must never run."""
    fake = tmp_path / "fake_pip"
    (fake / "pip").mkdir(parents = True)
    (fake / "pip" / "__init__.py").write_text("")
    (fake / "pip" / "__main__.py").write_text(
        "import sys\n"
        f"with open({str(log)!r}, 'a') as f:\n"
        "    f.write('pip-ran ' + ' '.join(sys.argv[1:]) + '\\n')\n"
    )
    (fake / "fake_unsloth_cli.py").write_text(
        "import sys\n"
        "def main():\n"
        f"    with open({str(log)!r}, 'a') as f:\n"
        "        f.write(sys.argv[0] + '-ran ' + ' '.join(sys.argv[1:]) + '\\n')\n"
    )
    info = fake / "fake_unsloth_cli-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-unsloth-cli\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text("[console_scripts]\nunsloth = fake_unsloth_cli:main\n")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    for tool in ("pip", "pip3", "python", "python3", "unsloth"):
        (decoy / tool).write_text(f'#!/bin/bash\necho "path-{tool}-ran $*" >> {shlex.quote(str(log))}\n')
        (decoy / tool).chmod(0o755)
    pythonpath = os.pathsep.join(filter(None, [str(fake), os.environ.get("PYTHONPATH", "")]))
    return {"PYTHONPATH": pythonpath, "decoy": str(decoy)}


def _torchrun_case(tmp_path, extra_env):
    root = tmp_path / "repo"
    (root / "python_scripts").mkdir(parents = True)
    (root / "fsdp2_scripts").mkdir()
    (root / "python_scripts" / SCRIPT_NAME).write_text(_FAKE_MIRROR)
    (root / "fsdp2_scripts" / "unsloth_fsdp2_launch.py").write_text(LAUNCH_PATH.read_text())
    launcher = root / "fsdp2_scripts" / SCRIPT_NAME
    launcher.write_text(_LAUNCHER_STUB.format(name = SCRIPT_NAME))
    # "Installs" are observable without installing anything: `pip` reaches a fake pip
    # module through this interpreter, and `uv` on PATH only records it was called.
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "uv").write_text('#!/bin/bash\necho "uv-ran $*" >> "$FAKE_LOG"\n')
    (shim / "uv").chmod(0o755)
    work, out, tmpdir = tmp_path / "work", tmp_path / "out", tmp_path / "tmp"
    for d in (work, out, tmpdir):
        d.mkdir()
    log = tmp_path / "shell.log"
    log.touch()
    fake = _fake_pip_env(tmp_path, log)
    env = _clean_env()
    env.update(
        PATH = f"{fake['decoy']}{os.pathsep}{shim}{os.pathsep}{env.get('PATH', '')}",
        PYTHONPATH = fake["PYTHONPATH"],
        FAKE_LOG = str(log), FAKE_OUT = str(out), TMPDIR = str(tmpdir),
    )
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, str(launcher), "--nproc-per-node", "2", "arg one", "--lr=1"],
        cwd = work, env = env, capture_output = True, text = True, timeout = 600,
    )
    ranks = {p.stem: json.loads(p.read_text()) for p in out.glob("rank*.json")}
    leftover = sorted(p.name for p in tmpdir.glob("unsloth_fsdp2_*"))
    return result, log.read_text().splitlines(), ranks, work, root, leftover


@pytest.mark.skipif(not _HAS_TORCH, reason = "torch is not importable")
def test_real_torchrun_two_ranks_on_cpu(tmp_path):
    result, log, ranks, work, root, leftover = _torchrun_case(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(ranks) == ["rank0", "rank1"], result.stdout + result.stderr
    # Shell steps ran once for the node, not once per rank.
    assert log.count("system-sys") == 1
    assert log.count("bash-cell") == 1
    assert log.count("capture-bang") == 1
    assert log.count("after-install") == 1
    assert log.count("getoutput-ran") == 1
    # By default an install of a missing package runs once per node, and one whose
    # package is already installed is skipped with a note naming the command.
    assert log.count("pip-ran install fake-pkg-xyz") == 1
    assert log.count("uv-ran pip install --system fake-pkg-abc") == 1
    assert not any(line.startswith("path-") for line in log)
    assert not any("pytest" in line for line in log)
    assert "skipped install step (already installed: pytest): pip install -q --upgrade pytest" in result.stderr
    assert "UNSLOTH_FSDP2_RUN_INSTALLS=1" in result.stderr
    # The marker folder is removed once both ranks are done with it.
    assert leftover == []
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
        # x = !cmd runs once per node and every rank gets local rank 0's output.
        assert seen["once_lines"] == ["0"]
        assert seen["fsdp2"] == "1"
        assert seen["argv"] == [target, "arg one", "--lr=1"]
        assert seen["file"] == target


@pytest.mark.skipif(not _HAS_TORCH, reason = "torch is not importable")
def test_real_torchrun_installs_opt_in_and_failing_rank(tmp_path):
    result, log, ranks, _, _, _ = _torchrun_case(
        tmp_path, {"UNSLOTH_FSDP2_RUN_INSTALLS": "1", "FAKE_FAIL_RANK": "1"},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert log.count("pip-ran install fake-pkg-xyz") == 1
    assert log.count("pip-ran install -q --upgrade pytest") == 1
    assert log.count("uv-ran pip install --system fake-pkg-abc") == 1
    assert log.count("system-sys") == 1
    assert "skipped install step" not in result.stderr
    assert "rank1" in ranks


@pytest.mark.skipif(not _HAS_TORCH, reason = "torch is not importable")
def test_real_torchrun_installs_opt_out(tmp_path):
    result, log, ranks, _, _, _ = _torchrun_case(tmp_path, {"UNSLOTH_FSDP2_RUN_INSTALLS": "0"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(line.startswith(("pip-ran", "uv-ran")) for line in log)
    assert "skipped install step (UNSLOTH_FSDP2_RUN_INSTALLS=0): pip install fake-pkg-xyz" in result.stderr
    assert log.count("after-install") == 1
    assert sorted(ranks) == ["rank0", "rank1"]


# ---------------------------------------------------------------------------------------
# Install decisions.
# ---------------------------------------------------------------------------------------
def _fake_dist(site, name, version):
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents = True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


@pytest.fixture
def installed(fl, tmp_path, monkeypatch):
    """A site folder on sys.path holding fake installed distributions, and a PATH with
    only the programs a test puts there."""
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(fl, "_dpkg_installed", lambda package: False)
    # Only the fake site counts, and the test interpreter counts as a virtual environment
    # (which the shell steps name to uv through VIRTUAL_ENV), whatever runs the tests.
    monkeypatch.setattr(fl, "_in_venv", lambda: True)
    for name in ("UV_PYTHON", "UV_SYSTEM_PYTHON", *fl._PIP_ENV_REDIRECTS):
        monkeypatch.delenv(name, raising = False)
    import importlib.metadata
    monkeypatch.setattr(fl, "_installed_versions", lambda: {
        fl._normalize(dist.metadata["Name"]): dist.version
        for dist in importlib.metadata.distributions(path = [str(site)])
    })

    def add_dist(name, version = "1.0"):
        _fake_dist(site, name, version)

    def add_program(name):
        (bin_dir / name).write_text("#!/bin/sh\n")
        (bin_dir / name).chmod(0o755)

    return add_dist, add_program


def _skip(fl, command):
    decision = fl.install_decision(command)
    assert decision is not None, command
    return decision[0]


# Lines exactly as they appear in python_scripts/, so the decision is tested on the real
# install steps rather than on a paraphrase of them.
_REAL_LINES = {
    "DeepSeek_R1_0528_Qwen3_(8B)_GRPO.py": "pip install langid -qq",
    "Gemma3_(270M).py": 'uv pip install -q --python litert_env/bin/python --torch-backend cpu '
                        'litert-torch==0.10.0 litert-lm==0.18.1 tensorflow-cpu==2.21.0 "transformers<5.21"',
    "Llama3_(8B)-Ollama.py": "command -v zstd >/dev/null 2>&1 || (apt-get -qq update && "
                             "apt-get -qq install -y zstd) >/dev/null 2>&1",
}
_OLLAMA_LINE = "curl -fsSL https://ollama.com/install.sh | sh"


def test_real_install_lines_are_still_in_the_scripts():
    for name, line in _REAL_LINES.items():
        assert repr(line)[1:-1] in (REPO_ROOT / "python_scripts" / name).read_text(), name
    assert _OLLAMA_LINE in (REPO_ROOT / "python_scripts" / "Llama3_(8B)-Ollama.py").read_text()


def test_install_runs_only_when_the_package_is_missing(fl, installed):
    add_dist, _ = installed
    line = _REAL_LINES["DeepSeek_R1_0528_Qwen3_(8B)_GRPO.py"]
    assert _skip(fl, line) is False
    add_dist("langid", "1.1.6")
    assert _skip(fl, line) is True
    assert "already installed: langid" in fl.install_decision(line)[1]


def test_install_into_another_environment_always_runs(fl, installed):
    add_dist, _ = installed
    for name in ("litert-torch", "litert-lm", "tensorflow-cpu", "transformers"):
        add_dist(name, "0.10.0")
    assert _skip(fl, _REAL_LINES["Gemma3_(270M).py"]) is False
    add_dist("langid")
    assert _skip(fl, "pip install --target vendored langid") is False
    assert _skip(fl, "pip install --prefix=/opt/x langid") is False
    assert _skip(fl, "uv venv -q env && uv pip install langid") is False


def test_apt_and_ollama_run_only_when_the_program_is_missing(fl, installed):
    _, add_program = installed
    apt = _REAL_LINES["Llama3_(8B)-Ollama.py"]
    assert _skip(fl, apt) is False
    assert _skip(fl, _OLLAMA_LINE) is False
    add_program("zstd")
    add_program("ollama")
    assert _skip(fl, apt) is True
    assert _skip(fl, _OLLAMA_LINE) is True


@pytest.mark.parametrize("command, skip", [
    ("pip install --upgrade -qqq langid", True),
    ("python -m pip install -qU langid --root-user-action=ignore", True),
    ("uv pip install -qqq 'langid==1.1.6'", True),
    ("uv pip install -qqq 'langid>=2'", False),
    ("pip install --no-deps --force-reinstall langid unsloth_zoo", True),
    ("pip install langid other-missing-pkg", False),
    ("pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git", True),
    ('pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git@main"', True),
    ("pip install git+https://github.com/x/whatever.git#egg=langid", True),
    ("pip install git+https://github.com/x/missing-repo.git", False),
    ("pip install unsloth  # Do this in local & cloud setups", True),
    ("pip install \"langid; python_version < '3'\" ", True),
    ("pip install -r requirements.txt", False),
    ("pip install -e .", False),
    ("pip install {_vllm}", False),
    ("pip uninstall -y langid", False),
    ("pip uninstall -y never-installed-pkg", True),
    ("unsloth install-kernels", False),
])
def test_install_decision_forms(fl, installed, command, skip):
    add_dist, _ = installed
    add_dist("langid", "1.1.6")
    add_dist("unsloth_zoo", "2026.10.1")
    add_dist("unsloth", "2026.10.1")
    assert _skip(fl, command) is skip


def test_non_install_commands_have_no_decision(fl):
    assert fl.install_decision("git clone https://github.com/x/y.git") is None
    assert fl.install_decision("uv venv -q --allow-existing --python 3.12 litert_env") is None


def _shim_pip(tmp_path, monkeypatch):
    log = tmp_path / "pip.log"
    fake = _fake_pip_env(tmp_path, log)
    monkeypatch.setenv("PYTHONPATH", fake["PYTHONPATH"])
    monkeypatch.setenv("PATH", f"{fake['decoy']}{os.pathsep}{os.environ['PATH']}")
    return log


def test_shell_steps_use_this_interpreter_not_path(fl, tmp_path, monkeypatch):
    """The install check reads this interpreter's packages, so pip / python in a shell
    step must be this interpreter, not the first ones on PATH."""
    log = _shim_pip(tmp_path, monkeypatch)
    shell = fl.FSDP2Shell()
    shell._system("pip install never-installed-pkg-xyz")
    shell._system("pip3 install never-installed-pkg-abc")
    shell._system("python -m pip install never-installed-pkg-def")
    shell._system("unsloth install-kernels never-installed-kernel-xyz")
    out = shell.getoutput("python -c 'import sys; print(sys.executable)'; python3 -c 'print(1)'")
    ran = log.read_text()
    assert "pip-ran install never-installed-pkg-xyz" in ran
    assert "pip-ran install never-installed-pkg-abc" in ran
    assert "pip-ran install never-installed-pkg-def" in ran
    # install-kernels installs into the interpreter its command runs on.
    assert "unsloth-ran install-kernels never-installed-kernel-xyz" in ran
    assert "path-" not in ran
    assert out == [sys.executable, "1"]
    # A bash started by the step sees the same functions.
    assert shell.getoutput("bash -c 'type -t pip python'") == ["function", "function"]


def test_wrapper_interpreter_does_not_recurse(fl, tmp_path, monkeypatch):
    """$UNSLOTH_FSDP2_PYTHON may be a bash wrapper (sys.executable is the wrapper under
    `exec -a`). One that runs `python3` / `pip` by name imports the exported functions and
    must reach PATH's command, not call itself again."""
    bindir = tmp_path / "dir (with) spaces"
    bindir.mkdir()
    wrapper = bindir / "py wrapper"
    wrapper.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        export WRAP_DEPTH=$((${{WRAP_DEPTH:-0}} + 1))
        [ "$WRAP_DEPTH" -gt 3 ] && {{ echo recursed >&2; exit 97; }}
        if [ "$1" = -m ] && [ "$2" = pip ]; then shift 2; pip "$@"; else python3 "$@"; fi
    """))
    wrapper.chmod(0o755)
    real = tmp_path / "real"
    real.mkdir()
    (real / "python3").symlink_to(sys.executable)
    (real / "pip").write_text("#!/bin/sh\necho real-pip \"$@\" depth=$WRAP_DEPTH\n")
    (real / "pip").chmod(0o755)
    monkeypatch.setenv("PATH", f"{real}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(sys, "executable", str(wrapper))
    monkeypatch.setenv("_UNSLOTH_FSDP2_PINNED", "1")  # inherited from an outer launcher
    shell = fl.FSDP2Shell()
    assert shell.getoutput("python3 -c 'print(6 * 7)'; echo rc=$?") == ["42", "rc=0"]
    assert shell.getoutput("pip --version; echo rc=$?") == ["real-pip --version depth=1", "rc=0"]


def test_pinned_commands_work_under_nounset(fl):
    """A %%bash cell or !cmd that runs `set -u` (nounset) still reaches the pinned
    python / pip: the recursion guard variable is unset there."""
    shell = fl.FSDP2Shell()
    assert shell.getoutput("set -u; python -c 'print(6 * 7)'; echo rc=$?") == ["42", "rc=0"]
    assert shell.getoutput("set -euo pipefail; pip --version >/dev/null; echo rc=$?") == ["rc=0"]
    assert shell.getoutput("set -u; bash -c 'set -u; python3 -c \"print(5)\"'") == ["5"]


@pytest.mark.parametrize("in_venv", [True, False])
def test_virtual_env_names_this_interpreter_for_uv(fl, monkeypatch, in_venv):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/other/venv")
    monkeypatch.setattr(sys, "prefix", "/this/venv")
    monkeypatch.setattr(sys, "base_prefix", "/usr" if in_venv else "/this/venv")
    env = fl._shell_env()
    assert env["UNSLOTH_FSDP2_PYTHON"] == sys.executable
    assert env["VIRTUAL_ENV"] == ("/this/venv" if in_venv else "/some/other/venv")


@pytest.mark.parametrize("command, variable, value", [
    ("uv pip install langid", "UV_PYTHON", "/other/env/bin/python"),
    ("uv pip install langid", "UV_PYTHON", "3.12"),
    ("pip install langid", "PIP_TARGET", "/other/site"),
    ("pip install langid", "PIP_PREFIX", "/other"),
    ("pip install langid", "PIP_ROOT", "/other"),
    ("pip install langid", "PIP_PYTHON", "/other/env/bin/python"),
    ("python -m pip uninstall -y missing-pkg", "PIP_TARGET", "/other/site"),
    ("unsloth install-kernels langid", "PIP_TARGET", "/other/site"),
])
def test_inherited_redirection_always_runs(fl, installed, monkeypatch, command, variable, value):
    """The variable forms of --python / --target / --prefix / --root: the install goes
    where the check did not look, so it runs instead of being skipped."""
    add_dist, _ = installed
    add_dist("langid")
    assert _skip(fl, command) is True
    monkeypatch.setenv(variable, value)
    skip, reason = fl.install_decision(command)
    assert skip is False and variable in reason


@pytest.mark.parametrize("command", [
    "PIP_TARGET=/other/site pip install langid",
    "UV_PYTHON=/other/env/bin/python uv pip install langid",
    "PIP_NO_CACHE_DIR=1 pip install langid",
    "env PIP_TARGET=/other/site pip install langid",
    "export PIP_PREFIX=/other; pip install langid",
    "cd /elsewhere && uv pip install langid",
    "source /other/env/bin/activate && pip install langid",
    "conda activate other && pip install langid",
    "sudo pip install langid",
    "exec pip install langid",
    "pip3.12 install langid",
    "/other/env/bin/pip install langid",
    "/other/env/bin/python -m pip install langid",
    "PIP_TARGET=/other/site unsloth install-kernels langid",
])
def test_installs_that_may_land_elsewhere_always_run(fl, installed, command):
    """Only a plain command is trusted to install into this interpreter: inline variables,
    environment or directory changes, sudo / env / exec, and pip / python by version or
    another path all run instead of being skipped."""
    add_dist, _ = installed
    add_dist("langid")
    skip, reason = fl.install_decision(command)
    assert skip is False, reason


def test_plain_commands_for_this_interpreter_still_skip(fl, installed):
    add_dist, _ = installed
    add_dist("langid")
    for command in ("pip install langid", "pip3 install langid", "python -m pip install langid",
                    "python3 -m pip install langid", f"{sys.executable} -m pip install langid",
                    "uv pip install langid", "mkdir -p out && pip install langid",
                    "unsloth install-kernels langid"):
        assert _skip(fl, command) is True, command


def test_redirection_variables_that_name_this_environment_still_skip(fl, installed, monkeypatch):
    add_dist, _ = installed
    add_dist("langid")
    for target in (sys.executable, sys.prefix):
        monkeypatch.setenv("UV_PYTHON", target)
        assert _skip(fl, "uv pip install langid") is True
    monkeypatch.setenv("UV_PYTHON", "/other/env/bin/python")
    assert _skip(fl, "pip install langid") is True  # UV_PYTHON is not read by pip
    monkeypatch.setenv("PIP_TARGET", "/other/site")
    monkeypatch.delenv("UV_PYTHON")
    assert _skip(fl, "uv pip install langid") is True  # nor PIP_* by uv


@pytest.mark.parametrize("in_venv, command, system_env, skip", [
    (True, "uv pip install langid", None, True),
    (True, "uv pip install --system -qqq langid", None, False),
    (True, "uv pip install langid", "true", False),
    # Outside a virtual environment uv takes VIRTUAL_ENV, CONDA_PREFIX or a .venv above
    # the working directory, or its own system interpreter: never provably this one.
    (False, "uv pip install langid", None, False),
    (False, "uv pip install --system -qqq langid", None, False),
    (False, "uv pip install langid", "true", False),
])
def test_uv_installs_skip_only_inside_this_virtual_environment(
        fl, installed, monkeypatch, in_venv, command, system_env, skip):
    add_dist, _ = installed
    add_dist("langid")
    if system_env:
        monkeypatch.setenv("UV_SYSTEM_PYTHON", system_env)
    monkeypatch.setattr(fl, "_in_venv", lambda: in_venv)
    assert _skip(fl, command) is skip


def test_pip_from_outside_a_virtual_environment_still_skips(fl, installed, monkeypatch):
    """pip is this interpreter's -m pip whatever uv would discover, so VIRTUAL_ENV,
    CONDA_PREFIX and a .venv do not matter to it."""
    add_dist, _ = installed
    add_dist("langid")
    monkeypatch.setattr(fl, "_in_venv", lambda: False)
    monkeypatch.setenv("VIRTUAL_ENV", "/other/venv")
    monkeypatch.setenv("CONDA_PREFIX", "/other/conda")
    assert _skip(fl, "pip install langid") is True


@pytest.mark.parametrize("setting, ran_missing, ran_present", [
    (None, True, False), ("1", True, True), ("0", False, False),
])
def test_single_process_install_rules_match_the_multi_gpu_ones(
        fl, tmp_path, monkeypatch, capfd, setting, ran_missing, ran_present):
    log = _shim_pip(tmp_path, monkeypatch)
    if setting is not None:
        monkeypatch.setenv("UNSLOTH_FSDP2_RUN_INSTALLS", setting)
    shell = fl.FSDP2Shell()
    assert shell.distributed is False
    shell._system("pip install never-installed-pkg-xyz")
    shell._system("pip install pytest")
    ran = log.read_text() if log.exists() else ""
    assert ("never-installed-pkg-xyz" in ran) is ran_missing
    assert ("install pytest" in ran) is ran_present
    err = capfd.readouterr().err
    if not ran_present:
        assert "skipped install step" in err and "pip install pytest" in err


# ---------------------------------------------------------------------------------------
# Launch environment.
# ---------------------------------------------------------------------------------------
def test_each_launch_gets_a_fresh_run_id(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    seen = []
    for _ in range(2):
        with pytest.raises(_Exec):
            fl.main(str(launcher), SCRIPT_NAME, [])
        seen.append(os.environ["UNSLOTH_FSDP2_RUN_ID"])
    assert len(calls) == 2 and seen[0] != seen[1] and all(seen)


def test_marker_folder_is_named_by_run_id_or_parent(fl, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_PORT", "29500")
    without = os.path.basename(fl.FSDP2Shell()._markers())
    assert f"p{os.getppid()}" in without
    monkeypatch.setenv("UNSLOTH_FSDP2_RUN_ID", "abc123")
    assert os.path.basename(fl.FSDP2Shell()._markers()).startswith("unsloth_fsdp2_abc123_")


_FAKE_LAUNCHER = textwrap.dedent('''\
    # Starts two local ranks the way deepspeed / a hand-rolled launcher does: LOCAL_RANK,
    # WORLD_SIZE and a fixed MASTER_PORT, no TORCHELASTIC_RUN_ID.
    import os, subprocess, sys
    procs = [subprocess.Popen([sys.executable] + sys.argv[1:],
                              env = {**os.environ, "LOCAL_RANK": str(r), "RANK": str(r),
                                     "WORLD_SIZE": "2", "MASTER_PORT": "29500"})
             for r in (0, 1)]
    sys.exit(max(p.wait() for p in procs))
''')

_STALE_TARGET = textwrap.dedent('''\
    import os, time
    if os.environ["LOCAL_RANK"] == "0":
        time.sleep(2)
    get_ipython().run_cell_magic("file", "shared.txt", os.environ["PAYLOAD"])
    with open(os.environ["OUT"] + "." + os.environ["LOCAL_RANK"], "w") as f:
        f.write(open("shared.txt").read())
''')


def test_markers_are_never_reused_by_a_later_launch(tmp_path):
    (tmp_path / "fake_launcher.py").write_text(_FAKE_LAUNCHER)
    (tmp_path / "target.py").write_text(_STALE_TARGET)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    for payload in ("first", "second"):
        env = _clean_env(TMPDIR = str(tmpdir), PAYLOAD = payload, OUT = str(tmp_path / payload))
        result = subprocess.run(
            [sys.executable, str(tmp_path / "fake_launcher.py"), str(LAUNCH_PATH), str(tmp_path / "target.py")],
            cwd = tmp_path, env = env, capture_output = True, text = True, timeout = 300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        # Rank 1 must have waited for this launch's write, not an earlier launch's marker.
        assert (tmp_path / f"{payload}.1").read_text() == payload
        assert (tmp_path / f"{payload}.0").read_text() == payload


_HANG_TARGET = 'get_ipython().system("echo ran > " + __import__("os").environ["OUT"])\n'


@pytest.mark.parametrize("leftover", ["0", "1"])
def test_a_local_rank_left_in_the_shell_does_not_hang(tmp_path, leftover):
    (tmp_path / "target.py").write_text(_HANG_TARGET)
    out = tmp_path / "out.txt"
    env = _clean_env(LOCAL_RANK = leftover, OUT = str(out), TMPDIR = str(tmp_path),
                     UNSLOTH_NPROC_PER_NODE = "1")
    result = subprocess.run(
        [sys.executable, str(LAUNCH_PATH), str(tmp_path / "target.py")],
        env = env, capture_output = True, text = True, timeout = 120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.read_text() == "ran\n"
    assert "left over in the shell" in result.stderr


def test_local_rank_without_a_world_is_not_distributed(fl, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert fl.FSDP2Shell().distributed is False
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert fl.FSDP2Shell().distributed is False
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert fl.FSDP2Shell().distributed is True


def test_marker_wait_times_out_with_a_clear_error(fl, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("UNSLOTH_FSDP2_RUN_ID", "timeout-test")
    monkeypatch.setenv("UNSLOTH_FSDP2_WAIT_TIMEOUT", "0.5")
    monkeypatch.setattr(fl.tempfile, "gettempdir", lambda: str(tmp_path))
    shell = fl.FSDP2Shell()
    with pytest.raises(TimeoutError, match = "UNSLOTH_FSDP2_WAIT_TIMEOUT"):
        shell._system("echo never")


@pytest.mark.parametrize("env", [
    {"SLURM_STEP_NUM_TASKS": "2"}, {"OMPI_COMM_WORLD_SIZE": "4"}, {"PMI_SIZE": "2"}, {"WORLD_SIZE": "2"},
])
def test_refuses_to_launch_from_each_slurm_or_mpi_task(fl, layout, monkeypatch, capsys, env):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert raised.value.code == 2 and calls == []
    err = capsys.readouterr().err
    assert "SLURM / MPI" in err and "torchrun" in err and "LOCAL_RANK" in err


def test_a_single_slurm_task_still_launches(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "1")
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert calls[0][1] == _expected(launcher, 2)


def test_ipython_run_uses_a_child_process(fl, layout, monkeypatch):
    import types
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: object()
    monkeypatch.setitem(sys.modules, "IPython", fake)
    ran = []

    def fake_run(command, env = None, **kwargs):
        ran.append((command, env))
        return subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(fl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), SCRIPT_NAME, [])
    assert raised.value.code == 3 and calls == []
    command, env = ran[0]
    assert command == _expected(launcher, 2)
    assert env["UNSLOTH_FSDP2"] == "1" and env["UNSLOTH_FSDP2_RUN_ID"]
    # The kernel's own environment is left alone.
    assert "UNSLOTH_FSDP2" not in os.environ and "UNSLOTH_FSDP2_RUN_ID" not in os.environ


def test_windows_waits_on_a_child_instead_of_execv(fl, layout, monkeypatch):
    # os.execv on Windows detaches the new process, drops argument quoting and exits 0
    # at once (bpo-19124, bpo-20451), so the launcher must wait on torchrun itself.
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    monkeypatch.setattr(fl, "_IS_WINDOWS", True)
    ran = []

    def fake_run(command, env = None, **kwargs):
        ran.append((command, env))
        return subprocess.CompletedProcess(command, 5)

    monkeypatch.setattr(fl.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), SCRIPT_NAME, ["a b (c)"])
    assert raised.value.code == 5 and calls == []
    command, env = ran[0]
    assert command == _expected(launcher, 2, "a b (c)")
    assert env["UNSLOTH_FSDP2"] == "1" and env["UNSLOTH_FSDP2_RUN_ID"]


_MP_TARGET = textwrap.dedent('''\
    import json, multiprocessing as mp, os
    def square(x):
        return x * x
    get_ipython().system("echo top-level >> " + os.environ["FAKE_LOG"])
    get_ipython().run_line_magic("cd", "sub")
    if __name__ == "__main__":
        results = {}
        for method in ("spawn", "forkserver"):
            with mp.get_context(method).Pool(2) as pool:
                results[method] = pool.map(square, [1, 2, 3])
        with open(os.environ["OUT"], "w") as f:
            json.dump(results, f)
''')


def test_spawn_and_forkserver_children_see_get_ipython(tmp_path):
    (tmp_path / "work" / "sub").mkdir(parents = True)
    (tmp_path / "target.py").write_text(_MP_TARGET)
    log, out = tmp_path / "shell.log", tmp_path / "out.json"
    env = _clean_env(FAKE_LOG = str(log), OUT = str(out), UNSLOTH_NPROC_PER_NODE = "1")
    result = subprocess.run(
        [sys.executable, str(LAUNCH_PATH), str(tmp_path / "target.py")],
        cwd = tmp_path / "work", env = env, capture_output = True, text = True, timeout = 180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out.read_text()) == {"spawn": [1, 4, 9], "forkserver": [1, 4, 9]}
    # The children re-import the script but do not repeat its shell steps.
    assert log.read_text().splitlines() == ["top-level"]
    assert "NameError" not in result.stderr


def test_cd_matches_ipython(fl, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    shell = fl.FSDP2Shell()
    assert shell.run_line_magic("cd", "missing-folder") is None
    assert os.getcwd() == str(tmp_path)
    assert "missing-folder" in capsys.readouterr().out
    assert shell.run_line_magic("cd", "sub") is None
    assert os.getcwd() == str(tmp_path / "sub")
    assert capsys.readouterr().out.strip() == str(tmp_path / "sub")


@pytest.mark.parametrize("args, visible, expected", [
    (["--nproc-per-node", "1"], 4, "--nproc-per-node=1 (4 GPU(s) visible): running single-process"),
    (["--nproc-per-node", "0"], 2, "--nproc-per-node=0 (2 GPU(s) visible): running single-process"),
    ([], 1, "1 GPU visible: running single-process"),
    ([], 0, "no GPU visible: running single-process"),
])
def test_single_process_note_names_what_was_asked(fl, layout, monkeypatch, capsys, args, visible, expected):
    _, launcher, _ = layout
    _stub(fl, monkeypatch, visible = visible)
    assert fl.main(str(launcher), SCRIPT_NAME, ["--dry-run", *args]) == 0
    assert expected in capsys.readouterr().out


def test_gpu_count_failure_prints_the_childs_error(fl, monkeypatch, capfd):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout = b"", stderr = b"Traceback\nImportError: boom\n")

    monkeypatch.setattr(fl.subprocess, "run", fake_run)
    assert fl._count_visible_gpus() == 0
    err = capfd.readouterr().err
    assert err.count("ImportError: boom") == 1 and "assuming none is visible" in err


# ---------------------------------------------------------------------------------------
# Marker protocol: private folder, safe writes, validated content, arrival + hash
# handshake, rank 0's decision / output / error reaching every rank.
# ---------------------------------------------------------------------------------------
@pytest.fixture
def ranked(fl, tmp_path, monkeypatch):
    """An in-process FSDP2Shell playing one local rank of two, markers under tmp_path."""
    monkeypatch.setattr(fl.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("UNSLOTH_FSDP2_RUN_ID", "unit")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("UNSLOTH_FSDP2_WAIT_TIMEOUT", "5")

    def make(rank):
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        shell = fl.FSDP2Shell()
        assert shell.distributed
        return shell

    def folder():
        return tmp_path / f"unsloth_fsdp2_unit_r0_{fl.socket.gethostname()}"

    return make, folder


def _done(fl, folder, step, template, kind = "system", **fields):
    payload = {"hash": fl._step_hash(kind, template), "code": 0, "output": None,
               "skipped": False, "error": None}
    payload.update(fields)
    folder.mkdir(mode = 0o700, exist_ok = True)
    (folder / f"{step}.done").write_text(json.dumps(payload))


def _arrive(fl, folder, step, template, kind = "system"):
    folder.mkdir(mode = 0o700, exist_ok = True)
    (folder / f"{step}.arrived.1").write_text(json.dumps({"hash": fl._step_hash(kind, template)}))


@pytest.mark.skipif(os.name == "nt", reason = "POSIX permissions")
def test_marker_folder_is_private(fl, ranked):
    make, folder = ranked
    shell = make(1)
    path = shell._markers()
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason = "POSIX symlinks and permissions")
def test_marker_folder_planted_by_someone_else_is_refused(fl, ranked, tmp_path):
    make, folder = ranked
    elsewhere = tmp_path / "attacker"
    elsewhere.mkdir()
    folder().symlink_to(elsewhere)
    with pytest.raises(RuntimeError, match = "symlink"):
        make(1)._markers()
    folder().unlink()
    folder().mkdir()
    folder().chmod(0o777)
    with pytest.raises(RuntimeError, match = "world-writable"):
        make(1)._markers()


@pytest.mark.skipif(os.name == "nt", reason = "POSIX symlinks")
def test_marker_writes_do_not_follow_planted_symlinks(fl, ranked, tmp_path):
    make, folder = ranked
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n")
    folder().mkdir(mode = 0o700)
    os.symlink(victim, folder() / "1.done")
    os.symlink(victim, folder() / f"1.done.{os.getpid()}.tmp")
    _arrive(fl, folder(), 1, "true")
    make(0)._system("true")
    assert victim.read_text() == "precious\n"
    assert not (folder() / "1.done").is_symlink()


def test_forged_or_malformed_markers_are_rejected(fl, ranked):
    make, folder = ranked
    folder().mkdir(mode = 0o700)
    (folder() / "1.done").write_text("0")
    with pytest.raises(RuntimeError, match = "malformed"):
        make(1)._system("true")


def test_peer_announces_arrival_and_rank0_waits_for_it(fl, ranked):
    make, folder = ranked
    os.environ["UNSLOTH_FSDP2_WAIT_TIMEOUT"] = "0.5"
    with pytest.raises(TimeoutError, match = "local rank 1 to arrive"):
        make(0)._system("echo hi")
    with pytest.raises(TimeoutError):
        make(1)._system("echo hi")
    arrived = json.loads((folder() / "1.arrived.1").read_text())
    assert arrived["hash"] == fl._step_hash("system", "echo hi")


def test_ranks_at_different_steps_fail_instead_of_pairing(fl, ranked):
    make, folder = ranked
    _done(fl, folder(), 1, "echo other")
    with pytest.raises(RuntimeError, match = "diverged"):
        make(1)._system("echo mine")
    _arrive(fl, folder(), 2, "echo other")
    shell = make(0)
    shell.counter = 1
    with pytest.raises(RuntimeError, match = "diverged"):
        shell._system("echo mine")
    recorded = json.loads((folder() / "2.done").read_text())
    assert "diverged" in recorded["error"]["message"]


def test_peers_follow_rank0_install_decision(fl, ranked, tmp_path, monkeypatch):
    make, folder = ranked
    log = _shim_pip(tmp_path, monkeypatch)
    command = "pip install never-installed-pkg-xyz"
    # Rank 0 recorded a skip (say it saw the package installed); rank 1 must not decide
    # again from its own view of the environment, where the package is missing.
    _done(fl, folder(), 1, command, skipped = True)
    assert make(1)._system(command) == 0
    assert not log.exists()


def test_getoutput_runs_once_and_shares_rank0_output(fl, ranked, tmp_path):
    make, folder = ranked
    flag = tmp_path / "ran"
    command = f"touch {flag}; echo mine"
    _done(fl, folder(), 1, command, kind = "getoutput", output = ["from rank 0"])
    assert make(1).getoutput(command) == ["from rank 0"]
    assert not flag.exists()


def test_getoutput_follows_the_install_rules(fl, ranked, tmp_path, monkeypatch):
    make, folder = ranked
    log = _shim_pip(tmp_path, monkeypatch)
    monkeypatch.setenv("UNSLOTH_FSDP2_RUN_INSTALLS", "0")
    _arrive(fl, folder(), 1, "pip install anything", kind = "getoutput")
    assert make(0).getoutput("pip install anything") == []
    assert not log.exists()
    assert json.loads((folder() / "1.done").read_text())["skipped"] is True


def test_file_write_errors_reach_every_rank(fl, ranked, tmp_path):
    make, folder = ranked
    target = tmp_path / "a_directory"
    target.mkdir()
    _arrive(fl, folder(), 1, f"%%file {target}\nhello\n", kind = "file")
    with pytest.raises(IsADirectoryError):
        make(0).run_cell_magic("file", str(target), "hello\n")
    recorded = json.loads((folder() / "1.done").read_text())
    assert recorded["error"]["type"] == "IsADirectoryError"
    with pytest.raises(IsADirectoryError, match = "raised on local rank 0"):
        make(1).run_cell_magic("file", str(target), "hello\n")


@pytest.mark.skipif(not _HAS_TORCH, reason = "torch is not importable")
def test_real_torchrun_divergent_ranks_fail_clearly(tmp_path):
    target = tmp_path / "diverge.py"
    target.write_text('import os\nget_ipython().system("echo rank-" + os.environ["RANK"])\n')
    result = subprocess.run(
        [sys.executable, str(LAUNCH_PATH), "--nproc-per-node", "2", str(target)],
        cwd = tmp_path, env = _clean_env(TMPDIR = str(tmp_path)),
        capture_output = True, text = True, timeout = 300,
    )
    assert result.returncode != 0
    assert "diverged at step 1" in result.stdout + result.stderr


# ---------------------------------------------------------------------------------------
# Adapter injection, credentials in logs, bash on Windows.
# ---------------------------------------------------------------------------------------
def test_worker_uses_the_adapter_even_inside_ipython(fl, layout, monkeypatch):
    import builtins
    tmp, launcher, _ = layout
    _stub(fl, monkeypatch, visible = 2)
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising = False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    ran = tmp / "ran.json"
    monkeypatch.setenv("FAKE_RAN", str(ran))
    assert fl.main(str(launcher), SCRIPT_NAME, []) == 0
    assert json.loads(ran.read_text())["has_ipython"] is True


def test_redaction(fl):
    token = "hf_" + "Q" * 30
    text = fl._redact(
        f"pip install --index-url https://u:{token}@pypi.example/simple --extra-index-url=https://x/y "
        f"-i https://z/w pkg; curl -H 'Authorization: Bearer abc' https://user:pw@host/a; "
        f"HF_TOKEN={token} echo {token}"
    )
    for secret in (token, "u:", "user:pw", "Bearer abc", "https://x/y", "https://z/w"):
        assert secret not in text, (secret, text)
    assert "pip install" in text and "pkg" in text


def test_logs_name_the_unexpanded_step_with_secrets_masked(fl, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("UNSLOTH_FSDP2_RUN_INSTALLS", "0")
    token = "hf_" + "Z" * 30  # a local the command expands, as {token} would in a notebook
    shell = fl.FSDP2Shell()
    shell.system("pip install --index-url https://me:{token}@pypi.example/simple somepkg")
    err = capfd.readouterr().err
    assert "skipped install step" in err and "somepkg" in err
    assert token not in err


def test_windows_needs_a_real_bash(fl, monkeypatch):
    monkeypatch.setattr(fl, "_IS_WINDOWS", True)
    monkeypatch.setattr(fl.shutil, "which", lambda name: r"C:\Windows\System32\bash.exe")
    with pytest.raises(RuntimeError, match = "Git Bash"):
        fl.FSDP2Shell._bash()
    monkeypatch.setattr(fl.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match = "no bash"):
        fl.FSDP2Shell._bash()
    monkeypatch.setattr(fl.shutil, "which", lambda name: r"C:\Program Files\Git\bin\bash.exe")
    assert fl.FSDP2Shell._bash().endswith("bash.exe")


# ---------------------------------------------------------------------------------------
# Launch argv corners and single-process scripts.
# ---------------------------------------------------------------------------------------
def test_double_separator_keeps_a_literal_separator(fl, layout, monkeypatch):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 2)
    assert fl._parse(["--", "--"]) == ({"nproc": None, "dry_run": False, "help": False}, ["--"])
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, ["--", "--"])
    assert calls[0][1] == _expected(launcher, 2, "--", "--")


def test_dry_run_names_the_adapter(fl, layout, monkeypatch, capsys):
    _, launcher, _ = layout
    _stub(fl, monkeypatch, visible = 2)
    fl.main(str(launcher), SCRIPT_NAME, ["--dry-run"])
    assert f"get_ipython adapter: {LAUNCH_PATH}" in capsys.readouterr().out


def test_single_process_script_runs_alone(fl, layout, monkeypatch, capfd):
    tmp, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 4)
    ran = tmp / "ran.json"
    monkeypatch.setenv("FAKE_RAN", str(ran))
    assert fl.main(str(launcher), SCRIPT_NAME, [], single_process = True, reason = "starts a server") == 0
    assert calls == [] and ran.exists()
    assert "starts a server" in capfd.readouterr().err


def test_single_process_script_refuses_more_processes_unless_forced(fl, layout, monkeypatch, capsys):
    _, launcher, _ = layout
    calls = _stub(fl, monkeypatch, visible = 4)
    with pytest.raises(SystemExit) as raised:
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node", "2"], single_process = True, reason = "port 8000")
    assert raised.value.code == 2 and calls == []
    err = capsys.readouterr().err
    assert "port 8000" in err and "UNSLOTH_FSDP2_FORCE=1" in err
    monkeypatch.setenv("UNSLOTH_FSDP2_FORCE", "1")
    with pytest.raises(_Exec):
        fl.main(str(launcher), SCRIPT_NAME, ["--nproc-per-node", "2"], single_process = True, reason = "port 8000")
    assert calls[0][1] == _expected(launcher, 2)


def test_single_process_script_refuses_a_multi_rank_worker(fl, layout, monkeypatch, capsys):
    _, launcher, _ = layout
    _stub(fl, monkeypatch, visible = 2)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        fl.main(str(launcher), SCRIPT_NAME, [], single_process = True, reason = "shared temp files")
    assert "shared temp files" in capsys.readouterr().err


_TRITON_KERNELS = ("git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356"
                   "#subdirectory=python/triton_kernels")


@pytest.mark.parametrize("command, skip", [
    # The repository is triton, but the subdirectory is another distribution: an
    # installed triton says nothing about it.
    (f"uv pip install -qqq {_TRITON_KERNELS}", False),
    ("pip install git+https://github.com/unslothai/unsloth.git@main#subdirectory=sub", False),
    # A repository URL with no explicit name is only checked for the known repos.
    ("pip install git+https://github.com/someone/triton.git", False),
    ("pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git", True),
    ("pip install --no-deps git+https://github.com/unslothai/unsloth.git@main", True),
    # Explicit names still count.
    ('pip install "triton @ git+https://github.com/someone/fork.git"', True),
    ("pip install git+https://github.com/someone/fork.git#egg=triton", True),
])
def test_url_requirements_need_an_explicit_name(fl, installed, command, skip):
    add_dist, _ = installed
    for name in ("triton", "unsloth", "unsloth_zoo"):
        add_dist(name, "3.4.0")
    assert _skip(fl, command) is skip


def test_triton_kernels_line_is_still_in_the_script():
    text = (REPO_ROOT / "python_scripts" / "gpt_oss_(20B)_Reinforcement_Learning_2048_Game.py").read_text()
    assert _TRITON_KERNELS in text
