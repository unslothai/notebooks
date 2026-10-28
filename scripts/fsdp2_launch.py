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

"""Run a ``python_scripts/<name>.py`` notebook mirror on every visible GPU with Unsloth's native FSDP2.

The generator copies this file to ``fsdp2_scripts/unsloth_fsdp2_launch.py``; each
``fsdp2_scripts/<name>.py`` is a thin launcher that calls ``main(__file__, "<name>.py")``.
It can also be used directly on any script:

    python scripts/fsdp2_launch.py "python_scripts/Qwen3_(4B)-GRPO.py" [script args]

Outside torchrun (no ``LOCAL_RANK`` / ``TORCHELASTIC_RUN_ID``) only three flags are read,
anywhere before a ``--`` separator; everything else goes to the script:

    --nproc-per-node N   processes to start (default: UNSLOTH_NPROC_PER_NODE, else the
                         number of visible GPUs). A count above the visible GPUs is
                         refused. With no GPU visible the count is taken as given, which
                         is how the CPU (gloo) tests drive two ranks.
    --dry-run            print the command and environment, then exit
    --help               this text

With 2+ processes this process is replaced by
``python -m torch.distributed.run --standalone --nproc-per-node=N <launcher> [args]``
with ``UNSLOTH_FSDP2=1`` unless the caller already set it (``UNSLOTH_FSDP2=0`` gives plain
DDP). With 0 or 1 the script runs here, single process.

Inside torchrun the target script runs through ``runpy`` with a small ``get_ipython``
stand-in, because the notebook mirrors still carry ``!cmd`` / ``%%bash`` / ``%%file`` /
``%%capture`` / ``%env`` / ``%cd`` calls. Shell commands and file writes happen once per
node (local rank 0) while the other local ranks wait for its return code. Package
installs are skipped unless ``UNSLOTH_FSDP2_RUN_INSTALLS=1``: prepare the environment
once instead of having every training launch reinstall it.

Only the standard library is imported here, so the parent process never loads torch.
"""

import contextlib
import io
import os
import re
import runpy
import shlex
import socket
import string
import subprocess
import sys
import tempfile
import time

__all__ = ["main", "SUPPORTED_CALLS", "is_install_command"]

_LAUNCHER_FLAGS = ("--nproc-per-node", "--nproc_per_node", "--dry-run", "--help", "-h")
_WAIT_NOTE_SECONDS = 60
_POLL_SECONDS = 0.2
_PREFIX = "[unsloth fsdp2]"

# Every get_ipython() call form the adapter implements, as (method, magic name).
# tests/test_fsdp2_adapter_covers_scripts.py checks every live call in
# python_scripts/ against this table, so a notebook that starts using another magic
# fails CI here instead of failing on a multi-GPU box.
SUPPORTED_CALLS = frozenset({
    ("system", None),
    ("getoutput", None),
    ("run_cell_magic", "bash"),
    ("run_cell_magic", "file"),
    ("run_cell_magic", "writefile"),
    ("run_cell_magic", "capture"),
    ("run_line_magic", "env"),
    ("run_line_magic", "cd"),
    ("run_line_magic", "capture"),
})

# Shell commands that change the Python / system environment. Under a multi-GPU launch
# they are skipped unless UNSLOTH_FSDP2_RUN_INSTALLS=1, then run on local rank 0 only.
_INSTALL_PATTERNS = (
    re.compile(r"(?<![\w.-])(?:pip3?|uv\s+pip|python[\d.]*\s+-m\s+pip)\b[^\n;&|]*?\b(?:install|uninstall)\b"),
    re.compile(r"(?<![\w.-])apt(?:-get)?\b[^\n;&|]*?\binstall\b"),
    re.compile(r"ollama\.com/install\.sh"),
    re.compile(r"(?<![\w.-])unsloth\s+install-kernels\b"),
)


def is_install_command(command):
    """Whether a shell command (one line or a whole ``%%bash`` body) installs packages."""
    return any(pattern.search(command) for pattern in _INSTALL_PATTERNS)


def _log(message):
    print(f"{_PREFIX} {message}", file = sys.__stderr__, flush = True)


def _short(command, limit = 160):
    command = " ".join(command.split())
    return command if len(command) <= limit else command[: limit - 3] + "..."


# ---------------------------------------------------------------------------------------
# IPython-style expansion: {expr} and $name, from the calling frame. Same rules as
# IPython's DollarFormatter: $$ is a literal $, $name inside single quotes is left
# alone, and any failure leaves the whole string untouched.
# ---------------------------------------------------------------------------------------
class _DollarFormatter(string.Formatter):
    _dollar = re.compile(r"(.*?)\$(\$?[\w\.]+)(?=([^']*'[^']*')*[^']*$)")

    def parse(self, format_string):
        for literal, field, spec, conversion in string.Formatter.parse(self, format_string):
            continue_from = 0
            text = ""
            for match in self._dollar.finditer(literal):
                before, name = match.group(1, 2)
                if name.startswith("$"):
                    text += before + name
                else:
                    yield (text + before, name, "", None)
                    text = ""
                continue_from = match.end()
            yield (text + literal[continue_from:], field, spec, conversion)

    def vformat(self, format_string, args, kwargs):
        pieces = []
        for literal, field, spec, conversion in self.parse(format_string):
            if literal:
                pieces.append(literal)
            if field is not None:
                if spec:
                    field = ":".join([field, spec])
                # The expression comes from the script being run, exactly as IPython
                # evaluates it for `!cmd {expr}` in the notebook.
                value = eval(field, dict(kwargs))
                pieces.append(self.format_field(self.convert_field(value, conversion), ""))
        return "".join(pieces)


_FORMATTER = _DollarFormatter()


def _expand(command, frame):
    namespace = dict(frame.f_globals)
    namespace.update(frame.f_locals)
    try:
        return _FORMATTER.vformat(command, (), namespace)
    except Exception:
        return command


# ---------------------------------------------------------------------------------------
# Cell transform for %%capture bodies.
# ---------------------------------------------------------------------------------------
_ASSIGN_BANG = re.compile(r"^(\s*)([A-Za-z_][\w\.\[\]'\", ]*?)\s*=\s*!(.*)$")


def _minimal_transform(body):
    """Rewrite ``!cmd``, ``x = !cmd``, ``%magic args`` and a leading ``%%magic`` the way
    IPython's TransformerManager does, for when IPython is not installed."""
    lines = body.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        head = lines[0].lstrip()[2:]
        name, _, line = head.partition(" ")
        rest = "\n".join(lines[1:]) + ("\n" if body.endswith("\n") else "")
        return f"get_ipython().run_cell_magic({name!r}, {line.strip()!r}, {rest!r})\n"
    out = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")) or _ASSIGN_BANG.match(line):
            # IPython joins backslash-continued escaped lines the same way
            # (inputtransformer2.assemble_continued_line).
            parts = [line]
            while parts[-1].rstrip().endswith("\\") and index + 1 < len(lines):
                index += 1
                parts.append(lines[index])
            line = " ".join([p.rstrip()[:-1] for p in parts[:-1]] + [parts[-1].rstrip()])
        indent = line[: len(line) - len(line.lstrip())]
        stripped = line.lstrip()
        assign = _ASSIGN_BANG.match(line)
        if stripped.startswith("!!"):
            out.append(f"{indent}get_ipython().getoutput({stripped[2:]!r})")
        elif stripped.startswith("!"):
            out.append(f"{indent}get_ipython().system({stripped[1:]!r})")
        elif stripped.startswith("%") and not stripped.startswith("%%"):
            name, _, argument = stripped[1:].partition(" ")
            out.append(f"{indent}get_ipython().run_line_magic({name!r}, {argument.strip()!r})")
        elif assign:
            target, command = assign.group(2), assign.group(3)
            out.append(f"{assign.group(1)}{target} = get_ipython().getoutput({command!r})")
        else:
            out.append(line)
        index += 1
    return "\n".join(out) + "\n"


def _transform_cell(body):
    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except Exception:
        return _minimal_transform(body)
    return TransformerManager().transform_cell(body)


# ---------------------------------------------------------------------------------------
# The get_ipython() stand-in.
# ---------------------------------------------------------------------------------------
class _Captured:
    def __init__(self, stdout, stderr):
        self.stdout = stdout
        self.stderr = stderr

    def show(self):
        sys.stdout.write(self.stdout)
        sys.stderr.write(self.stderr)

    __call__ = show


class FSDP2Shell:
    """The few ``get_ipython()`` operations the notebook mirrors use, made safe for N ranks.

    Shell commands and file writes run on local rank 0. The other local ranks wait for
    ``<tmp>/unsloth_fsdp2_<run>/<n>.done`` (a file, since there may be no process group
    yet), which holds rank 0's return code. ``n`` counts every system / bash / file call,
    skipped ones included, so it agrees across ranks that follow the same control flow.
    ``%env`` / ``%cd`` / ``getoutput`` run on every rank.
    """

    def __init__(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
        self.distributed = "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ
        self.run_installs = os.environ.get("UNSLOTH_FSDP2_RUN_INSTALLS", "0") == "1"
        self.counter = 0
        self._capture_depth = 0
        self._marker_dir = None
        self._skip_noted = False

    # -- synchronisation -------------------------------------------------------------
    def _markers(self):
        if self._marker_dir is None:
            # torchrun's default rendezvous id is the literal "none", so the id alone can
            # collide with an earlier run's markers. Every local rank of one launch shares
            # the elastic agent as parent, plus the restart count after an elastic retry.
            run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "local"
            parts = [run_id]
            if "TORCHELASTIC_RUN_ID" in os.environ:
                parts.append(str(os.getppid()))
            parts.append("r" + os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
            parts.append(socket.gethostname())
            name = "_".join(re.sub(r"[^\w.-]", "_", part) for part in parts)
            self._marker_dir = os.path.join(tempfile.gettempdir(), f"unsloth_fsdp2_{name}")
            os.makedirs(self._marker_dir, exist_ok = True)
        return self._marker_dir

    def _once(self, what, action):
        """Run ``action()`` (returns an int code) on local rank 0; other local ranks wait
        for its marker and return the same code."""
        self.counter += 1
        if not self.distributed:
            return action()
        marker = os.path.join(self._markers(), f"{self.counter}.done")
        if self.local_rank == 0:
            code = 1
            try:
                code = action()
            finally:
                partial = f"{marker}.{os.getpid()}.tmp"
                with open(partial, "w", encoding = "utf-8") as handle:
                    handle.write(str(code))
                os.replace(partial, marker)
            return code
        started = time.monotonic()
        next_note = started + _WAIT_NOTE_SECONDS
        while not os.path.exists(marker):
            time.sleep(_POLL_SECONDS)
            if time.monotonic() >= next_note:
                waited = int(time.monotonic() - started)
                _log(f"local rank {self.local_rank} still waiting ({waited}s) for local rank 0 "
                     f"to finish step {self.counter}: {_short(what)}")
                next_note += _WAIT_NOTE_SECONDS
        with open(marker, encoding = "utf-8") as handle:
            return int(handle.read().strip() or 1)

    def _skip_install(self, command):
        if self.local_rank == 0:
            _log(f"skipped install step: {_short(command)}")
            if not self._skip_noted:
                _log("prepare the environment once, or set UNSLOTH_FSDP2_RUN_INSTALLS=1 to run "
                     "install steps (local rank 0 runs them, the other ranks wait)")
                self._skip_noted = True

    # -- shell -----------------------------------------------------------------------
    @staticmethod
    def _bash():
        return "/bin/bash" if os.path.exists("/bin/bash") else "bash"

    def _shell(self, command):
        sys.stdout.flush()
        sys.stderr.flush()
        argv = [self._bash(), "-c", command]
        if self._capture_depth:
            result = subprocess.run(argv, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
            sys.stdout.write(result.stdout.decode("utf-8", "replace"))
            return result.returncode
        return subprocess.run(argv).returncode

    def _system(self, command):
        if is_install_command(command) and not self.run_installs:
            self.counter += 1
            self._skip_install(command)
            return 0
        return self._once(command, lambda: self._shell(command))

    def system(self, cmd):
        """``!cmd``: like IPython, a failing command does not raise; ``_exit_code`` holds it."""
        frame = sys._getframe(1)
        code = self._system(_expand(cmd, frame))
        frame.f_globals["_exit_code"] = code
        return None

    def getoutput(self, cmd):
        """``x = !cmd``: runs on every rank, returns the output lines."""
        command = _expand(cmd, sys._getframe(1))
        result = subprocess.run(
            [self._bash(), "-c", command], stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
        )
        return result.stdout.decode("utf-8", "replace").splitlines()

    # -- magics ----------------------------------------------------------------------
    def _unsupported(self, kind, name, frame):
        where = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        raise NotImplementedError(
            f"get_ipython().{kind}({name!r}) is not supported by the fsdp2 launcher ({where}). "
            f"Supported: {sorted(m for k, m in SUPPORTED_CALLS if k == kind)}"
        )

    def run_line_magic(self, magic_name, line, _stack_depth = 1):
        frame = sys._getframe(_stack_depth)
        line = _expand(line, frame)
        if magic_name == "env":
            return self._env(line)
        if magic_name == "cd":
            target = os.path.expanduser(line.strip().strip("'\"") or "~")
            os.chdir(target)
            print(os.getcwd())
            return None
        if magic_name == "capture":
            return None
        self._unsupported("run_line_magic", magic_name, frame)

    @staticmethod
    def _env(line):
        # Mirrors IPython's %env / %set_env, including keeping a trailing comment in the
        # value, so the variable a script sees is the one it sees in the notebook.
        if not line.strip():
            return {
                key: "<hidden>" if any(s in key.lower() for s in ("key", "token", "secret")) else value
                for key, value in os.environ.items()
            }
        split = "=" if "=" in line else " "
        bits = line.split(split, 1)
        if len(bits) == 1:
            key = line.strip()
            if key in os.environ:
                return os.environ[key]
            raise KeyError(f"Environment does not have key: {key}")
        var, value = bits[0].strip(), bits[1].strip()
        if re.match(r".*\s.*", var):
            raise ValueError(f"refusing to set env var with whitespace: {var!r}")
        os.environ[var] = value
        print(f"env: {var}={value}")
        return None

    def run_cell_magic(self, magic_name, line, cell):
        frame = sys._getframe(1)
        line = _expand(line, frame)
        if magic_name == "bash":
            return self._bash_cell(cell)
        if magic_name in ("file", "writefile"):
            return self._write_file(line, cell)
        if magic_name == "capture":
            return self._capture(line, cell, frame)
        self._unsupported("run_cell_magic", magic_name, frame)

    def _bash_cell(self, cell):
        # %%bash is not expanded by IPython; it raises on a non-zero exit, on every rank.
        code = self._system(cell)
        if code:
            raise subprocess.CalledProcessError(code, cell)
        return None

    def _write_file(self, line, cell):
        args = shlex.split(line)
        append = any(a in ("-a", "--append") for a in args)
        paths = [a for a in args if a not in ("-a", "--append")]
        if len(paths) != 1:
            raise NotImplementedError(f"%%file / %%writefile expects one path, got {line!r}")
        path = os.path.expanduser(paths[0])

        def write():
            exists = os.path.exists(path)
            if append:
                print(f"Appending to {path}")
            else:
                print(f"{'Overwriting' if exists else 'Writing'} {path}")
            with open(path, "a" if append else "w", encoding = "utf-8") as handle:
                handle.write(cell)
            return 0

        self._once(f"write {path}", write)
        return None

    def _capture(self, line, cell, frame):
        args = line.split()
        keep_stdout = "--no-stdout" not in args
        keep_stderr = "--no-stderr" not in args
        names = [a for a in args if not a.startswith("-")]
        out, err = io.StringIO(), io.StringIO()
        code = _transform_cell(cell)
        self._capture_depth += 1
        try:
            with contextlib.ExitStack() as stack:
                if keep_stdout:
                    stack.enter_context(contextlib.redirect_stdout(out))
                if keep_stderr:
                    stack.enter_context(contextlib.redirect_stderr(err))
                _exec_cell(code, frame.f_globals)
        finally:
            self._capture_depth -= 1
        if names:
            frame.f_globals[names[0]] = _Captured(out.getvalue(), err.getvalue())
        return None


def _exec_cell(code, namespace):
    # The cell body is the script's own %%capture cell, already part of the program
    # being run; this is the same exec IPython performs for it in the notebook.
    exec(code, namespace)


_SHELL = None


def get_ipython():
    global _SHELL
    if _SHELL is None:
        _SHELL = FSDP2Shell()
    return _SHELL


# ---------------------------------------------------------------------------------------
# Launcher.
# ---------------------------------------------------------------------------------------
def _usage(launcher, script_name):
    shown = launcher if script_name else f"{launcher} <script.py>"
    return f"usage: python {shown} [--nproc-per-node N] [--dry-run] [--help] [script args]\n\n{__doc__}"


def _parse(argv):
    """Pull the launcher flags out of argv; everything else is the script's."""
    options = {"nproc": None, "dry_run": False, "help": False}
    rest = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            rest.extend(argv[index + 1:])
            break
        if arg in ("--nproc-per-node", "--nproc_per_node"):
            if index + 1 >= len(argv):
                raise ValueError(f"{arg} needs a value")
            options["nproc"] = argv[index + 1]
            index += 2
            continue
        if arg.startswith(("--nproc-per-node=", "--nproc_per_node=")):
            options["nproc"] = arg.split("=", 1)[1]
        elif arg == "--dry-run":
            options["dry_run"] = True
        elif arg in ("--help", "-h"):
            options["help"] = True
        else:
            rest.append(arg)
        index += 1
    return options, rest


def _count_visible_gpus():
    """GPUs torch can see, from a short-lived child so this process never imports torch.
    Honours CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES; 0 when torch is missing or fails."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            stdout = subprocess.PIPE, stderr = subprocess.DEVNULL, timeout = 300,
        )
        return int(result.stdout.decode().strip().splitlines()[-1])
    except Exception:
        return 0


def _fail(message):
    print(f"{_PREFIX} error: {message}", file = sys.stderr, flush = True)
    raise SystemExit(2)


def _resolve_target(launcher, script_name, rest):
    if script_name is None:
        if not rest:
            _fail("no script given: python fsdp2_launch.py <script.py> [args]")
        target, rest = rest[0], rest[1:]
    else:
        target = os.path.join(os.path.dirname(launcher), os.pardir, "python_scripts", script_name)
    target = os.path.normpath(os.path.abspath(target))
    if not os.path.isfile(target):
        _fail(f"target script not found: {target}")
    return target, rest


def _relaunch_args(launcher, script_name, target, rest):
    payload = list(rest) if script_name else [target, *rest]
    # A script argument that looks like a launcher flag would be eaten by the worker's
    # own parse, so keep everything behind a separator in that case.
    if any(a == "--" or a.split("=", 1)[0] in _LAUNCHER_FLAGS for a in rest):
        payload = ["--", *payload]
    return [launcher, *payload]


def _run_worker(launcher, target, rest, set_fsdp2):
    if set_fsdp2:
        os.environ.setdefault("UNSLOTH_FSDP2", "1")
    sys.argv = [target, *rest]
    # Look like `python target.py`: the target's folder first on sys.path, and the
    # launcher folder (put there by `python launcher.py`) gone.
    drop = {os.path.dirname(os.path.abspath(__file__)), os.path.dirname(launcher)}
    sys.path[:] = [p for p in sys.path if not (p and os.path.abspath(p) in drop)]
    sys.path.insert(0, os.path.dirname(target))
    init_globals = {}
    import builtins
    if not hasattr(builtins, "get_ipython"):
        # A real IPython session keeps its own get_ipython.
        init_globals["get_ipython"] = get_ipython
    runpy.run_path(target, init_globals = init_globals, run_name = "__main__")


def main(launcher_path, script_name, argv = None):
    """Launch ``python_scripts/<script_name>`` next to ``launcher_path`` with FSDP2.

    ``script_name=None`` is the generic form: the first script argument is the target.
    """
    launcher = os.path.abspath(launcher_path)
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        options, rest = _parse(argv)
    except ValueError as error:
        _fail(str(error))
    if options["help"]:
        print(_usage(launcher, script_name))
        return 0
    target, rest = _resolve_target(launcher, script_name, rest)

    if "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ:
        # Already a torchrun worker (ours or the user's own torchrun): never launch again.
        if options["nproc"] is not None and os.environ.get("LOCAL_RANK", "0") == "0":
            _log("--nproc-per-node is ignored inside torchrun")
        if options["dry_run"]:
            print(f"would run {target} {shlex.join(rest)} in this torchrun worker".rstrip())
            return 0
        _run_worker(launcher, target, rest, set_fsdp2 = True)
        return 0

    requested = options["nproc"]
    source = "--nproc-per-node"
    if requested is None and os.environ.get("UNSLOTH_NPROC_PER_NODE", "").strip():
        requested, source = os.environ["UNSLOTH_NPROC_PER_NODE"].strip(), "UNSLOTH_NPROC_PER_NODE"
    if requested is not None:
        try:
            requested = int(requested)
        except ValueError:
            _fail(f"{source} must be an integer, got {requested!r}")
        if requested < 0:
            _fail(f"{source} must be >= 0, got {requested}")
    visible = _count_visible_gpus()
    if requested is None:
        nproc = visible
    else:
        nproc = requested
        if visible > 0 and requested > visible:
            _fail(f"{source}={requested} but only {visible} GPU(s) are visible "
                  f"(check CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES)")

    if nproc >= 2:
        fsdp2 = os.environ.get("UNSLOTH_FSDP2", "1")
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={nproc}", *_relaunch_args(launcher, script_name, target, rest),
        ]
        if visible == 0:
            _log(f"no GPU visible; starting {nproc} processes as requested")
        if fsdp2 == "0":
            _log("UNSLOTH_FSDP2=0: launching plain DDP torchrun without FSDP2")
        if options["dry_run"]:
            print(f"UNSLOTH_FSDP2={fsdp2}")
            print(shlex.join(command))
            return 0
        os.environ.setdefault("UNSLOTH_FSDP2", "1")
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, command)
        return 0  # only reached when os.execv is replaced, as in the tests

    if nproc == 1:
        note = "1 GPU visible: running single-process, FSDP2 needs 2+ GPUs"
    else:
        note = "no GPU visible: running single-process"
    if options["dry_run"]:
        print(note)
        print(shlex.join([sys.executable, target, *rest]))
        return 0
    _log(note)
    _run_worker(launcher, target, rest, set_fsdp2 = False)
    return 0


if __name__ == "__main__":
    main(__file__, None)
