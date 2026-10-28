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

Outside a launcher only three flags are read, anywhere before a ``--`` separator;
everything else goes to the script:

    --nproc-per-node N   processes to start (default: UNSLOTH_NPROC_PER_NODE, else the
                         number of visible GPUs). A count above the visible GPUs is
                         refused. With no GPU visible the count is taken as given, which
                         is how the CPU (gloo) tests drive two ranks.
    --dry-run            print the command and environment, then exit
    --help               this text

With 2+ processes this process is replaced by
``python -m torch.distributed.run --standalone --nproc-per-node=N <launcher> [args]``
with ``UNSLOTH_FSDP2=1`` unless the caller already set it (``UNSLOTH_FSDP2=0`` gives plain
DDP). Under IPython / Jupyter (``%run``) torchrun runs as a child process instead. With
0 or 1 the script runs here, single process. Started as one of several SLURM / MPI tasks
without ``LOCAL_RANK`` / ``WORLD_SIZE``, it refuses instead of starting one torchrun per
task.

A process is a worker (it runs the script and never launches again) under torchrun
(``TORCHELASTIC_RUN_ID``) or when ``LOCAL_RANK`` comes with ``WORLD_SIZE`` /
``LOCAL_WORLD_SIZE``. A ``LOCAL_RANK`` left over in the shell on its own is ignored.

Inside a worker the target script runs through ``runpy`` with a small ``get_ipython``
stand-in, because the notebook mirrors still carry ``!cmd`` / ``%%bash`` / ``%%file`` /
``%%capture`` / ``%env`` / ``%cd`` calls. With 2+ local ranks, shell commands and file
writes happen once per node (local rank 0) while the other local ranks wait for its
return code (at most ``UNSLOTH_FSDP2_WAIT_TIMEOUT`` seconds, default 6 hours).

Package installs (``pip`` / ``uv pip`` / ``python -m pip`` install, ``apt-get install``,
the ollama installer, ``unsloth install-kernels``) are handled per command:

    UNSLOTH_FSDP2_RUN_INSTALLS unset   skip it when everything it asks for is already
                                       there (every requested distribution installed at
                                       a matching version, or the apt / ollama binary on
                                       PATH), else run it once per node. Installs into
                                       another environment (``--python``, ``--target``,
                                       ``--prefix``, ``--root``, ``uv venv``) always run.
    UNSLOTH_FSDP2_RUN_INSTALLS=1       always run, once per node
    UNSLOTH_FSDP2_RUN_INSTALLS=0       always skip

Every skipped install prints one line. The single-process path uses the same rules.

Only the standard library is imported here, so the parent process never loads torch.
"""

import atexit
import contextlib
import io
import os
import re
import runpy
import shlex
import shutil
import socket
import string
import subprocess
import sys
import tempfile
import time
import uuid

__all__ = ["main", "SUPPORTED_CALLS", "is_install_command", "install_decision"]

_LAUNCHER_FLAGS = ("--nproc-per-node", "--nproc_per_node", "--dry-run", "--help", "-h")
_WAIT_NOTE_SECONDS = 60
_POLL_SECONDS = 0.2
_DEFAULT_WAIT_TIMEOUT = 6 * 3600
# How long local rank 0 waits at exit for the other local ranks to read the last marker
# before it removes the marker folder. Past this the folder is left behind; its name is
# unique to the launch, so it is never reused.
_CLEANUP_WAIT_SECONDS = 120
_PREFIX = "[unsloth fsdp2]"
# Set by the parent before it starts torchrun, so the marker folder is unique per launch.
_RUN_ID_ENV = "UNSLOTH_FSDP2_RUN_ID"
# Set in a worker for multiprocessing spawn / forkserver children (see _patch_spawn).
_MP_TARGET_ENV = "UNSLOTH_FSDP2_MP_TARGET"

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

# Shell commands that change the Python / system environment. install_decision() says
# whether one can be skipped; when it runs, it runs on local rank 0 only.
_INSTALL_PATTERNS = (
    re.compile(r"(?<![\w.-])(?:pip3?|uv\s+pip|python[\d.]*\s+-m\s+pip)\b[^\n;&|]*?\b(?:install|uninstall)\b"),
    re.compile(r"(?<![\w.-])apt(?:-get)?\b[^\n;&|]*?\binstall\b"),
    re.compile(r"ollama\.com/install\.sh"),
    re.compile(r"(?<![\w.-])unsloth\s+install-kernels\b"),
)


def is_install_command(command):
    """Whether a shell command (one line or a whole ``%%bash`` body) installs packages."""
    return any(pattern.search(command) for pattern in _INSTALL_PATTERNS)


# A command that builds another environment: whatever it installs is not for this one.
_OTHER_ENV = re.compile(r"(?<![\w.-])(?:uv\s+venv|virtualenv|python[\d.]*\s+-m\s+(?:venv|virtualenv))\b")
# A shell comment starts at a word boundary, so `...#egg=name` in a URL stays.
_COMMENT = re.compile(r"(?<!\S)#.*$")
_PIP_EXE = re.compile(r"^pip[\d.]*$")
_PYTHON_EXE = re.compile(r"^python[\d.]*$")
# pip / uv pip options that take a value as the next argument.
_PIP_VALUE_FLAGS = frozenset({
    "-c", "--constraint", "-i", "--index-url", "--extra-index-url", "--index",
    "--default-index", "-f", "--find-links", "--torch-backend", "-C", "--config-settings",
    "--no-binary", "--only-binary", "--platform", "--python-platform", "--python-version",
    "--implementation", "--abi", "--progress-bar", "--log", "--cache-dir", "--src",
    "--upgrade-strategy", "--index-strategy", "--trusted-host", "--proxy", "--retries",
    "--timeout", "--exists-action", "--keyring-provider", "--resolution", "--prerelease",
    "--exclude-newer", "--link-mode", "--overrides", "--build-constraint", "--report",
    "--global-option", "--root-user-action", "--group", "--extra", "-P", "--upgrade-package",
})
# Options that send the install to another interpreter or folder: never skipped.
_PIP_OTHER_ENV_FLAGS = frozenset({"--python", "-p", "--target", "-t", "--prefix", "--root"})
# Options whose effect cannot be checked from the argument list: never skipped.
_PIP_UNCHECKABLE_FLAGS = frozenset({"-r", "--requirement", "-e", "--editable"})


def _normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_versions():
    import importlib
    import importlib.metadata
    importlib.invalidate_caches()
    versions = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
        if name:
            versions.setdefault(_normalize(name), dist.version)
    return versions


def _name_from_location(spec):
    """Distribution name for a URL / VCS / archive requirement, or None."""
    egg = re.search(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    if egg:
        return egg.group(1)
    path = re.split(r"[?#]", spec, 1)[0].rstrip("/")
    last = path.rsplit("/", 1)[-1]
    if last.endswith(".whl"):
        return last.split("-", 1)[0]
    if "://" not in spec:
        return None  # a local path: its name is in its own metadata
    if path.startswith("git+") or last.endswith(".git") or "@" in last:
        last = last.split("@", 1)[0]
        if last.endswith(".git"):
            last = last[:-4]
        return last if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", last) else None
    return None


def _parse_requirement(spec):
    """(name, specifier or None, applies) for one requirement argument, or None when the
    name cannot be known without running pip."""
    if " @ " in spec or re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?\s*@", spec):
        return (re.split(r"[\[\s@]", spec, 1)[0], None, True)
    if "://" in spec or spec.startswith(("git+", ".", "/", "~")) or spec.endswith((".whl", ".tar.gz", ".zip")):
        name = _name_from_location(spec)
        return (name, None, True) if name else None
    try:
        from packaging.requirements import Requirement
    except Exception:
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(?:[<>=!~;]|$)", spec)
        return (match.group(1), None, True) if match else None
    try:
        requirement = Requirement(spec)
    except Exception:
        return None
    applies = True
    if requirement.marker is not None:
        try:
            applies = requirement.marker.evaluate()
        except Exception:
            applies = True
    return (requirement.name, requirement.specifier or None, applies)


def _version_ok(installed, specifier):
    if specifier is None:
        return True
    try:
        return specifier.contains(installed, prereleases = True)
    except Exception:
        return False


def _pip_args(tokens):
    """Arguments after ``pip`` / ``uv pip`` / ``python -m pip``, or None."""
    if not tokens:
        return None
    head = os.path.basename(tokens[0])
    if _PIP_EXE.match(head):
        return tokens[1:]
    if head == "uv" and tokens[1:2] == ["pip"]:
        return tokens[2:]
    if _PYTHON_EXE.match(head) and tokens[1:3] == ["-m", "pip"]:
        return tokens[3:]
    return None


def _pip_decision(args, versions):
    """(satisfied, detail) for one pip / uv pip command, or None if it is not an
    install / uninstall."""
    words = [a for a in args if not a.startswith("-")]
    if not words or words[0] not in ("install", "uninstall"):
        return None
    action = words[0]
    rest = args[args.index(action) + 1:]
    specs = []
    index = 0
    while index < len(rest):
        arg = rest[index]
        flag = arg.split("=", 1)[0]
        if flag in _PIP_OTHER_ENV_FLAGS:
            return (False, f"installs into another environment ({flag})")
        if flag in _PIP_UNCHECKABLE_FLAGS:
            return (False, f"cannot check {flag} without running it")
        if arg.startswith("-"):
            if flag in _PIP_VALUE_FLAGS and "=" not in arg:
                index += 1
        else:
            specs.append(arg)
        index += 1
    if not specs:
        return (False, "no requirement to check")
    names = []
    for spec in specs:
        parsed = _parse_requirement(spec)
        if parsed is None:
            return (False, f"cannot tell what {spec!r} installs")
        name, specifier, applies = parsed
        if not applies:
            continue
        installed = versions.get(_normalize(name))
        if action == "uninstall":
            if installed is not None:
                return (False, f"{name} is installed")
        elif installed is None:
            return (False, f"{name} is not installed")
        elif not _version_ok(installed, specifier):
            return (False, f"{name} {installed} does not match {specifier}")
        names.append(name)
    verb = "already removed" if action == "uninstall" else "already installed"
    return (True, f"{verb}: {', '.join(names) if names else 'nothing applies here'}")


def _dpkg_installed(package):
    try:
        with open("/var/lib/dpkg/status", encoding = "utf-8", errors = "replace") as handle:
            text = handle.read()
    except OSError:
        return False
    for block in text.split("\n\n"):
        if re.search(rf"^Package: {re.escape(package)}$", block, re.M) and \
                re.search(r"^Status: .* installed$", block, re.M):
            return True
    return False


def _command_segments(command):
    """The simple commands in a shell line or ``%%bash`` body, each as its argument list
    with redirections removed. Quotes are respected; None when it cannot be parsed."""
    joined = command.replace("\\\n", " ")
    segments = []
    for line in joined.splitlines():
        lexer = shlex.shlex(_COMMENT.sub("", line), posix = True, punctuation_chars = True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError:
            return None
        current = []
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token and all(ch in "();<>|&" for ch in token):
                if "<" in token or ">" in token:
                    if current and current[-1].isdigit():
                        current.pop()  # the fd in `2>&1`
                    skip_next = True
                    continue
                segments.append(current)
                current = []
                continue
            current.append(token)
        segments.append(current)
    return [segment for segment in segments if segment]


def install_decision(command):
    """Whether an install command can be skipped because what it asks for is already here.

    Returns None when ``command`` installs nothing, else ``(skip, reason)``. Only the
    smart default uses this; ``UNSLOTH_FSDP2_RUN_INSTALLS`` overrides it.
    """
    if not is_install_command(command):
        return None
    if _OTHER_ENV.search(command):
        return (False, "builds another environment")
    versions = None
    details = []
    if re.search(r"ollama\.com/install\.sh", command):
        if not shutil.which("ollama"):
            return (False, "ollama is not on PATH")
        details.append("ollama on PATH")
    segments = _command_segments(command)
    if segments is None:
        return (False, "cannot parse the command")
    for tokens in segments:
        while tokens and (re.match(r"^[A-Za-z_]\w*=", tokens[0]) or tokens[0] in ("sudo", "env", "exec")):
            tokens = tokens[1:]
        if not tokens:
            continue
        head = os.path.basename(tokens[0])
        if head in ("apt", "apt-get") and "install" in tokens:
            packages = [t for t in tokens[tokens.index("install") + 1:] if not t.startswith("-")]
            missing = [p for p in packages if not shutil.which(p) and not _dpkg_installed(p)]
            if not packages or missing:
                return (False, f"not installed: {', '.join(missing) or 'unknown packages'}")
            details.append("on PATH: " + ", ".join(packages))
            continue
        if head == "unsloth" and tokens[1:2] == ["install-kernels"]:
            kernels = [t for t in tokens[2:] if not t.startswith("-")]
            if not kernels:
                return (False, "unsloth install-kernels with no names")
            versions = _installed_versions() if versions is None else versions
            missing = [k for k in kernels if _normalize(k) not in versions]
            if missing:
                return (False, f"{', '.join(missing)} not installed")
            details.append("already installed: " + ", ".join(kernels))
            continue
        args = _pip_args(tokens)
        if args is None:
            continue
        versions = _installed_versions() if versions is None else versions
        decision = _pip_decision(args, versions)
        if decision is None:
            continue
        if not decision[0]:
            return decision
        details.append(decision[1])
    if not details:
        return (False, "cannot tell what it installs")
    return (True, "; ".join(details))


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


def _env_int(*names, default):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return default


def _local_world_size():
    return _env_int("LOCAL_WORLD_SIZE", "WORLD_SIZE", default = 1)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


class FSDP2Shell:
    """The few ``get_ipython()`` operations the notebook mirrors use, made safe for N ranks.

    With 2+ local ranks, shell commands and file writes run on local rank 0. The other
    local ranks wait for ``<tmp>/unsloth_fsdp2_<launch>/<n>.done`` (a file, since there
    may be no process group yet), which holds rank 0's return code. ``n`` counts every
    system / bash / file call, skipped installs included, so it agrees across ranks that
    follow the same control flow. ``%env`` / ``%cd`` / ``getoutput`` run on every rank.
    In a multiprocessing spawn / forkserver child, which re-imports the script, shell
    steps, file writes and ``%cd`` are skipped: the parent already ran them.
    """

    def __init__(self, mp_child = False):
        self.local_rank = _env_int("LOCAL_RANK", default = 0)
        self.local_world_size = _local_world_size()
        launched = "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ
        self.distributed = launched and self.local_world_size > 1
        self.mp_child = mp_child
        self.counter = 0
        self._capture_depth = 0
        self._marker_dir = None
        self._skip_noted = False
        self._child_noted = False

    # -- synchronisation -------------------------------------------------------------
    @staticmethod
    def _wait_timeout():
        value = os.environ.get("UNSLOTH_FSDP2_WAIT_TIMEOUT", "").strip()
        try:
            return float(value) if value else float(_DEFAULT_WAIT_TIMEOUT)
        except ValueError:
            return float(_DEFAULT_WAIT_TIMEOUT)

    def _markers(self):
        if self._marker_dir is None:
            # One folder per launch. The parent launcher sets UNSLOTH_FSDP2_RUN_ID; under
            # another launcher (the user's torchrun, deepspeed, ...) every local rank of
            # one launch shares its parent process, and torchrun's default rendezvous id
            # is the literal "none", so the parent pid is always part of the name.
            run_id = os.environ.get(_RUN_ID_ENV, "").strip()
            if run_id:
                parts = [run_id]
            else:
                parts = [os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "local",
                         "p" + str(os.getppid())]
            parts.append("r" + os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))
            parts.append(socket.gethostname())
            name = "_".join(re.sub(r"[^\w.-]", "_", part) for part in parts)
            self._marker_dir = os.path.join(tempfile.gettempdir(), f"unsloth_fsdp2_{name}")
            os.makedirs(self._marker_dir, exist_ok = True)
            if self.local_rank == 0:
                atexit.register(self._cleanup_markers)
        return self._marker_dir

    def _ack(self, step):
        """A non-zero local rank records that it has read marker ``step``."""
        path = os.path.join(self._markers(), f"seen.{self.local_rank}")
        partial = f"{path}.{os.getpid()}.tmp"
        try:
            with open(partial, "w", encoding = "utf-8") as handle:
                handle.write(f"{os.getpid()} {step}")
            os.replace(partial, path)
        except OSError:
            pass

    def _cleanup_markers(self, wait_seconds = None):
        """Local rank 0 at exit: remove the marker folder once every other local rank has
        read the last marker or exited. Gives up (leaving the folder) after a while."""
        folder = self._marker_dir
        if folder is None or not os.path.isdir(folder):
            return
        wait_seconds = _CLEANUP_WAIT_SECONDS if wait_seconds is None else wait_seconds
        deadline = time.monotonic() + wait_seconds
        pending = set(range(1, self.local_world_size))
        while pending:
            for rank in sorted(pending):
                try:
                    with open(os.path.join(folder, f"seen.{rank}"), encoding = "utf-8") as handle:
                        pid, step = (int(x) for x in handle.read().split())
                except (OSError, ValueError):
                    continue
                if step >= self.counter or not _pid_alive(pid):
                    pending.discard(rank)
            if not pending:
                break
            if time.monotonic() >= deadline:
                return
            time.sleep(_POLL_SECONDS)
        shutil.rmtree(folder, ignore_errors = True)

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
        timeout = self._wait_timeout()
        next_note = started + _WAIT_NOTE_SECONDS
        while not os.path.exists(marker):
            time.sleep(_POLL_SECONDS)
            now = time.monotonic()
            if now - started >= timeout:
                raise TimeoutError(
                    f"{_PREFIX} local rank {self.local_rank} waited {int(now - started)}s for local "
                    f"rank 0 to finish step {self.counter} ({_short(what)}) and gave up. Local rank 0 "
                    f"may have failed, taken another branch, or not be running; raise "
                    f"UNSLOTH_FSDP2_WAIT_TIMEOUT (seconds) if the step is just slow."
                )
            if now >= next_note:
                _log(f"local rank {self.local_rank} still waiting ({int(now - started)}s) for local rank 0 "
                     f"to finish step {self.counter}: {_short(what)}")
                next_note += _WAIT_NOTE_SECONDS
        with open(marker, encoding = "utf-8") as handle:
            code = int(handle.read().strip() or 1)
        self._ack(self.counter)
        return code

    def _install_action(self, command):
        """Run or skip one install command, on the process that runs shell steps."""
        setting = os.environ.get("UNSLOTH_FSDP2_RUN_INSTALLS", "").strip()
        if setting == "1":
            return self._shell(command)
        if setting == "0":
            self._note_skip(command, "UNSLOTH_FSDP2_RUN_INSTALLS=0")
            return 0
        skip, reason = install_decision(command)
        if skip:
            self._note_skip(command, reason)
            return 0
        _log(f"running install step ({reason}): {_short(command)}")
        return self._shell(command)

    def _note_skip(self, command, reason):
        _log(f"skipped install step ({reason}): {_short(command)}")
        if not self._skip_noted:
            _log("UNSLOTH_FSDP2_RUN_INSTALLS=1 runs every install step (once per node), "
                 "UNSLOTH_FSDP2_RUN_INSTALLS=0 skips them all")
            self._skip_noted = True

    def _child_skip(self):
        if not self._child_noted:
            _log("multiprocessing child: shell steps, file writes and %cd in the re-imported "
                 "script are skipped (the parent ran them)")
            self._child_noted = True
        return 0

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
        if self.mp_child:
            return self._child_skip()
        if is_install_command(command):
            return self._once(command, lambda: self._install_action(command))
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
            return self._cd(line)
        if magic_name == "capture":
            return None
        self._unsupported("run_line_magic", magic_name, frame)

    def _cd(self, line):
        # Like IPython's %cd: a missing folder prints the error and the script goes on.
        if self.mp_child:
            return self._child_skip()
        target = os.path.expanduser(line.strip().strip("'\"") or "~")
        try:
            os.chdir(target)
        except OSError as error:
            print(error)
            return None
        print(os.getcwd())
        return None

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

        if self.mp_child:
            self._child_skip()
            return None
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
    Honours CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES; 0 when torch is missing or fails,
    in which case the tail of the child's stderr is printed once."""
    stderr = b""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            stdout = subprocess.PIPE, stderr = subprocess.PIPE, timeout = 300,
        )
        stderr = result.stderr or b""
        if result.returncode == 0:
            return int(result.stdout.decode().strip().splitlines()[-1])
        reason = f"exit code {result.returncode}"
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
    tail = stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
    _log(f"could not count GPUs with torch ({reason}); assuming none is visible")
    for line in tail:
        _log(f"  {line}")
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


def _patch_spawn(target):
    """Let multiprocessing spawn / forkserver children re-import the script with
    ``get_ipython`` defined.

    Such a child normally re-runs ``target`` as ``__mp_main__`` through a plain
    ``runpy.run_path``, where ``get_ipython`` does not exist. Point it at this helper
    instead; run as ``__mp_main__`` (bottom of this file) it re-runs ``target`` with the
    stand-in in a child-mode shell and exposes the script's names as ``__mp_main__``'s.
    """
    import multiprocessing.spawn as mp_spawn

    original = mp_spawn.get_preparation_data
    if getattr(original, "_unsloth_fsdp2", False):
        return
    helper = os.path.normpath(os.path.abspath(__file__))
    wanted = os.path.normpath(target)

    def get_preparation_data(name):
        data = original(name)
        if os.path.normpath(data.get("init_main_from_path") or "") == wanted:
            data["init_main_from_path"] = helper
        return data

    get_preparation_data._unsloth_fsdp2 = True
    mp_spawn.get_preparation_data = get_preparation_data
    os.environ[_MP_TARGET_ENV] = wanted


def _real_ipython_active():
    ipython = sys.modules.get("IPython")
    try:
        return ipython is not None and ipython.get_ipython() is not None
    except Exception:
        return False


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
        # A real IPython session keeps its own get_ipython (and its own __main__, which
        # spawn children never re-import).
        init_globals["get_ipython"] = get_ipython
        _patch_spawn(target)
    runpy.run_path(target, init_globals = init_globals, run_name = "__main__")


def _launcher_env_problem():
    """Why starting torchrun here would start one torchrun per task, or None."""
    for name in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        if _env_int(name, default = 1) > 1:
            return f"{name}={os.environ[name].strip()}"
    if "LOCAL_RANK" not in os.environ and _env_int("WORLD_SIZE", default = 1) > 1:
        return f"WORLD_SIZE={os.environ['WORLD_SIZE'].strip()} without LOCAL_RANK"
    return None


def _worker_mode():
    """'worker' under a launcher, 'stale' for a LOCAL_RANK left in the shell, else None."""
    if "TORCHELASTIC_RUN_ID" in os.environ:
        return "worker"
    if "LOCAL_RANK" in os.environ:
        sized = any(os.environ.get(n, "").strip() for n in ("LOCAL_WORLD_SIZE", "WORLD_SIZE"))
        return "worker" if sized else "stale"
    return None


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

    mode = _worker_mode()
    if mode == "stale":
        # LOCAL_RANK alone, without WORLD_SIZE / LOCAL_WORLD_SIZE, is not a launcher.
        _log(f"LOCAL_RANK={os.environ['LOCAL_RANK']} is set without WORLD_SIZE / LOCAL_WORLD_SIZE; "
             f"treating it as left over in the shell and ignoring it")
        os.environ.pop("LOCAL_RANK")
    elif mode == "worker":
        # Already a launcher's worker (ours, the user's torchrun, deepspeed, ...): never
        # launch again. With one local process the shell steps simply run here.
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
        problem = _launcher_env_problem()
        if problem:
            _fail(f"this looks like one of several SLURM / MPI tasks ({problem}); starting torchrun "
                  f"here would start one torchrun per task. Run one task per node with torchrun "
                  f"(torchrun --nnodes ... --nproc-per-node N {shlex.quote(launcher)}), or set "
                  f"LOCAL_RANK, WORLD_SIZE and LOCAL_WORLD_SIZE for each task so it runs as a worker.")
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
        sys.stdout.flush()
        sys.stderr.flush()
        if _real_ipython_active():
            # %run from IPython / Jupyter: replacing the kernel process would kill it.
            env = dict(os.environ)
            env.setdefault("UNSLOTH_FSDP2", "1")
            env[_RUN_ID_ENV] = uuid.uuid4().hex
            code = subprocess.run(command, env = env).returncode
            if code:
                raise SystemExit(code)
            return 0
        os.environ.setdefault("UNSLOTH_FSDP2", "1")
        # Unique per launch: names the workers' marker folder (FSDP2Shell._markers).
        os.environ[_RUN_ID_ENV] = uuid.uuid4().hex
        os.execv(sys.executable, command)
        return 0  # only reached when os.execv is replaced, as in the tests

    if requested is not None:
        asked = f"{source}={requested}"
        if visible > requested:
            asked += f" ({visible} GPU(s) visible)"
        note = f"{asked}: running single-process" + (", FSDP2 needs 2+ processes" if nproc == 1 else "")
    elif nproc == 1:
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


if __name__ == "__mp_main__" and os.environ.get(_MP_TARGET_ENV):
    # A multiprocessing spawn / forkserver child of a worker (see _patch_spawn): re-run the
    # script as __mp_main__ the way multiprocessing would, with get_ipython defined.
    _SHELL = FSDP2Shell(mp_child = True)
    globals().update(runpy.run_path(
        os.environ[_MP_TARGET_ENV], init_globals = {"get_ipython": get_ipython}, run_name = "__mp_main__",
    ))

if __name__ == "__main__":
    main(__file__, None)
