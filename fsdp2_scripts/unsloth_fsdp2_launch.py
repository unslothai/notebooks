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
task, whatever the GPU count and also for a single-process script (each task would
otherwise run its own copy).

A process is a worker (it runs the script and never launches again) under torchrun
(``TORCHELASTIC_RUN_ID``) or when ``LOCAL_RANK`` comes with ``WORLD_SIZE`` /
``LOCAL_WORLD_SIZE``. A ``LOCAL_RANK`` left over in the shell on its own is ignored.

Inside a worker the target script runs through ``runpy`` with a small ``get_ipython``
stand-in, because the notebook mirrors still carry ``!cmd`` / ``%%bash`` / ``%%file`` /
``%%capture`` / ``%env`` / ``%cd`` calls. With 2+ local ranks, shell commands (``!cmd``,
``x = !cmd``, ``%%bash``) and ``%%file`` writes happen once per node (local rank 0) while
the other local ranks wait for its return code, output and any error (at most
``UNSLOTH_FSDP2_WAIT_TIMEOUT`` seconds, default 6 hours). Ranks that reach different
steps fail with an error instead of pairing them up. The handshake files live in a
private (0700) folder under the temp dir. Shell steps need bash; on Windows that means
Git Bash, not WSL's ``bash.exe``. Log lines name a step by its unexpanded text with
credentials (URL user info, index URLs, header values, tokens) masked.

A launcher generated with ``single_process=True`` belongs to a script that is not safe
on several ranks (it starts a server on a fixed port, installs or clones from Python,
or shares temp files); it runs single process with a note saying why, and more than
one process is refused unless ``UNSLOTH_FSDP2_FORCE=1``.

Package installs (``pip`` / ``uv pip`` / ``python -m pip`` install, ``apt-get install``,
the ollama installer, ``unsloth install-kernels``) are handled per command:

    UNSLOTH_FSDP2_RUN_INSTALLS unset   skip it when everything it asks for is already
                                       there (every requested distribution installed at
                                       a matching version, or the apt / ollama binary on
                                       PATH), else run it once per node. Only local
                                       rank 0 decides; the others follow it. Installs into
                                       another environment (``--python``, ``--target``,
                                       ``--prefix``, ``--root``, ``uv venv``,
                                       ``--system`` from a virtual environment, or the
                                       same through ``PIP_TARGET`` / ``PIP_PREFIX`` /
                                       ``PIP_ROOT`` / ``PIP_PYTHON`` / ``UV_PYTHON`` /
                                       ``UV_SYSTEM_PYTHON``) always run. So does an install
                                       that may land elsewhere: ``uv pip`` from a Python
                                       outside a virtual environment, a command line that
                                       sets variables or runs ``export`` / ``cd`` /
                                       ``source`` / ``sudo`` / ``env`` / ``conda``, and a
                                       pip or python named by version or other path.
    UNSLOTH_FSDP2_RUN_INSTALLS=1       always run, once per node
    UNSLOTH_FSDP2_RUN_INSTALLS=0       always skip

Every skipped install prints one line. The single-process path uses the same rules.

Shell steps use the interpreter running this launcher, the environment the install
check reads: ``python`` / ``python3`` are that interpreter, ``pip`` / ``pip3`` its
``-m pip`` and ``unsloth`` its ``unsloth`` command, whatever ``PATH`` holds, and from a
virtual environment ``VIRTUAL_ENV`` names it, so ``uv pip`` installs there too.

Only the standard library is imported here, so the parent process never loads torch.
"""

import atexit
import contextlib
import hashlib
import io
import json
import os
import re
import runpy
import shlex
import shutil
import socket
import stat
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
_IS_WINDOWS = os.name == "nt"

# Every get_ipython() call form the adapter implements, as (method, magic name);
# tests/test_fsdp2_adapter_covers_scripts.py checks python_scripts/ against it.
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


def _in_venv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


# Inherited variables that send pip / uv pip somewhere other than this interpreter, the
# same as the matching flags in _PIP_OTHER_ENV_FLAGS (pip reads any option as PIP_<NAME>).
_PIP_ENV_REDIRECTS = ("PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_PYTHON")


def _env_redirect(tool):
    """Why the environment sends this ``pip`` (``tool="pip"``) or ``uv pip`` somewhere the
    install check does not read, or None."""
    if tool == "uv":
        if not _in_venv():
            # Outside a virtual environment nothing names this interpreter for uv: it takes
            # VIRTUAL_ENV, CONDA_PREFIX or a .venv above the working directory, or for
            # --system its own pick of system interpreter.
            return "uv pip from a Python outside a virtual environment installs where uv finds one"
        target = os.environ.get("UV_PYTHON", "").strip()
        here = (os.path.abspath(sys.executable), os.path.abspath(sys.prefix))
        if target and os.path.abspath(os.path.expanduser(target)) not in here:
            return f"UV_PYTHON={target} sends uv pip to another interpreter"
        if _in_venv() and os.environ.get("UV_SYSTEM_PYTHON", "").strip().lower() in ("1", "true", "yes", "on"):
            return "UV_SYSTEM_PYTHON installs into the system Python, not this virtual environment"
        return None
    for name in _PIP_ENV_REDIRECTS:
        if os.environ.get(name, "").strip():
            return f"{name} sends pip to another environment"
    return None


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


# Repositories whose distribution name is known without running pip. Only these exact
# repos, and only without a #subdirectory=, map from the URL to a name.
_KNOWN_REPOS = {
    "github.com/unslothai/unsloth": "unsloth",
    "github.com/unslothai/unsloth-zoo": "unsloth_zoo",
    "github.com/unslothai/unsloth_zoo": "unsloth_zoo",
}


def _name_from_location(spec):
    """Distribution name for a URL / VCS / archive requirement, or None when only pip
    could tell (then the install runs). Explicit names (``#egg=``) and wheel file names
    count; a ``#subdirectory=`` or any other repository URL does not."""
    egg = re.search(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    if egg:
        return egg.group(1)
    if re.search(r"[#&]subdirectory=", spec):
        return None
    path = re.split(r"[?#]", spec, 1)[0].rstrip("/")
    last = path.rsplit("/", 1)[-1]
    if last.endswith(".whl"):
        return last.split("-", 1)[0]
    if "://" not in spec:
        return None  # a local path: its name is in its own metadata
    repo = re.sub(r"^(?:git\+)?[a-z]+://(?:[^/@]+@)?", "", path)
    repo = re.sub(r"@[^/]*$", "", repo)
    repo = re.sub(r"\.git$", "", repo).lower()
    return _KNOWN_REPOS.get(repo)


def _spec_extras(spec):
    match = re.match(r"^\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*\[([^\]]*)\]", spec)
    return {e.strip() for e in match.group(1).split(",") if e.strip()} if match else set()


def _parse_requirement(spec):
    """(name, specifier or None, applies, extras) for one requirement argument, or None
    when the name cannot be known without running pip."""
    extras = _spec_extras(spec)
    if " @ " in spec or re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?\s*@", spec):
        # `name @ url`: the name is explicit, whatever the URL holds.
        return (re.split(r"[\[\s@]", spec, 1)[0], None, True, extras)
    if "://" in spec or spec.startswith(("git+", ".", "/", "~")) or spec.endswith((".whl", ".tar.gz", ".zip")):
        name = _name_from_location(spec)
        return (name, None, True, set()) if name else None
    try:
        from packaging.requirements import Requirement
    except Exception:
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(?:[<>=!~;]|$)", spec)
        return (match.group(1), None, True, extras) if match else None
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
    return (requirement.name, requirement.specifier or None, applies, set(requirement.extras))


def _extras_problem(name, extras, versions):
    """Why ``name[extras]`` may still be missing something, or None once every dependency an
    extra adds is installed at a matching version (read from the installed metadata)."""
    if not extras:
        return None
    try:
        import importlib.metadata
        from packaging.requirements import Requirement
        requires = importlib.metadata.distribution(name).requires or []
    except Exception:
        return f"cannot check the extras {sorted(extras)} of {name}"
    declared = set()
    for line in requires:
        try:
            requirement = Requirement(line)
        except Exception:
            return f"cannot read a dependency of {name}"
        if requirement.marker is None:
            continue
        for extra in extras:
            try:
                wanted = requirement.marker.evaluate({"extra": extra})
                base = requirement.marker.evaluate({"extra": ""})
            except Exception:
                return f"cannot read a dependency of {name}"
            if wanted and not base:
                declared.add(extra)
                installed = versions.get(_normalize(requirement.name))
                if installed is None or not _version_ok(installed, requirement.specifier or None):
                    return f"{name}[{extra}] needs {requirement.name}{requirement.specifier}"
    unknown = sorted(extras - declared)
    if unknown:
        return f"{name} declares no extra {', '.join(unknown)}"
    return None


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
        if flag == "--system" and _in_venv():
            return (False, "installs into the system Python (--system), not this virtual environment")
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
        name, specifier, applies, extras = parsed
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
        else:
            problem = _extras_problem(name, extras, versions)
            if problem:
                return (False, problem)
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


_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
# Commands that change the variables, directory or user a later command runs with.
_STATE_COMMANDS = frozenset({
    "export", "unset", "set", "source", ".", "cd", "pushd", "popd", "eval", "sudo", "env",
    "exec", "conda", "mamba", "micromamba", "activate", "deactivate",
})
# The commands _SHELL_PREAMBLE pins to this interpreter.
_PINNED_COMMANDS = frozenset({"pip", "pip3", "python", "python3", "unsloth"})


def _target_doubt(segments, tokens):
    """Why the Python package install ``tokens`` (one of ``segments``) may not land in this
    interpreter, or None. Only a plain command is trusted: no variable assignments or
    environment / directory changes anywhere in the command line, a pinned command name or
    this interpreter's own path, and no pip / uv variable pointing elsewhere."""
    for segment in segments:
        if _ASSIGNMENT.match(segment[0]):
            return f"the command sets {segment[0].split('=', 1)[0]}, which can change where it installs"
        if segment[0] in _STATE_COMMANDS:
            return f"`{segment[0]}` in the command can change where it installs"
    if os.path.basename(tokens[0]) == "uv":
        return _env_redirect("uv")
    if tokens[0] not in _PINNED_COMMANDS and \
            os.path.abspath(os.path.expanduser(tokens[0])) != os.path.abspath(sys.executable):
        return f"{tokens[0]} is not this interpreter's command"
    return _env_redirect("pip")


_READ_ONLY_HEADS = {"true", ":", "tail", "head", "grep"}


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
    ollama_checked = False
    if re.search(r"ollama\.com/install\.sh", command):
        if not shutil.which("ollama"):
            return (False, "ollama is not on PATH")
        details.append("ollama on PATH")
        ollama_checked = True  # its curl | sh pipeline is the install itself
    segments = _command_segments(command)
    if segments is None:
        return (False, "cannot parse the command")
    for segment in segments:
        tokens = segment
        while tokens and (_ASSIGNMENT.match(tokens[0]) or tokens[0] in ("sudo", "env", "exec")):
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
            # It installs with the pip of the interpreter it runs on, which reads PIP_* too.
            doubt = _target_doubt(segments, segment)
            if doubt:
                return (False, doubt)
            details.append("already installed: " + ", ".join(kernels))
            continue
        args = _pip_args(tokens)
        if args is None:
            # Skipping the step skips every command in it, so it only goes when the rest is
            # read-only: apt's index refresh or a filter on the install's output.
            if head in _READ_ONLY_HEADS or (head in ("apt", "apt-get") and [t for t in tokens[1:] if not t.startswith("-")] == ["update"]) \
                    or (head == "command" and tokens[1:2] == ["-v"]) \
                    or (ollama_checked and head in ("curl", "sh", "bash")):
                continue
            return (False, f"also runs `{head}`")
        versions = _installed_versions() if versions is None else versions
        decision = _pip_decision(args, versions)
        if decision is None:
            words = [a for a in args if not a.startswith("-")]
            if words[:1] in ([], ["list"], ["show"], ["freeze"], ["check"]):
                continue
            return (False, f"also runs `{head} {words[0]}`")
        doubt = _target_doubt(segments, segment)
        if doubt:
            return (False, doubt)
        if not decision[0]:
            return decision
        details.append(decision[1])
    if not details:
        return (False, "cannot tell what it installs")
    return (True, "; ".join(details))


# install_decision() reads this interpreter's packages, so a shell step's pip must install
# into it rather than into whichever pip PATH finds first. On the command's first line so
# bash's line numbers in error messages still match the cell.
# Below the pinned interpreter (_UNSLOTH_FSDP2_PINNED set) the names fall back to PATH:
# if $UNSLOTH_FSDP2_PYTHON is a bash wrapper that runs `python3` by name, the exported
# function would otherwise call the wrapper again without end.
# ${...:-} so a step running under `set -u` (nounset) does not stop on the unset guard.
_PIN_GUARD = '[ -n "${_UNSLOTH_FSDP2_PINNED:-}" ] && { command %s "$@"; return; }; '
_SHELL_PREAMBLE = (
    'python() { ' + _PIN_GUARD % "python" + '_UNSLOTH_FSDP2_PINNED=1 "$UNSLOTH_FSDP2_PYTHON" "$@"; }; '
    'python3() { ' + _PIN_GUARD % "python3" + '_UNSLOTH_FSDP2_PINNED=1 "$UNSLOTH_FSDP2_PYTHON" "$@"; }; '
    'pip() { ' + _PIN_GUARD % "pip" + '_UNSLOTH_FSDP2_PINNED=1 "$UNSLOTH_FSDP2_PYTHON" -m pip "$@"; }; '
    'pip3() { ' + _PIN_GUARD % "pip3" + '_UNSLOTH_FSDP2_PINNED=1 "$UNSLOTH_FSDP2_PYTHON" -m pip "$@"; }; '
    # The `unsloth` console script of this interpreter, run the way the script runs it.
    'unsloth() { ' + _PIN_GUARD % "unsloth"
    + '_UNSLOTH_FSDP2_PINNED=1 "$UNSLOTH_FSDP2_PYTHON" -c "$_UNSLOTH_FSDP2_CLI" "$@"; }; '
    "export -f python python3 pip pip3 unsloth; "
)
_UNSLOTH_CLI = (
    "import sys; from importlib.metadata import entry_points; "
    "ep = next(iter(entry_points(group='console_scripts', name='unsloth')), None); "
    "ep is None and sys.exit('unsloth is not installed for ' + sys.executable); "
    "sys.argv[0] = 'unsloth'; sys.exit(ep.load()())"
)


def _shell_env():
    env = dict(os.environ)
    env["UNSLOTH_FSDP2_PYTHON"] = sys.executable
    env["_UNSLOTH_FSDP2_CLI"] = _UNSLOTH_CLI
    # A launcher started from another launcher's shell step pins its own interpreter again.
    env.pop("_UNSLOTH_FSDP2_PINNED", None)
    if _in_venv():
        env["VIRTUAL_ENV"] = sys.prefix
    return env


def _log(message):
    print(f"{_PREFIX} {message}", file = sys.__stderr__, flush = True)


# Secrets that must not reach a log line: URL userinfo, index URLs, header values and
# token-shaped strings. Messages name the unexpanded step and pass through this too.
_REDACTIONS = (
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s@'\"]+@"), r"\1***@"),
    (re.compile(r"(?i)((?:--(?:extra-)?index-url|--find-links|-[if])(?:\s+|=))(\S+)"), r"\1***"),
    (re.compile(r"(?i)((?:-H|--header)(?:\s+|=))('[^']*'|\"[^\"]*\"|\S+)"), r"\1***"),
    (re.compile(r"(?i)(authorization:\s*(?:bearer|token|basic)?\s*)\S+"), r"\1***"),
    (re.compile(r"(?i)(\b[\w-]*(?:token|secret|password|passwd|api_?key)[\w-]*\s*[=:]\s*)('[^']*'|\"[^\"]*\"|\S+)"), r"\1***"),
    (re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[abpr]-[A-Za-z0-9-]{10,})"), "***"),
)


def _redact(text):
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _short(command, limit = 160):
    command = " ".join(_redact(command).split())
    return command if len(command) <= limit else command[: limit - 3] + "..."


# IPython-style expansion ({expr}, $name) with DollarFormatter's rules: $$ is a literal
# $, $name inside single quotes is left alone, any failure leaves the string untouched.
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
    # Processes on this node: torchrun's LOCAL_WORLD_SIZE, DeepSpeed's LOCAL_SIZE, Open MPI's
    # and MPICH's local counts; the global WORLD_SIZE only when none of them is set.
    return _env_int("LOCAL_WORLD_SIZE", "LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS",
                    "WORLD_SIZE", default = 1)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


# Marker files. The folder lives in a possibly shared temp dir, so it must be ours: a
# private folder (0700, owned by this uid, not a symlink), files created exclusively
# without following symlinks, and content that is checked before it is believed.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _private_dir(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(path)
    problem = None
    if stat.S_ISLNK(info.st_mode):
        problem = "is a symlink"
    elif not stat.S_ISDIR(info.st_mode):
        problem = "is not a directory"
    elif hasattr(os, "getuid") and info.st_uid != os.getuid():
        problem = f"is owned by uid {info.st_uid}, not {os.getuid()}"
    elif not _IS_WINDOWS and info.st_mode & 0o022:
        problem = f"is group- or world-writable (mode {stat.S_IMODE(info.st_mode):o})"
    if problem:
        raise RuntimeError(
            f"{_PREFIX} refusing marker folder {path}: it {problem}. Remove it, or point "
            f"TMPDIR at a private folder."
        )
    return path


def _write_private(path, payload):
    """Write JSON ``payload`` to ``path`` atomically: an exclusive, no-follow temp file
    in the same private folder, then os.replace."""
    partial = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        json.dump(payload, handle)
    os.replace(partial, path)


def _read_private(path, kind):
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    with os.fdopen(fd, encoding = "utf-8") as handle:
        text = handle.read()
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    if not _valid_payload(payload, kind):
        raise RuntimeError(f"{_PREFIX} malformed {kind} marker {path}: {text[:200]!r}")
    return payload


def _valid_payload(payload, kind):
    if not isinstance(payload, dict) or not isinstance(payload.get("hash"), str):
        return False
    if kind == "arrival":
        return True
    if not isinstance(payload.get("code"), int) or isinstance(payload.get("code"), bool):
        return False
    if not isinstance(payload.get("skipped"), bool):
        return False
    output = payload.get("output")
    if output is not None and not (isinstance(output, list) and all(isinstance(x, str) for x in output)):
        return False
    error = payload.get("error")
    if error is not None and not (isinstance(error, dict) and isinstance(error.get("type"), str)
                                  and isinstance(error.get("message"), str)):
        return False
    return True


def _step_hash(kind, template):
    return hashlib.sha256(f"{kind}\0{template}".encode("utf-8", "replace")).hexdigest()[:16]


def _rebuild_error(error):
    """The exception local rank 0 recorded, re-raised on a peer."""
    import builtins
    cls = getattr(builtins, error["type"], None)
    message = f"{error['message']} (raised on local rank 0)"
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(message)
        except Exception:
            pass
    return RuntimeError(f"{error['type']}: {message}")


class _Divergence(RuntimeError):
    pass


class FSDP2Shell:
    """The few ``get_ipython()`` operations the notebook mirrors use, made safe for N ranks.

    With 2+ local ranks, shell commands (``!cmd``, ``x = !cmd``, ``%%bash``) and file
    writes run on local rank 0 only. Step ``n`` (counted per call on every rank) goes:
    each other local rank writes ``n.arrived.<rank>`` with a hash of the step's
    unexpanded text; rank 0 waits for all of them, checks the hashes agree, runs or
    skips the step, and writes ``n.done`` with its return code, output lines, whether
    it skipped an install, and any error. The others take all of that from the marker,
    so every rank sees rank 0's decision, output and failure. A hash mismatch means the
    ranks reached different steps and raises. Markers live in a private folder under the
    temp dir (``<tmp>/unsloth_fsdp2_<launch>``) because there may be no process group
    yet. ``%env`` / ``%cd`` run on every rank. In a multiprocessing spawn / forkserver
    child, which re-imports the script, shell steps, file writes and ``%cd`` are
    skipped: the parent already ran them.
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
            folder = _private_dir(os.path.join(tempfile.gettempdir(), f"unsloth_fsdp2_{name}"))
            self._marker_dir = folder
            if self.local_rank == 0:
                atexit.register(self._cleanup_markers)
        return self._marker_dir

    def _ack(self, step):
        """A non-zero local rank records that it has read marker ``step``."""
        try:
            _write_private(os.path.join(self._markers(), f"seen.{self.local_rank}"),
                           {"hash": "", "pid": os.getpid(), "step": step})
        except OSError:
            pass

    def _cleanup_markers(self, wait_seconds = None):
        """Local rank 0 at exit: remove the marker folder once every other local rank has
        read the last marker or exited. Gives up (leaving the folder) after a while."""
        folder = self._marker_dir
        if folder is None or not os.path.isdir(folder) or os.path.islink(folder):
            return
        wait_seconds = _CLEANUP_WAIT_SECONDS if wait_seconds is None else wait_seconds
        deadline = time.monotonic() + wait_seconds
        pending = set(range(1, self.local_world_size))
        while pending:
            for rank in sorted(pending):
                try:
                    seen = _read_private(os.path.join(folder, f"seen.{rank}"), "arrival")
                    pid, step = int(seen["pid"]), int(seen["step"])
                except (OSError, RuntimeError, KeyError, TypeError, ValueError):
                    continue
                if step >= self.counter or not _pid_alive(pid):
                    pending.discard(rank)
            if not pending:
                break
            if time.monotonic() >= deadline:
                return
            time.sleep(_POLL_SECONDS)
        shutil.rmtree(folder, ignore_errors = True)

    def _wait_for(self, path, waiting_for, template):
        started = time.monotonic()
        timeout = self._wait_timeout()
        next_note = started + _WAIT_NOTE_SECONDS
        while not os.path.exists(path):
            time.sleep(_POLL_SECONDS)
            now = time.monotonic()
            if now - started >= timeout:
                raise TimeoutError(
                    f"{_PREFIX} local rank {self.local_rank} waited {int(now - started)}s for "
                    f"{waiting_for} at step {self.counter} ({_short(template)}) and gave up. A rank "
                    f"may have failed, taken another branch, or not be running; raise "
                    f"UNSLOTH_FSDP2_WAIT_TIMEOUT (seconds) if the step is just slow."
                )
            if now >= next_note:
                _log(f"local rank {self.local_rank} still waiting ({int(now - started)}s) for "
                     f"{waiting_for} at step {self.counter}: {_short(template)}")
                next_note += _WAIT_NOTE_SECONDS

    def _divergence(self, rank, theirs, ours, template):
        return _Divergence(
            f"{_PREFIX} local ranks diverged at step {self.counter}: local rank {rank} reached a "
            f"different shell / file step (hash {theirs}) than local rank {self.local_rank} "
            f"(hash {ours}, {_short(template)}). Every rank must run the same get_ipython() "
            f"calls in the same order."
        )

    def _once(self, kind, template, action):
        """Run ``action()`` on local rank 0 and give every local rank its result.

        ``action`` returns ``{"code": int, "output": list or None, "skipped": bool}``.
        ``template`` is the step's unexpanded text: it names the step in messages and its
        hash must match across ranks.
        """
        self.counter += 1
        if not self.distributed:
            return action()
        folder = self._markers()
        digest = _step_hash(kind, template)
        marker = os.path.join(folder, f"{self.counter}.done")
        if self.local_rank != 0:
            _write_private(os.path.join(folder, f"{self.counter}.arrived.{self.local_rank}"), {"hash": digest})
            self._wait_for(marker, "local rank 0", template)
            done = _read_private(marker, "done")
            self._ack(self.counter)
            if done["hash"] != digest:
                raise self._divergence(0, done["hash"], digest, template)
            if done["error"] is not None:
                raise _rebuild_error(done["error"])
            return {"code": done["code"], "output": done["output"], "skipped": done["skipped"]}

        result = {"code": 1, "output": None, "skipped": False}
        error = None
        try:
            for rank in range(1, self.local_world_size):
                arrival = os.path.join(folder, f"{self.counter}.arrived.{rank}")
                self._wait_for(arrival, f"local rank {rank} to arrive", template)
                theirs = _read_private(arrival, "arrival")["hash"]
                if theirs != digest:
                    raise self._divergence(rank, theirs, digest, template)
            result = action()
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": _redact(str(exc))}
            raise
        finally:
            _write_private(marker, {"hash": digest, "code": int(result["code"]),
                                    "output": result["output"], "skipped": bool(result["skipped"]),
                                    "error": error})
        return result

    def _install_action(self, command, template, capture = False):
        """Run or skip one install command, on the process that runs shell steps. Only
        this process decides; the other local ranks follow its recorded result."""
        setting = os.environ.get("UNSLOTH_FSDP2_RUN_INSTALLS", "").strip()
        if setting == "1":
            return self._run(command, capture)
        if setting == "0":
            reason = "UNSLOTH_FSDP2_RUN_INSTALLS=0"
        else:
            skip, reason = install_decision(command)
            if not skip:
                _log(f"running install step ({_redact(reason)}): {_short(template)}")
                return self._run(command, capture)
        self._note_skip(template, reason)
        return {"code": 0, "output": [] if capture else None, "skipped": True}

    def _note_skip(self, template, reason):
        _log(f"skipped install step ({_redact(reason)}): {_short(template)}")
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
        if _IS_WINDOWS:
            path = shutil.which("bash")
            if not path or "system32" in path.lower():
                raise RuntimeError(
                    f"{_PREFIX} shell steps need bash, and only "
                    f"{'WSL bash (' + path + ')' if path else 'no bash'} was found on PATH. "
                    f"Install Git for Windows (Git Bash) and put its bash first on PATH, or run "
                    f"the script under WSL / Linux."
                )
            return path
        if os.path.exists("/bin/bash"):
            return "/bin/bash"
        path = shutil.which("bash")
        if not path:
            raise RuntimeError(f"{_PREFIX} shell steps need bash, and none was found on PATH.")
        return path

    def _run(self, command, capture = False):
        """Run ``command`` here. ``capture`` returns its output lines (``x = !cmd``)."""
        sys.stdout.flush()
        sys.stderr.flush()
        argv = [self._bash(), "-c", _SHELL_PREAMBLE + command]
        env = _shell_env()
        if capture:
            result = subprocess.run(argv, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = env)
            lines = result.stdout.decode("utf-8", "replace").splitlines()
            return {"code": result.returncode, "output": lines, "skipped": False}
        if self._capture_depth:
            result = subprocess.run(argv, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = env)
            sys.stdout.write(result.stdout.decode("utf-8", "replace"))
            return {"code": result.returncode, "output": None, "skipped": False}
        return {"code": subprocess.run(argv, env = env).returncode, "output": None, "skipped": False}

    def _shell(self, command):
        return self._run(command)["code"]

    def _system(self, command, template = None):
        template = command if template is None else template
        if self.mp_child:
            return self._child_skip()
        if is_install_command(command):
            return self._once("system", template, lambda: self._install_action(command, template))["code"]
        return self._once("system", template, lambda: self._run(command))["code"]

    def system(self, cmd):
        """``!cmd``: like IPython, a failing command does not raise; ``_exit_code`` holds it."""
        frame = sys._getframe(1)
        code = self._system(_expand(cmd, frame), cmd)
        frame.f_globals["_exit_code"] = code
        return None

    def getoutput(self, cmd):
        """``x = !cmd``: runs once per node like ``!cmd`` (installs follow the same rules);
        every local rank gets local rank 0's output lines."""
        command = _expand(cmd, sys._getframe(1))
        if self.mp_child:
            if is_install_command(command):
                self._child_skip()
                return []
            return self._run(command, capture = True)["output"]
        if is_install_command(command):
            action = lambda: self._install_action(command, cmd, capture = True)
        else:
            action = lambda: self._run(command, capture = True)
        return list(self._once("getoutput", cmd, action)["output"] or [])

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
        template = line
        line = _expand(line, frame)
        if magic_name == "bash":
            return self._bash_cell(cell)
        if magic_name in ("file", "writefile"):
            return self._write_file(line, cell, template)
        if magic_name == "capture":
            return self._capture(line, cell, frame)
        self._unsupported("run_cell_magic", magic_name, frame)

    def _bash_cell(self, cell):
        # %%bash is not expanded by IPython; it raises on a non-zero exit, on every rank.
        code = self._system(cell, cell)
        if code:
            raise subprocess.CalledProcessError(code, cell)
        return None

    def _write_file(self, line, cell, template = None):
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
            return {"code": 0, "output": None, "skipped": False}

        if self.mp_child:
            self._child_skip()
            return None
        # A failed write raises here on local rank 0 and, from its marker, on every peer.
        template = line if template is None else template
        self._once("file", f"%%file {template}\n{cell}", write)
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

    helper = os.path.normpath(os.path.abspath(__file__))
    wanted = os.path.normpath(target)
    spawns = [mp_spawn]
    try:
        # datasets' Dataset.map(num_proc = N) uses the multiprocess package, a fork of
        # multiprocessing with its own spawn module.
        import multiprocess.spawn as mpx_spawn
        spawns.append(mpx_spawn)
    except ImportError:
        pass
    for module in spawns:
        original = module.get_preparation_data
        if getattr(original, "_unsloth_fsdp2", False):
            continue

        def get_preparation_data(name, original = original):
            data = original(name)
            if os.path.normpath(data.get("init_main_from_path") or "") == wanted:
                data["init_main_from_path"] = helper
            return data

        get_preparation_data._unsloth_fsdp2 = True
        module.get_preparation_data = get_preparation_data
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
    real_ipython = hasattr(builtins, "get_ipython")
    if set_fsdp2 or not real_ipython:
        # A worker always uses the adapter, even where an IPython get_ipython exists:
        # only the adapter runs shell steps once per node. A single process inside a real
        # IPython session keeps IPython's own.
        init_globals["get_ipython"] = get_ipython
    if not real_ipython:
        # IPython has its own __main__, which spawn children never re-import.
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


def main(launcher_path, script_name, argv = None, single_process = False, reason = None):
    """Launch ``python_scripts/<script_name>`` next to ``launcher_path`` with FSDP2.

    ``script_name=None`` is the generic form: the first script argument is the target.
    ``single_process=True`` marks a script that is not safe to run on several ranks
    (``reason`` says why): it runs single process, and asking for more processes is
    refused unless ``UNSLOTH_FSDP2_FORCE=1``.
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
    adapter = f"get_ipython adapter: {os.path.abspath(__file__)}"
    pinned = single_process and os.environ.get("UNSLOTH_FSDP2_FORCE", "").strip() != "1"
    why = f"{os.path.basename(target)} is not safe to run on several ranks: {reason or 'see its launcher'}"

    mode = _worker_mode()
    if mode == "stale":
        # LOCAL_RANK alone, without WORLD_SIZE / LOCAL_WORLD_SIZE, is not a launcher.
        _log(f"LOCAL_RANK={os.environ['LOCAL_RANK']} is set without WORLD_SIZE / LOCAL_WORLD_SIZE; "
             f"treating it as left over in the shell and ignoring it")
        os.environ.pop("LOCAL_RANK")
    elif mode == "worker":
        # Already a launcher's worker (ours, the user's torchrun, deepspeed, ...): never
        # launch again. With one local process the shell steps simply run here.
        if pinned and _env_int("WORLD_SIZE", "LOCAL_WORLD_SIZE", default = 1) > 1:
            _fail(f"{why}. Run it with one process, or set UNSLOTH_FSDP2_FORCE=1 to run it on "
                  f"every rank anyway.")
        if options["nproc"] is not None and os.environ.get("LOCAL_RANK", "0") == "0":
            _log("--nproc-per-node is ignored inside torchrun")
        if options["dry_run"]:
            print(f"would run {target} {shlex.join(rest)} in this torchrun worker".rstrip())
            print(adapter)
            return 0
        _run_worker(launcher, target, rest, set_fsdp2 = True)
        return 0

    # Not a worker. Checked before any launch choice: with one GPU per task, or for a
    # single-process script, every task would otherwise run its own copy of the script.
    problem = _launcher_env_problem()
    if problem:
        _fail(f"this looks like one of several SLURM / MPI tasks ({problem}); running here would "
              f"start one torchrun or one copy of the script per task. Run one task per node with "
              f"torchrun (torchrun --nnodes ... --nproc-per-node N {shlex.quote(launcher)}), run a "
              f"single task, or set LOCAL_RANK, WORLD_SIZE and LOCAL_WORLD_SIZE for each task so "
              f"it runs as a worker.")

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
    if pinned:
        if requested is not None and requested > 1:
            _fail(f"{source}={requested} refused: {why}. Set UNSLOTH_FSDP2_FORCE=1 to launch it "
                  f"on {requested} processes anyway.")
        note = f"running single-process: {why} (UNSLOTH_FSDP2_FORCE=1 overrides)"
        if options["dry_run"]:
            print(note)
            print(shlex.join([sys.executable, target, *rest]))
            print(adapter)
            return 0
        _log(note)
        _run_worker(launcher, target, rest, set_fsdp2 = False)
        return 0
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
            print(adapter)
            print(shlex.join(command))
            return 0
        sys.stdout.flush()
        sys.stderr.flush()
        if _IS_WINDOWS or _real_ipython_active():
            # %run from IPython / Jupyter: replacing the kernel process would kill it.
            # Windows: os.execv starts a detached process with unquoted arguments (paths
            # with spaces split) and exits at once with code 0, so wait on a child instead.
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
        print(adapter)
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
