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

"""A notebook may not pin a vLLM that its own install cell cannot then run.

Two independent gates live here, because the T4 branch of the generated GRPO
install cells is squeezed from both directions at once.

Gate 1 -- the pinned vLLM must import the pinned transformers.

Up to 0.9.2, `vllm/transformers_utils/configs/ovis.py` ran

    AutoConfig.register("aimv2", AIMv2Config)

transformers 4.54 added a model type of its own by that name, and
`AutoConfig.register` refuses a name that is already taken:

    ValueError: 'aimv2' is already used by a Transformers config, pick another name.

That fires at import time from `vllm/config.py`, so it is not confined to Ovis:
`import vllm` fails outright, and with it every notebook that runs
`fast_inference = True` or starts `vllm.entrypoints.openai.api_server`. vLLM
dropped the bare registration in 0.10.0, keeping only `aimv2_visual_tokenizer`.
The T4 branch used to pin `vllm==0.9.2` beside `transformers==4.56.2`, which is
that pairing exactly.

unsloth does carry a mitigation for this one: `unsloth/import_fixes.py` rewrites
the installed vLLM's `ovis.py` in place when the version is below 0.10.1. It is
a repair of somebody else's source file on disk, conditional on that file being
writable and on the exact registration line still being there, so the gate below
is written against vLLM itself rather than against the repair holding.

Gate 2 -- the pinned vLLM must survive unsloth_zoo's standby guard.

Fixing gate 1 alone is not enough, and this is the gate the first attempt at
this pin was missing. Every generated GRPO install cell sets
`UNSLOTH_VLLM_STANDBY = "1"`, and with standby enabled
`unsloth_zoo.vllm_utils.patch_vllm` refuses two whole version ranges outright:

    if os.getenv("UNSLOTH_VLLM_STANDBY", "0") != "0":
        if Version("0.10.0") <= Version(vllm_version) < Version("0.11.0"):
            raise RuntimeError(
                "Unsloth: vLLM 0.10.x crashes with std::bad_alloc when standby mode is "
                "enabled due to insufficient memory headroom in CuMemAllocator.\\n"
                "Please update vLLM: pip install --upgrade vllm>=0.11.2"
            )
        if Version("0.14.0") <= Version(vllm_version) < Version("0.15.0"):
            raise RuntimeError(
                "Unsloth: vLLM 0.14.x has a known bug (cudaErrorIllegalAddress) in "
                "CuMemAllocator during sleep/wake cycles which crashes standby mode.\\n"
                "Please update vLLM: pip install --upgrade vllm>=0.15.1"
            )

A pin inside either range is a hard RuntimeError on the fourth cell of the
notebook, on real hardware, with a green CI. Nothing static can see the crash,
but the pin can be checked against the guard's own bounds, so it is.

The bounds are read out of the installed unsloth_zoo when there is one, so the
gate tracks upstream if the windows move. CI's notebook static-checks job
installs no torch, and importing unsloth_zoo would import torch, so the source
is located with `find_spec` and parsed rather than imported. When it is not
installed the literals above are used; they are transcribed from
`unsloth_zoo/vllm_utils.py::patch_vllm`.

Gate 2 deliberately does not encode the other half of the T4 squeeze, the
Turing ceiling: `vllm/v1/attention/backends/xformers.py` is present at 0.11.2
and deleted in 0.12.0, so a T4 has no attention backend above 0.11.2. That is a
property of vLLM's tree rather than of anything this repo can read at test
time, and it is written up above `installation_grpo_content` in
update_all_notebooks.py instead.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIRS = (REPO_ROOT / "nb", REPO_ROOT / "kaggle")

# (first transformers that breaks it, first vLLM that is fixed, why)
INCOMPATIBILITIES = [
    ((4, 54), (0, 10, 0),
     "vLLM below 0.10.0 registers its Ovis config as `aimv2`, a name "
     "transformers took in 4.54, so `import vllm` raises ValueError"),
]

# Fallback for gate 2, used only when unsloth_zoo is not importable. Each entry
# is (first bad version, first good version, why). Transcribed from
# unsloth_zoo/vllm_utils.py::patch_vllm; see the module docstring.
FALLBACK_STANDBY_WINDOWS = [
    ((0, 10, 0), (0, 11, 0),
     "vLLM 0.10.x crashes with std::bad_alloc when standby mode is enabled "
     "due to insufficient memory headroom in CuMemAllocator"),
    ((0, 14, 0), (0, 15, 0),
     "vLLM 0.14.x has a known bug (cudaErrorIllegalAddress) in CuMemAllocator "
     "during sleep/wake cycles which crashes standby mode"),
]

_VLLM_PIN = re.compile(r"vllm\s*==\s*([0-9]+(?:\.[0-9]+)*)")
_TRANSFORMERS_PIN = re.compile(r"transformers\s*==\s*([0-9]+(?:\.[0-9]+)*)")

# `os.environ["UNSLOTH_VLLM_STANDBY"] = "1"`, through the notebook's JSON
# escaping. The guard treats any value other than "0" as on, so the value is
# captured and compared rather than matched literally.
_STANDBY_SET = re.compile(
    r"""UNSLOTH_VLLM_STANDBY\\?["']\s*\]?\s*=\s*\\?["']([^"'\\]*)\\?["']""")

# The two ternary spellings the install cells use, generated and hand-tuned:
#   _vllm, _triton   = ('vllm==0.11.2', 'triton') if is_t4 else (...)
#   get_vllm, get_triton = ("vllm==0.11.2", "triton") if is_t4 else (...)
_T4_TERNARY = re.compile(
    r"""[\w]*vllm,\s*[\w]*triton\s*=\s*\(\s*\\?["']([^"'\\]+)\\?["']"""
    r"""[^\n]*?\bif\s+is_t4\b""")


def _version(text: str):
    return tuple(int(part) for part in text.split("."))


def _standby_windows():
    """The guard's windows, read from unsloth_zoo when it is installed.

    Returns the parsed windows, plus the source they came from for the failure
    message. Never imports unsloth_zoo: that would pull in torch, which the
    static-checks CI job does not install.
    """
    try:
        spec = importlib.util.find_spec("unsloth_zoo")
        origin = spec.origin if spec is not None else None
    except (ImportError, ValueError):
        origin = None
    if origin:
        source_file = Path(origin).parent / "vllm_utils.py"
        if source_file.is_file():
            text = source_file.read_text(encoding="utf-8", errors="replace")
            pairs = re.findall(
                r"""Version\(["']([0-9.]+)["']\)\s*<=\s*Version\(\s*vllm_version"""
                r"""\s*\)\s*<\s*Version\(["']([0-9.]+)["']\)""",
                text)
            if pairs:
                return ([(_version(lo), _version(hi),
                          "unsloth_zoo's standby guard rejects "
                          f"{lo} <= vllm < {hi}")
                         for lo, hi in pairs],
                        str(source_file))
    return FALLBACK_STANDBY_WINDOWS, "the literals in this test"


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code")


def _notebooks():
    found = []
    for directory in NB_DIRS:
        found += sorted(directory.glob("*.ipynb"))
    return found


def _enables_standby(source: str) -> bool:
    """True if the notebook turns standby on, by the guard's own rule.

    `patch_vllm` checks `!= "0"`, so anything other than "0" counts.
    """
    return any(value != "0" for value in _STANDBY_SET.findall(source))


def _conflicts(source: str):
    """Every (vllm, transformers, reason) pairing in this notebook that cannot
    import.

    Both pins are compared across the whole notebook rather than per cell: the
    transformers pin lives on the last line of the install cell and the vLLM
    pin four lines above it, but a notebook may also carry a second install
    cell, and any combination the reader can end up with counts.
    """
    vllms = {_version(m) for m in _VLLM_PIN.findall(source)}
    transformers = {_version(m) for m in _TRANSFORMERS_PIN.findall(source)}
    bad = []
    for breaks_at, fixed_at, reason in INCOMPATIBILITIES:
        for vllm in vllms:
            if vllm >= fixed_at:
                continue
            for transformers_pin in transformers:
                if transformers_pin >= breaks_at:
                    bad.append((vllm, transformers_pin, reason))
    return bad


def _standby_conflicts(source: str, windows=None):
    """Every vLLM pin that the standby guard would refuse, if standby is on."""
    if not _enables_standby(source):
        return []
    if windows is None:
        windows = _standby_windows()[0]
    bad = []
    for vllm in sorted({_version(m) for m in _VLLM_PIN.findall(source)}):
        for first_bad, first_good, reason in windows:
            if first_bad <= vllm < first_good:
                bad.append((vllm, reason))
    return bad


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_pinned_vllm_can_import_the_pinned_transformers(path):
    conflicts = _conflicts(_code(path))
    assert not conflicts, "\n".join(
        f"{path.relative_to(REPO_ROOT)} pins vllm=="
        + ".".join(str(p) for p in vllm)
        + " and transformers==" + ".".join(str(p) for p in transformers)
        + f": {reason}"
        for vllm, transformers, reason in conflicts)


@pytest.mark.parametrize(
    "path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_standby_notebook_pins_a_vllm_the_guard_allows(path):
    windows, came_from = _standby_windows()
    conflicts = _standby_conflicts(_code(path), windows)
    assert not conflicts, "\n".join(
        f"{path.relative_to(REPO_ROOT)} sets UNSLOTH_VLLM_STANDBY and pins "
        "vllm==" + ".".join(str(p) for p in vllm)
        + f": {reason}. patch_vllm raises RuntimeError on the install cell. "
        f"Bounds read from {came_from}."
        for vllm, reason in conflicts)


def test_the_t4_branch_pin_is_inside_every_bound():
    """The one line this whole file exists for, checked end to end.

    The T4 side of the ternary must clear the aimv2 floor and both standby
    windows at once. It is asserted here as well as per notebook so that the
    intersection is stated in one place, and so that deleting the pin (rather
    than mispinning it) is also a failure.
    """
    windows, _ = _standby_windows()
    floor = min(fixed_at for _, fixed_at, _ in INCOMPATIBILITIES)
    seen = set()
    for path in _notebooks():
        for spec in _T4_TERNARY.findall(_code(path)):
            match = _VLLM_PIN.fullmatch(spec.strip())
            assert match, (
                f"{path.relative_to(REPO_ROOT)} leaves the T4 side of the "
                f"is_t4 ternary unpinned: {spec!r}")
            pinned = _version(match.group(1))
            seen.add(pinned)
            assert pinned >= floor, (
                f"{path.relative_to(REPO_ROOT)} pins vllm=={spec} on the T4 "
                "branch, below the aimv2 floor "
                + ".".join(str(p) for p in floor))
            for first_bad, first_good, reason in windows:
                assert not first_bad <= pinned < first_good, (
                    f"{path.relative_to(REPO_ROOT)} pins vllm=={spec} on the "
                    f"T4 branch: {reason}")
    assert seen, "no notebook carries an is_t4 vLLM ternary any more; the " \
                 "regex above has gone stale and every case is vacuous"


def test_some_notebooks_really_pin_both():
    """A regex that stopped matching would leave every case above vacuous."""
    both = [p for p in _notebooks()
            if _VLLM_PIN.search(_code(p)) and _TRANSFORMERS_PIN.search(_code(p))]
    assert len(both) >= 40, f"only {len(both)} notebooks pin both"


def test_some_notebooks_really_enable_standby():
    """Same, for the standby detector: gate 2 is vacuous without it."""
    enabling = [p for p in _notebooks() if _enables_standby(_code(p))]
    assert len(enabling) >= 40, f"only {len(enabling)} notebooks set standby"


def test_the_guard_bounds_are_read_from_unsloth_zoo_when_it_is_installed():
    """The fallback literals are a backstop, not the normal path.

    If unsloth_zoo is importable and its windows have moved, this test is where
    that shows up, rather than silently in a stale fallback.
    """
    if importlib.util.find_spec("unsloth_zoo") is None:
        pytest.skip("unsloth_zoo is not installed (as in the static CI job)")
    windows, came_from = _standby_windows()
    assert came_from.endswith("vllm_utils.py"), (
        "unsloth_zoo is installed but its guard could not be parsed; the "
        "regex in _standby_windows has gone stale")
    assert [(lo, hi) for lo, hi, _ in windows] == \
           [(lo, hi) for lo, hi, _ in FALLBACK_STANDBY_WINDOWS], (
        f"unsloth_zoo's standby windows have moved (read from {came_from}). "
        "Update FALLBACK_STANDBY_WINDOWS and re-check the pin.")


# --------------------------------------------------------------------------

def test_the_pairing_that_shipped_is_rejected():
    """vllm 0.9.2 with transformers 4.56.2, the combination that broke."""
    assert _conflicts(
        '_vllm = "vllm==0.9.2"\n!uv pip install transformers==4.56.2')


def test_the_fixed_vllm_is_accepted():
    assert not _conflicts(
        '_vllm = "vllm==0.10.1"\n!uv pip install transformers==4.56.2')


def test_a_newer_vllm_is_accepted():
    assert not _conflicts(
        '_vllm = "vllm==0.15.1"\n!uv pip install transformers==4.56.2')


def test_an_old_vllm_with_an_old_transformers_is_left_alone():
    """The pairing only breaks from transformers 4.54; below that it worked,
    and flagging it would be a false positive."""
    assert not _conflicts(
        '_vllm = "vllm==0.9.2"\n!uv pip install transformers==4.53.2')


def test_an_unpinned_vllm_is_not_guessed_at():
    assert not _conflicts('_vllm = "vllm"\n!uv pip install transformers==4.56.2')


# --------------------------------------------------------------------------
# Gate 2 unit cases. `0.10.1` is the version the first attempt at this pin
# chose, and the one that failed on a real Colab T4.

_STANDBY_ON = 'os.environ["UNSLOTH_VLLM_STANDBY"] = "1"\n'


def test_the_t4_pin_that_failed_on_colab_is_rejected():
    assert _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.10.1"',
                              FALLBACK_STANDBY_WINDOWS)


def test_the_second_standby_window_is_rejected():
    assert _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.14.1"',
                              FALLBACK_STANDBY_WINDOWS)


def test_the_chosen_t4_pin_is_accepted():
    assert not _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.11.2"',
                                  FALLBACK_STANDBY_WINDOWS)


def test_the_non_t4_pin_is_accepted():
    """0.15.1 sits above both windows, which is why the non-T4 branch of the
    same line needed no change."""
    assert not _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.15.1"',
                                  FALLBACK_STANDBY_WINDOWS)


def test_the_window_edges_are_half_open():
    """0.11.0 and 0.15.0 are the first good versions, not the last bad ones."""
    assert not _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.11.0"',
                                  FALLBACK_STANDBY_WINDOWS)
    assert not _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.15.0"',
                                  FALLBACK_STANDBY_WINDOWS)
    assert _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.10.0"',
                              FALLBACK_STANDBY_WINDOWS)
    assert _standby_conflicts(_STANDBY_ON + '_vllm = "vllm==0.14.0"',
                              FALLBACK_STANDBY_WINDOWS)


def test_a_bad_pin_without_standby_is_left_alone():
    """The guard only fires with standby on, so neither should this."""
    assert not _standby_conflicts('_vllm = "vllm==0.10.1"',
                                  FALLBACK_STANDBY_WINDOWS)
    assert not _standby_conflicts(
        'os.environ["UNSLOTH_VLLM_STANDBY"] = "0"\n_vllm = "vllm==0.10.1"',
        FALLBACK_STANDBY_WINDOWS)


def test_any_truthy_standby_value_counts():
    """patch_vllm checks `!= "0"`, not `== "1"`."""
    assert _standby_conflicts(
        'os.environ["UNSLOTH_VLLM_STANDBY"] = "true"\n_vllm = "vllm==0.10.1"',
        FALLBACK_STANDBY_WINDOWS)
