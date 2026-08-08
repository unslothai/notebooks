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

"""Runtime-templated install-pin gate.

A ``nb/*.ipynb`` install cell may compute a pin at runtime and hand it to pip
through an IPython ``{var}`` expansion instead of a literal spec — for example
the Granite 4.0 / Nemotron Nano cells, which pick ``torch==2.7.1`` +
``mamba_ssm==2.2.5`` + ``causal_conv1d==1.5.2`` on everything up to sm_90 and a
newer trio on Blackwell.  ``scripts/molab_generate.py`` drops the install cell
entirely and rebuilds the PEP 723 header from
``molab_dependencies.plan_dependencies``, so any ``{var}`` the planner does not
know about vanishes from the generated header with no diagnostic.  That is how
those three pins were silently lost from four molab notebooks.

This module is the gate against a repeat.  Two obligations:

1. Every ``{var}`` a molab-targeted install cell feeds to pip is registered in
   ``molab_dependencies`` — either with a static PEP 508 spec
   (``_TEMPLATE_STATIC_SPECS``) or with an explicit reason for having none
   (``_TEMPLATE_NO_STATIC_SPEC``).  A new unregistered variable fails here.
2. A variable that IS registered with a static spec actually reaches
   ``plan.dependencies``.

Static only: no torch, no GPU, no marimo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import molab_dependencies as md  # noqa: E402
import molab_manifest as mm  # noqa: E402

# ---------------------------------------------------------------------------
# Notebooks in scope: every active manifest entry whose nb/ counterpart exists.
# ---------------------------------------------------------------------------


def _resolved_sources() -> list[tuple[str, Path]]:
    """``(stem, nb/<name>.ipynb)`` for each active manifest entry on disk."""
    out: list[tuple[str, Path]] = []
    for nb in mm.get_active_notebooks():
        try:
            path = md.resolve_nb_source(nb.source)
        except FileNotFoundError:
            # A stale manifest entry is test_molab_manifest.py's problem.
            continue
        out.append((nb.output.stem, path))
    return sorted(out)


_SOURCES: list[tuple[str, Path]] = _resolved_sources()

# The Mamba-hybrid notebooks whose kernels are chosen by the compute-capability
# branch.  Listed explicitly (not derived from the registry) so the expected
# pins are asserted against a second, independent statement of the intent.
_MAMBA_HYBRID_STEMS: tuple[str, ...] = (
    "Granite4.0",
    "Granite4.0_350M",
    "Nemotron-3-Nano-30B-A3B_A100",
    "Nemotron-Nano-3-30B-A3B_A100",
)

_MAMBA_HYBRID_PINS: frozenset[str] = frozenset(
    {"torch==2.7.1", "mamba_ssm==2.2.5", "causal_conv1d==1.5.2"}
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_template_registries_are_disjoint() -> None:
    """A variable cannot both have and not have a static resolution."""
    overlap = sorted(
        set(md._TEMPLATE_STATIC_SPECS) & set(md._TEMPLATE_NO_STATIC_SPEC)
    )
    assert not overlap, (
        "These template variables are registered in BOTH "
        "_TEMPLATE_STATIC_SPECS and _TEMPLATE_NO_STATIC_SPEC: "
        f"{overlap}.  Pick one."
    )


def test_static_specs_are_parseable_pep508_names() -> None:
    """Each static spec must start with a distribution name the planner can key."""
    for var, spec in sorted(md._TEMPLATE_STATIC_SPECS.items()):
        assert md.resolve_template_spec("{" + var + "}") == spec, (
            f"resolve_template_spec('{{{var}}}') did not return {spec!r}."
        )
        match = md._RE_PKG_NAME.match(spec)
        assert match is not None, (
            f"_TEMPLATE_STATIC_SPECS[{var!r}] = {spec!r} does not start with a "
            "PEP 508 distribution name, so plan_dependencies cannot key it."
        )


def test_no_static_spec_reasons_are_non_empty() -> None:
    """A deliberate drop must say why, the same contract DroppedItem carries."""
    for var, reason in sorted(md._TEMPLATE_NO_STATIC_SPEC.items()):
        assert reason.strip(), (
            f"_TEMPLATE_NO_STATIC_SPEC[{var!r}] has an empty reason.  "
            "Nothing may be dropped silently."
        )


# ---------------------------------------------------------------------------
# Obligation 1: no unregistered template variable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem,nb_path", _SOURCES, ids=[stem for stem, _ in _SOURCES]
)
def test_every_templated_pip_token_is_registered(
    stem: str, nb_path: Path
) -> None:
    """Every ``{var}`` pip token resolves to a spec or a registered reason.

    This is the recurrence gate.  Introduce a new runtime-computed pin in an
    install cell without teaching molab_dependencies about it and this fails,
    instead of the pin silently disappearing from ``molab/<stem>.py``.
    """
    unregistered: list[str] = []
    for token in md.iter_templated_tokens(nb_path):
        if md.resolve_template_spec(token) is not None:
            continue
        if md.template_drop_reason(token) is not None:
            continue
        unregistered.append(token)

    if unregistered:
        pytest.fail(
            f"UNREGISTERED TEMPLATED PIN: {nb_path.name} feeds pip the "
            f"runtime-templated token(s) {sorted(set(unregistered))}, which "
            "scripts/molab_dependencies.py has no resolution for.  The molab "
            "generator drops the install cell, so these would be missing "
            f"from molab/{stem}.py with no diagnostic.\n"
            "Add each variable to ONE of:\n"
            "  molab_dependencies._TEMPLATE_STATIC_SPECS    "
            "(variable -> the PEP 508 spec molab should pin)\n"
            "  molab_dependencies._TEMPLATE_NO_STATIC_SPEC  "
            "(variable -> why no static pin is correct)"
        )


# ---------------------------------------------------------------------------
# Obligation 2: a registered static spec actually reaches the plan
# ---------------------------------------------------------------------------

def _statically_resolved_specs(nb_path: Path) -> list[str]:
    """Sorted, de-duplicated static specs the notebook's ``{var}`` tokens map to."""
    specs = set()
    for token in md.iter_templated_tokens(nb_path):
        spec = md.resolve_template_spec(token)
        if spec is not None:
            specs.add(spec)
    return sorted(specs)


_STATICALLY_RESOLVED: list[tuple[str, Path, str]] = [
    (stem, nb_path, spec)
    for stem, nb_path in _SOURCES
    for spec in _statically_resolved_specs(nb_path)
]


@pytest.mark.parametrize(
    "stem,nb_path,spec",
    _STATICALLY_RESOLVED,
    ids=[f"{stem}::{spec}" for stem, _, spec in _STATICALLY_RESOLVED],
)
def test_statically_resolved_template_reaches_the_plan(
    stem: str, nb_path: Path, spec: str
) -> None:
    """A ``{var}`` with a registered static spec must land in the PEP 723 list.

    Calls the real planner and inspects ``plan.dependencies`` — no text search.
    """
    plan = md.plan_dependencies(nb_path)
    assert spec in plan.dependencies, (
        f"TEMPLATED PIN LOST: {nb_path.name} installs a runtime-templated "
        f"spec that molab_dependencies resolves to {spec!r}, but "
        "plan_dependencies() did not put it in the PEP 723 dependency list "
        f"(got {sorted(plan.dependencies)}).  The generated molab/{stem}.py "
        "would ship without that pin."
    )


@pytest.mark.parametrize("stem", _MAMBA_HYBRID_STEMS)
def test_mamba_hybrid_notebooks_keep_their_kernel_pins(stem: str) -> None:
    """Granite 4.0 / Nemotron Nano molab headers carry the Mamba kernel trio.

    These models are Mamba hybrids: a molab run without ``mamba_ssm`` /
    ``causal_conv1d``, or with those kernels resolved against an unpinned
    torch, cannot run the model.  The install cell picks the pins through
    ``{_torch}`` / ``{_mamba}`` / ``{_conv}``, so this asserts the planner
    resolves the non-Blackwell arm rather than dropping all three.
    """
    nb_path = REPO_ROOT / "nb" / f"{stem}.ipynb"
    if not nb_path.exists():
        pytest.skip(f"nb/{stem}.ipynb not present.")

    deps = set(md.plan_dependencies(nb_path).dependencies)
    missing = sorted(_MAMBA_HYBRID_PINS - deps)
    assert not missing, (
        f"MAMBA KERNEL PINS MISSING from the {stem} molab dependency plan: "
        f"{missing}.  Got {sorted(deps)}."
    )


# ---------------------------------------------------------------------------
# Belt and braces: no unexpanded brace survives into a committed header
# ---------------------------------------------------------------------------

_MOLAB_DIR = REPO_ROOT / "molab"
_GENERATED_FILES: list[Path] = (
    sorted(_MOLAB_DIR.glob("*.py")) if _MOLAB_DIR.exists() else []
)


@pytest.mark.parametrize(
    "py_file", _GENERATED_FILES, ids=lambda p: p.stem
)
def test_generated_header_has_no_unexpanded_template(py_file: Path) -> None:
    """No committed PEP 723 dependency may still contain a ``{var}`` brace.

    A resolution bug that substituted the variable NAME rather than its spec
    would produce an unparseable dependency; uv would fail the sandbox build
    at notebook start.
    """
    plan_specs = [
        line
        for line in py_file.read_text(encoding="utf-8").splitlines()
        if line.startswith("#     \"") and ("{" in line or "}" in line)
    ]
    assert not plan_specs, (
        f"{py_file.name} has PEP 723 dependencies with an unexpanded IPython "
        f"template: {plan_specs}"
    )
