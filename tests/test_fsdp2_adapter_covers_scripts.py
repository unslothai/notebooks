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

"""Every live ``get_ipython()`` call in ``python_scripts/`` is one the fsdp2 adapter implements.

The adapter raises ``NotImplementedError`` for an unknown magic only at run time,
after the model loads; this gate catches it at PR time. ``%%capture`` bodies are
transformed and walked too. With IPython installed, the fallback transform must match
IPython's AST on every ``%%capture`` body.

No torch; stdlib only (plus IPython when present).
"""
from __future__ import annotations

import ast
import importlib.util
import warnings
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_PATH = REPO_ROOT / "scripts" / "fsdp2_launch.py"
SCRIPTS = sorted((REPO_ROOT / "python_scripts").glob("*.py"))


def _load_launch():
    spec = importlib.util.spec_from_file_location("_fsdp2_launch_cover", str(LAUNCH_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FL = _load_launch()


def _parse(source, name):
    with warnings.catch_warnings():
        # Notebook mirrors carry regex strings with invalid escapes; not this test's job.
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source, filename = name)


def _calls(tree, where, capture_bodies):
    """(method, magic, where) for every get_ipython().<method>(...) in the tree."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name)
                and owner.func.id == "get_ipython"):
            continue
        method = node.func.attr
        magic = None
        if method in ("run_cell_magic", "run_line_magic"):
            first = node.args[0] if node.args else None
            magic = first.value if isinstance(first, ast.Constant) else f"<non-literal at line {node.lineno}>"
        found.append((method, magic, f"{where}:{node.lineno}"))
        if (method, magic) == ("run_cell_magic", "capture") and len(node.args) == 3:
            body = node.args[2]
            if isinstance(body, ast.Constant) and isinstance(body.value, str):
                capture_bodies.append((body.value, f"{where}:{node.lineno}"))
    return found


def _collect():
    calls, bodies = [], []
    for path in SCRIPTS:
        calls.extend(_calls(_parse(path.read_text(encoding = "utf-8"), str(path)), path.name, bodies))
    for body, where in list(bodies):
        transformed = FL._minimal_transform(body)
        calls.extend(_calls(_parse(transformed, where), f"{where} (%%capture body)", []))
    return calls, bodies


CALLS, CAPTURE_BODIES = _collect()


def test_corpus_is_not_empty():
    # A glob or walk that silently finds nothing would pass the real check vacuously.
    assert len(SCRIPTS) > 100
    files = {where.split(":")[0] for _, _, where in CALLS}
    assert len(files) > 50
    assert CAPTURE_BODIES


def test_every_live_get_ipython_call_is_supported():
    unsupported = Counter()
    examples = {}
    for method, magic, where in CALLS:
        if (method, magic) not in FL.SUPPORTED_CALLS:
            unsupported[(method, magic)] += 1
            examples.setdefault((method, magic), where)
    assert not unsupported, (
        "python_scripts/ uses get_ipython() calls the fsdp2 adapter (scripts/fsdp2_launch.py) "
        "does not implement; add them to FSDP2Shell and SUPPORTED_CALLS:\n"
        + "\n".join(f"  {m}({g!r}) x{n}, e.g. {examples[(m, g)]}" for (m, g), n in unsupported.items())
    )


def test_supported_table_matches_the_shell():
    for method, _ in FL.SUPPORTED_CALLS:
        assert callable(getattr(FL.FSDP2Shell, method, None)), method


def test_fallback_transform_matches_ipython():
    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except Exception:
        pytest.skip("IPython is not installed")
    manager = TransformerManager()
    mismatched = []
    for body, where in CAPTURE_BODIES:
        ours = ast.dump(_parse(FL._minimal_transform(body), where))
        theirs = ast.dump(_parse(manager.transform_cell(body), where))
        if ours != theirs:
            mismatched.append(where)
    assert not mismatched, f"fallback transform differs from IPython for: {mismatched[:10]}"
