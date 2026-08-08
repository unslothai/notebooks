# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
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
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A name live code reads must be bound by live code, not only by `if False:`.

`Gemma4_(12B)_Audio` loaded with `model, tokenizer = FastModel.from_pretrained`
and then called `processor` everywhere: the inference helper, the trainer's
`processing_class` and collator, the save and reload cells. `processor` was
bound in exactly one place, inside the `if False:` reload block, so the trainer
cell could only ever raise `NameError: name 'processor' is not defined`. Its two
sibling audio notebooks bind `model, processor`; that one was the odd one out.

A bound-nowhere check cannot see this, which is why it survived: the name IS
bound, just unreachably. The rule here is deliberately narrow -- flag only a
name that is bound SOMEWHERE and never in live code. That the author wrote the
binding is what makes the placement a mistake rather than a missing import.
"""

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"


def _strip_magics(source):
    """Blank out IPython line magics, keeping line numbers and indentation."""
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            out.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def _dead_nodes(tree):
    """Every node inside an `if False:` body. `if False:` is the notebooks'
    own idiom for "here is how you would reload it", not dead code to delete."""
    dead = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Constant) and test.value is False:
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    dead.add(id(sub))
    return dead


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef) \
    + _COMPREHENSIONS


def _scope_body(node):
    """The nodes evaluated inside this scope.

    A comprehension is one too in Python 3: its target does not escape, so
    `[processor for processor in items]` must not answer a later
    `use(processor)`. Its outermost iterable is the exception, evaluated in the
    enclosing scope, so the caller pushes that there instead.
    """
    if isinstance(node, _COMPREHENSIONS):
        parts = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        for index, generator in enumerate(node.generators):
            parts.append(generator.target)
            if index:
                parts.append(generator.iter)
            parts.extend(generator.ifs)
        return parts
    return node.body if isinstance(node.body, list) else [node.body]


def _analyse_scope(node, dead):
    """(live_binds, dead_binds, free_loads, offenders, escaping_binds) for one scope.

    Nested bodies are recursed into as their own scopes rather than unioned in.
    One flat set let any unrelated local -- a `processor` parameter on a helper,
    say -- stand in for the module-level binding, so a top-level
    `use(processor)` with no top-level binding read clean: the exact break this
    file exists to catch. A load only reaches the enclosing scope when its own
    scope does not bind the name.

    A scope that both binds a name only under `if False:` and reads it live is
    an offender in its own right, and has to be reported here: filtering it out
    of what escapes is what a correct scope walk does, and it would otherwise
    hide `def train(): if False: p = load(); return p` from the module.

    `escaping_binds` is the one thing a scope binds for its PARENT: a `:=` inside
    a comprehension. PEP 572 binds that in the containing scope, so treating it
    as comprehension-local made a later live read look unanswered, and with the
    name also bound under `if False:` the check reported an offender that is not
    one. Sixty comprehensions in nb/ use `:=`, so that false positive is on a
    construct the catalogue really contains.
    """
    live_binds, dead_binds, loads, nested = set(), set(), set(), []
    offenders = set()
    walrus_live, walrus_dead = set(), set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            live_binds.add(arg.arg)
        for arg in (args.vararg, args.kwarg):
            if arg is not None:
                live_binds.add(arg.arg)

    stack = _scope_body(node)
    while stack:
        current = stack.pop()
        is_dead = id(current) in dead
        if isinstance(current, _SCOPES):
            if hasattr(current, "name"):
                (dead_binds if is_dead else live_binds).add(current.name)
            nested.append(current)
            # Decorators, defaults, bases and the outermost comprehension
            # iterable are all evaluated in THIS scope.
            stack.extend(getattr(current, "decorator_list", None) or [])
            stack.extend(getattr(current, "bases", None) or [])
            args = getattr(current, "args", None)
            if args is not None:
                stack.extend(d for d in args.defaults if d is not None)
                stack.extend(d for d in (args.kw_defaults or []) if d is not None)
            if isinstance(current, _COMPREHENSIONS) and current.generators:
                stack.append(current.generators[0].iter)
            continue
        if isinstance(current, ast.NamedExpr):
            # Recorded as well as, not instead of, the ordinary Store below:
            # the name is usable in this scope too, it just also escapes.
            (walrus_dead if is_dead else walrus_live).add(current.target.id)
        if isinstance(current, ast.Name):
            if isinstance(current.ctx, ast.Store):
                (dead_binds if is_dead else live_binds).add(current.id)
            elif isinstance(current.ctx, ast.Load) and not is_dead:
                loads.add(current.id)
        elif isinstance(current, ast.alias):
            name = (current.asname or current.name).split(".")[0]
            (dead_binds if is_dead else live_binds).add(name)
        elif isinstance(current, ast.ExceptHandler) and current.name:
            (dead_binds if is_dead else live_binds).add(current.name)
        stack.extend(ast.iter_child_nodes(current))

    for child in nested:
        (_child_live, _child_dead, child_free, child_offenders,
         child_escaping) = _analyse_scope(child, dead)
        loads |= child_free
        offenders |= child_offenders
        live_binds |= child_escaping[0]
        dead_binds |= child_escaping[1]
    escaping = ((walrus_live, walrus_dead) if isinstance(node, _COMPREHENSIONS)
                else (set(), set()))
    if isinstance(node, ast.Module):
        # Module-level detection stays in `_offenders`, which sees every cell.
        return live_binds, dead_binds, loads, offenders, escaping
    offenders |= (loads & dead_binds) - live_binds
    # Bindings here are local, so neither they nor what they satisfy escape.
    return (live_binds, dead_binds, loads - live_binds - dead_binds, offenders,
            escaping)


def _scan(source):
    """(live_binds, dead_binds, live_loads, nested_offenders, _), or None."""
    try:
        tree = ast.parse(_strip_magics(source))
    except SyntaxError:
        return None
    return _analyse_scope(tree, _dead_nodes(tree))


def _offenders(path):
    """Names this notebook reads live and binds only inside `if False:`."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    live_binds, dead_binds, live_loads, nested = set(), set(), set(), set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        result = _scan("".join(cell.get("source", [])))
        if result is None:
            # An unparseable cell is skipped rather than guessed at, so this
            # check can only ever under-report. Silence beats a false failure
            # on 434 notebooks.
            continue
        live_binds |= result[0]
        dead_binds |= result[1]
        live_loads |= result[2]
        nested |= result[3]
    # Module level is decided across every cell, since a notebook binds in one
    # and reads in another. A nested scope was already decided on its own.
    return sorted(((live_loads & dead_binds) - live_binds) | nested)


_NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb")) if NB_DIR.is_dir() else []


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_no_live_name_is_bound_only_in_dead_code(path):
    names = _offenders(path)
    assert not names, (
        f"{path.name} reads {names} but binds them only inside `if False:`, "
        f"so those cells can only raise NameError"
    )


def test_the_check_would_have_caught_the_gemma4_audio_break():
    """Guards the rule itself: without it, the shipped notebook read clean."""
    broken = {
        "cells": [
            {"cell_type": "code", "source": [
                "model, tokenizer = FastModel.from_pretrained(name)\n"]},
            {"cell_type": "code", "source": [
                "trainer = SFTTrainer(processing_class = processor.tokenizer)\n"]},
            {"cell_type": "code", "source": [
                "if False:\n",
                "    model, processor = FastModel.from_pretrained(name)\n"]},
        ]
    }
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as handle:
        json.dump(broken, handle)
        handle.flush()
        assert _offenders(Path(handle.name)) == ["processor"]

    fixed = json.loads(json.dumps(broken))
    fixed["cells"][0]["source"] = [
        "model, processor = FastModel.from_pretrained(name)\n"]
    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as handle:
        json.dump(fixed, handle)
        handle.flush()
        assert _offenders(Path(handle.name)) == []


def _offenders_of(cells):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as handle:
        json.dump({"cells": [{"cell_type": "code", "source": [c]} for c in cells]},
                  handle)
        handle.flush()
        return _offenders(Path(handle.name))


def test_a_local_of_the_same_name_does_not_satisfy_a_top_level_load():
    """A helper's own `processor` parameter is not the notebook's `processor`."""
    assert _offenders_of([
        "def describe(processor):\n    return processor.tokenizer\n",
        "trainer = SFTTrainer(processing_class = processor.tokenizer)\n",
        "if False:\n    model, processor = FastModel.from_pretrained(name)\n",
    ]) == ["processor"]


def test_a_body_reading_its_own_parameter_is_still_not_a_free_load():
    assert _offenders_of([
        "if False:\n    processor = None\n",
        "def describe(processor):\n    return processor.tokenizer\n",
    ]) == []


def test_a_function_reading_a_module_global_is_still_checked():
    """The Gemma4 inference helper read `processor` with nothing binding it."""
    assert _offenders_of([
        "def infer(sample):\n    return processor(sample)\n",
        "if False:\n    model, processor = FastModel.from_pretrained(name)\n",
    ]) == ["processor"]


def test_a_name_bound_live_and_again_in_dead_code_is_not_flagged():
    """The reload blocks rebind `model` all over the repo; that is fine."""
    ok = {
        "cells": [
            {"cell_type": "code", "source": ["model = load()\n", "use(model)\n"]},
            {"cell_type": "code", "source": ["if False:\n", "    model = reload()\n"]},
        ]
    }
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False) as handle:
        json.dump(ok, handle)
        handle.flush()
        assert _offenders(Path(handle.name)) == []


def test_a_dead_only_binding_inside_a_helper_is_reported():
    """`train()` raises UnboundLocalError. Filtering the name out of what
    escapes is right; dropping it without a word is not."""
    assert _offenders_of([
        "def train():\n    if False:\n        processor = load()\n    return processor\n",
    ]) == ["processor"]


def test_a_comprehension_target_does_not_escape():
    """Comprehension scopes are their own in Python 3, so the loop variable
    cannot answer a later read."""
    assert _offenders_of([
        "if False:\n    processor = load()\n",
        "values = [processor for processor in items]\n",
        "use(processor)\n",
    ]) == ["processor"]


@pytest.mark.parametrize("comprehension", [
    "[p for p in items]",
    "{p for p in items}",
    "{p: 1 for p in items}",
    "(p for p in items)",
])
def test_every_comprehension_form_is_scoped(comprehension):
    assert _offenders_of([
        "if False:\n    p = load()\n",
        f"values = {comprehension}\n",
        "use(p)\n",
    ]) == ["p"]


def test_the_outermost_iterable_is_read_in_the_enclosing_scope():
    """`[x for x in processor]` really does read `processor` out here."""
    assert _offenders_of([
        "if False:\n    processor = load()\n",
        "values = [x for x in processor]\n",
    ]) == ["processor"]


def test_a_comprehension_reading_its_own_target_is_not_a_free_load():
    assert _offenders_of([
        "if False:\n    p = load()\n",
        "values = [p.name for p in items]\n",
    ]) == []

def test_a_walrus_in_a_comprehension_binds_in_the_enclosing_scope():
    """PEP 572: `:=` inside a comprehension binds outside it, not inside.

    Treating it as comprehension-local made the read in the third cell look
    unanswered, and with `processor` also bound under `if False:` the check
    reported an offender that is not one -- a false positive in a gate that
    runs over the whole catalogue.
    """
    assert _offenders_of([
        "if False:\n    processor = None\n",
        "values = [(processor := item) for item in items]\n",
        "use(processor)\n",
    ]) == []


def test_a_comprehension_target_still_does_not_escape():
    """The narrowing must not take the ordinary loop target with it."""
    assert _offenders_of([
        "if False:\n    processor = None\n",
        "values = [processor for processor in items]\n",
        "use(processor)\n",
    ]) == ["processor"]
