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

"""Shared notebook inventory helpers.

Walk the .ipynb tree under original_template/, nb/, kaggle/ and yield
(path, cell_index, source) tuples plus the canonical magic-strip
routine. Lifted from update_all_notebooks.py::validate_notebook_syntax
(lines 1478-1535) so the in-process generator gate and the pytest gate
agree on what "parseable Python" means.

Public surface used by tests/ and CI:

  iter_notebooks(roots=None)              -> Iterator[Path]
  iter_code_cells(path)                   -> Iterator[(int, str)]
  strip_jupyter_magics(source)            -> str    # cell-friendly
  extract_pip_pins_from_cell(source)      -> list[(pkg, version)]
  read_pin_constants()                    -> dict[str, str]
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK_ROOTS = ("original_template", "nb", "kaggle")

_PIN_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.\-]*)==([0-9][0-9A-Za-z.\-+_]*)")


def iter_notebooks(roots: Iterable[str] | None = None) -> Iterator[Path]:
    """Yield every .ipynb path under each root, in deterministic order."""
    if roots is None:
        roots = DEFAULT_NOTEBOOK_ROOTS
    for root in roots:
        d = REPO_ROOT / root
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.ipynb")):
            yield p


def iter_code_cells(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (cell_index, source_str) for every code cell in a notebook.
    Skips empty cells. Raises if the file is not valid JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        if not (src or "").strip():
            continue
        yield i, src


def strip_jupyter_magics(source: str) -> str:
    """Replace Jupyter magic / shell lines with ``pass`` placeholders so
    the result parses as plain Python. Mirrors the logic in
    update_all_notebooks.py::validate_notebook_syntax (1478-1535)."""
    clean_lines: list[str] = []
    in_shell_continuation = False
    in_cell_magic = False
    shell_block_indent = ""
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if in_cell_magic:
            clean_lines.append(shell_block_indent + "pass")
            continue
        if in_shell_continuation:
            clean_lines.append(shell_block_indent + "pass")
            in_shell_continuation = line.rstrip().endswith("\\")
            if not in_shell_continuation:
                shell_block_indent = ""
            continue
        if stripped.startswith("%%"):
            shell_block_indent = indent
            clean_lines.append(shell_block_indent + "pass")
            in_cell_magic = True
            continue
        if stripped.startswith(("!", "%")):
            shell_block_indent = indent
            clean_lines.append(shell_block_indent + "pass")
            in_shell_continuation = line.rstrip().endswith("\\")
            if not in_shell_continuation:
                shell_block_indent = ""
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def parse_cell_or_collect_error(source: str) -> SyntaxError | None:
    """Return None if the cell parses cleanly after magic stripping;
    otherwise return the SyntaxError captured."""
    clean = strip_jupyter_magics(source)
    if not clean.strip():
        return None
    try:
        ast.parse(clean)
    except SyntaxError as exc:
        return exc
    return None


def extract_pip_pins_from_cell(source: str) -> list[tuple[str, str]]:
    """Return the ``[(pkg, version), ...]`` pairs found in every
    ``!pip install`` / ``%pip install`` / ``uv pip install`` line in a
    cell. Only the literal ``pkg==X.Y.Z`` form is matched -- range and
    open pins are intentionally ignored so the test stays strict on the
    pinned set."""
    out: list[tuple[str, str]] = []
    for line in source.splitlines():
        s = line.strip()
        if not s:
            continue
        is_install = (
            s.startswith("!pip install")
            or s.startswith("%pip install")
            or s.startswith("!uv pip install")
            or s.startswith("%uv pip install")
            or s.startswith("uv pip install")
        )
        if not is_install:
            continue
        for pkg, ver in _PIN_RE.findall(s):
            out.append((pkg.replace("_", "-").lower(), ver))
    return out


def read_pin_constants(
    generator_path: Path | None = None,
) -> dict[str, str]:
    """Parse update_all_notebooks.py and extract the canonical PIN_*
    constants by AST. Returns a dict from package name (lowercase) to
    pinned version string.

    Specifically reads the module-level assigns at lines 107-117:
      PIN_TRANSFORMERS = "!pip install transformers==4.56.2"
      PIN_TRL          = "!pip install --no-deps trl==0.22.2"
      PIN_TOKENIZERS_SPEC = "tokenizers>=0.22.0,<=0.23.0"
    and any other ``PIN_<NAME>`` string assign that contains a literal
    ``<pkg>==<version>``."""
    if generator_path is None:
        generator_path = REPO_ROOT / "update_all_notebooks.py"
    pins: dict[str, str] = {}
    src = generator_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("PIN_")
        ):
            continue
        if not (
            isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        text = node.value.value
        for pkg, ver in _PIN_RE.findall(text):
            pins[pkg.replace("_", "-").lower()] = ver
    return pins


PINNED_PACKAGES = (
    "transformers",
    "trl",
    "tokenizers",
    "datasets",
    "peft",
    "accelerate",
    "bitsandbytes",
    "unsloth",
    "unsloth-zoo",
)
# Packages NOT in the strict set above are resolved dynamically by the
# generator (e.g. xformers / torchao / fbgemm-genai are picked from a
# dict keyed on torch minor; torch / triton come from per-env install
# helpers). Including them in the strict pin-check would fire false
# positives for any notebook whose dynamic-resolved version is not
# spelled literally in update_all_notebooks.py.


def extract_per_model_override_pins(
    generator_path: Path | None = None,
) -> set[tuple[str, str]]:
    """Walk every string constant in update_all_notebooks.py and return
    the set of ``(pkg, version)`` literals it mentions, restricted to
    the packages we pin-check (PINNED_PACKAGES). This is the programmatic
    source of truth for the legitimate-pin allowlist: any ``pkg==X.Y.Z``
    that appears anywhere in the generator script counts as legitimate.
    Catches the failure mode where someone hand-edits a generated
    notebook to bump a pin without updating the generator."""
    if generator_path is None:
        generator_path = REPO_ROOT / "update_all_notebooks.py"
    interesting = set(PINNED_PACKAGES)
    out: set[tuple[str, str]] = set()
    src = generator_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for pkg, ver in _PIN_RE.findall(node.value):
                p = pkg.replace("_", "-").lower()
                if p in interesting:
                    out.add((p, ver))
    return out
