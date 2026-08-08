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

"""No notebook may start a server with a backgrounded `!` shell command.

`Gemma3N_(2B)-Inference` launched its server with

    !nohup python -m sglang.launch_server ... > sglang.log &

IPython refuses that. `InteractiveShell.system_piped` raises
`OSError("Background processes not supported.")` for any command whose
stripped form ends in `&`, and `ipykernel.zmqshell.ZMQInteractiveShell` binds
`system = system_piped`. Colab is the exception: it replaces `ip.system` with
`system_raw`, which is why this never showed there. Every other kernel --
Kaggle, plain Jupyter, papermill -- could not run the cell at all.

`subprocess.Popen` works on every kernel and hands back a handle, which is also
what lets the wait be bounded: sglang's `wait_for_server` takes `timeout` and
`process`, and with neither it waits forever, exactly like the
`while ! grep -q ... ; do sleep 5; done` loop that shape replaces.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "nb"
TEMPLATE_DIR = REPO_ROOT / "original_template"

# A `!`-escaped shell command left in the background. `&&` is a conjunction, not
# backgrounding, so it is excluded; `system_piped` only inspects the very last
# character of the stripped command.
_BACKGROUNDED = re.compile(r"^\s*!.*(?<!&)&\s*$")


def _notebooks():
    return sorted(NB_DIR.glob("*.ipynb")) + sorted(TEMPLATE_DIR.glob("*.ipynb"))


def _offending_lines(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            if _BACKGROUNDED.match(line):
                bad.append(f"cell {index}: {line.strip()}")
    return bad


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_notebook_backgrounds_a_shell_command(path):
    bad = _offending_lines(path)
    assert not bad, (
        f"{path.name} ends a `!` command with `&`, which every kernel except "
        f"Colab's rejects with OSError('Background processes not supported.'). "
        f"Use subprocess.Popen instead:\n  " + "\n  ".join(bad)
    )


def test_the_pattern_matches_the_shape_it_is_meant_to_catch():
    """Guards the rule itself, so a regex edit cannot silently disarm it."""
    launch = ("!nohup python -m sglang.launch_server --model-path m "
              "--port 8000 > sglang.log &")
    assert _BACKGROUNDED.match(launch)
    assert _BACKGROUNDED.match("!python server.py &  ")
    assert not _BACKGROUNDED.match("!pip install a && pip install b")
    assert not _BACKGROUNDED.match("!pip install sglang")
    assert not _BACKGROUNDED.match("server = subprocess.Popen([...])  # &")


def test_a_live_kernel_rejects_the_background_form_and_accepts_popen():
    """Measured, not read off the IPython source, and on this platform.

    Runs on Linux, macOS and Windows: the child is `sys.executable`, so nothing
    here depends on a POSIX shell or on `sleep` existing.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    source = "\n".join([
        "import IPython, subprocess, sys",
        "ip = IPython.get_ipython()",
        "assert getattr(ip.system, '__name__', '') == 'system_piped', ip.system",
        "try:",
        "    ip.system('%s -c pass &' % sys.executable)",
        "    OLD = 'accepted'",
        "except OSError as exc:",
        "    OLD = 'OSError: %s' % exc",
        "child = subprocess.Popen([sys.executable, '-c', 'pass'])",
        "NEW = 'Popen rc=%s' % child.wait()",
        "print(OLD)",
        "print(NEW)",
    ])
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source)])
    nbclient.NotebookClient(
        notebook, timeout=120, allow_errors=True, kernel_name="python3").execute()

    outputs = notebook.cells[0].get("outputs", [])
    errors = [o for o in outputs if o.get("output_type") == "error"]
    assert not errors, f"probe cell raised: {errors[0].get('evalue')}"
    text = "".join(o.get("text", "") for o in outputs
                   if o.get("output_type") == "stream")
    assert "OSError: Background processes not supported." in text, text
    assert "Popen rc=0" in text, text
