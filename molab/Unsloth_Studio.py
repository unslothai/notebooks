# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
# ]
#
# [tool.uv]
# no-build-package = [
#     "bitsandbytes",
#     "triton",
#     "vllm",
#     "xformers",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run this, hit the **▶ Run all** button in the bottom-right corner - or use `Ctrl/Cmd + Shift + R`.
    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    To install Unsloth Studio on your local device, follow [our guide](https://unsloth.ai/docs/new/unsloth-studio/install). Unsloth Studio is licensed [AGPL-3.0](https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0).

    ### Unsloth Studio

    Train and run open models with [**Unsloth Studio**](https://unsloth.ai/docs/new/unsloth-studio/start). NEW! Installation should now only take 2 mins!

    [Features](https://unsloth.ai/docs/new/unsloth-studio#features) • [Quickstart](https://unsloth.ai/docs/new/unsloth-studio/start) • [Data Recipes](https://unsloth.ai/docs/new/unsloth-studio/data-recipe) • [Studio Chat](https://unsloth.ai/docs/new/unsloth-studio/chat) • [Export](https://unsloth.ai/docs/new/unsloth-studio/export)

    <p align="left"><img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/studio%20github%20landscape%20colab%20display.png" width="600"></p>
    """)
    return


@app.cell
def _():
    import os
    import pathlib
    import re
    import stat
    import subprocess
    import sys
    import tarfile
    import time
    import urllib.request

    # Grab the repo if it isn't here yet.
    if not pathlib.Path("unsloth").exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                "https://github.com/unslothai/unsloth.git",
            ],
            check=True,
        )
    repo = pathlib.Path("unsloth").resolve()

    # The UI ships unbuilt and there's no Node here, so fetch one to build with.
    _node = pathlib.Path("node-v22.12.0-linux-x64")
    if not _node.exists():
        _tar = pathlib.Path(f"{_node}.tar.xz")
        urllib.request.urlretrieve(
            f"https://nodejs.org/dist/v22.12.0/{_node}.tar.xz", _tar
        )
        with tarfile.open(_tar) as _t:
            _t.extractall()
    os.environ["PATH"] = (
        str((_node / "bin").resolve()) + os.pathsep + os.environ["PATH"]
    )

    # Build the UI and install the stack into the system Python. The env var
    # picks the no-venv path; drop it when setup.sh learns this host.
    subprocess.run(
        "chmod +x studio/setup.sh && ./studio/setup.sh --local",
        shell=True,
        check=True,
        cwd=str(repo),
        env={**os.environ, "COLAB_RELEASE_TAG": "molab"},
    )
    return os, pathlib, re, repo, stat, subprocess, sys, time, urllib


@app.cell
def _(os, pathlib, re, repo, stat, subprocess, sys, time, urllib):
    # Relax the server's frame headers before it starts so the page can
    # embed it below. Drop this once the backend reads UNSLOTH_STUDIO_EMBED.
    os.environ["UNSLOTH_STUDIO_EMBED"] = "1"
    sys.path.insert(0, str((repo / "studio" / "backend").resolve()))
    import main as _m  # noqa: E402

    _m._IS_COLAB = True
    from run import run_server  # noqa: E402

    run_server(
        host="0.0.0.0",
        port=8888,
        frontend_path=repo / "studio" / "frontend" / "dist",
        silent=True,
    )
    for _ in range(60):  # give the server a moment to come up
        try:
            urllib.request.urlopen(
                "http://localhost:8888/api/health", timeout=2
            ).close()
            break
        except Exception:
            time.sleep(1)

    # Reach the server from the browser through a cloudflared quick tunnel
    # (a public *.trycloudflare.com URL).
    _cf = pathlib.Path("cloudflared")
    if not _cf.exists():
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest"
            "/download/cloudflared-linux-amd64",
            _cf,
        )
    _cf.chmod(_cf.stat().st_mode | stat.S_IEXEC)
    _proc = subprocess.Popen(  # full path, else it won't be found
        [str(_cf.resolve()), "tunnel", "--url", "http://localhost:8888"],
        stderr=subprocess.PIPE,
        text=True,
    )
    studio_url = None
    for _line in _proc.stderr:
        _hit = re.search(r"https://[\w-]+\.trycloudflare\.com", _line)
        if _hit:
            studio_url = _hit.group(0)
            break
        if _proc.poll() is not None:
            break
    if not studio_url:
        raise RuntimeError("cloudflared did not return a tunnel URL")
    for _ in range(20):  # the tunnel goes live a few seconds later
        try:
            urllib.request.urlopen(studio_url, timeout=5).close()
            break
        except Exception:
            time.sleep(2)
    return (studio_url,)


@app.cell(hide_code=True)
def _(mo, studio_url):
    mo.vstack(
        [
            mo.md(
                f"### 🦥 Unsloth Studio is live\n\n"
                f"**[↗ Open in a new tab]({studio_url})**. Sign in as `unsloth`; "
                f"your password is in `.unsloth/studio/auth/.bootstrap_password`."
            ),
            mo.Html(
                f'<iframe src="{studio_url}" width="100%" height="820px"'
                ' allow="clipboard-read; clipboard-write"'
                ' style="border:none;"></iframe>'
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

    Some other resources:
    1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
    2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
    3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
    4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
    5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

      Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
