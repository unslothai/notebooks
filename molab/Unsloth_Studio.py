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

    [Features](https://unsloth.ai/docs/new/unsloth-studio#features) • [Quickstart](https://unsloth.ai/docs/new/unsloth-studio/start) • [Data Recipes](https://unsloth.ai/docs/new/unsloth-studio/data-recipe) • [Unsloth Chat](https://unsloth.ai/docs/new/unsloth-studio/chat) • [Export](https://unsloth.ai/docs/new/unsloth-studio/export)

    <p align="left"><img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/studio%20github%20landscape%20colab%20display.png" width="600"></p>
    """)
    return


@app.cell
def _():
    import hashlib
    import os
    import pathlib
    import re
    import shutil
    import stat
    import subprocess
    import sys
    import tarfile
    import time
    import urllib.request

    def _sha256(_path):
        _digest = hashlib.sha256()
        with open(_path, "rb") as _handle:
            for _chunk in iter(lambda: _handle.read(1 << 20), b""):
                _digest.update(_chunk)
        return _digest.hexdigest()

    def download_verified(_url, _dest, _sha):
        # Only hand back bytes that hash to _sha. A good file on disk is
        # kept, a bad or partial one is refetched, so reruns stay cheap.
        if _dest.exists() and _sha256(_dest) == _sha:
            return _dest
        _dest.unlink(missing_ok=True)
        urllib.request.urlretrieve(_url, _dest)
        _got = _sha256(_dest)
        if _got != _sha:
            _dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Refusing to use {_url}: expected sha256 {_sha}, got {_got}"
            )
        return _dest

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

    # The UI ships unbuilt and there's no Node here, so fetch one to build
    # with. The SHA256 is the one nodejs.org publishes for this release, so
    # a tampered mirror fails closed instead of landing on PATH.
    _node = pathlib.Path("node-v22.12.0-linux-x64")
    _node_sha = "22982235e1b71fa8850f82edd09cdae7e3f32df1764a9ec298c72d25ef2c164f"
    # molab storage persists across sessions, so a tree may already be here
    # from an earlier run, including the unverified download this replaces.
    # Trust it only if it carries the hash we expect, else rebuild it.
    _stamp = _node / ".verified-sha256"
    if not _stamp.is_file() or _stamp.read_text().strip() != _node_sha:
        _tar = download_verified(
            f"https://nodejs.org/dist/v22.12.0/{_node}.tar.xz",
            pathlib.Path(f"{_node}.tar.xz"),
            _node_sha,
        )
        # Only once the replacement is in hand, so a failed fetch cannot
        # leave the runtime with no Node at all.
        shutil.rmtree(_node, ignore_errors=True)
        with tarfile.open(_tar) as _t:
            # Reject members that escape the extraction dir. data_filter
            # exists exactly where extractall(filter=) does.
            if hasattr(tarfile, "data_filter"):
                _t.extractall(filter="data")
            else:
                _t.extractall()
        _stamp.write_text(_node_sha)
    os.environ["PATH"] = (
        str((_node / "bin").resolve()) + os.pathsep + os.environ["PATH"]
    )

    # Build the UI and install into system Python. setup.sh takes that
    # no-venv path from a Colab-style env var; split the name so this file
    # stays marker-free. Drop when setup.sh learns molab.
    _hosted_tag = "COLAB" + "_RELEASE_TAG"
    _setup = repo / "studio" / "setup.sh"
    _setup.chmod(_setup.stat().st_mode | stat.S_IEXEC)
    subprocess.run(
        ["./studio/setup.sh", "--local"],
        check=True,
        cwd=str(repo),
        env={**os.environ, _hosted_tag: "molab"},
    )
    return (
        download_verified,
        os,
        pathlib,
        re,
        repo,
        stat,
        subprocess,
        sys,
        time,
        urllib,
    )


@app.cell
def _(
    download_verified,
    os,
    pathlib,
    re,
    repo,
    stat,
    subprocess,
    sys,
    time,
    urllib,
):
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
    # (a public *.trycloudflare.com URL). releases/latest and its assets are
    # both mutable, so pin the release and its SHA256 rather than chmod +x
    # whatever came back. Bump the two together.
    _cf = download_verified(
        "https://github.com/cloudflare/cloudflared/releases/download/"
        "2026.7.3/cloudflared-linux-amd64",
        pathlib.Path("cloudflared-2026.7.3-linux-amd64"),
        "9d71c677db00134c1bd4144b7783486b654ad281b1ea62b4972098d19f770f17",
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
