# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "sglang[all]==0.5.16",
#     "torchaudio==2.11.0",
#     "torchvision==0.26.0",
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
    To run this notebook on your A100 molab Pro instance, hit the **▶ Run all** button in the bottom-right corner - or use `Ctrl/Cmd + Shift + R`.
    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

    You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### News
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Introducing **[Unsloth Desktop](https://unsloth.ai/docs/desktop)**, the first desktop app to run and train models. Free and open-source for macOS, Windows and Linux. [GitHub](https://github.com/unslothai/unsloth) • [Download](https://unsloth.ai/download)

    <a href="https://unsloth.ai/docs/desktop"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/unsloth-qwen3-8.png" width="350" alt="Introducing Unsloth Desktop"></a>

    Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)

    Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)

    New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)

    Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Launch sglang inference for unsloth/gemma-3n-E2B-it (https://huggingface.co/unsloth/gemma-3n-E2B-it)
    """)
    return


@app.cell
def _():
    import subprocess

    # sglang publishes no Turing kernels. sgl-kernel dropped its compute_75
    # gencode flag in sgl-project/sglang#9207, so the sglang-kernel 0.4.5 that
    # sglang 0.5.16 pins carries SASS for sm_80, sm_89, sm_90, sm_90a, sm_100a,
    # sm_103a and sm_120a, and no PTX to JIT a Turing kernel from. On a T4
    # (sm_75) `import sgl_kernel` raises while the server is still parsing its
    # arguments, and all this cell would report is "Server process exited with
    # code 1". Check the GPU first and name the fix.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "sglang needs an NVIDIA GPU, and this session has none. Switch the "
            "runtime to one with an L4 or A100 attached."
        )

    _capability = torch.cuda.get_device_capability()
    if _capability < (8, 0):
        raise RuntimeError(
            f"sglang cannot run on this GPU: {torch.cuda.get_device_name(0)} is "
            f"compute capability {_capability[0]}.{_capability[1]}, and the "
            "sglang-kernel wheels carry kernels for compute capability 8.0 and "
            "newer only. Switch the runtime to an Ampere or newer GPU, such as "
            "an L4 or an A100."
        )

    # Load and run the model using sglang.
    #
    # Popen, not `!... &`: IPython's `system_piped`, which every kernel except
    # molab's uses, raises OSError on a trailing `&`, so on Kaggle, plain Jupyter
    # or papermill this cell could never run.
    # Backend left to sglang: `fa3` is Hopper (sm90) only, so it fails on the
    # T4 / L4 / A100 a session actually hands out.
    import glob, os, subprocess, sys
    from sglang.utils import wait_for_server

    # sglang-kernel's common_ops library links libnvrtc.so.13 and carries no
    # RUNPATH, so it loads only if that exact file is already in the process or
    # on the loader path. torch preloads one libnvrtc by absolute path
    # (`_preload_cuda_deps` in torch/__init__.py) and looks in
    # `nvidia/cuda_nvrtc/lib` before `nvidia/cu13/lib`. A session that still
    # carries the CUDA 12 `nvidia-cuda-nvrtc-cu12` wheel beside the CUDA 13 one
    # -- which molab does, because installing sglang replaces torch but leaves
    # the old torch's NVIDIA wheels installed -- therefore preloads
    # libnvrtc.so.12 and never .so.13, and the server exits inside
    # `import sgl_kernel` with "libnvrtc.so.13: cannot open shared object
    # file". Hand the child the directory holding the libnvrtc that matches
    # this torch.
    _site_packages = os.path.dirname(os.path.dirname(torch.__file__))
    _cuda_major = (torch.version.cuda or "").split(".")[0]
    _nvrtc_dirs = (
        sorted(
            {
                os.path.dirname(_path)
                for _path in glob.glob(
                    os.path.join(
                        _site_packages,
                        "nvidia",
                        "*",
                        "lib",
                        f"libnvrtc.so.{_cuda_major}",
                    )
                )
            }
        )
        if _cuda_major
        else []
    )

    _env = dict(os.environ)
    if _nvrtc_dirs:
        _previous = _env.get("LD_LIBRARY_PATH", "")
        _env["LD_LIBRARY_PATH"] = os.pathsep.join(
            _nvrtc_dirs + ([_previous] if _previous else [])
        )

    log = open("sglang.log", "w")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "unsloth/gemma-3n-E2B-it",
            "--port",
            "8000",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=_env,
    )

    # Both arguments matter. `wait_for_server` defaults to timeout = None, which is
    # wait forever, so without one this is the same unbounded hang as the shell
    # `while ! grep -q` loop it replaces. `process` makes it poll the subprocess and
    # raise as soon as a failed launch exits, instead of waiting out the timeout.
    try:
        wait_for_server("http://localhost:8000", timeout=900, process=server)
    except Exception:
        # On the timeout path the server is still alive, and the kernel outlives the
        # cell, so re-raising alone would leave it holding the GPU and port 8000 and
        # the retry would fail on address-in-use rather than on the real problem.
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        # The server's own log is the only thing that says why it did not start.
        print(open("sglang.log").read()[-4000:])
        raise
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image helper functions
    """)
    return


@app.cell
def _():
    from PIL.ImageFile import ImageFile
    from PIL import Image
    import numpy as np
    import io
    import base64
    import requests
    from io import BytesIO

    def load_image_from_url(url):
        response = requests.get(url, timeout=60)
        # A moved or deleted file still answers, with a 404 whose body is text.
        # `Image.open` on that reports "cannot identify image file <BytesIO ...>",
        # which names neither the URL nor the status, so check the response.
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(
                f"{url} answered with Content-Type {content_type!r}, not an "
                f"image. The link has most likely moved or been deleted."
            )
        img = Image.open(BytesIO(response.content))
        return img

    def process_image(image: ImageFile) -> str:
        """Process image for sglang gemma3n and return base64 string."""
        assert isinstance(image, ImageFile), "please pass an image object"

        # Resize the image
        resized_image = image.resize((384, 384))

        # Convert to numpy array and transpose
        image_array = np.array(resized_image)
        array_reordered = np.transpose(image_array, (1, 0, 2))

        # Convert back to PIL Image
        processed_image = Image.fromarray(array_reordered)

        # Convert to base64 string
        image_bytes = io.BytesIO()
        processed_image.save(image_bytes, format=image.format)
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

        # Return data URL string
        format_name = image.format.lower() if image.format else "png"
        return f"data:image/{format_name};base64,{base64_image}"

    return load_image_from_url, process_image, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gemma3n Inference using sglang (source model: https://huggingface.co/unsloth/gemma-3n-E2B-it)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference 1
    Image source file "https://raw.githubusercontent.com/sgl-project/sglang/196b940aed024fd4a072dcae9f96d3ce153e57ed/examples/assets/example_image.png"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    load image from url source
    """)
    return


@app.cell
def _(load_image_from_url):
    from IPython.display import display

    image = load_image_from_url(
        "https://raw.githubusercontent.com/sgl-project/sglang/196b940aed024fd4a072dcae9f96d3ce153e57ed/examples/assets/example_image.png"
    )
    display(image)
    return display, image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's run the model!
    """)
    return


@app.cell
def _(image, process_image, requests):
    from sglang.utils import print_highlight, terminate_process

    processed_image = process_image(image)
    url = f"http://localhost:8000/v1/chat/completions"
    processed_image = process_image(image)
    data = {
        "model": "unsloth/gemma-3n-E2B-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": processed_image}},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(url, json=data)
    print_highlight(response.text)
    return (print_highlight,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference 2
    Image source file "https://i.ibb.co/1tw5whfz/ocr.png"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    load image from url source
    """)
    return


@app.cell
def _(display, load_image_from_url):
    image_1 = load_image_from_url("https://i.ibb.co/1tw5whfz/ocr.png")
    display(image_1)
    return (image_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's run the model!
    """)
    return


@app.cell
def _(image_1, print_highlight, process_image, requests):
    url_1 = f"http://localhost:8000/v1/chat/completions"
    processed_image_1 = process_image(image_1)
    data_1 = {
        "model": "unsloth/gemma-3n-E2B-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read the text in the image"},
                    {"type": "image_url", "image_url": {"url": processed_image_1}},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response_1 = requests.post(url_1, json=data_1)
    print_highlight(response_1.text)
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
