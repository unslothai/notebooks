# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "sglang[all]==0.5.16",
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
    Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://github.com/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)

    <table><tr>
    <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
    <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
    </tr></table>

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
    # Load and run the model using sglang.
    #
    # Popen, not `!... &`: IPython's `system_piped`, which every kernel except
    # molab's uses, raises OSError on a trailing `&`, so on Kaggle, plain Jupyter
    # or papermill this cell could never run.
    # Backend left to sglang: `fa3` is Hopper (sm90) only, so it fails on the
    # T4 / L4 / A100 a session actually hands out.
    import subprocess, sys
    from sglang.utils import wait_for_server

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
        response = requests.get(url)
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
    Image source file "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/test/lang/example_image.png"
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
        "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/test/lang/example_image.png"
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
