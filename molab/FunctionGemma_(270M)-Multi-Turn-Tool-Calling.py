# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "hf_transfer",
#     "huggingface_hub>=0.34.0",
#     "marimo",
#     "peft",
#     "protobuf",
#     "sentencepiece",
#     "torchao>=0.16.0",
#     "transformers>=4.56.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth.git",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
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
    <a href="https://github.com/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-Multi-Turn-Tool-Calling.ipynb" target="_parent"><img src="https://marimo.io/molab-shield.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run this, press the **Run** button beside each cell!
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


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 4096  # Can choose any sequence length!
    fourbit_models = [
        # 4bit Gemma 3 dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        # Function Gemma models
        "unsloth/functiongemma-270m-it",
        "unsloth/functiongemma-270m-it-unsloth-bnb-4bit",
        "unsloth/functiongemma-270m-it-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/functiongemma-270m-it",
        max_seq_length=max_seq_length,  # Choose any for long context!
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        load_in_16bit=True,  # [NEW!] Enables 16bit LoRA
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define tools for FunctionGemma

    We'll define multiple tools which FunctionGemma can call, including the below:
    ```
    get_today_date
    get_current_weather
    add_numbers
    multiply_numbers
    ```
    """)
    return


@app.cell
def _():
    def get_today_date():
        """
        Gets today's date

        Returns:
            today_date: Today's date in format 18 December 2025
        """
        from datetime import datetime

        today_date = datetime.today().strftime("%d %B %Y")
        return {"today_date": today_date}

    def get_current_weather(location: str, unit: str = "celsius"):
        """
        Gets the current weather in a given location.

        Args:
            location: The city and state, e.g. "San Francisco, CA, USA" or "Sydney, Australia"
            unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

        Returns:
            temperature: The current temperature in the given location
            weather: The current weather in the given location
        """
        if "San Francisco" in location.title():
            return {"temperature": 15, "weather": "sunny"}
        elif "Sydney" in location.title():
            return {"temperature": 25, "weather": "cloudy"}
        else:
            return {"temperature": 30, "weather": "rainy"}

    def add_numbers(x: float | str, y: float | str):
        """
        Adds 2 numbers together

        Args:
            x: First number
            y: Second number

        Returns:
            result: x + y
        """
        return {"result": float(x) + float(y)}

    def multiply_numbers(x: float | str, y: float | str):
        """
        Multiplies 2 numbers together

        Args:
            x: First number
            y: Second number

        Returns:
            result: x * y
        """
        return {"result": float(x) * float(y)}

    return add_numbers, get_current_weather, get_today_date, multiply_numbers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We save all the functions to a mapping and as tools
    """)
    return


@app.cell
def _(add_numbers, get_current_weather, get_today_date, multiply_numbers):
    FUNCTION_MAPPING = {
        "get_today_date": get_today_date,
        "get_current_weather": get_current_weather,
        "add_numbers": add_numbers,
        "multiply_numbers": multiply_numbers,
    }
    TOOLS = list(FUNCTION_MAPPING.values())
    return FUNCTION_MAPPING, TOOLS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then make some parsing code for FunctionGemma which we hide
    """)
    return


@app.cell
def _(FUNCTION_MAPPING, TOOLS, tokenizer):
    # @title FunctionGemma parsing code (expandible)
    import re

    def extract_tool_calls(text):
        def cast(v):
            try:
                return int(v)
            except:
                try:
                    return float(v)
                except:
                    return {"true": True, "false": False}.get(v.lower(), v.strip("'\""))

        return [
            {
                "name": name,
                "arguments": {
                    k: cast((v1 or v2).strip())
                    for k, v1, v2 in re.findall(
                        r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args
                    )
                },
            }
            for name, args in re.findall(
                r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>",
                text,
                re.DOTALL,
            )
        ]

    def process_tool_calls(output, messages):
        calls = extract_tool_calls(output)
        if not calls:
            return messages
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": call} for call in calls
                ],
            }
        )
        results = [
            {
                "name": c["name"],
                "response": FUNCTION_MAPPING[c["name"]](**c["arguments"]),
            }
            for c in calls
        ]
        messages.append({"role": "tool", "content": results})
        return messages

    def _do_inference(model, messages, max_new_tokens=128):
        inputs = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        output = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

        out = model.generate(
            **inputs.to(model.device),
            max_new_tokens=max_new_tokens,
            top_p=0.95,
            top_k=64,
            temperature=1.0,
        )
        generated_tokens = out[0][len(inputs["input_ids"][0]) :]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def do_inference(model, messages, print_assistant=True, max_new_tokens=128):
        output = _do_inference(model, messages, max_new_tokens=max_new_tokens)
        messages = process_tool_calls(output, messages)
        if messages[-1]["role"] == "tool":
            output = _do_inference(model, messages, max_new_tokens=max_new_tokens)
        messages.append({"role": "assistant", "content": output})
        if print_assistant:
            print(output)
        return messages

    return (do_inference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's call the model!
    """)
    return


@app.cell
def _(do_inference, model):
    messages = []
    messages.append({"role": "user", "content": "What's today's date?"})
    messages = do_inference(model, messages, max_new_tokens=128)
    return (messages,)


@app.cell
def _(do_inference, messages, model):
    messages.append(
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    )
    messages_1 = do_inference(model, messages, max_new_tokens=128)
    return (messages_1,)


@app.cell
def _(do_inference, messages_1, model):
    messages_1.append(
        {"role": "user", "content": "What's the weather like in Sydney, Australia?"}
    )
    messages_2 = do_inference(model, messages_1, max_new_tokens=128)
    return (messages_2,)


@app.cell
def _(do_inference, messages_2, model):
    messages_2.append({"role": "user", "content": "Add 112358 and 123456"})
    messages_3 = do_inference(model, messages_2, max_new_tokens=128)
    return (messages_3,)


@app.cell
def _(do_inference, messages_3, model):
    messages_3.append({"role": "user", "content": "Multiply 112358 and 123456"})
    messages_4 = do_inference(model, messages_3, max_new_tokens=128)
    return (messages_4,)


@app.cell
def _(do_inference, messages_4, model):
    messages_4.append({"role": "user", "content": "Do the addition of 2 and 231.111"})
    messages_5 = do_inference(model, messages_4, max_new_tokens=128)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

    Some other resources:
    1. Train your own reasoning model - Llama GRPO notebook [Open in molab](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    2. Saving finetunes to Ollama. [Free notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    3. Llama 3.2 Vision finetuning - Radiography use case. [Open in molab](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
    4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

      Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
    </div>

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
    """)
    return


if __name__ == "__main__":
    app.run()
