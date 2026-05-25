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
#     "transformers==5.3.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
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
    To run this, press the **Run** button beside each cell on your A100 molab Pro instance!
    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    To install Unsloth your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

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
    Goal: To convert `unsloth/Qwen3-30B-A3B-Instruct-2507` into a reasoning model via GRPO by using OpenR1's Math dataset.

    We first pre fine-tune the model to make GRPO skip trying to match formatting - this speeds GRPO up.
    """)
    return


@app.cell
def _():
    import os, torch

    # os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm" # switch to 'unsloth_triton' or 'native_torch'
    # grouped_mm is only supported on torch 2.9 or newer.
    # Make sure that we have at least 64GB VRAM because the model itself takes 60GB in 16bit
    return (torch,)


@app.cell
def _():
    model_name = "unsloth/Qwen3-30B-A3B-Instruct-2507"  # This is a very big model, might take a while for downloading
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower
    return lora_rank, max_seq_length, model_name


@app.cell
def _(lora_rank, max_seq_length, model_name):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,  # Can increase for longer reasoning traces
        load_in_4bit=False,
        fast_inference=False,  # Not supported for MoE (yet!)
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing=True,  # Reduces memory usage
        random_state=3407,
        bias="none",
    )
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GRPO chat template
    Since we're using a base model, we should set a chat template. You can make your own chat template as well!
    1. DeepSeek uses `<think>` and `</think>`, but this is **not** necessary - you can customize it however you like!
    2. A `system_prompt` is recommended to at least guide the model's responses.
    """)
    return


@app.cell
def _():
    reasoning_start = "<start_working_out>"  # Acts as think-open tag
    reasoning_end = "<end_working_out>"  # Acts as think-close tag
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""
    system_prompt
    return (
        reasoning_end,
        reasoning_start,
        solution_end,
        solution_start,
        system_prompt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a simple chat template below. Notice `add_generation_prompt` includes prepending `<start_working_out>` to guide the model to start its reasoning process.
    """)
    return


@app.cell
def _(reasoning_start, system_prompt, tokenizer):
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '{system_prompt}' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
        "{% endif %}"
    )

    # Replace with our specific template:
    chat_template = chat_template.replace(
        "'{system_prompt}'", f"'{system_prompt}'"
    ).replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how our chat template behaves on an example:
    """)
    return


@app.cell
def _(reasoning_end, reasoning_start, solution_end, solution_start, tokenizer):
    tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant",
                "content": f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}",
            },
            {"role": "user", "content": "What is 2+2?"},
        ],
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pre fine-tuning for formatting
    We now use a subset of NVIDIA's [Open Math Reasoning dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning) which was filtered to only include high quality DeepSeek R1 traces.

    We'll only filter ~59 or so examples to first "prime" / pre fine-tune the model to understand our custom GRPO formatting.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset
    import pandas as pd
    import numpy as np

    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

    # Try converting to number - if not, replace with NaN
    is_number = pd.to_numeric(
        pd.Series(dataset["expected_answer"]), errors="coerce"
    ).notnull()
    # Select only numbers
    dataset = dataset.iloc[np.where(is_number)[0]]

    dataset
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have to format the dataset to follow our GRPO style formatting:
    """)
    return


@app.cell
def _(
    dataset,
    reasoning_end,
    reasoning_start,
    solution_end,
    solution_start,
    system_prompt,
):
    def format_dataset(x):
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # Remove generated think tags
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # Strip newlines on left and right
        thoughts = thoughts.strip()
        # Add our custom formatting
        final_prompt = (
            reasoning_start
            + thoughts
            + reasoning_end
            + solution_start
            + expected_answer
            + solution_end
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]

    dataset["Messages"] = dataset.apply(format_dataset, axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check to see if it worked:
    """)
    return


@app.cell
def _(dataset, tokenizer):
    tokenizer.apply_chat_template(dataset["Messages"][0], tokenize=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's truncate the pre fine-tuning dataset to `max_seq_length/2` since we don't want too long reasoning traces.

    Note this might take 2 minutes!
    """)
    return


@app.cell
def _(dataset, max_seq_length, tokenizer):
    dataset["N"] = dataset["Messages"].apply(
        lambda x: len(tokenizer.apply_chat_template(x)["input_ids"])
    )
    dataset_1 = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()
    dataset_1.shape
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then tokenize the messages and convert it to a Hugging Face compatible dataset format:
    """)
    return


@app.cell
def _(dataset_1, tokenizer):
    from datasets import Dataset

    dataset_1["text"] = tokenizer.apply_chat_template(
        dataset_1["Messages"].values.tolist(), tokenize=False
    )
    dataset_2 = Dataset.from_pandas(dataset_1)
    dataset_2
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now pre fine-tune the model so it follows our custom GRPO formatting!
    """)
    return


@app.cell
def _(dataset_2, model, tokenizer):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_2,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,  # Use GA to mimic batch size!
            warmup_steps=5,
            max_steps=50,
            learning_rate=0.0002,  # Reduce to 2e-5 for long training runs
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check if the model has learnt to follow the custom format:
    """)
    return


@app.cell
def _(dataset_2, model, tokenizer):
    text = tokenizer.apply_chat_template(
        dataset_2[0]["Messages"][:2], tokenize=False, add_generation_prompt=True
    )
    from transformers import TextStreamer

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=0.1,
        max_new_tokens=128,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
        use_cache=True,
    )  # Must add for generation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Yes it did follow the formatting! Great! Let's remove some items before the GRPO step
    """)
    return


@app.cell
def _(dataset_2, torch):
    del dataset_2
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model, tokenizer):
    # Merge to 16bit
    if False:
        model.save_pretrained_merged(
            "qwen_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "qwen_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("qwen_lora")
        tokenizer.save_pretrained("qwen_lora")
    if False:
        model.push_to_hub("HF_USERNAME/qwen_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/qwen_lora", token="YOUR_HF_TOKEN")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

    Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
    * `q8_0` - Fast conversion. High resource use, but generally acceptable.
    * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
    * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

    [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    """)
    return


@app.cell
def _(model, tokenizer):
    # Save to 8bit Q8_0
    if False:
        model.save_pretrained_gguf(
            "qwen_finetune",
            tokenizer,
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )

    # Save to 16bit GGUF
    if False:
        model.save_pretrained_gguf(
            "qwen_finetune", tokenizer, quantization_method="f16"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )

    # Save to q4_k_m GGUF
    if False:
        model.save_pretrained_gguf(
            "qwen_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",  # Change hf to your username!
            tokenizer,
            quantization_method=[
                "q4_k_m",
                "q8_0",
                "q5_k_m",
            ],
            token="YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `qwen_finetune.Q8_0.gguf` file or `qwen_finetune.Q4_K_M.gguf` file in llama.cpp.

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
