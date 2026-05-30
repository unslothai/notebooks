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
#     "transformers==4.56.2",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo",
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
    To run this notebook, hit the **▶ Run all** button in the bottom-right corner - or use `Ctrl/Cmd + Shift + R`.
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
    from unsloth import FastVisionModel  # FastLanguageModel for LLMs
    import torch

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",  # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
        "unsloth/Pixtral-12B-2409-bnb-4bit",  # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",  # Pixtral base model
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",  # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",  # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2-VL-7B-Instruct",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    return FastVisionModel, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

    **[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!
    """)
    return


@app.cell
def _(FastVisionModel, model):
    model_1 = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We'll be using a sampled dataset of handwritten maths formulas. The goal is to convert these images into a computer readable form - ie in LaTeX form, so we can render it. This can be very useful for complex formulas.

    You can access the dataset [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR). The full dataset is [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take an overview look at the dataset. We shall see what the 3rd image is, and what caption it had.
    """)
    return


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    dataset[2]["image"]
    return


@app.cell
def _(dataset):
    dataset[2]["text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also render the LaTeX in the browser directly!
    """)
    return


@app.cell
def _(dataset):
    from IPython.display import display, Math, Latex

    latex = dataset[2]["text"]
    display(Math(latex))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To format the dataset, all vision finetuning tasks should be formatted as follows:

    ```python
    [
    { "role": "user",
      "content": [{"type": "text",  "text": Q}, {"type": "image", "image": image} ]
    },
    { "role": "assistant",
      "content": [{"type": "text",  "text": A} ]
    },
    ]
    ```
    """)
    return


@app.cell
def _():
    instruction = "Write the LaTeX representation for this image."

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        return {"messages": conversation}

    pass
    return (convert_to_conversation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's convert the dataset into the "correct" format for finetuning:
    """)
    return


@app.cell
def _(convert_to_conversation, dataset):
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    return (converted_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We look at how the conversations are structured for the first example:
    """)
    return


@app.cell
def _(converted_dataset):
    converted_dataset[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's first see before we do any finetuning what the model outputs for the first example!
    """)
    return


@app.cell
def _(FastVisionModel, dataset, model_1, tokenizer):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image = dataset[2]["image"]
    instruction_1 = "Write the LaTeX representation for this image."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_1}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!!

    We use our new `UnslothVisionDataCollator` which will help in our vision finetuning setup.
    """)
    return


@app.cell
def _(FastVisionModel, converted_dataset, model_1, tokenizer):
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model_1)
    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model_1, tokenizer),  # Must use!
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=0.0002,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # For Weights and Biases
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )
    return (trainer,)


@app.cell
def _(torch):
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return max_memory, start_gpu_memory


@app.cell
def _(trainer):
    trainer_stats = trainer.train()
    return (trainer_stats,)


@app.cell
def _(max_memory, start_gpu_memory, torch, trainer_stats):
    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Let's run the model! You can change the instruction and input - leave the output blank!

    We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.
    """)
    return


@app.cell
def _(FastVisionModel, TextStreamer, dataset, model_1, tokenizer):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image_1 = dataset[2]["image"]
    instruction_2 = "Write the LaTeX representation for this image."
    messages_1 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_2}],
        }
    ]
    input_text_1 = tokenizer.apply_chat_template(messages_1, add_generation_prompt=True)
    inputs_1 = tokenizer(
        image_1, input_text_1, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    text_streamer_1 = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_1,
        streamer=text_streamer_1,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    model_1.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, dataset, model_1, tokenizer):
    if False:
        from unsloth import FastVisionModel as _FastVisionModel

        _model, _tokenizer = _FastVisionModel.from_pretrained(
            model_name="qwen_lora", load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
        _FastVisionModel.for_inference(_model)
    image_2 = dataset[0]["image"]
    instruction_3 = "Write the LaTeX representation for this image."
    messages_2 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_3}],
        }
    ]
    input_text_2 = tokenizer.apply_chat_template(messages_2, add_generation_prompt=True)
    inputs_2 = tokenizer(
        image_2, input_text_2, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    text_streamer_2 = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_2,
        streamer=text_streamer_2,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Select ONLY 1 to save! (Both not needed!)
    if False:
        # Save locally to 16bit
        model_1.save_pretrained_merged("unsloth_finetune", tokenizer)
    if False:
        # To export and save to your Hugging Face account
        model_1.push_to_hub_merged(
            "YOUR_USERNAME/unsloth_finetune", tokenizer, token="YOUR_HF_TOKEN"
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
