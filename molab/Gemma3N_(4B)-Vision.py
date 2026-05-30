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
#     "timm",
#     "torchao>=0.16.0",
#     "torchcodec",
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


@app.cell
def _():
    # Only for Gemma 3N
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

    model, processor = FastVisionModel.from_pretrained(
        "unsloth/gemma-3n-E4B",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    return FastVisionModel, model, processor, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters for parameter efficient fine-tuning, allowing us to train only 1% of all model parameters efficiently.

    **[NEW]** We also support fine-tuning only the vision component, only the language component, or both. Additionally, you can choose to fine-tune the attention modules, the MLP layers, or both!
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
        r=32,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=32,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        target_modules="all-linear",  # Optional now! Can specify a list if needed
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We'll use a sampled dataset of handwritten math formulas. The objective is to convert these images into a computer-readable format—specifically LaTeX—so they can be rendered. This is particularly useful for complex expressions.

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
    Let's take an overview of the dataset. We'll examine the second image and its corresponding caption.
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
    We can also render LaTeX directly in the browser!
    """)
    return


@app.cell
def _(dataset):
    from IPython.display import display, Math, Latex

    latex = dataset[3]["text"]
    display(Math(latex))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To format the dataset, all vision fine-tuning tasks should follow this format:

    ```python
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
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
    The first example is now structured like below:
    """)
    return


@app.cell
def _(converted_dataset):
    converted_dataset[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets take the Gemma 3n instruction chat template and use it in our base model
    """)
    return


@app.cell
def _(processor):
    from unsloth import get_chat_template

    processor_1 = get_chat_template(processor, "gemma-3n")
    return (processor_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before fine-tuning, let us evaluate the base model's performance. We do not expect strong results, as it has not encountered this chat template before.
    """)
    return


@app.cell
def _(FastVisionModel, dataset, model_1, processor_1):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image = dataset[2]["image"]
    instruction_1 = "Write the LaTeX representation for this image."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_1}],
        }
    ]
    input_text = processor_1.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor_1(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    from transformers import TextStreamer

    text_streamer = TextStreamer(processor_1, skip_prompt=True)
    result = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can see it's absolutely terrible! It doesn't follow instructions at all
    """)
    return


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
def _(FastVisionModel, converted_dataset, model_1, processor_1):
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model_1)
    trainer = SFTTrainer(
        model=model_1,
        train_dataset=converted_dataset,
        processing_class=processor_1.tokenizer,
        data_collator=UnslothVisionDataCollator(model_1, processor_1),
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
            warmup_ratio=0.03,
            max_steps=60,
            learning_rate=0.0002,
            logging_steps=1,
            save_strategy="steps",
            optim="adamw_torch_fused",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
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
    Let's run the model! You can modify the instruction and input—just leave the output blank.

    We'll use the best hyperparameters for inference on Gemma: `top_p=0.95`, `top_k=64`, and `temperature=1.0`.
    """)
    return


@app.cell
def _(FastVisionModel, TextStreamer, dataset, model_1, processor_1):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image_1 = dataset[10]["image"]
    instruction_2 = "Write the LaTeX representation for this image."
    messages_1 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_2}],
        }
    ]
    input_text_1 = processor_1.apply_chat_template(
        messages_1, add_generation_prompt=True
    )
    inputs_1 = processor_1(
        image_1, input_text_1, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    text_streamer_1 = TextStreamer(processor_1, skip_prompt=True)
    result_1 = model_1.generate(
        **inputs_1,
        streamer=text_streamer_1,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, use Hugging Face’s `push_to_hub` for online saving, or `save_pretrained` for local storage.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
    """)
    return


@app.cell
def _(model_1, processor_1):
    model_1.save_pretrained("gemma_3n_lora")
    processor_1.save_pretrained("gemma_3n_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(FastVisionModel, TextStreamer, dataset, model_1, processor_1):
    if False:
        from unsloth import FastVisionModel as _FastVisionModel

        _model, _processor = _FastVisionModel.from_pretrained(
            model_name="gemma_3n_lora", load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
        _FastVisionModel.for_inference(_model)
    FastVisionModel.for_inference(model_1)
    sample = dataset[1]  # Enable for inference!
    image_2 = sample["image"].convert("RGB")
    messages_2 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": sample["text"]}, {"type": "image"}],
        }
    ]
    input_text_2 = processor_1.apply_chat_template(
        messages_2, add_generation_prompt=True
    )
    inputs_2 = processor_1(
        image_2, input_text_2, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    text_streamer_2 = TextStreamer(processor_1.tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_2,
        streamer=text_streamer_2,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
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
def _(model_1, processor_1):
    # Select ONLY 1 to save! (Both not needed!)
    if False:
        # Save locally to 16bit
        model_1.save_pretrained_merged("unsloth_finetune", processor_1)
    if False:
        # To export and save to your Hugging Face account
        model_1.push_to_hub_merged(
            "YOUR_USERNAME/unsloth_finetune", processor_1, token="YOUR_HF_TOKEN"
        )
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
