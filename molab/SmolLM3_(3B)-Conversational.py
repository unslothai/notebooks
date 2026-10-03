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

    You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it.

    This notebook fine-tunes **[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)** (Hugging Face's small hybrid-reasoning model) for conversational chat.
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
    <a name="Unsloth"></a>
    ### Unsloth
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, Bfloat16 for Ampere+
    load_in_4bit = True  # 4bit quantization to reduce memory. Can be False.

    # 4bit pre-quantized models we support for 4x faster downloading + no OOMs:
    fourbit_models = [
        "unsloth/SmolLM3-3B",  # This notebook's model
        "unsloth/Qwen3-4B-Instruct-2507",
        "unsloth/gemma-3-4b-it",
        "unsloth/Phi-4",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/SmolLM3-3B",
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        dtype=dtype,  # None for auto detection. Float16 for Tesla T4, Bfloat16 for Ampere+
        load_in_4bit=load_in_4bit,  # 4bit quantization to reduce memory. Can be False.
        full_finetuning=False,
    )
    return FastModel, max_seq_length, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
    """)
    return


@app.cell
def _(FastModel, model):
    model_1 = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # text-only model
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # 30% less VRAM, fits 2x larger batch sizes
        random_state=3407,
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style, and convert it to HuggingFace's `("role", "content")` format.

    Note on SmolLM3: its chat template defaults to `/think` reasoning mode, which adds a reasoning system prompt and expects `<think>...</think>` sections. FineTome has no reasoning traces, so we set `enable_thinking=False` (`/no_think`) so the template matches the data. In `/no_think` mode SmolLM3 still emits an empty `<think></think>` block before the answer, which is expected, so we keep it. We use SmolLM3's own chat template and don't override it with `get_chat_template`.
    """)
    return


@app.cell
def _(tokenizer):
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt

    dataset = load_dataset("mlabonne/FineTome-100k", split="train")  # -> {"role", "content"} format
    dataset = standardize_sharegpt(dataset)  # -> {"role", "content"} format

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,  # SmolLM3: fast /no_think mode, matches non-reasoning data
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)  # -> {"role", "content"} format
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We look at how the conversations are structured for item 5:
    """)
    return


@app.cell
def _(dataset):
    dataset[5]["conversations"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we see how SmolLM3's chat template transformed these conversations (note the short `/no_think` system prompt and the intentional empty `<think></think>` block before the assistant's answer):
    """)
    return


@app.cell
def _(dataset):
    dataset[5]["text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!
    """)
    return


@app.cell
def _(dataset, max_seq_length, model_1, tokenizer):
    from trl import SFTConfig, SFTTrainer
    from transformers import DataCollatorForSeq2Seq

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=0.0002,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use Unsloth's `train_on_responses_only` to train only on the assistant's replies and ignore loss on the user's inputs. Unsloth auto-detects SmolLM3's ChatML instruction/response markers from the chat template.
    """)
    return


@app.cell
def _(trainer):
    from unsloth.chat_templates import train_on_responses_only

    trainer_1 = train_on_responses_only(trainer)
    return (trainer_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We verify masking is done — only the assistant's reply should be unmasked (system + user show as blank):
    """)
    return


@app.cell
def _(tokenizer, trainer_1):
    space = tokenizer(" ", add_special_tokens=False).input_ids[0]
    tokenizer.decode(
        [space if x == -100 else x for x in trainer_1.train_dataset[5]["labels"]]
    )
    return


@app.cell
def _(torch):
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return


@app.cell
def _(trainer_1):
    trainer_stats = trainer_1.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Let's run the model! We keep `/no_think` (`enable_thinking=False`) to match how we trained it. We pass the full tokenized dict so `attention_mask` is set (avoids the pad==eos warning).
    """)
    return


@app.cell
def _(model_1, tokenizer):
    messages = [
        {"role": "user", "content": "Explain what a black hole is, in simple terms."}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,  # SmolLM3: fast /no_think mode, matches non-reasoning data
        return_tensors="pt",
        return_dict=True,  # returns input_ids + attention_mask
    ).to("cuda")
    from transformers import TextStreamer

    _ = model_1.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )  # returns input_ids + attention_mask
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 for VLLM / GGUF
    We save the LoRA adapters below. To merge to 16-bit or export GGUF (for llama.cpp / Ollama), uncomment the relevant lines.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Save the LoRA adapters (small)
    model_1.save_pretrained("smollm_lora")
    tokenizer.save_pretrained("smollm_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we're done! If you have any questions, join our [Discord](https://discord.gg/unsloth). Some other resources:
    - ⭐ Star Unsloth on [GitHub](https://github.com/unslothai/unsloth)
    - 📚 [Unsloth docs](https://unsloth.ai/docs) for all our models & guides
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
