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
    ### Unsloth
    """)
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import (
        get_chat_template,
    )
    import torch

    MODEL_ID = "unsloth/Qwen3-0.6B"
    QAT_SCHEME = "int8-int4"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,
        # ExecuTorch CPU quantization scheme
        # Quantize embedding to 8-bits, and quantize linear layers to 4-bits
        # with 8-bit dynamically quantized activations
        qat_scheme=QAT_SCHEME,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
    return MODEL_ID, QAT_SCHEME, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    Qwen3 has both reasoning and a non reasoning mode. So, we should use 2 datasets:

    1. We use the [Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) dataset which was used to win the [AIMO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard) (AI Mathematical Olympiad - Progress Prize 2) challenge! We sample 10% of verifiable reasoning traces that used DeepSeek R1, and which got > 95% accuracy.

    2. We also leverage [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. But we need to convert it to HuggingFace's normal multiturn format as well.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    return non_reasoning_dataset, reasoning_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see the structure of both datasets:
    """)
    return


@app.cell
def _(reasoning_dataset):
    reasoning_dataset
    return


@app.cell
def _(non_reasoning_dataset):
    non_reasoning_dataset
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now convert the reasoning dataset into conversational format:
    """)
    return


@app.function
def generate_conversation(examples):
    problems = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        )
    return {
        "conversations": conversations,
    }


@app.cell
def _(reasoning_dataset, tokenizer):
    reasoning_conversations = tokenizer.apply_chat_template(
        list(
            reasoning_dataset.map(generate_conversation, batched=True)["conversations"]
        ),
        tokenize=False,
    )
    return (reasoning_conversations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see the first transformed row:
    """)
    return


@app.cell
def _(reasoning_conversations):
    reasoning_conversations[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we take the non reasoning dataset and convert it to conversational format as well.

    We have to use Unsloth's `standardize_sharegpt` function to fix up the format of the dataset first.
    """)
    return


@app.cell
def _(non_reasoning_dataset, tokenizer):
    from unsloth.chat_templates import standardize_sharegpt

    dataset = standardize_sharegpt(non_reasoning_dataset)

    non_reasoning_conversations = tokenizer.apply_chat_template(
        list(dataset["conversations"]),
        tokenize=False,
    )
    return (non_reasoning_conversations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see the first row
    """)
    return


@app.cell
def _(non_reasoning_conversations):
    non_reasoning_conversations[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's see how long both datasets are:
    """)
    return


@app.cell
def _(non_reasoning_conversations, reasoning_conversations):
    print(len(reasoning_conversations))
    print(len(non_reasoning_conversations))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The non reasoning dataset is much longer. Let's assume we want the model to retain some reasoning capabilities, but we specifically want a chat model.

    Let's define a ratio of chat only data. The goal is to define some mixture of both sets of data.

    Let's select 75% reasoning and 25% chat based:
    """)
    return


@app.cell
def _():
    chat_percentage = 0.25
    return (chat_percentage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's sample the reasoning dataset by 75% (or whatever is 100% - chat_percentage)
    """)
    return


@app.cell
def _(chat_percentage, non_reasoning_conversations, reasoning_conversations):
    import pandas as pd

    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations) * (chat_percentage / (1 - chat_percentage))),
        random_state=2407,
    )
    print(len(reasoning_conversations))
    print(len(non_reasoning_subset))
    print(
        len(non_reasoning_subset)
        / (len(non_reasoning_subset) + len(reasoning_conversations))
    )
    return non_reasoning_subset, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally combine both datasets:
    """)
    return


@app.cell
def _(non_reasoning_subset, pd, reasoning_conversations):
    data = pd.concat(
        [pd.Series(reasoning_conversations), pd.Series(non_reasoning_subset)]
    )
    data.name = "text"

    from datasets import Dataset

    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=3407)
    return (combined_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
    """)
    return


@app.cell
def _(combined_dataset, model, tokenizer):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=100,
            learning_rate=5e-5,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
    """)
    return


@app.cell
def _(trainer):
    trainer_stats = trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Let's run the model via Unsloth native inference! According to the `Qwen-3` team, the recommended settings for reasoning inference are `temperature = 0.6, top_p = 0.95, top_k = 20`

    For normal chat based inference, `temperature = 0.7, top_p = 0.8, top_k = 20`
    """)
    return


@app.cell
def _(model, tokenizer):
    messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
        enable_thinking=False,  # Disable thinking
    )

    from transformers import TextStreamer

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=256,  # Increase for longer outputs!
        temperature=0.7,  # For non thinking
        top_p=0.8,
        top_k=20,  # For non thinking
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    return (TextStreamer,)


@app.cell
def _(TextStreamer, model, tokenizer):
    messages_1 = [{"role": "user", "content": "Solve (x + 2)^2 = 0."}]
    text_1 = tokenizer.apply_chat_template(
        messages_1, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    _ = model.generate(
        **tokenizer(text_1, return_tensors="pt").to("cuda"),
        max_new_tokens=1024,  # Increase for longer outputs!
        temperature=0.6,  # For non thinking
        top_p=0.95,
        top_k=20,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving
    The final model can be saved locally or pushed to the Hub to share with the community.
    """)
    return


@app.cell
def _(MODEL_ID, QAT_SCHEME, model, tokenizer):
    model_name = MODEL_ID.split("/")[-1]
    save_to = f"{model_name}-{QAT_SCHEME}-unsloth"

    # Save locally - this auto-detects QAT models and handles the conversion
    model.save_pretrained_torchao(save_to, tokenizer=tokenizer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once saved, we can export the model checkpoint to ExecuTorch.
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
