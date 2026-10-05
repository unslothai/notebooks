# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "marimo",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch==2.8.0",
#     "torchao>=0.16.0",
#     "torchcodec==0.7.0",
#     "torchvision",
#     "transformers==5.2.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo",
#     "uv",
#     "xformers>=0.0.33",
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


@app.cell
def _():
    import subprocess

    return


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

    A decision model doesn't write text. It reads an input, looks at the options you give it, and picks one with a probability. `FastDecisionModel` turns an LLM into one: it adds a small head that scores each option.

    Change `model_name` to `unsloth/Llama-3.2-3B-Instruct` or `unsloth/gemma-4-E4B-it` to try other models.
    """)
    return


@app.cell
def _():
    from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
    import torch

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=2048,  # Longest input. Long inputs keep the question and options.
        load_in_4bit=True,  # 4 bit quantization to reduce memory
    )
    return (
        DecisionTrainer,
        FastDecisionModel,
        is_bfloat16_supported,
        model,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update a small amount of parameters! The new decision head always trains.
    """)
    return


@app.cell
def _(FastDecisionModel, model):
    model_1 = FastDecisionModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We use the [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) dataset. Each row has a `state` (the input), the `questions` to decide about it, and the `gold` answers. There are three kinds of questions:

    * `choice`: pick one option, like which team should handle a ticket.
    * `noul`: yes or no.
    * `score`: pick a level, like how urgent something is.

    Let's see how row 0 looks like!
    """)
    return


@app.cell
def _():
    from datasets import load_dataset
    import json

    dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    print(dataset[0]["state"][:500])
    print(json.dumps(json.loads(dataset[0]["questions"]), indent=2)[:1000])
    print(dataset[0]["gold"])
    return dataset, load_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `build_dataset` turns every question into one example, and tells you how many it skipped and why. To use your own data, give it a list of rows with the same `state`, `questions` and `gold` fields.

    `split_holdout` keeps some rows out of training, so we can check accuracy and calibrate the model later.
    """)
    return


@app.cell
def _(FastDecisionModel, dataset, model_1, tokenizer):
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model_1)
    print(f"Skipped {report['skipped']} of {report['total']} decisions")
    train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)
    print(f"{len(train_items)} training decisions, {len(eval_items)} held out")
    return eval_items, train_items


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check accuracy before training. The head is new, so it's about the same as guessing.
    """)
    return


@app.cell
def _(FastDecisionModel, eval_items, model_1, tokenizer):
    FastDecisionModel.evaluate(model_1, tokenizer, eval_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, which reached 76% test accuracy in our run. For the full run, set `num_train_epochs = 2` and remove `max_steps`: it reached 81%, but takes about 3.5 hours on a free T4 GPU.
    """)
    return


@app.cell
def _(
    DecisionTrainer,
    eval_items,
    is_bfloat16_supported,
    model_1,
    tokenizer,
    train_items,
):
    from transformers import TrainingArguments

    trainer = DecisionTrainer(
        model=model_1,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=eval_items,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=10,
            max_steps=60,
            learning_rate=0.0002,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            eval_strategy="epoch",
            logging_steps=10,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
            seed=3407,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
    """)
    return


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
    First we calibrate the model on the held-out rows. Calibration adjusts the probabilities, so an answer given with 90% confidence is right about 90% of the time. `ece` is the calibration error, lower is better.
    """)
    return


@app.cell
def _(FastDecisionModel, eval_items, model_1, tokenizer):
    FastDecisionModel.calibrate(model_1, tokenizer, eval_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check accuracy on the dataset's test split, which the model never saw:
    """)
    return


@app.cell
def _(FastDecisionModel, load_dataset, model_1, tokenizer):
    test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model_1)
    FastDecisionModel.evaluate(model_1, tokenizer, test_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's make some decisions! Give `predict` an input and your questions. `answer` is the option for `choice`, `True` or `False` for `noul`, and the level number for `score`.
    """)
    return


@app.cell
def _(FastDecisionModel, model_1, tokenizer):
    FastDecisionModel.for_inference(model_1)
    answers = FastDecisionModel.predict(
        model_1,
        tokenizer,
        "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
        {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages, errors",
                    "sales": "pricing, new plans",
                },
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "today"],
            },
        },
    )
    for name, result in answers.items():
        print(
            name,
            result["answer"],
            {k: round(v, 3) for k, v in result["probabilities"].items()},
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model, use `save_pretrained` for a local save or `push_to_hub` for an online save. This saves the LoRA adapters, the decision head and its calibration.
    """)
    return


@app.cell
def _(model_1):
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # Online saving
    model_1.save_pretrained("qwen_lora")  # Local saving
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the model we just saved, set `False` to `True`:
    """)
    return


@app.cell
def _():
    if False:
        from unsloth import FastDecisionModel as _FastDecisionModel

        _model, _tokenizer = _FastDecisionModel.from_pretrained(
            model_name="qwen_lora", max_seq_length=2048, load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16

    We also support saving to `float16` directly with `save_pretrained_merged`. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "qwen_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:  # Pushing to HF Hub
        model_1.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

    To train decision models on your own data, read our [guide](https://unsloth.ai/docs/models/decision-model-training).

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
