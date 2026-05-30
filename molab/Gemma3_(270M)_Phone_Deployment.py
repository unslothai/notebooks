# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "executorch==1.1.0",
#     "hf_transfer",
#     "huggingface_hub>=0.34.0",
#     "marimo",
#     "optimum-executorch @ git+https://github.com/huggingface/optimum-executorch.git@v0.1.0",
#     "optimum==1.24.0",
#     "peft",
#     "protobuf",
#     "pytorch-tokenizers",
#     "sentencepiece",
#     "torchao==0.15.0",
#     "transformers==4.57.3",
#     "triton>=3.2.0",
#     "trl==0.25.1",
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

    # Models supported for Phone Deployment
    fourbit_models = [
        "unsloth/Qwen3-4B",  # Any Qwen3 model like 0.6B, 4B, 8B, 32B
        "unsloth/Qwen3-32B",
        "unsloth/Llama-3.1-8B-Instruct",  # Llama 3 models work
        "unsloth/Llama-3.3-70B-Instruct",
        "unsloth/gemma-3-270m-it",  # Gemma 3 models work
        "unsloth/gemma-3-27b-it",
        "unsloth/Qwen2.5-7B-Instruct",  # And more models!
        "unsloth/Phi-4-mini-instruct",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        full_finetuning=True,
        qat_scheme="int4",  # Gemma3 needs int4 due to large vocab (262K)
    )
    return model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:

    ```
    <bos><start_of_turn>user
    Hello!<end_of_turn>
    <start_of_turn>model
    Hey there!<end_of_turn>
    ```

    We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
    """)
    return


@app.cell
def _(tokenizer):
    from unsloth.chat_templates import get_chat_template

    tokenizer_1 = get_chat_template(tokenizer, chat_template="gemma-3")
    return (tokenizer_1,)


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We sample 10k rows to speed up training for the first successful phone deployment.
    """)
    return


@app.cell
def _(dataset):
    dataset_1 = dataset.shuffle(seed=3407).select(range(10000))
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!
    """)
    return


@app.cell
def _(dataset_1):
    from unsloth.chat_templates import standardize_data_formats

    dataset_2 = standardize_data_formats(dataset_1)
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how row 100 looks like!
    """)
    return


@app.cell
def _(dataset_2):
    dataset_2[100]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one.
    """)
    return


@app.cell
def _(dataset_2, tokenizer_1):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer_1.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset_3 = dataset_2.map(formatting_prompts_func, batched=True)
    return (dataset_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one.
    """)
    return


@app.cell
def _(dataset_3):
    dataset_3[100]["text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Fine-tuning requires careful experimentation. To avoid wasting hours on a broken pipeline, we start with a 5-step sanity check. This ensures the training stabilizes and the model exports correctly to your phone.

    Run this short test first. If the export succeeds, come back and set max_steps = -1 (or num_train_epochs = 1) for the full training run.
    """)
    return


@app.cell
def _(dataset_3, model, tokenizer_1):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer_1,
        train_dataset=dataset_3,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            max_steps=60,
            warmup_steps=5,
            learning_rate=5e-06,
            optim="adamw_torch",
            max_grad_norm=1.0,
            logging_steps=1,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
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
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    training_percentage = round(used_memory_for_training / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        f"Peak reserved memory for training % of max memory = {training_percentage} %."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models

    To save the model for phone deployment, we first save the model and tokenizer via `save_pretrained`.
    """)
    return


@app.cell
def _(model, tokenizer_1):
    # Save the model and tokenizer directly
    model.save_pretrained("gemma_phone_model")
    tokenizer_1.save_pretrained("gemma_phone_model")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then export directly from the local folder using Optimum Executorch as per [the documentation.](https://github.com/huggingface/optimum-executorch/blob/main/optimum/exporters/executorch/README.md)
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%file export_gemma_model.py
    # from optimum.executorch import ExecuTorchModelForCausalLM
    # import shutil
    # import os
    #
    # print("Exporting Gemma3-270M to ExecuTorch format...")
    #
    # # Export the trained model using Python API
    # et_model = ExecuTorchModelForCausalLM.from_pretrained(
    #     "gemma_phone_model",
    #     export = True,
    #     recipe = "xnnpack",
    #     task = "text-generation",
    # )
    #
    # # Copy .pte file to output directory
    # temp_dir = et_model._temp_dir.name
    # os.makedirs("gemma_output", exist_ok = True)
    #
    # for f in os.listdir(temp_dir):
    #     src = os.path.join(temp_dir, f)
    #
    # print("\nExport complete!")
    return


@app.cell
def _():
    import subprocess as _molab_subprocess

    _molab_subprocess.call(["python", "export_gemma_model.py"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we have the file Gemma3 model.pte of size 306M!
    """)
    return


@app.cell
def _():
    import subprocess as _molab_subprocess_2

    _molab_subprocess_2.call(["ls", "-lh", "gemma_output/model.pte"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Inference on Exported Model
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%file test_executorch.py
    # from transformers import AutoTokenizer
    # from optimum.executorch import ExecuTorchModelForCausalLM
    #
    # # Load the exported model for inference
    # et_model = ExecuTorchModelForCausalLM.from_pretrained("gemma_output", export = False)
    # tokenizer = AutoTokenizer.from_pretrained("gemma_phone_model")
    #
    # # Test generation
    # prompt = "<start_of_turn>user\nWhat is 2 + 2?<end_of_turn>\n<start_of_turn>model\n"
    # output = et_model.text_generation(tokenizer, prompt, max_seq_len = 50)
    # print(f"Generated: {output}")
    return


@app.cell
def _():
    import subprocess as _molab_subprocess_3

    _molab_subprocess_3.call(["python", "test_executorch.py"])
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
