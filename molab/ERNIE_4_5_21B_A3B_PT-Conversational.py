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
    To run this, press the **Run** button beside each cell on your L4 molab Pro instance!
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
    from unsloth import FastModel
    import torch

    fourbit_models = [
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",  # Qwen 14B 2x faster
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/Phi-4",
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit",  # [NEW] We support TTS models!
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/ERNIE-4.5-21B-A3B-PT",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=2048,  # Choose any for long context!
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastModel, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update a small amount of parameters!
    """)
    return


@app.cell
def _(FastModel, model):
    model_1 = FastModel.get_peft_model(
        model,
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
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
    We now use the `ERNIE-4.5` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. ERNIE-4.5 renders multi turn conversations like below:

    ```
    <|begin_of_sentence|>User: Hello
    Assistant: Hey there!<|end_of_sentence|>
    ```
    We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!
    """)
    return


@app.cell
def _(dataset):
    from unsloth.chat_templates import standardize_data_formats

    dataset_1 = standardize_data_formats(dataset)
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how row 100 looks like!
    """)
    return


@app.cell
def _(dataset_1):
    dataset_1[100]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have to apply the chat template for `ERNIE-4.5` onto the conversations, and save it to `text`.
    """)
    return


@app.cell
def _(dataset_1, tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            + tokenizer.eos_token
            for convo in convos
        ]
        return {"text": texts}

    dataset_2 = dataset_1.map(formatting_prompts_func, batched=True)
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how the chat template did!
    """)
    return


@app.cell
def _(dataset_2):
    dataset_2[100]["text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
    """)
    return


@app.cell
def _(dataset_2, model_1, tokenizer):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset_2,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,  # Use GA to mimic batch size!
            warmup_steps=5,
            max_steps=60,
            learning_rate=0.0002,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return (trainer,)


@app.cell
def _(tokenizer, trainer):
    tokenizer.decode(trainer.train_dataset[0]["input_ids"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!
    """)
    return


@app.cell
def _(trainer):
    from unsloth.chat_templates import train_on_responses_only

    trainer_1 = train_on_responses_only(
        trainer, instruction_part="User:", response_part="Assistant:"
    )
    return (trainer_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's verify masking the instruction part is done! Let's print the 100th row again.
    """)
    return


@app.cell
def _(tokenizer, trainer_1):
    tokenizer.decode(trainer_1.train_dataset[100]["input_ids"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's print the masked out example - you should see only the answer is present:
    """)
    return


@app.cell
def _(tokenizer, trainer_1):
    tokenizer.decode(
        [
            tokenizer.pad_token_id if x == -100 else x
            for x in trainer_1.train_dataset[100]["labels"]
        ]
    ).replace(tokenizer.pad_token, " ")
    return


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
def _(trainer_1):
    trainer_stats = trainer_1.train()
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
    Let's run the model via Unsloth native inference!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    messages = [{"role": "user", "content": "Continue the sequence: 1, 1, 2, 3, 5, 8,"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from transformers import TextStreamer

    _ = model_1.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=1000,  # Increase for longer outputs!
        streamer=TextStreamer(tokenizer, skip_prompt=True),
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
    model_1.save_pretrained("ernie_lora")
    tokenizer.save_pretrained("ernie_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _():
    if False:
        from unsloth import FastLanguageModel as _FastLanguageModel

        _model, _tokenizer = _FastLanguageModel.from_pretrained(
            model_name="ernie_lora", max_seq_length=2048, load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "ernie_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:  # Pushing to HF Hub
        model_1.push_to_hub_merged(
            "HF_USERNAME/ernie_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )
    if False:
        # Merge to 4bit
        model_1.save_pretrained_merged(
            "ernie_finetune_4bit", tokenizer, save_method="merged_4bit"
        )
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/ernie_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )  # Pushing to HF Hub
    if False:
        model_1.save_pretrained("ernie_lora")
        # Just LoRA adapters
        tokenizer.save_pretrained("ernie_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/ernie_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub(
            "HF_USERNAME/ernie_lora", token="YOUR_HF_TOKEN"
        )  # Pushing to HF Hub
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Save to 8bit Q8_0
    if False:
        model_1.save_pretrained_gguf("ernie_finetune", tokenizer)
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/ernie_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )
    if False:
        # Save to 16bit GGUF
        model_1.save_pretrained_gguf(
            "ernie_finetune", tokenizer, quantization_method="f16"
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/ernie_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )  # Pushing to HF Hub
    if False:
        model_1.save_pretrained_gguf(
            "ernie_finetune", tokenizer, quantization_method="q4_k_m"
        )
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/ernie_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )
    if False:  # Pushing to HF Hub
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.push_to_hub_gguf(
            "HF_USERNAME/ernie_finetune",
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
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

      <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
