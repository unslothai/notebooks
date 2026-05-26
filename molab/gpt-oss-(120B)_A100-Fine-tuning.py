# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "marimo",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch>=2.8.0",
#     "torchao>=0.16.0",
#     "torchvision",
#     "transformers==4.56.2",
#     "triton>=3.2.0",
#     "triton_kernels @ git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo",
#     "uv",
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
    We're about to demonstrate the power of the new OpenAI GPT-OSS 120B model through a finetuning example. To use our `MXFP4` inference example, use this [notebook](https://github.com/unslothai/notebooks/blob/main/nb/GPT_OSS_MXFP4_(20B)-Inference.ipynb) instead.
    """)
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 4096  # Choose any for long context!
    dtype = None  # None for auto detection

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # 20B model using bitsandbytes 4bit quantization
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b",  # 20B model using MXFP4 format
        "unsloth/gpt-oss-120b",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-120b",  # YOUR MODEL YOU USED FOR TRAINING
        dtype=dtype,  # None for auto detection
        max_seq_length=max_seq_length,  # Choose any for long context!
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastLanguageModel, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.
    """)
    return


@app.cell
def _(FastLanguageModel, model):
    model_1 = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
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
    ### Reasoning Effort
    The `gpt-oss` models from OpenAI include a feature that allows users to adjust the model's "reasoning effort." This gives you control over the trade-off between the model's performance and its response speed (latency) which by the amount of token the model will use to think.

    ----

    The `gpt-oss` models offer three distinct levels of reasoning effort you can choose from:

    * **Low**: Optimized for tasks that need very fast responses and don't require complex, multi-step reasoning.
    * **Medium**: A balance between performance and speed.
    * **High**: Provides the strongest reasoning performance for tasks that require it, though this results in higher latency.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    from transformers import TextStreamer

    messages = [{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",  # **NEW!** Set reasoning effort to low, medium or high
    ).to("cuda")
    _ = model_1.generate(
        **inputs, max_new_tokens=64, streamer=TextStreamer(tokenizer)
    )  # **NEW!** Set reasoning effort to low, medium or high
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Changing the `reasoning_effort` to `medium` will make the model think longer. We have to increase the `max_new_tokens` to occupy the amount of the generated tokens but it will give better and more correct answer
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer):
    messages_1 = [{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}]
    inputs_1 = tokenizer.apply_chat_template(
        messages_1,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="medium",  # **NEW!** Set reasoning effort to low, medium or high
    ).to("cuda")
    _ = model_1.generate(
        **inputs_1, max_new_tokens=64, streamer=TextStreamer(tokenizer)
    )  # **NEW!** Set reasoning effort to low, medium or high
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lastly we will test it using `reasoning_effort` to `high`
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer):
    messages_2 = [{"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}]
    inputs_2 = tokenizer.apply_chat_template(
        messages_2,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="high",  # **NEW!** Set reasoning effort to low, medium or high
    ).to("cuda")
    _ = model_1.generate(
        **inputs_2, max_new_tokens=64, streamer=TextStreamer(tokenizer)
    )  # **NEW!** Set reasoning effort to low, medium or high
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `HuggingFaceH4/Multilingual-Thinking` dataset will be utilized as our example. This dataset, available on Hugging Face, contains reasoning chain-of-thought examples derived from user questions that have been translated from English into four other languages. It is also the same dataset referenced in OpenAI's [cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) for fine-tuning. The purpose of using this dataset is to enable the model to learn and develop reasoning capabilities in these four distinct languages.
    """)
    return


@app.cell
def _(tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    dataset
    return dataset, formatting_prompts_func


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To format our dataset, we will apply our version of the GPT OSS prompt
    """)
    return


@app.cell
def _(dataset, formatting_prompts_func):
    from unsloth.chat_templates import standardize_sharegpt

    dataset_1 = standardize_sharegpt(dataset)
    dataset_1 = dataset_1.map(formatting_prompts_func, batched=True)
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at the dataset, and check what the 1st example shows
    """)
    return


@app.cell
def _(dataset_1):
    print(dataset_1[0]["text"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What is unique about GPT-OSS is that it uses OpenAI [Harmony](https://github.com/openai/harmony) format which support conversation structures, reasoning output, and tool calling.
    """)
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
def _(dataset_1, model_1, tokenizer):
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset_1,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            max_steps=30,
            learning_rate=0.0002,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
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
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer):
    messages_3 = [
        {
            "role": "system",
            "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems.",
        },
        {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
    ]
    inputs_3 = tokenizer.apply_chat_template(
        messages_3,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="medium",  # **NEW!** Set reasoning effort to low, medium or high
    ).to("cuda")
    _ = model_1.generate(
        **inputs_3, max_new_tokens=64, streamer=TextStreamer(tokenizer)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.

    **[NOTE]** Currently finetunes can only be loaded via Unsloth in the meantime - we're working on vLLM and GGUF exporting!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    model_1.save_pretrained("gpt_oss_lora")
    tokenizer.save_pretrained("gpt_oss_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run the finetuned model, you can do the below after setting `if False` to `if True` in a new instance.
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer):
    if False:
        from unsloth import FastLanguageModel as _FastLanguageModel

        _model, _tokenizer = _FastLanguageModel.from_pretrained(
            model_name="gpt_oss_lora",  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=1024,  # Choose any for long context!
            dtype=None,  # None for auto detection
            load_in_4bit=True,  # 4 bit quantization to reduce memory
        )
    messages_4 = [
        {
            "role": "system",
            "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems.",
        },
        {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
    ]
    inputs_4 = tokenizer.apply_chat_template(
        messages_4,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="high",  # **NEW!** Set reasoning effort to low, medium or high
    ).to("cuda")
    _ = model_1.generate(
        **inputs_4, max_new_tokens=64, streamer=TextStreamer(tokenizer)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM or mxfp4

    We also support saving to `float16` directly. Select `merged_16bit` for float16, `merged_4bit` for int4, and `mxfp4` for mxfp4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

    **[NOTE]** Due to the disk limit in molab, we have to save it using `mxfp4` format (4-bit) or we can use 'push_to_hub_merged' function instead.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to mxfp 4bit
    if False:
        model_1.save_pretrained_merged(
            "gpt_oss_finetune_4bit", tokenizer, save_method="mxfp4"
        )
    if False:  # Pushing to HF Hub
        model_1.push_to_hub_merged(
            "HF_USERNAME/gpt_oss_finetune_4bit",
            tokenizer,
            save_method="mxfp4",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )
    if False:
        # Merge and push to hub in 16bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/gpt_oss_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )  # Pushing to HF Hub
    if False:
        model_1.save_pretrained("gpt_oss_lora")
        # Just LoRA adapters
        tokenizer.save_pretrained("gpt_oss_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/gpt_oss_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub(
            "HF_USERNAME/gpt_oss_lora", token="YOUR_HF_TOKEN"
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


@app.cell
def _(model_1, tokenizer):
    # Save to 8bit Q8_0
    if False:
        model_1.save_pretrained_gguf("gpt_oss_finetune", tokenizer)
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gpt_oss_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )
    if False:
        # Save to 16bit GGUF
        model_1.save_pretrained_gguf(
            "gpt_oss_finetune", tokenizer, quantization_method="f16"
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gpt_oss_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )  # Pushing to HF Hub
    if False:
        model_1.save_pretrained_gguf(
            "gpt_oss_finetune", tokenizer, quantization_method="q4_k_m"
        )
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gpt_oss_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )
    if False:  # Pushing to HF Hub
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gpt_oss_finetune",
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
