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
    from unsloth import FastSentenceTransformer

    fourbit_models = [
        "unsloth/all-MiniLM-L6-v2",
        "unsloth/embeddinggemma-300m",
        "unsloth/Qwen3-Embedding-4B",
        "unsloth/Qwen3-Embedding-0.6B",
        "unsloth/all-mpnet-base-v2",
        "unsloth/gte-modernbert-base",
        "unsloth/bge-m3",
    ]  # More models at https://huggingface.co/unsloth

    model = FastSentenceTransformer.from_pretrained(
        model_name="unsloth/bge-m3",
        max_seq_length=512,  # Choose any for long context!
        full_finetuning=False,  # [NEW!] We have full finetuning now!
    )
    return FastSentenceTransformer, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update a small amount of parameters!
    """)
    return


@app.cell
def _(FastSentenceTransformer, model):
    model_1 = FastSentenceTransformer.get_peft_model(
        model,
        r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["key", "query", "dense", "value"],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=False,  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        task_type="FEATURE_EXTRACTION",
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We now use the `electroglyph/technical` dataset.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("electroglyph/technical", split="train")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at the dataset structure:
    """)
    return


@app.cell
def _(dataset):
    print("Dataset examples:")
    for i in range(6):
        print(dataset[i])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline Inference
    Let's test the base model before fine-tuning to see how it performs on our specific domain.
    """)
    return


@app.cell
def _(model_1):
    from sentence_transformers import util

    def test_inference(model, run_name="Run"):
        """Test model with a query and candidate sentences"""
        query = "apexification"
        candidates = [
            "a brick left by Yuki",
            "apples are a tasty treat",
            "the weed whacker uses an engine that runs on a mixture of gas and oil",
            "a type of cancer treatment that uses drugs to boost the body's immune response",
            "a plant hormone for regulating stress responses",
            "induces root tip closure in non-vital teeth",
        ]
        query_emb = model.encode(query, convert_to_tensor=True)  # Completely unrelated
        candidate_embs = model.encode(
            candidates, convert_to_tensor=True
        )  # Unrelated, but shares "ap-" prefix
        scores = util.cos_sim(query_emb, candidate_embs)[0]  # Unrelated
        results = []  # Medical context but wrong procedure
        for i, score in enumerate(scores):  # Scientific but unrelated field
            results.append(
                (candidates[i], score.item())
            )  # CORRECT - this is what apexification actually means
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"\n--- {run_name} Results for query: '{query}' ---")
        for text, score in results:
            print(f"{score:.4f} | {text}")

    test_inference(model_1, run_name="Pre-Training")
    return (test_inference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We use `MultipleNegativesRankingLoss`

     This loss function uses other positives in the same batch as negative examples, which is efficient for contrastive learning.
    """)
    return


@app.cell
def _(dataset, model_1):
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers
    from unsloth import is_bf16_supported

    loss = losses.MultipleNegativesRankingLoss(model_1)
    # This will use other positives in the same batch as negative examples
    trainer = SentenceTransformerTrainer(
        model=model_1,
        train_dataset=dataset,
        loss=loss,
        args=SentenceTransformerTrainingArguments(
            output_dir="output",
            num_train_epochs=2,
            per_device_train_batch_size=256,
            gradient_accumulation_steps=1,  # Use GA to mimic batch size!
            learning_rate=3e-05,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=50,
            warmup_ratio=0.03,
            report_to="none",
            lr_scheduler_type="constant_with_warmup",
            batch_sampler=BatchSamplers.NO_DUPLICATES,
        ),
    )
    return (trainer,)


@app.cell
def _():
    # @title Show current memory stats
    import torch

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return max_memory, start_gpu_memory, torch


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
    Let's run the model after training to see the improvements!
    """)
    return


@app.cell
def _(model_1, test_inference):
    test_inference(model_1, run_name="Post-Training")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit, scroll down!
    """)
    return


@app.cell
def _(model_1):
    model_1.save_pretrained("bge_m3_lora")
    model_1.tokenizer.save_pretrained("bge_m3_lora")
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
        from unsloth import FastSentenceTransformer as _FastSentenceTransformer

        _model = _FastSentenceTransformer.from_pretrained("bge_m3_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "bge_m3_finetune_16bit",
            tokenizer=model_1.tokenizer,
            save_method="merged_16bit",
        )
    if False:  # Pushing to HF Hub
        model_1.push_to_hub_merged(
            "HF_USERNAME/bge_m3_finetune_16bit",
            tokenizer=model_1.tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )
    if False:
        # Just LoRA adapters
        model_1.save_pretrained("bge_m3_lora")
    if False:
        model_1.push_to_hub(
            "HF_USERNAME/bge_m3_lora", token="YOUR_HF_TOKEN"
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
    """)
    return


@app.cell
def _(model_1):
    # Save to 8bit Q8_0
    if False:
        model_1.save_pretrained_gguf("bge_m3_finetune")
    if False:
        model_1.push_to_hub_gguf("HF_USERNAME/bge_m3_finetune", token="YOUR_HF_TOKEN")
    if False:
        # Save to 16bit GGUF
        model_1.save_pretrained_gguf("bge_m3_finetune", quantization_method="f16")
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/bge_m3_finetune",
            quantization_method="f16",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )  # Pushing to HF Hub
    if False:
        model_1.save_pretrained_gguf("bge_m3_finetune", quantization_method="q4_k_m")
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/bge_m3_finetune",
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",  # Get a token at https://huggingface.co/settings/tokens
        )
    if False:  # Pushing to HF Hub
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.push_to_hub_gguf(
            "HF_USERNAME/bge_m3_finetune",
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
