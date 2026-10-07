# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "hf_transfer",
#     "huggingface_hub>=1.31.0,<2.0",
#     "marimo",
#     "peft",
#     "protobuf",
#     "safetensors>=0.8.0",
#     "sentence-transformers>=6.1.0",
#     "sentencepiece",
#     "tokenizers>=0.23.1,<0.24",
#     "torchao>=0.16.0",
#     "torchcodec",
#     "transformers @ git+https://github.com/huggingface/transformers@92cd495f2720c064bc78eb2d93e28704c5bce51f",
#     "triton>=3.2.0",
#     "trl",
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
    Introducing **[Unsloth Desktop](https://unsloth.ai/docs/desktop)**, the first desktop app to run and train models. Free and open-source for macOS, Windows and Linux. [GitHub](https://github.com/unslothai/unsloth) • [Download](https://unsloth.ai/download)

    <a href="https://unsloth.ai/docs/desktop"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/unsloth-qwen3-8.png" width="350" alt="Introducing Unsloth Desktop"></a>

    Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)

    Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)

    New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)

    Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
    """)
    return


@app.cell
def _():
    import subprocess

    subprocess.run(["unsloth", "install-kernels"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth

    Fine-tune **EmbeddingGemma 2** for sound to caption retrieval with LoRA and compare Recall@K before and after.
    """)
    return


@app.cell
def _():
    from unsloth import FastSentenceTransformer
    import torch

    model = FastSentenceTransformer.from_pretrained(
        model_name="unsloth/embeddinggemma-2",
        max_seq_length=1024,  # audio is 25 tokens per second: a 30 s clip is 750 tokens
        config_kwargs={"vision_config": None},  # skip the vision tower: less VRAM
        full_finetuning=False,
    )
    return FastSentenceTransformer, model, torch


@app.cell
def _():
    # @title Small display helpers (thumbnails, audio players, video)
    import io, base64, html
    from IPython.display import display, HTML, Audio, Video
    from PIL import Image

    def show_images(images, captions=None, size=160):
        cells = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.copy().convert("RGB").resize((size, size)).save(buf, format="JPEG")
            cap = html.escape(captions[i]) if captions else ""
            cells.append(
                f'<div style="display:inline-block;margin:4px;width:{size}px;font-size:11px;vertical-align:top"><img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"><br>{cap}</div>'
            )
        display(HTML("".join(cells)))

    def recall_at_k(similarity, positives, ks=(1, 5, 10)):
        """similarity: [queries, items]; positives[q] = set of correct item indices."""
        ranking = similarity.argsort(dim=1, descending=True)
        out = {}
        for k in ks:
            top = ranking[:, :k].tolist()
            out[f"R@{k}"] = round(
                100
                * sum((len(set(t) & positives[q]) > 0 for q, t in enumerate(top)))
                / len(top),
                2,
            )
        return out

    return Audio, display, recall_at_k


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    Train on Clotho development (~300 clips), test on Clotho evaluation (209 clips).
    """)
    return


@app.cell
def _(Audio, display):
    import random, soundfile as sf, numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, HfApi
    from datasets import Dataset

    import os

    os.makedirs("clotho", exist_ok=True)

    def save_clip(
        row,
    ):  # write the wav once; sentence-transformers decodes + resamples audio files itself
        path = f"clotho/{row['index'].rsplit('/', 1)[-1]}.wav"
        open(path, "wb").write(row["audio"]["bytes"])
        return path

    def split_captions(text):
        return [c.strip() + "." for c in text.split(". ") if c.strip()]

    train_files = sorted(
        f.rfilename
        for f in HfApi().dataset_info("CLAPv2/Clotho").siblings
        if f.rfilename.startswith("data/train/")
    )[:10]
    random.seed(3407)
    pairs = []
    for f in train_files:
        for r in pq.read_table(
            hf_hub_download("CLAPv2/Clotho", f, repo_type="dataset")
        ).to_pylist():
            clip = save_clip(r)
            for c in split_captions(r["text"]):
                pairs.append({"anchor": c, "positive": clip})
    random.shuffle(pairs)
    train_dataset = Dataset.from_list(pairs)

    test_rows = pq.read_table(
        hf_hub_download(
            "mteb/Clotho", "data/test-00000-of-00005.parquet", repo_type="dataset"
        )
    ).to_pylist()
    test_items = [save_clip(r) for r in test_rows]
    test_texts, text_to_item = [], []
    for i, r in enumerate(test_rows):
        for c in split_captions(r["text"]):
            test_texts.append(c)
            text_to_item.append(i)
    print(
        f"train pairs: {len(train_dataset)}, test: {len(test_items)} clips / {len(test_texts)} captions"
    )
    print(pairs[0]["anchor"])
    display(Audio(filename=pairs[0]["positive"]))
    return test_items, test_texts, text_to_item, train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline Performance
    """)
    return


@app.cell
def _(model, recall_at_k, test_items, test_texts, text_to_item, torch):
    def evaluate(model):
        model.eval()
        with torch.no_grad():
            items = model.encode(
                [{"audio": x} for x in test_items], batch_size=8, convert_to_tensor=True
            )
            texts = model.encode(test_texts, batch_size=64, convert_to_tensor=True)
        sim = model.similarity(texts, items).float().cpu()
        item_to_texts = [
            set(j for j, it in enumerate(text_to_item) if it == i)
            for i in range(len(test_items))
        ]
        return {
            "text->audio": recall_at_k(
                sim, [{text_to_item[q]} for q in range(len(test_texts))]
            ),
            "audio->text": recall_at_k(sim.T, item_to_texts),
        }

    baseline = evaluate(model)
    print(baseline)
    return baseline, evaluate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We add LoRA adapters to the language model and the audio encoder (`finetune_audio_layers = True`).
    """)
    return


@app.cell
def _(FastSentenceTransformer, model):
    model_1 = FastSentenceTransformer.get_peft_model(
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
        finetune_audio_layers=True,  # also adapt the audio encoder
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # saves VRAM for the audio tower
        random_state=3407,
        task_type="FEATURE_EXTRACTION",
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    We train for 60 steps; set `num_train_epochs = 1` and `max_steps = -1` for a full run.
    """)
    return


@app.cell
def _(model_1, train_dataset):
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from unsloth import is_bf16_supported

    trainer = SentenceTransformerTrainer(
        model=model_1,
        train_dataset=train_dataset,
        loss=losses.MultipleNegativesRankingLoss(model_1),
        args=SentenceTransformerTrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            max_steps=60,
            learning_rate=0.0001,
            warmup_ratio=0.05,
            lr_scheduler_type="linear",
            logging_steps=5,
            bf16=is_bf16_supported(),
            fp16=not is_bf16_supported(),  # Unsloth switches EmbeddingGemma 2 to float32 math on T4
            remove_unused_columns=False,
            report_to="none",
            output_dir="outputs",
            save_strategy="no",
            seed=3407,
        ),
    )  # Unsloth switches EmbeddingGemma 2 to float32 math on T4
    return (trainer,)


@app.cell
def _(torch):
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return (max_memory,)


@app.cell
def _(trainer):
    trainer_stats = trainer.train()
    return (trainer_stats,)


@app.cell
def _(max_memory, torch, trainer_stats):
    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"Peak reserved memory = {used_memory} GB of {max_memory} GB.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluate after fine-tuning
    """)
    return


@app.cell
def _(baseline, evaluate, model_1):
    finetuned = evaluate(model_1)
    for direction in baseline:
        print(
            f"{direction:12s} before {baseline[direction]}  after {finetuned[direction]}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    """)
    return


@app.cell
def _(Audio, display, model_1, test_items, test_texts, text_to_item):
    item_embeddings = model_1.encode(
        [{"audio": x} for x in test_items], batch_size=8, convert_to_tensor=True
    )
    for query in ["birds chirping in a forest", "a car driving past on a wet road"]:
        scores = model_1.similarity(
            model_1.encode(query, convert_to_tensor=True), item_embeddings
        )[0]
        best = scores.argmax().item()
        print(f"{query} -> {test_texts[text_to_item.index(best)]} ({scores[best]:.3f})")
        display(Audio(filename=test_items[best]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving
    `save_pretrained` saves the LoRA adapters, `save_pretrained_merged` the full model.
    """)
    return


@app.cell
def _(model_1):
    model_1.save_pretrained("embeddinggemma_lora")
    model_1.tokenizer.save_pretrained("embeddinggemma_lora")
    return


@app.cell
def _(model_1):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "embeddinggemma_finetune_16bit",
            tokenizer=model_1.tokenizer,
            save_method="merged_16bit",
        )
    if False:  # Pushing to HF Hub
        model_1.push_to_hub_merged(
            "HF_USERNAME/embeddinggemma_finetune_16bit",
            tokenizer=model_1.tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
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
