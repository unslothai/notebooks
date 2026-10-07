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

    **EmbeddingGemma 2** embeds text, images, audio and video into one 768-dim space. Below are small search demos on Flickr30k, Clotho and MSR-VTT.
    """)
    return


@app.cell
def _():
    from unsloth import FastSentenceTransformer
    import torch

    model = FastSentenceTransformer.from_pretrained(
        model_name="unsloth/embeddinggemma-2",
        for_inference=True,  # inference only: no LoRA, no training patches
    )
    print(model)
    print("dtype:", next(model.parameters()).dtype)
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

    return Audio, Image, Video, display, recall_at_k, show_images


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Text search
    Queries use `prompt_name = "query"`, documents `prompt_name = "document"`.
    """)
    return


@app.cell
def _(model):
    documents = [
        "The aurora borealis is caused by charged particles from the solar wind colliding with oxygen and nitrogen in the upper atmosphere.",
        "Photosynthesis lets plants turn carbon dioxide and water into sugar using sunlight.",
        "The Great Barrier Reef is the world's largest coral reef system, off the coast of Queensland, Australia.",
        "Mount Everest's summit is 8,849 metres above sea level.",
        "A transformer is a neural network architecture built on self-attention.",
        "Unsloth makes fine-tuning large language models up to 2x faster with less memory.",
    ]
    query = "Why is the sky green and purple near the north pole at night?"

    doc_embeddings = model.encode(
        documents, prompt_name="document", convert_to_tensor=True
    )
    query_embedding = model.encode(query, prompt_name="query", convert_to_tensor=True)
    scores = model.similarity(query_embedding, doc_embeddings)[0]
    for i in scores.argsort(descending=True).tolist():
        print(f"{scores[i]:.3f} | {documents[i]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other prompts: `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, `SentenceSimilarity`, `Clustering`, `Classification`. Images, audio and video need no prompt.
    """)
    return


@app.cell
def _(model):
    for name, prompt in model.prompts.items():
        print(f"{name:26s} -> {prompt!r}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Image search on Flickr30k (1K test)
    """)
    return


@app.cell
def _(Image, show_images):
    import zipfile, csv, json, os
    from huggingface_hub import hf_hub_download

    repo = "nlphuji/flickr_1k_test_image_text_retrieval"
    zip_path = hf_hub_download(repo, "images_flickr_1k_test.zip", repo_type="dataset")
    csv_path = hf_hub_download(repo, "test_1k_flickr.csv", repo_type="dataset")
    rows = list(csv.DictReader(open(csv_path)))
    archive = zipfile.ZipFile(zip_path)
    name_to_member = {
        os.path.basename(n): n for n in archive.namelist() if n.endswith(".jpg")
    }
    flickr_images = [
        Image.open(archive.open(name_to_member[r["filename"]])).convert("RGB")
        for r in rows
    ]
    flickr_captions, caption_to_image = ([], [])
    for i_1, r in enumerate(rows):
        for c in json.loads(r["raw"]):
            flickr_captions.append(c)
            caption_to_image.append(i_1)
    print(len(flickr_images), "images,", len(flickr_captions), "captions")
    show_images(flickr_images[:6], [json.loads(r["raw"])[0] for r in rows[:6]])
    return (
        caption_to_image,
        flickr_captions,
        flickr_images,
        hf_hub_download,
        json,
        os,
        zipfile,
    )


@app.cell
def _(flickr_captions, flickr_images, model, show_images):
    import time

    start = time.time()
    image_embeddings = model.encode(
        flickr_images, batch_size=16, convert_to_tensor=True, show_progress_bar=True
    )
    caption_embeddings = model.encode(
        flickr_captions, batch_size=64, convert_to_tensor=True
    )
    print(
        f"Embedded {len(flickr_images)} images + {len(flickr_captions)} captions in {time.time() - start:.0f}s"
    )

    def search_images(text, k=5):
        q = model.encode(text, convert_to_tensor=True)
        scores = model.similarity(q, image_embeddings)[0]
        top = scores.topk(k)
        show_images(
            [flickr_images[i] for i in top.indices.tolist()],
            [f"{s:.3f}" for s in top.values.tolist()],
        )

    search_images("a dog jumping to catch a frisbee")
    search_images("people sitting at a cafe on a busy street")
    return caption_embeddings, image_embeddings


@app.cell
def _(flickr_images, image_embeddings, model, show_images):
    scores_1 = model.similarity(image_embeddings[7:8], image_embeddings)[0]
    show_images([flickr_images[i] for i in scores_1.topk(5).indices.tolist()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall@K:
    """)
    return


@app.cell
def _(
    caption_embeddings,
    caption_to_image,
    flickr_captions,
    flickr_images,
    image_embeddings,
    model,
    recall_at_k,
):
    text_to_image = model.similarity(caption_embeddings, image_embeddings).float().cpu()
    t2i = recall_at_k(
        text_to_image, [{caption_to_image[q]} for q in range(len(flickr_captions))]
    )
    image_to_texts = [
        set(j for j, img in enumerate(caption_to_image) if img == i)
        for i in range(len(flickr_images))
    ]
    i2t = recall_at_k(text_to_image.T, image_to_texts)
    print("Flickr30k 1K  text->image:", t2i)
    print("Flickr30k 1K  image->text:", i2t)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Matryoshka
    Use `truncate_dim` for 512, 256 or 128 dimensions.
    """)
    return


@app.cell
def _(
    caption_embeddings,
    caption_to_image,
    flickr_captions,
    image_embeddings,
    model,
    recall_at_k,
):
    import torch.nn.functional as F

    print(f"{'dims':>5s}  {'text->image R@1':>16s}  {'R@5':>6s}")
    for dims in [768, 512, 256, 128]:
        q = F.normalize(caption_embeddings[:, :dims].float(), dim=-1)
        d = F.normalize(image_embeddings[:, :dims].float(), dim=-1)
        r_1 = recall_at_k(
            (q @ d.T).cpu(),
            [{caption_to_image[i]} for i in range(len(flickr_captions))],
            ks=(1, 5),
        )
        print(f"{dims:5d}  {r_1['R@1']:16.2f}  {r_1['R@5']:6.2f}")
    small = model.encode(
        ["a dog on the beach"], truncate_dim=256, normalize_embeddings=True
    )
    print("truncated shape:", small.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Sound search on Clotho
    """)
    return


@app.cell
def _(Audio, display, hf_hub_download, model, os, recall_at_k):
    import soundfile as sf, numpy as np, pyarrow.parquet as pq

    clotho_path = hf_hub_download(
        "mteb/Clotho", "data/test-00000-of-00005.parquet", repo_type="dataset"
    )
    clotho = pq.read_table(clotho_path).to_pylist()
    os.makedirs("clotho", exist_ok=True)
    clips = []  # file paths: sentence-transformers decodes (and resamples) audio files itself
    for r_2 in (
        clotho
    ):  # file paths: sentence-transformers decodes (and resamples) audio files itself
        path = f"clotho/{r_2['index'].rsplit('_', 1)[-1]}.wav"
        open(path, "wb").write(r_2["audio"]["bytes"])
        clips.append(path)
    sound_captions, caption_to_clip = ([], [])
    for i_2, r_2 in enumerate(clotho):
        for c_1 in [c.strip() + "." for c in r_2["text"].split(". ") if c.strip()]:
            sound_captions.append(c_1)
            caption_to_clip.append(i_2)
    print(len(clips), "clips,", len(sound_captions), "captions")
    clip_embeddings = model.encode(
        [{"audio": c} for c in clips],
        batch_size=8,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    sound_caption_embeddings = model.encode(
        sound_captions, batch_size=64, convert_to_tensor=True
    )

    def search_sounds(text, k=2):
        q = model.encode(text, convert_to_tensor=True)
        scores = model.similarity(q, clip_embeddings)[0]
        for s, i in zip(*scores.topk(k)):
            print(f"{s:.3f} | {clotho[i]['text'].split('. ')[0]}")
            display(Audio(filename=clips[i]))

    search_sounds("heavy rain falling on a roof")
    r_2 = recall_at_k(
        model.similarity(sound_caption_embeddings, clip_embeddings).float().cpu(),
        [{caption_to_clip[q]} for q in range(len(sound_captions))],
    )
    print("Clotho (209 clips) text->audio:", r_2)
    return (clips,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5. Video search on MSR-VTT (first 40 test videos)
    """)
    return


@app.cell
def _(Video, display, hf_hub_download, json, model, os, recall_at_k, zipfile):
    from huggingface_hub import HfFileSystem

    test_1k = json.load(
        open(
            hf_hub_download(
                "friedrichor/MSR-VTT", "msrvtt_test_1k.json", repo_type="dataset"
            )
        )
    )[:40]
    os.makedirs("msrvtt", exist_ok=True)
    with HfFileSystem().open(
        "datasets/friedrichor/MSR-VTT/MSRVTT_Videos.zip", "rb"
    ) as f:
        remote = zipfile.ZipFile(f)
        members = {
            os.path.basename(n): n for n in remote.namelist() if n.endswith(".mp4")
        }
        for item in test_1k:
            target = f"msrvtt/{item['video']}"
            if not os.path.exists(target):
                open(target, "wb").write(remote.read(members[item["video"]]))
    videos = [f"msrvtt/{item['video']}" for item in test_1k]
    video_captions = [item["caption"] for item in test_1k]
    video_embeddings = model.encode(
        [{"video": v} for v in videos],
        batch_size=2,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    video_caption_embeddings = model.encode(video_captions, convert_to_tensor=True)
    r_3 = recall_at_k(
        model.similarity(video_caption_embeddings, video_embeddings).float().cpu(),
        [{i} for i in range(len(videos))],
        ks=(1, 5),
    )
    print("MSR-VTT (first 40 of 1K-A) text->video:", r_3)
    query_1 = "fish swimming around in an aquarium"
    best = (
        model.similarity(
            model.encode(query_1, convert_to_tensor=True), video_embeddings
        )[0]
        .argmax()
        .item()
    )
    print(
        "Query:",
        query_1,
        "| best match:",
        videos[best],
        "| its caption:",
        video_captions[best],
    )
    display(Video(videos[best], embed=True, width=320))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6. Interleaved queries
    Use `<|image|>`, `<|audio|>` and `<|video|>` placeholders in the text.
    """)
    return


@app.cell
def _(clips, flickr_images, image_embeddings, model, show_images):
    mixed_query = {
        "text": "A photo like <|image|> but at the beach, with a sound like <|audio|>",
        "image": flickr_images[7],
        "audio": clips[0],
    }
    mixed = model.encode(mixed_query, convert_to_tensor=True)
    print("one vector:", tuple(mixed.shape), "norm", round(mixed.norm().item(), 4))
    scores_2 = model.similarity(mixed, image_embeddings)[0]
    show_images([flickr_images[i] for i in scores_2.topk(5).indices.tolist()])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7. Load only what you need
    Text only is 271M parameters instead of 744M.
    """)
    return


@app.cell
def _(FastSentenceTransformer, model, torch):
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()
    variants = {
        "text only": {"vision_config": None, "audio_config": None},
        "text + vision": {"audio_config": None},
        "text + audio": {"vision_config": None},
        "full (omni)": {},
    }
    print(f"{'variant':16s} {'params':>10s} {'VRAM (GB)':>10s}")
    for name_1, config_kwargs in variants.items():
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        m = FastSentenceTransformer.from_pretrained(
            "unsloth/embeddinggemma-2", for_inference=True, config_kwargs=config_kwargs
        )
        params = sum((p.numel() for p in m.parameters()))
        print(
            f"{name_1:16s} {params / 1000000.0:9.1f}M {(torch.cuda.memory_allocated() - base) / 2**30:10.2f}"
        )
        del m
        gc.collect()
        torch.cuda.empty_cache()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Next steps
    * Fine-tune EmbeddingGemma 2 on your own text pairs: `EmbeddingGemma2_(300M).ipynb`
    * Fine-tune image <-> text retrieval: `EmbeddingGemma2_(300M)-Image_Text.ipynb`
    * Fine-tune audio <-> text retrieval: `EmbeddingGemma2_(300M)-Audio.ipynb`
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
