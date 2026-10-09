# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "hf_transfer",
#     "huggingface_hub>=1.5.0,<2.0",
#     "marimo",
#     "peft",
#     "protobuf",
#     "sentencepiece",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torchao>=0.16.0",
#     "transformers==5.11.0",
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
    To run this notebook on your A100 molab Pro instance, hit the **▶ Run all** button in the bottom-right corner - or use `Ctrl/Cmd + Shift + R`.
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

    **Goal: finetune DiffusionGemma to solve Sudoku.** Sudoku is a global-constraint task - the answer must be consistent across the whole 9x9 grid at once, and solving means going back to **revise** wrong cells. An autoregressive model commits each cell left to right and cannot take it back; DiffusionGemma denoises the entire grid in parallel and revises cells over steps, which is exactly the capability this task needs.

    DiffusionGemma needs a transformers build that ships the DiffusionGemma classes; `FastModel` auto-detects the diffusion architecture and routes to the transformers-only slow path.
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    # DiffusionGemma is a 26B-A4B block-diffusion MoE on the Gemma-4 backbone. FastModel auto-detects the
    # diffusion model_type and routes to the transformers-only FastDiffusionModel slow path.
    # It is ~52GB in bf16: the 128 MoE experts alone are ~46GB and stay bf16 (fused 3D params, so bnb 4bit
    # cannot shrink them). Use a >= ~50GB GPU (A100 80GB / H100 / B200); a 40GB GPU offloads weights to the
    # meta/CPU device and the run becomes impractically slow.
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_gb < 50:
            print(
                f"[warn] {free_gb:.0f}GB free < ~52GB needed: weights will offload (meta/CPU) and run very "
                "slowly. Use an 80GB GPU (A100 80GB / H100) for a real run."
            )

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/diffusiongemma-26B-A4B-it",
        dtype=torch.bfloat16,
        load_in_4bit=False,  # 4bit cannot shrink the ~46GB of MoE experts, so it needs ~50GB either way
        # token = "YOUR_HF_TOKEN",
    )
    processor = (  # diffusion checkpoints ship a processor (chat template + tokenizer)
        tokenizer  # diffusion checkpoints ship a processor (chat template + tokenizer)
    )
    vocab = model.config.text_config.vocab_size
    canvas_len = model.config.canvas_length  # 256-token generation canvas
    print("vocab", vocab, "| canvas", canvas_len)
    return FastModel, canvas_len, model, processor, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We add LoRA adapters so we only train a few percent of the parameters. We target the attention and the dense MLP of the shared Gemma-4 backbone; the 128 fused MoE experts stay frozen.
    """)
    return


@app.cell
def _(FastModel, model):
    model_1 = FastModel.get_peft_model(
        model, r=64, lora_alpha=128, use_gradient_checkpointing=False
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    # Sudoku dataset

    We generate puzzle -> solution pairs procedurally, each with a **unique** solution (verified by a solver), so the target is the only correct answer. The 9x9 grid is written as 9 newline-separated rows of compact digits; the Gemma tokenizer splits digits individually, so every cell is exactly one token at a fixed canvas position. `0` marks an empty cell.
    """)
    return


@app.cell
def _():
    import random

    def _solve_count(g, limit=2):
        best, best_cands = (
            -1,
            None,
        )  # count solutions up to `limit` via MRV backtracking; g is list[81], 0 = empty
        for i in range(81):
            if g[i] != 0:
                continue
            r, c = divmod(i, 9)
            used = set()
            for k in range(9):
                used.add(g[r * 9 + k])
                used.add(g[k * 9 + c])
            br, bc = (r // 3 * 3, c // 3 * 3)
            for dr in range(3):
                for dc in range(3):
                    used.add(g[(br + dr) * 9 + (bc + dc)])
            cands = [d for d in range(1, 10) if d not in used]
            if not cands:
                return 0
            if best_cands is None or len(cands) < len(best_cands):
                best, best_cands = (i, cands)
                if len(cands) == 1:
                    break
        if best == -1:
            return 1
        total = 0  # random complete solution by shuffled backtracking
        for d in best_cands:
            g[best] = d
            total = total + _solve_count(g, limit)
            g[best] = 0
            if total >= limit:
                break
        return total

    def _full_grid(rng):
        g = [0] * 81

        def fill(i):
            if i == 81:
                return True
            if g[i] != 0:
                return fill(i + 1)
            r, c = divmod(i, 9)
            used = set()
            for k in range(9):  # remove `holes` cells while keeping the solution unique
                used.add(g[r * 9 + k])
                used.add(g[k * 9 + c])
            br, bc = (r // 3 * 3, c // 3 * 3)
            for dr in range(3):
                for dc in range(3):
                    used.add(g[(br + dr) * 9 + (bc + dc)])
            cands = [d for d in range(1, 10) if d not in used]
            rng.shuffle(cands)
            for d in cands:
                g[i] = d
                if fill(i + 1):
                    return True
                g[i] = 0
            return False

        fill(0)
        return g  # holes_lo..holes_hi givens

    def _make_puzzle(full, holes, rng):
        g = full[:]
        order = list(range(81))
        rng.shuffle(order)
        removed = 0
        for i in order:
            if removed >= holes:
                break
            # A few thousand puzzles is enough to see learning; scale up for higher solve rates.
            saved = g[i]
            g[i] = 0
            if _solve_count(g[:], 2) != 1:
                g[i] = saved
            else:
                removed = removed + 1
        return (g, removed)

    def _grid_str(g):
        return "\n".join(
            ("".join((str(g[r * 9 + c]) for c in range(9))) for r in range(9))
        )

    PROMPT = "Solve this Sudoku puzzle. 0 marks an empty cell. Reply with the completed 9x9 grid.\n{puzzle}"

    def make_example(seed, holes_lo=36, holes_hi=46):
        rng = random.Random(seed)
        full = _full_grid(rng)
        holes = rng.randint(81 - holes_hi, 81 - holes_lo)  # holes_lo..holes_hi givens
        puz, _ = _make_puzzle(full, holes, rng)
        return {
            "messages": [
                {"role": "user", "content": PROMPT.format(puzzle=_grid_str(puz))},
                {"role": "assistant", "content": _grid_str(full)},
            ],
            "puzzle": "".join(map(str, puz)),
            "solution": "".join(map(str, full)),
        }

    N_TRAIN, N_EVAL = (3000, 200)
    train_rows = [make_example(s) for s in range(N_TRAIN)]
    eval_rows = [make_example(s) for s in range(10000, 10000 + N_EVAL)]
    print(len(train_rows), "train /", len(eval_rows), "eval puzzles")
    return eval_rows, train_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A training example - the user message is the puzzle, the assistant message is the solved grid:
    """)
    return


@app.cell
def _(train_rows):
    print(train_rows[0]["messages"][0]["content"])
    print("---")
    print(train_rows[0]["messages"][1]["content"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    # Block-diffusion finetuning

    DiffusionGemma is not trained the autoregressive way, so we use Unsloth's `DiffusionTrainer`, which takes ordinary prompt / completion data and applies DiffusionGemma's own block-diffusion objective (the reference recipe of the released checkpoint):

    * The reply is cut into 256-token canvases, one picked per example, with the tail past the end filled with `eos`.
    * The canvas is **corrupted** by replacing each token with probability `t` by a random token, with `t` drawn per example.
    * Half the time the model first denoises the canvas once and then conditions on its own guess (**self-conditioning**), as it does at generation time.
    * The loss is cross-entropy against the clean canvas over every position, plus an autoregressive loss on the causal encoder.
    """)
    return


@app.cell
def _(train_rows):
    from datasets import Dataset
    from unsloth import DiffusionTrainer, DiffusionConfig

    # prompt = the puzzle, completion = the solved grid; only the completion is denoised.
    dataset = Dataset.from_list(
        [
            {"prompt": [r["messages"][0]], "completion": [r["messages"][1]]}
            for r in train_rows
        ]
    )
    print(dataset[0])
    return DiffusionConfig, DiffusionTrainer, dataset


@app.cell
def _(DiffusionConfig, DiffusionTrainer, dataset, model_1, processor):
    trainer = DiffusionTrainer(
        model=model_1,
        processing_class=processor,
        train_dataset=dataset,
        args=DiffusionConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=500,  # full run in our report: 4000 steps, 8 GPUs
            learning_rate=0.0001,
            warmup_steps=15,
            lr_scheduler_type="cosine",
            adam_beta2=0.95,
            weight_decay=0.0,
            logging_steps=20,
            completion_only_loss=True,
            max_length=512,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
            seed=3407,
        ),
    )
    trainer_stats = (
        trainer.train()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Eval"></a>
    # Evaluate: refinement is the point

    We solve held-out puzzles by block-diffusion generation and score the **exact-solve rate**. Crucially we sweep the number of denoising steps: one shot (predict the grid once) is weak, but letting the model **revise over steps** is where it wins.
    """)
    return


@app.cell
def _(canvas_len, eval_rows, model_1, processor, torch):
    import copy

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    dev = next(
        (p.device for p in model_1.parameters() if p.device.type != "meta"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # compute device, not a "meta" param (offloaded weights report device = meta)

    def parse_grid(text):
        ds = [int(ch) for ch in text if ch.isdigit()]
        return ds[:81] if len(ds) >= 81 else None

    def solve(prompt, steps):
        ids = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(dev)
        gc = copy.deepcopy(model_1.generation_config)
        gc.max_denoising_steps = steps
        gc.max_new_tokens = canvas_len
        torch.manual_seed(0)
        with torch.no_grad():
            out = model_1.generate(input_ids=ids, generation_config=gc)
        seq = out.sequences[0, ids.shape[1] :]
        return parse_grid(tok.decode(seq.tolist(), skip_special_tokens=True))

    def eval_solve_rate(rows, steps, n=50):
        solved = 0
        for r in rows[:n]:
            g = solve(r["messages"][0]["content"], steps)
            solved = solved + (g is not None and g == [int(c) for c in r["solution"]])
        return solved / min(n, len(rows))

    model_1.eval()
    for s in (1, 16, 64):
        print(
            f"{s:>2}-step exact-solve rate: {eval_solve_rate(eval_rows, s) * 100:.1f}%"
        )
    return (solve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In our full run (4000 steps, the report on the model card), the finetune takes the base model from **1.5% to 89.5%** exact-solve on medium puzzles, and the one-shot vs refined gap is **18% -> 89.5%** purely from revising over steps. An autoregressive LoRA baseline on the **same** data reaches only 14.5% and overwrites about a third of the given clues, because once it emits a cell it cannot reconcile it against constraints that appear later in the grid - the diffusion model keeps 100% of the givens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    # Solve a puzzle
    """)
    return


@app.cell
def _(eval_rows, solve):
    puzzle = eval_rows[0]["messages"][0]["content"]
    print(puzzle, "\n--- solved ---")
    g = solve(puzzle, steps=64)
    print(
        "\n".join("".join(str(g[r * 9 + c]) for c in range(9)) for r in range(9))
        if g
        else "(no 81-digit grid)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    # Save the LoRA
    """)
    return


@app.cell
def _(model_1, processor):
    model_1.save_pretrained("diffusiongemma_lora")
    processor.save_pretrained("diffusiongemma_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp

    Prebuilt GGUFs are at [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF). DiffusionGemma needs the diffusiongemma build of llama.cpp and its `llama-diffusion-cli` runner (see that model card); the standard llama.cpp conversion does not cover this architecture yet.
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
