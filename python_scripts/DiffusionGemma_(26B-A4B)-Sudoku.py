#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on your A100 Google Colab Pro instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it

# ### News

# Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)
# 
# <table><tr>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
# </tr></table>
# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # ### Unsloth
# 
# **Goal: finetune DiffusionGemma to solve Sudoku.** Sudoku is a global-constraint task - the answer must be consistent across the whole 9x9 grid at once, and solving means going back to **revise** wrong cells. An autoregressive model commits each cell left to right and cannot take it back; DiffusionGemma denoises the entire grid in parallel and revises cells over steps, which is exactly the capability this task needs.
# 
# DiffusionGemma needs a transformers build that ships the DiffusionGemma classes; `FastModel` auto-detects the diffusion architecture and routes to the transformers-only slow path.

# In[ ]:


from unsloth import FastModel
import torch

# DiffusionGemma is a 26B-A4B block-diffusion MoE on the Gemma-4 backbone. FastModel auto-detects the
# diffusion model_type and routes to the transformers-only FastDiffusionModel slow path.
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/diffusiongemma-26B-A4B-it",
    dtype = torch.bfloat16,
    load_in_4bit = False,  # set True to fit a smaller GPU
    # token = "YOUR_HF_TOKEN",
)
processor = tokenizer  # diffusion checkpoints ship a processor (chat template + tokenizer)
vocab = model.config.text_config.vocab_size
canvas_len = model.config.canvas_length  # 256-token generation canvas
print("vocab", vocab, "| canvas", canvas_len)


# We add LoRA adapters so we only train a few percent of the parameters. We target the attention and the dense MLP of the shared Gemma-4 backbone; the 128 fused MoE experts stay frozen.

# In[ ]:


model = FastModel.get_peft_model(
    model,
    r = 64,            # LoRA rank
    lora_alpha = 128,  # alpha = 2*r
    use_gradient_checkpointing = False,
)


# <a name="Data"></a>
# # Sudoku dataset
# 
# We generate puzzle -> solution pairs procedurally, each with a **unique** solution (verified by a solver), so the target is the only correct answer. The 9x9 grid is written as 9 newline-separated rows of compact digits; the Gemma tokenizer splits digits individually, so every cell is exactly one token at a fixed canvas position. `0` marks an empty cell.

# In[ ]:


import random

def _solve_count(g, limit = 2):
    # count solutions up to `limit` via MRV backtracking; g is list[81], 0 = empty
    best, best_cands = -1, None
    for i in range(81):
        if g[i] != 0: continue
        r, c = divmod(i, 9); used = set()
        for k in range(9): used.add(g[r*9+k]); used.add(g[k*9+c])
        br, bc = (r//3)*3, (c//3)*3
        for dr in range(3):
            for dc in range(3): used.add(g[(br+dr)*9+(bc+dc)])
        cands = [d for d in range(1, 10) if d not in used]
        if not cands: return 0
        if best_cands is None or len(cands) < len(best_cands):
            best, best_cands = i, cands
            if len(cands) == 1: break
    if best == -1: return 1
    total = 0
    for d in best_cands:
        g[best] = d; total += _solve_count(g, limit); g[best] = 0
        if total >= limit: break
    return total

def _full_grid(rng):
    # random complete solution by shuffled backtracking
    g = [0]*81
    def fill(i):
        if i == 81: return True
        if g[i] != 0: return fill(i+1)
        r, c = divmod(i, 9); used = set()
        for k in range(9): used.add(g[r*9+k]); used.add(g[k*9+c])
        br, bc = (r//3)*3, (c//3)*3
        for dr in range(3):
            for dc in range(3): used.add(g[(br+dr)*9+(bc+dc)])
        cands = [d for d in range(1, 10) if d not in used]; rng.shuffle(cands)
        for d in cands:
            g[i] = d
            if fill(i+1): return True
            g[i] = 0
        return False
    fill(0); return g

def _make_puzzle(full, holes, rng):
    # remove `holes` cells while keeping the solution unique
    g = full[:]; order = list(range(81)); rng.shuffle(order); removed = 0
    for i in order:
        if removed >= holes: break
        saved = g[i]; g[i] = 0
        if _solve_count(g[:], 2) != 1: g[i] = saved
        else: removed += 1
    return g, removed

def _grid_str(g):
    return "\n".join("".join(str(g[r*9+c]) for c in range(9)) for r in range(9))

PROMPT = "Solve this Sudoku puzzle. 0 marks an empty cell. Reply with the completed 9x9 grid.\n{puzzle}"

def make_example(seed, holes_lo = 36, holes_hi = 46):
    rng = random.Random(seed); full = _full_grid(rng)
    holes = rng.randint(81 - holes_hi, 81 - holes_lo)  # holes_lo..holes_hi givens
    puz, _ = _make_puzzle(full, holes, rng)
    return {
        "messages": [
            {"role": "user", "content": PROMPT.format(puzzle = _grid_str(puz))},
            {"role": "assistant", "content": _grid_str(full)},
        ],
        "puzzle": "".join(map(str, puz)), "solution": "".join(map(str, full)),
    }

# A few thousand puzzles is enough to see learning; scale up for higher solve rates.
N_TRAIN, N_EVAL = 3000, 200
train_rows = [make_example(s) for s in range(N_TRAIN)]
eval_rows  = [make_example(s) for s in range(10_000, 10_000 + N_EVAL)]
print(len(train_rows), "train /", len(eval_rows), "eval puzzles")


# A training example - the user message is the puzzle, the assistant message is the solved grid:

# In[ ]:


print(train_rows[0]["messages"][0]["content"])
print("---")
print(train_rows[0]["messages"][1]["content"])


# <a name="Train"></a>
# # Block-diffusion finetuning
# 
# DiffusionGemma is not trained the autoregressive way (no `SFTTrainer`). Instead we use its own block-diffusion objective: pad the target solution to the 256-token canvas, **corrupt** the canvas by replacing each token with probability `t` by a random token, then ask the model to predict the clean grid. The loss is cross-entropy on the solution tokens plus the `eos` (the padding tail is ignored).

# In[ ]:


eos = (model.generation_config.eos_token_id or [1])
eos = eos[0] if isinstance(eos, (list, tuple)) else eos
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
pad = tok.pad_token_id if tok.pad_token_id is not None else eos

def build_examples(rows):
    out = []
    for r in rows:
        prompt_ids = processor.apply_chat_template(
            [r["messages"][0]], tokenize = True, add_generation_prompt = True, return_tensors = "pt")[0]
        ids = tok.encode(r["messages"][1]["content"], add_special_tokens = False)
        content = ids + [eos]
        n = len(content)
        if n > canvas_len: continue
        x0 = torch.tensor(content + [pad]*(canvas_len - n), dtype = torch.long)
        mask = torch.zeros(canvas_len, dtype = torch.bool); mask[:n] = True
        out.append((prompt_ids, x0, mask))
    return out

examples = build_examples(train_rows)
print("usable examples:", len(examples))


# In[ ]:


import time
dev = next(model.parameters()).device
STEPS, GRAD_ACCUM, LR, T_LO = 500, 4, 1e-4, 0.1  # full run in our report: 4000 steps, 8 GPUs

model.config.use_cache = True
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = LR,
                        betas = (0.9, 0.95), weight_decay = 0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr = LR, total_steps = STEPS,
                                            pct_start = 0.03, anneal_strategy = "cos")

def corrupt(x0):
    t = random.uniform(T_LO, 1.0)            # noise level
    xt = x0.to(dev).clone()
    m = torch.rand(canvas_len, device = dev) < t
    xt[m] = torch.randint(0, vocab, (canvas_len,), device = dev)[m]
    return xt.unsqueeze(0)

order = list(range(len(examples))); ptr = 0; t0 = time.time()
opt.zero_grad(set_to_none = True)
for step in range(1, STEPS + 1):
    step_loss = 0.0
    for _ in range(GRAD_ACCUM):
        if ptr >= len(order): random.shuffle(order); ptr = 0
        prompt_ids, x0, lm = examples[order[ptr]]; ptr += 1
        out = model(input_ids = prompt_ids.unsqueeze(0).to(dev), canvas_ids = corrupt(x0),
                    self_conditioning_logits = None)
        logits = out.logits[0].float()       # [canvas_len, vocab]
        m = lm.to(dev)
        loss = torch.nn.functional.cross_entropy(logits[m], x0.to(dev)[m])
        (loss / GRAD_ACCUM).backward(); step_loss += loss.item() / GRAD_ACCUM
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none = True)
    if step % 20 == 0:
        print(f"step {step:4d}/{STEPS} | loss {step_loss:.4f} | {time.time()-t0:.0f}s", flush = True)


# <a name="Eval"></a>
# # Evaluate: refinement is the point
# 
# We solve held-out puzzles by block-diffusion generation and score the **exact-solve rate**. Crucially we sweep the number of denoising steps: one shot (predict the grid once) is weak, but letting the model **revise over steps** is where it wins.

# In[ ]:


import copy

def parse_grid(text):
    ds = [int(ch) for ch in text if ch.isdigit()]
    return ds[:81] if len(ds) >= 81 else None

def solve(prompt, steps):
    ids = processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize = True,
                                        add_generation_prompt = True, return_tensors = "pt").to(dev)
    gc = copy.deepcopy(model.generation_config)
    gc.max_denoising_steps = steps; gc.max_new_tokens = canvas_len
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(input_ids = ids, generation_config = gc)
    seq = out.sequences[0, ids.shape[1]:]
    return parse_grid(tok.decode(seq.tolist(), skip_special_tokens = True))

def eval_solve_rate(rows, steps, n = 50):
    solved = 0
    for r in rows[:n]:
        g = solve(r["messages"][0]["content"], steps)
        solved += (g is not None and g == [int(c) for c in r["solution"]])
    return solved / min(n, len(rows))

model.eval()
for s in (1, 16, 64):
    print(f"{s:>2}-step exact-solve rate: {eval_solve_rate(eval_rows, s)*100:.1f}%")


# In our full run (4000 steps, the report on the model card), the finetune takes the base model from **1.5% to 89.5%** exact-solve on medium puzzles, and the one-shot vs refined gap is **18% -> 89.5%** purely from revising over steps. An autoregressive LoRA baseline on the **same** data reaches only 14.5% and overwrites about a third of the given clues, because once it emits a cell it cannot reconcile it against constraints that appear later in the grid - the diffusion model keeps 100% of the givens.

# <a name="Inference"></a>
# # Solve a puzzle

# In[ ]:


puzzle = eval_rows[0]["messages"][0]["content"]
print(puzzle, "\n--- solved ---")
g = solve(puzzle, steps = 64)
print("\n".join("".join(str(g[r*9+c]) for c in range(9)) for r in range(9)) if g else "(no 81-digit grid)")


# <a name="Save"></a>
# # Save the LoRA

# In[ ]:


model.save_pretrained("diffusiongemma_lora")
processor.save_pretrained("diffusiongemma_lora")
# model.push_to_hub("HF_ACCOUNT/diffusiongemma_lora", token = "YOUR_HF_TOKEN")

# Keep the adapter unmerged for inference (merging can corrupt the clipped LoRA linears).


# ### GGUF / llama.cpp
# 
# Prebuilt GGUFs are at [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF). DiffusionGemma needs the diffusiongemma build of llama.cpp and its `llama-diffusion-cli` runner (see that model card); the standard llama.cpp conversion does not cover this architecture yet.
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
# 2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
# 3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
# 4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
# 5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
