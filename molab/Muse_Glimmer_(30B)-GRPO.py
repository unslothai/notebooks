# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "marimo",
#     "safetensors>=0.8.0",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch>=2.8.0",
#     "torchao>=0.16.0",
#     "torchvision",
#     "transformers @ git+https://github.com/huggingface/transformers.git@fe95f5423d65951cf63055d519dd7fa5ae12eb8d",
#     "triton>=3.2.0",
#     "trl==1.9.2",
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
    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/blob/main/images/Discord button.png?raw=true" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
    </div>

    You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), and how to save it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hardware note: GRPO on this model needs more than 2x16 GB

    SFT, vision and multimodal fine-tuning of this checkpoint all run on a free Kaggle
    T4 x2 kernel. GRPO does not, and the reason is specific rather than general size.

    The 4-bit weights split evenly across the two cards, measured at 10.01 GiB on cuda:0
    and 10.30 GiB on cuda:1. The `lm_head` is 202048 x 6656 and lands on the last card,
    and GRPO's chunked log-softmax then materialises `tokens x 202048` logits plus a
    float32 copy of them on that same card. That leaves roughly 4.2 GiB for the logits,
    the rollout KV cache and the gradients together, and it is not enough.

    Two changes below take it as far as 2x16 GB will go, and both are worth keeping on
    any tier: `num_generations = 2` instead of 4, and capping the log-softmax at 128
    rows per chunk, which cuts peak logit memory from 994 MiB to 493 MiB at this vocab
    size. With both applied the run still stops, but the failing allocation falls from
    260 MiB to 50 MiB, so the shortfall is small rather than structural.

    Capping the last card with `max_memory` to reserve headroom does not help: accelerate
    balances rather than filling the first card, so the cap just pushes modules to CPU and
    bitsandbytes refuses. Forcing it would need a hand-written `device_map`.

    **Run this notebook on a single card with more memory** (A100 40GB, H100, or an L4 for
    a slower run). On one GPU `offload_embedding` also switches back on, which frees a
    further 2.5 GB, and none of the multi-GPU placement issues apply.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Goal: teach Muse Glimmer to answer grade school maths on the right channel with Reinforcement Learning

    Muse Glimmer already thinks before it answers. It writes its working out on a private `to=self` channel and
    then repeats itself on the `to=user` channel that the end user actually reads. That makes it a very
    natural fit for [GRPO](https://arxiv.org/abs/2402.03300): the reasoning is already separated from the
    answer, so a reward function can grade the answer channel on its own and let the model keep whatever
    private reasoning got it there.

    We use [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset), a set
    of competition maths problems (AIME, AMC, Omni-MATH) each with one short final answer. The reward is
    **verifiable**: we parse the number the model put on the `to=user` channel and compare it to the
    gold answer. There is no reward model, no judge and nothing to disagree about, which is the safest
    kind of RL task to start from.

    Four reasons this task suits Muse Glimmer specifically:

    1. The reward can be checked exactly, so a broken reward function shows up immediately as a flat
       reward curve rather than as a slow drift you only notice at the end.
    2. It is **hard enough to leave headroom**. That matters more than it sounds. GRPO learns from the
       spread of rewards inside a group, so if the model already gets every prompt right the advantage
       is zero and nothing happens. Muse Glimmer solves grade school maths almost perfectly, so GSM8K gives a
       flat reward at the ceiling from the first step. Competition problems do not.
    3. The prompts are short, a couple of hundred tokens, which matters a lot here because Muse Glimmer is a 28B
       model and GRPO generates several completions per prompt.
    4. It is a text only task. Muse Glimmer carries a 50 layer vision tower, and keeping it out of the rollout
       loop keeps both the step time and the memory down.

    ### One important thing about generation

    Most GRPO notebooks turn on `fast_inference = True` to generate rollouts with vLLM. **That is not
    available for Muse Glimmer.** vLLM has no `muse-glimmer` architecture, so there is nothing for it to load. This
    notebook generates rollouts with Unsloth and `transformers` instead, which is exactly what the
    Ministral Sudoku and gpt-oss 2048 RL notebooks do. It works, it is just slower per step. Keep
    `num_generations` small and be patient.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Installation

    Muse Glimmer is not in a released `transformers` yet, so we install a private build of `transformers` that
    has the `muse-glimmer` architecture in it. Everything else is a normal Unsloth install.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we pick up the token. Never paste a token into a cell. Read it from the environment, from
    Kaggle Secrets or from molab secrets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Muse Glimmer is a brand new architecture, so it needs a `transformers` build that knows about it. Support was merged in [PR #47867](https://github.com/huggingface/transformers/pull/47867) but is not in a tagged release yet, so we install from the merge commit. Everything used below is public - no Hugging Face token is required.
    """)
    return


@app.cell
def _():
    # Muse Glimmer support landed in transformers via PR #47867 but is not in a
    # tagged release yet, so install from the merge commit. Pinned rather than
    # tracking main so the notebook does not change underneath you. Run this BEFORE
    # anything imports transformers.
    # This build needs safetensors >= 0.8.0; some images ship 0.7.0.

    import transformers

    print("transformers:", transformers.__version__)
    from transformers import AutoConfig

    _cfg = AutoConfig.from_pretrained("unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit")
    print("architecture available:", _cfg.model_type)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which repo to load

    Two public repos hold the same weights, and both work with `FastModel.from_pretrained`:

    | repo | precision | size | use it when |
    |---|---|---|---|
    | [`unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit) | 4-bit (bitsandbytes nf4) | ~21 GB | fine-tuning on one or two consumer GPUs. This is what the cells below use. |
    | [`unsloth/Muse-Glimmer-30B`](https://huggingface.co/unsloth/Muse-Glimmer-30B) | bf16 | ~56 GB | full-precision inference or full fine-tuning on large-memory hardware. |

    GGUFs for llama.cpp are at
    [`unsloth/Muse-Glimmer-30B-GGUF`](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF),
    from 10 GB up to bf16, plus the vision projector.
    """)
    return


@app.cell
def _():
    import transformers as _molab_transformers

    print("transformers", _molab_transformers.__version__)
    from transformers import AutoConfig as _molab_AutoConfig

    _cfg = _molab_AutoConfig.from_pretrained(
        "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
    )
    print("architecture available:", _cfg.model_type)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Muse Glimmer

    Muse Glimmer is a 28B dense text and vision model. In 4-bit the weights are about 20.7 GiB before anything
    else, so we load the pre-quantised repository directly rather than quantising at load time.

    * `load_in_4bit = True` keeps the 416 text `Linear4bit` modules at nf4 with double quant, about
      11.7 GB. The embeddings and the vision tower stay in 16-bit.
    * `offload_embedding = True` moves the input embedding to CPU RAM. Muse Glimmer has a 202048 token
      vocabulary and untied embeddings, so `embed_tokens` alone is 2.5 GiB.
    * `fast_inference = False` because no released vLLM carries this architecture. See the note below.
    * `max_seq_length = 3072` is prompt plus completion. The prompts are only a couple of hundred
      tokens, so nearly all of that is completion budget. It has to be, see the length measurement
      further down.

    Muse Glimmer is registered as an image-text-to-text model, so `AutoModelForCausalLM` will not load it.
    `FastModel` picks the right class for you.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Muse Glimmer is a brand new architecture, so it needs a `transformers` build that knows about it. Support was merged in [PR #47867](https://github.com/huggingface/transformers/pull/47867) but is not in a tagged release yet, so we install from the merge commit. Everything used below is public - no Hugging Face token is required.
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import sys, torch

    MODEL_NAME = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
    lora_rank = 8  # Larger rank = smarter, but slower and more memory

    # Size the run from the hardware. Rollout width and sequence length are the two
    # memory levers: the log-softmax keeps every row's logits alive until backward
    # (the retained term scales with TOTAL rows, not chunk rows), so on 2 x 16 GB a
    # 3072-token sequence needs 7.57 GiB free on the head's card, which does not fit
    # alongside the weights. 1536 needs 4.10 GiB and leaves 1.29 GiB of margin.
    _n = max(torch.cuda.device_count(), 1)
    _gib = (
        min(torch.cuda.get_device_properties(i).total_memory for i in range(_n)) / 2**30
    )
    _big = _gib >= 39
    NUM_GENERATIONS = 4 if _big else 2
    BATCH_SIZE = NUM_GENERATIONS
    max_seq_length = 3072 if _big else 1536  # prompt + completion
    max_prompt_length = 384  # comfortably above the observed maximum
    max_completion_length = max_seq_length - max_prompt_length
    print(
        f"{_n} GPU(s), smallest {_gib:.1f} GiB -> num_generations={NUM_GENERATIONS}, "
        f"max_seq_length={max_seq_length}"
    )
    OFFLOAD_EMBEDDING = _n == 1

    # Head-aware placement. Returns None on a single GPU, and raises rather than
    # silently spilling to CPU when the request cannot fit.
    DEVICE_MAP = None
    if _n > 1:
        sys.path.insert(0, ".")
        from unsloth_zoo.device_map_planner import plan_device_map_for_pretrained

        _plan = plan_device_map_for_pretrained(
            MODEL_NAME,
            max_memory={i: f"{_gib:.2f}GiB" for i in range(_n)},
            rows_per_chunk=128,  # matches the log-softmax cap below
            retained_rows=BATCH_SIZE * max_seq_length,
            softcapped=True,
            temperature_scaled=True,
            # Activations follow layers: each layer moved onto card 0 costs it more peak
            # than it frees on the head card, so hold some back for card 0 itself.
            # Left to the planner. An explicit reserve is treated as a measured
            # figure the plan must honour exactly (unsloth_zoo sets steps = 1 when it
            # is explicit), so on 2 x 14.6 GiB a hard 3.2 GiB request cannot be met
            # alongside the weights and the logit headroom, and planning fails with
            # DeviceMapInfeasible. Auto-sizing derives the reserve from the actual
            # slack and still relaxes it on the non-head cards.
        )
        if _plan is not None:
            print(_plan.describe())
            DEVICE_MAP = _plan.device_map

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,  # prompt + completion
        load_in_4bit=True,  # 4-bit QLoRA. False needs an 80GB card
        offload_embedding=OFFLOAD_EMBEDDING,  # 202048 token vocabulary, keep it off the GPU
        fast_inference=False,  # no released vllm has this architecture
        device_map=DEVICE_MAP,
    )
    print(model.config.model_type, type(model).__name__)
    return (
        BATCH_SIZE,
        FastModel,
        NUM_GENERATIONS,
        lora_rank,
        max_seq_length,
        model,
        sys,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One thing has to be corrected before anything else. Muse Glimmer has two end tokens. `<|eot|>` ends a turn
    and is the one that should stop generation, and `<|end_of_text|>` is the raw pretraining end of
    document token. The tokenizer reports `<|end_of_text|>` as its `eos_token`, and TRL takes its stop
    token from the tokenizer rather than from the model's `generation_config`, which does list both.

    Left alone, that means rollouts never stop at the end of the reply. The model finishes its answer,
    emits `<|eot|>`, and then simply starts another assistant turn and keeps going until it hits the
    length limit. Every completion comes back truncated, and with `mask_truncated_completions` on, every
    completion gets masked out of the loss. Point the tokenizer at `<|eot|>` and it behaves.
    """)
    return


@app.cell
def _(model, tokenizer):
    # tokenizer is an Muse GlimmerProcessor, the text tokenizer is one level down
    _text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    print("eos before:", _text_tokenizer.eos_token, _text_tokenizer.eos_token_id)
    _text_tokenizer.eos_token = "<|eot|>"
    print("eos after :", _text_tokenizer.eos_token, _text_tokenizer.eos_token_id)

    # keep the model's own generation config listing both, so plain model.generate() also stops
    model.generation_config.eos_token_id = sorted(
        {
            _text_tokenizer.eos_token_id,
            _text_tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        }
    )
    print("generation_config.eos_token_id:", model.generation_config.eos_token_id)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the LoRA adapters. We only train the language layers. The vision tower is not used by this task
    at all, so adapting it would just cost memory for gradients that never see a useful signal.
    """)
    return


@app.cell
def _(FastModel, lora_rank, model):
    model_1 = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # text only task, leave the vision tower frozen
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### The Muse Glimmer chat format

    Muse Glimmer does not use ChatML and it does not use the Llama format. It uses ATEM, where every turn is
    `<|start|>ROLE<|message|>...<|eot|>` and the assistant additionally declares who it is talking to
    with `to=self` for private reasoning and `to=user` for the reply. The private channel is terminated
    by `<|eom|>`, which is deliberately not an end of generation token, so the model keeps going and
    opens a second assistant turn for the real answer.

    Rather than hardcoding those strings, we recover them from the tokenizer by rendering a probe
    conversation with a sentinel answer in it and looking at what ends up either side.
    """)
    return


@app.cell
def _(tokenizer):
    SENTINEL = "SENTINELANSWER"

    _probe = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "probe"},
            {"role": "assistant", "content": SENTINEL},
        ],
        tokenize=False,
    )
    _head, _tail = _probe.split(SENTINEL)

    ANSWER_PREFIX = _head[_head.rindex("<|start|>") :]  # opens the reply channel
    ANSWER_SUFFIX = _tail  # closes it

    print("generation prompt ends with:")
    print(
        repr(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "probe"}],
                tokenize=False,
                add_generation_prompt=True,
            )[-40:]
        )
    )
    print()
    print("ANSWER_PREFIX =", repr(ANSWER_PREFIX))
    print("ANSWER_SUFFIX =", repr(ANSWER_SUFFIX))
    return ANSWER_PREFIX, ANSWER_SUFFIX


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So a full Muse Glimmer completion looks like this, starting from the generation prompt which stops right
    after `<|start|>assistant`:

    ```
     to=self<|message|>48 / 2 = 24, and 48 + 24 = 72.<|eom|><|start|>assistant to=user<|message|>72<|eot|>
    ```

    There is one wrinkle that matters for the reward functions. TRL decodes completions with
    `skip_special_tokens=True` before handing them to us, and `<|start|>`, `<|message|>`, `<|eom|>` and
    `<|eot|>` are all registered as special tokens. So what a reward function actually receives is the
    same string with the markers removed:

    ```
     to=selfLet me work it out. 48 / 2 = 24, and 48 + 24 = 72.assistant to=user72
    ```

    The `to=self` and `to=user` recipients survive, because they are ordinary text rather than special
    tokens. That is enough to split the channels reliably. The helper below handles both forms, so it
    keeps working if you ever decode the rollouts yourself with the markers left in.
    """)
    return


@app.cell
def _(ANSWER_PREFIX, ANSWER_SUFFIX, tokenizer):
    RECIPIENT_USER = (  # "user"
        ANSWER_PREFIX.split("to=")[1].split("<|message|>")[0].strip()
    )  # "user"
    USER_CHANNEL = "to=" + RECIPIENT_USER
    print("splitting the reply channel on", repr(USER_CHANNEL))

    def split_channels(text):
        """Return (reasoning, answer) from an Muse Glimmer completion.

        Works whether or not the ATEM special tokens survived decoding. The reply is whatever
        follows the last to = user marker; anything before the first one is private reasoning.
        """
        if text is None:
            return "", ""
        idx = text.rfind(USER_CHANNEL)
        if idx == -1:
            return text.strip(), ""  # never opened a reply channel
        reasoning = text[:idx]
        answer = text[idx + len(USER_CHANNEL) :]
        # drop whatever markup is left over on either side
        for marker in (
            "<|message|>",
            "<|eom|>",
            "<|eot|>",
            "<|start|>",
            "assistant",
            "to=self",
        ):
            reasoning = reasoning.replace(marker, " ")
        answer = answer.replace("<|message|>", "")
        for marker in ("<|eom|>", "<|eot|>", "<|start|>"):
            answer = answer.split(marker)[0]
        return reasoning.strip(), answer.strip()

    # sanity check on both decodings of the same completion
    _with = (
        " to=self<|message|>48/2 = 24, 48+24 = 72.<|eom|>"
        + ANSWER_PREFIX
        + "72"
        + ANSWER_SUFFIX
    )
    # tokenizer is an Muse GlimmerProcessor, so text has to be passed by keyword
    _ids = tokenizer(text=_with, add_special_tokens=False)["input_ids"]
    if len(_ids) and isinstance(_ids[0], (list, tuple)):
        _ids = _ids[0]
    _without = tokenizer.decode(_ids, skip_special_tokens=True)
    print("markers kept   ->", split_channels(_with))
    print("markers dropped ->", split_channels(_without))
    assert split_channels(_with)[1] == "72" and split_channels(_without)[1] == "72"
    return (split_channels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data prep

    DeepScaleR answers are already reduced to a single final answer, but some of them are LaTeX
    expressions rather than plain numbers. We keep only the problems whose answer is a bare integer or
    decimal, so the reward function has something exact to compare against. That still leaves about
    25000 problems out of 40315.

    The Muse Glimmer chat template also takes a `reasoning_strength`, which it renders into the system turn as
    a `Reasoning strength: ...` line. It defaults to `high`, and on high the model will happily spend
    several hundred tokens restating the question before it answers. We set it to `low`, which is the
    same trick the gpt-oss RL notebook uses with `reasoning_effort`. TRL passes anything you put in a
    `chat_template_kwargs` column straight through to `apply_chat_template`, so it travels with the
    dataset.
    """)
    return


@app.cell
def _():
    import re
    from datasets import load_dataset

    REASONING_STRENGTH = (  # "low" / "medium" / "high", rendered into the system turn
        "low"  # "low" / "medium" / "high", rendered into the system turn
    )

    SYSTEM_PROMPT = (
        "You are solving competition mathematics problems.\n"
        "Work the problem out step by step in your private reasoning, then reply to the user with the "
        "final answer as a bare number and nothing else. No units, no words, no working out in the reply."
    )

    IS_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")

    raw = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    raw = raw.filter(
        lambda x: (
            IS_NUMERIC.match(str(x["answer"]).strip().replace(",", "")) is not None
        )
    )
    print("problems with a plain numeric answer:", len(raw))

    dataset = raw.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["problem"]},
            ],
            "answer": str(x["answer"]).strip().replace(",", ""),
            # the Muse Glimmer chat template takes a reasoning_strength, default "high". TRL forwards anything in
            # chat_template_kwargs straight to apply_chat_template, and "low" keeps the private reasoning
            # channel short enough that the reply fits inside max_completion_length
            "chat_template_kwargs": {"reasoning_strength": REASONING_STRENGTH},
        },
        remove_columns=raw.column_names,
    )

    print(dataset[0]["prompt"][1]["content"][:400])
    print("gold:", dataset[0]["answer"])
    return REASONING_STRENGTH, dataset, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We check how long the prompts actually are so we can set the length budget honestly instead of
    guessing. Anything longer than the cut off gets dropped, because a truncated question has no correct
    answer and would only inject noise into the reward.

    Prompts are the easy half. The measurement that actually decides this notebook is how long the
    *completions* need to be. Sampling 24 problems from each of two datasets at `reasoning_strength`
    low, and letting the model run to 2560 tokens:

    | dataset | median | 90th pct | hit the 2560 cap | answered correctly |
    |---|---|---|---|---|
    | GSM8K | 148 | 352 | 0 of 24 | 22 of 24 |
    | DeepScaleR, numeric answers | 454 | 2560 | 3 of 24 | 19 of 24 |

    Two things follow. Grade school problems are already solved, 22 out of 24, so GRPO on them would
    spend most steps with every rollout in the group scoring identically and an advantage of exactly
    zero. And competition problems need a completion budget in the low thousands, not the few hundred
    tokens a smaller model would need. Both of those are set accordingly below.
    """)
    return


@app.cell
def _(REASONING_STRENGTH, dataset, max_seq_length, tokenizer):
    import numpy as np

    def prompt_length(messages):
        """Token count of a rendered prompt.

        tokenizer is an Muse GlimmerProcessor, and its apply_chat_template returns a string here rather
        than ids, so render first and tokenize explicitly instead of taking len() of the result.
        """
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_strength=REASONING_STRENGTH,
        )
        ids = tokenizer(text=text, add_special_tokens=False)["input_ids"]
        if len(ids) and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)

    lengths = [prompt_length(p) for p in dataset["prompt"][:1000]]
    print(
        "prompt tokens: median",
        int(np.median(lengths)),
        " 90th pct",
        int(np.percentile(lengths, 90)),
        " max",
        max(lengths),
    )
    max_prompt_length_1 = 384
    max_completion_length_1 = max_seq_length - max_prompt_length_1
    print(
        "max_prompt_length",
        max_prompt_length_1,
        "max_completion_length",
        max_completion_length_1,
    )
    dataset_1 = dataset.filter(
        lambda x: prompt_length(x["prompt"]) <= max_prompt_length_1
    )
    print("kept", len(dataset_1), "examples")  # comfortably above the observed maximum
    return dataset_1, max_completion_length_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Muse Glimmer before any training

    Let us see what the base model does. Note that generation does not stop at `<|eom|>` on purpose, so
    you should see the private reasoning channel, then a second assistant turn with the reply.
    """)
    return


@app.cell
def _(
    REASONING_STRENGTH,
    dataset_1,
    max_completion_length_1,
    model_1,
    tokenizer,
):
    from transformers import TextStreamer

    text = tokenizer.apply_chat_template(
        dataset_1[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength=REASONING_STRENGTH,
    )
    inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )
    _ = model_1.generate(
        **inputs,
        max_new_tokens=max_completion_length_1,
        temperature=1.0,
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
    )
    print("\ngold answer:", dataset_1[0]["answer"])
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reward functions

    Three rewards, all cheap and all verifiable:

    1. `channel_format` rewards opening a `to=user` channel at all, with a small bonus for reasoning
       privately first. Without this the model can learn to dump its working out straight at the user.
    2. `answer_is_a_number` rewards putting exactly one number on the reply channel, which is what the
       system prompt asked for. This is the shaping term that gets the model moving early.
    3. `answer_is_correct` is the real objective: does the number match the gold answer.

    The magnitudes are deliberately ordered so that being correct dominates being tidy.
    """)
    return


@app.cell
def _(re, split_channels):
    NUMBER = re.compile("-?\\d+(?:\\.\\d+)?")

    def _completion_text(completion):
        if isinstance(completion, list):
            return completion[0]["content"] if completion else ""
        return completion

    def channel_format(completions, **kwargs):
        scores = []
        for completion in completions:
            reasoning, answer = split_channels(_completion_text(completion))
            score = 0.0  # reward opening the reply channel, and thinking first
            if answer:
                score = (  # reward opening the reply channel, and thinking first
                    score + 1.0
                )  # reward opening the reply channel, and thinking first
            else:  # opened a reply channel and put something on it
                score = score - 1.0  # never replied to the user
            if reasoning:  # thought about it privately first
                score = score + 0.5  # reward opening the reply channel, and thinking first
            scores.append(score)
        return scores

    def answer_is_a_number(completions, **kwargs):
        scores = []
        for completion in completions:
            _, answer = split_channels(_completion_text(completion))
            found = NUMBER.findall(answer.replace(",", ""))
            if (
                len(found) == 1 and answer.replace(",", "").strip() == found[0]
            ):  # a bare number, exactly as asked
                scores.append(1.0)
            elif len(found) >= 1:  # a number, but wrapped in other text
                scores.append(0.25)
            else:
                scores.append(-0.5)
        return scores

    PRINT_EVERY = 5
    _printed = 0

    def answer_is_correct(prompts, completions, answer, **kwargs):
        global _printed
        scores = []
        for completion, gold in zip(completions, answer):
            text = _completion_text(completion)
            reasoning, reply = split_channels(text)
            found = NUMBER.findall(reply.replace(",", ""))
            got = found[-1] if found else None
            correct = False
            if got is not None:
                try:
                    correct = abs(float(got) - float(gold)) < 1e-06
                except ValueError:
                    correct = False
            scores.append(4.0 if correct else -0.5)
            if _printed % PRINT_EVERY == 0:
                print("-" * 60)
                print("reasoning:", reasoning[:300])
                print("reply    :", repr(reply[:120]))
                print("got", got, "gold", gold, "->", "correct" if correct else "wrong")
            _printed = _printed + 1
        return scores

    return answer_is_a_number, answer_is_correct, channel_format


@app.cell
def _(
    ANSWER_PREFIX,
    ANSWER_SUFFIX,
    answer_is_a_number,
    answer_is_correct,
    channel_format,
):
    # quick check that the rewards behave before we spend any GPU time on them
    _good = (
        " to=self<|message|>48/2=24, 48+24=72.<|eom|>"
        + ANSWER_PREFIX
        + "72"
        + ANSWER_SUFFIX
    )
    _bad = (
        " to=self<|message|>I think it is about seventy.<|eom|>"
        + ANSWER_PREFIX
        + "about seventy"
        + ANSWER_SUFFIX
    )
    _none = " to=self<|message|>72<|eom|>"
    for name, c in [("good", _good), ("bad", _bad), ("no reply", _none)]:
        print(
            f"{name:9s}",
            channel_format([c]),
            answer_is_a_number([c]),
            answer_is_correct(prompts=[None], completions=[c], answer=["72"]),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    A few notes on the settings, because the memory budget on a 28B model in 4-bit is tight:

    * `num_generations = 4` is the group size. GRPO needs at least 2 to have anything to compare, and
      every extra completion is another full sequence held in memory during the log probability passes.
    * `per_device_train_batch_size = 4` with `num_generations = 4` means one unique question per step.
      Raise `gradient_accumulation_steps` rather than the batch size if you want smoother updates.
    * `max_completion_length = 2688` is sized from the measurement above, not guessed. It is the single
      most important setting here. Competition problems need a long private reasoning channel, and a
      completion that runs out of budget never reaches its reply, so it scores as a formatting failure
      no matter how good the reasoning was. At 896 tokens 16 of 20 steps came back fully truncated and
      the whole step contributed nothing.
    * We use GSPO, which does importance sampling at the sequence level instead of the token level and
      is noticeably more stable on long completions.
    * `mask_truncated_completions = True` drops rollouts that hit the length limit. A truncated
      completion has no reply channel, so it would otherwise be graded as a formatting failure for a
      reason that has nothing to do with the policy.
    * `optim = "adamw_8bit"` because at rank 8 the optimiser state is small, but there is no reason to
      spend the memory anyway.

    Leave compilation alone. Do not set `UNSLOTH_COMPILE_DISABLE=1` if you use gradient accumulation on
    this model, because on model classes that set `accepts_loss_kwargs = False`, and Muse Glimmer is one of
    them, disabling compilation makes the loss and the gradients get divided by the accumulation count a
    second time. It is completely silent and the effective learning rate ends up several times too
    small.
    """)
    return


@app.cell
def _(BATCH_SIZE, NUM_GENERATIONS, max_completion_length_1):
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=5e-06,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.1,
        logging_steps=1,
        log_completions=False,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=NUM_GENERATIONS,
        max_completion_length=max_completion_length_1,
        mask_truncated_completions=True,
        max_steps=200,
        save_steps=100,
        report_to="none",  # Can use Weights & Biases, TrackIO
        output_dir="outputs",
        importance_sampling_level="sequence",
        loss_type="dr_grpo",
    )
    return GRPOTrainer, training_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now the trainer. The column everyone watches is `reward`, and it should climb, but on this task
    the more informative one is **`frac_reward_zero_std`**. GRPO learns from the spread of rewards inside
    a group. If all four rollouts of a prompt score the same, the advantage is exactly zero and that step
    teaches the model nothing, however long it took to generate.

    That is not hypothetical here. Over a 10 step run every single group agreed with itself:

    | step | reward | reward_std | clipped | step time |
    |---|---|---|---|---|
    | 1 | 6.50 | 0.00 | 0.00 | 40 s |
    | 2 | -1.50 | 0.00 | 1.00 | 353 s |
    | 3 | 6.50 | 0.00 | 0.00 | 184 s |
    | 4 | 6.50 | 0.00 | 0.00 | 138 s |
    | 5 | 6.50 | 0.00 | 0.00 | 65 s |
    | 6 | -1.50 | 0.00 | 1.00 | 395 s |
    | 7 | -1.50 | 0.00 | 1.00 | 396 s |
    | 8 | 6.50 | 0.00 | 0.00 | 358 s |
    | 9 | -1.50 | 0.00 | 1.00 | 389 s |
    | 10 | 6.50 | 0.00 | 0.00 | 252 s |

    Muse Glimmer is strong enough that when it can do a problem all four rollouts get it, and when the problem
    needs more reasoning than the budget allows all four run out together. Ten steps is a small sample,
    and mixed groups do occur, but if `frac_reward_zero_std` sits near 1 for a hundred steps the run is
    not learning and the fix is to raise `num_generations` to 8 or 16 so that groups have a chance to
    disagree. That costs memory and step time in direct proportion.

    `rewards/answer_is_correct/mean` is the one that actually means the model got the maths right. The
    other two only tell you it is formatting its reply properly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-GPU fix for GRPO's chunked log-softmax

    On a 2x16 GB kernel this 28B checkpoint is sharded across both cards, so the
    `lm_head` can sit on `cuda:1` while the hidden states are still on `cuda:0`.
    `chunked_hidden_states_selective_log_softmax` matmuls the two directly, and the
    step dies with `found two different devices cuda:0, cuda:1`. The cell below
    co-locates the operands on the head's device and sends the result back.

    Verified on two GPUs: bitwise identical to upstream when everything is already on
    one device, and bitwise identical again when the head is moved to a second card.
    Keeping upstream's `_maybe_compile` decorator matters; running it eager instead
    shifts the logprobs by about 1e-2 through a different fp16 matmul path.
    """)
    return


@app.cell
def _(sys, torch):
    from unsloth_zoo.rl_replacements import _maybe_compile, torch_compile_options

    @_maybe_compile(dynamic=True, fullgraph=True, options=torch_compile_options)
    def chunked_hidden_states_selective_log_softmax(
        hidden_states,
        lm_head,
        index,
        chunks=4,
        logit_scale_multiply=0.0,
        logit_scale_divide=0.0,
        logit_softcapping=0.0,
        temperature=1.0,
    ):
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_index = index.reshape(-1)
        TOKENS_PER_CHUNK = 128
        rows = flat_hidden_states.shape[0]
        chunks = max(chunks, -(-rows // TOKENS_PER_CHUNK))
        chunks = min(chunks, max(rows, 1))
        chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
        chunked_index = torch.chunk(
            flat_index, chunks=chunks, dim=0
        )  # Muse Glimmer's vocab is 202048 wide, so each chunk materialises tokens x 202048 logits
        all_per_token_logps = []  # and then a float32 copy of them, all on whichever card holds the lm_head. At the
        for chunk_hidden_states, chunk_index in zip(
            chunked_hidden_states, chunked_index
        ):  # stock chunks = 4 that is hundreds of MB landing on one GPU, which is what runs a
            chunk_hidden_states = chunk_hidden_states.to(
                device=lm_head.device, dtype=lm_head.dtype
            )  # 2x16 GB kernel out of memory while the other card still has room. Cap the chunk
            chunk_index = chunk_index.to(
                lm_head.device
            )  # at TOKENS_PER_CHUNK rows instead. Pure loop splitting: the concatenated result is
            chunk_logits = (
                chunk_hidden_states @ lm_head.t()
            )  # unchanged, only the peak is lower.
            if logit_scale_multiply != 0.0:
                chunk_logits = chunk_logits * logit_scale_multiply
            if logit_scale_divide != 0.0:
                chunk_logits = chunk_logits / logit_scale_divide
            if logit_softcapping != 0.0:
                chunk_logits = logit_softcapping * torch.tanh(
                    chunk_logits / logit_softcapping
                )
            chunk_logits = chunk_logits.to(torch.float32)
            if temperature != 1.0:
                chunk_logits = (
                    chunk_logits / temperature
                )  # The only change from upstream: co-locate with the lm_head before the
            selected_logits = torch.gather(
                chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)
            ).squeeze(
                -1
            )  # matmul, and send the result back. On one GPU every .to() is a no-op.
            logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
            all_per_token_logps.append(
                (selected_logits - logsumexp_values).to(hidden_states.device)
            )
        all_per_token_logps = torch.concat(all_per_token_logps)
        return all_per_token_logps.reshape(
            (hidden_states.shape[0], hidden_states.shape[1])
        )

    import unsloth_zoo.rl_replacements as _rl

    _rl.chunked_hidden_states_selective_log_softmax = (
        chunked_hidden_states_selective_log_softmax
    )
    _rl.RL_REPLACEMENTS["grpo_selective_log_softmax"] = (
        chunked_hidden_states_selective_log_softmax
    )
    _n = 1
    for _name, _mod in list(sys.modules.items()):
        if "UnslothGRPOTrainer" in _name and hasattr(
            _mod, "chunked_hidden_states_selective_log_softmax"
        ):
            _mod.chunked_hidden_states_selective_log_softmax = (
                chunked_hidden_states_selective_log_softmax
            )
            _n = _n + 1
    print("patched chunked log-softmax in", _n, "module(s)")
    return


@app.cell
def _(
    GRPOTrainer,
    answer_is_a_number,
    answer_is_correct,
    channel_format,
    dataset_1,
    model_1,
    tokenizer,
    training_args,
):
    trainer = GRPOTrainer(
        model=model_1,
        processing_class=tokenizer,
        reward_funcs=[channel_format, answer_is_a_number, answer_is_correct],
        args=training_args,
        train_dataset=dataset_1,
    )
    trainer.train()
    return


@app.cell
def _(torch):
    peak = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"peak reserved VRAM {peak:.2f} GiB out of {total:.2f} GiB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Muse Glimmer after training

    Same question as before. The reply channel should now be a bare number.
    """)
    return


@app.cell
def _(
    REASONING_STRENGTH,
    TextStreamer,
    dataset_1,
    max_completion_length_1,
    model_1,
    split_channels,
    tokenizer,
):
    example = dataset_1[7]
    text_1 = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength=REASONING_STRENGTH,
    )
    inputs_1 = tokenizer(text=text_1, return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )
    out = model_1.generate(
        **inputs_1,
        max_new_tokens=max_completion_length_1,
        temperature=1.0,
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
    )
    completion = tokenizer.decode(
        out[0][inputs_1["input_ids"].shape[1] :], skip_special_tokens=False
    )
    reasoning, reply = split_channels(completion)
    print("\nparsed reply:", repr(reply))
    print("gold answer :", example["answer"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving

    Save the LoRA adapters. This saves the adapters only, not the full 28B model, so it is a few hundred
    megabytes rather than tens of gigabytes.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    model_1.save_pretrained("muse_glimmer_lora")
    tokenizer.save_pretrained("muse_glimmer_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verify the LoRA actually trained rather than staying at its initialisation:
    """)
    return


@app.cell
def _():
    from safetensors import safe_open

    n_nonzero = 0
    with safe_open("muse_glimmer_lora/adapter_model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert n_zeros.item() != tensor.numel()
            if n_zeros.item() < 1.0:
                n_nonzero = n_nonzero + 1
    print(n_nonzero, "adapter tensors are non zero")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Merging to 16-bit

    If you want a standalone checkpoint rather than adapters, merge back to 16-bit. The result is about
    56 GB on disk, and loading it needs the private `transformers` build just like the base model does.

    There is no GGUF export cell in this notebook on purpose. `save_pretrained_gguf` drives upstream
    llama.cpp, which has no `muse-glimmer` architecture, so it cannot convert this model and the cell would only
    fail after a long clone and build.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "muse_glimmer_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:
        model_1.push_to_hub_merged(
            "your_name/muse_glimmer_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            private=True,
        )
    # Just LoRA adapters
    if False:
        model_1.save_pretrained("muse_glimmer_lora")
        tokenizer.save_pretrained("muse_glimmer_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Notes on memory

    Muse Glimmer in 4-bit is 20.68 GiB of weights before a single token is generated: 11.72 GB of nf4 text
    `Linear4bit` modules, 5.38 GB of 16-bit embeddings and `lm_head` at 202048 x 6656 each and untied,
    and 3.44 GB of 16-bit vision tower. `offload_embedding = True` moves 2.7 GB of that to CPU RAM.

    The KV cache is small, which is unusual for a model this size and is worth knowing before you tune
    `num_generations`. Muse Glimmer has 52 layers but only 2 key value heads at head_dim 128, so one token of
    cache costs `2 (K and V) x 2 heads x 128 x 2 bytes = 1 KiB per layer`. On top of that, 39 of the 52
    layers are sliding window at 2048 tokens and only 13 are full attention, so the sliding layers stop
    growing once the sequence passes 2048 tokens.

    | sequence | 13 full layers | 39 sliding layers | total per sequence |
    |---|---|---|---|
    | 1024 | 13 MiB | 39 MiB | 52 MiB |
    | 2048 | 26 MiB | 78 MiB | 104 MiB |
    | 8192 | 104 MiB | 78 MiB | 182 MiB |

    At the settings in this notebook, four rollouts of at most 3072 tokens each, the entire KV cache is
    about 460 MiB. It is not what limits you.

    What limits you is the vocabulary. At 202048 tokens, one 16-bit logit row is 395 KiB, and the log
    probability passes that GRPO runs over the completions materialise those rows for every position of
    every rollout. That is why `num_generations` and `max_completion_length` are the two knobs that
    matter here, and why they are set low.

    Measured on one card at the settings above, 4-bit weights with the embedding offloaded, LoRA rank 8,
    `num_generations = 4` and `max_completion_length = 2688`: **peak reserved 28.19 GiB** over a full 10
    step run. Step time ranged from 40 to 396 seconds depending on how long the rollouts ran, so budget
    roughly 4 minutes a step for a run that is generating near the cap.

    Two 16 GB T4s will not run this. The weights alone are 20.68 GiB, or about 18 GiB with the embedding
    offloaded, which does not fit on one 16 GB card, and there is nothing in this notebook that splits
    the model across two. T4s are also sm_75 and have no bf16, so you would additionally need `float16`
    throughout. Plan on a single card with at least 40 GB.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### And we're done

    Some other resources:

    1. [Unsloth Reinforcement Learning docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) for GSPO, GAPO, Dr GRPO and the rest of the options
    2. [Memory efficient RL](https://unsloth.ai/docs/basics/memory-efficient-rl)
    3. [Unsloth notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
    </div>
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
