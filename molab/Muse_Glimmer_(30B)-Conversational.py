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
#     "transformers==5.15.0",
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
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i>
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
    Muse Glimmer is a brand new architecture, so it needs a `transformers` release that knows about it. Support shipped in [transformers 5.15.0](https://github.com/huggingface/transformers/releases/tag/v5.15.0), so we pin that exact version. Everything used below is public - no Hugging Face token is required.
    """)
    return


@app.cell
def _():
    # Muse Glimmer support ships in transformers 5.15.0, so install the release.
    # Pinned to that exact version so the notebook does not change underneath you.
    # Run this BEFORE anything imports transformers.

    import transformers

    print("transformers:", transformers.__version__)
    assert transformers.__version__ == "5.15.0", transformers.__version__
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth

    `FastModel` supports loading nearly any model now! This includes Vision and Text models!

    Muse Glimmer is a 28B dense text + vision model with 52 layers, a 202048 token vocabulary and untied
    embeddings. It is registered as an image-text-to-text model, so load it with `FastModel`.

    Two things matter a lot for memory here:

    * The embedding offload keeps `embed_tokens` in CPU RAM and only moves the looked-up rows to
      the GPU. Muse Glimmer's input embedding is 202048 x 6656 in 16-bit, so this gives back 2.5 GB at load
      time. There is a callback further down that keeps it that way once training starts.
    * The checkpoint is already 4-bit, and its own `quantization_config` is what transformers uses.
      That config leaves the embeddings, the `lm_head` and the whole vision tower unquantized, so there
      is nothing to restate here. Tesla T4s are `sm_75` and have no `bfloat16`; the loader falls back to
      `float16` on its own when the card cannot do it.
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    max_seq_length = (  # Muse Glimmer supports long context, but 1024 is what fits a small card
        1024  # Muse Glimmer supports long context, but 1024 is what fits a small card
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,  # Muse Glimmer supports long context, but 1024 is what fits a small card
    )
    return FastModel, max_seq_length, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update a small amount of parameters!

    Muse Glimmer has a vision tower, but this notebook is a text conversational finetune, so we leave the
    vision layers frozen.
    """)
    return


@app.cell
def _(FastModel, model):
    model_1 = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # Should leave on always!
        r=8,  # Larger = higher accuracy, but might overfit
        lora_alpha=8,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM, fits 2x larger batch sizes
        random_state=3407,
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep

    Muse Glimmer does not use ChatML and it does not use the Llama format. It uses **ATEM**, which routes every
    assistant turn to a named recipient. A two turn conversation renders like this:

    ```
    <|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.
    Knowledge cutoff: 2026-01-04.

    Reasoning strength: high.

    # Valid recipients: "self", "user".<|eot|><|start|>user<|message|>What is 17 times 23?<|eot|><|start|>assistant to=self<|message|>17*23 = 391.<|eom|><|start|>assistant to=user<|message|>391<|eot|>
    ```

    The assistant thinks out loud on the `to=self` channel, closes it with `<|eom|>`, then answers the
    user on the `to=user` channel and closes the turn with `<|eot|>`. `<|eot|>` ends generation,
    `<|eom|>` deliberately does not, which is how the model flows from thinking straight into
    answering.

    We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)
    dataset in ShareGPT style, the same multi turn conversational dataset the Llama conversational
    notebook uses.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now use `standardize_sharegpt` to convert ShareGPT style datasets into HuggingFace's generic
    format. This changes the dataset from looking like:
    ```
    {"from": "system", "value": "You are an assistant"}
    {"from": "human", "value": "What is 2+2?"}
    {"from": "gpt", "value": "It's 4."}
    ```
    to
    ```
    {"role": "system", "content": "You are an assistant"}
    {"role": "user", "content": "What is 2+2?"}
    {"role": "assistant", "content": "It's 4."}
    ```
    """)
    return


@app.cell
def _(dataset):
    from unsloth.chat_templates import standardize_sharegpt

    dataset_1 = standardize_sharegpt(dataset)
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how row 100 looks like!
    """)
    return


@app.cell
def _(dataset_1):
    dataset_1[100]["conversations"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Muse Glimmer ships its own ATEM chat template, so we apply the tokenizer's template directly instead of
    calling `get_chat_template`. We strip the leading `<|begin_of_text|>` with `removeprefix`, because
    the tokenizer adds one itself at tokenization time and the model expects exactly one.
    """)
    return


@app.cell
def _(dataset_1, tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix(tokenizer.bos_token)
            for convo in convos
        ]
        return {"text": texts}

    dataset_2 = dataset_1.map(formatting_prompts_func, batched=True)
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we see how the chat template transformed these conversations. Notice there is no
    `<|begin_of_text|>` token, and every answer sits on the `to=user` channel.
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
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1`
    for a full run, and turn off `max_steps=None`.

    `loss_type = "nll"` is deliberate. TRL's newer default chunked loss divides by the token count
    itself, and Muse Glimmer's model class also tells the trainer that it does not consume a token count, so
    with gradient accumulation switched on the loss and the gradients both end up divided by
    `gradient_accumulation_steps` a second time. Nothing errors, the effective learning rate is just
    silently too small. Plain `"nll"` is invariant to gradient accumulation, and it also re-enables
    Unsloth's own fused cross entropy, which measured 1.7 to 3.7 GiB cheaper than the chunked version
    at every sequence length we tried.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Controlling how much the model thinks

    This model reasons in a private `to=self` channel before it answers. The chat template
    renders `Reasoning strength: <level>.` into the system block, and it defaults to `high`,
    so if you never set it you are paying for the most expensive setting on every prompt.
    Pass `reasoning_strength` to `apply_chat_template` to change it.

    Measured on the 4-bit checkpoint, one multi-step word problem, greedy decoding:

    | reasoning_strength | private reasoning tokens | total completion tokens | answer |
    |---|---|---|---|
    | minimal | 248 | 255 | correct |
    | low | 247 | 254 | correct |
    | medium | 401 | 408 | correct |
    | high (default) | 502 | 509 | correct |

    All four reached the same right answer, and `high` spent roughly twice the tokens of
    `minimal` to get there. Reasoning tokens count against `max_new_tokens`, so a run that
    hits the ceiling mid-thought returns an empty answer, which looks exactly like a wrong
    one. Raise `max_new_tokens` before you raise the effort. One sample per setting is a
    smoke test rather than a measurement, so sweep your own task before locking a value in.
    """)
    return


@app.cell
def _(tokenizer):
    reasoning_strength = (  # minimal, low, medium, high. The template default is high.
        "low"  # minimal, low, medium, high. The template default is high.
    )

    messages = [
        {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
        tokenize=False,
        reasoning_strength=reasoning_strength,  # minimal, low, medium, high. The template default is high.
    )
    print([line for line in text.splitlines() if "Reasoning strength" in line])
    return


@app.cell
def _(dataset_2, max_seq_length, model_1, tokenizer):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset_2,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            loss_type="nll",  # Gradient accumulation invariant, see the note above
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=5,
            max_steps=60,
            learning_rate=0.0002,  # Reduce to 2e-5 for long training runs
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and
    ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

    The auto-detection built into `train_on_responses_only` is written for ChatML and Llama style
    templates, so we pass the ATEM markers explicitly. They are not guesses - they are the exact
    strings the template above emits, and both tokenize cleanly:

    * `<|start|>user<|message|>` is `['<|start|>', 'user', '<|message|>']`
    * `<|start|>assistant to=user<|message|>` is `['<|start|>', 'assistant', ' to', '=user', '<|message|>']`

    Pointing the response marker at `to=user` rather than at bare `<|start|>assistant` is on purpose.
    FineTome has no reasoning traces, so if we trained on everything after `<|start|>assistant` we
    would be teaching Muse Glimmer that the correct continuation of an assistant turn is to skip thinking and
    go straight to the answer, which erodes the `to=self` channel. Masking through the channel marker
    means the finetune only changes the wording of the answer and leaves the reasoning behaviour
    alone. If your dataset does carry reasoning (put it in `reasoning_content` on the assistant
    message, which the template renders as a `to=self` segment), use `<|start|>assistant` instead so
    the reasoning is trained too.
    """)
    return


@app.cell
def _(trainer):
    from unsloth.chat_templates import train_on_responses_only

    muse_glimmer_atem_kwargs = dict(
        instruction_part="<|start|>user<|message|>",
        response_part="<|start|>assistant to=user<|message|>",
    )
    trainer_1 = train_on_responses_only(trainer, **muse_glimmer_atem_kwargs)
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
    Now let's print the masked out example - you should see only the answer is present.

    Muse Glimmer is a vision capable model, so `FastModel` hands back a `MuseGlimmerProcessor` rather than a bare
    tokenizer. Calling it with a single positional argument would be read as an image, so reach for the
    inner text tokenizer when you want to tokenize a plain string.
    """)
    return


@app.cell
def _(tokenizer, trainer_1):
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    space = text_tokenizer(" ", add_special_tokens=False).input_ids[0]
    tokenizer.decode(
        [space if x == -100 else x for x in trainer_1.train_dataset[100]["labels"]]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One more memory step. The embedding offload puts `embed_tokens` in CPU RAM at load time, but
    the trainer moves the whole model onto the accelerator when `train()` starts, which quietly drags
    the 2.5 GB embedding back onto the GPU. The lookup hooks Unsloth installs read the weight's device
    live, so we can simply push it back to CPU once training has begun and it stays there.

    Measured on this notebook at `max_seq_length = 1024`, batch size 1, over 8 steps: peak reserved
    drops from 24.14 GB to 21.25 GB, and the per step losses match to three decimal places.
    """)
    return


@app.cell
def _(torch, trainer_1):
    from transformers import TrainerCallback

    class KeepEmbeddingOffloaded(TrainerCallback):
        # The trainer places the model on the accelerator at train() time, which undoes
        def on_train_begin(
            self, args, state, control, **kwargs
        ):  # offload_embedding. Put the input embedding back on the CPU once, after that has happened.
            embed_tokens = kwargs["model"].get_input_embeddings()
            hook = getattr(embed_tokens, "_hf_hook", None)
            if (
                getattr(hook, "execution_device", None) is not None
            ):  # Skip when accelerate owns placement: the embedding then carries a dispatch
                return control  # hook, which is why the loader declined the offload. hf_device_map cannot be
            if (
                embed_tokens.weight.device.type != "cpu"
            ):  # the test, since a single-GPU load is device_map = "sequential" -> {"": 0}.
                embed_tokens.to("cpu")
                torch.cuda.empty_cache()
            return control

    trainer_1.add_callback(KeepEmbeddingOffloaded())
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
    # Let's train the model!

    To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
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
    ### What this actually costs

    All of the numbers below are measured, on one card, with exactly the settings in this notebook -
    4-bit Muse Glimmer, LoRA r=8, `use_gradient_checkpointing = "unsloth"`, `max_seq_length = 1024`, batch
    size 1, gradient accumulation 4.

    | configuration | resident after load, GB | peak reserved while training, GB |
    |---|---|---|
    | no embedding offload | 20.72 | 24.14 |
    | embedding offload only | 18.22 | 24.14 |
    | embedding offload plus `KeepEmbeddingOffloaded` | 18.22 | **21.25** |

    The 20.72 GB of resident weights break down as the 416 4-bit text linears (11.72 GB), the
    untied `embed_tokens` and `lm_head` at 2.5 GB each in 16-bit, and the vision tower with its
    adapter and projection (3.44 GB). Offloading `embed_tokens` removes one of those 2.5 GB blocks.

    Note the middle row. The offload on its own buys nothing at the peak, because the trainer
    pulls the embedding back onto the GPU when training starts. Only the callback makes the saving
    survive into the training loop.

    None of these fit a single 16 GB T4. Two T4s give 32 GB in total and `device_map` sharding will
    spread the layers across both, but this notebook has not been run that way, so treat 2x T4 as
    untested. A single 24 GB card (RTX 3090, RTX 4090, L4, A10G) has the headroom. If you are close
    to the limit, the levers that move the number most are lowering `max_seq_length` and turning off
    `finetune_mlp_modules`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Let's run the model via Unsloth native inference!

    The generation prompt ends at `<|start|>assistant` and nothing more, so the model itself picks the
    channel. Expect it to open with ` to=self<|message|>`, think, close with `<|eom|>`, then start a
    fresh `<|start|>assistant to=user<|message|>` segment with the actual answer and stop at `<|eot|>`.
    We pass `skip_special_tokens = False` so you can watch both channels go past.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    messages_1 = [
        {
            "role": "user",
            "content": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,",
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages_1,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    outputs = model_1.generate(
        **inputs, max_new_tokens=512, temperature=1.0, top_p=0.95, top_k=64
    )
    print(
        tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use a `TextStreamer` for continuous inference - so you can see the generation token by
    token, instead of waiting the whole time!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    messages_2 = [{"role": "user", "content": "Why is the sky blue?"}]
    inputs_1 = tokenizer.apply_chat_template(
        messages_2,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    from transformers import TextStreamer

    _ = model_1.generate(
        **inputs_1,
        max_new_tokens=512,  # Muse Glimmer thinks at length, so give it room to reach the answer
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online
    save or `save_pretrained` for a local save.

    **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit, scroll
    down!
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
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer):
    if False:
        from unsloth import FastModel as _FastModel

        _model, _tokenizer = _FastModel.from_pretrained(
            model_name="muse_glimmer_lora", max_seq_length=1024  # YOUR MODEL YOU USED FOR TRAINING
        )
    messages_3 = [
        {"role": "user", "content": "Describe a tall tower in the capital of France."}
    ]
    inputs_2 = tokenizer.apply_chat_template(
        messages_3,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model_1.generate(
        **inputs_2,
        max_new_tokens=512,  # Muse Glimmer thinks at length, so give it room to reach the answer
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16

    We also support saving to 16-bit directly for deployment! We save it in the folder
    `muse-glimmer-finetune`. Set `if False` to `if True` to let it run!

    A merged Muse Glimmer is around 56 GB on disk, so make sure you have the room. The merge itself needs
    enough CPU RAM to hold the dequantized model, so prefer a machine with 64 GB or more, and expect
    it to take a while.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    if False:  # Change to True to save finetune!
        model_1.save_pretrained_merged("muse-glimmer-finetune", tokenizer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your
    upload location. Keep the repository private if the weights are not yours to share.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    if False:  # Change to True to upload finetune
        model_1.push_to_hub_merged(
            "HF_ACCOUNT/muse-glimmer-finetune", tokenizer, private=True
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A note on GGUF

    There is no GGUF export cell in this notebook on purpose. `save_pretrained_gguf` drives upstream
    `llama.cpp`, which has no `muse-glimmer` architecture yet, so the conversion fails partway through and
    leaves you with a broken file. Use the merged 16-bit save above for now.

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

      Join Discord if you need help + <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i>
    </div>

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
    """)
    return


if __name__ == "__main__":
    app.run()
