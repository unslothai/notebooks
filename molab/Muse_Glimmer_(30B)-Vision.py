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
    To run this, press the **Run** button beside each cell on a molab **A100** or **H100** instance.
    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/blob/main/images/Discord button.png?raw=true" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + star us on <a href="https://github.com/unslothai/unsloth">Github</a>
    </div>

    To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

    You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), and how to save it.
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

    `FastModel` loads Muse Glimmer through the generic vision path. The 4bit build we point at is
    `unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`, which is a private repo, so we pass the token explicitly.

    **Memory.** Muse Glimmer is a big model for a 4bit LoRA run. The resident weights split roughly like this:

    | part | dtype | size |
    |---|---|---|
    | 416 quantised text `Linear4bit` modules | nf4 with double quant | 11.72 GiB |
    | `embed_tokens` and `lm_head` (202048 x 6656 each, untied) | 16-bit | 5.38 GiB |
    | vision tower, adapter and projection | 16-bit | 3.44 GiB |
    | total | | **20.68 GiB** |

    The embeddings are 26% of that, and they are frozen during LoRA training, so we set
    `offload_embedding = True`. That keeps `embed_tokens` in CPU RAM and moves only the looked-up
    rows to the GPU, which takes 2.5 GiB off the resident footprint. `lm_head` stays on the GPU
    because it is needed for every logit.

    Measured on one 180 GiB card at `max_seq_length = 1024`, batch size 1, LoRA r=16:
    weights 18.22 GiB after load with the offload on (20.72 GiB without it), and **22.57 GiB peak
    reserved** through training. Plan for a 24 GiB card as the realistic minimum, and note that a
    single 16 GiB card cannot hold the weights at all.
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    max_seq_length = 1024
    OFFLOAD_EMBEDDING = torch.cuda.device_count() == 1
    print(
        f"visible GPUs: {torch.cuda.device_count()}, offload_embedding: {OFFLOAD_EMBEDDING}"
    )
    model, processor = FastModel.from_pretrained(
        model_name="unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",  # The adapters you just saved
        load_in_4bit=True,  # 4bit quantisation to reduce memory
        max_seq_length=max_seq_length,
        use_gradient_checkpointing="unsloth",  # "unsloth" for long context
        offload_embedding=OFFLOAD_EMBEDDING,  # Keeps the 202048 x 6656 embedding matrix in RAM
    )
    print(
        type(model).__name__, type(processor).__name__
    )
    return FastModel, OFFLOAD_EMBEDDING, max_seq_length, model, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Muse Glimmer registers as an image-text-to-text model, so `AutoModelForCausalLM` will not load it.
    `AutoProcessor` gives you an `Muse GlimmerProcessor`, which wraps `Muse GlimmerImageProcessor` and
    `Muse GlimmerVideoProcessor`. There is no audio tower on this model.
    """)
    return


@app.cell
def _():
    from transformers import AutoProcessor

    processor_1 = AutoProcessor.from_pretrained(
        "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
    )
    # Same object FastModel already returned; shown here so you know what you are holding.
    print("image token:", processor_1.image_token, processor_1.image_token_id)
    print("video token:", processor_1.video_token, processor_1.video_token_id)
    return (processor_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only train a small fraction of the parameters. You can
    fine-tune only the vision tower, only the language model, or both, and inside each you can pick
    attention, MLP or both.
    """)
    return


@app.cell
def _(FastModel, model):
    model_1 = FastModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    model_1.print_trainable_parameters()
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We use a sampled dataset of handwritten maths formulas. The task is to turn each image into
    LaTeX so it can be rendered again.

    The dataset is [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR), and the full version is
    [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    dataset[2]["image"]
    return


@app.cell
def _(dataset):
    dataset[2]["text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can render the LaTeX directly in the browser:
    """)
    return


@app.cell
def _(dataset):
    from IPython.display import display, Math

    display(Math(dataset[3]["text"]))
    return (display,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All vision fine-tuning data should be a list of messages, where image parts sit next to text
    parts inside `content`:

    ```python
    [
        {
            "role": "user",
            "content": [
                {"type": "text",  "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["text"]},
            ],
        },
    ]
    ```
    """)
    return


@app.cell
def _():
    instruction = "Write the LaTeX representation for this image."

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        return {"messages": conversation}

    pass
    return convert_to_conversation, instruction


@app.cell
def _(convert_to_conversation, dataset):
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    converted_dataset[0]["messages"][0]["content"][1]
    return (converted_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Muse Glimmer does not use ChatML or the Llama format. It uses ATEM, where every turn is
    `<|start|>{role}<|message|>...<|eot|>` and the assistant can emit a private reasoning channel
    addressed `to=self`, closed with `<|eom|>`, before the answer it addresses `to=user`.

    The image itself is a single `<|patch|>` placeholder in the template, and `Muse GlimmerProcessor` expands
    it to one token per merged 28x28 pixel block when it sees the real image. Let us look at the
    rendered text so the markers are not a guess.
    """)
    return


@app.cell
def _(converted_dataset, processor_1):
    example_text = processor_1.apply_chat_template(
        converted_dataset[0]["messages"], tokenize=False, add_generation_prompt=False
    )
    print(example_text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference before training

    Let us see what the base model does with this task first.

    `add_generation_prompt = True` ends the prompt at `<|start|>assistant`, which leaves the recipient
    open, so Muse Glimmer usually opens its private `to=self` reasoning channel first and answers after
    `<|eom|>`. Appending ` to=user<|message|>` pins it straight to the answer channel, which is the
    channel we are about to train.
    """)
    return


@app.cell
def _(dataset, instruction, model_1, processor_1):
    from transformers import TextStreamer

    def muse_glimmer_prompt(messages, answer_directly=True):
        text = processor_1.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return text + " to=user<|message|>" if answer_directly else text

    sample = dataset[2]
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    inputs = processor_1(
        text=[muse_glimmer_prompt(messages)],
        images=[[sample["image"].convert("RGB")]],
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    text_streamer = TextStreamer(processor_1.tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
    )
    return TextStreamer, muse_glimmer_prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    We run 30 steps to keep this quick. Set `num_train_epochs = 1` and `max_steps = None` for a full
    run.

    `UnslothVisionDataCollator` builds the batch, expands the image tokens and masks everything that
    is not an assistant answer. The masking strings have to be the real ATEM markers, not the
    ChatML or Llama ones other notebooks use, so we pass them explicitly.

    `loss_type = "nll"` is deliberate. TRL's newer default, `chunked_nll`, normalises the loss by the
    accumulated token count on its own, and on model classes that declare
    `accepts_loss_kwargs = False` (Muse Glimmer is one) the trainer then divides by
    `gradient_accumulation_steps` a second time. The loss and the gradients both come out
    `1/gradient_accumulation_steps` too small and nothing warns you. Plain `nll` is invariant to
    gradient accumulation and also re-enables Unsloth's own fused cross-entropy, which uses less VRAM
    here.
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
def _(processor_1):
    reasoning_strength = (  # minimal, low, medium, high. The template default is high.
        "low"  # minimal, low, medium, high. The template default is high.
    )
    messages_1 = [
        {"role": "user", "content": "What is 2 + 2? Reply with just the number."}
    ]
    text = processor_1.tokenizer.apply_chat_template(
        messages_1,
        add_generation_prompt=True,
        tokenize=False,
        reasoning_strength=reasoning_strength,  # minimal, low, medium, high. The template default is high.
    )
    print([line for line in text.splitlines() if "Reasoning strength" in line])
    return


@app.cell
def _(converted_dataset, max_seq_length, model_1, processor_1):
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    instruction_part = "<|start|>user<|message|>"
    # ATEM markers, read straight out of the rendered sample above.
    response_part = "<|start|>assistant to=user<|message|>"
    collator = UnslothVisionDataCollator(
        model_1,
        processor_1,
        max_seq_length=max_seq_length,
        train_on_responses_only=True,
        instruction_part=instruction_part,
        response_part=response_part,
    )
    trainer = SFTTrainer(
        model=model_1,
        train_dataset=converted_dataset,
        processing_class=processor_1.tokenizer,
        data_collator=collator,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=0.0002,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # For Weights and Biases or others
            loss_type="nll",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
        ),
    )
    return collator, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us check that only the assistant answer is being trained on:
    """)
    return


@app.cell
def _(collator, converted_dataset, processor_1):
    batch = collator([converted_dataset[0], converted_dataset[1]])
    labels = batch["labels"][0]
    print("trained tokens:", int((labels != -100).sum()), "of", labels.numel())
    print(processor_1.tokenizer.decode([t for t in labels.tolist() if t != -100]))
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
    Now run the fine-tuned model on a held out image.
    """)
    return


@app.cell
def _(
    TextStreamer,
    dataset,
    instruction,
    model_1,
    muse_glimmer_prompt,
    processor_1,
):
    sample_1 = dataset[10]
    messages_2 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    inputs_1 = processor_1(
        text=[muse_glimmer_prompt(messages_2)],
        images=[[sample_1["image"].convert("RGB")]],
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    text_streamer_1 = TextStreamer(processor_1.tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_1,
        streamer=text_streamer_1,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
    )
    print("\nGround truth:", sample_1["text"])
    return (sample_1,)


@app.cell
def _(display, sample_1):
    display(sample_1["image"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the adapters, use `save_pretrained` for a local save or `push_to_hub` for an online save.

    **Note:** this saves only the LoRA adapters, not the full model. For the merged 16-bit model,
    scroll down.
    """)
    return


@app.cell
def _(model_1, processor_1):
    model_1.save_pretrained("muse_glimmer_lora")
    processor_1.save_pretrained("muse_glimmer_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To load the adapters back for inference, change `False` to `True`:
    """)
    return


@app.cell
def _(
    OFFLOAD_EMBEDDING,
    TextStreamer,
    dataset,
    instruction,
    model_1,
    muse_glimmer_prompt,
    processor_1,
):
    if False:
        from unsloth import FastModel as _FastModel

        _model, _processor = _FastModel.from_pretrained(
            model_name="muse_glimmer_lora",  # The adapters you just saved
            load_in_4bit=True,  # 4bit quantisation to reduce memory
            max_seq_length=1024,
            offload_embedding=OFFLOAD_EMBEDDING,  # Keeps the 202048 x 6656 embedding matrix in RAM
        )
    sample_2 = dataset[1]
    messages_3 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    inputs_2 = processor_1(
        text=[muse_glimmer_prompt(messages_3)],
        images=[[sample_2["image"].convert("RGB")]],
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    text_streamer_2 = TextStreamer(processor_1.tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_2,
        streamer=text_streamer_2,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to 16-bit

    `save_pretrained_merged` folds the adapters back into the base weights and writes a normal
    16-bit checkpoint you can serve.

    Two things to know before you run it. The merge reads the **16-bit** base, not the 4bit one we
    trained on, and Muse Glimmer has no entry in Unsloth's 4bit-to-16bit name map yet, so we point
    `_name_or_path` at `unsloth/Muse-Glimmer-30B` by hand. And the output is about 56 GB, so you need that much
    free disk plus room to stage the download.

    GGUF export is not available for Muse Glimmer yet, since the architecture is not in upstream
    `llama.cpp`. Use the merged 16-bit save below instead.
    """)
    return


@app.cell
def _(model_1, processor_1):
    # Select ONLY 1 to save! (Both not needed!)
    model_1.config._name_or_path = "unsloth/Muse-Glimmer-30B"
    # The merge needs the 16bit base weights; Muse Glimmer is not in the 4bit-to-16bit name map yet.
    if False:
        model_1.save_pretrained_merged("muse_glimmer_finetune", processor_1)
    # Save locally to 16bit
    if False:
        # Push to your own private Hugging Face repo
        model_1.push_to_hub_merged(
            "your_name/muse_glimmer_finetune", processor_1, private=True
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we are done. If you have any questions on Unsloth, we have a
    [Discord](https://discord.gg/unsloth) channel.

    Some other resources:
    1. Train your own reasoning model - Llama GRPO notebook [Open in molab](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    2. Saving finetunes to Ollama. [Free notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    3. Llama 3.2 Vision finetuning - Radiography use case. [Open in molab](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
    4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

      Join Discord if you need help + star us on <a href="https://github.com/unslothai/unsloth">Github</a>
    </div>

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
    """)
    return


if __name__ == "__main__":
    app.run()
