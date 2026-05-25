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
#     "timm",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torchao>=0.16.0",
#     "torchcodec",
#     "transformers==5.5.0",
#     "triton>=3.2.0",
#     "trl",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
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
    To run this, press the **Run** button beside each cell!
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


@app.cell
def _():
    # For Gemma 4 vision/audio
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth

    `FastModel` supports loading nearly any model now! This includes Vision and Text models!
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch
    from huggingface_hub import snapshot_download

    fourbit_models = [
        # Gemma 4 models
        "unsloth/gemma-4-E2B-it",
        "unsloth/gemma-4-E2B",
        "unsloth/gemma-4-E2B-it",
        "unsloth/gemma-4-E4B",
        "unsloth/gemma-4-31B-it",
        "unsloth/gemma-4-31B",
        "unsloth/gemma-4-26B-A4B-it",
        "unsloth/gemma-4-26B-A4B",
    ]  # More models at https://huggingface.co/unsloth

    model, processor = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-E2B-it",  # YOUR MODEL YOU USED FOR TRAINING
        dtype=None,  # None for auto detection
        max_seq_length=8192,  # Choose any for long context!
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastModel, model, processor, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gemma 4 can process Text, Vision and Audio!

    Let's first experience how Gemma 4 can handle multimodal inputs. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64` but for this example we use `do_sample=False` for ASR.
    """)
    return


@app.cell
def _(model, processor):
    from transformers import TextStreamer

    # Helper function for inference
    def do_gemma_4_inference(messages, max_new_tokens=128):
        _ = model.generate(
            **processor.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Must add for generation
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda"),
            max_new_tokens=max_new_tokens,  # Increase for longer outputs!
            do_sample=False,
            streamer=TextStreamer(processor, skip_prompt=True),
        )

    return TextStreamer, do_gemma_4_inference


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h3>Let's Evaluate Gemma 4 Baseline Performance on German Transcription</h2>
    """)
    return


@app.cell
def _():
    from datasets import load_dataset, Audio, concatenate_datasets

    dataset = load_dataset("kadirnar/Emilia-DE-B000000", split="train")
    test_audio = dataset[7546]

    dataset = dataset.select(range(3000))

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return Audio, dataset, test_audio


@app.cell
def _(Audio, test_audio):
    from IPython.display import display

    print(test_audio["text"])
    Audio(test_audio["audio"]["array"], rate=test_audio["audio"]["sampling_rate"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And the translation of the audio from German to English is:

    > I—I hold myself directly accountable. That much is, of course, clear: namely, that there are political interests involved in trade—in the exchange of goods—and that political influences are at play. The question is: that should not be the alternative.
    """)
    return


@app.cell
def _(do_gemma_4_inference, test_audio):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes speech accurately.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": test_audio["audio"]["array"]},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        },
    ]

    do_gemma_4_inference(messages, max_new_tokens=256)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h3>Baseline Model Performance: 32.43% Word Error Rate (WER) for this sample !</h3>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Let's finetune Gemma 4!

    You can finetune the vision and text and audio parts
    """)
    return


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
        finetune_vision_layers=False,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=8,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "post",
            "linear_start",
            "linear_end",
            "embedding_projection",
            "ffw_layer_1",
            "ffw_layer_2",
            "output_proj",
        ],
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We adapt the `kadirnar/Emilia-DE-B000000` dataset for our German ASR task using Gemma 4 multi-modal chat format. Each audio-text pair is structured into a conversation with `system`, `user`, and `assistant` roles. The processor then converts this into the final training format:

    ```
    <bos><|turn>system
    You are an assistant that transcribes speech accurately.<turn|>
    <|turn>user
    <|audio|>Please transcribe this audio.<turn|>
    <|turn>model
    Ich, ich rechne direkt mich an.<turn|>
    """)
    return


@app.function
def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"]
        label = str(samples["text"][idx])

        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples


@app.cell
def _(dataset):
    dataset_1 = dataset.map(
        format_intersection_data, batched=True, batch_size=4, num_proc=4
    )
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
    """)
    return


@app.cell
def _(dataset_1, model_1, processor):
    # Use UnslothVisionDataCollator which handles audio token alignment correctly
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model_1,
        train_dataset=dataset_1,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model_1, processor),
        args=SFTConfig(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            warmup_ratio=0.03,
            max_steps=60,
            learning_rate=5e-05,
            logging_steps=1,
            save_strategy="steps",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=8192,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Let's train the model!

    To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
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
    Let's run the model via Unsloth native inference! According to the `Gemma-4` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64` but for this example we use `do_sample=False` for ASR.
    """)
    return


@app.cell
def _(do_gemma_4_inference, test_audio):
    messages_1 = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes speech accurately.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": test_audio["audio"]["array"]},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        },
    ]
    do_gemma_4_inference(messages_1, max_new_tokens=256)
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
def _(model_1, processor):
    model_1.save_pretrained("gemma_4_lora")
    processor.save_pretrained("gemma_4_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, model_1, processor):
    if False:
        from unsloth import FastModel as _FastModel

        _model, _processor = _FastModel.from_pretrained(
            model_name="gemma_4_lora", max_seq_length=2048, load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
    messages_2 = [
        {"role": "user", "content": [{"type": "text", "text": "What is Gemma-4?"}]}
    ]
    inputs = processor.apply_chat_template(
        messages_2,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model_1.generate(
        **inputs,
        max_new_tokens=128,  # Increase for longer outputs!
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(processor, skip_prompt=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly for deployment! We save it in the folder `gemma-4-finetune`. Set `if False` to `if True` to let it run!
    """)
    return


@app.cell
def _(model_1, processor):
    if False:  # Change to True to save finetune!
        model_1.save_pretrained_merged("gemma-4", processor)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, processor):
    if False:  # Change to True to upload finetune
        model_1.push_to_hub_merged(
            "HF_ACCOUNT/gemma-4-finetune", processor, token="YOUR_HF_TOKEN"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!
    """)
    return


@app.cell
def _(model_1, processor):
    if False:  # Change to True to save to GGUF
        model_1.save_pretrained_gguf(
            "gemma_4_finetune", processor, quantization_method="Q8_0"
        )  # For now only Q8_0, BF16, F16 supported
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, processor):
    if False:  # Change to True to upload GGUF
        model_1.push_to_hub_gguf(
            "HF_ACCOUNT/gemma_4_finetune",
            processor,
            quantization_method="Q8_0",  # For now only Q8_0, BF16, F16 supported
            token="YOUR_HF_TOKEN",
        )  # Only Q8_0, BF16, F16 supported
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `gemma-4-finetune.gguf` file or `gemma-4-finetune-Q4_K_M.gguf` file in llama.cpp.

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

      Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
    </div>

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
    """)
    return


if __name__ == "__main__":
    app.run()
