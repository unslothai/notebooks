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
#     "torchao>=0.16.0",
#     "torchcodec",
#     "transformers>=4.56.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth.git",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
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

    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        # Pretrained models
        "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
        # Other Gemma 3 quants
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3n-E4B-it",  # YOUR MODEL YOU USED FOR TRAINING
        dtype=None,  # None for auto detection
        max_seq_length=1024,  # Choose any for long context!
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastModel, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gemma 3N can process Text, Vision and Audio!

    Let's first experience how Gemma 3N can handle multimodal inputs. We use Gemma 3N's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`
    """)
    return


@app.cell
def _(model, tokenizer):
    from transformers import TextStreamer

    # Helper function for inference
    def do_gemma_3n_inference(messages, max_new_tokens=128):
        _ = model.generate(
            **tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Must add for generation
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda"),
            max_new_tokens=max_new_tokens,  # Increase for longer outputs!
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )

    return TextStreamer, do_gemma_3n_inference


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gemma 3N can see images!

    <img src="https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg" alt="Alt text" height="256">
    """)
    return


@app.cell
def _(do_gemma_3n_inference):
    sloth_link = "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sloth_link},
                {"type": "text", "text": "Which films does this animal feature in?"},
            ],
        }
    ]
    # You might have to wait 1 minute for Unsloth's auto compiler
    do_gemma_3n_inference(messages, max_new_tokens=256)
    return (sloth_link,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's make a poem about sloths!
    """)
    return


@app.cell
def _(do_gemma_3n_inference):
    messages_1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem about sloths."}],
        }
    ]
    do_gemma_3n_inference(messages_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gemma 3N can also hear!
    """)
    return


@app.cell
def _():
    from IPython.display import Audio, display

    Audio(
        "https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3"
    )
    return


@app.cell
def _():
    import subprocess

    #! wget -qqq https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3 -O audio.mp3
    subprocess.call(
        [
            "wget",
            "-qqq",
            "https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3",
            "-O",
            "audio.mp3",
        ]
    )
    return


@app.cell
def _(do_gemma_3n_inference):
    audio_file = "audio.mp3"
    messages_2 = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_file},
                {"type": "text", "text": "What is this audio about?"},
            ],
        }
    ]
    do_gemma_3n_inference(messages_2, max_new_tokens=256)
    return (audio_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Let's combine all 3 modalities together!
    """)
    return


@app.cell
def _(audio_file, do_gemma_3n_inference, sloth_link):
    messages_3 = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_file},
                {"type": "image", "image": sloth_link},
                {
                    "type": "text",
                    "text": "What is this audio and image about? How are they related?",
                },
            ],
        }
    ]
    do_gemma_3n_inference(messages_3, max_new_tokens=256)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Let's finetune Gemma 3N!

    You can finetune the vision and text parts for now through selection - the audio part can also be finetuned - we're working to make it selectable as well!
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
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # Should leave on always!
        r=8,  # Larger = higher accuracy, but might overfit
        lora_alpha=8,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:

    ```
    <bos><start_of_turn>user
    Hello!<end_of_turn>
    <start_of_turn>model
    Hey there!<end_of_turn>
    ```

    We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.
    """)
    return


@app.cell
def _(tokenizer):
    from unsloth.chat_templates import get_chat_template

    tokenizer_1 = get_chat_template(tokenizer, chat_template="gemma-3")
    return get_chat_template, tokenizer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We get the first 3000 rows of the dataset
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
    We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!
    """)
    return


@app.cell
def _(dataset):
    from unsloth.chat_templates import standardize_data_formats

    dataset_1 = standardize_data_formats(dataset)
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how row 100 looks like!
    """)
    return


@app.cell
def _(dataset_1):
    dataset_1[100]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one.
    """)
    return


@app.cell
def _(dataset_1, tokenizer_1):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer_1.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset_2 = dataset_1.map(formatting_prompts_func, batched=True)
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one.
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
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
    """)
    return


@app.cell
def _(dataset_2, model_1, tokenizer_1):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer_1,
        train_dataset=dataset_2,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
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
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!
    """)
    return


@app.cell
def _(trainer):
    from unsloth.chat_templates import train_on_responses_only

    trainer_1 = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    return (trainer_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's verify masking the instruction part is done! Let's print the 100th row again.  Notice how the sample only has a single `<bos>` as expected!
    """)
    return


@app.cell
def _(tokenizer_1, trainer_1):
    tokenizer_1.decode(trainer_1.train_dataset[100]["input_ids"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's print the masked out example - you should see only the answer is present:
    """)
    return


@app.cell
def _(tokenizer_1, trainer_1):
    tokenizer_1.decode(
        [
            tokenizer_1.pad_token_id if x == -100 else x
            for x in trainer_1.train_dataset[100]["labels"]
        ]
    ).replace(tokenizer_1.pad_token, " ")
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
    <a name="Inference"></a>
    ### Inference
    Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`
    """)
    return


@app.cell
def _(get_chat_template, model_1, tokenizer_1):
    tokenizer_2 = get_chat_template(tokenizer_1, chat_template="gemma-3")
    messages_4 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Continue the sequence: 1, 1, 2, 3, 5, 8,"}
            ],
        }
    ]
    inputs = tokenizer_2.apply_chat_template(
        messages_4,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    outputs = model_1.generate(
        **inputs, max_new_tokens=64, temperature=1.0, top_p=0.95, top_k=64
    )
    tokenizer_2.batch_decode(
        outputs
    )
    return (tokenizer_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer_2):
    messages_5 = [
        {"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]}
    ]
    inputs_1 = tokenizer_2.apply_chat_template(
        messages_5,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model_1.generate(
        **inputs_1,
        max_new_tokens=64,  # Increase for longer outputs!
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer_2, skip_prompt=True),
    )
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
def _(model_1, tokenizer_2):
    model_1.save_pretrained("gemma_3n_lora")
    tokenizer_2.save_pretrained("gemma_3n_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer_2):
    if False:
        from unsloth import FastModel as _FastModel

        _model, _tokenizer = _FastModel.from_pretrained(
            model_name="gemma_3n_lora", max_seq_length=2048, load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
    messages_6 = [
        {"role": "user", "content": [{"type": "text", "text": "What is Gemma-3N?"}]}
    ]
    inputs_2 = tokenizer_2.apply_chat_template(
        messages_6,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model_1.generate(
        **inputs_2,
        max_new_tokens=128,  # Increase for longer outputs!
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer_2, skip_prompt=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3N-finetune`. Set `if False` to `if True` to let it run!
    """)
    return


@app.cell
def _(model_1, tokenizer_2):
    if False:  # Change to True to save finetune!
        model_1.save_pretrained_merged("gemma-3N-finetune", tokenizer_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, tokenizer_2):
    if False:  # Change to True to upload finetune
        model_1.push_to_hub_merged(
            "HF_ACCOUNT/gemma-3N-finetune", tokenizer_2, token="YOUR_HF_TOKEN"
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
def _(model_1, tokenizer_2):
    if False:  # Change to True to save to GGUF
        model_1.save_pretrained_gguf(
            "gemma_3n_finetune", tokenizer_2, quantization_method="Q8_0"
        )  # For now only Q8_0, BF16, F16 supported
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, tokenizer_2):
    if False:  # Change to True to upload GGUF
        model_1.push_to_hub_gguf(
            "HF_ACCOUNT/gemma_3n_finetune",
            tokenizer_2,
            quantization_method="Q8_0",  # For now only Q8_0, BF16, F16 supported
            token="YOUR_HF_TOKEN",
        )  # Only Q8_0, BF16, F16 supported
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `gemma-3N-finetune.gguf` file or `gemma-3N-finetune-Q4_K_M.gguf` file in llama.cpp.

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
