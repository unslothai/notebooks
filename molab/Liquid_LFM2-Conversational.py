# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "causal-conv1d==1.5.0.post8",
#     "cut_cross_entropy",
#     "datasets>=3.4.1,<4.0.0",
#     "hf_transfer",
#     "huggingface_hub>=0.34.0",
#     "marimo",
#     "peft",
#     "protobuf",
#     "sentencepiece",
#     "torchao>=0.16.0",
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "triton>=3.2.0",
#     "trl",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "xformers>=0.0.33",
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

    To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://unsloth.ai/docs/get-started/install).

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
    # Need main branch for Liquid LFM2 models
    # Install Mamba kernels
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/LFM2-1.2B-unsloth-bnb-4bit",
        "unsloth/LFM2-700M-unsloth-bnb-4bit",
        "unsloth/LFM2-350M-unsloth-bnb-4bit",
        # Full 16bit unquantized models
        "unsloth/LFM2-1.2B",
        "unsloth/LFM2-700M",
        "unsloth/LFM2-350M",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/LFM2-1.2B",  # YOUR MODEL YOU USED FOR TRAINING
        dtype=None,  # None for auto detection
        max_seq_length=2048,  # Choose any for long context!
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "hf_...", # HF Token for gated models
    )
    return FastModel, model, tokenizer, torch


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
        finetune_vision_layers=False,  # LFM for now is just text only
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # Should leave on always!
        r=16,  # Larger = higher accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
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
    We now use the `LFM` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. LFM renders multi turn conversations like below:

    ```
    <|startoftext|><|im_start|>user
    Hello!<|im_end|>
    <|im_start|>assistant
    Hey there!<|im_end|>
    ```
    """)
    return


@app.cell
def _(tokenizer):
    tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hey there!"},
        ],
        tokenize=False,
    )
    return


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
    We now have to apply the chat template for `LFM` onto the conversations, and save it to `text`. We also remove the BOS token otherwise we'll get double BOS tokens!
    """)
    return


@app.cell
def _(dataset_1, tokenizer):
    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["conversations"], tokenize=False, add_generation_prompt=False
        )
        return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

    dataset_2 = dataset_1.map(formatting_prompts_func, batched=True)
    return (dataset_2,)


@app.cell
def _(dataset_2):
    dataset_2[0]["text"]
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
def _(dataset_2, model_1, tokenizer):
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset_2,
        eval_dataset=None,  # Can set up evaluation!
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
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
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
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
    Now let's print the masked out example - you should see only the answer is present:
    """)
    return


@app.cell
def _(tokenizer, trainer_1):
    tokenizer.decode(
        [
            tokenizer.pad_token_id if x == -100 else x
            for x in trainer_1.train_dataset[100]["labels"]
        ]
    ).replace(tokenizer.pad_token, " ")
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
    Let's run the model! You can change the instruction and input - leave the output blank!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    messages = [{"role": "user", "content": "Why is the sky blue?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    from transformers import TextStreamer

    _ = model_1.generate(
        **inputs,
        max_new_tokens=128,  # Increase for longer outputs!
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    return (TextStreamer,)


@app.cell
def _(TextStreamer, model_1, tokenizer):
    messages_1 = [{"role": "user", "content": "Write a poem about a sloth."}]
    inputs_1 = tokenizer.apply_chat_template(
        messages_1,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model_1.generate(
        **inputs_1,
        max_new_tokens=128,  # Increase for longer outputs!
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
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
def _(model_1, tokenizer):
    model_1.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, dtype, load_in_4bit, max_seq_length, model_1, tokenizer):
    if False:
        from unsloth import FastModel as _FastModel
        from transformers import Lfm2ForCausalLM as _Lfm2ForCausalLM

        _model, _tokenizer = _FastModel.from_pretrained(
            model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=max_seq_length,  # Choose any for long context!
            dtype=dtype,  # None for auto detection
            auto_model=_Lfm2ForCausalLM,
            load_in_4bit=load_in_4bit,  # 4 bit quantization to reduce memory
        )
        _FastModel.for_inference(_model)
    messages_2 = [{"role": "user", "content": "How do I code up a transformer?"}]
    inputs_2 = tokenizer.apply_chat_template(
        messages_2,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model_1.generate(
        **inputs_2,
        max_new_tokens=128,  # Increase for longer outputs!
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use Hugging Face's `AutoPeftModelForCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.
    """)
    return


@app.cell
def _(load_in_4bit):
    if False:
        from peft import AutoPeftModelForCausalLM as _AutoPeftModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        _model = _AutoPeftModelForCausalLM.from_pretrained(
            "lora_model", load_in_4bit=load_in_4bit
        )
        _tokenizer = _AutoTokenizer.from_pretrained("lora_model")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_16bit", token=""
        )
    if False:
        model_1.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_4bit", token=""
        )
    if False:
        model_1.save_pretrained("model")
        tokenizer.save_pretrained("model")
    if False:
        model_1.push_to_hub("hf/model", token="")
        tokenizer.push_to_hub("hf/model", token="")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

    Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
    * `q8_0` - Fast conversion. High resource use, but generally acceptable.
    * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
    * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

    [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Save to 8bit Q8_0
    if False:
        # Remember to go to https://huggingface.co/settings/tokens for a token!
        # And change hf to your username!
        model_1.save_pretrained_gguf("model", tokenizer)
    if False:
        # Save to 16bit GGUF
        model_1.push_to_hub_gguf("hf/model", tokenizer, token="")
    if False:
        model_1.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="f16", token=""
        )
    if False:
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    if False:
        model_1.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="q4_k_m", token=""
        )  # Change hf to your username!
    if False:
        model_1.push_to_hub_gguf(
            "hf/model",
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
            token="",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.

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

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
    """)
    return


if __name__ == "__main__":
    app.run()
