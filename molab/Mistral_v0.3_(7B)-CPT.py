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
#     "transformers==4.56.2",
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
    """)
    return


@app.cell
def _():
    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = (
        "1 # Run this to disable CCE since it is not supported for CPT"
    )
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/mistral-7b-v0.3-bnb-4bit",  # New Mistral v3 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 15 trillion tokens model 2x faster!
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",  # Gemma 2.2x faster!
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-v0.3",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        dtype=dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False.
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return (
        FastLanguageModel,
        dtype,
        load_in_4bit,
        max_seq_length,
        model,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

    We also add `embed_tokens` and `lm_head` to allow the model to learn out of distribution data.
    """)
    return


@app.cell
def _(FastLanguageModel, model):
    model_1 = FastLanguageModel.get_peft_model(
        model,
        r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=True,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We now use the Korean subset of the [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) to first continually pretrain the model. You can use **any language** you like! Go to [Wikipedia's List of Languages](https://en.wikipedia.org/wiki/List_of_Wikipedias) to find your own language!

    **[NOTE]** To train only on completions (ignoring the user's input) read our docs [here](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#training-on-completions-only-masking-out-inputs)

    **[NOTE]** Remember to add the **EOS_TOKEN** to the tokenized output! Otherwise you'll get infinite generations!

    If you want to use the `llama-3` template for ShareGPT datasets, try our conversational [notebook](https://github.com/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb)

    For text completions like novel writing, try this [notebook](https://github.com/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb).

    **[NOTE]** Use https://translate.google.com to translate from English to Korean!
    """)
    return


@app.cell
def _(tokenizer):
    # Wikipedia provides a title and an article text.
    # Use https://translate.google.com!
    _wikipedia_prompt = """Wikipedia Article
    ### Title: {}

    ### Article:
    {}"""
    # becomes:
    wikipedia_prompt = """위키피디아 기사
    ### 제목: {}

    ### 기사:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        titles = examples["title"]
        texts = examples["text"]
        outputs = []
        for title, text in zip(titles, texts):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = wikipedia_prompt.format(title, text) + EOS_TOKEN
            outputs.append(text)
        return {
            "text": outputs,
        }

    pass
    return (formatting_prompts_func,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We only use 1% of the dataset to speed things up! Use more for longer runs!
    """)
    return


@app.cell
def _(formatting_prompts_func):
    from datasets import load_dataset

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ko",
        split="train",
    )

    # We select 1% of the data to make training faster!
    dataset = dataset.train_test_split(train_size=0.01)["train"]

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    return dataset, load_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Continued Pretraining
    Now let's use Unsloth's `UnslothTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 20 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

    Also set `embedding_learning_rate` to be a learning rate at least 2x or 10x smaller than `learning_rate` to make continual pretraining work!
    """)
    return


@app.cell
def _(dataset, max_seq_length, model_1, tokenizer):
    from transformers import TrainingArguments
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    trainer = UnslothTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        dataset_num_proc=4,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            max_steps=120,
            warmup_steps=10,
            learning_rate=5e-05,
            embedding_learning_rate=1e-05,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return UnslothTrainer, UnslothTrainingArguments, trainer


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instruction Finetuning

    We now use the [Alpaca in GPT4 Dataset](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-korean) but translated in Korean!

    Go to [vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) for the original GPT4 dataset for Alpaca or [MultilingualSIFT project](https://github.com/FreedomIntelligence/MultilingualSIFT) for other translations of the Alpaca dataset.
    """)
    return


@app.cell
def _(load_dataset):
    alpaca_dataset = load_dataset(
        "FreedomIntelligence/alpaca-gpt4-korean", split="train"
    )
    return (alpaca_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We print 1 example:
    """)
    return


@app.cell
def _(alpaca_dataset):
    print(alpaca_dataset[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We again use https://translate.google.com/ to translate the Alpaca format into Korean
    """)
    return


@app.cell
def _(alpaca_dataset, tokenizer):
    _alpaca_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n{}"
    alpaca_prompt = "다음은 작업을 설명하는 명령입니다. 요청을 적절하게 완료하는 응답을 작성하세요.\n\n### 지침:\n{}\n\n### 응답:\n{}"
    EOS_TOKEN_1 = tokenizer.eos_token

    def formatting_prompts_func_1(conversations):
        texts = []
        conversations = conversations["conversations"]
        # Becomes:
        for convo in conversations:
            text = (
                alpaca_prompt.format(convo[0]["value"], convo[1]["value"]) + EOS_TOKEN_1
            )
            texts.append(text)
        return {"text": texts}

    alpaca_dataset_1 = alpaca_dataset.map(
        formatting_prompts_func_1, batched=True
    )
    return alpaca_dataset_1, alpaca_prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We again employ `UnslothTrainer` and do instruction finetuning!
    """)
    return


@app.cell
def _(
    UnslothTrainer,
    UnslothTrainingArguments,
    alpaca_dataset_1,
    max_seq_length,
    model_1,
    tokenizer,
):
    trainer_1 = UnslothTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=alpaca_dataset_1,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            max_steps=120,
            warmup_steps=10,
            learning_rate=5e-05,
            embedding_learning_rate=1e-05,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use TrackIO/WandB etc
        ),
    )
    return (trainer_1,)


@app.cell
def _(trainer_1):
    trainer_stats_1 = trainer_1.train()
    return (trainer_stats_1,)


@app.cell
def _(max_memory, start_gpu_memory, torch, trainer_stats_1):
    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats_1.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats_1.metrics['train_runtime'] / 60, 2)} minutes used for training."
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

    Remember to use https://translate.google.com/!
    """)
    return


@app.cell
def _(FastLanguageModel, alpaca_prompt, model_1, tokenizer):
    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model_1)  # Enable native 2x faster inference
    inputs = tokenizer(
        [alpaca_prompt.format("피보나치 수열을 계속하세요: 1, 1, 2, 3, 5, 8,", "")],
        return_tensors="pt",
    ).to("cuda")
    outputs = model_1.generate(**inputs, max_new_tokens=64, use_cache=True)
    tokenizer.batch_decode(
        outputs
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!
    """)
    return


@app.cell
def _(FastLanguageModel, alpaca_prompt, model_1, tokenizer):
    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model_1)  # Enable native 2x faster inference
    inputs_1 = tokenizer(
        [alpaca_prompt.format("한국음악은 어떤가요?", "")], return_tensors="pt"
    ).to("cuda")
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer)
    _ = model_1.generate(
        **inputs_1, streamer=text_streamer, max_new_tokens=128
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By using https://translate.google.com/ we get
    ```
    Korean music is classified into many types of music genres.

    This genre is classified into different music genres such as pop songs,

    rock songs, classical songs and pop songs, music groups consisting of drums, fans, instruments and singers
    ```
    """)
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
    model_1.save_pretrained("mistral_v0_lora")
    tokenizer.save_pretrained("mistral_v0_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(
    TextStreamer,
    alpaca_prompt,
    dtype,
    load_in_4bit,
    max_seq_length,
    model_1,
    tokenizer,
):
    if False:
        from unsloth import FastLanguageModel as _FastLanguageModel

        _model, _tokenizer = _FastLanguageModel.from_pretrained(
            model_name="mistral_v0_lora",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
            max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
            dtype=dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False.
        )
        _FastLanguageModel.for_inference(_model)
    inputs_2 = tokenizer(
        [alpaca_prompt.format("지구를 광범위하게 설명하세요.", "")], return_tensors="pt"
    ).to("cuda")
    # alpaca_prompt = You MUST copy from above!
    text_streamer_1 = TextStreamer(tokenizer)
    _ = model_1.generate(
        **inputs_2, streamer=text_streamer_1, max_new_tokens=128, repetition_penalty=0.1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By using https://translate.google.com/ we get
    ```
    Earth refers to all things including natural disasters such as local derailment

    and local depletion that occur in one space along with the suppression of water, gases, and living things.

    Most of the Earth's water comes from oceans, atmospheric water, underground water layers, and rivers and rivers.
    ```

    Yikes the language model is a bit whacky! Change the temperature and using sampling will definitely make the output much better!
    """)
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
            "mistral_v0_lora", load_in_4bit=load_in_4bit
        )
        _tokenizer = _AutoTokenizer.from_pretrained("mistral_v0_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "mistral_v0_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/mistral_v0_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_merged(
            "mistral_v0_finetune_4bit", tokenizer, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/mistral_v0_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained("mistral_v0_lora")
        tokenizer.save_pretrained("mistral_v0_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/mistral_v0_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/mistral_v0_lora", token="YOUR_HF_TOKEN")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

    Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
    * `q8_0` - Fast conversion. High resource use, but generally acceptable.
    * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
    * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Save to 8bit Q8_0
    if False:
        model_1.save_pretrained_gguf("mistral_v0_finetune", tokenizer)
    if False:
        # Save to 16bit GGUF
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )
    if False:
        model_1.save_pretrained_gguf(
            "mistral_v0_finetune", tokenizer, quantization_method="f16"
        )
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_gguf(
            "mistral_v0_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune",
            tokenizer,
            quantization_method="q5_k_m",
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

      <b>This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)</b>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
