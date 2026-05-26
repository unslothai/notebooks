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
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/llama-2-13b-bnb-4bit",
        "unsloth/codellama-34b-bnb-4bit",
        "unsloth/tinyllama-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",  # New Google 6 trillion tokens model 2.5x faster!
        "unsloth/gemma-2b-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
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
    """)
    return


@app.cell
def _(FastLanguageModel, model):
    model_1 = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We now use the `ChatML` format for conversation style finetunes. We use [Open Assistant conversations](https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style) in ShareGPT style. ChatML renders multi turn conversations like below:

    ```
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What's the capital of France?<|im_end|>
    <|im_start|>assistant
    Paris.
    ```

    **[NOTE]** To train only on completions (ignoring the user's input) read our docs [here](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#training-on-completions-only-masking-out-inputs)

    We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old` and our own optimized `unsloth` template.

    Normally one has to train `<|im_start|>` and `<|im_end|>`. We instead map `<|im_end|>` to be the EOS token, and leave `<|im_start|>` as is. This requires no additional training of additional tokens.

    Note ShareGPT uses `{"from": "human", "value" : "Hi"}` and not `{"role": "user", "content" : "Hi"}`, so we use `mapping` to map it.

    For text completions like novel writing, try this [notebook](https://github.com/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb).
    """)
    return


@app.cell
def _(tokenizer):
    from unsloth.chat_templates import get_chat_template

    tokenizer_1 = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping={  # ShareGPT style
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt",
        },
        map_eos_token=True,  # Maps <|im_end|> to </s> instead
    )

    def formatting_prompts_func(examples):
        convos = examples[
            "conversations"
        ]  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        texts = [
            tokenizer_1.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]  # ShareGPT style
        return {"text": texts}  # Maps <|im_end|> to </s> instead

    from datasets import load_dataset

    dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset, get_chat_template, tokenizer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how the `ChatML` format works by printing the 5th element
    """)
    return


@app.cell
def _(dataset):
    dataset[5]["conversations"]
    return


@app.cell
def _(dataset):
    print(dataset[5]["text"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you're looking to make your own chat template, that also is possible! You must use the Jinja templating regime. We provide our own stripped down version of the `Unsloth template` which we find to be more efficient, and leverages ChatML, Zephyr and Alpaca styles.

    More info on chat templates on [our wiki page!](https://github.com/unslothai/unsloth/wiki#chat-templates)
    """)
    return


@app.cell
def _(get_chat_template):
    unsloth_template = (
        "{{ bos_token }}"
        "{{ 'You are a helpful assistant to the user\n' }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '>>> User: ' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '>>> Assistant: ' }}"
        "{% endif %}"
    )
    unsloth_eos_token = "eos_token"

    if False:
        _tokenizer = get_chat_template(
            _tokenizer,
            chat_template=(unsloth_template, unsloth_eos_token),  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            mapping={  # ShareGPT style
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt",
            },
            map_eos_token=True,  # Maps <|im_end|> to </s> instead
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!!
    """)
    return


@app.cell
def _(dataset, max_seq_length, model_1, tokenizer_1):
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer_1,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=0.0002,
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
    Let's run the model! Since we're using `ChatML`, use `apply_chat_template` with `add_generation_prompt` set to `True` for inference.
    """)
    return


@app.cell
def _(FastLanguageModel, get_chat_template, model_1, tokenizer_1):
    tokenizer_2 = get_chat_template(
        tokenizer_1,
        chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping={  # ShareGPT style
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt",
        },
        map_eos_token=True,  # Maps <|im_end|> to </s> instead
    )
    FastLanguageModel.for_inference(model_1)
    messages = [
        {"from": "human", "value": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"}
    ]
    inputs = tokenizer_2.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model_1.generate(
        input_ids=inputs, max_new_tokens=64, use_cache=True
    )  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
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
def _(FastLanguageModel, model_1, tokenizer_2):
    FastLanguageModel.for_inference(model_1)  # Enable native 2x faster inference
    messages_1 = [
        {"from": "human", "value": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"}
    ]
    inputs_1 = tokenizer_2.apply_chat_template(
        messages_1, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer_2)
    _ = model_1.generate(
        input_ids=inputs_1, streamer=text_streamer, max_new_tokens=128, use_cache=True
    )  # Must add for generation
    return (TextStreamer,)


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
    model_1.save_pretrained("mistral_v0_lora")
    tokenizer_2.save_pretrained("mistral_v0_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, dtype, load_in_4bit, max_seq_length, model_1, tokenizer_2):
    if False:
        from unsloth import FastLanguageModel as _FastLanguageModel

        _model, _tokenizer = _FastLanguageModel.from_pretrained(
            model_name="mistral_v0_lora",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
            max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
            dtype=dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False.
        )
        _FastLanguageModel.for_inference(_model)
    messages_2 = [{"from": "human", "value": "What is a famous tall tower in Paris?"}]
    inputs_2 = tokenizer_2.apply_chat_template(
        messages_2, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    text_streamer_1 = TextStreamer(tokenizer_2)
    _ = model_1.generate(
        input_ids=inputs_2, streamer=text_streamer_1, max_new_tokens=128, use_cache=True
    )  # Must add for generation
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
def _(model_1, tokenizer_2):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "mistral_v0_finetune_16bit", tokenizer_2, save_method="merged_16bit"
        )
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/mistral_v0_finetune_16bit",
            tokenizer_2,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_merged(
            "mistral_v0_finetune_4bit", tokenizer_2, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/mistral_v0_finetune_4bit",
            tokenizer_2,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained("mistral_v0_lora")
        tokenizer_2.save_pretrained("mistral_v0_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/mistral_v0_lora", token="YOUR_HF_TOKEN")
        tokenizer_2.push_to_hub("HF_USERNAME/mistral_v0_lora", token="YOUR_HF_TOKEN")
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
def _(model_1, tokenizer_2):
    # Save to 8bit Q8_0
    if False:
        model_1.save_pretrained_gguf("mistral_v0_finetune", tokenizer_2)
    if False:
        # Save to 16bit GGUF
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune", tokenizer_2, token="YOUR_HF_TOKEN"
        )
    if False:
        model_1.save_pretrained_gguf(
            "mistral_v0_finetune", tokenizer_2, quantization_method="f16"
        )
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune",
            tokenizer_2,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_gguf(
            "mistral_v0_finetune", tokenizer_2, quantization_method="q4_k_m"
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/mistral_v0_finetune",
            tokenizer_2,
            quantization_method="q4_k_m",
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
