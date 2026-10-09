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
#     "transformers==4.57.6",
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Goal: Make `Llama-3.2-1B-Instruct` give better answers with **PPO** (Proximal Policy Optimization), the classic RLHF algorithm used to train InstructGPT and ChatGPT. A reward model scores each answer, and PPO nudges the model towards answers that score higher. We give the model a budget of 128 new tokens, so it also learns to answer directly and finish its answer.

    PPO uses 4 models:
    1. **Policy**: the model we train. We load it with Unsloth and add LoRA adapters.
    2. **Reference model**: the starting model, used as a KL penalty so the policy does not drift too far. With LoRA this is free: we simply turn the adapters off!
    3. **Reward model**: scores a full answer. We use [Skywork-Reward-V2-Llama-3.2-1B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.2-1B), which uses the same tokenizer as Llama 3.2.
    4. **Value model** (critic): predicts the final reward at every token, so PPO knows which tokens helped. We start it from the reward model and train it with LoRA.

    Compared to our [GRPO notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), PPO learns a value model instead of sampling a group of answers per prompt, and it works with any learned reward model.
    """)
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 512  # Prompt + response length
    lora_rank = 16  # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=max_seq_length,  # Prompt + response length
        load_in_4bit=True,  # False for LoRA 16bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )
    return FastLanguageModel, lora_rank, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reward model
    The reward model reads a whole conversation and returns one number: higher means a better answer. PPO feeds it the policy's tokens directly, so it must use the same tokenizer as the policy.
    """)
    return


@app.cell
def _(torch):
    from transformers import AutoModelForSequenceClassification
    from unsloth import is_bfloat16_supported

    reward_model_name = "Skywork/Skywork-Reward-V2-Llama-3.2-1B"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    ).to("cuda")
    return AutoModelForSequenceClassification, reward_model, reward_model_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check it prefers a good answer over a bad one:
    """)
    return


@app.cell
def _(reward_model, tokenizer, torch):
    def get_score(messages):
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            return reward_model(input_ids).logits[0, 0].item()

    question = {"role": "user", "content": "What is the capital of France?"}
    print(
        "Good answer:",
        get_score(
            [
                question,
                {"role": "assistant", "content": "The capital of France is Paris."},
            ]
        ),
    )
    print(
        "Bad answer: ",
        get_score(
            [question, {"role": "assistant", "content": "I think it is London."}]
        ),
    )
    return (get_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Value model
    The value model has the same shape as the reward model (a transformer with a 1 number head), so we start from the reward model's weights, which already know what a good answer looks like. We train it with LoRA adapters.
    """)
    return


@app.cell
def _(
    AutoModelForSequenceClassification,
    lora_rank,
    reward_model,
    reward_model_name,
):
    from peft import LoraConfig, get_peft_model

    value_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        torch_dtype=reward_model.dtype,
    )
    value_model = get_peft_model(
        value_model,
        LoraConfig(
            r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            lora_alpha=lora_rank * 2,  # *2 speeds up training
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    value_model.print_trainable_parameters()
    return (value_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We use the prompts from [UltraFeedback](https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt), a large set of diverse user requests. PPO only needs prompts: the model writes the answers itself, and the reward model grades them.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
    dataset[0]["prompt"]
    return dataset, load_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We apply the chat template and tokenize the prompts. We drop prompts longer than 128 tokens to leave room for the answer:
    """)
    return


@app.cell
def _(dataset, tokenizer):
    max_prompt_length = 128

    def tokenize(example):
        input_ids = tokenizer.apply_chat_template(
            example["prompt"], add_generation_prompt=True, tokenize=True
        )
        return {"input_ids": input_ids, "length": len(input_ids)}

    dataset_1 = dataset.map(
        tokenize, remove_columns=dataset.column_names
    )  # Must add for generation
    dataset_1 = dataset_1.filter(
        lambda x: x["length"] <= max_prompt_length
    ).remove_columns("length")
    eval_dataset = dataset_1.select(range(8))  # A few prompts to watch while training
    dataset_1 = dataset_1.select(range(8, len(dataset_1)))
    print(
        tokenizer.decode(dataset_1[0]["input_ids"])
    )  # A few prompts to watch while training
    return dataset_1, eval_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up the PPO Trainer! Each PPO step:
    1. Generates answers for `per_device_train_batch_size * gradient_accumulation_steps` prompts.
    2. Scores them with the reward model, and subtracts a KL penalty against the reference model.
    3. Updates the policy and value model for `num_ppo_epochs` passes over those answers.

    `missing_eos_penalty` lowers the score of answers that did not finish within `response_length` tokens, which teaches the model to finish its answers. `kl_coef` controls how strongly the model is kept close to the reference model: PPO will also learn the reward model's blind spots if it is allowed to drift too far.
    """)
    return


@app.cell
def _():
    from trl import PPOConfig, PPOTrainer

    training_args = PPOConfig(
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # 2 * 4 = 8 prompts per PPO step
        local_rollout_forward_batch_size=8,  # Generate all 8 answers at once
        num_ppo_epochs=4,
        num_mini_batches=1,
        total_episodes=8  # 150 PPO steps of 8 prompts. Increase for better results!
        * 150,  # 150 PPO steps of 8 prompts. Increase for better results!
        response_length=128,  # Max new tokens per answer
        temperature=0.7,
        stop_token="eos",
        missing_eos_penalty=1.0,
        kl_coef=0.05,
        num_sample_generations=0,  # Set > 0 to print sample answers during training
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        save_strategy="no",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use TrackIO/WandB etc
    )
    return PPOTrainer, training_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's run the trainer! The goal is to see `objective/scores` (the reward model's score) go up. `objective/kl` is how far the model moved from the reference model, summed over each answer's tokens: it rises at first and should then level off.

    | Steps   | objective/scores | objective/kl | loss/value_avg |
    |---------|------------------|--------------|----------------|
    | 1-25    | 2.27             | 4.0          | 1.70           |
    | 51-75   | 2.99             | 15.9         | 0.62           |
    | 126-150 | 3.35             | 21.1         | 0.48           |

    (Averages over 25 steps from our run on a free molab Tesla T4, which took about 45 minutes. Single steps are noisy, since each one only scores 8 answers.)
    """)
    return


@app.cell
def _(
    PPOTrainer,
    dataset_1,
    eval_dataset,
    model,
    reward_model,
    tokenizer,
    training_args,
    value_model,
):
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=model,
        ref_model=None,  # With LoRA, the reference model is the model with adapters disabled
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset_1,
        eval_dataset=eval_dataset,  # A few prompts to watch while training
    )
    trainer.train()  # With LoRA, the reference model is the model with adapters disabled
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Now let's compare the model before and after PPO! We answer a few new prompts with the LoRA adapters off (the original model) and on (after PPO), and score both with the reward model:
    """)
    return


@app.cell
def _(FastLanguageModel, get_score, load_dataset, model, tokenizer):
    test_prompts = load_dataset("trl-lib/ultrafeedback-prompt", split="test").select(
        range(32)
    )["prompt"]

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    def answer(prompts):
        texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
        )
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        finished = sum((row == tokenizer.eos_token_id).any().item() for row in outputs)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True), finished

    with model.disable_adapter():
        before, before_finished = answer(test_prompts)
    after, after_finished = answer(test_prompts)

    before_scores = [
        get_score(p + [{"role": "assistant", "content": a}])
        for p, a in zip(test_prompts, before)
    ]
    after_scores = [
        get_score(p + [{"role": "assistant", "content": a}])
        for p, a in zip(test_prompts, after)
    ]
    print(
        f"Before PPO: average reward = {sum(before_scores) / len(before_scores):.2f}, finished answers = {before_finished}/{len(test_prompts)}"
    )
    print(
        f"After PPO:  average reward = {sum(after_scores) / len(after_scores):.2f}, finished answers = {after_finished}/{len(test_prompts)}"
    )
    return after, before, test_prompts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In our run, the average reward went from 3.36 to 3.93, and the model finished 12 of 32 answers within 128 tokens instead of 2. Let's look at one example:
    """)
    return


@app.cell
def _(after, before, test_prompts):
    print("PROMPT:\n", test_prompts[0][-1]["content"])
    print("\nBEFORE PPO:\n", before[0])
    print("\nAFTER PPO:\n", after[0])
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
def _(model, tokenizer):
    model.save_pretrained("llama_lora")  # Local saving
    tokenizer.save_pretrained("llama_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model, tokenizer):
    # Merge to 16bit
    if False:
        model.save_pretrained_merged(
            "llama_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/llama_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "llama_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/llama_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("llama_lora")
        tokenizer.save_pretrained("llama_lora")
    if False:
        model.push_to_hub("HF_USERNAME/llama_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/llama_lora", token="YOUR_HF_TOKEN")
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

    [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    """)
    return


@app.cell
def _(model, tokenizer):
    # Save to 8bit Q8_0
    if False:
        model.save_pretrained_gguf(
            "llama_finetune",
            tokenizer,
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/llama_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )

    # Save to 16bit GGUF
    if False:
        model.save_pretrained_gguf(
            "llama_finetune", tokenizer, quantization_method="f16"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/llama_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )

    # Save to q4_k_m GGUF
    if False:
        model.save_pretrained_gguf(
            "llama_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/llama_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/llama_finetune",  # Change hf to your username!
            tokenizer,
            quantization_method=[
                "q4_k_m",
                "q8_0",
                "q5_k_m",
            ],
            token="YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `llama_finetune.Q8_0.gguf` file or `llama_finetune.Q4_K_M.gguf` file in llama.cpp.

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
