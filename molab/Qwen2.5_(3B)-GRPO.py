# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "cuda-tile==1.2.0",
#     "marimo",
#     "torchao>=0.16.0",
#     "torchvision",
#     "transformers>=4.56.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth.git",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
#     "uv",
#     "vllm>=0.11.0",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Load up `Qwen 2.5 3B Instruct`, and set parameters
    """)
    return


@app.cell
def _():
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch

    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,  # Can increase for longer reasoning traces
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vllm fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,  # Reduce if out of memory
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
        ],  # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Prep
    <a name="Data"></a>

    We directly leverage [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for data prep and all reward functions. You are free to create your own!
    """)
    return


@app.cell
def _():
    import re
    from datasets import load_dataset, Dataset

    SYSTEM_PROMPT = "\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n"
    # Load and prep dataset
    XML_COT_FORMAT = (
        "<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>\n"
    )

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def get_gsm8k_questions(split="train") -> Dataset:
        data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
        data = data.map(  # type: ignore
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )
        return data

    dataset = get_gsm8k_questions()

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    # uncomment middle messages for 1-shot prompting

    def strict_format_reward_func(completions, **kwargs) -> list[float]:  # type: ignore
        """Reward function that checks if the completion has a specific format."""  # type: ignore
        pattern = "^<reasoning>\\n.*?\\n</reasoning>\\n<answer>\\n.*?\\n</answer>\\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:  # type: ignore
        """Reward function that checks if the completion has a specific format."""  # type: ignore
        pattern = "<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        # Reward functions
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count = count + 0.125
        if text.count("\n</reasoning>\n") == 1:
            count = count + 0.125
        if text.count("\n<answer>\n") == 1:
            count = count + 0.125
            count = count - len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count = count + 0.125
            count = count - (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]

    return (
        SYSTEM_PROMPT,
        correctness_reward_func,
        dataset,
        int_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        xmlcount_reward_func,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations!
    """)
    return


@app.cell
def _():
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        use_vllm=True,  # use vllm for fast inference!
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=8,  # Decrease if out of memory
        max_prompt_length=256,
        max_completion_length=200,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
    )
    return GRPOTrainer, training_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

    You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

    | Step | Training Loss | reward    | reward_std | completion_length | kl       |
    |------|---------------|-----------|------------|-------------------|----------|
    | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
    | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
    | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
    """)
    return


@app.cell
def _(
    GRPOTrainer,
    correctness_reward_func,
    dataset,
    int_reward_func,
    model,
    soft_format_reward_func,
    strict_format_reward_func,
    tokenizer,
    training_args,
    xmlcount_reward_func,
):
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Now let's try the model we just trained! First, let's first try the model without any GRPO trained:
    """)
    return


@app.cell
def _(model, tokenizer):
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "How many r's are in strawberry?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )

    output
    return (SamplingParams,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now with the LoRA we just trained with GRPO - we first save the LoRA first!
    """)
    return


@app.cell
def _(model):
    model.save_lora("grpo_saved_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we load the LoRA and test:
    """)
    return


@app.cell
def _(SYSTEM_PROMPT, SamplingParams, model, tokenizer):
    text_1 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "How many r's are in strawberry?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params_1 = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
    output_1 = (
        model.fast_generate(
            text_1,
            sampling_params=sampling_params_1,
            lora_request=model.load_lora("grpo_saved_lora"),
        )[0]
        .outputs[0]
        .text
    )
    output_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our reasoning model is much better - it's not always correct, since we only trained it for an hour or so - it'll be better if we extend the sequence length and train for longer!
    """)
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
            "qwen_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "qwen_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("qwen_lora")
        tokenizer.save_pretrained("qwen_lora")
    if False:
        model.push_to_hub("HF_USERNAME/qwen_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/qwen_lora", token="YOUR_HF_TOKEN")
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
            "qwen_finetune",
            tokenizer,
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )

    # Save to 16bit GGUF
    if False:
        model.save_pretrained_gguf(
            "qwen_finetune", tokenizer, quantization_method="f16"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )

    # Save to q4_k_m GGUF
    if False:
        model.save_pretrained_gguf(
            "qwen_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",  # Change hf to your username!
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
    Now, use the `qwen_finetune.Q8_0.gguf` file or `qwen_finetune.Q4_K_M.gguf` file in llama.cpp.

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
