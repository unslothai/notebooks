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
    Load up `Gemma 3 1B Instruct`, and set parameters
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    max_seq_length = 1024  # Choose any for long context!

    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        # Other popular models!
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.3-70B",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Phi-4",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-1b-it",
        max_seq_length=max_seq_length,  # Choose any for long context!
        load_in_4bit=False,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastModel, max_seq_length, model, tokenizer


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
    ### Data Prep
    <a name="Data"></a>

    We're using OpenAI's famous GSM8K dataset!
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at the first row:
    """)
    return


@app.cell
def _(dataset):
    dataset[0]["question"]
    return


@app.cell
def _(dataset):
    dataset[0]["answer"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We notice all answers like about have a ####, so we extract it:
    """)
    return


@app.cell
def _(dataset):
    def extract_hash_answer(text):
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    extract_hash_answer(dataset[0]["answer"])
    return (extract_hash_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now create a system prompt which can be customized. We add 4 extra symbols for working out or thinking / reasoning sections and a final answer:
    """)
    return


@app.cell
def _():
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""
    system_prompt
    return (
        reasoning_end,
        reasoning_start,
        solution_end,
        solution_start,
        system_prompt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's map the dataset! and see the first row:
    """)
    return


@app.cell
def _(dataset, extract_hash_answer, system_prompt):
    dataset_1 = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    dataset_1[0]
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a regex format to match the reasoning sections and answers:
    """)
    return


@app.cell
def _(reasoning_end, reasoning_start, solution_end, solution_start):
    import re

    match_format = re.compile(
        rf"^[\s]{{0,}}"
        rf"{reasoning_start}.+?{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )
    return match_format, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We verify it works:
    """)
    return


@app.cell
def _(match_format):
    match_format.search(
        "<start_working_out>Let me think!<end_working_out><SOLUTION>2</SOLUTION>",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now want to create a reward function to match the format exactly - we reward it with 3 points if it succeeds:
    """)
    return


@app.cell
def _(match_format):
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if (
                match_format.search(response) is not None
            ):  # Match if format is seen exactly!
                score = score + 3.0
            scores.append(score)
        return scores

    return (match_format_exactly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If it fails, we want to reward the model if it at least follows the format partially, by counting each symbol:
    """)
    return


@app.cell
def _(reasoning_end, reasoning_start, solution_end, solution_start):
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score = score + (
                0.5 if response.count(reasoning_start) == 1 else -0.5
            )  # Count how many keywords are seen - we penalize if too many!
            score = score + (
                0.5 if response.count(reasoning_end) == 1 else -0.5
            )  # If we see 1, then plus some points!
            score = score + (0.5 if response.count(solution_start) == 1 else -0.5)
            score = score + (0.5 if response.count(solution_end) == 1 else -0.5)
            scores.append(score)
        return scores

    return (match_format_approximately,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:
    """)
    return


@app.cell
def _(match_format):
    def check_answer(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1) if (guess := match_format.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            if guess == true_answer:
                score = score + 3.0
            elif guess.strip() == true_answer.strip():
                score = score + 1.5
            else:
                try:
                    ratio = float(guess) / float(
                        true_answer
                    )  # Correct answer gets 3 points!
                    if ratio >= 0.9 and ratio <= 1.1:
                        score = score + 0.5
                    elif ratio >= 0.8 and ratio <= 1.2:  # Match if spaces are seen
                        score = score + 0.25
                    else:
                        score = score - 1.0
                except:  # We also reward it if the answer is close via ratios!
                    score = (
                        score - 0.5
                    )  # Ie if the answer is within some range, reward it!
            scores.append(score)
        return scores  # Penalize wrong answers  # Penalize

    return (check_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.
    """)
    return


@app.cell
def _(re, solution_start):
    match_numbers = re.compile(
        rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
    )
    match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>")
    return (match_numbers,)


@app.cell
def _(match_numbers):
    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        print(
            "*" * 20,
            f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip())
                scores.append(1.5 if guess == true_answer else 0.0)
            except:
                scores.append(0)
                continue
        return scores

    return (check_numbers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations!
    """)
    return


@app.cell
def _(max_seq_length):
    max_prompt_length = 256

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=50,
        save_steps=50,
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
    check_answer,
    check_numbers,
    dataset_1,
    match_format_approximately,
    match_format_exactly,
    model_1,
    tokenizer,
    training_args,
):
    trainer = GRPOTrainer(
        model=model_1,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args=training_args,
        train_dataset=dataset_1,
    )
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Now let's try the model we just trained!
    """)
    return


@app.cell
def _(model_1, system_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the sqrt of 101?"},
    ]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    from transformers import TextStreamer

    _ = model_1.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=64,  # Increase for longer outputs!
        temperature=1.0,
        top_p=0.95,
        top_k=64,
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
    model_1.save_pretrained("gemma_3_lora")
    tokenizer.save_pretrained("gemma_3_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly for deployment! We save it in the folder `gemma-3-finetune`. Set `if False` to `if True` to let it run!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    if False:  # Change to True to save finetune!
        model_1.save_pretrained_merged("gemma-3-finetune", tokenizer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    if False:  # Change to True to upload finetune
        model_1.push_to_hub_merged(
            "HF_ACCOUNT/gemma-3-finetune", tokenizer, token="YOUR_HF_TOKEN"
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
def _(model_1, tokenizer):
    if False:  # Change to True to save to GGUF
        model_1.save_pretrained_gguf(
            "gemma_3_finetune", tokenizer, quantization_method="Q8_0"
        )  # For now only Q8_0, BF16, F16 supported
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    if False:  # Change to True to upload GGUF
        model_1.push_to_hub_gguf(
            "HF_ACCOUNT/gemma_3_finetune",
            tokenizer,
            quantization_method="Q8_0",  # For now only Q8_0, BF16, F16 supported
            token="YOUR_HF_TOKEN",
        )  # Only Q8_0, BF16, F16 supported
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `gemma-3-finetune.gguf` file or `gemma-3-finetune-Q4_K_M.gguf` file in llama.cpp.

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
