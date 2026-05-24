# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "cuda-tile==1.2.0",
#     "langid",
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
    <div class="align-center"><a href="https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt"><img src="https://github.com/unslothai/notebooks/raw/main/assets/hf%20course.png" width="165"></a>
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    In this [Hugging Face](https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt) and Unsloth notebook, you will learn to transform DeepSeek R1 0528 Qwen3 (8B) GRPO into a Reasoning model using GRPO.

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
    Goal: To convert `DeepSeek-R1-0528-Qwen3-8B` into a reasoning model via GRPO by using OpenR1's Math dataset.

    We also use `langid` for language detection. Our main goal is to force the model to generate reasoning traces in Indonesian, and we create a reward function using `langid` to check this.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: langid !pip install langid -qq
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-0528-Qwen3-8B",
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
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )
    return max_seq_length, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GRPO Chat Template
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Distill Qwen3 from Deepseek has a chat template that is used to format the input and output of the model. This is used to make the model output in a chat format. Including the reasoning step. We have to use that chat template since the model is trained using it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how our chat template behaves on an example:
    """)
    return


@app.cell
def _(tokenizer):
    reasoning_start = None
    reasoning_end = None
    user_token = None
    assistant_token = None

    for token in tokenizer.get_added_vocab().keys():
        if "think" in token and "/" in token:
            reasoning_end = token
        elif "think" in token:
            reasoning_start = token
        elif "user" in token:
            user_token = token
        elif "assistant" in token:
            assistant_token = token

    system_prompt = f"""You are given a problem.
    Think about the problem and provide your working out.
    You must think in Bahasa Indonesia."""
    system_prompt
    return reasoning_end, reasoning_start, system_prompt


@app.cell
def _(tokenizer):
    print(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": f"<think>I think it's 2.2</think>2"},
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": f"<think>I think it's 2.2</think>2"},
            ],
            tokenize=False,
            add_generation_prompt=True,  # Must add for generation
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Prep
    <a name="Data"></a>

    We're using Hugging Face's [Open R1 Math dataset](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed). You can also utilize OpenAI's famous [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
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
    dataset[0]["prompt"]
    return


@app.cell
def _(dataset):
    dataset[0]["solution"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In GSM8K, we notice all answers like about have a ####, so we extract it. But for the Open R1 dataset, we can skip the below.
    """)
    return


@app.cell
def _(dataset):
    def extract_hash_answer(text):
        # if "####" not in text: return None
        # return text.split("####")[1].strip()
        return text

    extract_hash_answer(dataset[0]["solution"])
    return (extract_hash_answer,)


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
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": extract_hash_answer(x["solution"]),
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
def _(reasoning_end):
    import re

    # Add optional EOS token matching
    solution_end_regex = rf"{reasoning_end}(.*)"

    match_format = re.compile(solution_end_regex, re.DOTALL)
    match_format
    return match_format, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We verify it works:
    """)
    return


@app.cell
def _(match_format):
    match_format.findall(
        f"Let me think!</think>Hence, the solution is 2.",
    )
    return


@app.cell
def _(match_format):
    match_format.findall(
        f"<think>Let me think!</think>\n\nHence, the solution is 2",
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
def _(reasoning_end, reasoning_start):
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            score = score + (
                0.5 if response.count(reasoning_start) == 1 else -1.0
            )  # Count how many keywords are seen - we penalize if too many!
            score = score + (
                0.5 if response.count(reasoning_end) == 1 else -1.0
            )  # If we see 1, then plus some points!
            scores.append(score)
        return scores  # No need to reward the think tag since we always prepend it!

    return (match_format_approximately,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We want to extract the generated answer, and reward or penalize it! We also reward it based on how close the answer is to the true one via ratios:
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
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score = score + 5.0
            elif guess.strip() == true_answer.strip():
                score = score + 3.5
            else:
                try:
                    ratio = float(guess) / float(
                        true_answer
                    )  # Correct answer gets 5 points!
                    if ratio >= 0.9 and ratio <= 1.1:
                        score = score + 2.0
                    elif (
                        ratio >= 0.8 and ratio <= 1.2
                    ):  # Match if spaces are seen, but less reward
                        score = score + 1.5
                    else:
                        score = score - 2.5
                except:  # We also reward it if the answer is close via ratios!
                    score = (
                        score - 4.5
                    )  # Ie if the answer is within some range, reward it!
            scores.append(score)
        return scores  # Penalize wrong answers  # Penalize

    return (check_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Also sometimes it might not be 1 number as the answer, but like a sentence for example "The solution is $20" -> we extract 20.

    We also remove possible commas for example as in 123,456
    """)
    return


@app.cell
def _(re):
    match_numbers = re.compile(
        r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL
    )
    print(match_numbers.findall("  0.34  "))
    print(match_numbers.findall("  123,456  "))
    print(match_numbers.findall("  -0.234  "))
    print(match_numbers.findall("17"))
    return (match_numbers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we will try to enforce the thinking process to be in Bahasa Indonesia. This is a simple version of the `language consistency reward` that is used in DeepSeek R1 paper
    """)
    return


@app.cell
def _():
    import langid

    def get_lang(text: str) -> str:
        if not text:
            return "und"
        lang, _ = langid.classify(text)
        return lang

    print(get_lang("Hello, How are you"))  # This should return en
    print(get_lang("Aku berpikir kalau aku adalah kamu"))  # This should return id
    print(get_lang("我在这里"))  # This should return zh
    return (get_lang,)


@app.cell
def _(get_lang):
    def format_and_language_reward_func(completions, **kwargs):
        scores = []
        for completion_item in completions:
            if (
                not completion_item
                or not isinstance(completion_item[0], dict)
                or "content" not in completion_item[0]
            ):
                scores.append(-5.0)
                print(
                    f"Warning: Malformed completion item, assigning default low score: {completion_item}"
                )
                continue
            content = completion_item[0]["content"]
            lang = get_lang(content)
            if lang == "id":
                score = 5.0
            elif lang == "en":
                score = -3.0
            elif lang == "zh":
                score = -3.0
            else:
                score = -5.0
            scores.append(score)
        return scores

    return (format_and_language_reward_func,)


@app.cell
def _(format_and_language_reward_func):
    prompts = [
        [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
        [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
    ]
    completions = [
        [
            {
                "role": "assistant",
                "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>",
            }
        ],
        [
            {
                "role": "assistant",
                "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8.",
            }
        ],
    ]
    format_and_language_reward_func(prompts=prompts, completions=completions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now prepare our main function which will print out the generated responses and the true answer, along with another reward function which converts text to float via `float` and sees if it's the same.
    """)
    return


@app.cell
def _(match_numbers):
    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print(
                "*" * 20 + f"Question:\n{question}",
                f"\nAnswer:\n{answer[0]}",
                f"\nResponse:\n{responses[0]}",
                f"\nExtracted:\n{extracted_responses[0]}",
            )
        PRINTED_TIMES = PRINTED_TIMES + 1
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:  # Print only every few steps
                scores.append(-2.5)
                continue
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores  # Convert to numbers  # Remove commas like in 123,456

    return PRINTED_TIMES, PRINT_EVERY_STEPS, check_numbers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get the top 90% prompt length so we don't accidentally truncate them!

    Ie we'll remove the top 10% long prompts.
    """)
    return


@app.cell
def _(dataset_1, tokenizer):
    tokenized = dataset_1.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt=True, tokenize=True
            )
        },
        batched=True,
    )
    print(tokenizer.decode(tokenized[0]["tokens"]))
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    import numpy as np

    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print("Max Length = ", maximum_length)
    dataset_2 = dataset_1.select(
        np.where(np.array(tokenized["L"]) <= maximum_length)[0]
    )
    # Filter only samples smaller than 90% max length
    del tokenized
    return dataset_2, maximum_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations!
    """)
    return


@app.cell
def _(max_seq_length, maximum_length, tokenizer):
    max_prompt_length = maximum_length + 1  # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    from vllm import SamplingParams

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,  # + 1 just in case!
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=100,
        save_steps=100,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )
    return GRPOTrainer, SamplingParams, training_args


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
    dataset_2,
    format_and_language_reward_func,
    match_format_approximately,
    match_format_exactly,
    model,
    tokenizer,
    training_args,
):
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
            format_and_language_reward_func,
        ],
        args=training_args,
        train_dataset=dataset_2,
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
def _(SamplingParams, model):
    text = "What is the sqrt of 101?"
    sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    output = (
        model.fast_generate([text], sampling_params=sampling_params, lora_request=None)[
            0
        ]
        .outputs[0]
        .text
    )
    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now with the LoRA we just trained with GRPO - we first save the LoRA first!
    """)
    return


@app.cell
def _(model):
    model.save_lora("grpo_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verify LoRA is actually trained!
    """)
    return


@app.cell
def _():
    from safetensors import safe_open

    tensors = {}
    with safe_open("grpo_lora/adapter_model.safetensors", framework="pt") as f:
        # Verify both A and B are non zero
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert n_zeros.item() != tensor.numel()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we load the LoRA and test. We tested without using our custom system prompt which should not (or minimal) affect toward the model's original reasoning ability.:
    """)
    return


@app.cell
def _(SamplingParams, model, tokenizer):
    messages = [{"role": "user", "content": "Solve (x + 2)^2 = 0"}]
    text_1 = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    sampling_params_1 = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    output_1 = (
        model.fast_generate(
            text_1,
            sampling_params=sampling_params_1,
            lora_request=model.load_lora("grpo_lora"),
        )[0]
        .outputs[0]
        .text
    )
    output_1  # Must add for generation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let's test using our system prompt which should use the new language :
    """)
    return


@app.cell
def _(SamplingParams, model, system_prompt, tokenizer):
    messages_1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Solve (x + 2)^2 = 0"},
    ]
    text_2 = tokenizer.apply_chat_template(
        messages_1, add_generation_prompt=True, tokenize=False
    )
    sampling_params_2 = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    output_2 = (
        model.fast_generate(
            text_2,
            sampling_params=sampling_params_2,
            lora_request=model.load_lora("grpo_lora"),
        )[0]
        .outputs[0]
        .text
    )
    output_2  # Must add for generation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets compare our results with system prompt but without our LoRA
    """)
    return


@app.cell
def _(SamplingParams, model, system_prompt, tokenizer):
    messages_2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Solve (x + 2)^2 = 0"},
    ]
    text_3 = tokenizer.apply_chat_template(
        messages_2, add_generation_prompt=True, tokenize=False
    )
    sampling_params_3 = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    output_3 = (
        model.fast_generate(
            text_3, sampling_params=sampling_params_3, lora_request=None
        )[0]
        .outputs[0]
        .text
    )
    output_3  # Must add for generation
    return (sampling_params_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take 20 samples, and compare the the amount of using our LoRA and not using it, and see which one has better amount of correct language
    """)
    return


@app.cell
def _(dataset_2):
    sample_dataset = dataset_2.shuffle(seed=3407).select(range(20))
    sample_dataset
    return (sample_dataset,)


@app.cell
def _(
    get_lang,
    model,
    sample_dataset,
    sampling_params_3,
    system_prompt,
    tokenizer,
):
    with_lora_id_count = 0
    without_lora_id_count = 0
    print("Comparing language usage with and without LoRA on 20 samples:")
    print("=" * 60)
    for i, sample in enumerate(sample_dataset):
        messages_3 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"][1]["content"]},
        ]
        text_4 = tokenizer.apply_chat_template(
            messages_3, add_generation_prompt=True, tokenize=False
        )
        output_with_lora = (
            model.fast_generate(
                text_4,
                sampling_params=sampling_params_3,
                lora_request=model.load_lora("grpo_lora"),
            )[0]
            .outputs[0]
            .text
        )
        output_without_lora = (
            model.fast_generate(
                text_4, sampling_params=sampling_params_3, lora_request=None
            )[0]
            .outputs[0]
            .text
        )
        lang_with_lora = get_lang(output_with_lora)
        lang_without_lora = get_lang(output_without_lora)
        if lang_with_lora == "id":
            with_lora_id_count = with_lora_id_count + 1
        if lang_without_lora == "id":
            without_lora_id_count = without_lora_id_count + 1
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/20 samples...")
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(
        f"With LoRA - Indonesian responses: {with_lora_id_count}/20 ({with_lora_id_count / 20 * 100:.1f}%)"
    )
    print(
        f"Without LoRA - Indonesian responses: {without_lora_id_count}/20 ({without_lora_id_count / 20 * 100:.1f}%)"
    )
    print(
        f"Improvement: +{with_lora_id_count - without_lora_id_count} Indonesian responses with LoRA"
    )  # Print progress every 5 samples
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
            "deepseek_r1_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/deepseek_r1_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )

    # Merge to 4bit
    if False:
        model.save_pretrained_merged(
            "deepseek_r1_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
        )
    if False:
        model.push_to_hub_merged(
            "HF_USERNAME/deepseek_r1_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )

    # Just LoRA adapters
    if False:
        model.save_pretrained("deepseek_r1_lora")
        tokenizer.save_pretrained("deepseek_r1_lora")
    if False:
        model.push_to_hub("HF_USERNAME/deepseek_r1_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/deepseek_r1_lora", token="YOUR_HF_TOKEN")
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
            "deepseek_r1_finetune",
            tokenizer,
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/deepseek_r1_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )

    # Save to 16bit GGUF
    if False:
        model.save_pretrained_gguf(
            "deepseek_r1_finetune", tokenizer, quantization_method="f16"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/deepseek_r1_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )

    # Save to q4_k_m GGUF
    if False:
        model.save_pretrained_gguf(
            "deepseek_r1_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/deepseek_r1_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/deepseek_r1_finetune",  # Change hf to your username!
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
    Now, use the `deepseek_r1_finetune.Q8_0.gguf` file or `deepseek_r1_finetune.Q4_K_M.gguf` file in llama.cpp.

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
