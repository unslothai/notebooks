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
    <div class="align-center"><a href="https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt"><img src="https://github.com/unslothai/notebooks/raw/main/assets/hf%20course.png" width="165"></a>
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    In this [Hugging Face](https://huggingface.co/learn/nlp-course/en/chapter12/6?fw=pt) and Unsloth notebook, you will learn to transform Gemma3 (4B) Vision GRPO into a Reasoning model using GRPO.

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
    from unsloth import FastVisionModel  # FastLanguageModel for LLMs
    import torch

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",  # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
        "unsloth/Pixtral-12B-2409-bnb-4bit",  # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",  # Pixtral base model
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",  # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",  # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/gemma-3-4b-it",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    return FastVisionModel, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters for parameter efficient fine-tuning, allowing us to train only 1% of all model parameters efficiently.

    **[NEW]** We also support fine-tuning only the vision component, only the language component, or both. Additionally, you can choose to fine-tune the attention modules, the MLP layers, or both!
    """)
    return


@app.cell
def _(FastVisionModel, model):
    model_1 = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # False if not finetuning vision layers
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
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    AI4Math/MathVista is a dataset that involves using images to solve logic and math problems, for this notebook, it will only be math problems with numeric answers for simplicity.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    return GRPOConfig, GRPOTrainer, dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us see what our data looks like
    """)
    return


@app.cell
def _(dataset):
    dataset["decoded_image"][5]
    return


@app.cell
def _(dataset):
    dataset["question"][5], dataset["answer"][5]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The image of our data is a part of the math problem.

    To make the rewarding easy later on, let us filter only the numeric answer so that we can create reward function that gives score if reward is float or not.
    """)
    return


@app.cell
def _(dataset):
    def is_numeric_answer(example):
        try:
            float(example["answer"])
            return True
        except:
            return False

    dataset_1 = dataset.filter(is_numeric_answer)  #
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also resize the images to be 512 by 512 pixels to make the images manageable in context length. We also convert them to RGB so they are compatible with TRL's trainer!
    """)
    return


@app.cell
def _(dataset_1):
    # Filter have big images
    def resize_images(example):
        image = example["decoded_image"]
        image = image.resize((512, 512))
        example["decoded_image"] = image
        return example

    dataset_2 = dataset_1.map(resize_images)

    def convert_to_rgb(example):
        image = example["decoded_image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["decoded_image"] = image
        return example

    dataset_2 = dataset_2.map(convert_to_rgb)
    return (dataset_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the conversational template that is needed to collate the dataset
    """)
    return


@app.cell
def _(dataset_2):
    # Define the delimiter variables for clarity and easy modification
    REASONING_START = "<REASONING>"
    REASONING_END = "</REASONING>"
    SOLUTION_START = "<SOLUTION>"
    SOLUTION_END = "</SOLUTION>"

    def make_conversation(example):
        text_content = f"{example['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"  # Define placeholder constants if they are not defined globally
        prompt = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text_content}],
            }
        ]
        return {
            "prompt": prompt,
            "image": example["decoded_image"],
            "answer": example["answer"],
        }  # The user's text prompt

    train_dataset = dataset_2.map(make_conversation)
    train_dataset = train_dataset.remove_columns("image")
    train_dataset = train_dataset.rename_column(
        "decoded_image", "image"
    )
    return (
        REASONING_END,
        REASONING_START,
        SOLUTION_END,
        SOLUTION_START,
        train_dataset,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Applying Chat Template across the entire dataset
    """)
    return


@app.cell
def _(tokenizer, train_dataset):
    from unsloth_zoo.utils import Version

    if Version("trl") < Version("0.24.0"):
        # Only apply chat template for TRL < 0.24.0, otherwise TRL handles it
        train_dataset_1 = train_dataset.map(
            lambda example: {
                "prompt": tokenizer.apply_chat_template(
                    example["prompt"], tokenize=False, add_generation_prompt=False
                )
            }
        )
    return (train_dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use a basic formatting functions to see if reasoning starts and ends as well as if the answers were written correctly.
    """)
    return


@app.cell
def _(REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START):
    # Reward functions
    def formatting_reward_func(completions, **kwargs):
        import re

        thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
        answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
        scores = []
        for completion in completions:
            score = 0
            thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            if len(thinking_matches) == 1:
                score = score + 1.0
            if len(answer_matches) == 1:
                score = score + 1.0
            scores.append(score)
        return scores

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        import re

        answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
        responses = [
            re.findall(answer_pattern, completion, re.DOTALL)
            for completion in completions
        ]
        q = prompts[0]
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:{completions[0]}",
        )
        return [
            2.0 if len(r) == 1 and a == r[0].replace("\n", "") else 0.0
            for r, a in zip(responses, answer)
        ]

    return correctness_reward_func, formatting_reward_func


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is the first example prompt in the dataset
    """)
    return


@app.cell
def _(train_dataset_1):
    train_dataset_1[0]["prompt"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations!
    """)
    return


@app.cell
def _(GRPOConfig):
    training_args = GRPOConfig(
        learning_rate=5e-06,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=1024,
        max_completion_length=1024,
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
        max_steps=60,
        save_steps=60,
        max_grad_norm=0.1,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
    )
    return (training_args,)


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
    formatting_reward_func,
    model_1,
    tokenizer,
    train_dataset_1,
    training_args,
):
    trainer = GRPOTrainer(
        model=model_1,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[formatting_reward_func, correctness_reward_func],
        train_dataset=train_dataset_1,
    )
    trainer.train()  # Pass the processor to handle multimodal inputs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Let's run the model! You can modify the instruction and input—just leave the output blank.

    We'll use the best hyperparameters for inference on Gemma: `top_p=0.95`, `top_k=64`, and `temperature=1.0`.
    """)
    return


@app.cell
def _(
    FastVisionModel,
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    dataset_2,
    model_1,
    tokenizer,
):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image = dataset_2[100]["decoded_image"]
    instruction = f"{dataset_2[100]['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image, input_text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    result = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving, loading finetuned models
    To save the final model as LoRA adapters, use Hugging Face’s `push_to_hub` for online saving, or `save_pretrained` for local storage.

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
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(
    FastVisionModel,
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    TextStreamer,
    dataset_2,
    model_1,
    tokenizer,
):
    if False:
        from unsloth import FastVisionModel as _FastVisionModel

        _model, _processor = _FastVisionModel.from_pretrained(
            model_name="gemma_3_lora", load_in_4bit=True  # YOUR MODEL YOU USED FOR TRAINING
        )
        _FastVisionModel.for_inference(_model)
    FastVisionModel.for_inference(model_1)
    sample = dataset_2[1]  # Enable for inference!
    image_1 = sample["decoded_image"].convert("RGB")
    messages_1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{sample['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}",
                },
                {"type": "image"},
            ],
        }
    ]
    input_text_1 = tokenizer.apply_chat_template(messages_1, add_generation_prompt=True)
    inputs_1 = tokenizer(
        image_1, input_text_1, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    text_streamer_1 = TextStreamer(tokenizer, skip_prompt=True)
    result_1 = model_1.generate(
        **inputs_1,
        streamer=text_streamer_1,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Select ONLY 1 to save! (Both not needed!)
    if False:
        # Save locally to 16bit
        model_1.save_pretrained_merged("unsloth_finetune", tokenizer)
    if False:
        # To export and save to your Hugging Face account
        model_1.push_to_hub_merged(
            "YOUR_USERNAME/unsloth_finetune", tokenizer, token="YOUR_HF_TOKEN"
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

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
