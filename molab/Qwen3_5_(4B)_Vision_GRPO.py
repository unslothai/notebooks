# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "apache-tvm-ffi==0.1.9",
#     "bitsandbytes>=0.43.0",
#     "causal_conv1d==1.6.0",
#     "flash-linear-attention",
#     "marimo",
#     "tilelang==0.1.8",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch==2.8.0",
#     "torchao>=0.16.0",
#     "torchcodec==0.7.0",
#     "torchvision",
#     "transformers==5.2.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo",
#     "uv",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We're also introducing how you can do `GSPO` inside of Unsloth as well!

    The goal of this notebook is to make a vision language model solve maths problems via reinforcement learning given an image input like below:

    <img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/our_new_3_datasets.png" alt="Alt text" height="256">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    from unsloth import FastVisionModel
    import torch

    max_seq_length = 16384  # Must be this long for VLMs
    lora_rank = 16  # Larger rank = smarter, but slower

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=max_seq_length,  # Must be this long for VLMs
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=False,  # Enable vllm fast inference
    )
    return FastVisionModel, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In Unsloth, we share vLLM's weights directly, reducing VRAM usage by > 50%. vLLM also does not yet support LoRA on the vision layers, so we can only add them on the language layers. Vision GRPO still works though!
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
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Prep
    <a name="Data"></a>

    `AI4Math/MathVista` is a dataset that involves using images to solve logic and math problems.

    For this notebook, we will only use math problems with numeric answers for simplicity.
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
    We filter the dataset to keep only float or numeric answers:
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

    dataset_1 = dataset.filter(is_numeric_answer)
    return (dataset_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also resize the images to be 512 by 512 pixels to make the images manageable in context length. We also convert them to RGB so they are compatible for training!
    """)
    return


@app.cell
def _(dataset_1):
    # Resize to (512, 512)
    def resize_images(example):
        image = example["decoded_image"]
        image = image.resize((512, 512))
        example["decoded_image"] = image
        return example

    dataset_2 = dataset_1.map(resize_images)

    # Then convert to RGB
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
    We then create the conversational template that is needed to collate the dataset for RL:
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
        text_content = f"{example['question']}. Also first provide your reasoning or working out on how you would go about solving the question between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"  # Define placeholder constants if they are not defined globally
        prompt = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text_content}],
            }
        ]  # The user's text prompt
        return {
            "prompt": prompt,
            "image": example["decoded_image"],
            "answer": example["answer"],
        }

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
    Now let's apply the chat template across the entire dataset:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reward functions

    We now define some basic formatting rewards functions to see if reasoning starts and ends, and also another to see if the answers were written correctly.

    We also try to fix the `addCriterion` issue as described in our [blog post](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks)
    """)
    return


@app.cell
def _(REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START):
    # Reward functions
    import re

    def formatting_reward_func(completions, **kwargs):
        import re

        thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
        answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
        scores = []
        for completion in completions:
            if isinstance(completion, list):
                completion = completion[0]["content"] if completion else ""
            score = 0
            thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            if len(thinking_matches) == 1:
                score = score + 1.0
            if len(answer_matches) == 1:
                score = score + 1.0
            if len(completion) != 0:
                removal = completion.replace("addCriterion", "").replace("\n", "")
                if (len(completion) - len(removal)) / len(
                    completion
                ) >= 0.5:  # Fix up addCriterion issues
                    score = (
                        score - 2.0
                    )
            scores.append(score)  # Penalize on excessive addCriterion and newlines
        return scores

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
        completions = [
            (c[0]["content"] if c else "") if isinstance(c, list) else c
            for c in completions
        ]
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
def _(train_dataset):
    train_dataset[0]["prompt"]
    return


@app.cell
def _(train_dataset):
    train_dataset[100]["prompt"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    Now let's try the model on the hundredth sample of the train dataset without training.
    """)
    return


@app.cell
def _(model_1, tokenizer, train_dataset):
    image = train_dataset[100]["image"]
    prompt = train_dataset[100]["prompt"]
    inputs = tokenizer(image, prompt, add_special_tokens=False, return_tensors="pt").to(
        "cuda"
    )
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.0,
        min_p=0.1,
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up the `GRPO` Trainer and all configurations! Note we actually enable `GSPO` as well!
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
        log_completions=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=60,
        save_steps=60,
        max_grad_norm=0.1,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
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

    During inference, you might encounter `addCriterion` or some weird gibberish outputs. Please read our [blog post](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks) on why this occurs. It seems to be an inherent thing inside of the model, and we can ignore this.
    """)
    return


@app.cell
def _(
    GRPOTrainer,
    correctness_reward_func,
    formatting_reward_func,
    model_1,
    tokenizer,
    train_dataset,
    training_args,
):
    trainer = GRPOTrainer(
        model=model_1,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[formatting_reward_func, correctness_reward_func],
        train_dataset=train_dataset,
    )
    trainer.train()  # Pass the processor to handle multimodal inputs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's run the model! You can modify the instruction and input.
    """)
    return


@app.cell
def _(TextStreamer, model_1, tokenizer, train_dataset):
    image_1 = train_dataset[165]["image"]
    prompt_1 = train_dataset[165]["prompt"]
    inputs_1 = tokenizer(
        image_1, prompt_1, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    text_streamer_1 = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_1,
        streamer=text_streamer_1,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.0,
        min_p=0.1,
    )
    return


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
    model_1.save_pretrained("qwen_lora")
    tokenizer.save_pretrained("qwen_lora")
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
    <a name="Save"></a>
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "qwen_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_merged(
            "qwen_finetune_4bit", tokenizer, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/qwen_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained("qwen_lora")
        tokenizer.save_pretrained("qwen_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/qwen_lora", token="YOUR_HF_TOKEN")
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
def _(model_1, tokenizer):
    # Save to 8bit Q8_0
    if False:
        # Remember to go to https://huggingface.co/settings/tokens for a token!
        # And change hf to your username!
        model_1.save_pretrained_gguf("qwen_finetune", tokenizer)
    if False:
        # Save to 16bit GGUF
        model_1.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )
    if False:
        model_1.save_pretrained_gguf(
            "qwen_finetune", tokenizer, quantization_method="f16"
        )
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )
    if False:
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.save_pretrained_gguf(
            "qwen_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )  # Change hf to your username!
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune",
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
            token="YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Special Credits to [GAD-Cell](https://github.com/GAD-cell) for helping Unsloth create this notebook and bringing VLM GRPO into Unsloth!
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

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
    """)
    return


if __name__ == "__main__":
    app.run()
