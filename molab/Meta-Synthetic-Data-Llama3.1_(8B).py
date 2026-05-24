# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "cuda-tile==1.2.0",
#     "marimo",
#     "synthetic-data-kit==0.0.3",
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


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run this, press the **Run** button beside each cell on your A100 molab Pro instance!

    <a href="https://github.com/meta-llama/synthetic-data-kit"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/meta%20round%20logo.png" width="137"></a>
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
    ### Synthetic-data-kit
    """)
    return


@app.cell
def _(subprocess):
    # Load and run the model using vllm
    # we prepend "nohup" and postpend "&" to make the molab cell run in background
    #! nohup python -m vllm.entrypoints.openai.api_server --model unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit --trust-remote-code --dtype half --quantization bitsandbytes --max-model-len 10000 --tensor-parallel-size 1 --gpu-memory-utilization 0.7 --enable-chunked-prefill --port 8000 > vllm.log &
    subprocess.call(
        "nohup python -m vllm.entrypoints.openai.api_server --model unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit --trust-remote-code --dtype half --quantization bitsandbytes --max-model-len 10000 --tensor-parallel-size 1 --gpu-memory-utilization 0.7 --enable-chunked-prefill --port 8000 > vllm.log &",
        shell=True,
    )
    return


@app.cell
def _(subprocess):
    # tail vllm logs. Check server has been started correctly
    #! while ! grep -q "Application startup complete" vllm.log; do tail -n 1 vllm.log; sleep 5; done
    subprocess.call(
        [
            "while",
            "!",
            "grep",
            "-q",
            "Application startup complete",
            "vllm.log;",
            "do",
            "tail",
            "-n",
            "1",
            "vllm.log;",
            "sleep",
            "5;",
            "done",
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optional: Function to check if vllm server is running. Change False to True and run cell
    """)
    return


@app.cell
def _(requests):
    if False:

        def _is_vllm_server_running(api_base_url=None):
            """Simply check if vllm server is running and reachable."""
            print(api_base_url)
            try:
                response = requests.get(f"{api_base_url}/models", timeout=2)
                return response.status_code == 200
            except:
                return False

        _is_running = _is_vllm_server_running("http://localhost:8000/v1")
        if _is_running:
            print(f"vllm server is running.")
        else:
            print(f"vllm server is not available.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create data directories
    """)
    return


@app.cell
def _(subprocess):
    #! mkdir -p data/{pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final}
    subprocess.call(
        [
            "mkdir",
            "-p",
            "data/{pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final}",
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ingest source file

    Ingest source file "https://ai.meta.com/blog/llama-4-multimodal-intelligence/" . Can also use pdf, docx, ppt and youtube video
    """)
    return


@app.cell
def _(ctx):
    from synthetic_data_kit.core.ingest import process_file
    import os

    # Set variables directly
    doc_source = "https://ai.meta.com/blog/llama-4-multimodal-intelligence/"
    output_dir = "data/output"
    name = None  # Let the process determine the filename automatically
    config = (  # Use ctx if available, otherwise None
        ctx.config if "ctx" in locals() else None
    )  # Use ctx if available, otherwise None

    try:
        # Call process_file directly
        output_path = process_file(doc_source, output_dir, name, config)
        print(f"Text successfully extracted to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
    return os, process_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate QA pairs

    Generate QA pairs with the help of vllm and Llama-3.1-8B-Instruct-unsloth-bnb-4bit.
    set num_pairs to the number of required pairs
    """)
    return


@app.cell
def _(ctx, process_file):
    import requests
    import json

    input_file = "data/output/ai_meta_com.txt"
    output_dir_1 = "data/generated"
    config_path = ctx.config_path if "ctx" in locals() else None  # Use ctx if available
    # Set parameters
    api_base = "http://localhost:8000/v1"  # Default vllm API endpoint
    model = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
    content_type = "qa"  # Use ctx if available
    num_pairs = 10  # Default vllm API endpoint
    verbose = False
    with open(input_file, "r") as f:
        text_content = f.read()
    print("\nGenerating QA pairs...")
    try:
        # Read the content of the input file
        output_path_1 = process_file(
            input_file,
            output_dir_1,
            config_path,
            api_base,
            model,
            content_type,
            num_pairs,
            verbose,
        )
        if output_path_1:
            print(f"Content saved to {output_path_1}")
            try:
                with open(output_path_1, "r") as f:
                    output_content = f.read()
                print(
                    "\nGenerated content (first 500 chars):"
                )  # Call process_file directly with all parameters
                print(
                    output_content[:500] + "..."
                    if len(output_content) > 500
                    else output_content
                )
            except Exception as e:
                print(f"Could not read generated file: {e}")
        else:
            print("No output was generated")
    except Exception as e:
        print(f"Error: {e}")  # Additionally, print the content of the generated file
    return api_base, json, model, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Curate Data Pairs
    """)
    return


@app.cell
def _(api_base, ctx, model, os):
    from synthetic_data_kit.core.curate import curate_qa_pairs

    input_file_1 = "data/generated/ai_meta_com_qa_pairs.json"
    # Set all parameters directly
    cleaned_dir = "data/cleaned"
    base_name = os.path.splitext(os.path.basename(input_file_1))[0]
    output = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")
    threshold = None  # Use default threshold
    config_path_1 = ctx.config_path if "ctx" in locals() else None
    verbose_1 = False  # Use default threshold
    print("\nCurating generated pairs...")  # Use ctx if available
    try:
        result_path = curate_qa_pairs(
            input_file_1, output, threshold, api_base, model, config_path_1, verbose_1
        )
        print(f"Cleaned content saved to {result_path}")
        try:
            with open(result_path, "r") as f_1:
                output_content_1 = f_1.read()  # Call curate_qa_pairs directly
            print("\nGenerated content (first 500 chars):")
            print(
                output_content_1[:500] + "..."
                if len(output_content_1) > 500
                else output_content_1
            )
        except Exception as e:
            print(f"Could not read cleaned file: {e}")
    except Exception as e:
        print(f"Error: {e}")  # Display the content of the cleaned file
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save to chatML format
    """)
    return


@app.cell
def _(ctx, json, os):
    from synthetic_data_kit.core.save_as import convert_format

    input_file_2 = "data/cleaned/ai_meta_com_qa_pairs_cleaned.json"
    format_type = "ft"  # OpenAI fine-tuning format
    storage_format = "json"  # Default storage format
    # Set all parameters directly
    final_dir = "data/final"
    base_name_1 = os.path.splitext(os.path.basename(input_file_2))[
        0
    ]  # OpenAI fine-tuning format
    if storage_format == "hf":  # Default storage format
        output_path_2 = os.path.join(final_dir, f"{base_name_1}_{format_type}_hf")
    # Set up output path
    elif format_type == "jsonl":
        # os.makedirs(final_dir, exist_ok = True)
        output_path_2 = os.path.join(final_dir, f"{base_name_1}.jsonl")
    else:
        # Determine output file path
        output_path_2 = os.path.join(final_dir, f"{base_name_1}_{format_type}.json")
    config_1 = ctx.config if "ctx" in locals() else None
    try:
        result_path_1 = convert_format(
            input_file_2,
            output_path_2,
            format_type,
            config_1,
            storage_format=storage_format,  # Default storage format
        )
        print(f"Converted to {format_type} format and saved to {result_path_1}")
        try:
            if os.path.isfile(result_path_1):
                with open(result_path_1, "r") as f_2:
                    # Load config if available
                    output_content_2 = f_2.read()
                print("\nConverted content (first 500 chars):")
                print(
                    output_content_2[:500] + "..."
                    if len(output_content_2) > 500
                    else output_content_2
                )
            else:  # Call convert_format directly
                print(f"\nSaved as HF dataset directory at {result_path_1}")
                if os.path.exists(os.path.join(result_path_1, "dataset_info.json")):
                    with open(
                        os.path.join(result_path_1, "dataset_info.json"), "r"
                    ) as f_2:
                        info = json.load(f_2)
                    print(f"Dataset info: {info}")
        except Exception as e:
            print(f"Could not read converted file: {e}")
    except Exception as e:
        print(
            f"Error: {e}"
        )
    return


@app.cell
def _(subprocess):
    # kill vllm server. Takes around 5 seconds.
    print("Attempting to terminate the vllm server")
    #! pkill -f "vllm.entrypoints.openai.api_server"
    subprocess.call(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
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
    fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/llama-2-13b-bnb-4bit",
        "unsloth/codellama-34b-bnb-4bit",
        "unsloth/tinyllama-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
        "unsloth/gemma-2b-bnb-4bit",
    ]
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    model_1, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
        dtype=dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False.
    )
    return (
        FastLanguageModel,
        dtype,
        load_in_4bit,
        max_seq_length,
        model_1,
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
def _(FastLanguageModel, model_1):
    model_2 = FastLanguageModel.get_peft_model(
        model_1,
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
    return (model_2,)


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

    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    #     mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    #     map_eos_token = True, # Maps <|im_end|> to </s> instead
    # )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    from datasets import load_dataset, Dataset

    dataset = Dataset.from_json(
        "/content/data/final/ai_meta_com_qa_pairs_cleaned_ft.json"
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    return dataset, get_chat_template


@app.cell
def _(dataset):
    dataset[1]["messages"]
    return


@app.cell
def _(dataset):
    print(dataset[1]["text"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you're looking to make your own chat template, that also is possible! You

    ---

    must use the Jinja templating regime. We provide our own stripped down version of the `Unsloth template` which we find to be more efficient, and leverages ChatML, Zephyr and Alpaca styles.

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
            chat_template=(unsloth_template, unsloth_eos_token),  # You must provide a template and EOS token
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
    [link text](https://)<a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!!
    """)
    return


@app.cell
def _(dataset, max_seq_length, model_2, tokenizer):
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model_2,
        tokenizer=tokenizer,
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
    *italicized text*<a name="Inference"></a>
    ### Inference
    Let's run the model! Since we're using `ChatML`, use `apply_chat_template` with `add_generation_prompt` set to `True` for inference.
    """)
    return


@app.cell
def _(FastLanguageModel, get_chat_template, model_2, tokenizer):
    tokenizer_1 = get_chat_template(
        tokenizer,
        chat_template="chatml",  # You must provide a template and EOS token
        mapping={  # ShareGPT style
            "role": "from",
            "content": "value",
            "user": "human",
            "assistant": "gpt",
        },
        map_eos_token=True,  # Maps <|im_end|> to </s> instead
    )
    FastLanguageModel.for_inference(model_2)
    messages = [
        {"from": "human", "value": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"}
    ]
    inputs = tokenizer_1.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model_2.generate(
        input_ids=inputs, max_new_tokens=64, use_cache=True
    )  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    tokenizer_1.batch_decode(
        outputs
    )
    return (tokenizer_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use a TextStreamer for continuous inference - so you can see the generation token by token, instead of waiting the whole time!
    """)
    return


@app.cell
def _(FastLanguageModel, model_2, tokenizer_1):
    FastLanguageModel.for_inference(model_2)  # Enable native 2x faster inference
    messages_1 = [
        {"from": "human", "value": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"}
    ]
    inputs_1 = tokenizer_1.apply_chat_template(
        messages_1, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer_1)
    _ = model_2.generate(
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
def _(model_2, tokenizer_1):
    model_2.save_pretrained("meta_synthetic_data_lora")
    tokenizer_1.save_pretrained("meta_synthetic_data_lora")
    return


@app.cell
def _(TextStreamer, dtype, load_in_4bit, max_seq_length, model_2, tokenizer_1):
    if False:
        from unsloth import FastLanguageModel as _FastLanguageModel

        _model, _tokenizer = _FastLanguageModel.from_pretrained(
            model_name="meta_synthetic_data_lora",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
            max_seq_length=max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
            dtype=dtype,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False.
        )
        _FastLanguageModel.for_inference(_model)
    messages_2 = [{"from": "human", "value": "What is a famous tall tower in Paris?"}]
    inputs_2 = tokenizer_1.apply_chat_template(
        messages_2, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    text_streamer_1 = TextStreamer(tokenizer_1)
    _ = model_2.generate(
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
            "meta_synthetic_data_lora", load_in_4bit=load_in_4bit
        )
        _tokenizer = _AutoTokenizer.from_pretrained("meta_synthetic_data_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_2, tokenizer_1):
    # Merge to 16bit
    if False:
        model_2.save_pretrained_merged(
            "meta_synthetic_data_finetune_16bit",
            tokenizer_1,
            save_method="merged_16bit",
        )
    if False:
        # Merge to 4bit
        model_2.push_to_hub_merged(
            "HF_USERNAME/meta_synthetic_data_finetune_16bit",
            tokenizer_1,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_2.save_pretrained_merged(
            "meta_synthetic_data_finetune_4bit", tokenizer_1, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_2.push_to_hub_merged(
            "HF_USERNAME/meta_synthetic_data_finetune_4bit",
            tokenizer_1,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_2.save_pretrained("meta_synthetic_data_lora")
        tokenizer_1.save_pretrained("meta_synthetic_data_lora")
    if False:
        model_2.push_to_hub(
            "HF_USERNAME/meta_synthetic_data_lora", token="YOUR_HF_TOKEN"
        )
        tokenizer_1.push_to_hub(
            "HF_USERNAME/meta_synthetic_data_lora", token="YOUR_HF_TOKEN"
        )
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
def _(model_2, tokenizer_1):
    # Save to 8bit Q8_0
    if False:
        model_2.save_pretrained_gguf("meta_synthetic_data_finetune", tokenizer_1)
    if False:
        # Save to 16bit GGUF
        model_2.push_to_hub_gguf(
            "HF_USERNAME/meta_synthetic_data_finetune",
            tokenizer_1,
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_2.save_pretrained_gguf(
            "meta_synthetic_data_finetune", tokenizer_1, quantization_method="f16"
        )
    # Save to q4_k_m GGUF
    if False:
        model_2.push_to_hub_gguf(
            "HF_USERNAME/meta_synthetic_data_finetune",
            tokenizer_1,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_2.save_pretrained_gguf(
            "meta_synthetic_data_finetune", tokenizer_1, quantization_method="q4_k_m"
        )
    if False:
        model_2.push_to_hub_gguf(
            "HF_USERNAME/meta_synthetic_data_finetune",
            tokenizer_1,
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
