# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "decord",
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
    To run this, press the **Run** button beside each cell on your A100 molab Pro instance!
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
    from unsloth import FastVisionModel  # FastLanguageModel for LLMs
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",  # Qwen 3 vision support
        "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
        "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit",
        "unsloth/Qwen3-VL-32B-Thinking-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model_path = "unsloth/ERNIE-4.5-VL-28B-A3B-PT"
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        auto_model=AutoModelForCausalLM,
        load_in_4bit=False,  # Unsupported for this specific model variant
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing=False,
        attn_implementation="eager",
    )
    return AutoProcessor, FastVisionModel, model, model_path, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now load the processor
    """)
    return


@app.cell
def _(AutoProcessor, model, model_path):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.eval()
    model.add_image_preprocess(processor)
    return (processor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

    **[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!
    """)
    return


@app.cell
def _(FastVisionModel, model):
    model_1 = FastVisionModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
        ],
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    We'll be using a sampled dataset of handwritten maths formulas. The goal is to convert these images into a computer readable form - ie in LaTeX form, so we can render it. This can be very useful for complex formulas.

    You can access the dataset [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR). The full dataset is [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take an overview look at the dataset. We shall see what the 3rd image is, and what caption it had.
    """)
    return


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(dataset):
    dataset[2]["image"]
    return


@app.cell
def _(dataset):
    dataset[2]["text"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also render the LaTeX in the browser directly!
    """)
    return


@app.cell
def _(dataset):
    from IPython.display import display, Math, Latex

    latex = dataset[2]["text"]
    display(Math(latex))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To format the dataset, all vision finetuning tasks should be formatted as follows:

    ```python
    [
    { "role": "user",
      "content": [{"type": "text",  "text": Q}, {"type": "image", "image": image} ]
    },
    { "role": "assistant",
      "content": [{"type": "text",  "text": A} ]
    },
    ]
    ```
    """)
    return


@app.cell
def _():
    instruction = "Write the LaTeX representation for this image."

    def convert_to_conversation(sample):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
                "reasoning_content": "\n",  # If you leave this as "\n", you train the model to output empty thoughts.
            },
        ]
        return {"messages": conversation}

    return (convert_to_conversation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's convert the dataset into the "correct" format for finetuning:
    """)
    return


@app.cell
def _(convert_to_conversation, dataset):
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    return (converted_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We look at how the conversations are structured for the first example:
    """)
    return


@app.cell
def _(converted_dataset):
    converted_dataset[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's first see before we do any finetuning what the model outputs for the first example!
    """)
    return


@app.cell
def _(FastVisionModel, dataset, model_1, processor, tokenizer):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image = dataset[2]["image"]
    instruction_1 = "Write the LaTeX representation for this image."
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_1}],
        }
    ]
    text_prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(
        text=[text_prompt], images=[image], videos=[], padding=True, return_tensors="pt"
    )
    device = next(model_1.parameters()).device
    inputs = inputs.to(device)
    from transformers import TextStreamer

    text_streamer = TextStreamer(
        tokenizer, skip_prompt=True
    )
    _ = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=False,
        temperature=1.5,
        min_p=0.1,
    )
    return (TextStreamer,)


@app.cell
def _(torch):
    # @title Setup Collator & Trainer
    from trl import SFTTrainer, SFTConfig
    import torch.nn as nn
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Union

    @dataclass
    class ErnieVisionDataCollator:
        processor: Any
        tokenizer: Any
        ignore_index: int = -100
        max_seq_length: int = 2048
        train_on_responses_only: bool = False
        _img_patch_id: int = field(init=False, default=-1)

        def __post_init__(self):
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
            patch_token = "<|IMAGE_PLACEHOLDER|>"
            converted_id = self.tokenizer.convert_tokens_to_ids(patch_token)
            self._img_patch_id = converted_id if converted_id is not None else -1

        def _extract_visuals(self, msgs: List[Dict]) -> tuple:
            image_inputs, video_inputs = ([], [])
            needs_extraction = False
            for msg in msgs:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if "image" in part:
                                image_inputs.append(part["image"])
                            elif part.get("type") in ["image_url", "video_url"]:
                                needs_extraction = True
            if needs_extraction and (not image_inputs):
                try:
                    return self.processor.process_vision_info(msgs)
                except Exception:
                    return ([], [])
            return (image_inputs, video_inputs)

        def _mask_prompt(
            self,
            msgs: List[Dict],
            image_inputs: List,
            labels: torch.Tensor,
            full_input_ids: torch.Tensor,
        ) -> torch.Tensor:
            last_asst_idx = -1
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "assistant":
                    last_asst_idx = i
                    break
            if last_asst_idx == -1:
                return labels
            prompt_msgs = msgs[:last_asst_idx]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_inputs = self.processor(
                text=[prompt_text], images=image_inputs, return_tensors="pt"
            )
            prompt_ids = prompt_inputs["input_ids"][0]
            len_full = full_input_ids.size(0)
            len_prompt = prompt_ids.size(0)
            limit = min(len_full, len_prompt)
            matches = full_input_ids[:limit] == prompt_ids[:limit]
            mismatches = (~matches).nonzero(as_tuple=False)
            if len(mismatches) > 0:
                mask_len = mismatches[0].item()
            else:
                mask_len = limit
            labels[:mask_len] = self.ignore_index
            return labels

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            batch = {
                k: []
                for k in [
                    "input_ids",
                    "labels",
                    "token_type_ids",
                    "position_ids",
                    "images",
                    "grid_thw",
                    "image_type_ids",
                ]
            }
            for example in features:
                msgs = example.get("messages", example.get("conversations", []))
                image_inputs, video_inputs = self._extract_visuals(msgs)
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"][0]
                tt = inputs["token_type_ids"][0]
                pos = inputs["position_ids"][0]
                if input_ids[-1] != self.tokenizer.eos_token_id:
                    input_ids = torch.cat(
                        [input_ids, torch.tensor([self.tokenizer.eos_token_id])]
                    )
                    tt = torch.cat([tt, torch.tensor([0], dtype=tt.dtype)])
                    pos = torch.cat([pos, (pos[-1] + 1).unsqueeze(0)])
                labels = input_ids.clone()
                if self._img_patch_id != -1:
                    labels[labels == self._img_patch_id] = self.ignore_index
                if self.train_on_responses_only:
                    labels = self._mask_prompt(msgs, image_inputs, labels, input_ids)
                batch["input_ids"].append(input_ids)
                batch["labels"].append(labels)
                batch["token_type_ids"].append(torch.cat([tt, torch.tensor([0])]))
                batch["position_ids"].append(pos)
                if inputs.get("images") is not None:
                    batch["images"].append(inputs["images"])
                if inputs.get("grid_thw") is not None:
                    batch["grid_thw"].append(inputs["grid_thw"])
                if inputs.get("image_type_ids") is not None:
                    batch["image_type_ids"].append(inputs["image_type_ids"])
            padded_input = torch.nn.utils.rnn.pad_sequence(
                batch["input_ids"],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            padded_label = torch.nn.utils.rnn.pad_sequence(
                batch["labels"], batch_first=True, padding_value=self.ignore_index
            )
            padded_tt = torch.nn.utils.rnn.pad_sequence(
                batch["token_type_ids"], batch_first=True, padding_value=0
            )
            max_len = padded_input.shape[1]
            padded_pos = torch.zeros(
                (len(batch["position_ids"]), max_len, 3), dtype=torch.long
            )
            for i, p in enumerate(batch["position_ids"]):
                l = min(p.shape[0], max_len)
                padded_pos[i, :l, :] = p[:l]
            if padded_input.shape[1] > self.max_seq_length:
                padded_input = padded_input[:, : self.max_seq_length]
                padded_label = padded_label[:, : self.max_seq_length]
                padded_pos = padded_pos[:, : self.max_seq_length, :]
                padded_tt = padded_tt[:, : self.max_seq_length + 1]
            final_batch = {
                "input_ids": padded_input,
                "labels": padded_label,
                "attention_mask": padded_input.ne(self.tokenizer.pad_token_id).long(),
                "token_type_ids": padded_tt,
                "position_ids": padded_pos,
            }
            if batch["images"]:
                final_batch["images"] = torch.cat(batch["images"], dim=0)
            if batch["grid_thw"]:
                final_batch["grid_thw"] = torch.cat(batch["grid_thw"], dim=0)
            if batch["image_type_ids"]:
                final_batch["image_type_ids"] = torch.cat(
                    batch["image_type_ids"], dim=0
                )
            return final_batch

    class ErnieSFTTrainer(SFTTrainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs.get("labels")
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                loss = loss_fct(shift_logits, shift_labels)
                if hasattr(outputs, "router_loss") and outputs.router_loss is not None:
                    aux_loss = outputs.router_loss.to(loss.device)
                    loss = loss + aux_loss
            if return_outputs:
                return (loss, outputs)
            return loss

    return ErnieSFTTrainer, ErnieVisionDataCollator, SFTConfig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 30 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!!

    We use our new `ErnieVisionDataCollator` which will help in our vision finetuning setup.
    """)
    return


@app.cell
def _(
    ErnieSFTTrainer,
    ErnieVisionDataCollator,
    FastVisionModel,
    SFTConfig,
    converted_dataset,
    model_1,
    processor,
    tokenizer,
    torch,
):
    FastVisionModel.for_training(model_1)
    custom_collator = ErnieVisionDataCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=2048,
        train_on_responses_only=True,
    )
    trainer = ErnieSFTTrainer(
        model=model_1,
        tokenizer=processor.tokenizer,
        data_collator=custom_collator,
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            max_steps=30,
            learning_rate=0.0002,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            gradient_checkpointing=False,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )  # Enable for training!
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
    Let's run the model! You can change the instruction and input - leave the output blank!

    We use `min_p = 0.1` and `temperature = 1.5`. Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.
    """)
    return


@app.cell
def _(FastVisionModel, TextStreamer, dataset, model_1, processor, tokenizer):
    FastVisionModel.for_inference(model_1)  # Enable for inference!
    image_1 = dataset[2]["image"]
    instruction_2 = "Write the LaTeX representation for this image."
    messages_1 = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction_2}],
        }
    ]
    text_prompt_1 = processor.tokenizer.apply_chat_template(
        messages_1, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs_1 = processor(
        text=[text_prompt_1],
        images=[image_1],
        videos=[],
        padding=True,
        return_tensors="pt",
    )
    device_1 = next(model_1.parameters()).device
    inputs_1 = inputs_1.to(device_1)
    text_streamer_1 = TextStreamer(tokenizer, skip_prompt=True)
    # Move inputs to GPU
    _ = model_1.generate(
        **inputs_1,
        streamer=text_streamer_1,
        max_new_tokens=128,
        use_cache=False,
        temperature=1.5,
        min_p=0.1,
    )  # Placeholder required for the template
    return (inputs_1,)


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
    model_1.save_pretrained("ernie_lora")
    tokenizer.save_pretrained("ernie_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:
    """)
    return


@app.cell
def _(TextStreamer, inputs_1, model_1, tokenizer):
    if False:
        from unsloth import FastVisionModel as _FastVisionModel

        _model, _tokenizer = _FastVisionModel.from_pretrained(
            model_name="ernie_lora", load_in_4bit=False  # YOUR MODEL YOU USED FOR TRAINING
        )
        _FastVisionModel.for_inference(_model)
    text_streamer_2 = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(
        **inputs_1,
        streamer=text_streamer_2,
        max_new_tokens=128,
        use_cache=False,
        temperature=1.5,
        min_p=0.1,
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
