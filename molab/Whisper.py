# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "evaluate",
#     "hf_transfer",
#     "huggingface_hub>=0.34.0",
#     "jiwer",
#     "librosa",
#     "marimo",
#     "peft",
#     "protobuf",
#     "sentencepiece",
#     "soundfile",
#     "torchao>=0.16.0",
#     "torchcodec",
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


@app.cell
def _():
    from unsloth import FastModel
    from transformers import WhisperForConditionalGeneration
    import torch

    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        # Qwen3 new models
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        # Other very popular models!
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.3-70B",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Phi-4",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/whisper-large-v3",
        dtype=None,  # Leave as None for auto detection
        load_in_4bit=False,  # Set to True to do 4bit quantization which reduces memory
        auto_model=WhisperForConditionalGeneration,
        whisper_language="English",
        whisper_task="transcribe",
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastModel, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
    """)
    return


@app.cell
def _(FastModel, model):
    model_1 = FastModel.get_peft_model(
        model,
        r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "v_proj"],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        task_type=None,  # ** MUST set this for Whisper **
    )
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep

    We will use the `MrDragonFox/Elise`, which is designed for training TTS models. Ensure that your dataset follows the required format: **text, audio**. You can modify this section to accommodate your own dataset, but maintaining the correct structure is essential for optimal training.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    import numpy as np
    import tqdm

    model_1.generation_config.language = "<|en|>"
    # Set this to the language you want to train on
    model_1.generation_config.task = "transcribe"
    model_1.config.suppress_tokens = []
    model_1.generation_config.forced_decoder_ids = None

    def formatting_prompts_func(example):
        audio_arrays = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        features = tokenizer.feature_extractor(
            audio_arrays, sampling_rate=sampling_rate
        )
        tokenized_text = tokenizer.tokenizer(example["text"])
        return {
            "input_features": features.input_features[0],
            "labels": tokenized_text.input_ids,
        }

    from datasets import load_dataset, Audio

    dataset = load_dataset("MrDragonFox/Elise", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.train_test_split(test_size=0.06)
    train_dataset = [
        formatting_prompts_func(example)
        for example in tqdm.tqdm(dataset["train"], desc="Train split")
    ]
    test_dataset = [
        formatting_prompts_func(example)
        for example in tqdm.tqdm(dataset["test"], desc="Test split")
    ]
    return np, test_dataset, train_dataset


@app.cell
def _(np, tokenizer, torch):
    # @title Create compute_metrics and datacollator
    import evaluate
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import pdb

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions[0]
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(
            predictions=pred_str, references=label_str
        )  # replace -100 with the pad_token_id
        return {"wer": wer}

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    return DataCollatorSpeechSeq2SeqWithPadding, compute_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's use Hugging Face `Seq2SeqTrainer`! More docs here: [Transformers docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
    """)
    return


@app.cell
def _(
    DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics,
    model_1,
    test_dataset,
    tokenizer,
    train_dataset,
):
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    from unsloth import is_bf16_supported

    trainer = Seq2SeqTrainer(
        model=model_1,
        train_dataset=train_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer),
        eval_dataset=test_dataset,
        tokenizer=tokenizer.feature_extractor,
        compute_metrics=compute_metrics,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=0.0001,
            logging_steps=1,
            optim="adamw_8bit",
            fp16=not is_bf16_supported(),  # Use fp16 if bf16 is not supported
            bf16=is_bf16_supported(),  # Use bf16 if supported
            weight_decay=0.001,
            remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
            lr_scheduler_type="linear",
            label_names=["labels"],
            eval_steps=5,
            eval_strategy="steps",
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
    Let's run the model! Because we finetuned Whisper for speech recognition, we need to have a audio file.

    For example we use the Harvard Sentences audio dataset https://en.wikipedia.org/wiki/Harvard_sentences
    """)
    return


@app.cell
def _():
    import subprocess
    import subprocess

    subprocess.call(
        [
            "wget",
            "https://upload.wikimedia.org/wikipedia/commons/5/5b/Speech_12dB_s16.flac",
        ]
    )
    from IPython.display import Audio as _molab_Audio, display

    display(_molab_Audio("Speech_12dB_s16.flac", rate=24000))
    return


@app.cell
def _(FastModel, model_1, tokenizer, torch):
    from transformers import pipeline

    FastModel.for_inference(model_1)
    model_1.eval()
    whisper = pipeline(
        "automatic-speech-recognition",
        model=model_1,
        tokenizer=tokenizer.tokenizer,
        feature_extractor=tokenizer.feature_extractor,
        processor=tokenizer,
        return_language=True,
        torch_dtype=torch.float16,  # Remove the device parameter
    )
    # Create pipeline without specifying the device
    audio_file = "Speech_12dB_s16.flac"
    transcribed_text = whisper(audio_file)
    # Example usage
    print(transcribed_text["text"])  # Remove the device parameter
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
    model_1.save_pretrained("whisper_lora")
    tokenizer.save_pretrained("whisper_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving to float16

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "whisper_finetune_16bit", tokenizer, save_method=None
        )
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/whisper_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_merged(
            "whisper_finetune_4bit", tokenizer, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/whisper_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained("whisper_lora")
        tokenizer.save_pretrained("whisper_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/whisper_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/whisper_lora", token="YOUR_HF_TOKEN")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
