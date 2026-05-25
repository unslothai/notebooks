# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "MeCab",
#     "accelerate",
#     "argbind",
#     "bitsandbytes>=0.43.0",
#     "datasets==4.3.0",
#     "descript-audio-codec",
#     "descript-audiotools",
#     "einx",
#     "ffmpy",
#     "flatten_dict",
#     "ftfy",
#     "hf_transfer",
#     "huggingface_hub>=0.34.0",
#     "julius",
#     "loguru",
#     "marimo",
#     "omegaconf",
#     "openai-whisper",
#     "peft",
#     "protobuf",
#     "pyloudnorm",
#     "randomname",
#     "sentencepiece",
#     "tiktoken",
#     "torchao>=0.16.0",
#     "torchcodec",
#     "transformers==4.56.2",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "uroman",
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://github.com/unslothai/notebooks/blob/main/nb/Oute_TTS_(1B).ipynb" target="_parent"><img src="https://marimo.io/molab-shield.svg" alt="Open In Colab"/></a>
    """)
    return


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
    ### Unsloth

    `FastModel` supports loading nearly any model now! This includes Vision and Text models!

    Thank you to [Etherl](https://huggingface.co/Etherll) for creating this notebook!
    """)
    return


@app.cell
def _():
    from unsloth import FastModel
    import torch

    max_seq_length = 2048  # Choose any for long context!
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
        model_name="unsloth/Llama-OuteTTS-1.0-1B",
        max_seq_length=max_seq_length,  # Choose any for long context!
        dtype=None,  # Set to None for auto detection
        load_in_4bit=False,  # Set to True for 4bit which reduces memory
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    return FastModel, max_seq_length, model, tokenizer, torch


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
        r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "v_proj"],
        lora_alpha=128,
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

    We will use the `MrDragonFox/Elise`, which is designed for training TTS models. Ensure that your dataset follows the required format: **text, audio**, but maintaining the correct structure is essential for optimal training.
    """)
    return


@app.cell
def _():
    from datasets import load_dataset, Audio, Dataset

    dataset = load_dataset("MrDragonFox/Elise", split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    return Audio, Dataset, dataset


@app.cell
def _(Dataset, dataset, torch):
    # @title Tokenization Function
    from tqdm import tqdm
    import io
    import tempfile
    import sys

    sys.path.append("OuteTTS")
    import os
    import dac
    from outetts.version.v3.audio_processor import AudioProcessor
    from outetts.version.v3.prompt_processor import PromptProcessor
    from outetts.dac.interface import DacInterface

    # V3 Imports
    from outetts.models.config import ModelConfig
    import whisper
    from outetts.utils.preprocessing import text_normalizations
    import soundfile as sf  # Need a dummy config for AudioProcessor
    import numpy as np

    class DataCreationV3:
        def __init__(
            self,
            model_tokenizer_path: str,
            whisper_model_name: str = "turbo",
            device: str = None,
        ):
            self.device = (
                device if device else "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"Using device: {self.device}")
            dummy_config = ModelConfig(
                tokenizer_path=model_tokenizer_path,
                device=self.device,
                audio_codec_path=None,  # Let AudioProcessor use default DAC path
            )
            self.audio_processor = AudioProcessor(config=dummy_config)
            self.prompt_processor = PromptProcessor(model_tokenizer_path)
            print(f"Loading Whisper model: {whisper_model_name} on {self.device}")
            self.whisper_model = whisper.load_model(
                whisper_model_name, device=self.device
            )
            print("Whisper model loaded.")

        def create_speaker_representation(self, audio_bytes: bytes, transcript: str):
            """
            Creates a v3-compatible speaker dictionary using Whisper and AudioProcessor.  # Create a dummy ModelConfig mainly for device and paths needed by AudioProcessor/DacInterface
            """
            if not audio_bytes or not transcript:
                print(
                    "Missing audio bytes or transcript in create_speaker_representation."
                )
                return None  # Let AudioProcessor use default DAC path
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=True
                ) as tmp_audio_file:
                    tmp_audio_file.write(audio_bytes)
                    tmp_audio_file.flush()
                    whisper_result = self.whisper_model.transcribe(
                        tmp_audio_file.name, word_timestamps=True
                    )
                    normalized_transcript = text_normalizations(transcript)
                    words_with_timings = []
                    if whisper_result and "segments" in whisper_result:
                        for segment in whisper_result[
                            "segments"
                        ]:  # Renamed and adapted from the previous version
                            if "words" in segment:
                                for word_info in segment["words"]:
                                    cleaned_word = word_info["word"].strip()
                                    if cleaned_word:
                                        words_with_timings.append(
                                            {
                                                "word": cleaned_word,
                                                "start": float(word_info["start"]),
                                                "end": float(word_info["end"]),
                                            }
                                        )
                    else:
                        print(
                            f"Whisper did not return segments/words for: {transcript[:50]}..."
                        )
                        return None
                    if (
                        not words_with_timings
                    ):  # Whisper needs a file path, so save bytes to a temporary file
                        print(
                            f"No word timings extracted by Whisper for: {transcript[:50]}..."
                        )
                        return None
                    speaker_data_dict = {
                        "audio": {"bytes": audio_bytes},
                        "text": normalized_transcript,
                        "words": words_with_timings,
                    }
                    v3_speaker = self.audio_processor.create_speaker_from_dict(
                        speaker_data_dict
                    )  # Ensure data is written
                    return v3_speaker
            except Exception as e:  # 1. Get word timings using Whisper
                print(f"Error during speaker creation (Whisper/AudioProcessor): {e}")
                return None  # Use the provided transcript for consistency, but Whisper timings

        def process_dataset(self, dataset: Dataset):
            """
                  Processes a Hugging Face Dataset object in memory and yields training prompts.

                  Args:
                      dataset (Dataset): The Hugging Face dataset to process.
                                         Expected columns: 'text' (str) and 'audio' (dict with 'bytes').  # Use original word casing/punctuation from Whisper's output if needed,
            # but strip excess whitespace for consistency.
                  Yields:
                      str: The processed training prompt string for each valid row.  # Ignore empty strings
            """
            processed_count = 0
            skipped_count = 0
            for i, item in enumerate(tqdm(dataset, desc="Processing Dataset")):
                try:
                    transcript = item.get("text")
                    audio_info = item.get("audio")
                    if not transcript or not isinstance(
                        transcript, str
                    ):  # Indicate failure
                        print(
                            f"Row {i}: Skipping due to missing or invalid 'text' column."
                        )
                        skipped_count = skipped_count + 1
                        continue
                    audio_array = audio_info["array"]
                    buffer = io.BytesIO()
                    sf.write(
                        buffer,
                        audio_array.astype(np.float32),
                        audio_info["sampling_rate"],
                        format="WAV",
                        subtype="FLOAT",
                    )  # Prepare data dict for AudioProcessor
                    buffer.seek(0)
                    audio_bytes = buffer.getvalue()
                    speaker = self.create_speaker_representation(
                        audio_bytes, transcript
                    )  # Use the potentially normalized transcript
                    if speaker is None:
                        print(
                            f"Row {i}: Failed to create speaker representation for text: {transcript[:50]}... Skipping."
                        )
                        skipped_count = skipped_count + 1
                        continue  # 2. Use AudioProcessor to create the speaker representation
                    prompt = self.prompt_processor.get_training_prompt(speaker)
                    processed_count = processed_count + 1
                    yield prompt
                except KeyboardInterrupt:
                    print("Processing interrupted by user.")
                    break  # Indicate failure
                except Exception as e:
                    print(
                        f"Row {i}: Unhandled error processing item: {e}", exc_info=True
                    )
                    skipped_count = (
                        skipped_count + 1
                    )  # --- V3 Changes: run method is now a generator ---
                    continue
            print(
                f"Dataset processing finished. Processed: {processed_count}, Skipped: {skipped_count}"
            )

    if __name__ == "__main__":
        _MODEL_TOKENIZER_PATH = "OuteAI/Llama-OuteTTS-1.0-1B"
        _WHISPER_MODEL = "turbo"  # Or "small.en", "medium.en", "large-v2", etc.
        data_processor = DataCreationV3(
            model_tokenizer_path=_MODEL_TOKENIZER_PATH,
            whisper_model_name=_WHISPER_MODEL,
        )
        all_prompts = []
        print("Starting dataset processing...")
        procced_dataset = data_processor.process_dataset(dataset)
        for prompt in procced_dataset:
            if prompt:
                all_prompts.append({"text": prompt})
        dataset_1 = Dataset.from_list(all_prompts)
        print("Moving Whisper model to CPU")
        data_processor.whisper_model.to("cpu")  # Iterate directly over the dataset
        torch.cuda.empty_cache()
    return data_processor, dataset_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model
    Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support `DPOTrainer` and `GRPOTrainer` for reinforcement learning!!
    """)
    return


@app.cell
def _(dataset_1, max_seq_length, model_1, tokenizer):
    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        train_dataset=dataset_1,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # Choose any for long context!
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
    Let's run the model! You can change the prompts
    """)
    return


@app.cell
def _():
    input_text = "Hey there my name is Elise, and I'm a speech generation model that can sound like a person."
    return (input_text,)


@app.cell
def _(Audio, FastModel, data_processor, input_text, model_1, tokenizer, torch):
    # @title Run Inference
    import re
    from typing import Dict, Any
    import torchaudio.transforms as T
    from transformers import LogitsProcessor
    import transformers.generation.utils as generation_utils
    from transformers import AutoModelForCausalLM

    FastModel.for_inference(model_1)

    def get_audio(tokens):
        decoded_output = tokenizer.batch_decode(tokens, skip_special_tokens=False)[0]
        c1 = list(map(int, re.findall("<\\|c1_(\\d+)\\|>", decoded_output)))
        c2 = list(map(int, re.findall("<\\|c2_(\\d+)\\|>", decoded_output)))
        t = min(len(c1), len(c2))
        c1 = c1[:t]
        c2 = c2[:t]
        output = [c1, c2]
        if not output:
            print("No audio tokens found in the output")
            return None
        return data_processor.audio_processor.audio_codec.decode(
            torch.tensor([output], dtype=torch.int64).to(
                data_processor.audio_processor.audio_codec.device
            )
        )

    class RepetitionPenaltyLogitsProcessorPatch(LogitsProcessor):
        def __init__(self, penalty: float):
            penalty_last_n = 64
            print(
                "🔄 Using patched RepetitionPenaltyLogitsProcessor -> RepetitionPenaltyLogitsProcessorPatch | penalty_last_n: {penalty_last_n}"
            )
            if penalty_last_n is not None:
                if not isinstance(penalty_last_n, int) or penalty_last_n < 0:
                    raise ValueError(
                        f"`penalty_last_n` has to be a non-negative integer, but is {penalty_last_n}"
                    )
            if not isinstance(penalty, float) or penalty <= 0:
                raise ValueError(
                    f"`penalty` has to be a positive float, but is {penalty}"
                )
            self.penalty_last_n = penalty_last_n
            self.penalty = penalty

        @torch.no_grad()
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            """
            Args:
                input_ids (`torch.LongTensor`):
                    Indices of input sequence tokens in the vocabulary (shape `(batch_size, sequence_length)`).
                scores (`torch.FloatTensor`):
                    Prediction scores of a language modeling head (shape `(batch_size, vocab_size)`).

            Returns:
                `torch.FloatTensor`: The modified prediction scores.
            """
            if self.penalty_last_n == 0 or self.penalty == 1.0:
                return scores
            batch_size, seq_len = input_ids.shape
            vocab_size = scores.shape[-1]
            for b in range(batch_size):
                start_index = max(0, seq_len - self.penalty_last_n)
                window_indices = input_ids[b, start_index:]  # Shape: (window_len,)
                if window_indices.numel() == 0:
                    continue  # Check if penalties should be applied
                tokens_in_window = set(window_indices.tolist())
                for token_id in tokens_in_window:
                    if token_id >= vocab_size:
                        continue
                    logit = scores[b, token_id]
                    if logit <= 0:
                        logit = (
                            logit * self.penalty
                        )  # Process each batch item independently
                    else:
                        logit = logit / self.penalty  # 1. Determine the penalty window
                    scores[b, token_id] = logit
            return scores  # Shape: (window_len,)

    generation_utils.RepetitionPenaltyLogitsProcessor = (
        RepetitionPenaltyLogitsProcessorPatch
    )
    AutoModelForCausalLM.generate = (
        generation_utils.GenerationMixin.generate
    )  # Skip if window is empty
    if __name__ == "__main__":
        formated_text = "<|text_start|>" + input_text + "<|text_end|>"
        prompt_1 = "\n".join(
            ["<|im_start|>", formated_text, "<|audio_start|><|global_features_start|>"]
        )  # 2. Find unique tokens within the window
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=model_1.dtype):
                model_inputs = tokenizer([prompt_1], return_tensors="pt").to(
                    "cuda"
                )  # 3. Apply repetition penalty to the scores for this batch item
                print("Generating token sequence...")
                generated_ids = model_1.generate(
                    **model_inputs,
                    temperature=0.4,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    min_p=0.05,
                    max_new_tokens=2048,  # Limit generation length
                )
                print("Token sequence generated.")
        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]
        audio = get_audio(generated_ids)
        audio = audio.cpu()
        from IPython.display import display

        display(
            Audio(audio.squeeze(0), rate=24000)
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
    model_1.save_pretrained("oute_tts_lora")
    tokenizer.save_pretrained("oute_tts_lora")
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
            "oute_tts_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/oute_tts_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_merged(
            "oute_tts_finetune_4bit", tokenizer, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/oute_tts_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained("oute_tts_lora")
        tokenizer.save_pretrained("oute_tts_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/oute_tts_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/oute_tts_lora", token="YOUR_HF_TOKEN")
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
