# Unsloth Notebooks - Notebooks for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

"""API drift detectors for the unsloth / unsloth_zoo / trl / transformers /
peft / datasets surfaces actually invoked by the notebooks. One test per
high-frequency symbol. Each asserts the healthy upstream shape (class
exists; classmethod accepts at minimum the kwargs the notebooks pass).
On regression -> ``pytest.fail("DRIFT DETECTED: ...")`` -- never
``pytest.skip`` -- so CI goes red and the maintainer triages BEFORE a
user on Colab hits the crash.

Each test cites the call-site count measured against the cloned repo
(both ``original_template/`` and ``nb/`` aggregated). Mirrors the
pattern landed in unslothai/unsloth#5414 + unslothai/unsloth-zoo#637.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _signature_param_names(callable_obj) -> set[str]:
    """Set of parameter names accepted by ``callable_obj``. Returns the
    empty set on signature inspection failure (treated as 'cannot
    verify' rather than 'definitely missing')."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set()
    return set(sig.parameters)


def _accepts(callable_obj, kwargs: set[str]) -> tuple[bool, set[str]]:
    """True if every name in ``kwargs`` is either a named parameter on
    ``callable_obj`` OR the callable's signature has a ``**kwargs``
    catch-all (in which case we cannot prove a name is rejected and
    accept it conservatively). Returns (ok, missing_set)."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True, set()
    params = sig.parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_kw:
        return True, set()
    missing = kwargs - set(params)
    return (not missing), missing


# ===========================================================================
# unsloth (top-level imports + class methods)
# ===========================================================================


def test_unsloth_fast_language_model_class_present():
    """`from unsloth import FastLanguageModel` -- 490 import call sites
    across the notebooks. Without the class, every LoRA notebook crashes
    at cell 6."""
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastLanguageModel"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastLanguageModel is missing; "
            "490 notebook import sites would crash."
        )


def test_unsloth_fast_language_model_from_pretrained_kwargs():
    """`FastLanguageModel.from_pretrained` -- 506 call sites. Notebooks
    pass at minimum model_name / max_seq_length / dtype / load_in_4bit.
    Optional advanced kwargs measured in templates: load_in_8bit,
    full_finetuning, fast_inference, gpu_memory_utilization,
    max_lora_rank, trust_remote_code, token."""
    unsloth = pytest.importorskip("unsloth")
    required = {"model_name", "max_seq_length", "dtype", "load_in_4bit"}
    ok, missing = _accepts(unsloth.FastLanguageModel.from_pretrained, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastLanguageModel.from_pretrained signature "
            f"no longer accepts {sorted(missing)}; the most common notebook "
            f"call shape would crash with TypeError."
        )


def test_unsloth_fast_language_model_get_peft_model_kwargs():
    """`FastLanguageModel.get_peft_model` -- 304 call sites. Notebooks
    pass r / lora_alpha / lora_dropout / target_modules / bias /
    use_gradient_checkpointing / random_state / use_rslora /
    loftq_config."""
    unsloth = pytest.importorskip("unsloth")
    required = {
        "r", "lora_alpha", "lora_dropout", "target_modules", "bias",
        "use_gradient_checkpointing", "random_state",
    }
    ok, missing = _accepts(unsloth.FastLanguageModel.get_peft_model, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastLanguageModel.get_peft_model signature "
            f"no longer accepts {sorted(missing)}."
        )


def test_unsloth_fast_language_model_for_inference_callable():
    """`FastLanguageModel.for_inference` -- 370 call sites."""
    unsloth = pytest.importorskip("unsloth")
    if not callable(getattr(unsloth.FastLanguageModel, "for_inference", None)):
        pytest.fail(
            "DRIFT DETECTED: FastLanguageModel.for_inference is missing; "
            "370 inference-cell call sites would crash."
        )


def test_unsloth_fast_vision_model_class_and_methods():
    """`FastVisionModel.from_pretrained` (176) / .get_peft_model (99) /
    .for_inference (183) / .for_training (60)."""
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastVisionModel"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastVisionModel is missing; "
            "180 import call sites + 500+ method call sites would crash."
        )
    cls = unsloth.FastVisionModel
    missing_methods = [
        m for m in ("from_pretrained", "get_peft_model", "for_inference", "for_training")
        if not callable(getattr(cls, m, None))
    ]
    if missing_methods:
        pytest.fail(
            f"DRIFT DETECTED: FastVisionModel is missing methods "
            f"{missing_methods}."
        )


def test_unsloth_fast_vision_model_get_peft_model_vision_kwargs():
    """Vision-specific kwargs: finetune_vision_layers,
    finetune_language_layers, finetune_attention_modules,
    finetune_mlp_modules."""
    unsloth = pytest.importorskip("unsloth")
    required = {
        "finetune_vision_layers", "finetune_language_layers",
        "finetune_attention_modules", "finetune_mlp_modules",
    }
    ok, missing = _accepts(unsloth.FastVisionModel.get_peft_model, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: FastVisionModel.get_peft_model dropped "
            f"vision kwargs {sorted(missing)}."
        )


def test_unsloth_fast_model_class_and_methods():
    """`FastModel` -- 103 import sites + 103 from_pretrained + 67
    get_peft_model call sites (the modern unified entry point)."""
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastModel"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastModel is missing; the modern "
            "unified entry point used by 100+ notebooks would crash."
        )
    missing_methods = [
        m for m in ("from_pretrained", "get_peft_model")
        if not callable(getattr(unsloth.FastModel, m, None))
    ]
    if missing_methods:
        pytest.fail(
            f"DRIFT DETECTED: FastModel is missing methods {missing_methods}."
        )


def test_unsloth_is_bf16_supported_or_alias_callable():
    """`is_bf16_supported` (48 import sites) and the legacy alias
    `is_bfloat16_supported` (8). Either must be importable."""
    unsloth = pytest.importorskip("unsloth")
    has_new = callable(getattr(unsloth, "is_bf16_supported", None))
    has_old = callable(getattr(unsloth, "is_bfloat16_supported", None))
    if not (has_new or has_old):
        pytest.fail(
            "DRIFT DETECTED: neither unsloth.is_bf16_supported nor "
            "unsloth.is_bfloat16_supported is callable; 50+ notebooks "
            "fail at dtype probing."
        )


def test_unsloth_get_chat_template_callable():
    """`unsloth.get_chat_template` -- 160 (chat_templates) + 16 (notebook
    top-level) import sites."""
    unsloth = pytest.importorskip("unsloth")
    has_top = callable(getattr(unsloth, "get_chat_template", None))
    try:
        from unsloth.chat_templates import get_chat_template as _gct
        has_chat = callable(_gct)
    except Exception:
        has_chat = False
    if not (has_top or has_chat):
        pytest.fail(
            "DRIFT DETECTED: get_chat_template missing from both "
            "`unsloth` and `unsloth.chat_templates`; conversational "
            "notebooks would crash at template setup."
        )


def test_unsloth_train_on_responses_only_callable():
    """`train_on_responses_only` -- 100 import sites."""
    unsloth = pytest.importorskip("unsloth")
    has_top = callable(getattr(unsloth, "train_on_responses_only", None))
    try:
        from unsloth.chat_templates import train_on_responses_only as _t
        has_chat = callable(_t)
    except Exception:
        has_chat = False
    if not (has_top or has_chat):
        pytest.fail(
            "DRIFT DETECTED: train_on_responses_only missing from both "
            "`unsloth` and `unsloth.chat_templates`."
        )


def test_unsloth_vision_data_collator_class_present():
    """`UnslothVisionDataCollator` -- 56 import sites."""
    unsloth = pytest.importorskip("unsloth")
    has_top = hasattr(unsloth, "UnslothVisionDataCollator")
    try:
        from unsloth.trainer import UnslothVisionDataCollator as _c
        has_tr = _c is not None
    except Exception:
        has_tr = False
    if not (has_top or has_tr):
        pytest.fail(
            "DRIFT DETECTED: UnslothVisionDataCollator missing; vision "
            "fine-tuning notebooks would crash at data collation."
        )


def test_unsloth_fast_sentence_transformer_class_present():
    """`FastSentenceTransformer` -- 48 import sites."""
    unsloth = pytest.importorskip("unsloth")
    if not hasattr(unsloth, "FastSentenceTransformer"):
        pytest.fail(
            "DRIFT DETECTED: unsloth.FastSentenceTransformer is missing; "
            "embedding-model notebooks (BGE_M3, EmbeddingGemma, etc.) "
            "would crash at import."
        )


# ===========================================================================
# trl
# ===========================================================================


def test_trl_sft_trainer_class_present():
    """`from trl import SFTTrainer` -- 319 call sites."""
    trl = pytest.importorskip("trl")
    if not hasattr(trl, "SFTTrainer"):
        pytest.fail("DRIFT DETECTED: trl.SFTTrainer is missing.")


def test_trl_sft_trainer_or_config_accepts_dataset_text_field():
    """Notebooks pass `dataset_text_field` to one of SFTTrainer or
    SFTConfig (newer TRL moved it onto the config). Accept either."""
    trl = pytest.importorskip("trl")
    candidates = []
    if hasattr(trl, "SFTTrainer"):
        candidates.append(trl.SFTTrainer.__init__)
    if hasattr(trl, "SFTConfig"):
        candidates.append(trl.SFTConfig)
    if not candidates:
        pytest.fail("DRIFT DETECTED: neither trl.SFTTrainer nor trl.SFTConfig is importable.")
    for cb in candidates:
        ok, _missing = _accepts(cb, {"dataset_text_field"})
        if ok:
            return
    pytest.fail(
        "DRIFT DETECTED: dataset_text_field accepted by neither "
        "SFTTrainer.__init__ nor SFTConfig; 250+ SFT notebooks fail."
    )


def test_trl_sft_config_training_arg_kwargs():
    """SFTConfig must accept the canonical training arg shape passed by
    SFT notebooks: per_device_train_batch_size,
    gradient_accumulation_steps, learning_rate, optim, logging_steps,
    weight_decay, lr_scheduler_type, seed, warmup_steps."""
    trl = pytest.importorskip("trl")
    if not hasattr(trl, "SFTConfig"):
        pytest.fail("DRIFT DETECTED: trl.SFTConfig is missing.")
    required = {
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "optim", "logging_steps", "weight_decay",
        "lr_scheduler_type", "seed", "warmup_steps",
    }
    ok, missing = _accepts(trl.SFTConfig, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: trl.SFTConfig signature dropped kwargs "
            f"{sorted(missing)}."
        )


def test_trl_grpo_config_and_trainer_present():
    """GRPOConfig (137) / GRPOTrainer (137) -- modern reasoning loop."""
    trl = pytest.importorskip("trl")
    missing = [n for n in ("GRPOConfig", "GRPOTrainer") if not hasattr(trl, n)]
    if missing:
        pytest.fail(
            f"DRIFT DETECTED: trl is missing {missing}; GRPO reasoning "
            f"notebooks (Llama3.1 GRPO, Qwen3 GRPO, DeepSeek R1 GRPO) "
            f"would crash at import."
        )


def test_trl_grpo_config_accepts_canonical_kwargs():
    """GRPO-specific kwargs: num_generations, max_prompt_length,
    max_completion_length, beta."""
    trl = pytest.importorskip("trl")
    if not hasattr(trl, "GRPOConfig"):
        pytest.skip("GRPOConfig not present; see test_trl_grpo_config_and_trainer_present.")
    required = {"num_generations", "max_prompt_length", "max_completion_length"}
    ok, missing = _accepts(trl.GRPOConfig, required)
    if not ok:
        pytest.fail(
            f"DRIFT DETECTED: trl.GRPOConfig signature dropped kwargs "
            f"{sorted(missing)}."
        )


# ===========================================================================
# peft
# ===========================================================================


def test_peft_auto_peft_model_for_causal_lm_present():
    """`from peft import AutoPeftModelForCausalLM` -- 109 sites."""
    peft = pytest.importorskip("peft")
    if not hasattr(peft, "AutoPeftModelForCausalLM"):
        pytest.fail(
            "DRIFT DETECTED: peft.AutoPeftModelForCausalLM is missing; "
            "every notebook that loads back a saved adapter via the "
            "AutoPeft helper would crash."
        )
    cls = peft.AutoPeftModelForCausalLM
    if not callable(getattr(cls, "from_pretrained", None)):
        pytest.fail(
            "DRIFT DETECTED: AutoPeftModelForCausalLM.from_pretrained "
            "is not callable."
        )


# ===========================================================================
# transformers
# ===========================================================================


def test_transformers_text_streamer_class_present():
    """`from transformers import TextStreamer` -- 767 sites (most cited
    transformers symbol across the notebook tree)."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "TextStreamer"):
        pytest.fail(
            "DRIFT DETECTED: transformers.TextStreamer is missing; "
            "every inference cell crashes at TextStreamer construction."
        )


def test_transformers_auto_tokenizer_present():
    """`AutoTokenizer` -- 124 import sites."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "AutoTokenizer"):
        pytest.fail("DRIFT DETECTED: transformers.AutoTokenizer is missing.")
    if not callable(getattr(transformers.AutoTokenizer, "from_pretrained", None)):
        pytest.fail("DRIFT DETECTED: AutoTokenizer.from_pretrained not callable.")


def test_transformers_training_arguments_class_present():
    """`TrainingArguments` -- 68 import sites."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "TrainingArguments"):
        pytest.fail("DRIFT DETECTED: transformers.TrainingArguments is missing.")


def test_transformers_trainer_class_present():
    """`Trainer` -- 36 import sites."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Trainer"):
        pytest.fail("DRIFT DETECTED: transformers.Trainer is missing.")


def test_transformers_data_collator_for_seq2seq_present():
    """`DataCollatorForSeq2Seq` -- 22 import sites."""
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DataCollatorForSeq2Seq"):
        pytest.fail(
            "DRIFT DETECTED: transformers.DataCollatorForSeq2Seq is "
            "missing; conversational SFT notebooks fail at collator init."
        )


# ===========================================================================
# datasets
# ===========================================================================


def test_datasets_load_dataset_callable():
    """`from datasets import load_dataset` -- 415 / 479 sites."""
    datasets = pytest.importorskip("datasets")
    if not callable(getattr(datasets, "load_dataset", None)):
        pytest.fail(
            "DRIFT DETECTED: datasets.load_dataset is not callable; "
            "every dataset-loading cell crashes."
        )


def test_datasets_dataset_class_present():
    """`from datasets import Dataset` -- 151 sites."""
    datasets = pytest.importorskip("datasets")
    if not hasattr(datasets, "Dataset"):
        pytest.fail("DRIFT DETECTED: datasets.Dataset is missing.")
