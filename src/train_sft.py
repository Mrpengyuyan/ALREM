"""Train SFT with ALREM / Uniform / Matched LoRA.

Supports tasks: mgsm, flores, sparql.
Supports QLoRA 4-bit quantization.
Supports two-stage training (Stage 2 resumes from Stage 1 adapter checkpoint).
"""

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .alrem_rank_pattern import build_alpha_pattern, estimate_alrem_params, solve_r_match
from .data_flores import build_flores_train_text, load_flores_dataset
from .data_mgsm import build_mgsm_train_text, load_mgsm_dataset
from .lora_utils import (
    compute_lora_params_by_layer,
    compute_total_lora_params,
    estimate_lora_params_uniform,
    summarize_ranks,
)
from .prompts import FLORES_PROMPT, MGSM_PROMPT, build_sparql_train_text
from .step_time_logging import StepTimeLoggingCallback
from .utils import count_parameters, ensure_dir, load_yaml, save_json, save_yaml, set_seed, setup_logging

LOGGER = logging.getLogger("alrem.train_sft")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT with ALREM LoRA.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Override stage1_checkpoint from config.")
    return parser.parse_args()


def _ensure_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def _override_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg["max_eval_samples"] = args.max_eval_samples
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.stage1_checkpoint is not None:
        cfg["stage1_checkpoint"] = args.stage1_checkpoint
    return cfg


def _cfg_int(cfg: Dict[str, Any], *keys: str, default: int = 0) -> int:
    """Read the first matching key from cfg, return as int."""
    for key in keys:
        val = cfg.get(key)
        if val is not None:
            return int(val)
    return default


def _cfg_float(cfg: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        val = cfg.get(key)
        if val is not None:
            return float(val)
    return default


def _warn_if_conflicting_cfg(
    cfg: Dict[str, Any],
    logger: logging.Logger,
    new_key: str,
    old_key: str,
) -> None:
    """Warn when new/old compatibility keys are both set but differ."""
    if new_key in cfg and old_key in cfg and cfg.get(new_key) != cfg.get(old_key):
        logger.warning(
            "Config conflict: '%s'=%s overrides legacy '%s'=%s.",
            new_key,
            cfg.get(new_key),
            old_key,
            cfg.get(old_key),
        )


def _select_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _tokenize_dataset(
    tokenizer,
    dataset: Dataset,
    max_seq_len: int,
    add_special_tokens: bool = True,
) -> Dataset:
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=add_special_tokens,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tok, batched=True, remove_columns=["text"])


# ── SPARQL data loading ──────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file. Each line is a JSON object."""
    records: List[Dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    LOGGER.warning("Skip non-dict record at %s:%d", path, line_no)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skip invalid JSON at %s:%d (%s)", path, line_no, exc)
    return records


def _build_sparql_texts(
    records: List[Dict[str, Any]], tokenizer: Any
) -> List[str]:
    """Convert JSONL records to training text strings."""
    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "SPARQL task requires tokenizer chat template support. "
            "Current tokenizer does not provide a valid chat_template."
        )

    texts: List[str] = []
    skipped = 0
    for rec in records:
        question = str(rec.get("question", "")).strip()
        sparql = str(rec.get("sparql", "")).strip()
        if not question or not sparql:
            skipped += 1
            continue
        text = build_sparql_train_text(question, sparql, tokenizer)
        texts.append(text)
    if skipped > 0:
        LOGGER.warning("Skipped %d records with missing question/sparql fields.", skipped)
    return texts


def _load_sparql_data(
    cfg: Dict[str, Any], tokenizer: Any
) -> tuple:
    """Load SPARQL train/eval data based on config.

    Supports:
      - Stage 1 only (stage1_data_path set, no stage1_checkpoint)
      - Stage 2 only (stage2_data_path set, stage1_checkpoint set)
      - Can also be used standalone with just data_path / dev_path
    """
    stage1_checkpoint = cfg.get("stage1_checkpoint")

    # Determine which data paths to use
    if stage1_checkpoint:
        # Stage 2: use stage2 data
        train_path = cfg.get("stage2_data_path")
        dev_path = cfg.get("stage2_dev_path")
        stage_name = "Stage 2"
    else:
        # Stage 1: use stage1 data
        train_path = cfg.get("stage1_data_path")
        dev_path = cfg.get("stage1_dev_path")
        stage_name = "Stage 1"

    if not train_path:
        raise ValueError(
            f"SPARQL {stage_name}: no training data path configured. "
            f"Set {'stage2_data_path' if stage1_checkpoint else 'stage1_data_path'} in config."
        )

    LOGGER.info("SPARQL %s — train data: %s", stage_name, train_path)
    train_records = _load_jsonl(train_path)
    train_texts = _build_sparql_texts(train_records, tokenizer)

    eval_texts: List[str] = []
    if dev_path:
        LOGGER.info("SPARQL %s — dev data: %s", stage_name, dev_path)
        dev_records = _load_jsonl(dev_path)
        eval_texts = _build_sparql_texts(dev_records, tokenizer)

    LOGGER.info("SPARQL %s — train samples: %d, eval samples: %d",
                stage_name, len(train_texts), len(eval_texts))
    return train_texts, eval_texts


# ── QLoRA / quantization ─────────────────────────────────────────────────────

def _load_model_with_quantization(
    model_name: str,
    quantization: Optional[str],
    torch_dtype,
    trust_remote_code: bool = False,
):
    """Load a model, optionally with 4-bit quantization."""
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        # Prepare for k-bit training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        return model

    # Standard loading (no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model


# ── LoRA setup ────────────────────────────────────────────────────────────────

def _setup_lora(
    model,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> tuple:
    """Build LoRA config, estimate params, and return (peft_model, params_payload).

    For Stage 2 with stage1_checkpoint: loads the saved adapter instead of creating new.
    """
    target_modules = _ensure_list(cfg.get("target_modules", ["q_proj", "v_proj"]))

    # ALREM v2 module-aware support
    target_modules_attn = _ensure_list(cfg.get("target_modules_attention"))
    target_modules_mlp = _ensure_list(cfg.get("target_modules_mlp"))

    if target_modules_attn and target_modules_mlp:
        from .alrem_rank_pattern import estimate_alrem_v2_params
        target_modules = list(set(target_modules_attn + target_modules_mlp))
        attn_config = {
            "r_high": int(cfg.get("attention_r_high", 32)),
            "r_low": int(cfg.get("attention_r_low", 4)),
            "cut_ratio_early": cfg.get("attention_cut_ratio_early", 0.2),
            "cut_ratio_mid": cfg.get("attention_cut_ratio_mid", 0.8),
        }
        mlp_config = {
            "r_uniform": int(cfg.get("mlp_r_uniform", 16)),
        }
        alrem_target_params, rank_pattern, num_layers, early_end, mid_end = estimate_alrem_v2_params(
            model=model,
            attn_modules=target_modules_attn,
            mlp_modules=target_modules_mlp,
            attn_config=attn_config,
            mlp_config=mlp_config,
        )
        r_uniform = int(cfg.get("mlp_r_uniform", 16))
    else:
        r_high = int(cfg.get("r_high", 32))
        r_low = int(cfg.get("r_low", 8))
        r_uniform = int(cfg.get("r_uniform", 16))

        alrem_target_params, rank_pattern, num_layers, early_end, mid_end = estimate_alrem_params(
            model=model,
            target_modules=target_modules,
            r_high=r_high,
            r_low=r_low,
            cut_ratio_early=cfg.get("cut_ratio_early"),
            cut_ratio_mid=cfg.get("cut_ratio_mid"),
            early_end=cfg.get("early_end"),
            mid_end=cfg.get("mid_end"),
        )

    uniform_params = estimate_lora_params_uniform(model, target_modules, r_uniform)
    r_match, uniform_matched_params, rel_error = solve_r_match(
        target_params=alrem_target_params,
        model=model,
        target_modules=target_modules,
        r_min=1,
        r_max=int(cfg.get("r_match_max", 128)),
    )

    logger.info("Layer count: %d", num_layers)
    logger.info("Cut indices: early_end=%d mid_end=%d", early_end, mid_end)
    _, ranks_by_layer = summarize_ranks(rank_pattern)
    for key in sorted(ranks_by_layer, key=lambda x: (x == "layer_na", x)):
        uniq = sorted(set(ranks_by_layer[key]))
        logger.info("Rank %s: %s", key, uniq)

    logger.info("ALREM target LoRA params: %d", alrem_target_params)
    logger.info("Uniform r=%d params: %d", r_uniform, uniform_params)
    logger.info("Matched uniform r=%d params: %d (relative_error=%.4f)",
                r_match, uniform_matched_params, rel_error)

    # ── Stage 2: load existing adapter ──
    stage1_checkpoint = cfg.get("stage1_checkpoint")
    if stage1_checkpoint:
        ckpt_dir = Path(stage1_checkpoint)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"Stage 2 checkpoint path does not exist: {stage1_checkpoint}"
            )

        adapter_cfg_path = ckpt_dir / "adapter_config.json"
        if not adapter_cfg_path.exists():
            raise FileNotFoundError(
                "Missing adapter_config.json in stage1_checkpoint. "
                f"Expected: {adapter_cfg_path}"
            )

        with adapter_cfg_path.open("r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        adapter_base = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
        current_base = str(cfg.get("model_name_or_path", "")).strip()
        if adapter_base and current_base and adapter_base != current_base:
            raise ValueError(
                "Stage 2 base model mismatch.\n"
                f"  stage1_checkpoint base: {adapter_base}\n"
                f"  current config base:    {current_base}\n"
                "Use the exact same base model for Stage 1 and Stage 2."
            )

        adapter_modules = _ensure_list(adapter_cfg.get("target_modules"))
        if adapter_modules and target_modules and set(adapter_modules) != set(target_modules):
            raise ValueError(
                "Stage 2 target_modules mismatch with loaded adapter.\n"
                f"  adapter target_modules: {sorted(adapter_modules)}\n"
                f"  config target_modules:  {sorted(target_modules)}\n"
                "Use a matching Stage 1 checkpoint for this Stage 2 config."
            )

        method = str(cfg.get("method", "alrem")).lower()
        if method == "alrem":
            adapter_rank_pattern = adapter_cfg.get("rank_pattern")
            if not adapter_rank_pattern:
                raise ValueError(
                    "Stage 2 method=alrem requires a Stage 1 ALREM adapter with rank_pattern.\n"
                    "Loaded adapter has no rank_pattern (likely non-ALREM or incompatible checkpoint)."
                )
            if adapter_rank_pattern != rank_pattern:
                raise ValueError(
                    "Stage 2 ALREM rank pattern mismatch with loaded adapter.\n"
                    "This Stage 2 config does not match the Stage 1 adapter architecture.\n"
                    "Use a Stage 1 checkpoint trained with the same ALREM rank pattern."
                )
        elif method == "uniform":
            adapter_r = adapter_cfg.get("r")
            if adapter_r is not None and int(adapter_r) != int(r_uniform):
                raise ValueError(
                    "Stage 2 uniform rank mismatch with loaded adapter.\n"
                    f"  adapter r: {adapter_r}\n"
                    f"  config r_uniform: {r_uniform}\n"
                    "Use a Stage 1 uniform checkpoint with matching rank."
                )
        elif method == "matched":
            adapter_r = adapter_cfg.get("r")
            if adapter_r is not None and int(adapter_r) != int(r_match):
                raise ValueError(
                    "Stage 2 matched rank mismatch with loaded adapter.\n"
                    f"  adapter r: {adapter_r}\n"
                    f"  expected matched r: {r_match}\n"
                    "Use a Stage 1 matched checkpoint generated from the same base/method config."
                )

        logger.info("Stage 2: loading adapter from %s", stage1_checkpoint)
        peft_model = PeftModel.from_pretrained(
            model,
            stage1_checkpoint,
            is_trainable=True,
        )
        # Ensure all LoRA params are trainable
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        logger.info("Stage 2: adapter loaded and set to trainable.")
    else:
        # ── Stage 1 or standalone: create fresh adapter ──
        method = cfg.get("method", "alrem")
        alpha_mode = cfg.get("lora_alpha_mode", "2r")
        alpha_fixed = cfg.get("lora_alpha_fixed")
        lora_dropout = float(cfg.get("lora_dropout", 0.05))
        r_high = int(cfg.get("r_high", 32))

        if method == "alrem":
            alpha_pattern = build_alpha_pattern(rank_pattern, mode=alpha_mode, fixed=alpha_fixed)
            lora_config = LoraConfig(
                r=r_high,
                lora_alpha=2 * r_high if alpha_mode != "fixed" else int(alpha_fixed or 16),
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                rank_pattern=rank_pattern,
                alpha_pattern=alpha_pattern,
            )
        elif method in ("uniform", "matched"):
            r_use = r_uniform if method == "uniform" else r_match
            lora_alpha = 2 * r_use if alpha_mode != "fixed" else int(alpha_fixed or 16)
            lora_config = LoraConfig(
                r=r_use,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise ValueError("Unsupported method: %s" % method)

        peft_model = get_peft_model(model, lora_config)

    # ── Param summary ──
    base_model_params = count_parameters(model, trainable_only=False)
    trainable_params_total = count_parameters(peft_model, trainable_only=True)
    lora_params_total = compute_total_lora_params(model, peft_model)
    lora_params_by_layer = compute_lora_params_by_layer(peft_model)

    params_payload = {
        "base_model_params": base_model_params,
        "trainable_params_total": trainable_params_total,
        "lora_params_total": lora_params_total,
        "lora_params_by_layer": lora_params_by_layer,
        "alrem_target_params": alrem_target_params,
        "uniform_params": uniform_params,
        "uniform_matched_params": uniform_matched_params,
        "relative_error": rel_error,
        "r_match": r_match,
        "stage1_checkpoint": stage1_checkpoint or "",
    }

    logger.info("Trainable params total: %d", trainable_params_total)
    logger.info("LoRA params total (actual): %d", lora_params_total)

    return peft_model, params_payload


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = _override_config(cfg, args)

    output_dir = cfg.get("output_dir", "outputs")
    run_name = cfg.get("run_name", "run")
    run_dir = os.path.join(output_dir, run_name)
    ensure_dir(run_dir)

    logger = setup_logging(os.path.join(run_dir, "train.log"), logger_name="alrem")
    save_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_name = cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError("model_name_or_path must be set in config.")

    precision = cfg.get("precision", "bf16")
    torch_dtype = _select_dtype(precision)
    quantization = cfg.get("quantization")

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model (with optional quantization) ──
    model = _load_model_with_quantization(
        model_name=model_name,
        quantization=quantization,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if cfg.get("grad_ckpt", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ── Load data ──
    task = cfg.get("task", "mgsm")
    prompt_mgsm = cfg.get("mgsm_prompt", MGSM_PROMPT)
    prompt_flores = cfg.get("flores_prompt", FLORES_PROMPT)

    if task == "mgsm":
        train_examples = load_mgsm_dataset(cfg, cfg.get("train_split", "train"))
        eval_examples = load_mgsm_dataset(cfg, cfg.get("eval_split", "test"))
        train_texts = [build_mgsm_train_text(ex, prompt_template=prompt_mgsm) for ex in train_examples]
        eval_texts = [build_mgsm_train_text(ex, prompt_template=prompt_mgsm) for ex in eval_examples]
        add_special_tokens = True
    elif task == "flores":
        train_examples = load_flores_dataset(cfg, cfg.get("train_split", "train"))
        eval_examples = load_flores_dataset(cfg, cfg.get("eval_split", "dev"))
        train_texts = [build_flores_train_text(ex, prompt_template=prompt_flores) for ex in train_examples]
        eval_texts = [build_flores_train_text(ex, prompt_template=prompt_flores) for ex in eval_examples]
        add_special_tokens = True
    elif task == "sparql":
        train_texts, eval_texts = _load_sparql_data(cfg, tokenizer)
        # Chat template already injects model-specific special tokens.
        add_special_tokens = False
        if cfg.get("test_data_path"):
            logger.info(
                "SPARQL training uses train/dev only (model selection by eval loss). "
                "test_data_path is ignored during training."
            )
    else:
        raise ValueError("Unsupported task: %s" % task)

    max_train_samples = cfg.get("max_train_samples")
    max_eval_samples = cfg.get("max_eval_samples")
    if max_train_samples:
        train_texts = train_texts[: int(max_train_samples)]
    if max_eval_samples:
        eval_texts = eval_texts[: int(max_eval_samples)]

    if len(train_texts) == 0:
        raise ValueError(
            f"No training samples available for task={task}. "
            "Check data paths and required fields."
        )

    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})

    max_seq_len = int(cfg.get("max_seq_len", 1024))
    train_ds = _tokenize_dataset(
        tokenizer,
        train_ds,
        max_seq_len,
        add_special_tokens=add_special_tokens,
    )
    eval_ds = _tokenize_dataset(
        tokenizer,
        eval_ds,
        max_seq_len,
        add_special_tokens=add_special_tokens,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Setup LoRA / load adapter ──
    peft_model, params_payload = _setup_lora(model, cfg, logger)
    save_json(params_payload, os.path.join(run_dir, "params.json"))

    # ── Training arguments (backward compatible field names) ──
    _warn_if_conflicting_cfg(cfg, logger, "per_device_train_batch_size", "batch_size")
    _warn_if_conflicting_cfg(cfg, logger, "gradient_accumulation_steps", "grad_accum")
    _warn_if_conflicting_cfg(cfg, logger, "per_device_eval_batch_size", "eval_batch_size")

    batch_size = _cfg_int(cfg, "per_device_train_batch_size", "batch_size", default=1)
    eval_batch_size = _cfg_int(cfg, "per_device_eval_batch_size", "eval_batch_size", default=batch_size)
    grad_accum = _cfg_int(cfg, "gradient_accumulation_steps", "grad_accum", default=1)
    warmup_ratio = _cfg_float(cfg, "warmup_ratio", default=0.0)
    warmup_steps = _cfg_int(cfg, "warmup_steps", default=0)
    lr = _cfg_float(cfg, "learning_rate", default=2e-4)
    num_epochs = _cfg_float(cfg, "num_train_epochs", default=1.0)
    max_steps = _cfg_int(cfg, "max_steps", default=-1)
    logging_steps = _cfg_int(cfg, "logging_steps", default=10)
    save_steps = _cfg_int(cfg, "save_steps", default=500)
    eval_steps = _cfg_int(cfg, "eval_steps", default=500)
    save_total_limit = _cfg_int(cfg, "save_total_limit", default=2)
    weight_decay = _cfg_float(cfg, "weight_decay", default=0.0)

    eval_strategy = cfg.get("evaluation_strategy", "steps") if len(eval_ds) > 0 else "no"
    save_strategy = cfg.get("save_strategy", "steps")

    load_best = cfg.get("load_best_model_at_end", False)
    # Enable load_best if early stopping is configured
    early_stopping_patience = cfg.get("early_stopping_patience")
    if early_stopping_patience is not None:
        if len(eval_ds) == 0:
            raise ValueError(
                "early_stopping_patience is set, but eval dataset is empty. "
                "Provide dev data or disable early stopping."
            )
        load_best = True

    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio if warmup_ratio > 0 else 0.0,
        warmup_steps=warmup_steps if warmup_ratio == 0 else 0,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        report_to=[],
        run_name=run_name,
        seed=seed,
    )

    # ── Callbacks ──
    callbacks = []
    step_time_log_interval = _cfg_int(
        cfg,
        "step_time_log_interval",
        default=eval_steps if eval_steps > 0 else 10,
    )
    if step_time_log_interval > 0:
        callbacks.append(
            StepTimeLoggingCallback(
                interval_steps=step_time_log_interval,
                logger=logger,
            )
        )
    if early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(early_stopping_patience)
        ))

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    resume_ckpt = cfg.get("resume_from_checkpoint")
    train_start_time = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_elapsed = time.monotonic() - train_start_time
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    # ── Generate run report ──
    report = _build_run_report(
        cfg=cfg,
        run_dir=run_dir,
        params_payload=params_payload,
        train_result=train_result,
        trainer=trainer,
        train_elapsed=train_elapsed,
        train_samples=len(train_ds),
        eval_samples=len(eval_ds),
        task=task,
    )
    save_json(report, os.path.join(run_dir, "run_report.json"))
    _print_report_summary(report, logger)

    logger.info("Training complete. All outputs saved to %s", run_dir)


# ── Run report ────────────────────────────────────────────────────────────────

def _build_run_report(
    cfg: Dict[str, Any],
    run_dir: str,
    params_payload: Dict[str, Any],
    train_result: Any,
    trainer: Any,
    train_elapsed: float,
    train_samples: int,
    eval_samples: int,
    task: str,
) -> Dict[str, Any]:
    """Collect all training metadata into a structured report."""

    # ── Loss history from trainer.state.log_history ──
    log_history = getattr(trainer.state, "log_history", []) or []
    train_losses: List[Dict[str, Any]] = []
    eval_losses: List[Dict[str, Any]] = []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_losses.append({
                "step": entry.get("step"),
                "epoch": round(entry.get("epoch", 0), 4),
                "loss": round(entry["loss"], 6),
                "learning_rate": entry.get("learning_rate"),
            })
        if "eval_loss" in entry:
            eval_losses.append({
                "step": entry.get("step"),
                "epoch": round(entry.get("epoch", 0), 4),
                "eval_loss": round(entry["eval_loss"], 6),
            })

    # ── Best checkpoint ──
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)

    # ── GPU info ──
    gpu_info: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_mem / (1024 ** 3), 2),
            })
        try:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            peak_mb = None
    else:
        peak_mb = None

    # ── Train result metrics ──
    train_metrics: Dict[str, Any] = {}
    if hasattr(train_result, "metrics") and train_result.metrics:
        for k, v in train_result.metrics.items():
            train_metrics[k] = round(v, 6) if isinstance(v, float) else v

    # ── Stage info ──
    stage1_ckpt = cfg.get("stage1_checkpoint")
    stage = "stage2" if stage1_ckpt else "stage1"

    report: Dict[str, Any] = {
        # ── Identity ──
        "run_name": cfg.get("run_name", "run"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "stage": stage,
        "method": cfg.get("method", "unknown"),
        "model": cfg.get("model_name_or_path", ""),

        # ── Data ──
        "data": {
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "train_path": cfg.get("stage2_data_path" if stage1_ckpt else "stage1_data_path", ""),
            "eval_path": cfg.get("stage2_dev_path" if stage1_ckpt else "stage1_dev_path", ""),
        },

        # ── LoRA / Parameters ──
        "lora": {
            "method": cfg.get("method", ""),
            "target_modules": cfg.get("target_modules", []),
            "r_high": cfg.get("r_high"),
            "r_low": cfg.get("r_low"),
            "r_uniform": cfg.get("r_uniform"),
            "quantization": cfg.get("quantization", "none"),
            "lora_params_total": params_payload.get("lora_params_total", 0),
            "trainable_params_total": params_payload.get("trainable_params_total", 0),
            "base_model_params": params_payload.get("base_model_params", 0),
            "trainable_pct": round(
                params_payload.get("trainable_params_total", 0)
                / max(params_payload.get("base_model_params", 1), 1) * 100, 4
            ),
            "r_match": params_payload.get("r_match"),
            "stage1_checkpoint": stage1_ckpt or "",
        },

        # ── Hyperparameters ──
        "hyperparameters": {
            "learning_rate": cfg.get("learning_rate"),
            "num_train_epochs": cfg.get("num_train_epochs"),
            "batch_size": cfg.get("per_device_train_batch_size", cfg.get("batch_size")),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", cfg.get("grad_accum")),
            "effective_batch_size": (
                _cfg_int(cfg, "per_device_train_batch_size", "batch_size", default=1)
                * _cfg_int(cfg, "gradient_accumulation_steps", "grad_accum", default=1)
            ),
            "warmup_ratio": cfg.get("warmup_ratio"),
            "max_seq_len": cfg.get("max_seq_len"),
            "seed": cfg.get("seed"),
            "precision": cfg.get("precision", "bf16"),
            "grad_ckpt": cfg.get("grad_ckpt", False),
        },

        # ── Training results ──
        "training": {
            "wall_time_sec": round(train_elapsed, 1),
            "wall_time_min": round(train_elapsed / 60, 2),
            "total_steps": train_metrics.get("train_steps", getattr(trainer.state, "global_step", 0)),
            "final_train_loss": train_metrics.get("train_loss"),
            "final_eval_loss": eval_losses[-1]["eval_loss"] if eval_losses else None,
            "best_checkpoint": best_ckpt,
            "best_metric": round(best_metric, 6) if isinstance(best_metric, float) else best_metric,
            "early_stopped": bool(
                cfg.get("early_stopping_patience") is not None
                and int(getattr(trainer.state, "max_steps", 0) or 0) > 0
                and int(getattr(trainer.state, "global_step", 0) or 0)
                < int(getattr(trainer.state, "max_steps", 0) or 0)
            ),
            "metrics": train_metrics,
        },

        # ── Loss curves ──
        "loss_history": {
            "train": train_losses,
            "eval": eval_losses,
        },

        # ── Environment ──
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": gpu_info,
            "peak_gpu_memory_mb": round(peak_mb, 1) if peak_mb else None,
        },

        "output_dir": run_dir,
    }

    return report


def _print_report_summary(report: Dict[str, Any], logger: logging.Logger) -> None:
    """Print a human-readable summary block to the log."""
    sep = "=" * 64
    logger.info(sep)
    logger.info("  RUN REPORT: %s", report.get("run_name", "?"))
    logger.info(sep)

    logger.info("  Task:   %s  |  Stage: %s  |  Method: %s",
                report["task"], report["stage"], report["lora"]["method"])
    logger.info("  Model:  %s", report["model"])
    if report["lora"]["stage1_checkpoint"]:
        logger.info("  Stage1: %s", report["lora"]["stage1_checkpoint"])

    logger.info("-" * 64)
    data = report["data"]
    logger.info("  Data    train=%d  eval=%d", data["train_samples"], data["eval_samples"])

    lora = report["lora"]
    logger.info("  LoRA    trainable=%s (%.4f%% of %s)  quant=%s",
                f"{lora['trainable_params_total']:,}",
                lora["trainable_pct"],
                f"{lora['base_model_params']:,}",
                lora["quantization"])
    if lora["method"] == "alrem":
        logger.info("          r_high=%s  r_low=%s", lora["r_high"], lora["r_low"])
    elif lora["method"] in ("uniform", "matched"):
        r_val = lora["r_match"] if lora["method"] == "matched" else lora["r_uniform"]
        logger.info("          r=%s", r_val)

    logger.info("-" * 64)
    hp = report["hyperparameters"]
    logger.info("  Hyper   lr=%s  epochs=%s  eff_batch=%d  seq_len=%s",
                hp["learning_rate"], hp["num_train_epochs"],
                hp["effective_batch_size"], hp["max_seq_len"])

    logger.info("-" * 64)
    tr = report["training"]
    logger.info("  Training  steps=%s  time=%.1f min",
                tr["total_steps"], tr["wall_time_min"])
    if tr["final_train_loss"] is not None:
        logger.info("  Loss      train=%.6f", tr["final_train_loss"])
    if tr["final_eval_loss"] is not None:
        logger.info("            eval=%.6f", tr["final_eval_loss"])
    if tr["best_checkpoint"]:
        logger.info("  Best      %s (metric=%.6f)",
                    tr["best_checkpoint"],
                    tr["best_metric"] if isinstance(tr["best_metric"], (int, float)) else 0)

    env = report["environment"]
    if env["gpu"]:
        gpu = env["gpu"][0]
        logger.info("-" * 64)
        logger.info("  GPU       %s (%.1f GB)", gpu["name"], gpu["total_memory_gb"])
        if env["peak_gpu_memory_mb"]:
            logger.info("  Peak mem  %.1f MB", env["peak_gpu_memory_mb"])

    logger.info(sep)
    logger.info("  Report saved: %s/run_report.json", report["output_dir"])
    logger.info(sep)


if __name__ == "__main__":
    main()
