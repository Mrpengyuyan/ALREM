"""ICL baseline generation for multilingual Text-to-SPARQL.

This script generates predictions without loading any LoRA adapter.
Outputs are written in the same schema as adapter evaluation:
  - predictions.jsonl: idx, qid, language, question, gold_sparql, pred_sparql,
    generation_time_sec, mode, run_id, protocol_id
  - run_metadata.json: run/protocol/decode metadata for unified evaluation auditing
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from .prompts import build_sparql_infer_text
from .run_identity import (
    build_run_id,
    validate_protocol_id,
    validate_result_partition,
    validate_run_id,
)
from .utils import ensure_dir, load_yaml, save_json, set_seed, setup_logging

LOGGER = logging.getLogger("alrem.run_icl")

DEFAULT_EVAL_PROTOCOL_PATH = str(
    Path(__file__).resolve().parents[1] / "configs" / "sparql_eval_shared.yaml"
)
ALLOWED_PRED_MODES = {"adapter", "icl_zero", "icl_fewshot"}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skip invalid JSON at %s:%d (%s)", path, line_no, exc)
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _compute_protocol_id(protocol_cfg: Dict[str, Any]) -> str:
    protocol_id = str(protocol_cfg.get("protocol_id", "")).strip()
    protocol_name = str(protocol_cfg.get("protocol_name", "")).strip()
    protocol_version = str(protocol_cfg.get("protocol_version", "")).strip()
    derived = ""
    if protocol_name and protocol_version:
        derived = f"{protocol_name}:{protocol_version}"
    elif protocol_name:
        derived = protocol_name
    elif protocol_cfg:
        derived = "eval_protocol"

    if protocol_id and derived and protocol_id != derived:
        raise ValueError(
            "Eval protocol_id mismatch with protocol_name/protocol_version: "
            f"protocol_id={protocol_id} derived={derived}"
        )
    resolved = protocol_id or derived
    if not resolved:
        return ""
    return validate_protocol_id(resolved)


def _parse_language_list(csv_text: Optional[str]) -> Optional[List[str]]:
    if csv_text is None:
        return None
    tokens = [token.strip().lower() for token in csv_text.split(",")]
    langs = [token for token in tokens if token]
    return langs if langs else None


def _normalize_language_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_language_list(value)
    if isinstance(value, list):
        langs = [str(token).strip().lower() for token in value if str(token).strip()]
        return langs if langs else None
    return None


def _resolve_bool(cli_flag: bool, cfg_value: Any, protocol_value: Any, default: bool = False) -> bool:
    if cli_flag:
        return True
    if cfg_value is not None:
        return bool(cfg_value)
    if protocol_value is not None:
        return bool(protocol_value)
    return default


def _filter_by_languages(
    records: List[Dict[str, Any]],
    test_languages: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if not test_languages:
        return records
    allowed = set(test_languages)
    return [
        sample
        for sample in records
        if str(sample.get("language", "unk")).strip().lower() in allowed
    ]


def _normalize_mode(mode: str) -> str:
    lower = mode.strip().lower()
    if lower in {"few", "fewshot", "few_shot", "icl_fewshot"}:
        return "few"
    if lower in {"zero", "zeroshot", "zero_shot", "icl_zero"}:
        return "zero"
    raise ValueError(f"Unsupported ICL mode: {mode}. Expected 'zero' or 'few'.")


def _validate_few_shot_pool_path(pool_path: str, test_data_path: str) -> None:
    pool = Path(pool_path).resolve()
    test_data = Path(test_data_path).resolve()
    if not pool.exists():
        raise FileNotFoundError(f"few_shot_pool path not found: {pool}")
    if pool == test_data:
        raise ValueError("few_shot_pool must not be the same file as test_data.")
    if "test" in pool.name.lower():
        raise ValueError(
            "few_shot_pool appears to be a test file. "
            "Please use train/dev data only."
        )


def _build_few_shot_pool(pool_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    pool: List[Dict[str, str]] = []
    for item in pool_data:
        question = str(item.get("question", "")).strip()
        sparql = str(item.get("sparql", "")).strip()
        if not question or not sparql:
            continue
        pool.append(
            {
                "question": question,
                "sparql": sparql,
                "qid": str(item.get("qid", "")).strip(),
            }
        )
    if not pool:
        raise ValueError("few_shot_pool has no valid question/sparql pairs.")
    return pool


def _collect_non_empty_qids(records: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for item in records:
        qid = str(item.get("qid", "")).strip()
        if qid:
            out.add(qid)
    return out


def _normalize_signature_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _collect_qa_signatures(records: List[Dict[str, Any]]) -> Set[str]:
    signatures: Set[str] = set()
    for item in records:
        question = _normalize_signature_text(item.get("question", ""))
        sparql = _normalize_signature_text(item.get("sparql", ""))
        if not question or not sparql:
            continue
        digest = hashlib.md5(f"{question}\t{sparql}".encode("utf-8")).hexdigest()
        signatures.add(digest)
    return signatures


def _validate_no_qid_overlap(pool_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> None:
    pool_qids = _collect_non_empty_qids(pool_data)
    test_qids = _collect_non_empty_qids(test_data)
    overlap_qids = sorted(pool_qids & test_qids) if pool_qids and test_qids else []
    if overlap_qids:
        preview = ", ".join(overlap_qids[:10])
        suffix = "..." if len(overlap_qids) > 10 else ""
        raise ValueError(
            "few_shot_pool has qid overlap with test_data (data leakage risk). "
            f"overlap_count={len(overlap_qids)} overlap_qids={preview}{suffix}"
        )

    pool_signatures = _collect_qa_signatures(pool_data)
    test_signatures = _collect_qa_signatures(test_data)
    overlap_signatures = pool_signatures & test_signatures
    if overlap_signatures:
        overlap_preview = sorted(overlap_signatures)
        preview = ", ".join(overlap_preview[:10])
        suffix = "..." if len(overlap_signatures) > 10 else ""
        raise ValueError(
            "few_shot_pool overlaps test_data on normalized (question,sparql) pairs "
            "(data leakage risk when qid is missing). "
            f"overlap_count={len(overlap_signatures)} overlap_hashes={preview}{suffix}"
        )


def _validate_decoding_args(
    *,
    max_seq_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> None:
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if do_sample:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 when do_sample=True, got {temperature}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1] when do_sample=True, got {top_p}")


def _sample_few_shot_examples(
    pool: List[Dict[str, str]],
    current_qid: str,
    forbidden_qids: Set[str],
    k: int,
    seed: int,
    sample_index: int,
) -> List[Dict[str, str]]:
    if k <= 0:
        return []
    blocked = set(forbidden_qids)
    if current_qid:
        blocked.add(current_qid)
    candidates = [item for item in pool if item.get("qid", "") not in blocked]
    if not candidates:
        return []
    rng = random.Random(seed + sample_index)
    if len(candidates) <= k:
        picked = list(candidates)
        rng.shuffle(picked)
        return [{"question": p["question"], "sparql": p["sparql"]} for p in picked]
    picked = rng.sample(candidates, k)
    return [{"question": p["question"], "sparql": p["sparql"]} for p in picked]


def _infer_generation_device(model: Any) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_base_model_for_icl(
    model_name: str,
    quantization: Optional[str] = "4bit",
    precision: str = "bf16",
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    LOGGER.info("Loaded base model for ICL: %s", model_name)
    return model, tokenizer


def generate_sparql(
    model: Any,
    tokenizer: Any,
    question: str,
    max_seq_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    from transformers import GenerationConfig

    prompt = build_sparql_infer_text(
        question=question,
        tokenizer=tokenizer,
        few_shot_examples=few_shot_examples,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )
    input_device = _infer_generation_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_config)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICL baseline generation for SPARQL.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--eval_protocol", type=str, default=None, help="Shared eval protocol YAML path.")
    parser.add_argument("--model_name", type=str, default=None, help="Base model name.")
    parser.add_argument("--test_data", type=str, default=None, help="Test JSONL path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Run directory name.")
    parser.add_argument("--mode", type=str, default=None, help="ICL mode: zero or few.")
    parser.add_argument("--few_shot_pool", type=str, default=None, help="Few-shot pool JSONL path.")
    parser.add_argument("--few_shot_k", type=int, default=None, help="Number of shots for few-shot mode.")
    parser.add_argument("--few_shot_seed", type=int, default=None, help="Seed for few-shot sampling.")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization: 4bit or none.")
    parser.add_argument("--precision", type=str, default=None, help="bf16/fp16/fp32.")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Input max sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Generation max new tokens.")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--test_languages", type=str, default=None, help="Comma-separated language filter.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of test samples.")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config) or {}

    eval_protocol_arg = getattr(args, "eval_protocol", None)
    eval_protocol_path = eval_protocol_arg or cfg.get("eval_protocol_path") or DEFAULT_EVAL_PROTOCOL_PATH
    if not eval_protocol_path:
        raise ValueError("Eval protocol path is required.")
    protocol_file = Path(eval_protocol_path)
    if not protocol_file.exists():
        raise FileNotFoundError(f"Eval protocol file not found: {eval_protocol_path}")

    protocol_cfg: Dict[str, Any] = load_yaml(str(protocol_file)) or {}
    eval_protocol_loaded_path = str(protocol_file)
    strict_schema = bool(protocol_cfg.get("strict_schema", True))
    protocol_name = str(protocol_cfg.get("protocol_name", "")).strip()
    protocol_version = str(protocol_cfg.get("protocol_version", "")).strip()
    protocol_id = _compute_protocol_id(protocol_cfg)
    if strict_schema and "protocol_id" not in protocol_cfg:
        raise ValueError("strict_schema=True requires protocol_id in eval protocol config.")
    main_table_protocol = bool(protocol_cfg.get("main_table_protocol", False))
    if main_table_protocol and not strict_schema:
        raise ValueError("main_table_protocol=true requires strict_schema=true.")
    protocol_result_partition = validate_result_partition(
        protocol_cfg.get("result_partition"),
        strict_schema=strict_schema,
    )

    enforce_protocol = bool(protocol_cfg.get("enforce_protocol", True))
    protocol_languages = _normalize_language_list(protocol_cfg.get("test_languages"))

    model_name = args.model_name or cfg.get("model_name_or_path") or protocol_cfg.get("model_name_or_path")
    test_data_path = args.test_data or cfg.get("test_data_path") or protocol_cfg.get("test_data_path")
    output_root = args.output_dir or cfg.get("output_dir", "outputs")
    mode = _normalize_mode(args.mode or cfg.get("icl_mode", "zero"))
    run_name = args.run_name or cfg.get("run_name", f"sparql_icl_{mode}")
    quantization = args.quantization or cfg.get("quantization", protocol_cfg.get("quantization", "4bit"))
    precision = args.precision or cfg.get("precision", "bf16")
    max_seq_len = int(
        args.max_seq_len if args.max_seq_len is not None else cfg.get("max_seq_len", protocol_cfg.get("max_seq_len", 512))
    )
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else cfg.get("max_new_tokens", protocol_cfg.get("max_new_tokens", 256))
    )
    do_sample = _resolve_bool(
        cli_flag=args.do_sample,
        cfg_value=cfg.get("do_sample"),
        protocol_value=protocol_cfg.get("do_sample"),
        default=False,
    )
    temperature = float(
        args.temperature if args.temperature is not None else cfg.get("temperature", protocol_cfg.get("temperature", 1.0))
    )
    top_p = float(
        args.top_p if args.top_p is not None else cfg.get("top_p", protocol_cfg.get("top_p", 1.0))
    )
    _validate_decoding_args(
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    few_shot_k = int(args.few_shot_k if args.few_shot_k is not None else cfg.get("few_shot_k", 4))
    few_shot_seed = int(args.few_shot_seed if args.few_shot_seed is not None else cfg.get("few_shot_seed", 42))
    test_languages = _parse_language_list(args.test_languages)
    if test_languages is None:
        test_languages = _normalize_language_list(cfg.get("test_languages"))
    if test_languages is None:
        test_languages = protocol_languages
    if enforce_protocol and protocol_languages is not None:
        if test_languages is None or set(test_languages) != set(protocol_languages):
            raise ValueError(
                "test_languages mismatch with eval protocol. "
                f"protocol={sorted(protocol_languages)} current={sorted(test_languages or [])}"
            )
    if strict_schema and not test_languages:
        raise ValueError("strict_schema=True requires explicit test_languages.")
    if enforce_protocol:
        protocol_task = str(protocol_cfg.get("task", "")).strip().lower()
        cfg_task = str(cfg.get("task", "sparql")).strip().lower()
        if protocol_task and cfg_task and protocol_task != cfg_task:
            raise ValueError(f"Task mismatch between config ({cfg_task}) and eval protocol ({protocol_task}).")
        decode_mismatches = []
        for key, current in (
            ("max_seq_len", max_seq_len),
            ("max_new_tokens", max_new_tokens),
            ("do_sample", do_sample),
            ("temperature", temperature),
            ("top_p", top_p),
        ):
            if key in protocol_cfg and protocol_cfg.get(key) != current:
                decode_mismatches.append((key, protocol_cfg.get(key), current))
        if decode_mismatches:
            mismatch_text = ", ".join(f"{k}: protocol={p} current={c}" for k, p, c in decode_mismatches)
            raise ValueError(f"Decoding config mismatch with eval protocol: {mismatch_text}")
        if "test_data_path" in protocol_cfg and protocol_cfg.get("test_data_path") != test_data_path:
            raise ValueError(
                "test_data_path mismatch with eval protocol: "
                f"protocol={protocol_cfg.get('test_data_path')} current={test_data_path}"
            )
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    if not model_name:
        raise ValueError("model_name is required (via --model_name or config model_name_or_path).")
    if not test_data_path:
        raise ValueError("test_data is required (via --test_data or config test_data_path).")
    if not Path(test_data_path).exists():
        raise FileNotFoundError(f"test_data not found: {test_data_path}")

    test_data = _load_jsonl(test_data_path)

    few_shot_pool_path = args.few_shot_pool or cfg.get("few_shot_pool_path")
    few_shot_pool: List[Dict[str, str]] = []
    test_qids: Set[str] = _collect_non_empty_qids(test_data)
    if mode == "few":
        if not few_shot_pool_path:
            raise ValueError("few-shot mode requires few_shot_pool_path.")
        _validate_few_shot_pool_path(few_shot_pool_path, test_data_path)
        pool_data = _load_jsonl(few_shot_pool_path)
        _validate_no_qid_overlap(pool_data=pool_data, test_data=test_data)
        few_shot_pool = _build_few_shot_pool(pool_data)

    run_dir = Path(output_root) / run_name
    ensure_dir(str(run_dir))
    logger = setup_logging(str(run_dir / "run_icl.log"), logger_name="alrem")
    set_seed(seed)

    logger.info("ICL mode: %s", mode)
    logger.info("Model: %s", model_name)
    logger.info("Test data: %s", test_data_path)
    logger.info("Run dir: %s", run_dir)
    logger.info("Eval protocol: %s", eval_protocol_loaded_path or "(none)")
    if protocol_id:
        logger.info("Protocol id: %s", protocol_id)
    logger.info("Result partition: %s", protocol_result_partition)
    logger.info(
        "Decoding: max_seq_len=%d max_new_tokens=%d do_sample=%s temperature=%.3f top_p=%.3f",
        max_seq_len,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
    )
    if mode == "few":
        logger.info(
            "Few-shot settings: pool=%s k=%d seed=%d pool_size=%d",
            few_shot_pool_path,
            few_shot_k,
            few_shot_seed,
            len(few_shot_pool),
        )

    if args.max_samples:
        test_data = test_data[: args.max_samples]
    if cfg.get("max_samples") is not None and args.max_samples is None:
        test_data = test_data[: int(cfg["max_samples"])]

    before = len(test_data)
    test_data = _filter_by_languages(test_data, test_languages)
    if test_languages:
        logger.info(
            "Applied language filter %s: %d -> %d",
            sorted(set(test_languages)),
            before,
            len(test_data),
        )
    if not test_data:
        raise ValueError("No test samples available after filtering.")

    model, tokenizer = load_base_model_for_icl(
        model_name=model_name,
        quantization=quantization,
        precision=precision,
    )

    prediction_mode = "icl_zero" if mode == "zero" else "icl_fewshot"
    if prediction_mode not in ALLOWED_PRED_MODES:
        raise ValueError(f"Unsupported prediction mode generated: {prediction_mode}")
    run_id = build_run_id(
        run_name=run_name,
        mode=prediction_mode,
        protocol_id=protocol_id,
        seed=seed,
    )
    validate_run_id(run_id)
    predictions: List[Dict[str, Any]] = []
    total = len(test_data)
    for idx, sample in enumerate(test_data):
        question = str(sample.get("question", "")).strip()
        gold_sparql = str(sample.get("sparql", "")).strip()
        language = str(sample.get("language", "unk")).strip() or "unk"
        qid = str(sample.get("qid", "")).strip()

        if not question:
            LOGGER.warning("Skip sample %d/%d: empty question", idx + 1, total)
            continue

        few_shot_examples: Optional[List[Dict[str, str]]] = None
        if mode == "few":
            few_shot_examples = _sample_few_shot_examples(
                pool=few_shot_pool,
                current_qid=qid,
                forbidden_qids=test_qids,
                k=few_shot_k,
                seed=few_shot_seed,
                sample_index=idx,
            )

        start_time = time.monotonic()
        pred_sparql = generate_sparql(
            model=model,
            tokenizer=tokenizer,
            question=question,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            few_shot_examples=few_shot_examples,
        )
        elapsed = time.monotonic() - start_time

        predictions.append(
            {
                "idx": idx,
                "qid": qid,
                "language": language,
                "question": question,
                "gold_sparql": gold_sparql,
                "pred_sparql": pred_sparql,
                "generation_time_sec": round(elapsed, 3),
                "mode": prediction_mode,
                "run_id": run_id,
                "protocol_id": protocol_id,
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            logger.info("Generated %d/%d", idx + 1, total)

    preds_path = run_dir / "predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved predictions: %s (%d rows)", preds_path, len(predictions))

    run_report = {
        "run_id": run_id,
        "mode": prediction_mode,
        "model_name_or_path": model_name,
        "test_data_path": test_data_path,
        "few_shot_pool_path": few_shot_pool_path if mode == "few" else None,
        "few_shot_k": few_shot_k if mode == "few" else 0,
        "few_shot_seed": few_shot_seed if mode == "few" else None,
        "num_test_samples": len(test_data),
        "num_predictions": len(predictions),
        "quantization": quantization,
        "precision": precision,
        "max_seq_len": max_seq_len,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "test_languages": test_languages,
        "output_dir": str(run_dir),
        "eval_protocol": {
            "path": eval_protocol_loaded_path,
            "protocol_name": protocol_name,
            "protocol_version": protocol_version,
            "protocol_id": protocol_id,
            "enforce_protocol": enforce_protocol,
            "strict_schema": strict_schema,
            "main_table_protocol": main_table_protocol,
            "result_partition": protocol_result_partition,
            "decoding": {
                "max_seq_len": max_seq_len,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            },
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_report, str(run_dir / "run_report.json"))
    logger.info("Saved run report: %s", run_dir / "run_report.json")

    run_metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "mode": prediction_mode,
        "method": prediction_mode,
        "task": "sparql",
        "model_name_or_path": model_name,
        "adapter_path": None,
        "test_data_path": test_data_path,
        "test_languages": test_languages or [],
        "max_seq_len": max_seq_len,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "cache_dir": cfg.get("sparql_cache_dir") or protocol_cfg.get("cache_dir", "data/sparql/cache"),
        "offline_only": _resolve_bool(
            cli_flag=False,
            cfg_value=cfg.get("offline_only"),
            protocol_value=protocol_cfg.get("offline_only"),
            default=False,
        ),
        "result_partition": protocol_result_partition,
        "protocol_name": protocol_name,
        "protocol_version": protocol_version,
        "protocol_id": protocol_id,
        "seed": seed,
        "timestamp": run_report["timestamp_utc"],
        "timestamp_utc": run_report["timestamp_utc"],
        "output_dir": str(run_dir),
    }
    validate_protocol_id(str(run_metadata.get("protocol_id", "")))
    validate_run_id(str(run_metadata.get("run_id", "")))
    expected_run_id = build_run_id(
        run_name=str(run_metadata.get("run_name", "")),
        mode=str(run_metadata.get("mode", "")),
        protocol_id=str(run_metadata.get("protocol_id", "")),
        seed=int(run_metadata.get("seed")),
    )
    if strict_schema and run_metadata["run_id"] != expected_run_id:
        raise ValueError(
            "run_metadata run_id does not follow naming rule: "
            f"expected={expected_run_id} actual={run_metadata['run_id']}"
        )
    save_json(run_metadata, str(run_dir / "run_metadata.json"))
    logger.info("Saved run metadata: %s", run_dir / "run_metadata.json")


if __name__ == "__main__":
    main()
