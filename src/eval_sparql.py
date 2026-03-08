"""SPARQL evaluation pipeline.

Loads a trained adapter, generates SPARQL queries for the test set,
executes them via SPARQLCache, and computes:
  1. Execution Accuracy (EA)
  2. Executable Rate
  3. Normalized Exact Match (NormEM)
  4. Answer-level F1
  5. Cross-lingual Consistency (CLC-Ans / CLC-Struct)
  6. Error classification
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .prompts import build_sparql_infer_text
from .run_identity import (
    build_run_id,
    validate_protocol_id,
    validate_result_partition,
    validate_run_id,
)
from .sparql_executor import SPARQLCache
from .utils import ensure_dir, load_yaml, save_json, set_seed, setup_logging

LOGGER = logging.getLogger("alrem.eval_sparql")

DEFAULT_EVAL_PROTOCOL_PATH = str(
    Path(__file__).resolve().parents[1] / "configs" / "sparql_eval_shared.yaml"
)
ALLOWED_ERROR_TYPES = {
    "generation_empty",
    "syntax_or_parse_error",
    "execution_error",
    "wrong_answer",
}
ALLOWED_PRED_MODES = {"adapter", "icl_zero", "icl_fewshot"}
SUPPORTED_PRIMARY_METRICS = {"EA", "ER", "CLC-Ans", "CLC-Struct"}
SUPPORTED_AUX_METRICS = {"NormEM", "AnswerF1"}
PARSE_ERROR_HINTS = (
    "parse",
    "syntax",
    "malformed",
    "lexical",
    "bad query",
    "unexpected token",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skip invalid JSON at %s:%d (%s)", path, line_no, exc)
    return records


def _load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}.")
    return obj


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


def _resolve_allowed_error_types(protocol_cfg: Dict[str, Any], strict_schema: bool) -> Set[str]:
    configured = protocol_cfg.get("allowed_error_types")
    if configured is None:
        return set(ALLOWED_ERROR_TYPES)
    parsed = {str(item).strip() for item in configured if str(item).strip()}
    if strict_schema and parsed != ALLOWED_ERROR_TYPES:
        raise ValueError(
            "allowed_error_types in eval protocol must exactly match fixed schema. "
            f"expected={sorted(ALLOWED_ERROR_TYPES)} configured={sorted(parsed)}"
        )
    if not parsed:
        return set(ALLOWED_ERROR_TYPES)
    unknown = parsed - ALLOWED_ERROR_TYPES
    if unknown:
        msg = f"Unsupported allowed_error_types in protocol: {sorted(unknown)}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning("%s; fallback to fixed schema.", msg)
        return set(ALLOWED_ERROR_TYPES)
    return parsed


def _resolve_metric_names(
    protocol_cfg: Dict[str, Any],
    *,
    strict_schema: bool,
) -> Tuple[List[str], List[str]]:
    configured_primary = protocol_cfg.get("primary_metrics") or sorted(SUPPORTED_PRIMARY_METRICS)
    configured_aux = protocol_cfg.get("aux_metrics") or sorted(SUPPORTED_AUX_METRICS)

    primary = [str(item).strip() for item in configured_primary if str(item).strip()]
    aux = [str(item).strip() for item in configured_aux if str(item).strip()]

    bad_primary = sorted(set(primary) - SUPPORTED_PRIMARY_METRICS)
    bad_aux = sorted(set(aux) - SUPPORTED_AUX_METRICS)
    if bad_primary or bad_aux:
        msg = (
            "Unsupported metric names in eval protocol: "
            f"primary={bad_primary} aux={bad_aux}"
        )
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning("%s; fallback to supported defaults.", msg)
        return sorted(SUPPORTED_PRIMARY_METRICS), sorted(SUPPORTED_AUX_METRICS)
    return primary, aux


def _cleanup_internal_prediction_fields(records: List[Dict[str, Any]]) -> None:
    for record in records:
        internal_keys = [key for key in record.keys() if key.startswith("__")]
        for key in internal_keys:
            record.pop(key, None)


def _find_default_run_metadata_path(predictions_file: str) -> str:
    return str(Path(predictions_file).resolve().parent / "run_metadata.json")


def _normalize_sparql(sparql: str) -> str:
    """Normalize SPARQL for exact-match comparison."""
    text = sparql.strip()
    # Remove markdown code fences if present
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    # Collapse whitespace
    text = " ".join(text.split())
    # Lowercase keywords (but preserve URIs/literals)
    for kw in ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE", "WHERE", "FILTER",
               "OPTIONAL", "UNION", "ORDER", "GROUP", "HAVING", "LIMIT",
               "OFFSET", "DISTINCT", "REDUCED", "BY", "AS", "VALUES",
               "BIND", "SERVICE", "PREFIX", "BASE", "INSERT", "DELETE",
               "NOT", "EXISTS", "MINUS", "IN", "COUNT", "SUM", "AVG",
               "MIN", "MAX", "SAMPLE", "GROUP_CONCAT", "SEPARATOR",
               "YEAR", "MONTH", "DAY", "STR", "LANG", "LANGMATCHES",
               "DATATYPE", "BOUND", "SAMETERM", "ISIRI", "ISURI",
               "ISBLANK", "ISLITERAL", "REGEX", "REPLACE", "SUBSTR",
               "STRLEN", "UCASE", "LCASE", "ENCODE_FOR_URI", "CONTAINS",
               "STRSTARTS", "STRENDS", "STRBEFORE", "STRAFTER",
               "CONCAT", "COALESCE", "IF", "NOW", "UUID", "STRUUID",
               "MD5", "SHA1", "SHA256", "SHA384", "SHA512", "ASC", "DESC"):
        text = re.sub(rf"\b{kw}\b", kw, text, flags=re.IGNORECASE)
    return text


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


def _is_syntax_or_parse_error(message: str, error_type: str = "") -> bool:
    text = f"{error_type} {message}".strip().lower()
    return any(hint in text for hint in PARSE_ERROR_HINTS)


def _build_error_detail(
    *,
    stage: str,
    code: str,
    message: str = "",
    raw_error_type: str = "",
    raw_exception: str = "",
    cache_related: bool = False,
    pred_executable: bool = False,
    gold_executable: bool = False,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "code": code,
        "message": message,
        "raw_error_type": raw_error_type,
        "raw_exception": raw_exception,
        "cache_related": bool(cache_related),
        "pred_executable": bool(pred_executable),
        "gold_executable": bool(gold_executable),
    }


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


def _to_str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_prediction_record(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    gold_sparql = _to_str_or_empty(record.get("gold_sparql"))
    if not gold_sparql:
        gold_sparql = _to_str_or_empty(record.get("sparql"))
    pred_sparql = _to_str_or_empty(record.get("pred_sparql"))
    if not pred_sparql:
        pred_sparql = _to_str_or_empty(record.get("prediction"))
    if not gold_sparql:
        LOGGER.warning("Skip prediction #%d: missing gold SPARQL", idx)
        return None

    try:
        generation_time = float(record.get("generation_time_sec", 0.0))
    except (TypeError, ValueError):
        generation_time = 0.0

    try:
        norm_idx = int(record.get("idx", idx))
    except (TypeError, ValueError):
        norm_idx = idx

    raw_mode = _to_str_or_empty(record.get("mode"))
    raw_run_id = _to_str_or_empty(record.get("run_id"))
    raw_protocol_id = _to_str_or_empty(record.get("protocol_id"))

    normalized: Dict[str, Any] = {
        "idx": norm_idx,
        "qid": _to_str_or_empty(record.get("qid")) or _to_str_or_empty(record.get("id")),
        "language": _to_str_or_empty(record.get("language")) or "unk",
        "question": _to_str_or_empty(record.get("question")),
        "gold_sparql": gold_sparql,
        "pred_sparql": pred_sparql,
        "generation_time_sec": round(generation_time, 3),
        "mode": raw_mode or "adapter",
        "run_id": raw_run_id,
        "protocol_id": raw_protocol_id,
        "__missing_mode": not bool(raw_mode),
        "__missing_run_id": not bool(raw_run_id),
        "__missing_protocol_id": not bool(raw_protocol_id),
    }
    return normalized


def _load_predictions_jsonl(path: str) -> List[Dict[str, Any]]:
    raw_records = _load_jsonl(path)
    normalized: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        normalized_record = _normalize_prediction_record(record, idx)
        if normalized_record is None:
            continue
        normalized.append(normalized_record)
    if not normalized:
        raise ValueError(
            f"No valid prediction records loaded from {path}. "
            "Expected fields include gold_sparql and pred_sparql (or prediction)."
        )
    return normalized


def _validate_predictions_schema(
    records: List[Dict[str, Any]],
    *,
    strict_schema: bool,
    expected_protocol_id: str,
    expected_run_id: str,
    allowed_modes: Set[str],
) -> None:
    if not records:
        raise ValueError("No prediction records available for schema validation.")

    invalid_modes = []
    missing_mode = 0
    missing_run_id = 0
    missing_protocol = 0
    mismatch_protocol = 0
    mismatch_run_id = 0
    invalid_run_ids: List[Tuple[int, str]] = []
    for idx, record in enumerate(records):
        mode_raw = str(record.get("mode", "")).strip().lower()
        run_id_raw = str(record.get("run_id", "")).strip()
        if bool(record.get("__missing_mode", "mode" not in record)) or not mode_raw:
            missing_mode += 1
        if bool(record.get("__missing_run_id", "run_id" not in record)) or not run_id_raw:
            missing_run_id += 1
        elif run_id_raw:
            try:
                validate_run_id(run_id_raw)
            except ValueError:
                invalid_run_ids.append((idx, run_id_raw))
        if expected_run_id and run_id_raw and run_id_raw != expected_run_id:
            mismatch_run_id += 1
        if mode_raw and mode_raw not in allowed_modes:
            invalid_modes.append((idx, mode_raw))
        protocol_id = str(record.get("protocol_id", "")).strip()
        if expected_protocol_id:
            if not protocol_id:
                missing_protocol += 1
            elif protocol_id != expected_protocol_id:
                mismatch_protocol += 1

    if invalid_modes:
        sample = ", ".join(f"{i}:{m}" for i, m in invalid_modes[:5])
        msg = f"Found unsupported prediction modes: {sample}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    if missing_mode:
        msg = f"Prediction records missing mode: {missing_mode}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    if missing_run_id:
        msg = f"Prediction records missing run_id: {missing_run_id}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    if invalid_run_ids:
        sample = ", ".join(f"{i}:{v}" for i, v in invalid_run_ids[:5])
        msg = f"Prediction records have invalid run_id format: {sample}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    if expected_run_id and mismatch_run_id:
        msg = f"Prediction run_id mismatch: mismatch_count={mismatch_run_id} expected={expected_run_id}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    if expected_protocol_id and (missing_protocol or mismatch_protocol):
        msg = (
            "Prediction protocol_id mismatch: "
            f"missing={missing_protocol} mismatch={mismatch_protocol} expected={expected_protocol_id}"
        )
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)


# ── Model loading ────────────────────────────────────────────────────────────

def _validate_run_metadata_schema(
    metadata: Dict[str, Any],
    *,
    strict_schema: bool,
    enforce_protocol: bool,
    expected_protocol_id: str,
    expected_result_partition: str,
    allowed_modes: Set[str],
    protocol_cfg: Dict[str, Any],
    predictions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    required_fields = {
        "run_id",
        "run_name",
        "mode",
        "method",
        "model_name_or_path",
        "adapter_path",
        "test_data_path",
        "test_languages",
        "max_seq_len",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "cache_dir",
        "offline_only",
        "result_partition",
        "protocol_name",
        "protocol_version",
        "protocol_id",
        "seed",
    }
    missing_required = sorted(k for k in required_fields if k not in metadata)
    if missing_required:
        msg = f"run_metadata missing required fields: {missing_required}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    mode = str(metadata.get("mode", "")).strip().lower()
    if mode and mode not in allowed_modes:
        msg = f"run_metadata mode is unsupported: {mode}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    metadata_protocol_id = str(metadata.get("protocol_id", "")).strip()
    if metadata_protocol_id:
        try:
            validate_protocol_id(metadata_protocol_id)
        except ValueError as exc:
            if strict_schema:
                raise
            LOGGER.warning("%s", exc)

    if expected_protocol_id and metadata_protocol_id != expected_protocol_id:
        msg = (
            "run_metadata protocol_id mismatch: "
            f"expected={expected_protocol_id} actual={metadata_protocol_id or '<missing>'}"
        )
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    metadata_result_partition = ""
    try:
        metadata_result_partition = validate_result_partition(
            metadata.get("result_partition"),
            strict_schema=strict_schema,
        )
    except ValueError as exc:
        if strict_schema:
            raise
        LOGGER.warning("%s", exc)
    if expected_result_partition and metadata_result_partition and metadata_result_partition != expected_result_partition:
        msg = (
            "run_metadata result_partition mismatch with eval protocol: "
            f"expected={expected_result_partition} metadata={metadata_result_partition}"
        )
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)

    metadata_run_id = str(metadata.get("run_id", "")).strip()
    if metadata_run_id:
        try:
            validate_run_id(metadata_run_id)
        except ValueError as exc:
            if strict_schema:
                raise
            LOGGER.warning("%s", exc)
    try:
        metadata_seed = int(metadata.get("seed"))
    except (TypeError, ValueError):
        metadata_seed = None
        msg = f"run_metadata seed is not a valid int: {metadata.get('seed')}"
        if strict_schema:
            raise ValueError(msg)
        LOGGER.warning(msg)
    metadata_run_name = str(metadata.get("run_name", "")).strip()
    expected_run_id_from_metadata = ""
    if metadata_run_name and mode and metadata_protocol_id and metadata_seed is not None:
        expected_run_id_from_metadata = build_run_id(
            run_name=metadata_run_name,
            mode=mode,
            protocol_id=metadata_protocol_id,
            seed=metadata_seed,
        )
        if metadata_run_id and metadata_run_id != expected_run_id_from_metadata:
            msg = (
                "run_metadata run_id does not follow naming rule: "
                f"expected={expected_run_id_from_metadata} actual={metadata_run_id}"
            )
            if strict_schema:
                raise ValueError(msg)
            LOGGER.warning(msg)

    if predictions:
        run_ids = {str(rec.get("run_id", "")).strip() for rec in predictions if str(rec.get("run_id", "")).strip()}
        modes = {str(rec.get("mode", "")).strip().lower() for rec in predictions if str(rec.get("mode", "")).strip()}
        protocol_ids = {
            str(rec.get("protocol_id", "")).strip()
            for rec in predictions
            if str(rec.get("protocol_id", "")).strip()
        }

        if run_ids and metadata_run_id and metadata_run_id not in run_ids:
            msg = (
                "run_metadata run_id mismatch with predictions: "
                f"metadata={metadata_run_id} predictions={sorted(run_ids)}"
            )
            if strict_schema:
                raise ValueError(msg)
            LOGGER.warning(msg)

        if modes and mode and mode not in modes:
            msg = f"run_metadata mode mismatch with predictions: metadata={mode} predictions={sorted(modes)}"
            if strict_schema:
                raise ValueError(msg)
            LOGGER.warning(msg)

        if expected_protocol_id and protocol_ids and expected_protocol_id not in protocol_ids:
            msg = (
                "Predictions protocol_id mismatch with eval protocol: "
                f"expected={expected_protocol_id} predictions={sorted(protocol_ids)}"
            )
            if strict_schema:
                raise ValueError(msg)
            LOGGER.warning(msg)

    if enforce_protocol:
        protocol_languages = _normalize_language_list(protocol_cfg.get("test_languages"))
        metadata_languages = _normalize_language_list(metadata.get("test_languages"))
        if protocol_languages is not None:
            if metadata_languages is None or set(metadata_languages) != set(protocol_languages):
                msg = (
                    "run_metadata test_languages mismatch with eval protocol. "
                    f"protocol={sorted(protocol_languages)} metadata={sorted(metadata_languages or [])}"
                )
                if strict_schema:
                    raise ValueError(msg)
                LOGGER.warning(msg)

        for key in (
            "max_seq_len",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "cache_dir",
            "offline_only",
            "test_data_path",
            "result_partition",
        ):
            if key not in protocol_cfg:
                continue
            protocol_value = protocol_cfg.get(key)
            metadata_value = metadata.get(key)
            if protocol_value != metadata_value:
                msg = (
                    f"run_metadata {key} mismatch with eval protocol: "
                    f"protocol={protocol_value} metadata={metadata_value}"
                )
                if strict_schema:
                    raise ValueError(msg)
                LOGGER.warning(msg)


def load_model_for_eval(
    model_name: str,
    adapter_path: str,
    quantization: Optional[str] = "4bit",
    precision: str = "bf16",
) -> Tuple[Any, Any]:
    """Load base model + adapter for inference."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
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

    # Load adapter
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    LOGGER.info("Model loaded: base=%s adapter=%s", model_name, adapter_path)
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def _infer_generation_device(model: Any) -> torch.device:
    """Infer a safe input device for generation when device_map='auto' is used."""
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


def generate_sparql(
    model,
    tokenizer,
    question: str,
    max_seq_len: int = 512,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Generate a SPARQL query for a single question."""
    from transformers import GenerationConfig

    prompt = build_sparql_infer_text(question, tokenizer, few_shot_examples)
    # Prompt already contains chat-template special tokens.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )
    input_device = _infer_generation_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

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

    # Decode only the newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def batch_generate(
    model,
    tokenizer,
    test_data: List[Dict[str, Any]],
    max_seq_len: int = 512,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    run_id: str = "",
    protocol_id: str = "",
) -> List[Dict[str, Any]]:
    """Generate SPARQL for all test samples."""
    results: List[Dict[str, Any]] = []
    total = len(test_data)

    for idx, sample in enumerate(test_data):
        question = str(sample.get("question", "")).strip()
        gold_sparql = str(sample.get("sparql", "")).strip()
        language = str(sample.get("language", "unk")).strip()
        qid = str(sample.get("qid", "")).strip()

        if not question:
            LOGGER.warning("Skip sample %d/%d: empty question", idx + 1, total)
            continue

        start_time = time.monotonic()
        pred_sparql = generate_sparql(
            model, tokenizer, question,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            few_shot_examples=few_shot_examples,
        )
        elapsed = time.monotonic() - start_time

        results.append({
            "idx": idx,
            "qid": qid,
            "language": language,
            "question": question,
            "gold_sparql": gold_sparql,
            "pred_sparql": pred_sparql,
            "generation_time_sec": round(elapsed, 3),
            "mode": "adapter",
            "run_id": run_id,
            "protocol_id": protocol_id,
        })

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            LOGGER.info("Generated %d/%d", idx + 1, total)

    return results


# ── Metrics ──────────────────────────────────────────────────────────────────

def _execute_and_compare(
    pred_sparql: str,
    gold_sparql: str,
    cache: SPARQLCache,
    offline_only: bool = True,
) -> Dict[str, Any]:
    """Execute pred and gold SPARQL, compare results."""
    result: Dict[str, Any] = {
        "pred_executable": False,
        "gold_executable": False,
        "execution_match": False,
        "normalized_em": False,
        "semantic_equivalent": False,
        "answer_f1_eligible": False,
        "pred_answers": [],
        "gold_answers": [],
        "error_type": "",
        "error_detail": {},
    }

    if not pred_sparql.strip():
        result["error_type"] = "generation_empty"
        result["error_detail"] = _build_error_detail(
            stage="generation",
            code="empty_prediction",
            message="Model output is empty.",
            pred_executable=result["pred_executable"],
            gold_executable=result["gold_executable"],
        )
        return result

    # Normalized EM
    norm_pred = _normalize_sparql(pred_sparql)
    norm_gold = _normalize_sparql(gold_sparql)
    result["normalized_em"] = (norm_pred == norm_gold)

    # Execute gold
    try:
        gold_result = cache.execute(gold_sparql, offline_only=offline_only)
        if gold_result.get("ok", False):
            result["gold_executable"] = True
            result["gold_answers"] = gold_result.get("normalized_answers", [])
        else:
            result["error_type"] = "execution_error"
            result["error_detail"] = _build_error_detail(
                stage="gold_execute",
                code="gold_query_failed",
                message=str(gold_result.get("error", "")),
                raw_error_type=str(gold_result.get("error_type", "")),
                pred_executable=result["pred_executable"],
                gold_executable=result["gold_executable"],
            )
    except FileNotFoundError:
        if offline_only:
            raise
        LOGGER.debug("Gold SPARQL cache miss (offline): %s...", gold_sparql[:80])
    except Exception as exc:
        LOGGER.debug("Gold SPARQL execution error: %s", exc)
        result["error_type"] = "execution_error"
        result["error_detail"] = _build_error_detail(
            stage="gold_execute",
            code="gold_execute_exception",
            message=str(exc),
            raw_error_type=type(exc).__name__,
            raw_exception=repr(exc),
            pred_executable=result["pred_executable"],
            gold_executable=result["gold_executable"],
        )

    # Execute pred
    try:
        pred_result = cache.execute(pred_sparql, offline_only=offline_only)
        if pred_result.get("ok", False):
            result["pred_executable"] = True
            result["pred_answers"] = pred_result.get("normalized_answers", [])
        else:
            raw_message = str(pred_result.get("error", "")).strip()
            raw_error_type = str(pred_result.get("error_type", "")).strip()
            classified = (
                "syntax_or_parse_error"
                if _is_syntax_or_parse_error(raw_message, raw_error_type)
                else "execution_error"
            )
            result["error_type"] = classified
            result["error_detail"] = _build_error_detail(
                stage="pred_execute",
                code="pred_query_failed",
                message=raw_message,
                raw_error_type=raw_error_type,
                pred_executable=result["pred_executable"],
                gold_executable=result["gold_executable"],
            )
    except FileNotFoundError:
        if offline_only:
            raise
        result["error_type"] = "execution_error"
        result["error_detail"] = _build_error_detail(
            stage="pred_execute",
            code="cache_miss",
            message="Cache miss while offline_only=False fallback path.",
            cache_related=True,
            pred_executable=result["pred_executable"],
            gold_executable=result["gold_executable"],
        )
        LOGGER.debug("Pred SPARQL cache miss (offline): %s...", pred_sparql[:80])
    except Exception as exc:
        classified = (
            "syntax_or_parse_error"
            if _is_syntax_or_parse_error(str(exc), type(exc).__name__)
            else "execution_error"
        )
        result["error_type"] = classified
        result["error_detail"] = _build_error_detail(
            stage="pred_execute",
            code="pred_execute_exception",
            message=str(exc),
            raw_error_type=type(exc).__name__,
            raw_exception=repr(exc),
            pred_executable=result["pred_executable"],
            gold_executable=result["gold_executable"],
        )
        LOGGER.debug("Pred SPARQL execution error: %s", exc)

    # Execution Accuracy: both executable and same answer set
    if result["pred_executable"] and result["gold_executable"]:
        pred_set = set(result["pred_answers"])
        gold_set = set(result["gold_answers"])
        result["execution_match"] = (pred_set == gold_set)
        result["answer_f1_eligible"] = True

        # Semantic equivalent: EM=0 but execution results match
        if not result["normalized_em"] and result["execution_match"]:
            result["semantic_equivalent"] = True

    # Classify error type for non-matching results
    if not result["execution_match"] and result["pred_executable"]:
        result["error_type"] = "wrong_answer"
        detail_code = "empty_answer_set" if not result["pred_answers"] else "answer_mismatch"
        result["error_detail"] = _build_error_detail(
            stage="compare",
            code=detail_code,
            message="Predicted executable query does not match gold answers.",
            pred_executable=result["pred_executable"],
            gold_executable=result["gold_executable"],
        )

    if result["execution_match"]:
        result["error_type"] = ""
        result["error_detail"] = {}
    elif not result["pred_executable"] and not result["error_type"]:
        result["error_type"] = "execution_error"
        result["error_detail"] = _build_error_detail(
            stage="pred_execute",
            code="pred_not_executable",
            message="Predicted query is not executable.",
            pred_executable=result["pred_executable"],
            gold_executable=result["gold_executable"],
        )

    return result


def compute_answer_f1(
    pred_answers: List[str],
    gold_answers: List[str],
) -> Dict[str, float]:
    """Compute set-level precision, recall, F1 over normalized answer sets."""
    pred_set: Set[str] = set(pred_answers)
    gold_set: Set[str] = set(gold_answers)

    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred_set:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_clc(
    results: List[Dict[str, Any]],
    expected_languages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Cross-Lingual Consistency by grouping on canonical qid.

    CLC-Ans: fraction of qid groups where ALL languages got the same answer set.
    CLC-Struct: fraction of qid groups where ALL languages generated the same
                normalized SPARQL.
    """
    # Group by qid
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        qid = r.get("qid", "").strip()
        if qid:
            groups[qid].append(r)

    required_langs: Optional[Set[str]] = None
    if expected_languages:
        required_langs = {
            str(lang).strip().lower()
            for lang in expected_languages
            if str(lang).strip()
        }

    if not groups:
        return {
            "clc_ans": 0.0,
            "clc_struct": 0.0,
            "num_groups": 0,
            "ans_consistent_groups": 0,
            "struct_consistent_groups": 0,
            "expected_languages": sorted(required_langs) if required_langs else [],
            "incomplete_group_count": 0,
            "incomplete_groups": [],
        }

    ans_consistent = 0
    struct_consistent = 0
    total_groups = 0
    incomplete_groups: List[Dict[str, Any]] = []

    for qid, items in groups.items():
        present_langs = {
            str(item.get("language", "")).strip().lower()
            for item in items
            if str(item.get("language", "")).strip()
        }
        if required_langs is not None:
            missing = sorted(required_langs - present_langs)
            if missing:
                incomplete_groups.append(
                    {
                        "qid": qid,
                        "present_languages": sorted(present_langs),
                        "missing_languages": missing,
                    }
                )
                continue
        elif len(items) < 2:
            # Need at least 2 languages to measure consistency when no language set is specified.
            continue
        total_groups += 1

        # CLC-Struct: all normalized pred SPARQLs are the same
        norm_preds = set()
        for item in items:
            norm_preds.add(_normalize_sparql(item.get("pred_sparql", "")))
        if len(norm_preds) == 1:
            struct_consistent += 1

        # CLC-Ans: all execution answer sets are identical
        answer_sets = []
        all_executable = True
        for item in items:
            exec_info = item.get("exec_info", {})
            if not exec_info.get("pred_executable", False):
                all_executable = False
                break
            answer_sets.append(frozenset(exec_info.get("pred_answers", [])))

        if all_executable and len(set(answer_sets)) == 1:
            ans_consistent += 1

    return {
        "clc_ans": ans_consistent / max(total_groups, 1),
        "clc_struct": struct_consistent / max(total_groups, 1),
        "num_groups": total_groups,
        "ans_consistent_groups": ans_consistent,
        "struct_consistent_groups": struct_consistent,
        "expected_languages": sorted(required_langs) if required_langs else [],
        "incomplete_group_count": len(incomplete_groups),
        "incomplete_groups": incomplete_groups,
    }


def compute_all_metrics(
    results: List[Dict[str, Any]],
    cache: SPARQLCache,
    offline_only: bool = True,
    expected_languages: Optional[List[str]] = None,
    allowed_error_types: Optional[Set[str]] = None,
    fail_on_cache_miss: bool = True,
) -> Dict[str, Any]:
    """Compute all 6 metric categories over evaluation results."""
    total = len(results)
    if total == 0:
        return {
            "error": "No results to evaluate.",
            "total_samples": 0,
            "execution_accuracy": 0.0,
            "executable_rate": 0.0,
            "normalized_em": 0.0,
            "answer_f1_macro": 0.0,
            "answer_f1_macro_executable_only": 0.0,
            "semantic_equivalent_count": 0,
            "ea_count": 0,
            "executable_count": 0,
            "em_count": 0,
            "f1_executable_samples": 0,
            "per_language": {},
            "error_distribution": {},
            "cross_lingual_consistency": {
                "clc_ans": 0.0,
                "clc_struct": 0.0,
                "num_groups": 0,
                "ans_consistent_groups": 0,
                "struct_consistent_groups": 0,
                "expected_languages": [],
                "incomplete_group_count": 0,
                "incomplete_groups": [],
            },
        }

    effective_allowed_error_types = set(allowed_error_types or ALLOWED_ERROR_TYPES)

    # Execute and compare each result
    ea_count = 0
    executable_count = 0
    normalized_em_count = 0
    semantic_equiv_count = 0
    f1_sum = 0.0
    f1_exec_sum = 0.0
    f1_exec_count = 0
    error_types: Dict[str, int] = defaultdict(int)

    per_language: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "ea": 0, "executable": 0, "em": 0, "f1_sum": 0.0}
    )

    for r in results:
        try:
            exec_info = _execute_and_compare(
                r["pred_sparql"], r["gold_sparql"], cache, offline_only
            )
        except FileNotFoundError as exc:
            if fail_on_cache_miss:
                qid = str(r.get("qid", "")).strip() or "<no-qid>"
                lang = str(r.get("language", "")).strip() or "unk"
                raise RuntimeError(
                    "Offline evaluation cache miss detected. "
                    f"qid={qid}, language={lang}. {exc}"
                ) from exc
            exec_info = {
                "pred_executable": False,
                "gold_executable": False,
                "execution_match": False,
                "normalized_em": False,
                "semantic_equivalent": False,
                "answer_f1_eligible": False,
                "pred_answers": [],
                "gold_answers": [],
                "error_type": "execution_error",
                "error_detail": _build_error_detail(
                    stage="execute",
                    code="cache_miss",
                    message=str(exc),
                    cache_related=True,
                    raw_exception=repr(exc),
                    pred_executable=False,
                    gold_executable=False,
                ),
            }
        r["exec_info"] = exec_info  # attach for CLC

        lang = r.get("language", "unk")
        per_language[lang]["total"] += 1

        if exec_info["pred_executable"]:
            executable_count += 1
            per_language[lang]["executable"] += 1

        if exec_info["execution_match"]:
            ea_count += 1
            per_language[lang]["ea"] += 1

        if exec_info["normalized_em"]:
            normalized_em_count += 1
            per_language[lang]["em"] += 1

        if exec_info["semantic_equivalent"]:
            semantic_equiv_count += 1

        if exec_info.get("answer_f1_eligible", False):
            f1_info = compute_answer_f1(
                exec_info.get("pred_answers", []),
                exec_info.get("gold_answers", []),
            )
            f1_exec_sum += f1_info["f1"]
            f1_exec_count += 1
        else:
            f1_info = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        r["f1_info"] = f1_info
        f1_sum += f1_info["f1"]
        per_language[lang]["f1_sum"] += f1_info["f1"]

        if exec_info.get("error_type"):
            error_type = str(exec_info["error_type"]).strip()
            if error_type not in effective_allowed_error_types:
                error_type = "execution_error"
            error_types[error_type] += 1

    # Aggregate
    metrics: Dict[str, Any] = {
        "total_samples": total,
        "execution_accuracy": round(ea_count / total, 4),
        "executable_rate": round(executable_count / total, 4),
        "normalized_em": round(normalized_em_count / total, 4),
        "answer_f1_macro": round(f1_sum / total, 4),
        "answer_f1_macro_executable_only": round(f1_exec_sum / max(f1_exec_count, 1), 4),
        "semantic_equivalent_count": semantic_equiv_count,
        "ea_count": ea_count,
        "executable_count": executable_count,
        "em_count": normalized_em_count,
        "f1_executable_samples": f1_exec_count,
    }

    # Per-language breakdown
    per_lang_metrics: Dict[str, Dict[str, Any]] = {}
    for lang, stats in sorted(per_language.items()):
        n = stats["total"]
        per_lang_metrics[lang] = {
            "total": n,
            "execution_accuracy": round(stats["ea"] / max(n, 1), 4),
            "executable_rate": round(stats["executable"] / max(n, 1), 4),
            "normalized_em": round(stats["em"] / max(n, 1), 4),
            "answer_f1_macro": round(stats["f1_sum"] / max(n, 1), 4),
        }
    metrics["per_language"] = per_lang_metrics

    # Error classification
    metrics["error_distribution"] = dict(sorted(error_types.items(), key=lambda kv: -kv[1]))
    metrics["error_type_schema"] = sorted(effective_allowed_error_types)

    # Cross-lingual consistency
    clc = compute_clc(results, expected_languages=expected_languages)
    metrics["cross_lingual_consistency"] = clc

    return metrics


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


# ── Main CLI ─────────────────────────────────────────────────────────────────

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPARQL generation.")
    parser.add_argument("--config", type=str, default=None,
                        help="Training config YAML (reads model/data paths from it).")
    parser.add_argument("--eval_protocol", type=str, default=None,
                        help="Shared evaluation protocol YAML path.")
    parser.add_argument("--predictions_file", type=str, default=None,
                        help="Path to an existing predictions JSONL to evaluate only.")
    parser.add_argument("--run_metadata_file", type=str, default=None,
                        help="Path to run_metadata.json (required for strict protocol validation).")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to trained adapter (overrides config).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Base model name (overrides config).")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test JSONL file.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="SPARQL cache directory.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write evaluation results.")
    parser.add_argument("--offline_only", action="store_true",
                        help="Only use cached SPARQL results, never query endpoint.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples.")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max tokens to generate per query.")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Max input sequence length for prompt tokenization.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling for generation.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (used when do_sample=True).")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Sampling top-p (used when do_sample=True).")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization mode (4bit or none).")
    parser.add_argument(
        "--test_languages",
        type=str,
        default=None,
        help="Comma-separated language filter for evaluation (e.g., en,de,es,ru).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_eval_args()

    # Load config if provided
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
    protocol_allowed_modes = {
        str(mode).strip().lower()
        for mode in protocol_cfg.get("allowed_modes", sorted(ALLOWED_PRED_MODES))
        if str(mode).strip()
    } or set(ALLOWED_PRED_MODES)
    allowed_error_types = _resolve_allowed_error_types(protocol_cfg, strict_schema=strict_schema)
    primary_metric_names, aux_metric_names = _resolve_metric_names(
        protocol_cfg,
        strict_schema=strict_schema,
    )
    fail_on_cache_miss = bool(protocol_cfg.get("fail_on_cache_miss", True))

    # Resolve parameters (CLI overrides config)
    predictions_file = args.predictions_file or cfg.get("predictions_file")
    run_metadata_file = args.run_metadata_file or cfg.get("run_metadata_file")
    model_name = args.model_name or cfg.get("model_name_or_path") or protocol_cfg.get("model_name_or_path")
    adapter_path = args.adapter_path or cfg.get("adapter_path")
    if not adapter_path:
        adapter_path = os.path.join(
            cfg.get("output_dir", "outputs"),
            cfg.get("run_name", "run"),
        )
    test_data_path = args.test_data or cfg.get("test_data_path") or protocol_cfg.get("test_data_path")
    cache_dir = (
        args.cache_dir
        or cfg.get("sparql_cache_dir")
        or protocol_cfg.get("cache_dir")
        or "data/sparql/cache"
    )
    if args.output_dir:
        output_dir = args.output_dir
    elif predictions_file:
        output_dir = str(Path(predictions_file).resolve().parent / "eval_results")
    else:
        output_dir = os.path.join(adapter_path, "eval_results")
    offline_only = _resolve_bool(
        cli_flag=args.offline_only,
        cfg_value=cfg.get("offline_only"),
        protocol_value=protocol_cfg.get("offline_only"),
        default=False,
    )
    quantization = args.quantization or cfg.get("quantization", protocol_cfg.get("quantization", "4bit"))
    max_seq_len = int(
        args.max_seq_len
        if args.max_seq_len is not None
        else cfg.get("max_seq_len", protocol_cfg.get("max_seq_len", 512))
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
        args.temperature
        if args.temperature is not None
        else cfg.get("temperature", protocol_cfg.get("temperature", 1.0))
    )
    top_p = float(
        args.top_p
        if args.top_p is not None
        else cfg.get("top_p", protocol_cfg.get("top_p", 1.0))
    )
    _validate_decoding_args(
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
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
        for key, current in (
            ("cache_dir", cache_dir),
            ("offline_only", offline_only),
            ("test_data_path", test_data_path),
        ):
            if key in protocol_cfg and protocol_cfg.get(key) != current:
                raise ValueError(
                    f"{key} mismatch with eval protocol: protocol={protocol_cfg.get(key)} current={current}"
                )

    seed = args.seed

    if not predictions_file:
        if not model_name:
            raise ValueError("model_name is required (via --model_name or config).")
        if not test_data_path:
            raise ValueError("test_data is required (via --test_data or config test_data_path).")
        if not Path(test_data_path).exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")
    else:
        if not Path(predictions_file).exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        if not run_metadata_file:
            run_metadata_file = _find_default_run_metadata_path(predictions_file)
        if enforce_protocol and strict_schema and not Path(run_metadata_file).exists():
            raise FileNotFoundError(
                "Strict protocol validation requires run_metadata.json. "
                f"Not found: {run_metadata_file}"
            )

    ensure_dir(output_dir)
    logger = setup_logging(os.path.join(output_dir, "eval.log"), logger_name="alrem")
    set_seed(seed)

    logger.info("Predictions file mode: %s", bool(predictions_file))
    if predictions_file:
        logger.info("Predictions file: %s", predictions_file)
        logger.info("Run metadata file: %s", run_metadata_file or "(none)")
    else:
        logger.info("Model: %s", model_name)
        logger.info("Adapter: %s", adapter_path)
        logger.info("Test data: %s", test_data_path)
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Offline only: %s", offline_only)
    logger.info("fail_on_cache_miss: %s", fail_on_cache_miss)
    logger.info("Eval protocol: %s", eval_protocol_loaded_path or "(none)")
    if protocol_id:
        logger.info("Protocol id: %s", protocol_id)
    logger.info("Result partition: %s", protocol_result_partition)
    logger.info("Output dir: %s", output_dir)
    logger.info(
        "Decoding: max_seq_len=%d max_new_tokens=%d do_sample=%s temperature=%.3f top_p=%.3f",
        max_seq_len,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
    )
    logger.info("Primary metrics (protocol): %s", primary_metric_names)
    logger.info("Aux metrics (protocol): %s", aux_metric_names)

    results: List[Dict[str, Any]]
    run_metadata: Dict[str, Any] = {}
    if predictions_file:
        results = _load_predictions_jsonl(predictions_file)
        if run_metadata_file and Path(run_metadata_file).exists():
            run_metadata = _load_json(run_metadata_file)
            _validate_run_metadata_schema(
                run_metadata,
                strict_schema=strict_schema,
                enforce_protocol=enforce_protocol,
                expected_protocol_id=protocol_id,
                expected_result_partition=protocol_result_partition,
                allowed_modes=protocol_allowed_modes,
                protocol_cfg=protocol_cfg,
                predictions=results,
            )
        elif enforce_protocol and strict_schema:
            raise FileNotFoundError(
                "Strict protocol validation requires run_metadata.json in predictions mode."
            )
        logger.info("Loaded %d predictions from file", len(results))
        if args.max_samples:
            results = results[: args.max_samples]
        before = len(results)
        results = _filter_by_languages(results, test_languages)
        if test_languages:
            logger.info(
                "Applied prediction language filter %s: %d -> %d samples",
                sorted(set(test_languages)),
                before,
                len(results),
            )
        if not results:
            raise ValueError("No prediction samples left after filtering.")
    else:
        # Load test data
        test_data = _load_jsonl(test_data_path)
        if args.max_samples:
            test_data = test_data[: args.max_samples]
        before = len(test_data)
        test_data = _filter_by_languages(test_data, test_languages)
        if test_languages:
            logger.info(
                "Applied test language filter %s: %d -> %d samples",
                sorted(set(test_languages)),
                before,
                len(test_data),
            )
        logger.info("Test samples: %d", len(test_data))
        if len(test_data) == 0:
            raise ValueError(f"No test samples loaded from: {test_data_path}")

        # Load model
        model, tokenizer = load_model_for_eval(
            model_name=model_name,
            adapter_path=adapter_path,
            quantization=quantization,
            precision=cfg.get("precision", "bf16"),
        )

        # Generate
        logger.info("Starting generation...")
        run_name_for_id = str(cfg.get("run_name", "")).strip() or Path(output_dir).name
        run_id = build_run_id(
            run_name=run_name_for_id,
            mode="adapter",
            protocol_id=protocol_id,
            seed=seed,
        )
        results = batch_generate(
            model, tokenizer, test_data,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            run_id=run_id,
            protocol_id=protocol_id,
        )
        logger.info("Generation complete: %d results", len(results))
        run_metadata = {
            "run_id": run_id,
            "run_name": run_name_for_id,
            "mode": "adapter",
            "method": str(cfg.get("method", "adapter")).strip().lower() or "adapter",
            "task": "sparql",
            "model_name_or_path": model_name,
            "adapter_path": adapter_path,
            "test_data_path": test_data_path,
            "test_languages": test_languages or [],
            "max_seq_len": max_seq_len,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "cache_dir": cache_dir,
            "offline_only": offline_only,
            "result_partition": protocol_result_partition,
            "protocol_name": protocol_name,
            "protocol_version": protocol_version,
            "protocol_id": protocol_id,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": output_dir,
        }
        _validate_run_metadata_schema(
            run_metadata,
            strict_schema=strict_schema,
            enforce_protocol=enforce_protocol,
            expected_protocol_id=protocol_id,
            expected_result_partition=protocol_result_partition,
            allowed_modes=protocol_allowed_modes,
            protocol_cfg=protocol_cfg,
            predictions=results,
        )

    expected_run_id = str(run_metadata.get("run_id", "")).strip() if run_metadata else ""
    _validate_predictions_schema(
        results,
        strict_schema=strict_schema,
        expected_protocol_id=protocol_id,
        expected_run_id=expected_run_id,
        allowed_modes=protocol_allowed_modes,
    )
    _cleanup_internal_prediction_fields(results)

    # Save raw predictions
    preds_path = os.path.join(output_dir, "predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Predictions saved to %s", preds_path)

    metadata_path = os.path.join(output_dir, "run_metadata.json")
    if run_metadata:
        save_json(run_metadata, metadata_path)
        logger.info("Run metadata saved to %s", metadata_path)

    # Compute metrics
    cache = SPARQLCache(cache_dir=cache_dir, force_offline=offline_only)
    logger.info("Computing metrics (offline_only=%s)...", offline_only)
    metrics = compute_all_metrics(
        results,
        cache,
        offline_only=offline_only,
        expected_languages=test_languages,
        allowed_error_types=allowed_error_types,
        fail_on_cache_miss=fail_on_cache_miss,
    )
    metric_values = {
        "EA": metrics["execution_accuracy"],
        "ER": metrics["executable_rate"],
        "CLC-Ans": metrics.get("cross_lingual_consistency", {}).get("clc_ans", 0.0),
        "CLC-Struct": metrics.get("cross_lingual_consistency", {}).get("clc_struct", 0.0),
        "NormEM": metrics["normalized_em"],
        "AnswerF1": metrics["answer_f1_macro"],
    }
    metrics["metric_protocol"] = {
        "primary_metrics": primary_metric_names,
        "aux_metrics": aux_metric_names,
        "metric_values": {k: metric_values[k] for k in primary_metric_names + aux_metric_names if k in metric_values},
    }
    metrics["eval_protocol"] = {
        "path": eval_protocol_loaded_path,
        "protocol_name": protocol_name,
        "protocol_version": protocol_version,
        "protocol_id": protocol_id,
        "enforce_protocol": enforce_protocol,
        "strict_schema": strict_schema,
        "main_table_protocol": main_table_protocol,
        "result_partition": protocol_result_partition,
        "fail_on_cache_miss": fail_on_cache_miss,
        "allowed_error_types": sorted(allowed_error_types),
        "primary_metrics": primary_metric_names,
        "aux_metrics": aux_metric_names,
        "test_languages": test_languages or [],
        "decoding": {
            "max_seq_len": max_seq_len,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    if run_metadata:
        metrics["run_metadata"] = {
            "path": metadata_path,
            "run_id": run_metadata.get("run_id", ""),
            "mode": run_metadata.get("mode", ""),
            "method": run_metadata.get("method", ""),
            "result_partition": run_metadata.get("result_partition", ""),
        }

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, metrics_path)
    logger.info("Metrics saved to %s", metrics_path)

    # Save detailed results (with exec_info and f1_info)
    detailed_path = os.path.join(output_dir, "detailed_results.jsonl")
    with open(detailed_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    logger.info("Detailed results saved to %s", detailed_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Execution Accuracy: %.2f%%", metrics["execution_accuracy"] * 100)
    logger.info("Executable Rate:    %.2f%%", metrics["executable_rate"] * 100)
    logger.info("Normalized EM:      %.2f%%", metrics["normalized_em"] * 100)
    logger.info("Answer F1 (macro):  %.4f", metrics["answer_f1_macro"])
    logger.info(
        "Answer F1 (exec-only): %.4f (n=%d)",
        metrics.get("answer_f1_macro_executable_only", 0.0),
        metrics.get("f1_executable_samples", 0),
    )
    clc = metrics.get("cross_lingual_consistency", {})
    logger.info("CLC-Ans:  %.2f%% (%d/%d groups)",
                clc.get("clc_ans", 0) * 100,
                clc.get("ans_consistent_groups", 0),
                clc.get("num_groups", 0))
    logger.info("CLC-Struct: %.2f%% (%d/%d groups)",
                clc.get("clc_struct", 0) * 100,
                clc.get("struct_consistent_groups", 0),
                clc.get("num_groups", 0))
    logger.info("CLC incomplete groups: %d", clc.get("incomplete_group_count", 0))
    logger.info("-" * 60)
    logger.info("Per-language breakdown:")
    for lang, stats in metrics.get("per_language", {}).items():
        logger.info("  %s: EA=%.2f%% Exec=%.2f%% EM=%.2f%% F1=%.4f (n=%d)",
                     lang,
                     stats["execution_accuracy"] * 100,
                     stats["executable_rate"] * 100,
                     stats["normalized_em"] * 100,
                     stats["answer_f1_macro"],
                     stats["total"])
    logger.info("-" * 60)
    logger.info("Error distribution: %s", metrics.get("error_distribution", {}))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
