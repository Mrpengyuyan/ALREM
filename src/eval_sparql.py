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
from .sparql_executor import SPARQLCache
from .utils import ensure_dir, load_yaml, save_json, set_seed, setup_logging

LOGGER = logging.getLogger("alrem.eval_sparql")


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


# ── Model loading ────────────────────────────────────────────────────────────

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
    max_new_tokens: int = 256,
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
        max_length=512,
        add_special_tokens=False,
    )
    input_device = _infer_generation_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
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
    max_new_tokens: int = 256,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
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
            max_new_tokens=max_new_tokens,
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
        "error_type": "generation_empty",
    }

    if not pred_sparql.strip():
        result["error_type"] = "generation_empty"
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
    except FileNotFoundError:
        if offline_only:
            raise
        LOGGER.debug("Gold SPARQL cache miss (offline): %s...", gold_sparql[:80])
    except Exception as exc:
        LOGGER.debug("Gold SPARQL execution error: %s", exc)

    # Execute pred
    try:
        pred_result = cache.execute(pred_sparql, offline_only=offline_only)
        if pred_result.get("ok", False):
            result["pred_executable"] = True
            result["pred_answers"] = pred_result.get("normalized_answers", [])
            result["error_type"] = ""  # No error if executable
        else:
            result["error_type"] = "execution_error"
    except FileNotFoundError:
        if offline_only:
            raise
        result["error_type"] = "cache_miss"
        LOGGER.debug("Pred SPARQL cache miss (offline): %s...", pred_sparql[:80])
    except Exception as exc:
        result["error_type"] = "execution_exception"
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
        if not result["pred_answers"]:
            result["error_type"] = "empty_result"
        else:
            result["error_type"] = "wrong_answer"

    if not result["pred_executable"] and not result["error_type"]:
        result["error_type"] = "not_executable"

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

    if not groups:
        return {"clc_ans": 0.0, "clc_struct": 0.0, "num_groups": 0}

    ans_consistent = 0
    struct_consistent = 0
    total_groups = 0

    for qid, items in groups.items():
        if len(items) < 2:
            # Need at least 2 languages to measure consistency
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
    }


def compute_all_metrics(
    results: List[Dict[str, Any]],
    cache: SPARQLCache,
    offline_only: bool = True,
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
            "cross_lingual_consistency": {"clc_ans": 0.0, "clc_struct": 0.0, "num_groups": 0},
        }

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
            qid = str(r.get("qid", "")).strip() or "<no-qid>"
            lang = str(r.get("language", "")).strip() or "unk"
            raise RuntimeError(
                "Offline evaluation cache miss detected. "
                f"qid={qid}, language={lang}. {exc}"
            ) from exc
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
            error_types[exec_info["error_type"]] += 1

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

    # Cross-lingual consistency
    clc = compute_clc(results)
    metrics["cross_lingual_consistency"] = clc

    return metrics


# ── Main CLI ─────────────────────────────────────────────────────────────────

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPARQL generation.")
    parser.add_argument("--config", type=str, default=None,
                        help="Training config YAML (reads model/data paths from it).")
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
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per query.")
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
        cfg = load_yaml(args.config)

    # Resolve parameters (CLI overrides config)
    model_name = args.model_name or cfg.get("model_name_or_path")
    adapter_path = args.adapter_path or os.path.join(
        cfg.get("output_dir", "outputs"),
        cfg.get("run_name", "run"),
    )
    test_data_path = args.test_data or cfg.get("test_data_path")
    cache_dir = args.cache_dir or cfg.get("sparql_cache_dir", "data/sparql/cache")
    output_dir = args.output_dir or os.path.join(adapter_path, "eval_results")
    offline_only = args.offline_only or cfg.get("offline_only", False)
    quantization = args.quantization or cfg.get("quantization", "4bit")
    test_languages = _parse_language_list(args.test_languages)
    if test_languages is None:
        cfg_languages = cfg.get("test_languages")
        if isinstance(cfg_languages, list):
            test_languages = [str(lang).strip().lower() for lang in cfg_languages if str(lang).strip()]
    seed = args.seed

    if not model_name:
        raise ValueError("model_name is required (via --model_name or config).")
    if not test_data_path:
        raise ValueError("test_data is required (via --test_data or config test_data_path).")
    if not Path(test_data_path).exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    ensure_dir(output_dir)
    logger = setup_logging(os.path.join(output_dir, "eval.log"), logger_name="alrem")
    set_seed(seed)

    logger.info("Model: %s", model_name)
    logger.info("Adapter: %s", adapter_path)
    logger.info("Test data: %s", test_data_path)
    logger.info("Cache dir: %s", cache_dir)
    logger.info("Offline only: %s", offline_only)
    logger.info("Output dir: %s", output_dir)

    # Load test data
    test_data = _load_jsonl(test_data_path)
    if args.max_samples:
        test_data = test_data[: args.max_samples]
    if test_languages:
        allowed = set(test_languages)
        before = len(test_data)
        test_data = [
            sample for sample in test_data
            if str(sample.get("language", "unk")).strip().lower() in allowed
        ]
        logger.info(
            "Applied test language filter %s: %d -> %d samples",
            sorted(allowed),
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
    results = batch_generate(
        model, tokenizer, test_data,
        max_new_tokens=args.max_new_tokens,
    )
    logger.info("Generation complete: %d results", len(results))

    # Save raw predictions
    preds_path = os.path.join(output_dir, "predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Predictions saved to %s", preds_path)

    # Compute metrics
    cache = SPARQLCache(cache_dir=cache_dir, force_offline=offline_only)
    logger.info("Computing metrics (offline_only=%s)...", offline_only)
    metrics = compute_all_metrics(results, cache, offline_only=offline_only)

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
