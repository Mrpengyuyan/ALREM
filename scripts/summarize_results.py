import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ALLOWED_RESULT_PARTITIONS = {
    "unified_codechain",
    "external_reproduced",
    "external_reported",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_markdown_table(headers: List[str], rows: List[List[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(x) for x in row) + " |\n")


def _find_metric_files(outputs_dir: Path) -> List[Path]:
    metric_paths: List[Path] = []
    for root, _, files in os.walk(outputs_dir):
        if "metrics.json" in files:
            metric_paths.append(Path(root) / "metrics.json")
    return sorted(metric_paths)


def _infer_run_root(metrics_path: Path) -> Path:
    if metrics_path.parent.name == "eval_results":
        return metrics_path.parent.parent
    return metrics_path.parent


def _infer_stage(run_name: str) -> str:
    lowered = run_name.lower()
    if "stage1" in lowered or "_s1" in lowered or lowered.startswith("s1"):
        return "stage1"
    if "stage2" in lowered or "_s2" in lowered or lowered.startswith("s2"):
        return "stage2"
    if "icl" in lowered:
        return "icl"
    return "unknown"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _infer_alrem_variant(
    lowered_run_tokens: str,
    cfg: Dict[str, Any],
    run_report: Dict[str, Any],
) -> str:
    lora = run_report.get("lora", {}) or {}
    r_high = _coerce_int(cfg.get("r_high"))
    if r_high is None:
        r_high = _coerce_int(lora.get("r_high"))
    r_low = _coerce_int(cfg.get("r_low"))
    if r_low is None:
        r_low = _coerce_int(lora.get("r_low"))

    if "reverse" in lowered_run_tokens:
        return "reverse_sandwich"
    if r_high is not None and r_low is not None and r_high < r_low:
        return "reverse_sandwich"
    if "strong" in lowered_run_tokens:
        return "alrem_strong"
    if r_high is not None and r_low is not None and r_high > r_low and r_low <= 4:
        return "alrem_strong"
    return "alrem_main"


def _infer_method(run_name: str, cfg: Dict[str, Any], run_report: Dict[str, Any]) -> str:
    mode = str(run_report.get("mode", "") or cfg.get("mode", "")).strip().lower()
    if mode in {"icl_zero", "icl_fewshot"}:
        return mode

    method = str(cfg.get("method", "") or run_report.get("method", "")).strip().lower()
    lowered = " ".join(
        [
            run_name.lower(),
            str(cfg.get("run_name", "")).lower(),
            str(run_report.get("run_name", "")).lower(),
        ]
    )
    if method == "uniform":
        return "uniform_lora"
    if method == "matched":
        return "parameter_matched_lora"
    if method == "alrem":
        return _infer_alrem_variant(lowered, cfg, run_report)

    if "icl_few" in lowered:
        return "icl_fewshot"
    if "icl_zero" in lowered or "icl" in lowered:
        return "icl_zero"
    if method:
        return method
    return "unknown"


def _normalize_partition(value: Any) -> str:
    partition = str(value or "").strip().lower()
    if not partition:
        return "unknown"
    if partition in ALLOWED_RESULT_PARTITIONS:
        return partition
    return partition


def _infer_result_partition(
    metrics: Dict[str, Any],
    cfg: Dict[str, Any],
    run_report: Dict[str, Any],
) -> str:
    candidates = [
        (metrics.get("eval_protocol", {}) or {}).get("result_partition"),
        (metrics.get("run_metadata", {}) or {}).get("result_partition"),
        (run_report.get("eval_protocol", {}) or {}).get("result_partition"),
        run_report.get("result_partition"),
        cfg.get("result_partition"),
    ]
    for candidate in candidates:
        normalized = _normalize_partition(candidate)
        if normalized != "unknown":
            return normalized
    return "unknown"


def _is_sparql_metrics(metrics: Dict[str, Any]) -> bool:
    return "execution_accuracy" in metrics and "executable_rate" in metrics


def _to_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def _to_float_str(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def summarize(outputs_dir: Path, out_dir: Path) -> None:
    main_rows: List[List[Any]] = []
    external_rows: List[List[Any]] = []
    per_lang_rows: List[List[Any]] = []
    clc_rows: List[List[Any]] = []
    error_rows: List[List[Any]] = []

    for metrics_path in _find_metric_files(outputs_dir):
        metrics = _load_json(metrics_path)
        if not _is_sparql_metrics(metrics):
            continue

        run_root = _infer_run_root(metrics_path)
        run_name = run_root.name
        cfg_path = run_root / "config.yaml"
        run_report_path = run_root / "run_report.json"

        cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
        run_report = _load_json(run_report_path) if run_report_path.exists() else {}

        method = _infer_method(run_name, cfg, run_report)
        stage = _infer_stage(run_name)
        result_partition = _infer_result_partition(metrics, cfg, run_report)

        clc = metrics.get("cross_lingual_consistency", {}) or {}
        eval_protocol = metrics.get("eval_protocol", {}) or {}

        summary_row = [
            run_name,
            method,
            stage,
            result_partition,
            _to_pct(metrics.get("execution_accuracy", 0.0)),
            _to_pct(metrics.get("executable_rate", 0.0)),
            _to_pct(metrics.get("normalized_em", 0.0)),
            _to_float_str(metrics.get("answer_f1_macro", 0.0)),
            _to_float_str(metrics.get("answer_f1_macro_executable_only", 0.0)),
            _to_pct(clc.get("clc_ans", 0.0)),
            _to_pct(clc.get("clc_struct", 0.0)),
            int(metrics.get("total_samples", 0)),
            str(eval_protocol.get("protocol_id", "")),
            str(metrics_path.parent),
        ]
        if result_partition == "unified_codechain":
            main_rows.append(summary_row)
        else:
            external_rows.append(summary_row)

        per_language = metrics.get("per_language", {}) or {}
        for lang, stats in sorted(per_language.items()):
            per_lang_rows.append(
                [
                    run_name,
                    result_partition,
                    lang,
                    _to_pct(stats.get("execution_accuracy", 0.0)),
                    _to_pct(stats.get("executable_rate", 0.0)),
                    _to_pct(stats.get("normalized_em", 0.0)),
                    _to_float_str(stats.get("answer_f1_macro", 0.0)),
                    int(stats.get("total", 0)),
                ]
            )

        clc_rows.append(
            [
                run_name,
                result_partition,
                int(clc.get("num_groups", 0)),
                int(clc.get("incomplete_group_count", 0)),
                int(clc.get("ans_consistent_groups", 0)),
                int(clc.get("struct_consistent_groups", 0)),
                ",".join(clc.get("expected_languages", []) or []),
            ]
        )

        error_rows.append(
            [
                run_name,
                result_partition,
                ",".join(metrics.get("error_type_schema", []) or []),
                json.dumps(metrics.get("error_distribution", {}), ensure_ascii=False),
            ]
        )

    main_rows.sort(key=lambda row: row[0])
    external_rows.sort(key=lambda row: (row[3], row[0]))
    per_lang_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    clc_rows.sort(key=lambda row: (row[1], row[0]))
    error_rows.sort(key=lambda row: (row[1], row[0]))

    headers = [
        "run",
        "method",
        "stage",
        "result_partition",
        "EA",
        "ExecRate",
        "NormEM",
        "F1",
        "F1_exec_only",
        "CLC_Ans",
        "CLC_Struct",
        "samples",
        "protocol_id",
        "metrics_dir",
    ]
    _write_markdown_table(headers, main_rows, out_dir / "main_results.md")
    _write_markdown_table(headers, external_rows, out_dir / "external_results.md")

    _write_markdown_table(
        ["run", "result_partition", "language", "EA", "ExecRate", "NormEM", "F1", "samples"],
        per_lang_rows,
        out_dir / "per_language.md",
    )

    _write_markdown_table(
        ["run", "result_partition", "num_groups", "incomplete_groups", "ans_consistent", "struct_consistent", "expected_languages"],
        clc_rows,
        out_dir / "clc_groups.md",
    )

    _write_markdown_table(
        ["run", "result_partition", "error_type_schema", "error_distribution"],
        error_rows,
        out_dir / "error_distribution.md",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize SPARQL evaluation results from outputs directory.")
    parser.add_argument("--outputs_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    summarize(Path(args.outputs_dir), Path(args.out_dir))
