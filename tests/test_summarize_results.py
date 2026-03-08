import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _build_metrics(result_partition: str) -> Dict[str, Any]:
    return {
        "execution_accuracy": 0.5,
        "executable_rate": 0.8,
        "normalized_em": 0.4,
        "answer_f1_macro": 0.3,
        "answer_f1_macro_executable_only": 0.35,
        "total_samples": 10,
        "cross_lingual_consistency": {
            "clc_ans": 0.1,
            "clc_struct": 0.2,
            "num_groups": 3,
            "incomplete_group_count": 0,
            "ans_consistent_groups": 1,
            "struct_consistent_groups": 1,
            "expected_languages": ["en", "de", "es", "ru"],
        },
        "per_language": {
            "en": {
                "execution_accuracy": 0.5,
                "executable_rate": 0.8,
                "normalized_em": 0.4,
                "answer_f1_macro": 0.3,
                "total": 3,
            }
        },
        "error_type_schema": [
            "generation_empty",
            "syntax_or_parse_error",
            "execution_error",
            "wrong_answer",
        ],
        "error_distribution": {"wrong_answer": 2},
        "eval_protocol": {
            "protocol_id": "sparql_eval_shared:v1",
            "result_partition": result_partition,
        },
    }


def test_summarize_results_partition_and_method_inference(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "summarize_results.py"
    outputs_dir = tmp_path / "outputs"
    out_dir = tmp_path / "tables"

    run_main = outputs_dir / "sparql_s2_alrem"
    _write_yaml(run_main / "config.yaml", {"method": "alrem", "r_high": 32, "r_low": 8})
    _write_json(run_main / "run_report.json", {"method": "alrem", "lora": {"r_high": 32, "r_low": 8}})
    _write_json(run_main / "eval_results" / "metrics.json", _build_metrics("unified_codechain"))

    run_reverse = outputs_dir / "sparql_s2_reverse_sandwich"
    _write_yaml(run_reverse / "config.yaml", {"method": "alrem", "r_high": 4, "r_low": 32})
    _write_json(run_reverse / "run_report.json", {"method": "alrem", "lora": {"r_high": 4, "r_low": 32}})
    _write_json(run_reverse / "eval_results" / "metrics.json", _build_metrics("unified_codechain"))

    run_external = outputs_dir / "sparql_s2_alrem_strong_external"
    _write_yaml(run_external / "config.yaml", {"method": "alrem", "r_high": 32, "r_low": 4})
    _write_json(run_external / "run_report.json", {"method": "alrem", "lora": {"r_high": 32, "r_low": 4}})
    _write_json(run_external / "eval_results" / "metrics.json", _build_metrics("external_reported"))

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--outputs_dir",
            str(outputs_dir),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"summarize_results failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    main_table = (out_dir / "main_results.md").read_text(encoding="utf-8")
    external_table = (out_dir / "external_results.md").read_text(encoding="utf-8")
    per_language = (out_dir / "per_language.md").read_text(encoding="utf-8")

    assert "sparql_s2_alrem" in main_table
    assert "alrem_main" in main_table
    assert "sparql_s2_reverse_sandwich" in main_table
    assert "reverse_sandwich" in main_table
    assert "sparql_s2_alrem_strong_external" not in main_table

    assert "sparql_s2_alrem_strong_external" in external_table
    assert "alrem_strong" in external_table
    assert "external_reported" in external_table
    assert "| sparql_s2_alrem |" not in external_table

    assert "result_partition" in per_language
