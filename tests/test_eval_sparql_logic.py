import json
import logging
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src import eval_sparql as eval_sparql_mod
from src.eval_sparql import (
    batch_generate,
    _filter_by_languages,
    _load_predictions_jsonl,
    _normalize_prediction_record,
    _parse_language_list,
    compute_all_metrics,
)
from src.run_identity import build_run_id


def _write_jsonl(path: Path, rows) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_eval_protocol(
    path: Path,
    *,
    test_data_path: str,
    test_languages: list[str],
    cache_dir: str,
    max_seq_len: int = 512,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> None:
    payload = {
        "protocol_name": "unit_test_protocol",
        "protocol_version": "v1",
        "protocol_id": "unit_test_protocol:v1",
        "task": "sparql",
        "enforce_protocol": True,
        "strict_schema": True,
        "main_table_protocol": True,
        "result_partition": "unified_codechain",
        "test_data_path": test_data_path,
        "test_languages": test_languages,
        "max_seq_len": max_seq_len,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "cache_dir": cache_dir,
        "offline_only": True,
        "fail_on_cache_miss": True,
        "allowed_modes": ["adapter", "icl_zero", "icl_fewshot"],
        "allowed_error_types": [
            "generation_empty",
            "syntax_or_parse_error",
            "execution_error",
            "wrong_answer",
        ],
        "primary_metrics": ["EA", "ER", "CLC-Ans", "CLC-Struct"],
        "aux_metrics": ["NormEM", "AnswerF1"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _isolated_logger(*args, **kwargs):  # type: ignore[no-untyped-def]
    logger = logging.getLogger("pytest.isolated.eval")
    logger.handlers = []
    logger.propagate = True
    return logger


class FakeExecCache:
    def __init__(self, table: Dict[str, Dict[str, Any]]) -> None:
        self.table = table

    def execute(self, sparql_query: str, offline_only: bool = False) -> Dict[str, Any]:
        query = sparql_query.strip()
        if query in self.table:
            return self.table[query]
        raise FileNotFoundError(f"cache miss: {query}")


def test_parse_language_list() -> None:
    assert _parse_language_list(None) is None
    assert _parse_language_list("en,de,es") == ["en", "de", "es"]
    assert _parse_language_list(" en , , ru ") == ["en", "ru"]


def test_compute_all_metrics_offline_cache_miss_raises_runtime_error() -> None:
    cache = FakeExecCache(
        {
            "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }": {
                "ok": True,
                "normalized_answers": ["x=http://www.wikidata.org/entity/Q1"],
            },
        }
    )
    results = [
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
            "language": "en",
            "qid": "q1",
        }
    ]
    with pytest.raises(RuntimeError, match="Offline evaluation cache miss detected"):
        compute_all_metrics(results, cache=cache, offline_only=True)  # type: ignore[arg-type]


def test_compute_all_metrics_offline_cache_miss_can_continue() -> None:
    cache = FakeExecCache(
        {
            "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }": {
                "ok": True,
                "normalized_answers": ["x=http://www.wikidata.org/entity/Q1"],
            },
        }
    )
    results = [
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
            "language": "en",
            "qid": "q1",
        }
    ]
    metrics = compute_all_metrics(
        results,
        cache=cache,  # type: ignore[arg-type]
        offline_only=True,
        fail_on_cache_miss=False,
    )
    assert metrics["total_samples"] == 1
    assert metrics["execution_accuracy"] == 0.0
    assert metrics["error_distribution"]["execution_error"] == 1


def test_compute_all_metrics_normal_path() -> None:
    cache = FakeExecCache(
        {
            "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }": {
                "ok": True,
                "normalized_answers": ["x=http://www.wikidata.org/entity/Q1"],
            },
            "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }": {
                "ok": True,
                "normalized_answers": ["x=http://www.wikidata.org/entity/Q2"],
            },
        }
    )
    results = [
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "language": "en",
            "qid": "q_same",
        },
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
            "language": "de",
            "qid": "q_diff",
        },
    ]
    metrics = compute_all_metrics(results, cache=cache, offline_only=True)  # type: ignore[arg-type]

    assert metrics["total_samples"] == 2
    assert metrics["ea_count"] == 1
    assert metrics["executable_count"] == 2
    assert metrics["execution_accuracy"] == 0.5
    assert set(metrics["per_language"].keys()) == {"en", "de"}


def test_compute_all_metrics_error_detail_extended_schema() -> None:
    cache = FakeExecCache(
        {
            "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }": {
                "ok": True,
                "normalized_answers": ["x=http://www.wikidata.org/entity/Q1"],
            },
            "SELECT BAD WHERE {": {
                "ok": False,
                "error": "Parse error near WHERE",
                "error_type": "ParserError",
            },
        }
    )
    results = [
        {
            "pred_sparql": "SELECT BAD WHERE {",
            "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "language": "en",
            "qid": "q_parse",
        }
    ]
    compute_all_metrics(results, cache=cache, offline_only=True)  # type: ignore[arg-type]
    detail = results[0]["exec_info"]["error_detail"]
    assert detail["stage"] == "pred_execute"
    assert detail["code"] == "pred_query_failed"
    assert "raw_exception" in detail
    assert "pred_executable" in detail
    assert "gold_executable" in detail
    assert isinstance(detail["pred_executable"], bool)
    assert isinstance(detail["gold_executable"], bool)


def test_normalize_prediction_record_requires_gold() -> None:
    rec = {"pred_sparql": "SELECT ?x WHERE {?x wdt:P31 wd:Q5}"}
    assert _normalize_prediction_record(rec, idx=0) is None


def test_normalize_prediction_record_handles_none_values() -> None:
    rec = {
        "idx": "7",
        "qid": "q7",
        "language": None,
        "question": None,
        "gold_sparql": None,
        "sparql": "SELECT ?x WHERE {?x wdt:P31 wd:Q5}",
        "pred_sparql": None,
        "prediction": "ASK { wd:Q1 wdt:P31 ?x }",
    }
    normalized = _normalize_prediction_record(rec, idx=1)
    assert normalized is not None
    assert normalized["idx"] == 7
    assert normalized["language"] == "unk"
    assert normalized["question"] == ""
    assert normalized["gold_sparql"].startswith("SELECT")
    assert normalized["pred_sparql"].startswith("ASK")
    assert normalized["mode"] == "adapter"


def test_load_predictions_jsonl_and_language_filter(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "preds.jsonl"
    rows = [
        {
            "idx": 0,
            "qid": "q1",
            "language": "en",
            "question": "q",
            "gold_sparql": "SELECT ?x WHERE {?x wdt:P31 wd:Q5}",
            "pred_sparql": "SELECT ?x WHERE {?x wdt:P31 wd:Q5}",
        },
        {
            "idx": 1,
            "qid": "q2",
            "language": "de",
            "question": "q",
            "sparql": "SELECT ?x WHERE {?x wdt:P31 wd:Q146}",
            "prediction": "SELECT ?x WHERE {?x wdt:P31 wd:Q146}",
        },
        {
            "idx": 2,
            "qid": "q3",
            "language": "ru",
            "question": "q",
            "pred_sparql": "SELECT ?x WHERE {?x wdt:P31 wd:Q1}",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    loaded = _load_predictions_jsonl(str(path))
    assert len(loaded) == 2
    assert loaded[1]["gold_sparql"].startswith("SELECT")
    assert loaded[1]["pred_sparql"].startswith("SELECT")

    filtered = _filter_by_languages(loaded, ["de"])
    assert len(filtered) == 1
    assert filtered[0]["language"] == "de"


def test_compute_all_metrics_clc_requires_complete_language_group() -> None:
    cache = FakeExecCache(
        {
            "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }": {"ok": True, "normalized_answers": ["x=Q1"]},
            "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }": {"ok": True, "normalized_answers": ["x=Q2"]},
        }
    )
    results = [
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "language": "en",
            "qid": "q_complete",
            "mode": "adapter",
        },
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            "language": "de",
            "qid": "q_complete",
            "mode": "adapter",
        },
        {
            "pred_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
            "gold_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
            "language": "en",
            "qid": "q_incomplete",
            "mode": "adapter",
        },
    ]
    metrics = compute_all_metrics(
        results,
        cache=cache,  # type: ignore[arg-type]
        offline_only=True,
        expected_languages=["en", "de"],
    )
    clc = metrics["cross_lingual_consistency"]
    assert clc["num_groups"] == 1
    assert clc["incomplete_group_count"] == 1
    assert clc["incomplete_groups"][0]["qid"] == "q_incomplete"
    assert clc["incomplete_groups"][0]["missing_languages"] == ["de"]


def test_batch_generate_includes_adapter_mode(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(eval_sparql_mod, "generate_sparql", lambda *args, **kwargs: "ASK { ?s ?p ?o }")
    out = batch_generate(
        model=object(),
        tokenizer=object(),
        test_data=[
            {
                "qid": "q1",
                "language": "en",
                "question": "q1",
                "sparql": "ASK { ?s ?p ?o }",
            }
        ],
    )
    assert len(out) == 1
    assert out[0]["mode"] == "adapter"


def test_eval_main_predictions_mode_smoke(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    protocol_path = tmp_path / "eval_protocol.yaml"
    cache_dir = str(tmp_path / "cache")
    _write_eval_protocol(
        protocol_path,
        test_data_path=str(tmp_path / "test_scope.jsonl"),
        test_languages=["en", "de"],
        cache_dir=cache_dir,
    )

    pred_path = tmp_path / "preds.jsonl"
    run_name = "predictions_mode_smoke"
    protocol_id = "unit_test_protocol:v1"
    run_id = build_run_id(
        run_name=run_name,
        mode="adapter",
        protocol_id=protocol_id,
        seed=42,
    )
    _write_jsonl(
        pred_path,
        [
            {
                "idx": 0,
                "qid": "q1",
                "language": "en",
                "question": "q1",
                "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "mode": "adapter",
                "run_id": run_id,
                "protocol_id": protocol_id,
            },
            {
                "idx": 1,
                "qid": "q2",
                "language": "de",
                "question": "q2",
                "gold_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
                "pred_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
                "mode": "adapter",
                "run_id": run_id,
                "protocol_id": protocol_id,
            },
        ],
    )
    run_metadata_path = tmp_path / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_name,
                "mode": "adapter",
                "method": "adapter",
                "model_name_or_path": "dummy-model",
                "adapter_path": "dummy-adapter",
                "test_data_path": str(tmp_path / "test_scope.jsonl"),
                "test_languages": ["en", "de"],
                "max_seq_len": 512,
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "cache_dir": cache_dir,
                "offline_only": True,
                "result_partition": "unified_codechain",
                "protocol_name": "unit_test_protocol",
                "protocol_version": "v1",
                "protocol_id": protocol_id,
                "seed": 42,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    table = {
        "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }": {"ok": True, "normalized_answers": ["x=Q1"]},
        "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }": {"ok": True, "normalized_answers": ["x=Q2"]},
    }

    class _FakeCache:
        def __init__(self, cache_dir: str, force_offline: bool = False) -> None:
            self.cache_dir = cache_dir
            self.force_offline = force_offline

        def execute(self, sparql_query: str, offline_only: bool = False) -> Dict[str, Any]:
            q = sparql_query.strip()
            if q in table:
                return table[q]
            raise FileNotFoundError(f"cache miss: {q}")

    args = SimpleNamespace(
        config=None,
        eval_protocol=str(protocol_path),
        predictions_file=str(pred_path),
        run_metadata_file=str(run_metadata_path),
        adapter_path=None,
        model_name=None,
        test_data=None,
        cache_dir=cache_dir,
        output_dir=str(tmp_path / "eval_out"),
        offline_only=True,
        max_samples=None,
        max_new_tokens=None,
        max_seq_len=None,
        do_sample=False,
        temperature=None,
        top_p=None,
        quantization=None,
        test_languages="en,de",
        seed=42,
    )
    monkeypatch.setattr(eval_sparql_mod, "parse_eval_args", lambda: args)
    monkeypatch.setattr(eval_sparql_mod, "SPARQLCache", _FakeCache)
    monkeypatch.setattr(eval_sparql_mod, "setup_logging", _isolated_logger)

    eval_sparql_mod.main()

    metrics_path = tmp_path / "eval_out" / "metrics.json"
    preds_out_path = tmp_path / "eval_out" / "predictions.jsonl"
    assert metrics_path.exists()
    assert preds_out_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["total_samples"] == 2
    assert metrics["ea_count"] == 2
    assert metrics["execution_accuracy"] == 1.0
    assert "eval_protocol" in metrics
    assert set(metrics.get("error_type_schema", [])) == {
        "generation_empty",
        "syntax_or_parse_error",
        "execution_error",
        "wrong_answer",
    }


def test_eval_main_adapter_mode_smoke_with_mocks(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_path = tmp_path / "test.jsonl"
    protocol_path = tmp_path / "eval_protocol.yaml"
    cache_dir = str(tmp_path / "cache")
    _write_eval_protocol(
        protocol_path,
        test_data_path=str(test_path),
        test_languages=["en"],
        cache_dir=cache_dir,
        max_seq_len=128,
        max_new_tokens=64,
    )
    _write_jsonl(
        test_path,
        [
            {
                "qid": "q1",
                "language": "en",
                "question": "q1",
                "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            }
        ],
    )

    calls = {"load_model": 0, "batch_generate": 0}

    def _fake_load_model_for_eval(**kwargs):  # type: ignore[no-untyped-def]
        calls["load_model"] += 1
        return "model", "tokenizer"

    def _fake_batch_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["batch_generate"] += 1
        expected_run_id = build_run_id(
            run_name="eval_adapter_out",
            mode="adapter",
            protocol_id="unit_test_protocol:v1",
            seed=7,
        )
        return [
            {
                "idx": 0,
                "qid": "q1",
                "language": "en",
                "question": "q1",
                "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "generation_time_sec": 0.01,
                "mode": "adapter",
                "run_id": expected_run_id,
                "protocol_id": "unit_test_protocol:v1",
            }
        ]

    class _FakeCache:
        def __init__(self, cache_dir: str, force_offline: bool = False) -> None:
            pass

        def execute(self, sparql_query: str, offline_only: bool = False) -> Dict[str, Any]:
            return {"ok": True, "normalized_answers": ["x=Q1"]}

    args = SimpleNamespace(
        config=None,
        eval_protocol=str(protocol_path),
        predictions_file=None,
        run_metadata_file=None,
        adapter_path="dummy-adapter",
        model_name="dummy-model",
        test_data=str(test_path),
        cache_dir=cache_dir,
        output_dir=str(tmp_path / "eval_adapter_out"),
        offline_only=True,
        max_samples=None,
        max_new_tokens=64,
        max_seq_len=128,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        quantization="none",
        test_languages="en",
        seed=7,
    )
    monkeypatch.setattr(eval_sparql_mod, "parse_eval_args", lambda: args)
    monkeypatch.setattr(eval_sparql_mod, "load_model_for_eval", _fake_load_model_for_eval)
    monkeypatch.setattr(eval_sparql_mod, "batch_generate", _fake_batch_generate)
    monkeypatch.setattr(eval_sparql_mod, "SPARQLCache", _FakeCache)
    monkeypatch.setattr(eval_sparql_mod, "setup_logging", _isolated_logger)

    eval_sparql_mod.main()

    assert calls["load_model"] == 1
    assert calls["batch_generate"] == 1
    metrics = json.loads((tmp_path / "eval_adapter_out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["total_samples"] == 1
    assert metrics["ea_count"] == 1
    assert "eval_protocol" in metrics


def test_eval_main_predictions_missing_file_raises(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_data_path = tmp_path / "test.jsonl"
    _write_jsonl(
        test_data_path,
        [{"qid": "q1", "language": "en", "question": "q", "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"}],
    )
    protocol_path = tmp_path / "eval_protocol.yaml"
    _write_eval_protocol(
        protocol_path,
        test_data_path=str(test_data_path),
        test_languages=["en"],
        cache_dir=str(tmp_path / "cache"),
    )

    args = SimpleNamespace(
        config=None,
        eval_protocol=str(protocol_path),
        predictions_file=str(tmp_path / "missing_preds.jsonl"),
        run_metadata_file=None,
        adapter_path=None,
        model_name=None,
        test_data=None,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "eval_out"),
        offline_only=True,
        max_samples=None,
        max_new_tokens=None,
        max_seq_len=None,
        do_sample=False,
        temperature=None,
        top_p=None,
        quantization=None,
        test_languages=None,
        seed=42,
    )
    monkeypatch.setattr(eval_sparql_mod, "parse_eval_args", lambda: args)
    monkeypatch.setattr(eval_sparql_mod, "setup_logging", _isolated_logger)

    with pytest.raises(FileNotFoundError, match="Predictions file not found"):
        eval_sparql_mod.main()


def test_eval_main_invalid_sampling_args_raises(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_data_path = tmp_path / "test.jsonl"
    _write_jsonl(
        test_data_path,
        [{"qid": "q1", "language": "en", "question": "q", "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"}],
    )
    protocol_path = tmp_path / "eval_protocol.yaml"
    _write_eval_protocol(
        protocol_path,
        test_data_path=str(test_data_path),
        test_languages=["en"],
        cache_dir=str(tmp_path / "cache"),
    )

    pred_path = tmp_path / "preds.jsonl"
    _write_jsonl(
        pred_path,
        [
            {
                "idx": 0,
                "qid": "q1",
                "language": "en",
                "question": "q",
                "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
            }
        ],
    )

    args = SimpleNamespace(
        config=None,
        eval_protocol=str(protocol_path),
        predictions_file=str(pred_path),
        run_metadata_file=None,
        adapter_path=None,
        model_name=None,
        test_data=None,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "eval_out"),
        offline_only=True,
        max_samples=None,
        max_new_tokens=64,
        max_seq_len=128,
        do_sample=True,
        temperature=1.0,
        top_p=1.5,
        quantization=None,
        test_languages="en",
        seed=42,
    )
    monkeypatch.setattr(eval_sparql_mod, "parse_eval_args", lambda: args)
    monkeypatch.setattr(eval_sparql_mod, "setup_logging", _isolated_logger)

    with pytest.raises(ValueError, match="top_p must be in \\(0, 1\\]"):
        eval_sparql_mod.main()
