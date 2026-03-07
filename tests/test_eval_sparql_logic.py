import json
import logging
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict

import pytest

from src import eval_sparql as eval_sparql_mod
from src.eval_sparql import (
    _filter_by_languages,
    _load_predictions_jsonl,
    _normalize_prediction_record,
    _parse_language_list,
    compute_all_metrics,
)


def _write_jsonl(path: Path, rows) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def test_eval_main_predictions_mode_smoke(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pred_path = tmp_path / "preds.jsonl"
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
            },
            {
                "idx": 1,
                "qid": "q2",
                "language": "de",
                "question": "q2",
                "gold_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
                "pred_sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
            },
        ],
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
        predictions_file=str(pred_path),
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


def test_eval_main_adapter_mode_smoke_with_mocks(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_path = tmp_path / "test.jsonl"
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
        return [
            {
                "idx": 0,
                "qid": "q1",
                "language": "en",
                "question": "q1",
                "gold_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "pred_sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                "generation_time_sec": 0.01,
            }
        ]

    class _FakeCache:
        def __init__(self, cache_dir: str, force_offline: bool = False) -> None:
            pass

        def execute(self, sparql_query: str, offline_only: bool = False) -> Dict[str, Any]:
            return {"ok": True, "normalized_answers": ["x=Q1"]}

    args = SimpleNamespace(
        config=None,
        predictions_file=None,
        adapter_path="dummy-adapter",
        model_name="dummy-model",
        test_data=str(test_path),
        cache_dir=str(tmp_path / "cache"),
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


def test_eval_main_predictions_missing_file_raises(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    args = SimpleNamespace(
        config=None,
        predictions_file=str(tmp_path / "missing_preds.jsonl"),
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
