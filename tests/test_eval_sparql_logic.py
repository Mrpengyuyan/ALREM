from typing import Any, Dict

import pytest

from src.eval_sparql import _parse_language_list, compute_all_metrics


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
