import hashlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.sparql_executor import SPARQLCache


class DummySPARQLCache(SPARQLCache):
    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir=cache_dir, min_interval_sec=0.0)
        self.remote_calls: List[str] = []

    def _remote_query(self, sparql_query: str) -> Dict[str, Any]:
        self.remote_calls.append(sparql_query)
        if "FAIL" in sparql_query:
            raise RuntimeError("forced remote failure")
        if sparql_query.strip().upper().startswith("ASK"):
            return {"boolean": True}
        return {
            "head": {"vars": ["x"]},
            "results": {
                "bindings": [
                    {
                        "x": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/Q42",
                        }
                    }
                ]
            },
        }


def _cache_file_for_query(cache_dir: Path, query: str) -> Path:
    key = hashlib.md5(query.strip().encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def test_execute_cache_first_and_md5_cache_key(tmp_path: Path) -> None:
    cache = DummySPARQLCache(str(tmp_path))
    query = "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }"

    first = cache.execute(query, offline_only=False)
    assert first["ok"] is True
    assert first["from_cache"] is False
    assert len(cache.remote_calls) == 1

    cache_file = _cache_file_for_query(tmp_path, query)
    assert cache_file.exists()

    second = cache.execute(query, offline_only=False)
    assert second["ok"] is True
    assert second["from_cache"] is True
    assert len(cache.remote_calls) == 1


def test_execute_offline_cache_miss_raises(tmp_path: Path) -> None:
    cache = DummySPARQLCache(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        cache.execute("SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }", offline_only=True)


def test_execute_failed_remote_not_cached(tmp_path: Path) -> None:
    cache = DummySPARQLCache(str(tmp_path))
    query = "FAIL SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }"

    first = cache.execute(query, offline_only=False)
    assert first["ok"] is False
    assert first["status"] == "error"
    assert len(cache.remote_calls) == 1
    assert not _cache_file_for_query(tmp_path, query).exists()

    second = cache.execute(query, offline_only=False)
    assert second["ok"] is False
    assert len(cache.remote_calls) == 2


def test_pre_cache_gold_deduplicates_queries(tmp_path: Path) -> None:
    cache = DummySPARQLCache(str(tmp_path))
    dataset = [
        {"sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"},
        {"sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"},
        {"sparql": "ASK { wd:Q2 wdt:P31 ?x }"},
        {"sparql": ""},
    ]

    cache.pre_cache_gold(dataset)
    assert len(cache.remote_calls) == 2
