from typing import Any, Dict

from src.entity_filter import extract_entities, filter_high_stakes_subset, get_entity_types


class FakeCache:
    def execute(self, sparql_query: str, offline_only: bool = False) -> Dict[str, Any]:
        if "wd:Q100" in sparql_query:
            return {
                "ok": True,
                "raw": {
                    "results": {
                        "bindings": [
                            {
                                "type": {
                                    "type": "uri",
                                    "value": "http://www.wikidata.org/entity/Q12136",
                                }
                            }
                        ]
                    }
                },
                "normalized_answers": [],
            }
        if "wd:Q200" in sparql_query:
            return {
                "ok": True,
                "raw": {"results": {"bindings": []}},
                "normalized_answers": [
                    "type=http://www.wikidata.org/entity/Q7748",
                ],
            }
        if "wd:Q300" in sparql_query:
            return {
                "ok": False,
                "error": "forced failure",
                "raw": {},
                "normalized_answers": [],
            }
        return {"ok": True, "raw": {"results": {"bindings": []}}, "normalized_answers": []}


def test_extract_entities_prefix_and_uri_dedupe() -> None:
    sparql = """
    SELECT ?x WHERE {
      wd:Q42 wdt:P31 ?x .
      wd:Q42 wdt:P279 ?x .
      <http://www.wikidata.org/entity/Q100> wdt:P31 ?x .
    }
    """
    entities = extract_entities(sparql)
    assert entities == ["Q42", "Q100"]


def test_get_entity_types_raw_and_fallback() -> None:
    cache = FakeCache()
    result = get_entity_types(["Q100", "wd:Q200", "Q300", "INVALID"], cache)  # type: ignore[arg-type]

    assert result["Q100"] == ["Q12136"]
    assert result["Q200"] == ["Q7748"]
    assert result["Q300"] == []
    assert "INVALID" not in result


def test_filter_high_stakes_subset_by_keyword_and_type() -> None:
    cache = FakeCache()
    data = [
        {
            "question": "What medicine treats this symptom?",
            "sparql": "SELECT ?x WHERE { wd:Q300 wdt:P31 ?x }",
            "language": "en",
            "qid": "k1",
        },
        {
            "question": "Question without explicit risk keywords",
            "sparql": "SELECT ?x WHERE { wd:Q200 wdt:P31 ?x }",
            "language": "en",
            "qid": "k2",
        },
        {
            "question": "Neutral question",
            "sparql": "SELECT ?x WHERE { wd:Q999 wdt:P31 ?x }",
            "language": "en",
            "qid": "k3",
        },
    ]
    subset = filter_high_stakes_subset(data, cache=cache)  # type: ignore[arg-type]
    qids = {item["qid"] for item in subset}

    assert qids == {"k1", "k2"}
    by_qid = {item["qid"]: item for item in subset}
    assert by_qid["k1"]["risk_tags"] == ["medical"]
    assert by_qid["k2"]["risk_tags"] == ["legal"]
