import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.data_sparql import (
    format_text2sparql_infer,
    format_text2sparql_train,
    load_lcquad2,
    load_lcquad2_from_local,
    load_qald9plus_test,
    load_qald9plus_train,
    split_qald_train_dev,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_qald_question_item(qid: str, sparql: str, langs: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": qid,
        "question": [{"language": k, "string": v} for k, v in langs.items()],
        "query": {"sparql": sparql},
    }


def test_load_lcquad2_from_local_success_and_dedupe(tmp_path: Path) -> None:
    lcquad_file = tmp_path / "lcquad2_train.json"
    _write_json(
        lcquad_file,
        [
            {
                "corrected_question": "Who is the spouse of Barack Obama?",
                "sparql_wikidata": "SELECT  ?x  WHERE { wd:Q76 wdt:P26 ?x }",
            },
            {
                "corrected_question": "Who is the spouse of Barack Obama?",
                "sparql_wikidata": "SELECT  ?x  WHERE { wd:Q76 wdt:P26 ?x }",
            },
            {
                "question": "invalid without sparql",
            },
        ],
    )

    data = load_lcquad2_from_local(str(tmp_path))
    assert len(data) == 1
    assert data[0]["question"] == "Who is the spouse of Barack Obama?"
    assert data[0]["sparql"] == "SELECT ?x WHERE { wd:Q76 wdt:P26 ?x }"


def test_load_lcquad2_from_local_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_lcquad2_from_local("Z:/this/path/does/not/exist")


def test_load_lcquad2_split_reproducible(tmp_path: Path) -> None:
    rows: List[Dict[str, str]] = []
    for idx in range(10):
        rows.append(
            {
                "question": f"q_{idx}",
                "sparql": f"SELECT ?x WHERE {{ wd:Q{idx+1} wdt:P31 ?x }}",
            }
        )
    _write_json(tmp_path / "lcquad2_raw.json", rows)

    train_a, dev_a = load_lcquad2(str(tmp_path), seed=123)
    train_b, dev_b = load_lcquad2(str(tmp_path), seed=123)

    assert train_a == train_b
    assert dev_a == dev_b
    assert len(train_a) == 9
    assert len(dev_a) == 1


def test_load_qald9plus_train_test_language_filter_and_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    train_payload = {
        "questions": [
            _build_qald_question_item(
                "q1",
                "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }",
                {"en": "q1 en", "de": "q1 de", "es": "q1 es"},
            ),
            _build_qald_question_item(
                "q2",
                "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }",
                {"en": "q2 en", "de": "q2 de", "es": "q2 es"},
            ),
        ]
    }
    test_payload = {
        "questions": [
            _build_qald_question_item(
                "q3",
                "SELECT ?x WHERE { wd:Q3 wdt:P31 ?x }",
                {"en": "q3 en", "de": "q3 de", "es": "q3 es"},
            )
        ]
    }
    _write_json(tmp_path / "qald_9_plus_train_wikidata.json", train_payload)
    _write_json(tmp_path / "qald_9_plus_test_wikidata.json", test_payload)

    train_data = load_qald9plus_train(str(tmp_path), languages=["en", "de"])
    assert len(train_data) == 4
    assert set(item["language"] for item in train_data) == {"en", "de"}
    assert set(item["qid"] for item in train_data) == {"q1", "q2"}

    caplog.clear()
    test_data = load_qald9plus_test(str(tmp_path), languages=None)
    assert len(test_data) == 3
    assert set(item["language"] for item in test_data) == {"en", "de", "es"}
    assert "missing language='ru'" in caplog.text


def test_split_qald_train_dev_grouped_by_qid() -> None:
    train_data = [
        {"question": "q1 en", "sparql": "s1", "language": "en", "qid": "q1"},
        {"question": "q1 de", "sparql": "s1", "language": "de", "qid": "q1"},
        {"question": "q2 en", "sparql": "s2", "language": "en", "qid": "q2"},
        {"question": "q2 de", "sparql": "s2", "language": "de", "qid": "q2"},
    ]
    train_split, dev_split = split_qald_train_dev(train_data, dev_ratio=0.5, seed=42)

    train_qids = {row["qid"] for row in train_split}
    dev_qids = {row["qid"] for row in dev_split}
    assert train_qids
    assert dev_qids
    assert train_qids.isdisjoint(dev_qids)
    assert len(train_split) + len(dev_split) == len(train_data)


def test_format_text2sparql_train_and_infer() -> None:
    train_prompt = format_text2sparql_train("Who is Q42?", "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }")
    assert "prompt_text" in train_prompt
    assert "messages" in train_prompt
    assert train_prompt["messages"][-1]["role"] == "assistant"

    infer_prompt = format_text2sparql_infer(
        "Who is Q1?",
        few_shot_examples=[{"question": "Who is Q2?", "sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }"}],
    )
    assert "prompt_text" in infer_prompt
    assert infer_prompt["messages"][0]["role"] == "system"
    assert infer_prompt["messages"][-1]["role"] == "user"


def test_format_text2sparql_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        format_text2sparql_train("", "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }")
    with pytest.raises(ValueError):
        format_text2sparql_infer("")
