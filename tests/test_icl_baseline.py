import json
import logging
from types import SimpleNamespace
from pathlib import Path

import pytest

from src import run_icl_baseline as icl_baseline
from src.run_icl_baseline import (
    _build_few_shot_pool,
    _collect_non_empty_qids,
    _normalize_mode,
    _sample_few_shot_examples,
    _validate_no_qid_overlap,
    _validate_few_shot_pool_path,
)


def _write_jsonl(path: Path, rows) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _isolated_logger(*args, **kwargs):  # type: ignore[no-untyped-def]
    logger = logging.getLogger("pytest.isolated.icl")
    logger.handlers = []
    logger.propagate = True
    return logger


def test_normalize_mode() -> None:
    assert _normalize_mode("zero") == "zero"
    assert _normalize_mode("few_shot") == "few"
    with pytest.raises(ValueError):
        _normalize_mode("invalid")


def test_validate_few_shot_pool_path_rejects_test_file(tmp_path: Path) -> None:
    pool = tmp_path / "qald_test.jsonl"
    test_data = tmp_path / "qald9plus_test.jsonl"
    pool.write_text("", encoding="utf-8")
    test_data.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="appears to be a test file"):
        _validate_few_shot_pool_path(str(pool), str(test_data))


def test_build_few_shot_pool_and_sample_deterministic() -> None:
    pool = _build_few_shot_pool(
        [
            {"qid": "q1", "question": "Q1", "sparql": "S1"},
            {"qid": "q2", "question": "Q2", "sparql": "S2"},
            {"qid": "q3", "question": "Q3", "sparql": "S3"},
            {"qid": "q4", "question": "Q4", "sparql": "S4"},
        ]
    )
    s1 = _sample_few_shot_examples(
        pool,
        current_qid="q1",
        forbidden_qids={"q1", "qx"},
        k=2,
        seed=42,
        sample_index=7,
    )
    s2 = _sample_few_shot_examples(
        pool,
        current_qid="q1",
        forbidden_qids={"q1", "qx"},
        k=2,
        seed=42,
        sample_index=7,
    )
    assert s1 == s2
    assert len(s1) == 2
    assert all(item["question"] != "Q1" for item in s1)


def test_build_few_shot_pool_requires_valid_pairs() -> None:
    with pytest.raises(ValueError, match="no valid question/sparql pairs"):
        _build_few_shot_pool([{"qid": "q1", "question": "Q1", "sparql": ""}])


def test_collect_non_empty_qids() -> None:
    records = [{"qid": "q1"}, {"qid": ""}, {"qid": " q2 "}, {}]
    assert _collect_non_empty_qids(records) == {"q1", "q2"}


def test_validate_no_qid_overlap_rejects_leakage() -> None:
    pool_data = [{"qid": "q1"}, {"qid": "q2"}]
    test_data = [{"qid": "q2"}, {"qid": "q3"}]
    with pytest.raises(ValueError, match="qid overlap"):
        _validate_no_qid_overlap(pool_data=pool_data, test_data=test_data)


def test_main_zero_shot_smoke(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_data_path = tmp_path / "test.jsonl"
    _write_jsonl(
        test_data_path,
        [
            {"qid": "q1", "language": "en", "question": "q1", "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"},
            {"qid": "q2", "language": "de", "question": "q2", "sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }"},
            {"qid": "q3", "language": "ru", "question": "q3", "sparql": "SELECT ?x WHERE { wd:Q3 wdt:P31 ?x }"},
            {"qid": "q4", "language": "en", "question": "", "sparql": "SELECT ?x WHERE { wd:Q4 wdt:P31 ?x }"},
        ],
    )

    args = SimpleNamespace(
        config=None,
        model_name="dummy-model",
        test_data=str(test_data_path),
        output_dir=str(tmp_path / "outputs"),
        run_name="icl_zero_smoke",
        mode="zero",
        few_shot_pool=None,
        few_shot_k=None,
        few_shot_seed=None,
        quantization="none",
        precision="bf16",
        max_seq_len=128,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        test_languages="en,de",
        max_samples=None,
        seed=123,
    )
    monkeypatch.setattr(icl_baseline, "parse_args", lambda: args)
    monkeypatch.setattr(icl_baseline, "load_base_model_for_icl", lambda **kwargs: ("m", "t"))
    monkeypatch.setattr(icl_baseline, "generate_sparql", lambda **kwargs: "ASK { wd:Q1 wdt:P31 ?x }")
    monkeypatch.setattr(icl_baseline, "setup_logging", _isolated_logger)

    icl_baseline.main()

    run_dir = tmp_path / "outputs" / "icl_zero_smoke"
    preds_path = run_dir / "predictions.jsonl"
    report_path = run_dir / "run_report.json"
    assert preds_path.exists()
    assert report_path.exists()

    preds = [json.loads(line) for line in preds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(preds) == 2
    assert {row["language"] for row in preds} == {"en", "de"}
    assert all(row["mode"] == "icl_zero" for row in preds)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_predictions"] == 2
    assert report["mode"] == "icl_zero"
    assert report["seed"] == 123


def test_main_few_shot_missing_pool_raises(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_data_path = tmp_path / "test.jsonl"
    _write_jsonl(
        test_data_path,
        [{"qid": "q1", "language": "en", "question": "q1", "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"}],
    )
    args = SimpleNamespace(
        config=None,
        model_name="dummy-model",
        test_data=str(test_data_path),
        output_dir=str(tmp_path / "outputs"),
        run_name="icl_few_should_fail",
        mode="few",
        few_shot_pool=None,
        few_shot_k=1,
        few_shot_seed=42,
        quantization="none",
        precision="bf16",
        max_seq_len=128,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        test_languages="en",
        max_samples=None,
        seed=42,
    )
    monkeypatch.setattr(icl_baseline, "parse_args", lambda: args)
    monkeypatch.setattr(icl_baseline, "setup_logging", _isolated_logger)

    with pytest.raises(ValueError, match="few-shot mode requires few_shot_pool_path"):
        icl_baseline.main()


def test_main_few_shot_smoke(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    test_data_path = tmp_path / "test.jsonl"
    pool_path = tmp_path / "pool.jsonl"
    _write_jsonl(
        test_data_path,
        [
            {"qid": "q1", "language": "en", "question": "test-q1", "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"},
            {"qid": "q2", "language": "en", "question": "test-q2", "sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }"},
        ],
    )
    _write_jsonl(
        pool_path,
        [
            {"qid": "p1", "language": "en", "question": "pool-q1", "sparql": "SELECT ?x WHERE { wd:Q10 wdt:P31 ?x }"},
            {"qid": "p2", "language": "en", "question": "pool-q2", "sparql": "SELECT ?x WHERE { wd:Q20 wdt:P31 ?x }"},
            {"qid": "p3", "language": "en", "question": "pool-q3", "sparql": "SELECT ?x WHERE { wd:Q30 wdt:P31 ?x }"},
        ],
    )

    args = SimpleNamespace(
        config=None,
        model_name="dummy-model",
        test_data=str(test_data_path),
        output_dir=str(tmp_path / "outputs"),
        run_name="icl_few_smoke",
        mode="few",
        few_shot_pool=str(pool_path),
        few_shot_k=2,
        few_shot_seed=42,
        quantization="none",
        precision="bf16",
        max_seq_len=128,
        max_new_tokens=32,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        test_languages="en",
        max_samples=None,
        seed=999,
    )
    monkeypatch.setattr(icl_baseline, "parse_args", lambda: args)
    monkeypatch.setattr(icl_baseline, "load_base_model_for_icl", lambda **kwargs: ("m", "t"))
    monkeypatch.setattr(icl_baseline, "setup_logging", _isolated_logger)

    seen_examples = []

    def _fake_generate(**kwargs):  # type: ignore[no-untyped-def]
        examples = kwargs.get("few_shot_examples") or []
        seen_examples.append(examples)
        return "ASK { wd:Q1 wdt:P31 ?x }"

    monkeypatch.setattr(icl_baseline, "generate_sparql", _fake_generate)

    icl_baseline.main()

    assert len(seen_examples) == 2
    assert all(len(examples) == 2 for examples in seen_examples)
    for examples in seen_examples:
        assert all(ex["question"].startswith("pool-") for ex in examples)
