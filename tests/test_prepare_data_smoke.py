import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_cache_hit(cache_dir: Path, sparql: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(sparql.strip().encode("utf-8")).hexdigest()
    payload = {
        "query": sparql.strip(),
        "ok": True,
        "status": "success",
        "error": "",
        "endpoint": "offline-test",
        "from_cache": False,
        "normalized_answers": [],
        "raw": {},
        "timestamp": 0,
    }
    with (cache_dir / f"{key}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _qald_item(qid: str, sparql: str, langs: Dict[str, str]) -> Dict[str, Any]:
    return {
        "id": qid,
        "question": [{"language": lang, "string": text} for lang, text in langs.items()],
        "query": {"sparql": sparql},
    }


def _signature(row: Dict[str, Any]) -> str:
    question = " ".join(str(row.get("question", "")).strip().split()).lower()
    sparql = " ".join(str(row.get("sparql", "")).strip().split()).lower()
    if not question or not sparql:
        return ""
    return hashlib.md5(f"{question}\t{sparql}".encode("utf-8")).hexdigest()


def test_prepare_data_offline_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "prepare_data.py"

    lcquad_src = tmp_path / "raw" / "lcquad"
    qald_src = tmp_path / "raw" / "qald"
    output_dir = tmp_path / "out"
    cache_dir = output_dir / "cache"

    _write_json(
        lcquad_src / "lcquad2_train.json",
        [
            {"question": "who is q1", "sparql": "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"},
            {"question": "who is q2", "sparql": "SELECT ?x WHERE { wd:Q2 wdt:P31 ?x }"},
            {"question": "who is q3", "sparql": "SELECT ?x WHERE { wd:Q3 wdt:P31 ?x }"},
        ],
    )

    train_queries: List[str] = [
        "SELECT ?x WHERE { wd:Q10 wdt:P31 ?x }",
        "SELECT ?x WHERE { wd:Q20 wdt:P31 ?x }",
    ]
    test_queries: List[str] = [
        "SELECT ?x WHERE { wd:Q30 wdt:P31 ?x }",
    ]
    _write_json(
        qald_src / "qald_9_plus_train_wikidata.json",
        {
            "questions": [
                _qald_item("t1", train_queries[0], {"en": "t1 en", "de": "t1 de", "es": "t1 es"}),
                _qald_item("t2", train_queries[1], {"en": "t2 en", "de": "t2 de", "es": "t2 es"}),
            ]
        },
    )
    _write_json(
        qald_src / "qald_9_plus_test_wikidata.json",
        {
            "questions": [
                _qald_item("u1", test_queries[0], {"en": "u1 en", "de": "u1 de", "es": "u1 es"}),
            ]
        },
    )

    for query in train_queries + test_queries:
        _write_cache_hit(cache_dir, query)

    cmd = [
        sys.executable,
        str(script_path),
        "--output-dir",
        str(output_dir),
        "--lcquad-source",
        str(lcquad_src),
        "--qald-source",
        str(qald_src),
        "--offline-only",
        "--qald-test-languages",
        "en,de,es,ru",
        "--allow-incomplete-test-languages",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"prepare_data smoke failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    expected_files = [
        output_dir / "lcquad2_stage1_train.jsonl",
        output_dir / "lcquad2_stage1_dev.jsonl",
        output_dir / "qald9plus_stage2_train.jsonl",
        output_dir / "qald9plus_stage2_dev.jsonl",
        output_dir / "qald9plus_test.jsonl",
        output_dir / "qald9plus_icl_few_shot_pool.jsonl",
        output_dir / "prepare_stats.json",
    ]
    for file_path in expected_files:
        assert file_path.exists(), f"missing output file: {file_path}"
    assert not (output_dir / "qald9plus_high_stakes_test.jsonl").exists()
    assert not (output_dir / "archive" / "qald9plus_high_stakes_test.jsonl").exists()

    stats = json.loads((output_dir / "prepare_stats.json").read_text(encoding="utf-8"))
    assert stats["lcquad_train_samples"] > 0
    assert stats["qald_train_samples"] > 0
    assert stats["qald_test_samples"] > 0
    assert stats["qald_high_stakes_status"] in {"ok", "skipped_offline_cache_miss", "skipped_disabled"}
    assert stats["qald_high_stakes_path"] == ""
    assert stats["qald_icl_few_shot_pool_samples"] > 0

    pool_path = output_dir / "qald9plus_icl_few_shot_pool.jsonl"
    test_path = output_dir / "qald9plus_test.jsonl"
    pool_rows = [
        json.loads(line)
        for line in pool_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_rows = [
        json.loads(line)
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool_qids: Set[str] = {str(row.get("qid", "")).strip() for row in pool_rows if str(row.get("qid", "")).strip()}
    test_qids: Set[str] = {str(row.get("qid", "")).strip() for row in test_rows if str(row.get("qid", "")).strip()}
    assert not (pool_qids & test_qids)

    pool_signatures = {sig for sig in (_signature(row) for row in pool_rows) if sig}
    test_signatures = {sig for sig in (_signature(row) for row in test_rows) if sig}
    assert not (pool_signatures & test_signatures)
