import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sparql import (  # noqa: E402
    load_lcquad2,
    load_qald9plus_test,
    load_qald9plus_train,
    split_qald_train_dev,
)
from src.entity_filter import filter_high_stakes_subset  # noqa: E402
from src.sparql_executor import SPARQLCache  # noqa: E402


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("alrem")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _resolve_path(path_text: str) -> Path:
    path_obj = Path(path_text)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def _parse_languages(csv_text: Optional[str]) -> Optional[List[str]]:
    if csv_text is None:
        return None
    tokens = [token.strip() for token in csv_text.split(",")]
    langs = [token for token in tokens if token]
    return langs if langs else None


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _normalize_signature_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _qa_signature(record: Dict[str, Any]) -> str:
    question = _normalize_signature_text(record.get("question", ""))
    sparql = _normalize_signature_text(record.get("sparql", ""))
    if not question or not sparql:
        return ""
    return hashlib.md5(f"{question}\t{sparql}".encode("utf-8")).hexdigest()


def _collect_signatures(records: List[Dict[str, Any]]) -> Set[str]:
    signatures: Set[str] = set()
    for item in records:
        sig = _qa_signature(item)
        if sig:
            signatures.add(sig)
    return signatures


def _filter_by_signatures(
    records: List[Dict[str, Any]],
    *,
    forbidden_signatures: Set[str],
) -> Tuple[List[Dict[str, Any]], int]:
    if not forbidden_signatures:
        return list(records), 0
    kept: List[Dict[str, Any]] = []
    removed = 0
    for item in records:
        sig = _qa_signature(item)
        if sig and sig in forbidden_signatures:
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def _count_signature_overlap(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> int:
    a_signatures = _collect_signatures(a)
    b_signatures = _collect_signatures(b)
    return len(a_signatures & b_signatures)


def _build_icl_few_shot_pool(
    records: List[Dict[str, Any]],
    *,
    allowed_languages: List[str],
) -> List[Dict[str, Any]]:
    allowed = {str(lang).strip().lower() for lang in allowed_languages if str(lang).strip()}
    out: List[Dict[str, Any]] = []
    for item in records:
        question = str(item.get("question", "")).strip()
        sparql = str(item.get("sparql", "")).strip()
        if not question or not sparql:
            continue
        language = str(item.get("language", "unk")).strip().lower() or "unk"
        if allowed and language not in allowed:
            continue
        qid = str(item.get("qid", "")).strip()
        if qid:
            qid = f"pool_{qid}"
        else:
            sig_input = question + "\t" + sparql
            qid = f"pool_auto_{hashlib.md5(sig_input.encode('utf-8')).hexdigest()[:16]}"
        out.append(
            {
                "question": question,
                "sparql": sparql,
                "language": language,
                "qid": qid,
            }
        )
    return out


def _language_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for item in records:
        lang = str(item.get("language", "unk")).strip().lower() or "unk"
        counter[lang] += 1
    return dict(sorted(counter.items(), key=lambda kv: kv[0]))


def _print_stats(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    logger.info("========== Data Preparation Summary ==========")
    for key, value in stats.items():
        logger.info("%s: %s", key, value)
    logger.info("=============================================")


def _collect_unique_sparql_queries(records: List[Dict[str, Any]]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for item in records:
        query = str(item.get("sparql", "")).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        unique.append(query)
    return unique


def _validate_gold_cache_offline(
    cache: SPARQLCache,
    records: List[Dict[str, Any]],
    logger: logging.Logger,
    max_report: int = 5,
) -> None:
    queries = _collect_unique_sparql_queries(records)
    logger.info("Offline cache validation for %d unique gold SPARQL queries.", len(queries))
    missing: List[str] = []
    for query in queries:
        try:
            cache.execute(query, offline_only=True)
        except FileNotFoundError:
            missing.append(query)
    if missing:
        preview = "\n".join(f"- {q[:200]}" for q in missing[:max_report])
        raise RuntimeError(
            "offline_only=True but gold SPARQL cache is incomplete.\n"
            f"Missing cache entries: {len(missing)} / {len(queries)}\n"
            f"Examples:\n{preview}\n"
            "Please run prepare_data.py once without --offline-only to pre-cache gold queries."
        )
    logger.info("Offline cache validation passed (all gold queries cached).")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LC-QuAD 2.0 + QALD-9-plus data for ALREM experiments.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sparql",
        help="Output directory (default: data/sparql, relative to code/).",
    )
    parser.add_argument(
        "--lcquad-source",
        type=str,
        default=None,
        help="Optional local LC-QuAD file or directory. If omitted, use output-dir for local-first + download fallback.",
    )
    parser.add_argument(
        "--qald-source",
        type=str,
        default=None,
        help="Optional local QALD-9-plus file or directory. If omitted, use output-dir for local-first + download fallback.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lcquad-max-samples", type=int, default=None, help="Optional max sample count for LC-QuAD.")
    parser.add_argument("--qald-dev-ratio", type=float, default=0.15, help="QALD train/dev group split ratio.")
    parser.add_argument(
        "--qald-train-languages",
        type=str,
        default=None,
        help="Comma-separated language list for QALD train (default: all available).",
    )
    parser.add_argument(
        "--qald-test-languages",
        type=str,
        default="en,de,es,ru",
        help="Comma-separated language list for QALD test (default: en,de,es,ru).",
    )
    parser.add_argument(
        "--allow-incomplete-test-languages",
        action="store_true",
        help=(
            "Allow QALD test qid groups with missing target languages. "
            "By default, preparation fails fast on incomplete groups."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for SPARQL query results (default: <output-dir>/cache).",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Do not call remote SPARQL endpoint on cache miss.",
    )
    parser.add_argument(
        "--build-high-stakes-subset",
        action="store_true",
        help="Build optional high-stakes subset artifact (disabled by default in core protocol).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(output_dir / "prepare_data.log")

    lcquad_source = _resolve_path(args.lcquad_source) if args.lcquad_source else output_dir
    qald_source = _resolve_path(args.qald_source) if args.qald_source else output_dir
    cache_dir = _resolve_path(args.cache_dir) if args.cache_dir else (output_dir / "cache")

    train_langs = _parse_languages(args.qald_train_languages)
    test_langs = _parse_languages(args.qald_test_languages)
    if not test_langs:
        raise ValueError("qald-test-languages cannot be empty.")

    logger.info("Output directory: %s", output_dir)
    logger.info("LC-QuAD source: %s", lcquad_source)
    logger.info("QALD source: %s", qald_source)
    logger.info("SPARQL cache directory: %s", cache_dir)
    logger.info("offline_only: %s", args.offline_only)
    logger.info("strict_test_languages: %s", not args.allow_incomplete_test_languages)
    logger.info("build_high_stakes_subset: %s", args.build_high_stakes_subset)

    # 1-2) LC-QuAD (local-first, optional download) -> train/dev JSONL.
    lcquad_train, lcquad_dev = load_lcquad2(
        data_dir=str(lcquad_source),
        max_samples=args.lcquad_max_samples,
        seed=args.seed,
    )
    lcquad_train_path = output_dir / "lcquad2_stage1_train.jsonl"
    lcquad_dev_path = output_dir / "lcquad2_stage1_dev.jsonl"
    _write_jsonl(lcquad_train, lcquad_train_path)
    _write_jsonl(lcquad_dev, lcquad_dev_path)

    # 3-4) QALD (local-first, optional download) -> grouped train/dev + test JSONL.
    qald_train_all = load_qald9plus_train(
        data_dir=str(qald_source),
        languages=train_langs,
    )
    qald_train, qald_dev = split_qald_train_dev(
        train_data=qald_train_all,
        dev_ratio=args.qald_dev_ratio,
        seed=args.seed,
    )
    qald_test = load_qald9plus_test(
        data_dir=str(qald_source),
        languages=test_langs,
        strict_languages=not args.allow_incomplete_test_languages,
    )

    # Enforce split hygiene by removing train/dev records that duplicate test
    # on normalized (question, sparql) signatures.
    test_signatures = _collect_signatures(qald_test)
    qald_train, removed_train_vs_test = _filter_by_signatures(
        qald_train,
        forbidden_signatures=test_signatures,
    )
    qald_dev, removed_dev_vs_test = _filter_by_signatures(
        qald_dev,
        forbidden_signatures=test_signatures,
    )
    if removed_train_vs_test or removed_dev_vs_test:
        logger.warning(
            "Removed split-overlap samples by (question,sparql) signature: "
            "train_vs_test=%d dev_vs_test=%d",
            removed_train_vs_test,
            removed_dev_vs_test,
        )

    # Keep train/dev disjoint on the same signature criterion.
    train_signatures = _collect_signatures(qald_train)
    qald_dev, removed_dev_vs_train = _filter_by_signatures(
        qald_dev,
        forbidden_signatures=train_signatures,
    )
    if removed_dev_vs_train:
        logger.warning(
            "Removed train/dev overlap samples by (question,sparql) signature: dev_vs_train=%d",
            removed_dev_vs_train,
        )

    # Hard gate: no split overlap should remain after cleaning.
    remain_train_test = _count_signature_overlap(qald_train, qald_test)
    remain_dev_test = _count_signature_overlap(qald_dev, qald_test)
    remain_train_dev = _count_signature_overlap(qald_train, qald_dev)
    if remain_train_test or remain_dev_test or remain_train_dev:
        raise RuntimeError(
            "Split overlap remains after cleaning. "
            f"train_vs_test={remain_train_test} "
            f"dev_vs_test={remain_dev_test} "
            f"train_vs_dev={remain_train_dev}"
        )

    qald_train_path = output_dir / "qald9plus_stage2_train.jsonl"
    qald_dev_path = output_dir / "qald9plus_stage2_dev.jsonl"
    qald_test_path = output_dir / "qald9plus_test.jsonl"
    qald_icl_pool_path = output_dir / "qald9plus_icl_few_shot_pool.jsonl"
    _write_jsonl(qald_train, qald_train_path)
    _write_jsonl(qald_dev, qald_dev_path)
    _write_jsonl(qald_test, qald_test_path)
    qald_icl_pool = _build_icl_few_shot_pool(
        qald_train,
        allowed_languages=test_langs,
    )
    _write_jsonl(qald_icl_pool, qald_icl_pool_path)

    # 5) Pre-cache gold SPARQL results.
    cache = SPARQLCache(cache_dir=str(cache_dir), force_offline=args.offline_only)
    gold_records = qald_train + qald_dev + qald_test
    if args.offline_only:
        _validate_gold_cache_offline(cache=cache, records=gold_records, logger=logger)
    else:
        cache.pre_cache_gold(gold_records)

    # 6) Optional stress-test subset (not required by the core paper protocol).
    # Keep this as a side artifact and do not mix it into the main result table.
    high_stakes_status = "skipped_disabled"
    high_stakes: List[Dict[str, Any]] = []
    high_stakes_path: Optional[Path] = None
    if args.build_high_stakes_subset:
        high_stakes_status = "ok"
        try:
            high_stakes = filter_high_stakes_subset(qald_test, cache=cache)
        except FileNotFoundError as exc:
            if args.offline_only:
                logger.warning(
                    "Skip high-stakes subset construction in offline mode due to missing cache: %s",
                    exc,
                )
                high_stakes = []
                high_stakes_status = "skipped_offline_cache_miss"
            else:
                raise
        high_stakes_path = output_dir / "archive" / "qald9plus_high_stakes_test.jsonl"
        _write_jsonl(high_stakes, high_stakes_path)

    # 7) Print statistics.
    stats: Dict[str, Any] = {
        "lcquad_train_samples": len(lcquad_train),
        "lcquad_dev_samples": len(lcquad_dev),
        "qald_train_samples": len(qald_train),
        "qald_dev_samples": len(qald_dev),
        "qald_test_samples": len(qald_test),
        "qald_icl_few_shot_pool_samples": len(qald_icl_pool),
        "qald_train_languages": _language_distribution(qald_train),
        "qald_dev_languages": _language_distribution(qald_dev),
        "qald_test_languages": _language_distribution(qald_test),
        "overlap_removed": {
            "train_vs_test_by_signature": removed_train_vs_test,
            "dev_vs_test_by_signature": removed_dev_vs_test,
            "dev_vs_train_by_signature": removed_dev_vs_train,
        },
        "overlap_remaining": {
            "train_vs_test_by_signature": remain_train_test,
            "dev_vs_test_by_signature": remain_dev_test,
            "train_vs_dev_by_signature": remain_train_dev,
        },
        "qald_high_stakes_samples": len(high_stakes),
        "qald_high_stakes_status": high_stakes_status,
        "qald_high_stakes_path": str(high_stakes_path) if high_stakes_path else "",
        "output_files": [
            str(lcquad_train_path),
            str(lcquad_dev_path),
            str(qald_train_path),
            str(qald_dev_path),
            str(qald_test_path),
            str(qald_icl_pool_path),
        ],
    }
    if high_stakes_path:
        stats["output_files"].append(str(high_stakes_path))
    stats_path = output_dir / "prepare_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    stats["stats_file"] = str(stats_path)
    _print_stats(logger, stats)


if __name__ == "__main__":
    main()
