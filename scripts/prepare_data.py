import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        default="en,de,fr,ru",
        help="Comma-separated language list for QALD test (default: en,de,fr,ru).",
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
    )

    qald_train_path = output_dir / "qald9plus_stage2_train.jsonl"
    qald_dev_path = output_dir / "qald9plus_stage2_dev.jsonl"
    qald_test_path = output_dir / "qald9plus_test.jsonl"
    _write_jsonl(qald_train, qald_train_path)
    _write_jsonl(qald_dev, qald_dev_path)
    _write_jsonl(qald_test, qald_test_path)

    # 5) Pre-cache gold SPARQL results.
    cache = SPARQLCache(cache_dir=str(cache_dir), force_offline=args.offline_only)
    gold_queries = qald_train + qald_dev + qald_test
    cache.pre_cache_gold(gold_queries)

    # 6) Build legal/medical high-stakes subset for stress-test case study.
    high_stakes = filter_high_stakes_subset(qald_test, cache=cache)
    high_stakes_path = output_dir / "qald9plus_high_stakes_test.jsonl"
    _write_jsonl(high_stakes, high_stakes_path)

    # 7) Print statistics.
    stats: Dict[str, Any] = {
        "lcquad_train_samples": len(lcquad_train),
        "lcquad_dev_samples": len(lcquad_dev),
        "qald_train_samples": len(qald_train),
        "qald_dev_samples": len(qald_dev),
        "qald_test_samples": len(qald_test),
        "qald_train_languages": _language_distribution(qald_train),
        "qald_dev_languages": _language_distribution(qald_dev),
        "qald_test_languages": _language_distribution(qald_test),
        "qald_high_stakes_samples": len(high_stakes),
        "output_files": [
            str(lcquad_train_path),
            str(lcquad_dev_path),
            str(qald_train_path),
            str(qald_dev_path),
            str(qald_test_path),
            str(high_stakes_path),
        ],
    }
    stats_path = output_dir / "prepare_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    stats["stats_file"] = str(stats_path)
    _print_stats(logger, stats)


if __name__ == "__main__":
    main()
