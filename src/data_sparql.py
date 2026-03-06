import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

LOGGER = logging.getLogger("alrem.data_sparql")

SYSTEM_PROMPT = (
    "You are a SPARQL query generator for Wikidata. Given a natural language question, "
    "generate the corresponding SPARQL query. Output ONLY the SPARQL query, nothing else."
)

DEFAULT_QALD_TEST_LANGUAGES = ["en", "de", "es", "ru"]

_PREPARED_OUTPUT_FILES = {
    "lcquad2_stage1_train.jsonl",
    "lcquad2_stage1_dev.jsonl",
    "qald9plus_stage2_train.jsonl",
    "qald9plus_stage2_dev.jsonl",
    "qald9plus_test.jsonl",
    "qald9plus_high_stakes_test.jsonl",
}

_SPARQL_PATTERN = re.compile(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", flags=re.IGNORECASE)

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except Exception:
    detect = None


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        # Typical multilingual text fields: {"language": "en", "string": "..."}.
        for key in ("string", "text", "value", "question", "query", "sparql", "en"):
            if key in value:
                candidate = _to_text(value[key])
                if candidate:
                    return candidate
        for nested in value.values():
            candidate = _to_text(nested)
            if candidate:
                return candidate
        return ""
    if isinstance(value, list):
        for item in value:
            candidate = _to_text(item)
            if candidate:
                return candidate
        return ""
    return str(value).strip()


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def _looks_like_sparql(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    if _SPARQL_PATTERN.search(candidate):
        return True
    if "wd:" in candidate and "wdt:" in candidate:
        return True
    if "WHERE" in candidate.upper():
        return True
    return False


def _case_insensitive_get(rec: Dict[str, Any], key: str) -> Any:
    if key in rec:
        return rec[key]
    lower_map = {str(k).lower(): k for k in rec.keys()}
    real_key = lower_map.get(key.lower())
    if real_key is None:
        return None
    return rec[real_key]


def _extract_json_like(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skip invalid JSONL at %s:%d (%s)", path, line_no, exc)
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
                else:
                    LOGGER.warning("Skip non-dict JSONL record at %s:%d", path, line_no)
        return out

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]

    if isinstance(obj, dict):
        for key in ("questions", "data", "items", "examples", "records"):
            value = obj.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [obj]

    return []


def _find_candidate_files(path: Path, file_patterns: Sequence[str]) -> List[Path]:
    if path.is_file():
        return [path]

    if not path.exists():
        return []

    found: List[Path] = []
    seen = set()
    for pattern in file_patterns:
        for item in path.rglob(pattern):
            if not item.is_file():
                continue
            if item in seen:
                continue
            seen.add(item)
            found.append(item)
    return found


def _filter_prepared_artifacts(files: Sequence[Path]) -> List[Path]:
    filtered: List[Path] = []
    skipped = 0
    for file_path in files:
        if file_path.name.lower() in _PREPARED_OUTPUT_FILES:
            skipped += 1
            continue
        filtered.append(file_path)
    if skipped:
        LOGGER.info("Ignored %d prepared output file(s) when scanning local raw data.", skipped)
    return filtered


def _extract_lcquad_fields(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    question_keys = [
        "corrected_question",
        "question",
        "paraphrased_question",
        "NNQT_question",
        "nl_question",
        "text",
    ]
    sparql_keys = [
        "sparql_wikidata",
        "sparql",
        "query",
        "sparql_query",
        "target_sparql",
        "gold_sparql",
    ]

    question = ""
    sparql = ""

    for key in question_keys:
        value = _case_insensitive_get(record, key)
        candidate = _to_text(value)
        if candidate and not _looks_like_sparql(candidate):
            question = candidate
            break

    for key in sparql_keys:
        value = _case_insensitive_get(record, key)
        if isinstance(value, dict):
            for nested_key in ("sparql", "query", "value", "text"):
                nested_candidate = _to_text(value.get(nested_key))
                if nested_candidate and _looks_like_sparql(nested_candidate):
                    sparql = nested_candidate
                    break
            if sparql:
                break
        candidate = _to_text(value)
        if candidate and _looks_like_sparql(candidate):
            sparql = candidate
            break

    if not sparql:
        for value in record.values():
            candidate = _to_text(value)
            if candidate and _looks_like_sparql(candidate):
                sparql = candidate
                break

    if not question:
        for key, value in record.items():
            if "question" not in str(key).lower():
                continue
            candidate = _to_text(value)
            if candidate and not _looks_like_sparql(candidate):
                question = candidate
                break

    if not question or not sparql:
        return None

    return {"question": question.strip(), "sparql": _normalize_whitespace(sparql)}


def _dedupe_records(records: Iterable[Dict[str, str]], key_fields: Sequence[str]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for rec in records:
        key = tuple(rec.get(field, "").strip() for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _split_records(records: List[Dict[str, str]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not records:
        return [], []
    if len(records) == 1:
        return records, []

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    return shuffled[:split_idx], shuffled[split_idx:]


def load_lcquad2_from_local(path: str) -> List[Dict[str, str]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"LC-QuAD 2.0 local path does not exist: {path_obj}")

    primary_patterns = [
        "*lcquad*2*.jsonl",
        "*lcquad*2*.json",
        "*lc-quad*2*.jsonl",
        "*lc-quad*2*.json",
        "*train-data-answer*.json",
        "*test-data-answer*.json",
    ]
    files = _find_candidate_files(path_obj, primary_patterns)
    files = _filter_prepared_artifacts(files)
    if not files:
        raise FileNotFoundError(
            "No LC-QuAD-like JSON/JSONL files found. "
            f"Expected file names containing lcquad/train-data-answer/test-data-answer under: {path_obj}"
        )

    parsed: List[Dict[str, str]] = []
    skipped_count = 0
    for file_path in files:
        try:
            raw_records = _extract_json_like(file_path)
        except Exception as exc:
            LOGGER.warning("Failed to parse LC-QuAD file %s (%s)", file_path, exc)
            continue
        for raw in raw_records:
            normalized = _extract_lcquad_fields(raw)
            if normalized is None:
                skipped_count += 1
                continue
            parsed.append(normalized)

    parsed = _dedupe_records(parsed, key_fields=("question", "sparql"))
    if not parsed:
        raise ValueError(
            "Found local LC-QuAD files but could not parse valid samples. "
            "Please ensure records contain both question and SPARQL fields."
        )

    if skipped_count:
        LOGGER.warning("Skipped %d LC-QuAD records due to missing/invalid fields.", skipped_count)
    LOGGER.info("Loaded %d LC-QuAD samples from local path: %s", len(parsed), path_obj)
    return parsed


def _write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _download_json_from_url(url: str, timeout_sec: int = 60) -> Any:
    with urlopen(url, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def download_lcquad2(data_dir: str) -> str:
    base_dir = Path(data_dir)
    if base_dir.suffix:
        base_dir = base_dir.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    out_file = base_dir / "lcquad2_downloaded.jsonl"
    errors: List[str] = []

    # Attempt 1: Hugging Face datasets.
    try:
        from datasets import load_dataset

        hf_candidates = ["lc_quad2", "lc_quAD2", "kgqa/lcquad2", "ALREM/lcquad2"]
        for dataset_name in hf_candidates:
            try:
                ds = load_dataset(dataset_name)
                all_rows: List[Dict[str, Any]] = []
                for split_name in ds.keys():
                    all_rows.extend([dict(row) for row in ds[split_name]])

                parsed: List[Dict[str, str]] = []
                for row in all_rows:
                    normalized = _extract_lcquad_fields(row)
                    if normalized is not None:
                        parsed.append(normalized)
                parsed = _dedupe_records(parsed, key_fields=("question", "sparql"))
                if parsed:
                    _write_jsonl(parsed, out_file)
                    LOGGER.info(
                        "Downloaded LC-QuAD from Hugging Face dataset '%s' to %s",
                        dataset_name,
                        out_file,
                    )
                    return str(out_file)
            except Exception as exc:
                errors.append(f"[HF:{dataset_name}] {exc}")
    except Exception as exc:
        errors.append(f"[HF:import] {exc}")

    # Attempt 2: Explicit URLs.
    url_candidates = [
        "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/train-data-answer.json",
        "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/test-data-answer.json",
    ]
    downloaded: List[Dict[str, str]] = []
    for url in url_candidates:
        try:
            obj = _download_json_from_url(url)
            raw_records: List[Dict[str, Any]]
            if isinstance(obj, list):
                raw_records = [item for item in obj if isinstance(item, dict)]
            elif isinstance(obj, dict):
                raw_records = []
                for key in ("questions", "data", "items", "records"):
                    value = obj.get(key)
                    if isinstance(value, list):
                        raw_records = [item for item in value if isinstance(item, dict)]
                        break
                if not raw_records:
                    raw_records = [obj]
            else:
                raw_records = []
            for record in raw_records:
                normalized = _extract_lcquad_fields(record)
                if normalized is not None:
                    downloaded.append(normalized)
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"[URL:{url}] {exc}")
        except Exception as exc:
            errors.append(f"[URL:{url}] {exc}")

    downloaded = _dedupe_records(downloaded, key_fields=("question", "sparql"))
    if downloaded:
        _write_jsonl(downloaded, out_file)
        LOGGER.info("Downloaded LC-QuAD from explicit URLs to %s", out_file)
        return str(out_file)

    joined_errors = "\n".join(errors[-10:])
    raise RuntimeError(
        "Failed to download LC-QuAD 2.0 from remote sources.\n"
        f"Tried output dir: {base_dir}\n"
        f"Errors:\n{joined_errors}\n\n"
        "Please place LC-QuAD original JSON/JSONL files manually under this directory and retry."
    )


def load_lcquad2(data_dir: str, max_samples: Optional[int] = None, seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    source_path = Path(data_dir)
    parsed: List[Dict[str, str]]

    try:
        parsed = load_lcquad2_from_local(str(source_path))
    except FileNotFoundError:
        downloaded_path = download_lcquad2(str(source_path))
        parsed = load_lcquad2_from_local(downloaded_path)
    except ValueError:
        # Local files exist but parsing failed: do not silently override with remote data.
        raise

    if max_samples is not None and max_samples > 0 and len(parsed) > max_samples:
        rng = random.Random(seed)
        indices = list(range(len(parsed)))
        rng.shuffle(indices)
        parsed = [parsed[idx] for idx in indices[:max_samples]]

    train_data, dev_data = _split_records(parsed, train_ratio=0.9, seed=seed)
    LOGGER.info("LC-QuAD split: train=%d dev=%d", len(train_data), len(dev_data))
    return train_data, dev_data


def _safe_lang(code: str) -> str:
    return code.strip().lower()


def _detect_lang(question: str) -> Optional[str]:
    if detect is None:
        return None
    cleaned = question.strip()
    if not cleaned:
        return None
    try:
        return _safe_lang(detect(cleaned))
    except Exception:
        return None


def _extract_sparql_from_record(record: Dict[str, Any]) -> str:
    query_candidates = ["sparql", "query", "sparql_query", "target_sparql", "gold_sparql"]
    for key in query_candidates:
        value = _case_insensitive_get(record, key)
        if isinstance(value, dict):
            for nested_key in ("sparql", "query", "value", "text"):
                nested_val = _to_text(value.get(nested_key))
                if nested_val and _looks_like_sparql(nested_val):
                    return _normalize_whitespace(nested_val)
        candidate = _to_text(value)
        if candidate and _looks_like_sparql(candidate):
            return _normalize_whitespace(candidate)
    for val in record.values():
        candidate = _to_text(val)
        if candidate and _looks_like_sparql(candidate):
            return _normalize_whitespace(candidate)
    return ""


def _extract_language_questions(record: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    question_fields = ["question", "questions", "translations", "utterances"]
    lang_fields = ["language", "lang", "locale"]

    # Standard QALD format: question=[{language:string}, ...]
    for field in question_fields:
        value = _case_insensitive_get(record, field)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                lang = _to_text(item.get("language") or item.get("lang") or item.get("locale"))
                text = _to_text(item.get("string") or item.get("text") or item.get("question"))
                if lang and text:
                    out[_safe_lang(lang)] = text
        elif isinstance(value, dict):
            # Either {"language":"en","string":"..."} or {"en":"...", "de":"..."}
            if any(k in value for k in ("language", "lang", "locale")):
                lang = _to_text(value.get("language") or value.get("lang") or value.get("locale"))
                text = _to_text(value.get("string") or value.get("text") or value.get("question"))
                if lang and text:
                    out[_safe_lang(lang)] = text
            else:
                for k, v in value.items():
                    text = _to_text(v)
                    lang_code = _safe_lang(str(k))
                    if text and len(lang_code) <= 5:
                        out[lang_code] = text
        elif isinstance(value, str):
            text = value.strip()
            if text:
                lang = ""
                for lf in lang_fields:
                    lang = _to_text(_case_insensitive_get(record, lf))
                    if lang:
                        break
                if not lang:
                    detected = _detect_lang(text)
                    if detected:
                        lang = detected
                if not lang:
                    lang = "unk"
                out[_safe_lang(lang)] = text

    # Fallback for flattened schema: {"question": "...", "language": "..."}.
    if not out:
        text = _to_text(_case_insensitive_get(record, "question"))
        if text:
            lang = _to_text(_case_insensitive_get(record, "language")) or _detect_lang(text) or "unk"
            out[_safe_lang(lang)] = text

    return out


def _resolve_qid(record: Dict[str, Any], sparql: str, lang_questions: Dict[str, str]) -> str:
    qid_candidates = ["qid", "id", "uid", "question_id", "_id", "questionId"]
    for key in qid_candidates:
        value = _case_insensitive_get(record, key)
        text = _to_text(value)
        if text:
            return text

    canonical_key = sparql
    if not canonical_key and lang_questions:
        canonical_key = "||".join(f"{lang}:{q}" for lang, q in sorted(lang_questions.items()))
    hashed = hashlib.md5(canonical_key.encode("utf-8")).hexdigest()[:16]
    return f"auto_{hashed}"


def _iter_qald_records(raw_obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw_obj, list):
        for item in raw_obj:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(raw_obj, dict):
        if "questions" in raw_obj and isinstance(raw_obj["questions"], list):
            for item in raw_obj["questions"]:
                if isinstance(item, dict):
                    yield item
            return

        for key in ("data", "items", "records", "examples", "train", "test"):
            value = raw_obj.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return

        # Single-record fallback.
        if any(k in raw_obj for k in ("question", "query", "sparql")):
            yield raw_obj


def _guess_split(file_path: Path, obj: Any) -> str:
    lower_name = file_path.name.lower()
    if "train" in lower_name:
        return "train"
    if "test" in lower_name:
        return "test"
    if "dev" in lower_name or "valid" in lower_name:
        return "dev"

    if isinstance(obj, dict):
        dataset_meta = obj.get("dataset")
        if isinstance(dataset_meta, dict):
            dataset_id = _to_text(dataset_meta.get("id")).lower()
            if "train" in dataset_id:
                return "train"
            if "test" in dataset_id:
                return "test"
    return "unknown"


def load_qald9plus_from_local(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"QALD-9-plus local path does not exist: {path_obj}")

    primary_patterns = [
        "*qald*9*plus*train*.json",
        "*qald*9*plus*test*.json",
        "*qald*train*.json",
        "*qald*test*.json",
        "*qald*.jsonl",
        "*qald*.json",
    ]
    files = _find_candidate_files(path_obj, primary_patterns)
    files = _filter_prepared_artifacts(files)
    if not files:
        raise FileNotFoundError(
            "No QALD-like JSON/JSONL files found. "
            f"Expected file names containing qald/train/test under: {path_obj}"
        )

    bundle: Dict[str, Any] = {"train": [], "test": [], "dev": [], "unknown": [], "source_files": []}
    for file_path in files:
        try:
            # _extract_json_like returns list; for QALD we need raw object too.
            if file_path.suffix.lower() == ".jsonl":
                data_obj = _extract_json_like(file_path)
            else:
                with file_path.open("r", encoding="utf-8") as handle:
                    data_obj = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Failed to parse QALD file %s (%s)", file_path, exc)
            continue

        split = _guess_split(file_path, data_obj)
        bundle[split].append({"file": str(file_path), "data": data_obj})
        bundle["source_files"].append(str(file_path))

    total_objects = sum(len(bundle[key]) for key in ("train", "test", "dev", "unknown"))
    if total_objects == 0:
        raise ValueError(
            "Found QALD files locally, but none could be parsed. "
            "Please check file encoding/content."
        )
    LOGGER.info("Loaded QALD local bundle from %s with %d source object(s).", path_obj, total_objects)
    return bundle


def download_qald9plus(data_dir: str) -> str:
    base_dir = Path(data_dir)
    if base_dir.suffix:
        base_dir = base_dir.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = base_dir / "qald9plus_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    # Attempt 1: Hugging Face datasets.
    try:
        from datasets import load_dataset

        hf_candidates = ["qald", "qald_9_plus", "qald/qald-9-plus", "ALREM/qald9plus"]
        for dataset_name in hf_candidates:
            try:
                ds = load_dataset(dataset_name)
                wrote_any = False
                for split_name in ds.keys():
                    rows = [dict(row) for row in ds[split_name]]
                    if not rows:
                        continue
                    out_path = raw_dir / f"qald9plus_{split_name}_downloaded.jsonl"
                    _write_jsonl(rows, out_path)
                    wrote_any = True
                if wrote_any:
                    LOGGER.info("Downloaded QALD-9-plus from Hugging Face dataset '%s'.", dataset_name)
                    return str(raw_dir)
            except Exception as exc:
                errors.append(f"[HF:{dataset_name}] {exc}")
    except Exception as exc:
        errors.append(f"[HF:import] {exc}")

    # Attempt 2: Explicit URLs.
    url_to_filename = {
        "https://raw.githubusercontent.com/KGQA/QALD_9_plus/master/data/qald_9_plus_train_wikidata.json": "qald_9_plus_train_wikidata.json",
        "https://raw.githubusercontent.com/KGQA/QALD_9_plus/master/data/qald_9_plus_test_wikidata.json": "qald_9_plus_test_wikidata.json",
    }
    wrote_urls = False
    for url, filename in url_to_filename.items():
        try:
            obj = _download_json_from_url(url)
            out_path = raw_dir / filename
            _ensure_parent_dir(out_path)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(obj, handle, ensure_ascii=False, indent=2)
            wrote_urls = True
        except Exception as exc:
            errors.append(f"[URL:{url}] {exc}")

    if wrote_urls:
        LOGGER.info("Downloaded QALD-9-plus raw files to %s", raw_dir)
        return str(raw_dir)

    joined_errors = "\n".join(errors[-10:])
    raise RuntimeError(
        "Failed to download QALD-9-plus from remote sources.\n"
        f"Tried output dir: {raw_dir}\n"
        f"Errors:\n{joined_errors}\n\n"
        "Please place QALD-9-plus raw JSON/JSONL files manually under this directory and retry."
    )


def _normalize_qald_samples(bundle_entries: List[Dict[str, Any]], languages: Optional[List[str]]) -> List[Dict[str, str]]:
    lang_set = set(_safe_lang(lang) for lang in languages) if languages else None
    normalized: List[Dict[str, str]] = []

    for entry in bundle_entries:
        file_name = entry.get("file", "<unknown>")
        raw_obj = entry.get("data")
        for raw_record in _iter_qald_records(raw_obj):
            sparql = _extract_sparql_from_record(raw_record)
            if not sparql:
                LOGGER.warning("Skip QALD record without SPARQL in %s", file_name)
                continue

            lang_map = _extract_language_questions(raw_record)
            if not lang_map:
                LOGGER.warning("Skip QALD record without language question variants in %s", file_name)
                continue

            qid = _resolve_qid(raw_record, sparql=sparql, lang_questions=lang_map)

            if lang_set is not None:
                for lang in sorted(lang_set):
                    question = lang_map.get(lang)
                    if not question:
                        LOGGER.warning(
                            "QALD qid=%s missing language='%s' in %s, skipped.",
                            qid,
                            lang,
                            file_name,
                        )
                        continue
                    normalized.append(
                        {
                            "question": question.strip(),
                            "sparql": sparql,
                            "language": lang,
                            "qid": qid,
                        }
                    )
            else:
                for lang, question in lang_map.items():
                    if not question.strip():
                        continue
                    normalized.append(
                        {
                            "question": question.strip(),
                            "sparql": sparql,
                            "language": _safe_lang(lang),
                            "qid": qid,
                        }
                    )

    normalized = _dedupe_records(normalized, key_fields=("qid", "language", "question", "sparql"))
    return normalized


def _load_qald_bundle_with_fallback(data_dir: str) -> Dict[str, Any]:
    source_path = Path(data_dir)
    try:
        return load_qald9plus_from_local(str(source_path))
    except FileNotFoundError:
        downloaded_dir = download_qald9plus(str(source_path))
        return load_qald9plus_from_local(downloaded_dir)


def _select_qald_split_entries(bundle: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    entries = bundle.get(split_name, [])
    if entries:
        return entries

    unknown_entries = bundle.get("unknown", [])
    if unknown_entries:
        source_files = bundle.get("source_files", [])
        raise ValueError(
            f"Ambiguous QALD split: could not detect explicit '{split_name}' files.\n"
            "Refusing to fallback to 'unknown' split to avoid train/test leakage.\n"
            "Please rename files to include 'train'/'test' in filenames or place train/test in separate directories.\n"
            f"Detected source files: {source_files}"
        )

    raise ValueError(f"Could not find QALD {split_name} split from local/downloaded sources.")


def load_qald9plus_train(data_dir: str, languages: Optional[List[str]] = None) -> List[Dict[str, str]]:
    bundle = _load_qald_bundle_with_fallback(data_dir)
    entries = _select_qald_split_entries(bundle, "train")

    data = _normalize_qald_samples(entries, languages=languages)
    if not data:
        raise ValueError("QALD train split loaded but no valid samples after normalization.")
    LOGGER.info("Loaded QALD train samples: %d", len(data))
    return data


def load_qald9plus_test(data_dir: str, languages: Optional[List[str]] = None) -> List[Dict[str, str]]:
    requested_languages = languages or DEFAULT_QALD_TEST_LANGUAGES
    bundle = _load_qald_bundle_with_fallback(data_dir)
    entries = _select_qald_split_entries(bundle, "test")

    data = _normalize_qald_samples(entries, languages=requested_languages)
    if not data:
        raise ValueError("QALD test split loaded but no valid samples after normalization.")
    LOGGER.info("Loaded QALD test samples: %d", len(data))
    return data


def split_qald_train_dev(
    train_data: List[Dict[str, str]],
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in (0, 1), got {dev_ratio}")
    if not train_data:
        return [], []

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for item in train_data:
        qid = item.get("qid", "").strip()
        if not qid:
            fallback = item.get("sparql", "") or item.get("question", "")
            qid = f"auto_{hashlib.md5(fallback.encode('utf-8')).hexdigest()[:16]}"
            item["qid"] = qid
        grouped.setdefault(qid, []).append(item)

    qids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(qids)

    if len(qids) == 1:
        return grouped[qids[0]], []

    dev_count = int(round(len(qids) * dev_ratio))
    dev_count = max(1, min(dev_count, len(qids) - 1))
    dev_ids = set(qids[:dev_count])

    train_split: List[Dict[str, str]] = []
    dev_split: List[Dict[str, str]] = []
    for qid, items in grouped.items():
        if qid in dev_ids:
            dev_split.extend(items)
        else:
            train_split.extend(items)

    LOGGER.info(
        "QALD grouped split by qid: total_groups=%d train=%d dev=%d",
        len(qids),
        len(train_split),
        len(dev_split),
    )
    return train_split, dev_split


def format_text2sparql_train(question: str, sparql: str) -> Dict[str, Any]:
    question_text = question.strip()
    sparql_text = sparql.strip()
    if not question_text:
        raise ValueError("question must be non-empty for training prompt.")
    if not sparql_text:
        raise ValueError("sparql must be non-empty for training prompt.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
        {"role": "assistant", "content": sparql_text},
    ]
    prompt_text = (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"Question: {question_text}\n"
        f"SPARQL: {sparql_text}"
    )
    return {"prompt_text": prompt_text, "messages": messages}


def format_text2sparql_infer(question: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    question_text = question.strip()
    if not question_text:
        raise ValueError("question must be non-empty for inference prompt.")

    examples = few_shot_examples or []
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    prompt_lines = [f"System: {SYSTEM_PROMPT}", ""]
    for example in examples:
        ex_question = _to_text(example.get("question"))
        ex_sparql = _to_text(example.get("sparql"))
        if not ex_question or not ex_sparql:
            LOGGER.warning("Skip invalid few-shot example with missing question/sparql fields.")
            continue
        messages.append({"role": "user", "content": ex_question})
        messages.append({"role": "assistant", "content": ex_sparql})
        prompt_lines.append(f"Question: {ex_question}")
        prompt_lines.append(f"SPARQL: {ex_sparql}")
        prompt_lines.append("")

    messages.append({"role": "user", "content": question_text})
    prompt_lines.append(f"Question: {question_text}")
    prompt_lines.append("SPARQL:")

    prompt_text = "\n".join(prompt_lines).strip()
    return {"prompt_text": prompt_text, "messages": messages}
