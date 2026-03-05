import json
from typing import Any, Dict, Iterable, List, Tuple

from datasets import load_dataset

from .prompts import FLORES_PROMPT, format_flores_prompt


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def parse_language_pairs(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    raw_pairs = cfg.get("language_pairs") or []
    for p in raw_pairs:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            pairs.append((str(p[0]), str(p[1])))
        elif isinstance(p, str) and "-" in p:
            src, tgt = p.split("-", 1)
            pairs.append((src, tgt))
    return pairs


def _filter_by_pairs(
    examples: Iterable[Dict[str, Any]], pairs: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    if not pairs:
        return list(examples)
    pair_set = set(pairs)
    return [ex for ex in examples if (ex["src_lang"], ex["tgt_lang"]) in pair_set]


def _normalize_record(rec: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    src_lang = rec.get(cfg.get("src_lang_field", "src_lang"), rec.get("src_lang"))
    tgt_lang = rec.get(cfg.get("tgt_lang_field", "tgt_lang"), rec.get("tgt_lang"))
    source = rec.get(cfg.get("source_field", "source"), rec.get("source"))
    target = rec.get(cfg.get("target_field", "target"), rec.get("target"))
    return {
        "src_lang": src_lang if src_lang is not None else "",
        "tgt_lang": tgt_lang if tgt_lang is not None else "",
        "source": source if source is not None else "",
        "target": target if target is not None else "",
    }


def _get_data_path(cfg: Dict[str, Any], split: str) -> str:
    if split == "train":
        return cfg.get("data_path_train") or cfg.get("data_path") or ""
    if split in ("validation", "valid", "eval", "test", "dev"):
        return cfg.get("data_path_eval") or cfg.get("data_path") or ""
    return cfg.get("data_path") or ""


def load_flores_online(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    name = cfg.get("dataset_name", "facebook/flores")
    pairs = parse_language_pairs(cfg)
    examples: List[Dict[str, Any]] = []
    for src, tgt in pairs:
        src_ds = load_dataset(name, src, split=split)
        tgt_ds = load_dataset(name, tgt, split=split)
        if "id" in src_ds.column_names and "id" in tgt_ds.column_names:
            src_map = {rec["id"]: rec for rec in src_ds}
            tgt_map = {rec["id"]: rec for rec in tgt_ds}
            shared_ids = sorted(set(src_map) & set(tgt_map))
            for idx in shared_ids:
                srec = src_map[idx]
                trec = tgt_map[idx]
                source = srec.get("sentence") or srec.get("text") or ""
                target = trec.get("sentence") or trec.get("text") or ""
                examples.append(
                    {"src_lang": src, "tgt_lang": tgt, "source": source, "target": target}
                )
        else:
            n = min(len(src_ds), len(tgt_ds))
            for i in range(n):
                srec = src_ds[i]
                trec = tgt_ds[i]
                source = srec.get("sentence") or srec.get("text") or ""
                target = trec.get("sentence") or trec.get("text") or ""
                examples.append(
                    {"src_lang": src, "tgt_lang": tgt, "source": source, "target": target}
                )
    return examples


def load_flores_dataset(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    data_path = _get_data_path(cfg, split)
    if data_path:
        raw = load_jsonl(data_path)
        examples = [_normalize_record(rec, cfg) for rec in raw]
    else:
        examples = load_flores_online(cfg, split)
    return _filter_by_pairs(examples, parse_language_pairs(cfg))


def build_flores_prompt(example: Dict[str, Any], prompt_template: str = FLORES_PROMPT) -> str:
    return format_flores_prompt(
        source=example["source"],
        src_lang=example["src_lang"],
        tgt_lang=example["tgt_lang"],
        template=prompt_template,
    )


def build_flores_train_text(
    example: Dict[str, Any], prompt_template: str = FLORES_PROMPT
) -> str:
    prompt = build_flores_prompt(example, prompt_template=prompt_template)
    return prompt + " " + example["target"]
