import json
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from .prompts import MGSM_PROMPT, format_mgsm_prompt


def _get_field(rec: Dict[str, Any], primary: Optional[str], fallbacks: List[str]) -> Any:
    if primary and primary in rec:
        return rec[primary]
    for f in fallbacks:
        if f in rec:
            return rec[f]
    return None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def normalize_mgsm_record(rec: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    q_field = cfg.get("question_field", "question")
    a_field = cfg.get("answer_field", "answer")
    l_field = cfg.get("language_field", "language")
    question = _get_field(rec, q_field, ["question", "problem", "query"])
    answer = _get_field(rec, a_field, ["answer", "solution", "target"])
    language = _get_field(rec, l_field, ["language", "lang"])
    return {
        "question": question if question is not None else "",
        "answer": answer if answer is not None else "",
        "language": language if language is not None else "unk",
    }


def load_mgsm_dataset(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    data_path = cfg.get("data_path")
    if data_path:
        raw = load_jsonl(data_path)
    else:
        name = cfg.get("dataset_name", "mgsm")
        config = cfg.get("dataset_config")
        raw = load_dataset(name, config, split=split)
    examples = [normalize_mgsm_record(rec, cfg) for rec in raw]
    languages = cfg.get("languages")
    if languages:
        lang_set = set(languages)
        examples = [ex for ex in examples if ex["language"] in lang_set]
    return examples


def build_mgsm_prompt(example: Dict[str, Any], prompt_template: str = MGSM_PROMPT) -> str:
    return format_mgsm_prompt(example["question"], template=prompt_template)


def build_mgsm_train_text(example: Dict[str, Any], prompt_template: str = MGSM_PROMPT) -> str:
    prompt = build_mgsm_prompt(example, prompt_template=prompt_template)
    return prompt + " " + example["answer"]
