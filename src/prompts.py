"""Prompt templates and formatting functions for all tasks."""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# MGSM / Flores (existing, unchanged)
# ---------------------------------------------------------------------------
MGSM_PROMPT = "Question: {q}\nAnswer:"
FLORES_PROMPT = "Translate from {src_lang} to {tgt_lang}:\n{source}\nTranslation:"


def format_mgsm_prompt(question: str, template: str = MGSM_PROMPT) -> str:
    return template.format(q=question)


def format_flores_prompt(
    source: str,
    src_lang: str,
    tgt_lang: str,
    template: str = FLORES_PROMPT,
) -> str:
    return template.format(src_lang=src_lang, tgt_lang=tgt_lang, source=source)


# ---------------------------------------------------------------------------
# SPARQL (new)
# ---------------------------------------------------------------------------
SPARQL_SYSTEM_PROMPT = (
    "You are a SPARQL query generator for Wikidata. Given a natural language question, "
    "generate the corresponding SPARQL query. Output ONLY the SPARQL query, nothing else."
)


def build_sparql_train_text(
    question: str,
    sparql: str,
    tokenizer: Any,
) -> str:
    """Build a complete training string using the tokenizer's chat template.

    Returns a single string ready for tokenization, including all special tokens
    that the model was pretrained with (e.g. <|im_start|> for Qwen).
    """
    messages = [
        {"role": "system", "content": SPARQL_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": sparql.strip()},
    ]
    # add_generation_prompt=False because the assistant turn is complete.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


def build_sparql_infer_text(
    question: str,
    tokenizer: Any,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build an inference prompt string using the tokenizer's chat template.

    Returns a string ending right before the assistant's first token,
    so that the model generates the SPARQL query.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SPARQL_SYSTEM_PROMPT},
    ]
    for ex in (few_shot_examples or []):
        q = ex.get("question", "").strip()
        s = ex.get("sparql", "").strip()
        if q and s:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": s})

    messages.append({"role": "user", "content": question.strip()})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text
