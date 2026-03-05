import re
from typing import List


_FRACTION_RE = re.compile(r"-?\d+/\d+")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_final_answer(text: str) -> str:
    if text is None:
        return ""
    if "####" in text:
        text = text.split("####")[-1]
    text = text.strip()
    frac_matches = _FRACTION_RE.findall(text)
    if frac_matches:
        return frac_matches[-1].replace(",", "")
    num_matches = _NUMBER_RE.findall(text)
    if num_matches:
        return num_matches[-1].replace(",", "")
    return ""


def compute_accuracy(preds: List[str], refs: List[str]) -> float:
    if not preds:
        return 0.0
    correct = 0
    for p, r in zip(preds, refs):
        p_ans = extract_final_answer(p)
        r_ans = extract_final_answer(r)
        if p_ans == r_ans and p_ans != "":
            correct += 1
    return correct / max(len(preds), 1)


def compute_chrf(preds: List[str], refs: List[str]) -> float:
    try:
        import evaluate

        metric = evaluate.load("chrf")
        score = metric.compute(predictions=preds, references=refs)["score"]
        return float(score)
    except Exception:
        from sacrebleu.metrics import CHRF

        metric = CHRF()
        return float(metric.corpus_score(preds, [refs]).score)


def compute_bleu(preds: List[str], refs: List[str]) -> float:
    try:
        import evaluate

        metric = evaluate.load("bleu")
        score = metric.compute(predictions=preds, references=refs)["bleu"]
        return float(score * 100.0)
    except Exception:
        from sacrebleu.metrics import BLEU

        metric = BLEU()
        return float(metric.corpus_score(preds, [refs]).score)
