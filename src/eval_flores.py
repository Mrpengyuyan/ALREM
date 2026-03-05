import argparse
import os
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_flores import build_flores_prompt, load_flores_dataset
from .metrics import compute_bleu, compute_chrf
from .prompts import FLORES_PROMPT
from .utils import load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLORES translation.")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def _select_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _batch_generate(model, tokenizer, prompts: List[str], max_new_tokens: int, device):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    attn = inputs["attention_mask"]
    lengths = attn.sum(dim=1).tolist()
    texts = []
    for i, length in enumerate(lengths):
        gen_ids = outputs[i, int(length) :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
    return texts


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    cfg = load_yaml(os.path.join(run_dir, "config.yaml"))

    model_name = cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError("model_name_or_path must be set in config.")

    precision = cfg.get("precision", "bf16")
    torch_dtype = _select_dtype(precision)

    tokenizer = AutoTokenizer.from_pretrained(run_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(base_model, run_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt_flores = cfg.get("flores_prompt", FLORES_PROMPT)
    eval_examples = load_flores_dataset(cfg, cfg.get("eval_split", "dev"))
    max_eval_samples = args.max_eval_samples or cfg.get("max_eval_samples")
    if max_eval_samples:
        eval_examples = eval_examples[: int(max_eval_samples)]

    prompts = [build_flores_prompt(ex, prompt_template=prompt_flores) for ex in eval_examples]
    refs = [ex["target"] for ex in eval_examples]
    pairs = [f"{ex['src_lang']}-{ex['tgt_lang']}" for ex in eval_examples]

    batch_size = int(cfg.get("eval_batch_size", cfg.get("batch_size", 1)))
    max_new_tokens = int(cfg.get("gen_max_new_tokens", 128))

    preds: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        preds.extend(_batch_generate(model, tokenizer, batch_prompts, max_new_tokens, device))

    by_pair: Dict[str, Dict[str, float]] = {}
    for pair in sorted(set(pairs)):
        idxs = [i for i, p in enumerate(pairs) if p == pair]
        pair_preds = [preds[i] for i in idxs]
        pair_refs = [refs[i] for i in idxs]
        chrf = compute_chrf(pair_preds, pair_refs)
        bleu = compute_bleu(pair_preds, pair_refs)
        by_pair[pair] = {"chrf": chrf, "bleu": bleu, "num_samples": len(idxs)}

    overall_chrf = compute_chrf(preds, refs)
    overall_bleu = compute_bleu(preds, refs)
    metrics = {
        "task": "flores",
        "overall": {"chrf": overall_chrf, "bleu": overall_bleu, "num_samples": len(refs)},
        "by_pair": by_pair,
    }
    save_json(metrics, os.path.join(run_dir, "metrics.json"))


if __name__ == "__main__":
    main()
