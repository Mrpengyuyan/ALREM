# ALREM: Sandwich Rank Allocation for LoRA

This repo provides a complete, runnable prototype of **ALREM** (Sandwich Rank Allocation for LoRA) for multilingual SFT and evaluation on MGSM (math reasoning) and FLORES (translation). It supports:
- **ALREM** per-layer rank patterns via PEFT `rank_pattern` and `alpha_pattern`.
- **Uniform LoRA** baseline.
- **Parameter-matched uniform** baseline that matches ALREM LoRA parameter count within ~1%.

Artifacts per run:
- `outputs/<run_name>/config.yaml`
- `outputs/<run_name>/train.log`
- `outputs/<run_name>/metrics.json`
- `outputs/<run_name>/params.json`

## Setup

```bash
pip install -r requirements.txt
```

## Model setup

Edit the config files and replace the placeholder:

```
model_name_or_path: "YOUR_LLAMA_3B_REPO_ID"
```

Use a HuggingFace repo ID you have access to (e.g., a 3B LLaMA model).

## Data

### MGSM
Online (default):
- `dataset_name: "mgsm"`
- If this fails, try other variants (examples: `"openai/MGSM"`, `"taesiri/mgsm"`), or use offline mode.

Offline:
- Provide a JSONL file with fields:
  - `language`, `question`, `answer`
- Set `data_path` in the config.

### FLORES
Online (default):
- `dataset_name: "facebook/flores"`
- `language_pairs` should use dataset config names, e.g. `"eng_Latn-fra_Latn"`.

Offline:
- Provide a JSONL file with fields:
  - `src_lang`, `tgt_lang`, `source`, `target`
- Set `data_path` (or `data_path_train` / `data_path_eval`) in the config.

## Training

ALREM MGSM:

```bash
python -m src.train_sft --config configs/alrem_mgsm.yaml
```

Uniform MGSM:

```bash
python -m src.train_sft --config configs/uniform_mgsm.yaml
```

ALREM FLORES:

```bash
python -m src.train_sft --config configs/alrem_flores.yaml
```

Uniform FLORES:

```bash
python -m src.train_sft --config configs/uniform_flores.yaml
```

Parameter-matched uniform:
- Set `method: matched` in the config (and keep the same `r_high/r_low` and cut ratios).

Fast sanity check (requirement):

```bash
python -m src.train_sft --config configs/uniform_mgsm.yaml --max_train_samples 200 --max_eval_samples 200
```

## Evaluation

MGSM:

```bash
python -m src.eval_mgsm --run_dir outputs/<run_name>
```

FLORES:

```bash
python -m src.eval_flores --run_dir outputs/<run_name>
```

## Summarize results

```bash
python scripts/summarize_results.py --outputs_dir outputs --out_dir paper_tables
```

This produces:
- `paper_tables/main_results.md`
- `paper_tables/efficiency.md`
- `paper_tables/ablations.md`

## Config knobs

All configs support:
- `task`: `mgsm` | `flores`
- `method`: `alrem` | `uniform` | `matched`
- `r_high`, `r_low`, `r_uniform`
- `cut_ratio_early`, `cut_ratio_mid` (or `early_end`, `mid_end`)
- `target_modules` (e.g., `["q_proj","v_proj"]` or `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`)
- `lora_dropout`, `lora_alpha_mode` (`"2r"` or `"fixed"`), `lora_alpha_fixed`
- `learning_rate`, `batch_size`, `grad_accum`, `num_train_epochs`, `max_steps`, `warmup_steps`, `weight_decay`
- `max_seq_len`, `max_train_samples`, `max_eval_samples`
- `precision` (`bf16`/`fp16`), `grad_ckpt`, `seed`
- `output_dir`, `run_name`

## Notes
- Per-layer ranks are implemented via PEFT `LoraConfig(rank_pattern=..., alpha_pattern=...)`.
- Parameter-matched uniform baseline automatically solves `r_match` to match ALREM LoRA params.
- `params.json` includes the matched relative error and LoRA parameter stats.

## FAQ

1) Dataset load fails:
- Use offline JSONL, or update `dataset_name` / `dataset_config` in your config.

2) OOM on a single GPU:
- Reduce `batch_size`, increase `grad_accum`, and/or reduce `max_seq_len`.

3) Want more modules:
- Update `target_modules` to include attention and FFN projections.
