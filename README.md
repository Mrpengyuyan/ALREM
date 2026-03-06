# ALREM Research Codebase

This repository contains runnable research code for ALREM-based LoRA experiments across:
- MGSM (math reasoning)
- FLORES (translation)
- Multilingual Text-to-SPARQL (LC-QuAD 2.0 -> QALD-9-plus, two-stage training)

Implemented LoRA methods:
- `alrem` (sandwich rank pattern via `rank_pattern` / `alpha_pattern`)
- `uniform`
- `matched` (parameter-matched uniform baseline)

## What Is Stable In This Version
- Two-stage SPARQL training is integrated in `src/train_sft.py`.
- Stage2 adapter compatibility checks are enforced (base model, target modules, rank settings).
- SPARQL execution is cache-first and supports strict offline evaluation.
- Preflight checks and Linux run scripts are included.
- Unit tests are available under `tests/`.

## Repository Layout

```text
code/
├── configs/
├── scripts/
├── src/
├── tests/
├── docs/
│   └── SPARQL_SERVER_RUNBOOK.md
├── data/
├── outputs/
└── requirements.txt
```

## Environment Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies include:
- `torch`, `transformers`, `peft`, `datasets`, `accelerate`
- `SPARQLWrapper`, `langdetect`, `bitsandbytes`

## SPARQL Pipeline (Recommended Main Path)

### 1) Prepare data (local-first, with optional download)

From `code/`:

```bash
python scripts/prepare_data.py
```

Common options:

```bash
python scripts/prepare_data.py \
  --output-dir data/sparql \
  --lcquad-source data/downloads/lcquad2 \
  --qald-source data/downloads/qald9plus \
  --qald-test-languages en,de,es,ru
```

Offline cache-only mode:

```bash
python scripts/prepare_data.py --offline-only
```

### 2) Preflight check

```bash
bash scripts/preflight_sparql.sh configs/sparql_stage1_alrem.yaml
bash scripts/preflight_sparql.sh configs/sparql_stage2_alrem.yaml
```

### 3) Stage1 training (LC-QuAD 2.0)

```bash
bash scripts/run_stage1.sh
```

Or choose another config:

```bash
CONFIG=configs/sparql_stage1_uniform.yaml bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage1_param_match.yaml bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage1_alrem_strong.yaml bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage1_reverse_sandwich.yaml bash scripts/run_stage1.sh
```

### 4) Stage2 training (QALD-9-plus)

```bash
bash scripts/run_stage2.sh
```

Or choose another config:

```bash
CONFIG=configs/sparql_stage2_uniform.yaml bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_param_match.yaml bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_alrem_strong.yaml bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_reverse_sandwich.yaml bash scripts/run_stage2.sh
```

Override Stage1 checkpoint when needed:

```bash
STAGE1_CKPT=outputs/sparql_s1_alrem bash scripts/run_stage2.sh
```

### 5) Evaluation

```bash
bash scripts/run_eval.sh
```

Optional overrides:

```bash
CONFIG=configs/sparql_stage2_alrem.yaml \
ADAPTER=outputs/sparql_s2_alrem \
TEST_LANGUAGES=en,de,es,ru \
OFFLINE_ONLY=true \
bash scripts/run_eval.sh
```

### 6) Minimal smoke test

```bash
CONFIG=configs/sparql_stage1_alrem.yaml RUN_NAME=smoke_s1 MAX_TRAIN=32 MAX_EVAL=16 bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage2_alrem.yaml STAGE1_CKPT=outputs/smoke_s1 RUN_NAME=smoke_s2 MAX_TRAIN=32 MAX_EVAL=16 bash scripts/run_stage2.sh
```

## SPARQL Config Matrix

Stage1:
- `configs/sparql_stage1_uniform.yaml`
- `configs/sparql_stage1_param_match.yaml`
- `configs/sparql_stage1_alrem.yaml`
- `configs/sparql_stage1_alrem_strong.yaml`
- `configs/sparql_stage1_reverse_sandwich.yaml`

Stage2:
- `configs/sparql_stage2_uniform.yaml`
- `configs/sparql_stage2_param_match.yaml`
- `configs/sparql_stage2_alrem.yaml`
- `configs/sparql_stage2_alrem_strong.yaml`
- `configs/sparql_stage2_reverse_sandwich.yaml`

Important:
- Do not mix Stage1 and Stage2 checkpoints from different base models.
- Do not use strong/reverse Stage2 configs with normal Stage1 checkpoints.
- Stage2 must use architecture-compatible Stage1 adapters.

## Training Outputs

Each run writes into `outputs/<run_name>/`:
- `config.yaml`
- `train.log`
- `params.json`
- `run_report.json`
- adapter/model files

`train.log` now includes elapsed-time progress lines like:
- `[0:11:58] Rank:0; Step: 10/8000;`

Default interval is `eval_steps`; override with:
- `step_time_log_interval` in YAML

## Testing

Run all tests:

```bash
python -m pytest -q tests
```

Run only SPARQL prep smoke test:

```bash
python -m pytest -q tests/test_prepare_data_smoke.py
```

## Legacy MGSM / FLORES Usage

Training:

```bash
python -m src.train_sft --config configs/alrem_mgsm.yaml
python -m src.train_sft --config configs/uniform_mgsm.yaml
python -m src.train_sft --config configs/alrem_flores.yaml
python -m src.train_sft --config configs/uniform_flores.yaml
```

Evaluation:

```bash
python -m src.eval_mgsm --run_dir outputs/<run_name>
python -m src.eval_flores --run_dir outputs/<run_name>
```

## Summarize Tables

```bash
python scripts/summarize_results.py --outputs_dir outputs --out_dir paper_tables
```

## Additional Documentation

For server deployment and end-to-end order of operations:
- `docs/SPARQL_SERVER_RUNBOOK.md`
