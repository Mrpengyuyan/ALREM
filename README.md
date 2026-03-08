# ALREM SPARQL Research Codebase

This repository is now focused on multilingual Text-to-SPARQL experiments:
- Stage 1: LC-QuAD 2.0 (English)
- Stage 2: QALD-9-plus (multilingual)
- Unified evaluation: EA / ExecRate / NormEM / F1 / CLC
- Task baselines: ICL zero-shot / few-shot

Implemented LoRA methods:
- `alrem` (main)
- `uniform`
- `matched` (parameter-matched uniform)
- `alrem_strong`
- `alrem_reverse_sandwich`

## Repository Layout

```text
code/
├─ configs/
├─ scripts/
├─ src/
├─ tests/
├─ docs/
│  ├─ SPARQL_EXPERIMENT_PLAN_AND_REPO_STEPS.md
│  └─ SPARQL_SERVER_RUNBOOK.md
├─ data/
├─ outputs/
└─ requirements.txt
```

## Environment Setup

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies:
- `torch`, `transformers`, `peft`, `datasets`, `accelerate`
- `SPARQLWrapper`, `langdetect`, `bitsandbytes`

## End-to-End Pipeline

### 1) Prepare data

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

Notes:
- Default is strict test-language completeness (missing target language causes fail-fast).
- To allow incomplete language groups: `--allow-incomplete-test-languages`.
- High-stakes subset is optional and disabled by default. Enable with `--build-high-stakes-subset`.

### 2) Preflight checks

```bash
bash scripts/preflight_sparql.sh configs/sparql_stage1_alrem.yaml
bash scripts/preflight_sparql.sh configs/sparql_stage2_alrem.yaml
```

### 3) Stage1 training

```bash
bash scripts/run_stage1.sh
```

### 4) Stage2 training

```bash
bash scripts/run_stage2.sh
```

Train -> eval -> summary in one chained run:

```bash
RUN_EVAL=true OFFLINE_ONLY=true RUN_SUMMARY=true bash scripts/run_stage2.sh
```

Override Stage1 checkpoint:

```bash
STAGE1_CKPT=outputs/sparql_s1_alrem bash scripts/run_stage2.sh
```

### 5) ICL baselines

Zero-shot:

```bash
bash scripts/run_icl.sh
```

Few-shot:

```bash
CONFIG=configs/sparql_icl_few_shot.yaml bash scripts/run_icl.sh
```

Run ICL then evaluate immediately under the shared protocol:

```bash
RUN_EVAL=true EVAL_PROTOCOL=configs/sparql_eval_shared.yaml bash scripts/run_icl.sh
```

Run ICL -> eval -> summary in one chained run:

```bash
RUN_EVAL=true RUN_SUMMARY=true bash scripts/run_icl.sh
```

### 6) Evaluation

Adapter mode:

```bash
bash scripts/run_eval.sh
```

Predictions-only mode (shared outlet for ICL/adapter):

```bash
PREDICTIONS_FILE=outputs/sparql_icl_zero/predictions.jsonl \
RUN_METADATA_FILE=outputs/sparql_icl_zero/run_metadata.json \
CACHE_DIR=data/sparql/cache \
OFFLINE_ONLY=true \
bash scripts/run_eval.sh
```

Evaluate then regenerate paper tables:

```bash
RUN_SUMMARY=true SUMMARY_OUTPUTS_DIR=outputs SUMMARY_OUT_DIR=paper_tables bash scripts/run_eval.sh
```

Evaluation always uses the shared protocol by default:
- `configs/sparql_eval_shared.yaml`
- see `docs/EVAL_PROTOCOL.md`
- main-table runs require `strict_schema=true` and `result_partition=unified_codechain`
- `run_id` is now canonicalized as `<run_name>__<mode>__<protocol_id_tag>__s<seed>`

## Config Matrix

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

ICL:
- `configs/sparql_icl_zero_shot.yaml`
- `configs/sparql_icl_few_shot.yaml`

## Run Outputs

Training run directory `outputs/<run_name>/` typically contains:
- `config.yaml`
- `train.log`
- `params.json`
- `run_report.json`
- adapter checkpoints

Evaluation output directory contains:
- `predictions.jsonl`
- `run_metadata.json`
- `metrics.json`
- `detailed_results.jsonl`

## Tests

Run all tests:

```bash
python -m pytest -q tests
```

Run core SPARQL tests:

```bash
python -m pytest -q tests/test_data_sparql.py tests/test_eval_sparql_logic.py tests/test_icl_baseline.py
```

## Result Tables

Generate SPARQL summary tables from `outputs/`:

```bash
python scripts/summarize_results.py --outputs_dir outputs --out_dir paper_tables
```

Tables generated:
- `paper_tables/main_results.md` (only `unified_codechain`)
- `paper_tables/external_results.md` (`external_*` partitions)
- `paper_tables/per_language.md`
- `paper_tables/clc_groups.md`
- `paper_tables/error_distribution.md`

Table policy:
- Keep `external_reported` baselines in a separate section from `unified_codechain`.
- Do not run same-class significance tests across these two sections.
