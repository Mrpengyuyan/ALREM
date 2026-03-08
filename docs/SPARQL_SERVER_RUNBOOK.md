# SPARQL Server Runbook (CIKM 2026)

## 1) Server Target Layout
Keep the same relative structure on server:

```text
<SERVER_ROOT>/
└── code/
    ├── configs/
    ├── scripts/
    ├── src/
    ├── data/
    │   └── sparql/
    ├── outputs/
    └── requirements.txt
```

Recommended: avoid changing YAML paths by keeping `code/` as the execution root.

## 2) Files You Must Upload

1. Source/config/scripts:
- `code/src/`
- `code/configs/`
- `code/scripts/`
- `code/requirements.txt`
- `code/docs/SPARQL_SERVER_RUNBOOK.md`

2. Data:
- `code/data/sparql/lcquad2_stage1_train.jsonl`
- `code/data/sparql/lcquad2_stage1_dev.jsonl`
- `code/data/sparql/qald9plus_stage2_train.jsonl`
- `code/data/sparql/qald9plus_stage2_dev.jsonl`
- `code/data/sparql/qald9plus_test.jsonl`
- `code/data/sparql/cache/` (optional but recommended)

3. Optional pretrained Stage1 adapters (if you skip Stage1 training on server):
- `code/outputs/sparql_s1_alrem/`
- `code/outputs/sparql_s1_uniform/`
- `code/outputs/sparql_s1_param_match/`
- `code/outputs/sparql_s1_alrem_strong/`
- `code/outputs/sparql_s1_reverse_sandwich/`

Each adapter directory must contain `adapter_config.json`.

## 3) Environment Setup (on server)

From `<SERVER_ROOT>/code`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your cluster requires a specific CUDA/PyTorch wheel, install the required `torch` first, then run `pip install -r requirements.txt`.

## 4) Preflight Validation

Run before any training:

```bash
bash scripts/preflight_sparql.sh configs/sparql_stage1_alrem.yaml
```

For Stage2:

```bash
bash scripts/preflight_sparql.sh configs/sparql_stage2_alrem.yaml
```

This checks:
- Python version
- Core package imports
- Required config keys
- Data path existence
- Stage2 checkpoint and `adapter_config.json`

## 5) Run Commands

### Stage1

```bash
bash scripts/run_stage1.sh
# or
CONFIG=configs/sparql_stage1_uniform.yaml bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage1_param_match.yaml bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage1_alrem_strong.yaml bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage1_reverse_sandwich.yaml bash scripts/run_stage1.sh
```

### Stage2

```bash
bash scripts/run_stage2.sh
# or
CONFIG=configs/sparql_stage2_uniform.yaml bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_param_match.yaml bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_alrem_strong.yaml bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_reverse_sandwich.yaml bash scripts/run_stage2.sh
```

Override checkpoint if needed:

```bash
STAGE1_CKPT=outputs/sparql_s1_alrem bash scripts/run_stage2.sh
```

## 6) Minimal Smoke Test (first run)

```bash
CONFIG=configs/sparql_stage1_alrem.yaml RUN_NAME=smoke_s1 MAX_TRAIN=32 MAX_EVAL=16 bash scripts/run_stage1.sh
CONFIG=configs/sparql_stage2_alrem.yaml STAGE1_CKPT=outputs/smoke_s1 RUN_NAME=smoke_s2 MAX_TRAIN=32 MAX_EVAL=16 bash scripts/run_stage2.sh
CONFIG=configs/sparql_stage2_alrem.yaml STAGE1_CKPT=outputs/smoke_s1 RUN_NAME=smoke_s2 RUN_EVAL=true OFFLINE_ONLY=true RUN_SUMMARY=true bash scripts/run_stage2.sh
```

## 7) Frequent Failure Modes

1. `Stage 2 checkpoint path does not exist`:
- Ensure `stage1_checkpoint` points to an adapter output dir with `adapter_config.json`.

2. `rank pattern mismatch` in Stage2:
- Stage2 config and Stage1 checkpoint architecture do not match.
- Use a matching Stage1 checkpoint for that Stage2 ablation.

3. `No training samples available`:
- JSONL fields missing `question`/`sparql`, or wrong data path.

4. OOM:
- Keep `per_device_train_batch_size=1`, increase `gradient_accumulation_steps`, or reduce `max_seq_len`.

## 8) What Not To Do

- Do not mix different base models between Stage1 and Stage2.
- Do not use strong/reverse Stage2 configs with normal Stage1 checkpoints.
- Do not run Stage2 before confirming Stage1 adapter compatibility.
