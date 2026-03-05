#!/usr/bin/env bash
# ============================================================
# preflight_sparql.sh — Validate server readiness for SPARQL runs
# ============================================================
# Usage (from project root):
#   bash scripts/preflight_sparql.sh
#   bash scripts/preflight_sparql.sh configs/sparql_stage1_alrem.yaml
#   bash scripts/preflight_sparql.sh configs/sparql_stage2_alrem.yaml
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${1:-configs/sparql_stage1_alrem.yaml}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}"
  exit 1
fi

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Config: ${CONFIG_PATH}"

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"[ERROR] Python>=3.10 required, got {sys.version}")
print(f"[OK] Python version: {sys.version.split()[0]}")
PY

python - <<'PY'
modules = ["yaml", "torch", "transformers", "datasets", "peft", "accelerate"]
missing = []
for name in modules:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("[ERROR] Missing required Python packages: " + ", ".join(missing))
print("[OK] Core Python packages are available.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] nvidia-smi found."
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "[WARN] nvidia-smi not found. GPU status cannot be validated."
fi

python - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
root = Path.cwd()
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")

def info(msg: str) -> None:
    print(f"[OK] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

common_keys = [
    "model_name_or_path",
    "task",
    "method",
    "quantization",
    "target_modules",
    "learning_rate",
    "num_train_epochs",
    "max_seq_len",
    "grad_ckpt",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "logging_steps",
    "evaluation_strategy",
    "eval_steps",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "seed",
    "output_dir",
]

missing_common = [k for k in common_keys if k not in cfg]
if missing_common:
    fail(f"Config missing common keys: {missing_common}")
info("Common config keys present.")

task = str(cfg.get("task", "")).strip().lower()
if task != "sparql":
    fail(f"Expected task=sparql, got task={task!r}")
info("task=sparql")

is_stage2 = bool(cfg.get("stage1_checkpoint"))
if is_stage2:
    stage_keys = [
        "stage1_checkpoint",
        "stage2_data_path",
        "stage2_dev_path",
        "test_data_path",
        "test_languages",
        "sparql_cache_dir",
        "early_stopping_patience",
    ]
    missing = [k for k in stage_keys if k not in cfg]
    if missing:
        fail(f"Stage2 config missing keys: {missing}")

    ckpt_dir = root / str(cfg["stage1_checkpoint"])
    if not ckpt_dir.exists():
        fail(f"Stage2 checkpoint path does not exist: {ckpt_dir}")
    adapter_cfg = ckpt_dir / "adapter_config.json"
    if not adapter_cfg.exists():
        fail(f"Missing adapter_config.json under checkpoint: {adapter_cfg}")

    for key in ("stage2_data_path", "stage2_dev_path", "test_data_path"):
        p = root / str(cfg[key])
        if not p.exists():
            fail(f"Missing data file for {key}: {p}")

    cache_dir = root / str(cfg["sparql_cache_dir"])
    if not cache_dir.exists():
        warn(f"Cache dir does not exist yet: {cache_dir} (allowed, but slower for first run)")
    info("Stage2 paths and checkpoint validated.")
else:
    stage_keys = ["stage1_data_path", "stage1_dev_path"]
    missing = [k for k in stage_keys if k not in cfg]
    if missing:
        fail(f"Stage1 config missing keys: {missing}")
    for key in stage_keys:
        p = root / str(cfg[key])
        if not p.exists():
            fail(f"Missing data file for {key}: {p}")
    info("Stage1 paths validated.")

print("[OK] Preflight validation passed.")
PY

echo "[DONE] preflight_sparql.sh passed."
