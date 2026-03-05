#!/usr/bin/env bash
# ============================================================
# run_stage1.sh — Run Stage 1 training on LC-QuAD 2.0
# ============================================================
# Usage (from project root):
#   bash scripts/run_stage1.sh                          # default: ALREM
#   CONFIG=configs/sparql_stage1_uniform.yaml bash scripts/run_stage1.sh
#   CONFIG=configs/sparql_stage1_param_match.yaml bash scripts/run_stage1.sh
#
# Optional overrides via environment variables:
#   OUTPUT_DIR=outputs  RUN_NAME=my_run  MAX_TRAIN=1000
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/sparql_stage1_alrem.yaml}"

echo "=========================================="
echo " Stage 1 Training"
echo " Config: ${CONFIG}"
echo "=========================================="

CMD=(python -m src.train_sft --config "${CONFIG}")

# Optional CLI overrides
if [ -n "${OUTPUT_DIR:-}" ]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${RUN_NAME:-}" ]; then
    CMD+=(--run_name "${RUN_NAME}")
fi
if [ -n "${MAX_TRAIN:-}" ]; then
    CMD+=(--max_train_samples "${MAX_TRAIN}")
fi
if [ -n "${MAX_EVAL:-}" ]; then
    CMD+=(--max_eval_samples "${MAX_EVAL}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "=========================================="
echo " Stage 1 complete."
echo "=========================================="
