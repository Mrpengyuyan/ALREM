#!/usr/bin/env bash
# ============================================================
# run_stage2.sh — Run Stage 2 training on QALD-9-plus
# ============================================================
# Usage (from project root):
#   bash scripts/run_stage2.sh                          # default: ALREM
#   CONFIG=configs/sparql_stage2_uniform.yaml bash scripts/run_stage2.sh
#
# Override Stage 1 checkpoint:
#   STAGE1_CKPT=outputs/sparql_s1_alrem bash scripts/run_stage2.sh
#
# Optional overrides:
#   OUTPUT_DIR=outputs  RUN_NAME=my_run  MAX_TRAIN=500
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/sparql_stage2_alrem.yaml}"

echo "=========================================="
echo " Stage 2 Training"
echo " Config: ${CONFIG}"
echo "=========================================="

CMD=(python -m src.train_sft --config "${CONFIG}")

# Override stage1 checkpoint if provided via env
if [ -n "${STAGE1_CKPT:-}" ]; then
    CMD+=(--stage1_checkpoint "${STAGE1_CKPT}")
fi
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
echo " Stage 2 complete."
echo "=========================================="
