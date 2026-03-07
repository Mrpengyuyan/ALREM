#!/usr/bin/env bash
# ============================================================
# run_icl.sh — Run SPARQL ICL baseline generation
# ============================================================
# Usage (from project root):
#   bash scripts/run_icl.sh
#   CONFIG=configs/sparql_icl_few_shot.yaml bash scripts/run_icl.sh
#
# Optional overrides:
#   MODE=zero|few
#   MODEL=Qwen/Qwen2.5-7B-Instruct
#   TEST_DATA=data/sparql/qald9plus_test.jsonl
#   FEW_SHOT_POOL=data/sparql/qald9plus_stage2_train.jsonl
#   FEW_SHOT_K=4
#   OUTPUT_DIR=outputs
#   RUN_NAME=sparql_icl_zero_custom
#   TEST_LANGUAGES=en,de,es,ru
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/sparql_icl_zero_shot.yaml}"

echo "=========================================="
echo " SPARQL ICL Baseline"
echo " Config: ${CONFIG}"
echo "=========================================="

CMD=(python -m src.run_icl_baseline --config "${CONFIG}")

if [ -n "${MODE:-}" ]; then
    CMD+=(--mode "${MODE}")
fi
if [ -n "${MODEL:-}" ]; then
    CMD+=(--model_name "${MODEL}")
fi
if [ -n "${TEST_DATA:-}" ]; then
    CMD+=(--test_data "${TEST_DATA}")
fi
if [ -n "${FEW_SHOT_POOL:-}" ]; then
    CMD+=(--few_shot_pool "${FEW_SHOT_POOL}")
fi
if [ -n "${FEW_SHOT_K:-}" ]; then
    CMD+=(--few_shot_k "${FEW_SHOT_K}")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${RUN_NAME:-}" ]; then
    CMD+=(--run_name "${RUN_NAME}")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${TEST_LANGUAGES:-}" ]; then
    CMD+=(--test_languages "${TEST_LANGUAGES}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "=========================================="
echo " ICL generation complete."
echo "=========================================="
