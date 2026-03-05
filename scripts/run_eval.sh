#!/usr/bin/env bash
# ============================================================
# run_eval.sh — Evaluate a trained SPARQL adapter
# ============================================================
# Usage (from project root):
#   bash scripts/run_eval.sh
#   CONFIG=configs/sparql_stage2_uniform.yaml bash scripts/run_eval.sh
#   ADAPTER=outputs/sparql_s2_alrem TEST_DATA=data/sparql/qald9plus_test.jsonl bash scripts/run_eval.sh
#
# By default reads paths from CONFIG's Stage 2 yaml.
# Override any parameter via environment variables.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/sparql_stage2_alrem.yaml}"

echo "=========================================="
echo " SPARQL Evaluation"
echo " Config: ${CONFIG}"
echo "=========================================="

CMD=(python -m src.eval_sparql --config "${CONFIG}")

if [ -n "${ADAPTER:-}" ]; then
    CMD+=(--adapter_path "${ADAPTER}")
fi
if [ -n "${MODEL:-}" ]; then
    CMD+=(--model_name "${MODEL}")
fi
if [ -n "${TEST_DATA:-}" ]; then
    CMD+=(--test_data "${TEST_DATA}")
fi
if [ -n "${CACHE_DIR:-}" ]; then
    CMD+=(--cache_dir "${CACHE_DIR}")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${OFFLINE_ONLY:-false}" = "true" ]; then
    CMD+=(--offline_only)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "=========================================="
echo " Evaluation complete."
echo "=========================================="
