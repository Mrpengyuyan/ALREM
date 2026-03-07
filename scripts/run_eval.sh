#!/usr/bin/env bash
# ============================================================
# run_eval.sh — Evaluate a trained SPARQL adapter
# ============================================================
# Usage (from project root):
#   bash scripts/run_eval.sh
#   CONFIG=configs/sparql_stage2_uniform.yaml bash scripts/run_eval.sh
#   ADAPTER=outputs/sparql_s2_alrem TEST_DATA=data/sparql/qald9plus_test.jsonl bash scripts/run_eval.sh
#   PREDICTIONS_FILE=outputs/sparql_icl_zero/predictions.jsonl bash scripts/run_eval.sh
#
# By default reads paths from CONFIG's Stage 2 yaml (adapter mode).
# In predictions-only mode, CONFIG is optional.
# Override any parameter via environment variables.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PREDICTIONS_FILE="${PREDICTIONS_FILE:-}"
if [ -z "${PREDICTIONS_FILE}" ]; then
    CONFIG="${CONFIG:-configs/sparql_stage2_alrem.yaml}"
else
    CONFIG="${CONFIG:-}"
fi

echo "=========================================="
echo " SPARQL Evaluation"
if [ -n "${CONFIG}" ]; then
    echo " Config: ${CONFIG}"
else
    echo " Config: (none)"
fi
echo "=========================================="

CMD=(python -m src.eval_sparql)
if [ -n "${CONFIG}" ]; then
    CMD+=(--config "${CONFIG}")
fi

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
if [ -n "${PREDICTIONS_FILE}" ]; then
    CMD+=(--predictions_file "${PREDICTIONS_FILE}")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if [ -n "${TEST_LANGUAGES:-}" ]; then
    CMD+=(--test_languages "${TEST_LANGUAGES}")
fi
if [ "${OFFLINE_ONLY:-false}" = "true" ]; then
    CMD+=(--offline_only)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "=========================================="
echo " Evaluation complete."
echo "=========================================="
