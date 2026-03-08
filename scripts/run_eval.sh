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
# Optional:
#   RUN_SUMMARY=true
#   SUMMARY_OUTPUTS_DIR=outputs
#   SUMMARY_OUT_DIR=paper_tables
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "[ERROR] python/python3 not found in PATH."
        exit 1
    fi
fi

PREDICTIONS_FILE="${PREDICTIONS_FILE:-}"
if [ -z "${PREDICTIONS_FILE}" ]; then
    CONFIG="${CONFIG:-configs/sparql_stage2_alrem.yaml}"
else
    CONFIG="${CONFIG:-}"
fi
EVAL_PROTOCOL="${EVAL_PROTOCOL:-configs/sparql_eval_shared.yaml}"

echo "=========================================="
echo " SPARQL Evaluation"
if [ -n "${CONFIG}" ]; then
    echo " Config: ${CONFIG}"
else
    echo " Config: (none)"
fi
echo "=========================================="

CMD=("${PYTHON_BIN}" -m src.eval_sparql)
if [ -n "${CONFIG}" ]; then
    CMD+=(--config "${CONFIG}")
fi
if [ -n "${EVAL_PROTOCOL}" ]; then
    if [ ! -f "${EVAL_PROTOCOL}" ]; then
        echo "[ERROR] Eval protocol file not found: ${EVAL_PROTOCOL}"
        exit 1
    fi
    CMD+=(--eval_protocol "${EVAL_PROTOCOL}")
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
if [ -n "${RUN_METADATA_FILE:-}" ]; then
    CMD+=(--run_metadata_file "${RUN_METADATA_FILE}")
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
OFFLINE_FLAG="$(printf '%s' "${OFFLINE_ONLY:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "${OFFLINE_FLAG}" = "true" ]; then
    CMD+=(--offline_only)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

RUN_SUMMARY_FLAG="$(printf '%s' "${RUN_SUMMARY:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "${RUN_SUMMARY_FLAG}" = "true" ]; then
    SUMMARY_OUTPUTS="${SUMMARY_OUTPUTS_DIR:-}"
    if [ -z "${SUMMARY_OUTPUTS}" ] && [ -n "${PREDICTIONS_FILE}" ]; then
        PRED_PARENT="$("${PYTHON_BIN}" - "${PREDICTIONS_FILE}" <<'PY'
import sys
from pathlib import Path

pred_path = Path(sys.argv[1]).resolve()
print(str(pred_path.parent))
PY
)"
        SUMMARY_OUTPUTS="${PRED_PARENT}"
    fi
    if [ -z "${SUMMARY_OUTPUTS}" ] && [ -n "${CONFIG}" ]; then
        SUMMARY_OUTPUTS="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    print("")
    raise SystemExit(0)
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
print(str(cfg.get("output_dir", "")).strip())
PY
)"
    fi
    if [ -z "${SUMMARY_OUTPUTS}" ]; then
        SUMMARY_OUTPUTS="outputs"
    fi
    SUMMARY_OUT="${SUMMARY_OUT_DIR:-paper_tables}"
    SUMMARY_CMD=(
        "${PYTHON_BIN}" scripts/summarize_results.py
        --outputs_dir "${SUMMARY_OUTPUTS}"
        --out_dir "${SUMMARY_OUT}"
    )
    echo "Running summary: ${SUMMARY_CMD[*]}"
    "${SUMMARY_CMD[@]}"
fi

echo "=========================================="
echo " Evaluation complete."
echo "=========================================="
