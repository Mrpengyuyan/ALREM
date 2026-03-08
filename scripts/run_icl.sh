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
#   EVAL_PROTOCOL=configs/sparql_eval_shared.yaml
#   RUN_EVAL=true
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

CONFIG="${CONFIG:-configs/sparql_icl_zero_shot.yaml}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-configs/sparql_eval_shared.yaml}"

echo "=========================================="
echo " SPARQL ICL Baseline"
echo " Config: ${CONFIG}"
echo " Eval Protocol: ${EVAL_PROTOCOL}"
echo "=========================================="

CMD=("${PYTHON_BIN}" -m src.run_icl_baseline --config "${CONFIG}")
if [ -n "${EVAL_PROTOCOL}" ]; then
    if [ ! -f "${EVAL_PROTOCOL}" ]; then
        echo "[ERROR] Eval protocol file not found: ${EVAL_PROTOCOL}"
        exit 1
    fi
    CMD+=(--eval_protocol "${EVAL_PROTOCOL}")
fi

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

RUN_EVAL_FLAG="$(printf '%s' "${RUN_EVAL:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "${RUN_EVAL_FLAG}" = "true" ]; then
    EFFECTIVE_OUTPUT="${OUTPUT_DIR:-}"
    if [ -z "${EFFECTIVE_OUTPUT}" ]; then
        EFFECTIVE_OUTPUT="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
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
    if [ -z "${EFFECTIVE_OUTPUT}" ]; then
        EFFECTIVE_OUTPUT="outputs"
    fi

    EFFECTIVE_RUN_NAME="${RUN_NAME:-}"
    if [ -z "${EFFECTIVE_RUN_NAME}" ]; then
        EFFECTIVE_RUN_NAME="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    print("")
    raise SystemExit(0)
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
print(str(cfg.get("run_name", "")).strip())
PY
)"
    fi
    if [ -z "${EFFECTIVE_RUN_NAME}" ]; then
        MODE_NORM="$(printf '%s' "${MODE:-zero}" | tr '[:upper:]' '[:lower:]')"
        if [ "${MODE_NORM}" = "few" ] || [ "${MODE_NORM}" = "fewshot" ] || [ "${MODE_NORM}" = "few_shot" ]; then
            EFFECTIVE_RUN_NAME="sparql_icl_fewshot_k4"
        else
            EFFECTIVE_RUN_NAME="sparql_icl_zero"
        fi
    fi
    PRED_PATH="${EFFECTIVE_OUTPUT}/${EFFECTIVE_RUN_NAME}/predictions.jsonl"
    META_PATH="${EFFECTIVE_OUTPUT}/${EFFECTIVE_RUN_NAME}/run_metadata.json"
    if [ ! -f "${PRED_PATH}" ]; then
        echo "[ERROR] Predictions file not found for RUN_EVAL: ${PRED_PATH}"
        exit 1
    fi
    if [ ! -f "${META_PATH}" ]; then
        echo "[ERROR] run_metadata.json not found for RUN_EVAL: ${META_PATH}"
        exit 1
    fi
    EVAL_CMD=(
        "${PYTHON_BIN}" -m src.eval_sparql
        --predictions_file "${PRED_PATH}"
        --run_metadata_file "${META_PATH}"
        --eval_protocol "${EVAL_PROTOCOL}"
    )
    if [ -n "${TEST_LANGUAGES:-}" ]; then
        EVAL_CMD+=(--test_languages "${TEST_LANGUAGES}")
    fi
    if [ -n "${CACHE_DIR:-}" ]; then
        EVAL_CMD+=(--cache_dir "${CACHE_DIR}")
    fi
    OFFLINE_FLAG="$(printf '%s' "${OFFLINE_ONLY:-false}" | tr '[:upper:]' '[:lower:]')"
    if [ "${OFFLINE_FLAG}" = "true" ]; then
        EVAL_CMD+=(--offline_only)
    fi
    echo "Running eval: ${EVAL_CMD[*]}"
    "${EVAL_CMD[@]}"
fi

RUN_SUMMARY_FLAG="$(printf '%s' "${RUN_SUMMARY:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "${RUN_SUMMARY_FLAG}" = "true" ]; then
    SUMMARY_OUTPUTS="${SUMMARY_OUTPUTS_DIR:-}"
    if [ -z "${SUMMARY_OUTPUTS}" ]; then
        SUMMARY_OUTPUTS="${OUTPUT_DIR:-}"
    fi
    if [ -z "${SUMMARY_OUTPUTS}" ]; then
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
echo " ICL generation complete."
echo "=========================================="
