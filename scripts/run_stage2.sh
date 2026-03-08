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
#   RUN_EVAL=true
#   EVAL_PROTOCOL=configs/sparql_eval_shared.yaml
#   OFFLINE_ONLY=true
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

CONFIG="${CONFIG:-configs/sparql_stage2_alrem.yaml}"

echo "=========================================="
echo " Stage 2 Training"
echo " Config: ${CONFIG}"
echo "=========================================="

CMD=("${PYTHON_BIN}" -m src.train_sft --config "${CONFIG}")

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
        EFFECTIVE_RUN_NAME="sparql_s2_alrem"
    fi

    EFFECTIVE_ADAPTER="${ADAPTER:-${EFFECTIVE_OUTPUT}/${EFFECTIVE_RUN_NAME}}"

    EVAL_PROTOCOL_PATH="${EVAL_PROTOCOL:-}"
    if [ -z "${EVAL_PROTOCOL_PATH}" ]; then
        EVAL_PROTOCOL_PATH="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    print("")
    raise SystemExit(0)
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
print(str(cfg.get("eval_protocol_path", "")).strip())
PY
)"
    fi
    if [ -z "${EVAL_PROTOCOL_PATH}" ]; then
        EVAL_PROTOCOL_PATH="configs/sparql_eval_shared.yaml"
    fi
    if [ ! -f "${EVAL_PROTOCOL_PATH}" ]; then
        echo "[ERROR] Eval protocol file not found: ${EVAL_PROTOCOL_PATH}"
        exit 1
    fi

    EVAL_CMD=(
        "${PYTHON_BIN}" -m src.eval_sparql
        --config "${CONFIG}"
        --adapter_path "${EFFECTIVE_ADAPTER}"
        --eval_protocol "${EVAL_PROTOCOL_PATH}"
    )
    if [ -n "${MODEL:-}" ]; then
        EVAL_CMD+=(--model_name "${MODEL}")
    fi
    if [ -n "${TEST_DATA:-}" ]; then
        EVAL_CMD+=(--test_data "${TEST_DATA}")
    fi
    if [ -n "${CACHE_DIR:-}" ]; then
        EVAL_CMD+=(--cache_dir "${CACHE_DIR}")
    fi
    if [ -n "${TEST_LANGUAGES:-}" ]; then
        EVAL_CMD+=(--test_languages "${TEST_LANGUAGES}")
    fi
    if [ -n "${EVAL_OUTPUT_DIR:-}" ]; then
        EVAL_CMD+=(--output_dir "${EVAL_OUTPUT_DIR}")
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
echo " Stage 2 complete."
echo "=========================================="
