#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/benchmark_outputs"
LIMIT="${1:-20}"

mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${OUT_DIR}/techqa_benchmark_${LIMIT}_${TS}.log"
JSONL_PATH="${OUT_DIR}/techqa_eval_results_${LIMIT}_${TS}.jsonl"
SUMMARY_PATH="${OUT_DIR}/techqa_eval_summary_${LIMIT}_${TS}.json"

# User requested conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rag_env

PYTHON_BIN="python"

echo "[Benchmark] Running TechQA CRAG eval with limit=${LIMIT}"
echo "[Benchmark] Log: ${LOG_PATH}"

pushd "${ROOT_DIR}" >/dev/null
"${PYTHON_BIN}" "${ROOT_DIR}/techqa_evaluate.py" \
  --limit "${LIMIT}" \
  --output "${JSONL_PATH}" \
  --summary "${SUMMARY_PATH}" \
  --data_json "${ROOT_DIR}/techqa_train.json" 2>&1 | tee "${LOG_PATH}"
popd >/dev/null

echo "[Benchmark] Results JSONL: ${JSONL_PATH}"
echo "[Benchmark] Summary JSON: ${SUMMARY_PATH}"
echo "[Benchmark] Completed"
