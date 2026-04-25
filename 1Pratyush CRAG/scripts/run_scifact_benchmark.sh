#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/benchmark_outputs"
LIMIT="${1:-20}"

mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${OUT_DIR}/scifact_benchmark_${LIMIT}_${TS}.log"
JSONL_PATH="${OUT_DIR}/scifact_eval_results_${LIMIT}_${TS}.jsonl"
SUMMARY_PATH="${OUT_DIR}/scifact_eval_summary_${LIMIT}_${TS}.json"

PYTHON_BIN="/home/pratyush/Documents/CRAG/.venv/bin/python"

echo "[Benchmark] Running SciFact CRAG eval with limit=${LIMIT}"
echo "[Benchmark] Log: ${LOG_PATH}"

pushd "${ROOT_DIR}" >/dev/null
"${PYTHON_BIN}" "${ROOT_DIR}/main.py" --mode eval --eval_limit "${LIMIT}" 2>&1 | tee "${LOG_PATH}"
popd >/dev/null

cp "${ROOT_DIR}/crag_eval_results_${LIMIT}.jsonl" "${JSONL_PATH}"

"${PYTHON_BIN}" - "${JSONL_PATH}" "${SUMMARY_PATH}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

if not rows:
    raise SystemExit("No rows found in JSONL output")

initial_accuracy = statistics.mean(float(r["initial_correct"]) for r in rows)
final_accuracy = statistics.mean(float(r["final_correct"]) for r in rows)
accuracy_gain = final_accuracy - initial_accuracy
avg_faithfulness = statistics.mean(float(r["faithfulness_score"]) for r in rows)
avg_retrieval_improvement = statistics.mean(float(r["retrieval_improvement"]) for r in rows)
initial_wrong = sum(1 for r in rows if int(r["initial_correct"]) == 0)
corrected_successes = sum(1 for r in rows if int(r["initial_correct"]) == 0 and int(r["final_correct"]) == 1)
correction_success_rate = (corrected_successes / initial_wrong) if initial_wrong > 0 else 0.0

summary = {
    "dataset": "SciFact validation/dev (HF fallback)",
    "num_samples": len(rows),
    "initial_accuracy": initial_accuracy,
    "final_accuracy": final_accuracy,
    "accuracy_gain": accuracy_gain,
    "avg_faithfulness": avg_faithfulness,
    "avg_retrieval_improvement": avg_retrieval_improvement,
    "correction_success_rate": correction_success_rate,
    "source_jsonl": str(jsonl_path),
}

summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[Benchmark] Results JSONL: ${JSONL_PATH}"
echo "[Benchmark] Summary JSON: ${SUMMARY_PATH}"
echo "[Benchmark] Completed"
