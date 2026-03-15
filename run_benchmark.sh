#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Automated train/test benchmark for Levels 2 and 3
#
# Usage:
#   bash run_benchmark.sh
#
# This script runs:
#   1) train mode (fit=train, eval=val)
#   2) test mode  (fit=val, eval=test)
# and prints an MCC summary from both JSON outputs.
# ===========================================================================
set -euo pipefail

DATASET="${DATASET:-non_human}"
EMBEDDING="${EMBEDDING:-8M}"
LEVELS=("2" "3")
OUTPUT_ROOT="${OUTPUT_ROOT:-results/benchmark_${DATASET}_${EMBEDDING}_l2_l3}"
BENCHMARK_MLP_CAL_RESTARTS="${BENCHMARK_MLP_CAL_RESTARTS:-5}"
BENCHMARK_MLP_ENSEMBLE="${BENCHMARK_MLP_ENSEMBLE:-7}"

export BENCHMARK_MLP_CAL_RESTARTS
export BENCHMARK_MLP_ENSEMBLE

run_phase() {
    local mode="$1"

    python semantic_screening_models.py \
        --dataset "${DATASET}" \
        --embedding "${EMBEDDING}" \
        --levels "${LEVELS[@]}" \
        --output_dir "${OUTPUT_ROOT}" \
        "--${mode}"
}

echo "============================================================"
echo " Benchmark automation | dataset=${DATASET} embedding=${EMBEDDING}"
echo " Levels: ${LEVELS[*]} | patience=10% of epochs (auto)"
echo " MLP tuning: cal_restarts=${BENCHMARK_MLP_CAL_RESTARTS}, ensemble=${BENCHMARK_MLP_ENSEMBLE}"
echo " Output root: ${OUTPUT_ROOT}"
echo "============================================================"

echo ""
echo "============================================================"
echo " TRAIN phase (fit=train, eval=val)"
echo "============================================================"
run_phase "train"

echo ""
echo "============================================================"
echo " TEST phase (fit=val, eval=test)"
echo "============================================================"
run_phase "test"

echo ""
echo "============================================================"
echo " MCC summary from train/test benchmark_comparison.json"
echo "============================================================"
OUTPUT_ROOT_ENV="${OUTPUT_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

# The orchestrator appends /train and /test to --output_dir.
output_root = Path(os.environ["OUTPUT_ROOT_ENV"])
targets = [
    "level2_meanpool_knn",
    "level2_meanpool_mlp",
    "level3_attnpool_knn",
    "level3_attnpool_mlp",
]

paths = {
    "train": output_root / "train" / "benchmark_comparison.json",
    "test": output_root / "test" / "benchmark_comparison.json",
}

print(f"{'phase':>6}  {'model':<20} {'MCC':>8} {'MCC_std':>8}")
print("-" * 52)

for phase, path in paths.items():
    if not path.exists():
        print(f"{phase:>6}  {'(missing benchmark_comparison.json)':<20}")
        continue

    with path.open() as fh:
        payload = json.load(fh)

    results = payload.get("results", {})
    missing_in_phase = []
    for model in targets:
        row = results.get(model, {})
        if not row:
            missing_in_phase.append(model)
        mcc = row.get("mcc")
        mcc_std = row.get("mcc_std")
        mcc_txt = f"{mcc:.4f}" if isinstance(mcc, (float, int)) else "N/A"
        std_txt = f"{mcc_std:.4f}" if isinstance(mcc_std, (float, int)) else "N/A"
        print(f"{phase:>6}  {model:<20} {mcc_txt:>8} {std_txt:>8}")

    if missing_in_phase:
        print(f"{phase:>6}  missing_models: {missing_in_phase}")
        print(f"{phase:>6}  available_keys: {sorted(results.keys())[:12]}")

print("")
print("Focus model: level3_attnpool_mlp (primary MCC target).")
PY

echo ""
echo "============================================================"
echo " Done. Results saved under:"
echo "   ${OUTPUT_ROOT}/train"
echo "   ${OUTPUT_ROOT}/test"
echo "============================================================"
