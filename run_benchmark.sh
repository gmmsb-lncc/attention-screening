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
LEVELS_CSV="${LEVELS_CSV:-3,3a}"
IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/benchmark_${DATASET}_${EMBEDDING}_15_03_2026}"
EPOCHS="${EPOCHS:-500}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-mcc}"

# Strong MCC profile (fast to adopt, no code changes required).
BENCHMARK_MLP_USE_CV="${BENCHMARK_MLP_USE_CV:-1}"
BENCHMARK_MLP_FOLDS="${BENCHMARK_MLP_FOLDS:-5}"
BENCHMARK_MLP_CAL_RESTARTS="${BENCHMARK_MLP_CAL_RESTARTS:-7}"
BENCHMARK_MLP_ENSEMBLE="${BENCHMARK_MLP_ENSEMBLE:-11}"
BENCHMARK_MLP_OVERSAMPLE="${BENCHMARK_MLP_OVERSAMPLE:-1}"
BENCHMARK_LEVEL3_USE_AUX_CHANNEL="${BENCHMARK_LEVEL3_USE_AUX_CHANNEL:-1}"

# Scientific-rigor safeguards (test cannot guide tuning).
BENCHMARK_REQUIRE_TRAIN_SELECTION="${BENCHMARK_REQUIRE_TRAIN_SELECTION:-1}"
BENCHMARK_STRICT_LEVEL_COMPLETENESS="${BENCHMARK_STRICT_LEVEL_COMPLETENESS:-1}"

export BENCHMARK_MLP_USE_CV
export BENCHMARK_MLP_FOLDS
export BENCHMARK_MLP_CAL_RESTARTS
export BENCHMARK_MLP_ENSEMBLE
export BENCHMARK_MLP_OVERSAMPLE
export BENCHMARK_LEVEL3_USE_AUX_CHANNEL
export BENCHMARK_REQUIRE_TRAIN_SELECTION
export BENCHMARK_STRICT_LEVEL_COMPLETENESS

run_phase() {
    local mode="$1"

    python semantic_screening_models.py \
        --dataset "${DATASET}" \
        --embedding "${EMBEDDING}" \
        --levels "${LEVELS[@]}" \
        --epochs "${EPOCHS}" \
        --model_selection_metric "${MODEL_SELECTION_METRIC}" \
        --output_dir "${OUTPUT_ROOT}" \
        "--${mode}"
}

echo "============================================================"
echo " Benchmark automation | dataset=${DATASET} embedding=${EMBEDDING}"
echo " Levels: ${LEVELS[*]} | epochs=${EPOCHS} | patience=10% of epochs (auto)"
echo " Model selection metric: ${MODEL_SELECTION_METRIC}"
echo " MLP tuning: cv=${BENCHMARK_MLP_USE_CV}, folds=${BENCHMARK_MLP_FOLDS}, cal_restarts=${BENCHMARK_MLP_CAL_RESTARTS}, ensemble=${BENCHMARK_MLP_ENSEMBLE}, oversample=${BENCHMARK_MLP_OVERSAMPLE}"
echo " Level3 aux channel: ${BENCHMARK_LEVEL3_USE_AUX_CHANNEL}"
echo " Rigor: require_train_selection=${BENCHMARK_REQUIRE_TRAIN_SELECTION}, strict_completeness=${BENCHMARK_STRICT_LEVEL_COMPLETENESS}"
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
    "level3_attnpool_knn",
    "level3_attnpool_mlp",
    "level3a_attnpool_mlp",
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
print("Focus model: level3a_attnpool_mlp (primary MCC target).")
PY

echo ""
echo "============================================================"
echo " Done. Results saved under:"
echo "   ${OUTPUT_ROOT}/train"
echo "   ${OUTPUT_ROOT}/test"
echo "============================================================"
