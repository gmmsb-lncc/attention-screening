#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Automated train/test benchmark for Levels 3 and 3a
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
BENCHMARK_LEVEL3_SELECTION_METRIC="${BENCHMARK_LEVEL3_SELECTION_METRIC:-downstream_mcc}"
BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY="${BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY:-10}"
FOCUS_MODEL="${FOCUS_MODEL:-level3a_attnpool_mlp}"
TARGET_TEST_MCC="${TARGET_TEST_MCC:-0.6}"
MAX_TEST_MCC_STD="${MAX_TEST_MCC_STD:-0.08}"
BENCHMARK_ENFORCE_RIGOR="${BENCHMARK_ENFORCE_RIGOR:-1}"

# Strong MCC profile (fast to adopt, no code changes required).
BENCHMARK_MLP_USE_CV="${BENCHMARK_MLP_USE_CV:-1}"
BENCHMARK_MLP_FOLDS="${BENCHMARK_MLP_FOLDS:-5}"
BENCHMARK_MLP_CAL_RESTARTS="${BENCHMARK_MLP_CAL_RESTARTS:-7}"
BENCHMARK_MLP_ENSEMBLE="${BENCHMARK_MLP_ENSEMBLE:-11}"
BENCHMARK_MLP_OVERSAMPLE="${BENCHMARK_MLP_OVERSAMPLE:-1}"
BENCHMARK_LEVEL3_USE_AUX_CHANNEL="${BENCHMARK_LEVEL3_USE_AUX_CHANNEL:-1}"

# OOF threshold refinement (re-calibrates decision threshold on full training set).
BENCHMARK_MLP_OOF_THRESHOLD="${BENCHMARK_MLP_OOF_THRESHOLD:-1}"
BENCHMARK_MLP_OOF_FOLDS="${BENCHMARK_MLP_OOF_FOLDS:-5}"

# Scientific-rigor safeguards (test cannot guide tuning).
BENCHMARK_REQUIRE_TRAIN_SELECTION="${BENCHMARK_REQUIRE_TRAIN_SELECTION:-1}"
BENCHMARK_STRICT_LEVEL_COMPLETENESS="${BENCHMARK_STRICT_LEVEL_COMPLETENESS:-1}"

export BENCHMARK_MLP_USE_CV
export BENCHMARK_MLP_FOLDS
export BENCHMARK_MLP_CAL_RESTARTS
export BENCHMARK_MLP_ENSEMBLE
export BENCHMARK_MLP_OVERSAMPLE
export BENCHMARK_LEVEL3_USE_AUX_CHANNEL
export BENCHMARK_LEVEL3_SELECTION_METRIC
export BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY
export BENCHMARK_MLP_OOF_THRESHOLD
export BENCHMARK_MLP_OOF_FOLDS
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
echo " MLP OOF threshold: enabled=${BENCHMARK_MLP_OOF_THRESHOLD}, oof_folds=${BENCHMARK_MLP_OOF_FOLDS}"
echo " Level3 aux channel: ${BENCHMARK_LEVEL3_USE_AUX_CHANNEL}"
echo " Level3 checkpoint selection: ${BENCHMARK_LEVEL3_SELECTION_METRIC} (eval_every=${BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY})"
echo " Rigor: require_train_selection=${BENCHMARK_REQUIRE_TRAIN_SELECTION}, strict_completeness=${BENCHMARK_STRICT_LEVEL_COMPLETENESS}"
echo " Acceptance gate: focus_model=${FOCUS_MODEL}, target_test_mcc>=${TARGET_TEST_MCC}, max_test_mcc_std<=${MAX_TEST_MCC_STD}"
echo " Output root: ${OUTPUT_ROOT}"
echo "============================================================"

if [[ "${BENCHMARK_ENFORCE_RIGOR}" == "1" ]]; then
    if [[ "${BENCHMARK_REQUIRE_TRAIN_SELECTION}" != "1" || "${BENCHMARK_STRICT_LEVEL_COMPLETENESS}" != "1" ]]; then
        echo "ERROR: BENCHMARK_ENFORCE_RIGOR=1 requires BENCHMARK_REQUIRE_TRAIN_SELECTION=1 and BENCHMARK_STRICT_LEVEL_COMPLETENESS=1"
        exit 2
    fi
fi

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
OUTPUT_ROOT_ENV="${OUTPUT_ROOT}" \
FOCUS_MODEL_ENV="${FOCUS_MODEL}" \
TARGET_TEST_MCC_ENV="${TARGET_TEST_MCC}" \
MAX_TEST_MCC_STD_ENV="${MAX_TEST_MCC_STD}" \
python - <<'PY'
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

focus_model = os.environ["FOCUS_MODEL_ENV"]
target_test_mcc = float(os.environ["TARGET_TEST_MCC_ENV"])
max_test_mcc_std = float(os.environ["MAX_TEST_MCC_STD_ENV"])

test_focus_mcc = None
test_focus_std = None

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
        if phase == "test" and model == focus_model:
            test_focus_mcc = mcc if isinstance(mcc, (float, int)) else None
            test_focus_std = mcc_std if isinstance(mcc_std, (float, int)) else None

    if missing_in_phase:
        print(f"{phase:>6}  missing_models: {missing_in_phase}")
        print(f"{phase:>6}  available_keys: {sorted(results.keys())[:12]}")

print("")
print(f"Focus model: {focus_model} (primary MCC target).")

gate_ok = True
if test_focus_mcc is None:
    gate_ok = False
    print(f"GATE FAIL: Missing test MCC for focus model '{focus_model}'.")
else:
    if test_focus_mcc < target_test_mcc:
        gate_ok = False
        print(
            f"GATE FAIL: test MCC for {focus_model} is {test_focus_mcc:.4f} "
            f"(< target {target_test_mcc:.4f})."
        )
    else:
        print(
            f"GATE PASS: test MCC for {focus_model} is {test_focus_mcc:.4f} "
            f"(>= target {target_test_mcc:.4f})."
        )

if test_focus_std is None:
    gate_ok = False
    print(f"GATE FAIL: Missing test MCC std for focus model '{focus_model}'.")
elif test_focus_std > max_test_mcc_std:
    gate_ok = False
    print(
        f"GATE FAIL: test MCC std for {focus_model} is {test_focus_std:.4f} "
        f"(> max {max_test_mcc_std:.4f})."
    )
else:
    print(
        f"GATE PASS: test MCC std for {focus_model} is {test_focus_std:.4f} "
        f"(<= max {max_test_mcc_std:.4f})."
    )

if not gate_ok:
    raise SystemExit(2)
PY

echo ""
echo "============================================================"
echo " Done. Results saved under:"
echo "   ${OUTPUT_ROOT}/train"
echo "   ${OUTPUT_ROOT}/test"
echo "============================================================"
