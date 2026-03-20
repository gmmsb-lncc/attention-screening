#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Automated train/test benchmark (any level)
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
LEVELS_CSV="${LEVELS_CSV:-3a}"
IFS=',' read -r -a LEVELS <<< "${LEVELS_CSV}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/benchmark_${DATASET}_${EMBEDDING}_16_03_2026}"
EPOCHS="${EPOCHS:-500}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-mcc}"
BENCHMARK_LEVEL3_SELECTION_METRIC="${BENCHMARK_LEVEL3_SELECTION_METRIC:-val_loss}"
BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY="${BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY:-10}"
# ---------- Auto-derive focus model from first requested level ----------
_derive_focus_model() {
    case "$1" in
        1a)  echo "level1a_fp_mlp" ;;
        1b)  echo "level1b_ligmean_mlp" ;;
        1c)  echo "level1c_ligattn_mlp" ;;
        2)   echo "level2_meanpool_mlp" ;;
        3)   echo "level3_attnpool_mlp" ;;
        3a)  echo "level3a_attnpool_mlp" ;;
        4)   echo "level4_crossatt_mlp" ;;
        4a)  echo "level4a_crossatt_mlp" ;;
        4lora) echo "level4_lora_mlp" ;;
        4cnn) echo "level4_cnn_mlp" ;;
        5)   echo "level5_da_mlp" ;;
        5b)  echo "level5b_da_mlp" ;;
        6a)  echo "level6a_ban_mlp" ;;
        6b)  echo "level6b_ban_mlp" ;;
        *)   echo "level${1}_mlp" ;;
    esac
}
FOCUS_MODEL="${FOCUS_MODEL:-$(_derive_focus_model "${LEVELS[0]}")}"
TARGET_TEST_MCC="${TARGET_TEST_MCC:-0.6}"
MAX_TEST_MCC_STD="${MAX_TEST_MCC_STD:-0.08}"
BENCHMARK_ENFORCE_RIGOR="${BENCHMARK_ENFORCE_RIGOR:-1}"

# Strong MCC profile (fast to adopt, no code changes required).
BENCHMARK_MLP_USE_CV="${BENCHMARK_MLP_USE_CV:-1}"
BENCHMARK_MLP_FOLDS="${BENCHMARK_MLP_FOLDS:-5}"
BENCHMARK_MLP_CAL_RESTARTS="${BENCHMARK_MLP_CAL_RESTARTS:-3}"
BENCHMARK_MLP_ENSEMBLE="${BENCHMARK_MLP_ENSEMBLE:-5}"
BENCHMARK_MLP_OVERSAMPLE="${BENCHMARK_MLP_OVERSAMPLE:-0}"
BENCHMARK_LEVEL3_USE_AUX_CHANNEL="${BENCHMARK_LEVEL3_USE_AUX_CHANNEL:-0}"
BENCHMARK_LEVEL3_FULL_TRAIN_FEATURES="${BENCHMARK_LEVEL3_FULL_TRAIN_FEATURES:-0}"

# Level 3 interaction features (product, abs-diff, cosine). Default OFF for baseline.
BENCHMARK_LEVEL3_INTERACTION_FEATURES="${BENCHMARK_LEVEL3_INTERACTION_FEATURES:-0}"

# Level 3 aux_head trained with interaction features (product + abs_diff).
# Aligns the training signal with the 4×hidden features extracted for the
# downstream MLP, improving representation quality.  (Default: 1 = enabled)
BENCHMARK_LEVEL3_AUX_INTERACTIONS="${BENCHMARK_LEVEL3_AUX_INTERACTIONS:-0}"

# Level 3 hyperparameter overrides (leave empty for auto-scaled defaults).
# LR: learning rate (default: uses --learning_rate CLI value; recommended: 5e-4 for 150M+).
# DROPOUT: dropout rate (default: 0.3; try 0.2 for richer embeddings).
# WEIGHT_DECAY: AdamW weight decay (default: 0.02; try 0.04 for 150M+).
BENCHMARK_LEVEL3_LR="${BENCHMARK_LEVEL3_LR:-}"
BENCHMARK_LEVEL3_DROPOUT="${BENCHMARK_LEVEL3_DROPOUT:-}"
BENCHMARK_LEVEL3_WEIGHT_DECAY="${BENCHMARK_LEVEL3_WEIGHT_DECAY:-}"

BENCHMARK_LEVEL4_INTERACTION_FEATURES="${BENCHMARK_LEVEL4_INTERACTION_FEATURES:-1}"

# OOF threshold refinement (re-calibrates decision threshold on training set OOF predictions).
BENCHMARK_MLP_OOF_THRESHOLD="${BENCHMARK_MLP_OOF_THRESHOLD:-1}"
BENCHMARK_MLP_OOF_FOLDS="${BENCHMARK_MLP_OOF_FOLDS:-5}"

# Full refit: final ensemble trains on 100% of data without early stopping.
BENCHMARK_MLP_FULL_REFIT="${BENCHMARK_MLP_FULL_REFIT:-0}"

# Level 3 architecture scaling (auto-scaled by default, set explicitly for 650M).
BENCHMARK_LEVEL3_HIDDEN_DIM="${BENCHMARK_LEVEL3_HIDDEN_DIM:-}"

# Multi-layer MoLFormer features: comma-separated layer indices to append.
# Empty = disabled (default). Example: "4,5,6" for last 3 transformer layers.
BENCHMARK_LEVEL3_MULTILAYER_LAYERS="${BENCHMARK_LEVEL3_MULTILAYER_LAYERS:-}"

# Level 4 regularisation (tuned for 650M embeddings by default).
BENCHMARK_LEVEL4_DROPOUT="${BENCHMARK_LEVEL4_DROPOUT:-0.30}"
BENCHMARK_LEVEL4_WEIGHT_DECAY="${BENCHMARK_LEVEL4_WEIGHT_DECAY:-0.06}"

# Level 4 CNN train-to-zero mode: disable early stopping, train until
# both train_loss and val_loss drop below threshold (default 0.01).
BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO="${BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO:-0}"
BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR="${BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR:-0.01}"

# Scientific-rigor safeguards (test cannot guide tuning).
BENCHMARK_REQUIRE_TRAIN_SELECTION="${BENCHMARK_REQUIRE_TRAIN_SELECTION:-1}"
BENCHMARK_STRICT_LEVEL_COMPLETENESS="${BENCHMARK_STRICT_LEVEL_COMPLETENESS:-1}"

export BENCHMARK_MLP_USE_CV
export BENCHMARK_MLP_FOLDS
export BENCHMARK_MLP_CAL_RESTARTS
export BENCHMARK_MLP_ENSEMBLE
export BENCHMARK_MLP_OVERSAMPLE
export BENCHMARK_LEVEL3_USE_AUX_CHANNEL
export BENCHMARK_LEVEL3_FULL_TRAIN_FEATURES
export BENCHMARK_LEVEL3_INTERACTION_FEATURES
export BENCHMARK_LEVEL3_AUX_INTERACTIONS
export BENCHMARK_LEVEL3_LR
export BENCHMARK_LEVEL3_DROPOUT
export BENCHMARK_LEVEL3_WEIGHT_DECAY
export BENCHMARK_LEVEL3_SELECTION_METRIC
export BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY
export BENCHMARK_LEVEL4_INTERACTION_FEATURES
export BENCHMARK_MLP_OOF_THRESHOLD
export BENCHMARK_MLP_OOF_FOLDS
export BENCHMARK_MLP_FULL_REFIT
export BENCHMARK_REQUIRE_TRAIN_SELECTION
export BENCHMARK_STRICT_LEVEL_COMPLETENESS
export BENCHMARK_LEVEL3_HIDDEN_DIM
export BENCHMARK_LEVEL3_MULTILAYER_LAYERS
export BENCHMARK_LEVEL4_DROPOUT
export BENCHMARK_LEVEL4_WEIGHT_DECAY
export BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO
export BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR

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
echo " MLP full refit: ${BENCHMARK_MLP_FULL_REFIT} (final ensemble trains without early stopping)"
echo " Level3 aux channel: ${BENCHMARK_LEVEL3_USE_AUX_CHANNEL}"
echo " Level3 aux interactions: ${BENCHMARK_LEVEL3_AUX_INTERACTIONS} (aux_head trains with product+abs_diff)"
echo " Level3 full train features: ${BENCHMARK_LEVEL3_FULL_TRAIN_FEATURES} (transfer learning mode)"
echo " Level3 checkpoint selection: ${BENCHMARK_LEVEL3_SELECTION_METRIC} (eval_every=${BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY})"
echo " Level3 hidden_dim: ${BENCHMARK_LEVEL3_HIDDEN_DIM:-auto}"
echo " Level3 lr: ${BENCHMARK_LEVEL3_LR:-auto}, dropout: ${BENCHMARK_LEVEL3_DROPOUT:-0.3}, weight_decay: ${BENCHMARK_LEVEL3_WEIGHT_DECAY:-0.02}"
echo " Level4 interaction features: ${BENCHMARK_LEVEL4_INTERACTION_FEATURES}"
echo " Level4 dropout: ${BENCHMARK_LEVEL4_DROPOUT}, weight_decay: ${BENCHMARK_LEVEL4_WEIGHT_DECAY}"
echo " Level4 CNN train-to-zero: ${BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO} (threshold: ${BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR})"
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
LEVELS_CSV_ENV="${LEVELS_CSV}" \
OUTPUT_ROOT_ENV="${OUTPUT_ROOT}" \
FOCUS_MODEL_ENV="${FOCUS_MODEL}" \
TARGET_TEST_MCC_ENV="${TARGET_TEST_MCC}" \
MAX_TEST_MCC_STD_ENV="${MAX_TEST_MCC_STD}" \
python - <<'PY'
import json
import os
from pathlib import Path

# ---------- Build targets list from requested levels ----------
_LEVEL_MODEL_MAP = {
    "1a": ["level1a_fp_knn", "level1a_fp_mlp"],
    "1b": ["level1b_ligmean_knn", "level1b_ligmean_mlp"],
    "1c": ["level1c_ligattn_knn", "level1c_ligattn_mlp"],
    "2":  ["level2_meanpool_knn", "level2_meanpool_mlp"],
    "3":  ["level3_attnpool_knn", "level3_attnpool_mlp"],
    "3a": ["level3a_attnpool_mlp"],
    "4":  ["level4_crossatt_knn", "level4_crossatt_mlp"],
    "4a": ["level4a_crossatt_mlp"],
    "4lora": ["level4_lora_mlp"],
    "4cnn": ["level4_cnn_mlp"],
    "5":  ["level5_da_knn", "level5_da_mlp"],
    "5b": ["level5b_da_knn", "level5b_da_mlp"],
    "6a": ["level6a_ban_knn", "level6a_ban_mlp"],
    "6b": ["level6b_ban_knn", "level6b_ban_mlp"],
}

levels_csv = os.environ["LEVELS_CSV_ENV"]
targets = []
for lvl in levels_csv.split(","):
    lvl = lvl.strip()
    targets.extend(_LEVEL_MODEL_MAP.get(lvl, [f"level{lvl}_mlp"]))

# The orchestrator appends /train and /test to --output_dir.
output_root = Path(os.environ["OUTPUT_ROOT_ENV"])

paths = {
    "train": output_root / "train" / "benchmark_comparison.json",
    "test": output_root / "test" / "benchmark_comparison.json",
}

col_w = max(20, max((len(t) for t in targets), default=20) + 2)
print(f"{'phase':>6}  {'model':<{col_w}} {'MCC':>8} {'MCC_std':>8}")
print("-" * (26 + col_w))

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
        print(f"{phase:>6}  {model:<{col_w}} {mcc_txt:>8} {std_txt:>8}")
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
