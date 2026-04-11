#!/usr/bin/env bash
# ===========================================================================
# run_dtkinase_drugban.sh
#
# Trains DT-Kinase (Level4CNN v7) on all three DrugBAN datasets using:
#   - Random split (7:1:2)     → matches DrugBAN "in-domain" evaluation
#   - Cold pair split          → matches DrugBAN "cross-domain" evaluation
#
# Results are saved under DrugBAN/results/dtkinase/{dataset}/{split}/
#
# Usage:
#   bash run_dtkinase_drugban.sh                         # full run (all datasets, both splits)
#   DATASET=human SPLIT=random bash run_dtkinase_drugban.sh  # single combo
#   ESM=150M bash run_dtkinase_drugban.sh                # ESM-2 150M embeddings
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# --- Configuration (override via env vars) ---------------------------------
DATASET="${DATASET:-all}"          # human | biosnap | bindingdb | all
SPLIT="${SPLIT:-all}"              # random | cold | all
ESM="${ESM:-8M}"                   # 8M | 150M | 650M
EPOCHS="${EPOCHS:-100}"            # DrugBAN paper uses 100
LR="${LR:-5e-5}"                   # DrugBAN paper uses 5e-5
BATCH_SIZE="${BATCH_SIZE:-64}"     # DrugBAN paper uses 64
SEEDS="${SEEDS:-42 123 456 789 1024}"
PATIENCE="${PATIENCE:-10}"         # 10% of 100 epochs
VARIANT="${VARIANT:-v7}"           # v7 = dot-product interaction maps
NUM_HEADS="${NUM_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-32}"
CNN_CHANNELS="${CNN_CHANNELS:-64}"
DROPOUT="${DROPOUT:-0.3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
NO_DOUBLE="${NO_DOUBLE:-0}"        # use float64 by default for precision
FORCE="${FORCE:-0}"                # skip if results.json exists unless =1

# --- Environment ------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-base}"

echo "==========================================================="
echo " DT-Kinase × DrugBAN Benchmark"
echo "==========================================================="
echo " Datasets:     ${DATASET}"
echo " Splits:       ${SPLIT}"
echo " ESM-2:        ${ESM}"
echo " Epochs:       ${EPOCHS}  LR: ${LR}  Batch: ${BATCH_SIZE}"
echo " Seeds:        ${SEEDS}"
echo " Patience:     ${PATIENCE}"
echo " CNN variant:  ${VARIANT}  heads=${NUM_HEADS}  head_dim=${HEAD_DIM}"
echo " CNN channels: ${CNN_CHANNELS}  dropout=${DROPOUT}  wd=${WEIGHT_DECAY}"
echo " Use float64:  $([ "${NO_DOUBLE}" = "1" ] && echo "NO (float32)" || echo "YES (float64)")"
echo " Force rerun:  ${FORCE}"
echo " Script dir:   ${SCRIPT_DIR}"
echo " Repo root:    ${REPO_ROOT}"
echo "==========================================================="
echo ""

# --- Activate conda env if needed -------------------------------------------
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null || echo "[WARN] Could not activate '${CONDA_ENV}', using current env"
fi

# --- Build command -----------------------------------------------------------
CMD=(
    python "${SCRIPT_DIR}/run_dtkinase_on_drugban_datasets.py"
    "--dataset" "${DATASET}"
    "--split"   "${SPLIT}"
    "--esm"     "${ESM}"
    "--epochs"  "${EPOCHS}"
    "--lr"      "${LR}"
    "--batch-size" "${BATCH_SIZE}"
    "--seeds"   ${SEEDS}
    "--patience" "${PATIENCE}"
    "--variant" "${VARIANT}"
    "--num-heads" "${NUM_HEADS}"
    "--head-dim" "${HEAD_DIM}"
    "--cnn-channels" "${CNN_CHANNELS}"
    "--dropout" "${DROPOUT}"
    "--weight-decay" "${WEIGHT_DECAY}"
)

if [ "${NO_DOUBLE}" = "1" ]; then
    CMD+=("--no-double")
fi

if [ "${FORCE}" = "1" ]; then
    CMD+=("--force")
fi

# --- Run from repo root (so 'benchmark' module is importable) ---------------
echo "Running: ${CMD[*]}"
echo ""
cd "${REPO_ROOT}"
"${CMD[@]}"

echo ""
echo "==========================================================="
echo " All done. Results under: DrugBAN/results/dtkinase/"
echo "==========================================================="
