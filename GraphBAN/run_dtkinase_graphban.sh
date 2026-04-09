#!/usr/bin/env bash
# ===========================================================================
# run_dtkinase_graphban.sh
#
# Trains DT-Kinase (Level4CNN v7) on GraphBAN's benchmark datasets using:
#   - Transductive split → in-domain evaluation
#   - Inductive split    → OOD/cold evaluation
#
# GraphBAN datasets: biosnap, bindingdb, kiba, c.elegans, pdb
# GraphBAN seeds:    12, 14, 16, 18, 20 (per-seed splits from upstream)
#
# Results are saved under GraphBAN/results/graphban_dataset_tests/{ds}/{mode}/
#
# Usage:
#   bash run_dtkinase_graphban.sh                              # full run (5 datasets × 2 modes)
#   DATASET=biosnap SPLIT=transductive bash run_dtkinase_graphban.sh  # single combo
#   ESM=150M bash run_dtkinase_graphban.sh                     # ESM-2 150M embeddings
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# --- Configuration (override via env vars) ---------------------------------
DATASET="${DATASET:-all}"          # biosnap | bindingdb | kiba | c.elegans | pdb | all
SPLIT="${SPLIT:-all}"              # transductive | inductive | all
ESM="${ESM:-8M}"                   # 8M | 150M | 650M
EPOCHS="${EPOCHS:-100}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEEDS="${SEEDS:-12 14 16 18 20}"   # GraphBAN canonical seeds
PATIENCE="${PATIENCE:-10}"
VARIANT="${VARIANT:-v7}"
NUM_HEADS="${NUM_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-32}"
CNN_CHANNELS="${CNN_CHANNELS:-64}"
DROPOUT="${DROPOUT:-0.3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
NO_DOUBLE="${NO_DOUBLE:-0}"
FORCE="${FORCE:-0}"

echo "==========================================================="
echo " DT-Kinase × GraphBAN Benchmark"
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

# --- Verify data exists ----------------------------------------------------
DATA_DIR="${SCRIPT_DIR}/upstream_data/Data"
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: GraphBAN upstream data not found at ${DATA_DIR}"
    echo ""
    echo "To download, run:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  git clone --depth 1 --filter=blob:none --sparse \\"
    echo "      https://github.com/HamidHadipour/GraphBAN.git upstream_data"
    echo "  cd upstream_data && git sparse-checkout set Data"
    exit 1
fi

# --- Build command -----------------------------------------------------------
CMD=(
    python "${SCRIPT_DIR}/run_dtkinase_on_graphban_datasets.py"
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
echo " All done. Results under: GraphBAN/results/graphban_dataset_tests/"
echo "==========================================================="
