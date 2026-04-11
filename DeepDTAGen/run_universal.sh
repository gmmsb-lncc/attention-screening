#!/usr/bin/env bash
# ===========================================================================
# run_universal.sh
#
# Complete pipeline for DeepDTAGen on universal kinase datasets:
#   1. Convert universal data to DeepDTAGen format
#   2. Evaluate pretrained models (bindingdb, davis, kiba) on universal test sets
#   3. Train from scratch on universal data
#
# Usage:
#   bash run_universal.sh                  # Full pipeline
#   bash run_universal.sh --eval-only      # Only evaluate pretrained models
#   bash run_universal.sh --train-only     # Only train on universal data
#
# Prerequisites:
#   1. Run setup_env.sh first
#   2. Pretrained models in models/ (extract pretrained_models.zip)
#   3. Universal data in ../DrugBAN/datasets/kinase/{non_human,human,all}/scaffold/
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="deepdtagen"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
cd "${SCRIPT_DIR}"

# Parse args
EVAL_ONLY=false
TRAIN_ONLY=false
for arg in "$@"; do
    case $arg in
        --eval-only)  EVAL_ONLY=true ;;
        --train-only) TRAIN_ONLY=true ;;
    esac
done

echo "============================================"
echo " DeepDTAGen — Universal Kinase Pipeline"
echo " Python: $(python --version 2>&1)"
echo " PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo " CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo " GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "============================================"

mkdir -p results saved_models logs

# ── Step 1: Convert universal data ────────────────────────────────────────
echo ""
echo "[Step 1] Converting universal kinase data to DeepDTAGen format..."
if [ ! -f "data/processed/kinase_non_human_test.pt" ]; then
    python convert_universal_data.py
else
    echo "  [SKIP] Converted data already exists."
fi

# ── Step 2: Evaluate pretrained models ────────────────────────────────────
if [ "${TRAIN_ONLY}" = false ]; then
    echo ""
    echo "[Step 2] Evaluating pretrained models on universal test sets..."
    python eval_pretrained_universal.py 2>&1 | tee results/log_pretrained_eval.txt
fi

# ── Step 3: Train on universal data ──────────────────────────────────────
if [ "${EVAL_ONLY}" = false ]; then
    echo ""
    echo "[Step 3] Training DeepDTAGen on universal kinase datasets..."
    for ds in non_human human all; do
        echo ""
        echo "================================================================"
        echo " Training: ${ds}"
        echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
        python train_universal.py --dataset "${ds}" --epochs 200 --eval_every 10 \
            2>&1 | tee "results/log_train_${ds}.txt"
        echo " [DONE] ${ds} at $(date '+%Y-%m-%d %H:%M:%S')"
    done
fi

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Pretrained eval: results/deepdtagen_pretrained_universal.csv"
echo " Training results: results/deepdtagen_trained_*.csv"
echo "============================================"
