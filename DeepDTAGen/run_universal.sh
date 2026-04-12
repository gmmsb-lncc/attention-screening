#!/usr/bin/env bash
# ===========================================================================
# run_universal.sh
#
# Complete pipeline for DeepDTAGen on universal kinase datasets:
#   1. Convert universal scaffold-split data to DeepDTAGen format
#   2. Evaluate pretrained models on universal test sets
#   3. Train from scratch on universal data (non_human, human, all)
#
# Usage:
#   bash run_universal.sh                  # Full pipeline
#   bash run_universal.sh --eval-only      # Only evaluate pretrained models
#   bash run_universal.sh --train-only     # Only train on universal data
#
# Prerequisites:
#   1. Run setup_env.sh first
#   2. Universal scaffold splits in scaffolds_splits/output/
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="deepdtagen"

# Scaffold splits path (relative to parent docktkinase directory)
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${SCRIPT_DIR}/../scaffolds_splits/output}"

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
echo " Scaffold splits: ${SCAFFOLD_DIR}"
echo "============================================"

mkdir -p results saved_models logs data/processed

# ── Step 1: Convert universal data ────────────────────────────────────────
echo ""
echo "[Step 1] Converting universal kinase data to DeepDTAGen format..."
if [ ! -f "data/processed/kinase_non_human_test.pt" ]; then
    python convert_universal_data.py --scaffold-dir "${SCAFFOLD_DIR}"
else
    echo "  [SKIP] Converted data already exists. Delete data/processed/ to regenerate."
fi

# ── Step 2: Evaluate pretrained models ────────────────────────────────────
if [ "${TRAIN_ONLY}" = false ]; then
    echo ""
    echo "[Step 2] Evaluating pretrained models on universal test sets..."
    if [ -f "eval_pretrained_universal.py" ]; then
        python eval_pretrained_universal.py 2>&1 | tee results/log_pretrained_eval.txt
    else
        echo "  [SKIP] eval_pretrained_universal.py not found."
    fi
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

        # Check that train/test data exists
        if [ ! -f "data/processed/kinase_${ds}_train.pt" ]; then
            echo "  [ERROR] data/processed/kinase_${ds}_train.pt not found!"
            echo "  Run Step 1 first."
            continue
        fi

        python train_universal.py --dataset "${ds}" --epochs 200 --eval_every 10 \
            2>&1 | tee "results/log_train_${ds}.txt"
        echo " [DONE] ${ds} at $(date '+%Y-%m-%d %H:%M:%S')"
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Results directory: results/"
if [ -f "results/deepdtagen_pretrained_universal.csv" ]; then
    echo " Pretrained eval: results/deepdtagen_pretrained_universal.csv"
fi
echo " Training logs: results/log_train_*.txt"
echo "============================================"
