#!/usr/bin/env bash
# ===========================================================================
# run_training.sh
#
# Train DeepDTAGen on all 3 datasets (Davis, KIBA, BindingDB).
# Designed for execution on the GPU machine (RTX 4090).
#
# Usage:
#   bash run_training.sh                  # Train all 3 datasets
#   bash run_training.sh davis            # Train only on Davis
#   bash run_training.sh kiba             # Train only on KIBA
#   bash run_training.sh bindingdb        # Train only on BindingDB
#
# Prerequisites:
#   1. Run setup_env.sh first to create the conda environment
#   2. Ensure data CSVs are in data/ (extract data.rar if needed)
#   3. Ensure data/processed/*.pt files exist (run create_data.py if not)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="deepdtagen"

# ── Activate conda ─────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

cd "${SCRIPT_DIR}"

echo "============================================"
echo " DeepDTAGen Training"
echo " Python: $(python --version 2>&1)"
echo " PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo " CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo " GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "============================================"
echo ""

# ── Ensure directories exist ──────────────────────────────────────────────
mkdir -p saved_models Affinities logs

# ── Check if processed data exists, create if needed ──────────────────────
if [ ! -f "data/processed/davis_train.pt" ]; then
    echo "[INFO] Processed data not found. Running create_data.py..."
    python create_data.py
fi

# ── Dataset mapping: name -> index ────────────────────────────────────────
declare -A DS_MAP=( ["davis"]=0 ["kiba"]=1 ["bindingdb"]=2 )

# ── Determine which datasets to train ─────────────────────────────────────
if [ $# -ge 1 ]; then
    DATASETS=("$1")
else
    DATASETS=("davis" "kiba" "bindingdb")
fi

# ── Train ─────────────────────────────────────────────────────────────────
for ds in "${DATASETS[@]}"; do
    idx="${DS_MAP[$ds]}"
    echo ""
    echo "================================================================"
    echo " Training on: ${ds} (index=${idx})"
    echo " Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
    
    python training.py "${idx}" 2>&1 | tee "logs/training_${ds}_$(date +%s).log"
    
    echo ""
    echo " [DONE] ${ds} finished at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
done

echo ""
echo "============================================"
echo " All training completed!"
echo " Models saved in: saved_models/"
echo " Affinities in:   Affinities/"
echo " Logs in:         logs/"
echo "============================================"
