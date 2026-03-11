#!/usr/bin/env bash
# Setup conda environment for GraphBAN baseline.
# Run: bash setup_env.sh
set -euo pipefail

ENV_NAME="graphban"

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Environment '${ENV_NAME}' already exists. Activating..."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' (Python 3.11)..."
    conda create -y --name "${ENV_NAME}" python=3.11
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Detect CUDA for PyTorch, torch-geometric, and DGL installation
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    echo "[INFO] Detected CUDA ${CUDA_VER}"
    if [ "${CUDA_MAJOR}" -ge 12 ]; then
        TORCH_INDEX="cu121"
        DGL_CUDA="cu121"
    else
        TORCH_INDEX="cu118"
        DGL_CUDA="cu118"
    fi
    echo "[INFO] Installing PyTorch with CUDA (${TORCH_INDEX})..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

    echo "[INFO] Installing torch-geometric..."
    pip install torch-geometric
    TORCH_VER=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${TORCH_INDEX}.html"

    echo "[INFO] Installing DGL with CUDA (${DGL_CUDA})..."
    pip uninstall dgl -y 2>/dev/null || true
    pip install dgl -f "https://data.dgl.ai/wheels/${DGL_CUDA}/repo.html"
else
    echo "[INFO] No GPU detected. Installing CPU-only versions..."
    pip install torch torchvision torchaudio

    echo "[INFO] Installing torch-geometric (CPU)..."
    pip install torch-geometric
    TORCH_VER=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html"

    echo "[INFO] Installing DGL (CPU)..."
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
fi

pip install dgllife

echo "[INFO] Installing remaining dependencies..."
pip install --force-reinstall setuptools packaging
pip install torchmetrics transformers fair-esm
pip install "numpy<2" rdkit-pypi scikit-learn pandas prettytable yacs tqdm pyarrow

echo "[INFO] Cloning GraphBAN source code..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHBAN_SRC="${SCRIPT_DIR}/src"

if [ -d "${GRAPHBAN_SRC}/.git" ]; then
    echo "[INFO] GraphBAN source already cloned at ${GRAPHBAN_SRC}"
else
    git clone https://github.com/HamidHadipour/GraphBAN.git "${GRAPHBAN_SRC}"
    echo "[INFO] GraphBAN cloned to ${GRAPHBAN_SRC}"
fi

echo ""
echo "====================================="
echo " GraphBAN environment ready!"
echo " Activate: conda activate ${ENV_NAME}"
echo "====================================="
