#!/usr/bin/env bash
# Setup conda environment for GraphBAN baseline.
# Run: bash setup_env.sh
# All packages installed via conda. Only fair-esm uses pip (not on conda).
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

# ── PyTorch + DGL + PyG ────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    CUDA_MINOR=$(echo "${CUDA_VER}" | cut -d. -f2)
    echo "[INFO] Detected CUDA ${CUDA_VER}"

    # CUDA channel tags
    if [ "${CUDA_MAJOR}" -ge 12 ]; then
        PYTORCH_CUDA="12.1"
        DGL_LABEL="cu121"
        PYG_CUDA="cu121"
    else
        PYTORCH_CUDA="11.8"
        DGL_LABEL="cu118"
        PYG_CUDA="cu118"
    fi

    echo "[INFO] Installing PyTorch ${PYTORCH_CUDA} via conda..."
    conda install -y pytorch torchvision torchaudio \
        pytorch-cuda=${PYTORCH_CUDA} \
        -c pytorch -c nvidia

    echo "[INFO] Installing PyTorch-Geometric via conda (pyg channel)..."
    conda install -y pyg pytorch-scatter pytorch-sparse pytorch-cluster \
        pytorch-spline-conv \
        -c pyg

    echo "[INFO] Installing DGL (CUDA ${DGL_LABEL}) via conda..."
    conda install -y dgl \
        -c "dglteam/label/${DGL_LABEL}"
else
    echo "[INFO] No GPU detected. Installing CPU-only versions via conda..."

    conda install -y pytorch torchvision torchaudio cpuonly \
        -c pytorch

    echo "[INFO] Installing PyTorch-Geometric (CPU) via conda..."
    conda install -y pyg pytorch-scatter pytorch-sparse pytorch-cluster \
        pytorch-spline-conv \
        -c pyg

    echo "[INFO] Installing DGL (CPU) via conda..."
    conda install -y dgl \
        -c dglteam
fi

# ── DGLlife ────────────────────────────────────────────────────────────────
echo "[INFO] Installing dgllife via conda..."
conda install -y dgllife -c conda-forge

# ── Remaining dependencies (all via conda-forge) ──────────────────────────
echo "[INFO] Installing remaining dependencies via conda-forge..."
conda install -y \
    "numpy<2" \
    scikit-learn \
    pandas \
    tqdm \
    pyarrow \
    rdkit \
    prettytable \
    yacs \
    torchmetrics \
    transformers \
    setuptools \
    packaging \
    -c conda-forge

# ── fair-esm: pip only (not available on conda) ───────────────────────────
echo "[INFO] Installing fair-esm via pip (not available on conda)..."
pip install fair-esm

# ── GraphBAN source ────────────────────────────────────────────────────────
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
