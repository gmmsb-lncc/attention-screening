#!/usr/bin/env bash
# Setup conda environment for DeepDTAGen reproduction.
#
# Usage:
#   bash setup_env.sh
#
# Following the same pattern as DrugBAN/GraphBAN setup scripts.
#
# Installation order (CRITICAL — changing order breaks things):
#   1. conda: Python 3.10 + scientific deps (numpy, sklearn, rdkit, etc.)
#      ⚠ Do NOT include pytorch here (conda-forge pulls CPU-only builds)
#   2. pip: PyTorch from pytorch.org/whl (GPU or CPU)
#   3. pip: PyTorch-Geometric (torch-geometric, torch-scatter, etc.)
#   4. pip: fairseq, einops (DeepDTAGen-specific deps)
#   5. Verify all imports
#
# Key differences from DrugBAN/GraphBAN:
#   - No DGL/dgllife needed (DeepDTAGen uses PyG, not DGL)
#   - No ESM-2/transformers needed (trains embeddings from scratch)
#   - Requires fairseq (for TransformerDecoderLayer)
#   - Requires einops (for Rearrange layer)

set -euo pipefail

ENV_NAME="deepdtagen"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

# ── Create environment ─────────────────────────────────────────────────────
if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Environment '${ENV_NAME}' already exists. Activating..."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -y --name "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

# ── Detect GPU/CUDA ──────────────────────────────────────────────────────
GPU_MODE=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
    CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    echo "[INFO] Detected NVIDIA GPU (CUDA ${CUDA_VER})."

    if [ "${CUDA_MAJOR}" -ge 12 ]; then
        PIP_CUDA="cu121"
    else
        PIP_CUDA="cu118"
    fi
    GPU_MODE=true
else
    echo "[INFO] No GPU detected. Installing CPU-compatible stack."
fi

# ── Step 1: Conda scientific deps (NO pytorch) ────────────────────────────
echo "[INFO] Installing scientific dependencies via conda..."
conda install -y \
    "numpy<2" \
    pandas \
    scikit-learn \
    scipy \
    tqdm \
    rdkit \
    networkx \
    -c conda-forge

# ── Step 2: PyTorch stack via pip (official index) ─────────────────────────
# LESSON LEARNED: conda's channel resolver unreliably picks CPU-only builds.
# pip with --index-url is deterministic and guaranteed compatible.
echo "[INFO] Cleaning any existing PyTorch packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [ -n "${SITE_PKGS}" ]; then
    for pkg_dir in torch torchvision torchaudio; do
        rm -rf "${SITE_PKGS}/${pkg_dir}" "${SITE_PKGS}/${pkg_dir}"*.dist-info 2>/dev/null || true
    done
fi

echo "[INFO] Installing PyTorch stack via pip..."
if [ "${GPU_MODE}" = true ]; then
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url "https://download.pytorch.org/whl/${PIP_CUDA}"
else
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu
fi

# ── Step 3: PyTorch-Geometric via pip (AFTER pytorch) ─────────────────────
# DeepDTAGen uses GCNConv from torch_geometric.nn
# LESSON LEARNED: torch-scatter/sparse need prebuilt wheels, not source builds.
echo "[INFO] Installing PyTorch-Geometric via pip..."
pip install torch-geometric

# Install PyG extensions from prebuilt wheels (avoids compilation errors)
TORCH_VER=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
if [ "${GPU_MODE}" = true ]; then
    PYG_WHEEL_TAG="torch-${TORCH_VER}+${PIP_CUDA}"
else
    PYG_WHEEL_TAG="torch-${TORCH_VER}+cpu"
fi
echo "[INFO] Installing PyG extensions from wheels: ${PYG_WHEEL_TAG}"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/${PYG_WHEEL_TAG}.html"

# ── Step 4: DeepDTAGen-specific deps ──────────────────────────────────────
# einops: provides Rearrange layer used in the model
echo "[INFO] Installing einops..."
pip install einops

# fairseq: DeepDTAGen uses only TransformerDecoderLayer, TransformerEncoderLayer,
# and FairseqIncrementalDecoder. Full fairseq has C++ build issues on modern
# setups, so we provide a local shim (fairseq/ directory) that reimplements
# these 3 classes using standard PyTorch nn.MultiheadAttention.
# The shim is architecture-compatible and produces identical outputs.
if [ -d "${SCRIPT_DIR}/fairseq" ] && [ -f "${SCRIPT_DIR}/fairseq/__init__.py" ]; then
    echo "[INFO] Local fairseq shim found. Skipping fairseq package install."
else
    echo "[WARN] No local fairseq shim. Attempting pip install (may fail on macOS)..."
    pip install "pip<24.1" && pip install "omegaconf>=2.0.5,<2.1" && \
    pip install hydra-core==1.0.7 antlr4-python3-runtime==4.8 && \
    pip install fairseq --no-deps || \
    echo "[ERROR] fairseq install failed. Create local shim or install manually."
fi

# ── Step 5: Extract data if needed ────────────────────────────────────────
DATA_DIR="${SCRIPT_DIR}/data"
if [ ! -f "${DATA_DIR}/kiba_train.csv" ]; then
    echo "[INFO] Extracting data from data.rar..."
    if command -v bsdtar >/dev/null 2>&1; then
        bsdtar xf "${SCRIPT_DIR}/data.rar" -C "${SCRIPT_DIR}/"
        echo "[INFO] Data extracted with bsdtar."
    elif command -v unrar >/dev/null 2>&1; then
        unrar x "${SCRIPT_DIR}/data.rar" "${SCRIPT_DIR}/"
        echo "[INFO] Data extracted with unrar."
    else
        echo "[WARN] No RAR extractor found. Please extract data.rar manually."
    fi
fi

# ── Step 6: Create required directories ───────────────────────────────────
mkdir -p "${SCRIPT_DIR}/saved_models"
mkdir -p "${SCRIPT_DIR}/Affinities"
mkdir -p "${SCRIPT_DIR}/logs"

# ── Verification ──────────────────────────────────────────────────────────
echo ""
echo "[INFO] Verifying installation..."
python3 - << 'PYEOF'
import sys
ok = True
checks = [
    ("torch",           "import torch; print(f'  torch {torch.__version__} | CUDA={torch.cuda.is_available()}')"),
    ("torch_geometric", "import torch_geometric; print(f'  torch_geometric {torch_geometric.__version__}')"),
    ("GCNConv",         "from torch_geometric.nn import GCNConv; print('  GCNConv OK')"),
    ("rdkit",           "from rdkit import Chem; print('  rdkit OK')"),
    ("fairseq",         "from fairseq.models import FairseqIncrementalDecoder; print('  fairseq OK')"),
    ("einops",          "from einops.layers.torch import Rearrange; print('  einops OK')"),
    ("networkx",        "import networkx; print(f'  networkx {networkx.__version__}')"),
    ("sklearn",         "import sklearn; print(f'  sklearn {sklearn.__version__}')"),
    ("scipy",           "import scipy; print(f'  scipy {scipy.__version__}')"),
]
for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}", file=sys.stderr)
        ok = False
if ok:
    print("\n  All checks passed.")
else:
    print("\n  Some checks failed — review errors above.", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "============================================"
echo " DeepDTAGen environment ready!"
echo " Activate : conda activate ${ENV_NAME}"
echo " Train    : python training.py 0  (davis)"
echo "          : python training.py 1  (kiba)"
echo "          : python training.py 2  (bindingdb)"
echo "============================================"
