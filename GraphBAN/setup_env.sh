#!/usr/bin/env bash
# Setup conda environment for GraphBAN baseline.
#
# Usage:
#   bash setup_env.sh
#
# Tested on: diamante-02 (RTX 4090, CUDA 12.4, Python 3.10)
#
# Installation order (CRITICAL — changing order breaks things):
#   1. conda: Python 3.10 + scientific deps (numpy, sklearn, rdkit, etc.)
#      ⚠ Do NOT include pytorch or dgl here (conda-forge pulls CPU-only builds)
#   2. pip: PyTorch 2.4.1 + torchvision + torchaudio from pytorch.org/whl/cu121
#      ⚠ conda pytorch from conda-forge installs CPU-only builds that break
#        torchvision::nms operator registration (RuntimeError)
#   3. pip: PyTorch-Geometric (torch-geometric, torch-scatter, etc.)
#   4. conda: DGL from dglteam/label/cu121 channel (--no-update-deps!)
#      ⚠ pip DGL wheels from data.dgl.ai return HTTP 403
#      ⚠ conda-forge DGL pulls its own pytorch 2.3.1 (overrides pip 2.4.1)
#      ⚠ Must pip uninstall dgl first to avoid ClobberError
#      ⚠ DGL graphbolt .so expects wrong PyTorch — patch __init__.py after install
#   5. pip: dgllife, fair-esm
#   6. Clone GraphBAN source + apply upstream patches
#   7. Verify all imports
#
# Common errors and fixes:
#   torchvision::nms does not exist → PyTorch was from conda-forge (CPU build)
#     Fix: pip install from pytorch.org/whl/cu121
#   DGL graphbolt libgraphbolt_pytorch_X.so not found → ABI mismatch
#     Fix: echo "# disabled" > dgl/graphbolt/__init__.py
#   GraphBAN models.py calls exit() → Colab-only import
#     Fix: patch_upstream.py removes the problematic import
#
set -euo pipefail

ENV_NAME="graphban"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Create environment ─────────────────────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Environment '${ENV_NAME}' already exists. Activating..."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -y --name "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ── Detect CUDA ───────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    echo "[INFO] Detected CUDA ${CUDA_VER}"

    if [ "${CUDA_MAJOR}" -ge 12 ]; then
        PYTORCH_CUDA="12.1"
        DGL_LABEL="cu121"
    else
        PYTORCH_CUDA="11.8"
        DGL_LABEL="cu118"
    fi
    GPU_MODE=true
else
    echo "[INFO] No GPU detected. Installing CPU-only versions."
    GPU_MODE=false
fi

# ── Step 1: Conda scientific deps (NO pytorch, NO dgl) ────────────────────
# Install these first via conda. pytorch and dgl come via pip to avoid
# conda-forge pulling in incompatible CPU-only pytorch builds.
echo "[INFO] Installing scientific dependencies via conda..."
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
    safetensors \
    huggingface_hub \
    setuptools \
    packaging \
    -c conda-forge

# ── Step 2: PyTorch stack via pip (official index) ─────────────────────────
# LESSON LEARNED: conda's channel resolver unreliably picks CPU-only builds
# from conda-forge even with --override-channels, causing torchvision::nms
# ABI failures. pip with --index-url is deterministic and guaranteed compatible.
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
        --index-url "https://download.pytorch.org/whl/cu${PYTORCH_CUDA//./}"
else
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu
fi

# ── Step 3: PyTorch-Geometric via pip (AFTER pytorch) ─────────────────────
echo "[INFO] Installing PyTorch-Geometric via pip..."
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv

# ── Step 4: DGL + dgllife (AFTER pytorch, via conda from dglteam) ──────────
# pip wheels from data.dgl.ai return 403. conda from dglteam channel is reliable.
# conda DGL won't override pip-installed PyTorch.
echo "[INFO] Installing DGL via conda (dglteam channel)..."
pip uninstall -y dgl 2>/dev/null || true
if [ "${GPU_MODE}" = true ]; then
    conda install -y dgl -c "dglteam/label/cu${PYTORCH_CUDA//./}" --no-update-deps
else
    conda install -y dgl -c dglteam --no-update-deps
fi

echo "[INFO] Installing dgllife via pip..."
pip install dgllife

# ── Step 4b: Patch DGL graphbolt to prevent exit(1) crash ─────────────────
echo "[INFO] Patching DGL graphbolt __init__.py to prevent CUDA version crash..."
DGL_GB=$(find "$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    -path "*/dgl/graphbolt/__init__.py" 2>/dev/null | head -1)
if [ -n "${DGL_GB}" ] && [ -f "${DGL_GB}" ]; then
    echo "# graphbolt disabled by setup_env.sh — GraphBAN does not use it" > "${DGL_GB}"
    echo "[INFO] Patched: ${DGL_GB}"
else
    echo "[WARN] Could not locate dgl/graphbolt/__init__.py — skipping patch."
fi

# ── Step 5: ESM (fair-esm) — not available via conda ─────────────────────
echo "[INFO] Installing fair-esm (pip only — not on conda-forge)..."
pip install fair-esm


# ── Step 6: Clone GraphBAN source ────────────────────────────────────────
echo "[INFO] Setting up GraphBAN source code..."
GRAPHBAN_SRC="${SCRIPT_DIR}/src"

if [ -d "${GRAPHBAN_SRC}/.git" ]; then
    echo "[INFO] GraphBAN source already cloned at ${GRAPHBAN_SRC}"
else
    git clone https://github.com/HamidHadipour/GraphBAN.git "${GRAPHBAN_SRC}"
    echo "[INFO] GraphBAN cloned to ${GRAPHBAN_SRC}"
fi

# ── Step 6b: Patch upstream source for known runtime issues ───────────────
# Applies idempotent fixes:
# - Colab-only exit() in models.py
# - Mutable graph reuse bug in inductive_mode/dataloader.py (negative virtual nodes)
python3 "${SCRIPT_DIR}/patch_upstream.py" --src "${GRAPHBAN_SRC}"

# ── Verification ──────────────────────────────────────────────────────────
echo ""
echo "[INFO] Verifying installation..."
python3 - << 'PYEOF'
import sys
ok = True
checks = [
    ("torch",         "import torch; assert torch.cuda.is_available() or True; print(f'  torch {torch.__version__} | CUDA={torch.cuda.is_available()}')"),
    ("torchvision",   "import torchvision; print(f'  torchvision {torchvision.__version__}')"),
    ("dgl",           "import dgl; print(f'  dgl {dgl.__version__}')"),
    ("dgllife",       "from dgllife.utils import smiles_to_bigraph; print('  dgllife OK')"),
    ("rdkit",         "from rdkit import Chem; print('  rdkit OK')"),
    ("transformers",  "from transformers import AutoTokenizer; print('  transformers OK')"),
    ("safetensors",   "import safetensors; print('  safetensors OK')"),
    ("sklearn",       "import sklearn; print(f'  sklearn {sklearn.__version__}')"),
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
echo " GraphBAN environment ready!"
echo " Activate : conda activate ${ENV_NAME}"
echo " Run      : python run_baseline.py --dataset non_human"
echo "============================================"
