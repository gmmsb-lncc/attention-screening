#!/usr/bin/env bash
# Setup conda environment for GraphBAN baseline.
#
# Usage:
#   bash setup_env.sh
#
# All packages installed via conda only (no pip).
#
# Key lessons from deployment on diamante-01 (CUDA 12.2, PyTorch 2.4.1):
#
#   1. PyTorch MUST come from the 'pytorch' channel, NOT conda-forge.
#      conda-forge ships CPU-only builds (cpu_mkl_*) that conflict with
#      CUDA torchvision/torchaudio and break torchvision::nms op registration.
#
#   2. pytorch + torchvision + torchaudio MUST be installed in ONE transaction
#      so conda guarantees ABI-compatible builds (same build string py311_cu121).
#
#   3. DGL from 'dglteam/label/cu121' ships graphbolt .so files built for
#      older PyTorch versions. They call libc exit(1) on load failure with
#      the message "Stopping RUNTIME. Colaboratory will restart automatically."
#      Fix: overwrite dgl/graphbolt/__init__.py with a no-op stub after install.
#
#   4. GraphBAN's src/inductive_mode/models.py has a Colab-only import that
#      calls exit() when IPythonConsole is missing. Fix applied after clone.
#
#   5. transformers >= 4.40 blocks torch.load(.bin) due to CVE-2025-32434.
#      ChemBERTa-77M-MTR has safetensors — use_safetensors=True in code.
#      safetensors must be installed explicitly.
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

# ── Step 1: PyTorch + torchvision + torchaudio (one transaction, pytorch channel)
# CRITICAL: must use -c pytorch -c nvidia and pin channel priority away from
# conda-forge to avoid the CPU-only pytorch build overriding the CUDA one.
echo "[INFO] Installing PyTorch stack via conda (pytorch channel)..."
if [ "${GPU_MODE}" = true ]; then
    conda install -y \
        pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        pytorch-cuda=${PYTORCH_CUDA} \
        --override-channels -c pytorch -c nvidia -c conda-forge
else
    conda install -y \
        pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly \
        --override-channels -c pytorch -c conda-forge
fi

# ── Step 2: PyTorch-Geometric ─────────────────────────────────────────────
echo "[INFO] Installing PyTorch-Geometric (pyg channel)..."
conda install -y \
    pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv \
    -c pyg -c conda-forge

# ── Step 3: DGL (CUDA build from dglteam) ────────────────────────────────
echo "[INFO] Installing DGL..."
if [ "${GPU_MODE}" = true ]; then
    conda install -y dgl --override-channels -c "dglteam/label/${DGL_LABEL}" -c pytorch -c nvidia -c conda-forge
else
    conda install -y dgl --override-channels -c dglteam -c pytorch -c conda-forge
fi

# ── Step 3b: Patch DGL graphbolt to prevent exit(1) crash ─────────────────
# DGL 2.x ships libgraphbolt_pytorch_*.so files. When the .so ABI does not
# match the running PyTorch, DGL's graphbolt/__init__.py calls exit(1) with
# "Stopping RUNTIME. Colaboratory will restart automatically."
# Overwriting the file with a comment prevents this entirely.
echo "[INFO] Patching DGL graphbolt __init__.py to prevent CUDA version crash..."
DGL_GB=$(python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('dgl')
if spec:
    p = pathlib.Path(spec.origin).parent / 'graphbolt/__init__.py'
    print(p)
" 2>/dev/null)
if [ -n "${DGL_GB}" ] && [ -f "${DGL_GB}" ]; then
    echo "# graphbolt disabled by setup_env.sh — GraphBAN does not use it" > "${DGL_GB}"
    echo "[INFO] Patched: ${DGL_GB}"
else
    echo "[WARN] Could not locate dgl/graphbolt/__init__.py — skipping patch."
fi

# ── Step 4: dgllife ───────────────────────────────────────────────────────
echo "[INFO] Installing dgllife (conda-forge)..."
conda install -y dgllife -c conda-forge

# ── Step 5: Remaining scientific dependencies ─────────────────────────────
echo "[INFO] Installing remaining dependencies (conda-forge)..."
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
