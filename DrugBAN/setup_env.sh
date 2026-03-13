#!/usr/bin/env bash
# Setup conda environment for DrugBAN baseline.
#
# Usage:
#   bash setup_env.sh
#
# Notes:
# - Creates/updates conda env named "drugban"
# - Installs core dependencies with conda
# - Clones upstream DrugBAN source into DrugBAN/src if missing

set -euo pipefail

ENV_NAME="drugban"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
UPSTREAM_URL="https://github.com/peizhenbai/DrugBAN.git"
# Optional: pin upstream source for reproducibility.
# Set to a commit hash or git tag, e.g. "a1b2c3d" or "v1.0.0".
UPSTREAM_REF=""

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Environment '${ENV_NAME}' already exists."
else
  echo "[INFO] Creating environment '${ENV_NAME}' (Python 3.11)..."
  conda create -y --name "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

GPU_MODE=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_MODE=true
  CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
  echo "[INFO] Detected NVIDIA GPU${CUDA_VER:+ (CUDA ${CUDA_VER})}."
else
  echo "[INFO] No GPU detected. Installing CPU-compatible stack."
fi

echo "[INFO] Installing PyTorch stack..."
if [ "${GPU_MODE}" = true ]; then
  conda install -y \
    pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    pytorch-cuda=12.1 \
    --override-channels -c pytorch -c nvidia -c conda-forge
else
  conda install -y \
    pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly \
    --override-channels -c pytorch -c conda-forge
fi

echo "[INFO] Installing scientific dependencies..."
conda install -y \
  "numpy<2" \
  pandas \
  scikit-learn \
  tqdm \
  rdkit \
  prettytable \
  yacs \
  torchmetrics \
  pyyaml \
  pyarrow \
  transformers \
  safetensors \
  huggingface_hub \
  dgl \
  dgllife \
  -c conda-forge

if [ -d "${SRC_DIR}/.git" ]; then
  echo "[INFO] DrugBAN source already present at ${SRC_DIR}"
  git -C "${SRC_DIR}" fetch --all --tags --prune
else
  if [ -d "${SRC_DIR}" ] && [ "$(find "${SRC_DIR}" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then
    echo "[WARN] ${SRC_DIR} exists and is not empty, but is not a git clone."
    echo "[WARN] Skipping clone to avoid overwriting files."
  else
    rm -rf "${SRC_DIR}"
    echo "[INFO] Cloning DrugBAN source from ${UPSTREAM_URL} ..."
    git clone "${UPSTREAM_URL}" "${SRC_DIR}"
  fi
fi

if [ -n "${UPSTREAM_REF}" ]; then
  echo "[INFO] Checking out pinned upstream ref: ${UPSTREAM_REF}"
  git -C "${SRC_DIR}" checkout "${UPSTREAM_REF}"
  git -C "${SRC_DIR}" submodule update --init --recursive
else
  echo "[INFO] No UPSTREAM_REF pin set; using repository default branch HEAD."
fi

echo "[INFO] Running import checks..."
python - << 'PYEOF'
import sys
checks = [
    ("torch", "import torch; print('torch', torch.__version__)"),
    ("torchvision", "import torchvision; print('torchvision', torchvision.__version__)"),
    ("dgl", "import dgl; print('dgl', dgl.__version__)"),
    ("dgllife", "from dgllife.utils import smiles_to_bigraph; print('dgllife OK')"),
    ("rdkit", "from rdkit import Chem; print('rdkit OK')"),
    ("transformers", "from transformers import AutoTokenizer; print('transformers OK')"),
    ("sklearn", "import sklearn; print('sklearn', sklearn.__version__)"),
]
ok = True
for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}", file=sys.stderr)
if not ok:
    sys.exit(1)
PYEOF

echo ""
echo "============================================"
echo " DrugBAN environment ready"
echo " Activate : conda activate ${ENV_NAME}"
echo " Prepare  : python prepare_data.py --dataset non_human"
echo "============================================"
