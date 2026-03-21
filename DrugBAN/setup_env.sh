#!/usr/bin/env bash
# Setup conda environment for DrugBAN baseline.
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
#   3. conda: DGL from dglteam/label/cu121 channel (--no-update-deps!)
#      ⚠ pip DGL wheels from data.dgl.ai return HTTP 403
#      ⚠ conda-forge DGL pulls its own pytorch 2.3.1 (overrides pip 2.4.1)
#      ⚠ Must pip uninstall dgl first to avoid ClobberError
#      ⚠ DGL graphbolt .so expects wrong PyTorch — patch __init__.py after install
#   4. pip: dgllife (safe after DGL + pytorch are installed)
#   5. Clone upstream DrugBAN source into src/
#   6. Verify all imports
#
# Common errors and fixes:
#   torchvision::nms does not exist → PyTorch was installed from conda-forge
#     Fix: pip install from pytorch.org/whl/cu121
#   No module 'torchdata.datapipes' → torchdata >= 0.8 removed datapipes
#     Fix: not needed when DGL comes from conda dglteam channel
#   DGL graphbolt libgraphbolt_pytorch_X.so not found → ABI mismatch
#     Fix: echo "# disabled" > dgl/graphbolt/__init__.py
#   'configs' is not a package → our configs/ dir shadows upstream configs.py
#     Fix: importlib.util.spec_from_file_location in run_baseline.py

set -euo pipefail

ENV_NAME="drugban"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
UPSTREAM_URL="https://github.com/peizhenbai/DrugBAN.git"
UPSTREAM_REF=""

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Environment '${ENV_NAME}' already exists."
else
  echo "[INFO] Creating environment '${ENV_NAME}' (Python 3.10)..."
  conda create -y --name "${ENV_NAME}" python=3.10
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

# ── Step 1: Conda scientific deps (NO pytorch, NO dgl) ────────────────────
# Install these first via conda. pytorch and dgl come via pip to avoid
# conda-forge pulling in incompatible CPU-only pytorch builds.
echo "[INFO] Installing scientific dependencies via conda..."
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
  -c conda-forge

# ── Step 2: PyTorch stack via pip (official index) ─────────────────────────
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
    CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    if [ "${CUDA_MAJOR}" -ge 12 ]; then
        PIP_CUDA="cu121"
    else
        PIP_CUDA="cu118"
    fi
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url "https://download.pytorch.org/whl/${PIP_CUDA}"
else
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cpu
fi

# ── Step 3: DGL + dgllife (AFTER pytorch, via conda from dglteam) ──────────
# pip wheels from data.dgl.ai return 403. conda from dglteam channel is reliable.
# conda DGL won't override pip-installed PyTorch.
echo "[INFO] Installing DGL via conda (dglteam channel)..."
pip uninstall -y dgl 2>/dev/null || true
if [ "${GPU_MODE}" = true ]; then
    conda install -y dgl -c "dglteam/label/${PIP_CUDA}" --no-update-deps
else
    conda install -y dgl -c dglteam --no-update-deps
fi

echo "[INFO] Installing dgllife via pip..."
pip install dgllife

# ── Step 4: Patch DGL graphbolt to prevent exit(1) crash ──────────────────
# DGL can't be imported before patching, so we find the path with sed/find.
echo "[INFO] Patching DGL graphbolt..."
DGL_GB=$(find "$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    -path "*/dgl/graphbolt/__init__.py" 2>/dev/null | head -1)
if [ -n "${DGL_GB}" ] && [ -f "${DGL_GB}" ]; then
    echo "# graphbolt disabled by setup_env.sh" > "${DGL_GB}"
    echo "[INFO] Patched: ${DGL_GB}"
else
    echo "[WARN] Could not locate dgl/graphbolt/__init__.py — skipping patch."
fi

# ── Step 5: Clone DrugBAN source ──────────────────────────────────────────
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

# ── Step 5b: Patch max_drug_nodes default in dataloader.py ────────────
# Upstream hardcodes max_drug_nodes=290 but human dataset has molecules
# up to 310 atoms. Read MAX_NODES from config and patch the default.
MAX_NODES=$(grep 'MAX_NODES' configs/kinase.yaml 2>/dev/null | head -1 | awk -F: '{print $2}' | awk '{print $1}')
if [ -n "${MAX_NODES}" ] && [ "${MAX_NODES}" != "290" ] && [ -f "${SRC_DIR}/dataloader.py" ]; then
    sed -i "s/max_drug_nodes=290/max_drug_nodes=${MAX_NODES}/g" "${SRC_DIR}/dataloader.py"
    echo "[INFO] Patched dataloader.py: max_drug_nodes=290 → ${MAX_NODES}"
fi

# ── Step 6: Verify installation ──────────────────────────────────────────
echo ""
echo "[INFO] Verifying installation..."
python3 - << 'PYEOF'
import sys
ok = True
checks = [
    ("torch",        "import torch; print(f'  torch {torch.__version__} | CUDA={torch.cuda.is_available()}')"),
    ("torchvision",  "import torchvision; print(f'  torchvision {torchvision.__version__}')"),
    ("dgl",          "import dgl; print(f'  dgl {dgl.__version__}')"),
    ("dgllife",      "from dgllife.utils import smiles_to_bigraph; print('  dgllife OK')"),
    ("rdkit",        "from rdkit import Chem; print('  rdkit OK')"),
    ("transformers", "from transformers import AutoTokenizer; print('  transformers OK')"),
    ("sklearn",      "import sklearn; print(f'  sklearn {sklearn.__version__}')"),
]
for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        ok = False
        print(f"  [FAIL] {name}: {e}", file=sys.stderr)
if ok:
    print("\n  All checks passed.")
else:
    print("\n  Some checks failed — review errors above.", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "============================================"
echo " DrugBAN environment ready"
echo " Activate : conda activate ${ENV_NAME}"
echo " Run      : python run_baseline.py --dataset non_human"
echo "============================================"
