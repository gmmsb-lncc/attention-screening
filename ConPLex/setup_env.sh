#!/usr/bin/env bash
# Setup conda environment for ConPLex (PNAS 2023) baseline.
#
# Usage:
#   bash setup_env.sh
#
# Tested on:
#   - MacBook Pro M1 (macOS, CPU-only, Python 3.10) ✓
#   - diamante-02 (RTX 4090, CUDA 12.4, Python 3.10) — target GPU machine
#
# Installation order (CRITICAL — changing order breaks things):
#   1. conda: Python 3.10 + scientific deps
#   2. pip: PyTorch from pytorch.org/whl
#   3. pip: DGL + graphbolt patch
#   4. pip: ConPLex-specific deps (fair-esm, dscript, mol2vec, deepchem, etc.)
#   5. pip: version pins to fix known compat issues
#   6. Verify all imports
#
# Bugs found and fixed during testing:
#   - DGL graphbolt .so missing for PyTorch 2.4.1 → patch __init__.py
#   - pandas 2.x breaks rdkit PandasTools → pin pandas<2.1
#   - setuptools 72+ removes pkg_resources → pin setuptools<71
#   - deepchem, mol2vec not in requirements.txt but required by molecule.py
#   - pytorch-lightning required by data.py
#   - molecule.py had hardcoded MIT CSAIL AFS path → fixed to relative path
#
# ConPLex architecture:
#   - Protein encoders: ProtBert (default), ESM-1b, ProSE, D-SCRIPT, ProtT5
#   - Drug encoders: Morgan Fingerprints (default), Mol2Vec, MolR (GNN)
#   - Prediction: contrastive co-embedding with cosine distance
#
# Reference: Singh et al. (2023) PNAS 120(24) doi:10.1073/pnas.2220778120

set -euo pipefail

ENV_NAME="conplex"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " ConPLex Environment Setup"
echo " Dir: ${SCRIPT_DIR}"
echo "============================================"

# ── Verify conda is available ─────────────────────────────────────────────
CONDA_SH=""
for p in \
  "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "/usr/local/anaconda3/etc/profile.d/conda.sh"; do
  if [ -f "$p" ]; then CONDA_SH="$p"; break; fi
done

if [ -n "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
elif ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# ── Create or reuse conda environment ─────────────────────────────────────
if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Environment '${ENV_NAME}' already exists."
else
  echo "[INFO] Creating environment '${ENV_NAME}' (Python 3.10)..."
  conda create -y --name "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

# ── Detect GPU ────────────────────────────────────────────────────────────
GPU_MODE=false
PIP_CUDA="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_MODE=true
  CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
  echo "[INFO] Detected NVIDIA GPU${CUDA_VER:+ (CUDA ${CUDA_VER})}."
  CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
  if [ "${CUDA_MAJOR}" -ge 12 ]; then
    PIP_CUDA="cu121"
  else
    PIP_CUDA="cu118"
  fi
else
  echo "[INFO] No GPU detected. Installing CPU-compatible stack."
fi

# ── Step 1: Conda scientific deps (NO pytorch, NO dgl) ────────────────────
echo ""
echo "[INFO] Step 1/6: Installing scientific dependencies via conda..."
conda install -y \
  "numpy<2" \
  "pandas>=1.5,<2.1" \
  scipy \
  scikit-learn \
  matplotlib \
  tqdm \
  rdkit \
  pyyaml \
  h5py \
  -c conda-forge

# ── Step 2: PyTorch stack via pip (official index) ─────────────────────────
echo ""
echo "[INFO] Step 2/6: Installing PyTorch stack via pip..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)
if [ -n "${SITE_PKGS}" ]; then
    for pkg_dir in torch torchvision torchaudio; do
        rm -rf "${SITE_PKGS}/${pkg_dir}" "${SITE_PKGS}/${pkg_dir}"*.dist-info 2>/dev/null || true
    done
fi

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url "https://download.pytorch.org/whl/${PIP_CUDA}"

# ── Step 3: DGL via pip + graphbolt patch ──────────────────────────────────
echo ""
echo "[INFO] Step 3/6: Installing DGL via pip..."
pip uninstall -y dgl 2>/dev/null || true

# Try conda first (more reliable on Linux GPU), fallback to pip
if [ "${GPU_MODE}" = true ]; then
    conda install -y dgl -c "dglteam/label/${PIP_CUDA}" --no-update-deps 2>/dev/null \
      || pip install dgl -f https://data.dgl.ai/wheels/repo.html
else
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
fi

# Patch DGL graphbolt to prevent exit(1) crash
echo "[INFO] Patching DGL graphbolt..."
DGL_GB=$(python3 -c "
import site, os
sp = site.getsitepackages()[0]
p = os.path.join(sp, 'dgl', 'graphbolt', '__init__.py')
if os.path.exists(p): print(p)
" 2>/dev/null)
if [ -n "${DGL_GB}" ] && [ -f "${DGL_GB}" ]; then
    echo "# graphbolt disabled by setup_env.sh" > "${DGL_GB}"
    echo "[INFO] Patched: ${DGL_GB}"
else
    echo "[WARN] Could not locate dgl/graphbolt/__init__.py — skipping patch."
fi

# ── Step 4: ConPLex-specific pip dependencies ─────────────────────────────
echo ""
echo "[INFO] Step 4/6: Installing ConPLex-specific dependencies via pip..."

pip install \
  fair-esm \
  dscript \
  omegaconf \
  wandb \
  pysmiles \
  deepchem \
  gensim \
  pytorch-lightning \
  torchmetrics \
  transformers \
  huggingface_hub \
  "mol2vec @ git+https://github.com/samoturk/mol2vec"

# pytdc (TDC benchmark) — installed separately because its pandas>=2.1 req
# conflicts with rdkit, but it works fine with pandas 2.0.x at runtime
pip install pytdc --no-deps 2>/dev/null || pip install pytdc

# ── Step 5: Pin versions to fix known compatibility issues ────────────────
echo ""
echo "[INFO] Step 5/6: Applying version pins for compatibility..."

# setuptools 72+ removed pkg_resources, needed by TDC
pip install "setuptools>=65,<71" 2>/dev/null || true

# Ensure pandas stays compatible with rdkit PandasTools
pip install "pandas>=1.5,<2.1" 2>/dev/null || true

# ── Step 6: Verify installation ──────────────────────────────────────────
echo ""
echo "[INFO] Step 6/6: Verifying installation..."
python3 << 'PYEOF'
import sys, warnings
warnings.filterwarnings("ignore")
ok = True
checks = [
    ("torch",        "import torch; print(f'  torch {torch.__version__} | CUDA={torch.cuda.is_available()}')"),
    ("dgl",          "import dgl; print(f'  dgl {dgl.__version__}')"),
    ("esm",          "import esm; print(f'  fair-esm OK')"),
    ("dscript",      "from dscript.language_model import lm_embed; print('  dscript OK')"),
    ("transformers", "from transformers import AutoTokenizer, AutoModel; print('  transformers OK')"),
    ("rdkit",        "from rdkit import Chem; from rdkit.Chem import AllChem; print('  rdkit OK')"),
    ("omegaconf",    "from omegaconf import OmegaConf; print('  omegaconf OK')"),
    ("torchmetrics", "import torchmetrics; print(f'  torchmetrics {torchmetrics.__version__}')"),
    ("pysmiles",     "import pysmiles; print('  pysmiles OK')"),
    ("deepchem",     "import deepchem as dc; print(f'  deepchem {dc.__version__}')"),
    ("mol2vec",      "from mol2vec.features import mol2alt_sentence; print('  mol2vec OK')"),
    ("wandb",        "import wandb; print(f'  wandb {wandb.__version__}')"),
    ("sklearn",      "import sklearn; print(f'  sklearn {sklearn.__version__}')"),
    ("gensim",       "import gensim; print(f'  gensim {gensim.__version__}')"),
    ("lightning",    "import pytorch_lightning; print(f'  pytorch-lightning {pytorch_lightning.__version__}')"),
    ("pkg_resources","import pkg_resources; print('  pkg_resources OK')"),
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

# ── Print summary ────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " ConPLex environment ready"
echo " Activate : conda activate ${ENV_NAME}"
echo " Train    : cd ${SCRIPT_DIR} && python train_DTI.py --exp-id test --config configs/default_config.yaml --wandb-save False"
echo " Datasets : ${SCRIPT_DIR}/dataset/{DAVIS,BIOSNAP,BindingDB,DUDe}"
echo " Weights  : ${SCRIPT_DIR}/models/"
echo "============================================"
