#!/usr/bin/env bash
# =============================================================================
# Unified "baseline" Python venv — pip/venv equivalent to setup_baseline_env.sh
# =============================================================================
#
# Creates a single Python venv hosting all 4 committee models (DT-Kinase,
# DrugBAN, GraphBAN, ConPLex). Use this if you prefer venv over conda.
#
# Trade-offs vs conda baseline env:
#   + No conda dependency
#   + Lighter than conda (smaller disk footprint)
#   - DGL CUDA wheels need a dedicated index URL (workaround included)
#   - PyTorch CUDA detection less robust than conda's CUDA toolkit handling
#
# Tested on:
#   Linux + CUDA 12.1   (full feature, GPU forwards work)
#   macOS arm64 + CPU   (full feature, slower forwards but functional)
#
# Usage:
#   bash scripts/inference/setup_baseline_venv.sh                # auto-detect CUDA
#   bash scripts/inference/setup_baseline_venv.sh --cpu          # force CPU build
#   bash scripts/inference/setup_baseline_venv.sh --force        # remove + recreate
#   VENV_DIR=/custom/path bash scripts/inference/setup_baseline_venv.sh
#
# After install:
#   source env_baseline/bin/activate
#   python attention_screening.py "..." --single-env baseline
#
# Note: `--single-env baseline` in committee.py uses `conda run -n baseline`,
# so when running under venv the orchestrator instead inherits the active
# Python via PYBIN env var:
#
#   PYBIN=$(which python) python attention_screening.py "..."
#
# (committee.py defaults to `conda run -n {env}`; see TODO in committee.py
#  to support venv-style direct invocation as well).
# =============================================================================

set -eu
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO}/env_baseline}"

TORCH_VERSION="2.4.1"
TORCH_CUDA="cu121"

# Parse flags
FORCE=0
USE_CPU=0
for arg in "$@"; do
    case "${arg}" in
        --force) FORCE=1 ;;
        --cpu)   USE_CPU=1 ;;
        *) echo "unknown flag: ${arg}" >&2; exit 1 ;;
    esac
done

# Auto-detect CUDA (Linux + nvidia-smi present)
if [[ "${USE_CPU}" -eq 0 ]]; then
    if [[ "$(uname -s)" == "Darwin" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[detect] no NVIDIA GPU detected → falling back to CPU build"
        USE_CPU=1
    fi
fi

echo "=============================================================="
echo " Setup unified 'baseline' Python venv (4-model committee)"
echo "  venv dir : ${VENV_DIR}"
echo "  PyTorch  : ${TORCH_VERSION}$([[ ${USE_CPU} -eq 0 ]] && echo "+${TORCH_CUDA}" || echo " (CPU)")"
echo "=============================================================="

# Reset venv if --force
if [[ -d "${VENV_DIR}" ]]; then
    if [[ "${FORCE}" -eq 1 ]]; then
        echo "[reset] removing existing ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
    else
        echo "[skip] venv already exists at ${VENV_DIR}. Use --force to recreate."
        echo "       activate via: source ${VENV_DIR}/bin/activate"
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Stage 1: create venv + bootstrap pip
# -----------------------------------------------------------------------------
echo "[1/6] creating venv"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools >/dev/null

# -----------------------------------------------------------------------------
# Stage 2: PyTorch (CUDA from pytorch.org wheels; CPU from PyPI)
# -----------------------------------------------------------------------------
echo "[2/6] installing PyTorch"
if [[ "${USE_CPU}" -eq 1 ]]; then
    pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" torchvision torchaudio
else
    pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

# -----------------------------------------------------------------------------
# Stage 3: DGL (CUDA from dgl-team wheels; CPU from PyPI)
# -----------------------------------------------------------------------------
echo "[3/6] installing DGL"
if [[ "${USE_CPU}" -eq 1 ]]; then
    pip install --no-cache-dir dgl \
        || { echo "WARN: DGL install failed; DrugBAN+GraphBAN will be unavailable" >&2; }
else
    pip install --no-cache-dir dgl \
        -f "https://data.dgl.ai/wheels/torch-2.4/${TORCH_CUDA}/repo.html" \
        || { echo "WARN: DGL CUDA wheel install failed; trying CPU fallback" >&2;
             pip install --no-cache-dir dgl; }
fi

# -----------------------------------------------------------------------------
# Stage 4: rest of the requirements (rdkit, transformers pinned, etc.)
# -----------------------------------------------------------------------------
echo "[4/6] installing remaining requirements (requirements-baseline.txt)"
pip install --no-cache-dir -r "${REPO}/requirements-baseline.txt"

# -----------------------------------------------------------------------------
# Stage 5: dscript with --no-deps (its TF/sklearn extras conflict)
# -----------------------------------------------------------------------------
echo "[5/6] installing dscript (--no-deps for ConPLex featurizer)"
pip install --no-cache-dir --no-deps dscript \
    || echo "  WARN: dscript install failed; ConPLex featurization may break"

# -----------------------------------------------------------------------------
# Stage 6: patch DGL graphbolt + verify imports
# -----------------------------------------------------------------------------
echo "[6/6] patching DGL graphbolt + verifying"
python <<'PYEOF'
import os, pathlib
try:
    import dgl
    p = pathlib.Path(os.path.dirname(dgl.__file__))/"graphbolt/__init__.py"
    if p.exists() and "graphbolt disabled" not in p.read_text():
        p.write_text("# graphbolt disabled — ABI mismatch with PyTorch wheels\n")
        print(f"  patched {p}")
except ImportError:
    print("  DGL not installed; skipping graphbolt patch")
PYEOF

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo " Verification — importing all 4 model dependencies"
echo "=============================================================="
python <<'PYEOF'
import sys
results = {}
for name, imports in [
    ("PyTorch+CUDA",     ["torch"]),
    ("DGL",              ["dgl"]),
    ("dgllife",          ["dgllife"]),
    ("torch-geometric",  ["torch_geometric"]),
    ("transformers",     ["transformers"]),
    ("RDKit",            ["rdkit"]),
    ("fair-esm",         ["esm"]),
    ("pytorch-lightning",["pytorch_lightning"]),
    ("omegaconf",        ["omegaconf"]),
    ("dscript (ConPLex)",["dscript"]),
]:
    try:
        for imp in imports:
            __import__(imp)
        results[name] = "OK"
    except ImportError as e:
        results[name] = f"FAIL: {e}"

import torch
gpu_status = (f"{torch.cuda.device_count()} GPU(s)" if torch.cuda.is_available()
              else "CPU only")

print()
for name, status in results.items():
    mark = "✓" if status == "OK" else "✗"
    print(f"  {mark} {name:25s} {status}")
print()
print(f"  GPU       : {gpu_status}")
print(f"  PyTorch   : {torch.__version__}")
try:
    import dgl, transformers, rdkit
    print(f"  DGL       : {dgl.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  rdkit     : {rdkit.__version__}")
except ImportError:
    pass

n_fail = sum(1 for s in results.values() if s != "OK")
sys.exit(1 if n_fail > 0 else 0)
PYEOF

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=============================================================="
    echo " SUCCESS — venv ready at ${VENV_DIR}"
    echo "=============================================================="
    echo "Activate:  source ${VENV_DIR}/bin/activate"
    echo "Demo:      python attention_screening.py 'CC(=O)Oc1ccccc1C(=O)O'"
else
    echo ""
    echo "=============================================================="
    echo " PARTIAL FAILURE — some imports failed (see above)"
    echo "=============================================================="
fi
