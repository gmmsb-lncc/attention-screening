#!/usr/bin/env bash
# =============================================================================
# Unified "baseline" conda env: hosts all 4 committee models in one env
# =============================================================================
#
# Creates a single conda environment named "baseline" that holds the union
# of dependencies for DT-Kinase, DrugBAN, GraphBAN, and ConPLex inference.
# This eliminates the need for 3 separate per-baseline envs and lets
# committee.py run all 4 forwards in the same Python process (or simple
# subprocess invocations) without `conda run -n {env}` switching.
#
# Cost of unification: ~3-4 GB env, ~10 min install. The pinned versions
# are the intersection of what works for all four models. Tested matrix:
#
#   Python 3.10                (DrugBAN/GraphBAN tested; DT-K + ConPLex agnostic)
#   PyTorch 2.4.1 + cu121      (DrugBAN/GraphBAN tested; DT-K + ConPLex compat)
#   DGL 2.x cu121              (DrugBAN/GraphBAN required; DT-K + ConPLex unused)
#   transformers 4.39.3        (MoLFormer custom code requires < 4.40 for ONNX)
#   RDKit                      (all 4 use)
#   fair-esm                   (GraphBAN ESM-1b; DT-K uses local llm/ESM clone)
#   pytorch-lightning + torchmetrics + omegaconf  (ConPLex)
#   dscript                    (ConPLex protein featurizer ProtBert wrapper)
#
# Critical install order (DO NOT REARRANGE — see DrugBAN/setup_env.sh notes):
#   1. conda Python + pure-conda scientific deps (no PyTorch/DGL!)
#   2. pip PyTorch from pytorch.org/whl/cu121 (NOT conda-forge — CPU build)
#   3. conda DGL from dglteam/label/cu121 (--no-update-deps! NOT pip wheels)
#   4. pip dgllife (after DGL + PyTorch are settled)
#   5. pip transformers (pinned < 4.40 for MoLFormer ONNX dependency)
#   6. pip ConPLex extras (dscript, lightning, etc.)
#   7. patch DGL graphbolt __init__.py (ABI mismatch between DGL build and PyTorch)
#
# Usage:
#   bash scripts/inference/setup_baseline_env.sh
#   bash scripts/inference/setup_baseline_env.sh --force    # remove + recreate
#
# After installation:
#   conda activate baseline
#   python attention_screening.py "..." --single-env baseline
# =============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-baseline}"
PYTHON_VERSION="3.10"
TORCH_VERSION="2.4.1"
TORCH_CUDA="cu121"
TRANSFORMERS_VERSION="4.39.3"

FORCE=""
USE_CPU=0
for arg in "$@"; do
    case "${arg}" in
        --force) FORCE="--force" ;;
        --cpu)   USE_CPU=1 ;;
        *) echo "unknown flag: ${arg}" >&2; exit 1 ;;
    esac
done

# Auto-detect CUDA when --cpu not requested
if [[ "${USE_CPU}" -eq 0 ]]; then
    if [[ "$(uname -s)" == "Darwin" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[detect] no NVIDIA GPU detected → falling back to CPU build"
        USE_CPU=1
    fi
fi

echo "=============================================================="
echo " Setup unified 'baseline' conda env (4-model committee)"
echo "  env name : ${ENV_NAME}"
echo "  Python   : ${PYTHON_VERSION}"
echo "  PyTorch  : ${TORCH_VERSION}$([[ ${USE_CPU} -eq 0 ]] && echo "+${TORCH_CUDA}" || echo " (CPU)")"
echo "  DGL      : $([[ ${USE_CPU} -eq 0 ]] && echo "2.x+${TORCH_CUDA}" || echo "CPU build")"
echo "  transf.  : ${TRANSFORMERS_VERSION}"
echo "=============================================================="

# Detect conda
if ! command -v conda >/dev/null 2>&1; then
    echo "FATAL: conda not on PATH. Install miniconda/anaconda first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Recreate if --force
if conda info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${FORCE}" == "--force" ]]; then
        echo "[reset] removing existing ${ENV_NAME} env"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "[skip] ${ENV_NAME} env already exists. Use --force to recreate."
        echo "       activate via: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Stage 1: conda base — Python + pure scientific deps (NO pytorch/dgl)
# -----------------------------------------------------------------------------
echo "[1/7] conda create + scientific base packages"
conda create -n "${ENV_NAME}" -y \
    -c conda-forge \
    "python=${PYTHON_VERSION}" \
    pip wheel setuptools \
    numpy "<2" \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    rdkit \
    networkx \
    h5py \
    pyyaml \
    tqdm \
    yacs

conda activate "${ENV_NAME}"

# -----------------------------------------------------------------------------
# Stage 2: pip PyTorch from pytorch.org wheels (CUDA build, NOT conda-forge)
# -----------------------------------------------------------------------------
if [[ "${USE_CPU}" -eq 1 ]]; then
    echo "[2/7] pip PyTorch ${TORCH_VERSION} (CPU build from PyPI)"
    pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision" "torchaudio"
else
    echo "[2/7] pip PyTorch ${TORCH_VERSION}+${TORCH_CUDA}"
    pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision" "torchaudio" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

# -----------------------------------------------------------------------------
# Stage 3: DGL — conda dglteam channel (CUDA) or pip (CPU)
# -----------------------------------------------------------------------------
# guard against pre-existing pip dgl
pip uninstall -y dgl 2>/dev/null || true
if [[ "${USE_CPU}" -eq 1 ]]; then
    echo "[3/7] pip DGL (CPU build)"
    pip install --no-cache-dir dgl \
        || echo "  WARN: DGL CPU install failed; DrugBAN+GraphBAN unavailable" >&2
else
    echo "[3/7] conda DGL from dglteam/label/${TORCH_CUDA}"
    conda install -y -n "${ENV_NAME}" \
        -c "dglteam/label/${TORCH_CUDA}" \
        --no-update-deps \
        "dgl"
fi

# -----------------------------------------------------------------------------
# Stage 4: pip dgllife + GraphBAN PyG dependencies
# -----------------------------------------------------------------------------
echo "[4/7] pip dgllife + torch-geometric stack (GraphBAN GCN/GIN)"
pip install --no-cache-dir \
    "dgllife" \
    "torch-geometric" \
    fair-esm

# -----------------------------------------------------------------------------
# Stage 5: pip transformers pinned + ConPLex extras
# -----------------------------------------------------------------------------
echo "[5/7] pip transformers ${TRANSFORMERS_VERSION} + ConPLex deps"
pip install --no-cache-dir \
    "transformers==${TRANSFORMERS_VERSION}" \
    "tokenizers<0.20" \
    "huggingface_hub" \
    "accelerate" \
    pytorch-lightning \
    torchmetrics \
    omegaconf \
    pytdc \
    sentencepiece \
    biopython

# ConPLex needs `dscript` for ProtBert ESM-style featurization.
# dscript's deepchem/mol2vec extras pin old TF — install with --no-deps.
pip install --no-cache-dir --no-deps dscript || \
    echo "  WARN: dscript install failed; ConPLex featurization may break"

# -----------------------------------------------------------------------------
# Stage 6: pytest (for committee tests) + pyarrow (optional parquet support)
# -----------------------------------------------------------------------------
echo "[6/7] pip dev tools (pytest)"
pip install --no-cache-dir pytest

# -----------------------------------------------------------------------------
# Stage 7: patch DGL graphbolt (ABI mismatch with PyTorch wheels)
# -----------------------------------------------------------------------------
echo "[7/7] patch DGL graphbolt __init__.py (ABI workaround)"
GRAPHBOLT_INIT="$(python -c 'import dgl, os; print(os.path.join(os.path.dirname(dgl.__file__), "graphbolt", "__init__.py"))')"
if [[ -f "${GRAPHBOLT_INIT}" ]]; then
    if grep -q "graphbolt disabled" "${GRAPHBOLT_INIT}" 2>/dev/null; then
        echo "  graphbolt already patched"
    else
        echo "# graphbolt disabled — ABI mismatch with PyTorch ${TORCH_VERSION}" > "${GRAPHBOLT_INIT}"
        echo "  patched ${GRAPHBOLT_INIT}"
    fi
else
    echo "  graphbolt not found at ${GRAPHBOLT_INIT}; skipping patch"
fi

# -----------------------------------------------------------------------------
# Verification: import all 4 model namespaces
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
import transformers, dgl, rdkit
print(f"  DGL       : {dgl.__version__}")
print(f"  transformers: {transformers.__version__}")
print(f"  rdkit     : {rdkit.__version__}")

n_fail = sum(1 for s in results.values() if s != "OK")
sys.exit(1 if n_fail > 0 else 0)
PYEOF

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=============================================================="
    echo " SUCCESS — env '${ENV_NAME}' ready for 4-model committee"
    echo "=============================================================="
    echo "Activate:    conda activate ${ENV_NAME}"
    echo "Run demo:    python attention_screening.py 'CC(=O)Oc1ccccc1C(=O)O' --single-env"
else
    echo ""
    echo "=============================================================="
    echo " PARTIAL FAILURE — some imports failed (see above)"
    echo "=============================================================="
    echo "The committee will work for the models whose imports succeeded."
    echo "To investigate, manually:"
    echo "  conda activate ${ENV_NAME}"
    echo "  python -c 'import <failing_module>'"
fi
