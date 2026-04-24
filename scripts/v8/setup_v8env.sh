#!/usr/bin/env bash
# =============================================================================
# Conda env setup for DT-Kinase v8 POC.
#
# Creates TWO isolated envs to avoid dependency conflicts between ADMET-AI
# (which pulls chemprop + lightning + accelerate + torchmetrics and forces
# torch>=2.11 + cu13) and the v8 training pipeline (which needs torch
# 2.5.1+cu124 for the diamante-02 driver 12.4.0):
#
#   1. v8env    — training + ChemBERTa/BioBERT precompute + Pfam/Taxonomy/
#                 ClassyFire/UniProt async fetch. torch 2.5.1+cu124 (GPU).
#   2. admetenv — ADMET-AI precompute only. Runs on CPU (disabled via
#                 CUDA_VISIBLE_DEVICES=""). Whatever torch admet_ai wants.
#
# The orchestrator run_v8_benchmark.sh activates the appropriate env per
# stage.
#
# Usage:
#     bash scripts/v8/setup_v8env.sh
#
# Env overrides:
#     V8ENV_NAME    default v8env
#     ADMETENV_NAME default admetenv
#     PYTHON_VER    default 3.11
#     TORCH_VER     default 2.5.1
# =============================================================================
set -uo pipefail

V8ENV_NAME="${V8ENV_NAME:-v8env}"
ADMETENV_NAME="${ADMETENV_NAME:-admetenv}"
PYTHON_VER="${PYTHON_VER:-3.11}"
TORCH_VER="${TORCH_VER:-2.5.1}"
TORCHVISION_VER="${TORCHVISION_VER:-0.20.1}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

if ! command -v conda &>/dev/null; then
    echo "[fatal] conda not on PATH" >&2; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# v8env: training + encoders (GPU torch 2.5.1+cu124)
# ---------------------------------------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "${V8ENV_NAME}"; then
    echo "== creating conda env '${V8ENV_NAME}' (python=${PYTHON_VER}) =="
    conda create -n "${V8ENV_NAME}" "python=${PYTHON_VER}" -y
else
    echo "== conda env '${V8ENV_NAME}' already exists, reusing =="
fi

conda activate "${V8ENV_NAME}"
echo "== active: $(which python)  $(python --version) =="
pip install --upgrade pip

echo "== [v8env] installing torch==${TORCH_VER}+cu124 FIRST =="
# torchvision intentionally NOT installed — v8 is text-only. When present,
# transformers eagerly imports torchvision.transforms (via image_utils.py)
# and crashes if the torchvision C++ extension was built for a different
# torch. Absence → transformers detects no-vision and skips the import.
pip install --index-url "${CUDA_INDEX}" "torch==${TORCH_VER}"
# Remove torchvision if a prior install pulled it in (idempotent)
pip uninstall -y torchvision 2>/dev/null || true

echo "== [v8env] installing minimal non-torch-pulling deps =="
# Only packages that don't transitively drag torch in. transformers is ok
# because we pin tokenizers + huggingface_hub explicitly. DGL, lightning,
# chemprop, accelerate, torchmetrics are DELIBERATELY excluded — they would
# force a torch bump.
pip install \
    "numpy<2" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "tqdm>=4.65" \
    "pyyaml>=6" \
    "rdkit>=2023.3" \
    "biopython>=1.80" \
    "pyhmmer>=0.10" \
    "aiohttp>=3.9" \
    "safetensors" \
    "sentencepiece" \
    "tokenizers>=0.15" \
    "huggingface_hub>=0.23" \
    "filelock" "packaging" "requests" "regex" \
    "scipy"

echo "== [v8env] installing transformers with --no-deps =="
pip install --no-deps "transformers>=4.30,<5"

# Verify torch is still 2.5.1
ACTUAL_TORCH="$(python -c 'import torch; print(torch.__version__)')"
if [[ "${ACTUAL_TORCH}" != "${TORCH_VER}"* ]]; then
    echo "[fatal] torch was clobbered: ${ACTUAL_TORCH} (expected ${TORCH_VER}+cu124)" >&2
    exit 1
fi

echo "== [v8env] verify CUDA =="
python - <<PY
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.zeros(1).cuda(); _ = x + 1
    print("kernel launch: OK")
else:
    print("WARN: CUDA not available — check driver vs wheel cu124")
PY

# ---------------------------------------------------------------------------
# admetenv: isolated ADMET-AI precompute (CPU only, any torch)
# ---------------------------------------------------------------------------
conda deactivate
if ! conda env list | awk '{print $1}' | grep -qx "${ADMETENV_NAME}"; then
    echo
    echo "== creating conda env '${ADMETENV_NAME}' (python=${PYTHON_VER}) =="
    conda create -n "${ADMETENV_NAME}" "python=${PYTHON_VER}" -y
else
    echo "== conda env '${ADMETENV_NAME}' already exists, reusing =="
fi

conda activate "${ADMETENV_NAME}"
echo "== [admetenv] active: $(which python) =="
pip install --upgrade pip

echo "== [admetenv] installing admet_ai (will bring lightning/chemprop/etc) =="
# Isolated env — safe to let admet_ai pick whatever torch it wants.
# At runtime, precompute_admet_ligand.py forces CUDA_VISIBLE_DEVICES=""
# so torch runs on CPU regardless of its CUDA build.
pip install "admet_ai" "numpy<2" "pandas>=2.0" "tqdm>=4.65" "pyyaml>=6" "rdkit>=2023.3"

echo
echo "[done] Two envs ready:"
echo "    conda activate ${V8ENV_NAME}   # training + ChemBERTa/BioBERT/Pfam/etc"
echo "    conda activate ${ADMETENV_NAME}   # ADMET-AI precompute only"
echo
echo "Run POC (orchestrator switches envs automatically):"
echo "    V8ENV=${V8ENV_NAME} ADMETENV=${ADMETENV_NAME} bash scripts/v8/run_v8_benchmark.sh"
