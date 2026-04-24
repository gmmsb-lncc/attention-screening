#!/usr/bin/env bash
# =============================================================================
# Conda env setup for DT-Kinase v8 POC.
# Targets diamante-02 (NVIDIA driver 12.4.0 → torch 2.5.x+cu124).
#
# Creates a fresh `v8env` conda env, installs a CUDA-12.4-compatible torch,
# and all v8 runtime dependencies. Idempotent — safe to re-run.
#
# Usage:
#     bash scripts/v8/setup_v8env.sh           # creates env "v8env"
#     ENV_NAME=v8 bash scripts/v8/setup_v8env.sh
# =============================================================================
set -uo pipefail

ENV_NAME="${ENV_NAME:-v8env}"
PYTHON_VER="${PYTHON_VER:-3.11}"
TORCH_VER="${TORCH_VER:-2.5.1}"
TORCHVISION_VER="${TORCHVISION_VER:-0.20.1}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

# Source conda.sh so `conda activate` works inside this script
if ! command -v conda &>/dev/null; then
    echo "[fatal] conda not on PATH" >&2; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create env if missing
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "== creating conda env '${ENV_NAME}' (python=${PYTHON_VER}) =="
    conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y
else
    echo "== conda env '${ENV_NAME}' already exists, reusing =="
fi

conda activate "${ENV_NAME}"
echo "== active: $(which python)  $(python --version) =="

# --- Install torch + torchvision from CUDA 12.4 wheel index -----------------
echo "== installing torch==${TORCH_VER}+cu124 =="
pip install --upgrade pip
pip install --index-url "${CUDA_INDEX}" \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}"

# --- Runtime deps for v8 ----------------------------------------------------
# numpy<2 keeps compatibility with older packages that don't yet support np2
pip install \
    "numpy<2" \
    "transformers>=4.30,<5" \
    "admet_ai" \
    "pyhmmer>=0.10" \
    "aiohttp>=3.9" \
    "biopython>=1.80" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "tqdm>=4.65" \
    "pyyaml>=6" \
    "rdkit>=2023.3" \
    "dgl" \
    "safetensors" \
    "accelerate" \
    "sentencepiece"

# Optional: Hugging Face token (users can export HF_TOKEN to avoid rate limits)
echo
echo "== verify torch CUDA =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cc:", torch.cuda.get_device_capability(0))
else:
    print("WARN: CUDA not initialized from torch — verify with:")
    print("  python -c 'import torch; torch.zeros(1).cuda()'")
PY

echo
echo "[done] v8env ready. Activate with:"
echo "    conda activate ${ENV_NAME}"
echo
echo "Run v8 POC:"
echo "    V7_ENV=${ENV_NAME} bash scripts/v8/run_v8_benchmark.sh"
