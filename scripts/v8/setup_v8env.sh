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

# --- Install order matters --------------------------------------------------
# admet_ai / accelerate / transformers ship transitive torch requirements
# that can silently pull a newer torch (e.g. 2.6+) incompatible with
# the CUDA 12.4 driver. Strategy:
#   1. Install all non-torch runtime deps first (with --no-deps where they
#      declare a bare "torch" requirement, so their torch pin is ignored).
#   2. Install torch + torchvision LAST via the cu124 index.
#   3. Verify the active torch is ${TORCH_VER}+cu124.
# DGL removed — not needed by v8 (only GraphBAN baseline uses it).
pip install --upgrade pip

echo "== installing non-torch runtime deps =="
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
    "sentencepiece"

echo "== installing transformers (without its torch transitive) =="
pip install --no-deps "transformers>=4.30,<5"
# transformers runtime deps that we actually need, pinned explicitly:
pip install "tokenizers>=0.15" "huggingface_hub>=0.23" "filelock>=3.12" "packaging" "requests" "regex"

echo "== installing admet_ai (without its torch transitive) =="
pip install --no-deps "admet_ai"
# admet_ai runtime deps (beyond what we already have):
pip install --no-deps "lightning>=2.0" "chemprop>=2.0"
# Lightning + Chemprop also try to pull torch — use --no-deps and install
# their smaller dep surface manually:
pip install "lightning-utilities" "fsspec" "torchmetrics<1.5"

echo "== installing torch==${TORCH_VER}+cu124 LAST (force reinstall) =="
pip install --force-reinstall --index-url "${CUDA_INDEX}" \
    "torch==${TORCH_VER}" \
    "torchvision==${TORCHVISION_VER}"

# Optional: Hugging Face token (users can export HF_TOKEN to avoid rate limits)
echo
echo "== verify torch version + CUDA =="
python - <<PY
import torch
expected = "${TORCH_VER}"
actual = torch.__version__
ok_ver = actual.startswith(expected)
print(f"torch: {actual}  {'OK' if ok_ver else 'MISMATCH (expected '+expected+')'}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cc:", torch.cuda.get_device_capability(0))
    # Attempt actual CUDA kernel launch
    x = torch.zeros(1).cuda(); _ = x + 1
    print("kernel launch: OK")
else:
    print("WARN: torch built without CUDA OR driver mismatch.")
if not ok_ver:
    print("FAIL — torch was clobbered by a transitive dependency. Re-run setup.")
    import sys; sys.exit(1)
PY

echo
echo "[done] v8env ready. Activate with:"
echo "    conda activate ${ENV_NAME}"
echo
echo "Run v8 POC:"
echo "    V7_ENV=${ENV_NAME} bash scripts/v8/run_v8_benchmark.sh"
