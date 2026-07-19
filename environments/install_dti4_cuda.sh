#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${DTI4_ENV_NAME:-dti4-cuda}"

conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environments/dti4-cuda.yml"
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n "${ENV_NAME}" python -m pip install \
  -r "${ROOT_DIR}/environments/requirements-dti4-cuda.txt"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/environments/verify_dti4.py" \
  --require-cuda --require-topology
