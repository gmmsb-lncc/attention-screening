#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CHEMGLAM_ENV_NAME:-chemglam-cuda}"

conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environments/chemglam-cuda.yml"
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n "${ENV_NAME}" python -m pip install \
  -r "${ROOT_DIR}/environments/requirements-chemglam.txt"
conda run -n "${ENV_NAME}" python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

