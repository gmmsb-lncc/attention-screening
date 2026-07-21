#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CMADTI_ENV_NAME:-cmadti-cuda}"
conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environments/cmadti-cuda.yml"
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.4.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
conda install -y -n "${ENV_NAME}" -c dglteam/label/cu121 --no-update-deps dgl
conda run -n "${ENV_NAME}" python -m pip install \
  dgllife==0.3.2 transformers==4.46.3 tokenizers sentencepiece safetensors
SITE_PACKAGES="$(conda run -n "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0])' | tail -1)"
GRAPHBOLT_INIT="${SITE_PACKAGES}/dgl/graphbolt/__init__.py"
if [[ -f "${GRAPHBOLT_INIT}" ]]; then
  echo "# graphbolt disabled: CMA-DTI does not use GraphBolt" > "${GRAPHBOLT_INIT}"
fi
conda run -n "${ENV_NAME}" python -c \
  "import torch,dgl,dgllife,transformers,rdkit; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
conda run --no-capture-output -n "${ENV_NAME}" \
  python "${ROOT_DIR}/scripts/cmadti/smoke_model.py"
