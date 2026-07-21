#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CMADTI_ENV_NAME:-cmadti-cuda}"
PACKAGE_CACHE=""

cleanup_package_cache() {
  if [[ -n "${PACKAGE_CACHE}" && -d "${PACKAGE_CACHE}" ]]; then
    rm -rf -- "${PACKAGE_CACHE}"
  fi
}
trap cleanup_package_cache EXIT

FORCE=0
for argument in "$@"; do
  case "${argument}" in
    --force) FORCE=1 ;;
    *) echo "unknown argument: ${argument}" >&2; exit 2 ;;
  esac
done

if [[ ! -f "${ROOT_DIR}/CMA-DTI/dataloader.py" ]]; then
  echo "[setup] initializing pinned CMA-DTI submodule"
  # The upstream example datasets are Git-LFS objects and are not used by our
  # canonical wrapper.  Avoid downloading them on production hosts.
  GIT_LFS_SKIP_SMUDGE=1 git -C "${ROOT_DIR}" submodule update --init CMA-DTI
fi
if [[ ! -f "${ROOT_DIR}/CMA-DTI/dataloader.py" ]]; then
  echo "FATAL: CMA-DTI submodule is unavailable at ${ROOT_DIR}/CMA-DTI" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "[setup] removing incomplete/existing environment: ${ENV_NAME}"
    conda env remove -y -n "${ENV_NAME}"
  else
    echo "FATAL: conda environment '${ENV_NAME}' already exists." >&2
    echo "Re-run with --force to rebuild it cleanly:" >&2
    echo "  bash environments/install_cmadti_cuda.sh --force" >&2
    exit 1
  fi
fi

PACKAGE_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/cmadti-conda-pkgs.XXXXXX")"
export CONDA_PKGS_DIRS="${PACKAGE_CACHE}"
echo "[setup] using isolated Conda package cache: ${CONDA_PKGS_DIRS}"

echo "[1/6] creating conda base (including conda-owned networkx)"
conda env create -n "${ENV_NAME}" -f "${ROOT_DIR}/environments/cmadti-cuda.yml"
echo "[2/6] installing PyTorch CUDA wheels"
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.4.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
echo "[3/6] installing CUDA DGL without replacing conda dependencies"
conda install -y -n "${ENV_NAME}" -c dglteam/label/cu121 --no-update-deps dgl
echo "[4/6] installing CMA-DTI Python dependencies"
conda run -n "${ENV_NAME}" python -m pip install \
  dgllife==0.3.2 transformers==4.46.3 tokenizers sentencepiece safetensors
echo "[5/6] disabling unused GraphBolt ABI loader"
SITE_PACKAGES="$(conda run -n "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0])' | tail -1)"
GRAPHBOLT_INIT="${SITE_PACKAGES}/dgl/graphbolt/__init__.py"
if [[ -f "${GRAPHBOLT_INIT}" ]]; then
  # Conda may hard-link environment files to its package cache.  Writing to
  # GRAPHBOLT_INIT in place would then corrupt the cached DGL package.  Create
  # a new inode and atomically replace only the environment copy instead.
  GRAPHBOLT_REPLACEMENT="${GRAPHBOLT_INIT}.cmadti"
  echo "# graphbolt disabled: CMA-DTI does not use GraphBolt" > "${GRAPHBOLT_REPLACEMENT}"
  mv -f -- "${GRAPHBOLT_REPLACEMENT}" "${GRAPHBOLT_INIT}"
fi
echo "[6/6] verifying imports and a real DGL/CUDA forward-backward"
conda run -n "${ENV_NAME}" python -c \
  "import torch,dgl,dgllife,transformers,rdkit; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
conda run --no-capture-output -n "${ENV_NAME}" \
  python "${ROOT_DIR}/scripts/cmadti/smoke_model.py"
