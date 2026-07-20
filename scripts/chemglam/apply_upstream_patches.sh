#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION="7b12d946c181a37f6012b9dc3b002275de070314"
MOLFORMER_PATCH="${ROOT_DIR}/scripts/chemglam/patches/pin_molformer_revision.patch"
PREDICTION_PATCH="${ROOT_DIR}/scripts/chemglam/patches/memory_efficient_prediction.patch"

if grep -q "${REVISION}" "${ROOT_DIR}/ChemGLaM/chemglam/model/chemglam.py"; then
  echo "ChemGLaM MoLFormer revision already pinned: ${REVISION}"
else
  git -C "${ROOT_DIR}/ChemGLaM" apply --unidiff-zero "${MOLFORMER_PATCH}"
  echo "Pinned MoLFormer remote code revision: ${REVISION}"
fi

if grep -q 'temporary_path = prediction_path.with_suffix' "${ROOT_DIR}/ChemGLaM/predict.py"; then
  echo "ChemGLaM memory-efficient prediction patch already applied"
else
  git -C "${ROOT_DIR}/ChemGLaM" apply --unidiff-zero "${PREDICTION_PATCH}"
  echo "Applied ChemGLaM memory-efficient prediction patch"
fi
