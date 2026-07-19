#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REVISION="7b12d946c181a37f6012b9dc3b002275de070314"
PATCH_FILE="${ROOT_DIR}/scripts/chemglam/patches/pin_molformer_revision.patch"

if grep -q "${REVISION}" "${ROOT_DIR}/ChemGLaM/chemglam/model/chemglam.py"; then
  echo "ChemGLaM MoLFormer revision already pinned: ${REVISION}"
  exit 0
fi

git -C "${ROOT_DIR}/ChemGLaM" apply --unidiff-zero "${PATCH_FILE}"
echo "Pinned MoLFormer remote code revision: ${REVISION}"
