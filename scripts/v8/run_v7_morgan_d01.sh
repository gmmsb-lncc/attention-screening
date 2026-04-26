#!/usr/bin/env bash
# d01 — v7+F + Morgan FP topological auxiliary (NH 2-seed quick test)
# Pre-requisito: precompute Morgan via scripts/v8/precompute_ligand_morgan.py
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MORGAN_DIR="${MORGAN_DIR:-results/morgan_fp}"

# Pre-compute Morgan se não existir
if [ ! -d "${MORGAN_DIR}" ] || [ ! -f "${MORGAN_DIR}/_summary.json" ]; then
    echo "[stage 0] Precomputing Morgan FP →${MORGAN_DIR}"
    python3 scripts/v8/precompute_ligand_morgan.py \
        --splits-dir scaffolds_splits/output \
        --out-dir "${MORGAN_DIR}" \
        || { echo "[fatal] precompute Morgan falhou"; exit 1; }
fi

export BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_DIR="${REPO_ROOT}/${MORGAN_DIR}"
export BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_BITS="${BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_BITS:-1024}"
export BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_PROJ="${BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_PROJ:-32}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

export SEEDS="${SEEDS:-42 123}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_plus_F_morgan.yaml}"
export CORPUS="${CORPUS:-non_human}"

echo "=============================================================="
echo "v7+F + Morgan FP (d01)"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}"
echo "  seeds:         ${SEEDS}"
echo "  Morgan dir:    ${BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_DIR}"
echo "  Morgan bits:   ${BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_BITS}"
echo "  Morgan proj:   ${BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_PROJ}"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
