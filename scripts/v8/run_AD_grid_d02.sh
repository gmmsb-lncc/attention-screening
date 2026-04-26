#!/usr/bin/env bash
# =============================================================================
# Grid sweep over Direção A (layers_lig) × Direção D (lr_mult_lig).
#
# Stack: v7+F LEGACY default (lição 17) + per-side asym adapter (lição 19).
# 6 cells: lr_mult_lig ∈ {3, 5, 8} × layers_lig ∈ {2, 3}, attn_heads_lig=12 fixed.
#
# Each cell: 3-seed (42, 123, 456). Total: 6 cells × 3 seeds = 18 runs.
# Estimated time on diamante-02 (cuDNN OFF, RTX 4090): ~5-7h total.
#
# All outputs land in results/grid_AD/<lr>x_<layers>L/test/...
# After completion, run aggregate.py to summarise.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-non_human}"
SEEDS="${SEEDS:-42 123 456}"
LR_MULTS="${LR_MULTS:-3 5 8}"
LAYERS_LIST="${LAYERS_LIST:-2 3}"
ATTN_HEADS_LIG="${ATTN_HEADS_LIG:-12}"
LR_MULT_PROT="${LR_MULT_PROT:-2.0}"

GRID_ROOT="results/grid_AD"
mkdir -p "${GRID_ROOT}"

LOG_FILE="${GRID_ROOT}/grid_AD_$(date +%Y%m%d_%H%M).log"
echo "[grid] log → ${LOG_FILE}"

echo "=============================================================="
echo "A+D grid sweep — d02"
echo "  corpus:       ${CORPUS}"
echo "  seeds:        ${SEEDS}"
echo "  lr_mult_lig:  ${LR_MULTS}"
echo "  layers_lig:   ${LAYERS_LIST}"
echo "  attn_heads:   ${ATTN_HEADS_LIG}"
echo "  lr_mult_prot: ${LR_MULT_PROT}"
echo "  output root:  ${GRID_ROOT}"
echo "=============================================================="

CELL=0
TOTAL=$(echo "${LR_MULTS} ${LAYERS_LIST}" | wc -w)
TOTAL_CELLS=$(( $(echo ${LR_MULTS} | wc -w) * $(echo ${LAYERS_LIST} | wc -w) ))

for lr in ${LR_MULTS}; do
    for nl in ${LAYERS_LIST}; do
        CELL=$((CELL + 1))
        TAG="lr${lr}_L${nl}"
        OUT_ROOT="${GRID_ROOT}/${TAG}"
        echo ""
        echo "============================================================"
        echo "[${CELL}/${TOTAL_CELLS}] cell=${TAG}  (lr_mult_lig=${lr}, layers_lig=${nl})"
        echo "  output: ${OUT_ROOT}"
        echo "============================================================"

        # Skip if already complete (idempotent re-runs).
        if [ -f "${OUT_ROOT}/test/benchmark_comparison.json" ]; then
            echo "[${CELL}/${TOTAL_CELLS}] SKIP — already complete"
            continue
        fi

        # Move any stale partial output aside.
        if [ -d "${OUT_ROOT}" ]; then
            mv "${OUT_ROOT}" "${OUT_ROOT}_partial_$(date +%H%M)" 2>/dev/null || true
        fi

        # Per-cell yaml so output_root reflects the cell tag.
        TMP_CONFIG="$(mktemp --suffix=.yaml)"
        sed "s|output_root:.*|output_root: ${OUT_ROOT}|" \
            configs/v7_plus_F.yaml > "${TMP_CONFIG}"

        SEEDS="${SEEDS}" V7_CONFIG="${TMP_CONFIG}" CORPUS="${CORPUS}" \
            BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS_LIG="${nl}" \
            BENCHMARK_LEVEL4CNN_ADAPTER_ATTN_HEADS_LIG="${ATTN_HEADS_LIG}" \
            BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_PROT="${LR_MULT_PROT}" \
            BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_LIG="${lr}" \
            bash scripts/v8/run_v7_yaml.sh 2>&1 | tee -a "${LOG_FILE}"

        rm -f "${TMP_CONFIG}"
    done
done

echo ""
echo "=============================================================="
echo "GRID DONE. Aggregate:"
echo "  python3 scripts/v8/aggregate_AD_grid.py --root ${GRID_ROOT}"
echo "=============================================================="
