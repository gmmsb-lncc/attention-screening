#!/usr/bin/env bash
# run_drugban_universal.sh
#
# Run DrugBAN baseline (as published) on all three kinase datasets
# using the UNIVERSAL scaffold split for fair comparison with DT-Kinase.
#
# Usage:
#   conda activate drugban
#   bash run_drugban_universal.sh [OPTIONS]
#
# Options (passed through to run_baseline.py):
#   --max-epoch N    Override epochs (default: 100)
#   --batch-size N   Override batch size (default: 64)
#   --seeds N...     Override seeds (default: 42 123 456 789 1024)
#   --no-da          Disable CDAN domain adaptation
#   --num-workers N  DataLoader workers
#
# Environment variables:
#   DRUGBAN_OUTPUT_ROOT  Base output dir (default: DrugBAN/results/universal)
#   DRUGBAN_DATASETS     Space-separated datasets to run (default: "non_human human all")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y_%m_%d)"
ENV_NAME="drugban"

# Datasets to run (override with DRUGBAN_DATASETS env var)
DATASETS="${DRUGBAN_DATASETS:-non_human human all}"

# Output root (override with DRUGBAN_OUTPUT_ROOT env var)
OUTPUT_ROOT="${DRUGBAN_OUTPUT_ROOT:-${SCRIPT_DIR}/results/universal_${TIMESTAMP}}"

# ── Check conda environment ───────────────────────────────────────────────────
if ! conda info --envs 2>/dev/null | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "ERROR: conda environment '${ENV_NAME}' not found."
    echo "Run first: bash setup_env.sh"
    exit 1
fi

# ── Print banner ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "  DrugBAN Baseline — Universal Scaffold Split Evaluation"
echo "  Architecture: as published (Bai et al., Nat. Mach. Intell. 2023)"
echo "  Datasets:     ${DATASETS}"
echo "  Output root:  ${OUTPUT_ROOT}"
echo "================================================================"
echo ""

# ── Prepare data (universal split) ───────────────────────────────────────────
echo "[1/2] Preparing datasets from universal scaffold split..."
conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/prepare_data.py" --all --output-dir "${SCRIPT_DIR}"
echo ""

# ── Train + evaluate per dataset ─────────────────────────────────────────────
echo "[2/2] Training DrugBAN on each dataset..."
echo ""

FAILED_DATASETS=()

for DATASET in ${DATASETS}; do
    echo "────────────────────────────────────────────────────────────────"
    echo "  Dataset: ${DATASET}"
    echo "────────────────────────────────────────────────────────────────"

    OUT_DIR="${OUTPUT_ROOT}/${DATASET}"
    mkdir -p "${OUT_DIR}"

    if conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_baseline.py" \
        --dataset "${DATASET}" \
        --output-dir "${OUT_DIR}" \
        "$@"; then
        echo "  ✓ ${DATASET} completed → ${OUT_DIR}/drugban_results.json"
    else
        echo "  ✗ ${DATASET} FAILED"
        FAILED_DATASETS+=("${DATASET}")
    fi
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  DrugBAN Universal Evaluation — Complete"
echo "================================================================"
echo ""
echo "Results saved under: ${OUTPUT_ROOT}"
echo ""
for DATASET in ${DATASETS}; do
    RESULT="${OUTPUT_ROOT}/${DATASET}/drugban_results.json"
    if [[ -f "${RESULT}" ]]; then
        # Extract test MCC mean from JSON (requires python)
        MCC=$(conda run -n "${ENV_NAME}" python -c "
import json, sys
try:
    with open('${RESULT}') as f:
        d = json.load(f)
    v = d['aggregate']['test']['mcc']
    print(f\"  {v['mean']:.4f} ± {v['std']:.4f}\")
except Exception as e:
    print(f'  (could not parse: {e})')
" 2>/dev/null || echo "  (parse error)")
        echo "  ${DATASET:10} Test MCC: ${MCC}"
    fi
done
echo ""

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo "WARNING: The following datasets failed: ${FAILED_DATASETS[*]}"
    exit 1
fi
