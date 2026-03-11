#!/usr/bin/env bash
# ===========================================================================
# run_benchmark.sh — Execute train then test benchmark phases
#
# Usage:
#   bash run_benchmark.sh
#
# Results are saved under results/output_dir/{train,test}/
# ===========================================================================
set -euo pipefail

DATASET="human"
EMBEDDING="8M"
LEVELS="0"                          # 0 = classical ML (1a, 1b, 1c, 3)
OUTPUT_BASE="results/output_dir"

echo "============================================================"
echo " Phase 1/2 — TRAIN (fit=80% train, eval=10% val)"
echo "============================================================"
python semantic_screening_models.py \
    --dataset "${DATASET}" \
    --embedding "${EMBEDDING}" \
    --levels ${LEVELS} \
    --train \
    --output_dir "${OUTPUT_BASE}/train"

echo ""
echo "============================================================"
echo " Phase 2/2 — TEST (fit=10% val, eval=10% test)"
echo "============================================================"
python semantic_screening_models.py \
    --dataset "${DATASET}" \
    --embedding "${EMBEDDING}" \
    --levels ${LEVELS} \
    --test \
    --output_dir "${OUTPUT_BASE}/test"

echo ""
echo "============================================================"
echo " Done. Results saved in:"
echo "   ${OUTPUT_BASE}/train/"
echo "   ${OUTPUT_BASE}/test/"
echo "============================================================"
