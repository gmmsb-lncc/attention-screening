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
LEVELS="0"
PATIENCE="30"
OUTPUT_BASE="results/non_human_10_03_2026_v4"

echo "============================================================"
echo " Phase 1/2 — TRAIN (fit=80% train, eval=10% val)"
echo "============================================================"
python semantic_screening_models.py \
    --dataset "${DATASET}" \
    --embedding "${EMBEDDING}" \
    --levels ${LEVELS} \
    --train \
    --patience ${PATIENCE} \
    --output_dir "${OUTPUT_BASE}"

echo ""
echo "============================================================"
echo " Phase 2/2 — TEST (fit=10% val, eval=10% test)"
echo "============================================================"
python semantic_screening_models.py \
    --dataset "${DATASET}" \
    --embedding "${EMBEDDING}" \
    --levels ${LEVELS} \
    --test \
    --patience ${PATIENCE} \
    --output_dir "${OUTPUT_BASE}/"

echo ""
echo "============================================================"
echo " Done. Results saved in:"
echo "   ${OUTPUT_BASE}/train/"
echo "   ${OUTPUT_BASE}/test/"
echo "============================================================"
