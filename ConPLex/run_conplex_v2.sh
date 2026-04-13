#!/usr/bin/env bash
# run_conplex_v2.sh — Run ConPLex v2 (val-calibrated threshold) for all reps
#
# Usage:  bash run_conplex_v2.sh
# Assumes: running from ConPLex/ directory on the GPU machine

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_DIR="./results_v2"

echo "═══════════════════════════════════════════════════════════════"
echo " ConPLex v2 — MCC-optimal threshold from validation"
echo " Protocol: inference(val) → sweep(τ*) → apply(test)"
echo " Device: cuda:${DEVICE}"
echo "═══════════════════════════════════════════════════════════════"

for dataset in non_human human; do
    echo ""
    echo "━━━ Dataset: ${dataset} ━━━"
    for rep in 0 1 2; do
        exp_id="conplex_v2_${dataset}_rep${rep}"
        ckpt="best_models/trained_${dataset}_rep${rep}/trained_${dataset}_rep${rep}_best_model.pt"
        data="dataset/kinase_${dataset}"
        
        if [ ! -f "$ckpt" ]; then
            echo "  ⚠ Checkpoint not found: $ckpt — skipping"
            continue
        fi
        
        echo ""
        echo "  ▶ ${exp_id}"
        python eval_conplex_v2.py \
            --checkpoint "$ckpt" \
            --data-dir "$data" \
            --exp-id "$exp_id" \
            --device "$DEVICE" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR" \
            --n-thresholds 2000
    done
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " All done. Results in: ${OUTPUT_DIR}/"
echo "═══════════════════════════════════════════════════════════════"

# ── Aggregate summary ──────────────────────────────────────────────────
echo ""
echo "SUMMARY (test MCC: v1 → v2):"
for dataset in non_human human; do
    echo "  --- ${dataset} ---"
    for rep in 0 1 2; do
        result_file="${OUTPUT_DIR}/conplex_v2_${dataset}_rep${rep}/results_v2.json"
        if [ -f "$result_file" ]; then
            v1_mcc=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['test']['at_fixed_0.5']['MCC']:.4f}\")")
            v2_mcc=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['test']['at_calibrated']['MCC']:.4f}\")")
            tau=$(python3 -c "import json; d=json.load(open('$result_file')); print(f\"{d['optimal_threshold']:.4f}\")")
            echo "    rep${rep}: MCC ${v1_mcc} → ${v2_mcc} (τ*=${tau})"
        fi
    done
done
