#!/usr/bin/env bash
# =============================================================================
# #7 — Cross-species transfer evaluation on v7
#
# Trains v7 on one corpus and evaluates zero-shot (no re-training) on the
# other corpus's test partition. This is the direct test of Objective 5
# (multi-species extensibility) that the thesis currently claims but does
# not empirically support.
#
# Two directions:
#   (a) H -> NH: train on Human scaffold split, evaluate on Non-Human test
#   (b) NH -> H: train on Non-Human scaffold split, evaluate on Human test
#
# Reports: MCC, AUROC, F1, precision, recall x 5 seeds for each direction.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="${OUT_ROOT:-results/thesis_followups/cross_species}"
SEEDS=(${SEEDS:-42 123 456 789 1024})

run_direction() {
    local train_ds="$1"
    local eval_ds="$2"
    local label="${train_ds}_to_${eval_ds}"
    local out_dir="${OUT_ROOT}/${label}"
    mkdir -p "${out_dir}"

    echo "=== Direction: ${label} ==="

    for seed in "${SEEDS[@]}"; do
        echo "--- seed ${seed} ---"
        # Phase 1: train on train_ds (reuse existing benchmark run output)
        train_out="${OUT_ROOT}/_ckpts/${train_ds}/seed_${seed}"
        mkdir -p "${train_out}"
        BENCHMARK_SEED="${seed}" python3 run_from_config.py configs/v7.yaml \
            --dataset "${train_ds}" \
            --output-root "${train_out}" \
            2>&1 | tee "${train_out}/train.log"

        # Phase 2: load best checkpoint, evaluate on eval_ds test
        # (requires a thin driver — see eval_checkpoint_on_dataset.py below)
        seed_out="${out_dir}/seed_${seed}"
        mkdir -p "${seed_out}"
        python3 scripts/thesis_followups/eval_checkpoint_on_dataset.py \
            --checkpoint "${train_out}/best_model.pt" \
            --eval-dataset "${eval_ds}" \
            --split test \
            --output "${seed_out}/metrics.json" \
            --seed "${seed}" \
            2>&1 | tee "${seed_out}/eval.log"
    done
}

# H -> NH
run_direction "human" "non_human"
# NH -> H
run_direction "non_human" "human"

echo
echo "Done. Aggregate with:"
echo "  python3 scripts/thesis_followups/bootstrap_ci.py \\"
echo "      --paths ${OUT_ROOT}/{human_to_non_human,non_human_to_human}/seed_*/metrics.json \\"
echo "      --compare h_to_nh nh_to_h"
