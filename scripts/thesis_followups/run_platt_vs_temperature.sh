#!/usr/bin/env bash
# =============================================================================
# #5 — Platt vs Temperature scaling calibration comparison on v7
#
# Runs the v7 benchmark twice per (dataset, seed) combination:
#   1) Platt scaling     (BENCHMARK_LEVEL4CNN_PLATT=1, TEMPERATURE=0)  [current default]
#   2) Temperature scaling (PLATT=0, TEMPERATURE=1)
#
# All other hyperparameters come from configs/v7.yaml. The test-set MCC,
# AUROC, F1, precision, recall across 5 seeds is the deliverable.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASETS=("${DATASETS:-non_human human}")
OUT_ROOT="${OUT_ROOT:-results/thesis_followups/platt_vs_temperature}"

run_one() {
    local dataset="$1"
    local mode="$2"   # platt | temperature
    local out_dir="${OUT_ROOT}/${dataset}/${mode}"
    mkdir -p "${out_dir}"

    if [[ "${mode}" == "platt" ]]; then
        export BENCHMARK_LEVEL4CNN_PLATT=1
        export BENCHMARK_LEVEL4CNN_TEMPERATURE=0
    else
        export BENCHMARK_LEVEL4CNN_PLATT=0
        export BENCHMARK_LEVEL4CNN_TEMPERATURE=1
    fi

    echo "=== ${dataset} / ${mode} ==="
    python3 run_from_config.py configs/v7.yaml \
        --dataset "${dataset}" \
        --output-root "${out_dir}" \
        2>&1 | tee "${out_dir}/run.log"
}

for ds in ${DATASETS[@]}; do
    run_one "${ds}" "platt"
    run_one "${ds}" "temperature"
done

echo
echo "Done. Aggregate with:"
echo "  python3 scripts/thesis_followups/bootstrap_ci.py \\"
echo "      --paths ${OUT_ROOT}/*/*/metrics.json \\"
echo "      --compare platt temperature"
