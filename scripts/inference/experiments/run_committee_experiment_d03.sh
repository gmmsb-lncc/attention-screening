#!/usr/bin/env bash
# =============================================================================
# Committee vs Individual Models — empirical evaluation runner
#
# Hypothesis under test: the 4-model consensus committee outperforms (or
# at least matches) each of the four individual models on the canonical
# in-domain test sets, when each individual model is itself the 5-seed
# ensemble of its own canonical seeds.
#
# Protocol:
#   - For each corpus (Non-Human, Human, All):
#     * Load raw_predictions.npz across 5 seeds {42,123,456,789,1024} for
#       each of the 4 models.
#     * Compute prob = mean over 5 seeds; threshold = mean over 5 calibration
#       sidecars.
#     * Build committee: prob = mean over 4 models; threshold = mean over
#       the 4 model thresholds.
#     * Evaluate MCC + AUROC + F1 + Accuracy on universal_test.tsv labels.
#   - Per-pair paired bootstrap (B=10000, IC95 percentile) for the deltas
#     committee − each individual model.
#
# Output:
#   results/inference/committee_experiment/<corpus>_metrics.csv
#   results/inference/committee_experiment/<corpus>_paired_bootstrap.csv
#   results/inference/committee_experiment/REPORT.md
#
# Pure NumPy / pandas / scikit-learn — no GPU needed, no model encoders
# loaded. Inputs are the raw_predictions.npz versioned in the repository.
#
# Expected wall-clock on diamante-03 (single thread):
#   - Non-Human (1.7k pairs): ~10s
#   - Human    (40k pairs):   ~3 min
#   - All      (41k pairs):   ~3 min
#   - Total:                  ~7 min
#
# Usage:
#   bash scripts/inference/experiments/run_committee_experiment_d03.sh
#   B=2000 bash scripts/inference/experiments/run_committee_experiment_d03.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

V8ENV="${V8ENV:-env}"
B="${B:-10000}"  # bootstrap resamples (set lower for faster turnaround)

activate_env() {
    local env="$1"
    set +u
    if [ -f "${REPO_ROOT}/${env}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/${env}/bin/activate"
    elif command -v conda > /dev/null && conda env list | grep -q " ${env} "; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${env}"
    else
        echo "FATAL: env '${env}' not found" >&2
        exit 1
    fi
    set -u
}

activate_env "${V8ENV}"

echo "=============================================================="
echo " Committee vs Individual Models — empirical evaluation"
echo "=============================================================="
echo "  env:         ${V8ENV}"
echo "  bootstrap:   B = ${B}"
echo "  output dir:  ${REPO_ROOT}/results/inference/committee_experiment/"
echo "  start:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

START_TS=$(date +%s)

# Run with -u (unbuffered) so progress prints incrementally even when
# stdout is captured by a CI/log harness. The script's own bootstrap loop
# emits a progress line every 500 resamples per comparison.
BENCHMARK_BOOTSTRAP_B="${B}" \
    python3 -u scripts/inference/experiments/committee_vs_individual.py
EXIT=$?

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "=============================================================="
echo " DONE — exit=${EXIT}"
echo "  elapsed:  ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo "  end:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="
echo ""
echo "Inspect results:"
echo "  cat results/inference/committee_experiment/REPORT.md"
echo "  cat results/inference/committee_experiment/{non_human,human,all}_metrics.csv"
echo "  cat results/inference/committee_experiment/{non_human,human,all}_paired_bootstrap.csv"

exit "${EXIT}"
