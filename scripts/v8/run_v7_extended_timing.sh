#!/usr/bin/env bash
# =============================================================================
# Cold-start v7 extended-budget training — single-seed timing test
#
# Hypothesis: 0.506 NH 5-seed plateau (Lição 21, §6.13) is constrained by
# training budget (patience=15, epochs=500) rather than architectural cap.
#
# Test protocol:
#   - Cold-start (from scratch, full optimizer state consistent)
#   - Single seed (42) for timing measurement
#   - Corpus = all (largest, dominates training cost)
#   - patience=50, epochs=2000 (configs/v7_extended.yaml)
#
# Decision rule (after this run completes):
#   - If MCC test improves ≥ +0.015 over vanilla v7 (~0.506 NH / 0.440 All):
#     promote to 5-seed final via SEEDS="42 123 456 789 1024".
#   - If MCC stagnates within ±0.010: plateau hypothesis confirmed for
#     budget axis; pivot to non-budget directions (encoder maior, LoRA-e2e).
#   - Stdout shows wall-clock; multiply by 5 to estimate full 5-seed cost.
#
# Usage:
#   bash scripts/v8/run_v7_extended_timing.sh
#   CORPUS=non_human bash scripts/v8/run_v7_extended_timing.sh
#   SEEDS="42 123" bash scripts/v8/run_v7_extended_timing.sh    # 2-seed
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-all}"
V8ENV="${V8ENV:-env}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
V7_CONFIG="${V7_CONFIG:-configs/v7_extended.yaml}"
export SEEDS="${SEEDS:-42}"

# Preserve the resumable training_checkpoint.pt after training completes,
# so a follow-up run with higher epochs/patience can pick up exactly
# where this run stopped (full optimizer + LR scheduler + best_state +
# early-stop counter restored). Avoids the 5 cold-start vs warm-start
# confounds documented in the methodology log.
export BENCHMARK_LEVEL4CNN_KEEP_CHECKPOINT="${BENCHMARK_LEVEL4CNN_KEEP_CHECKPOINT:-1}"

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
        echo "FATAL: env '${env}' not found (looked for ./${env}/bin/activate and conda env)" >&2
        exit 1
    fi
    set -u
}

activate_env "${V8ENV}"

echo "=============================================================="
echo " v7 Extended-Budget Cold-Start (timing test)"
echo "=============================================================="
echo "  config:    ${V7_CONFIG}"
echo "  corpus:    ${CORPUS}"
echo "  seeds:     ${SEEDS}"
echo "  env:       ${V8ENV}"
echo "  patience:  50 (vs 15 vanilla)"
echo "  epochs:    2000 (vs 500 vanilla)"
echo "  resume:    KEEP_CHECKPOINT=${BENCHMARK_LEVEL4CNN_KEEP_CHECKPOINT}"
echo "             (preserves training_checkpoint.pt for clean continuation)"
echo "  start:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

START_TS=$(date +%s)

python3 run_from_config.py "${V7_CONFIG}" \
    --dataset "${CORPUS}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=============================================================="
echo " DONE — exit=${EXIT_CODE}"
echo "  elapsed:   ${ELAPSED}s (${HOURS}h ${MINUTES}m)"
echo "  estimated 5-seed total: $((ELAPSED * 5))s ($((ELAPSED * 5 / 3600))h)"
echo "  end:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="

exit "${EXIT_CODE}"
