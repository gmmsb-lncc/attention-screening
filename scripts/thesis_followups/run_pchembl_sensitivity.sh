#!/usr/bin/env bash
# =============================================================================
# #6 — pChEMBL activity threshold sensitivity analysis on v7
#
# The thesis uses pChEMBL >= 6.0 (1 uM) as the active/inactive cutoff and
# mentions 10 uM (pChEMBL 5.0) only for class-balance analysis. This sweep
# re-trains v7 with tau_bio in {5.5, 6.0, 6.5, 7.0, 7.5} and reports
# MCC/AUROC/F1 at each threshold, revealing sensitivity to the cutoff.
#
# IMPORTANT: the threshold is read from benchmark/config.py
# (PCHEMBL_ACTIVITY_THRESHOLD). We override it via env var, which the
# benchmark respects through BENCHMARK_PCHEMBL_THRESHOLD (added if absent;
# see usage note below).
#
# If BENCHMARK_PCHEMBL_THRESHOLD is not yet wired into the code, apply the
# minimal patch documented at the bottom of this script before running.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASETS=("${DATASETS:-non_human human}")
THRESHOLDS=(${THRESHOLDS:-5.5 6.0 6.5 7.0 7.5})
OUT_ROOT="${OUT_ROOT:-results/thesis_followups/pchembl_sensitivity}"

for ds in ${DATASETS[@]}; do
    for tau in ${THRESHOLDS[@]}; do
        out_dir="${OUT_ROOT}/${ds}/tau_${tau}"
        mkdir -p "${out_dir}"
        export BENCHMARK_PCHEMBL_THRESHOLD="${tau}"
        echo "=== ${ds} / pChEMBL >= ${tau} ==="
        python3 run_from_config.py configs/v7.yaml \
            --dataset "${ds}" \
            --output-root "${out_dir}" \
            2>&1 | tee "${out_dir}/run.log"
    done
done

echo
echo "Done. Expected deliverable: sensitivity curve (MCC vs tau_bio) per dataset."

# ---------------------------------------------------------------------------
# Required one-time code patch (if not yet applied)
# ---------------------------------------------------------------------------
# In benchmark/config.py, replace the constant with an env-overridable value:
#
#   PCHEMBL_ACTIVITY_THRESHOLD = float(
#       os.getenv("BENCHMARK_PCHEMBL_THRESHOLD", "6.0")
#   )  # IC50 <= 1000 nM at tau=6.0
#
# All call sites (benchmark/levels/matrix_utils.py:261, level4.py:778,
# level1.py:29) already import this constant, so the sweep just works.
