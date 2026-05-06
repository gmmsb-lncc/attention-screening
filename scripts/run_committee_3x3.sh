#!/usr/bin/env bash
# =============================================================================
# Build 3×3 cross-corpus committee matrix on the canonical test set
# =============================================================================
#
# Generates the (train_corpus × test_corpus) MCC/AUROC/F1 matrix for the
# 3-model human_kinome committee (DT-Kinase + DrugBAN + ConPLex), plus
# 9 confusion matrices, in the style of the thesis Anexo A.
#
# Inputs (must already exist in the repo):
#   - DT-Kinase diagonal:    results/{benchmark_*_8M_*}/test/level4_cnn_8M/{c}/seed_*/raw_predictions.npz
#   - DT-Kinase off-diag:    results/cross_matrix/dtkinase/{train}_to_{test}/seed_*/metrics.json
#   - DrugBAN diagonal:      DrugBAN/results_universal/results_universal/{c}/seed_*/raw_predictions.npz
#   - DrugBAN off-diag:      results/cross_matrix/drugban/{train}_to_{test}/seed_*/raw_predictions.npz
#   - ConPLex diagonal:      ConPLex/results_universal/{c}/seed_*/raw_predictions.npz
#   - ConPLex off-diag:      results/cross_matrix/conplex/{train}_to_{test}/seed_*/raw_predictions.npz
#   - scaffolds_splits/output/universal_test.tsv
#
# Output dir (default): results/inference/committee_3x3_human_kinome/
#
# Usage:
#   bash scripts/run_committee_3x3.sh                          # default 3-model
#   bash scripts/run_committee_3x3.sh --models dtkinase,drugban,graphban,conplex
#   bash scripts/run_committee_3x3.sh --out-dir custom/path
#
# Env knobs:
#   PYBIN  Python interpreter (default: $REPO/env/bin/python)
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PYBIN:-${REPO}/env/bin/python}"
SCRIPT="${REPO}/scripts/inference/experiments/committee_3x3_human_kinome.py"

if [[ ! -x "${PYBIN}" ]]; then
    echo "FATAL: Python interpreter not found at ${PYBIN}" >&2
    echo "       set PYBIN=/path/to/python or activate the right env" >&2
    exit 1
fi

if [[ ! -f "${SCRIPT}" ]]; then
    echo "FATAL: aggregator missing at ${SCRIPT}" >&2
    exit 1
fi

echo "=============================================================="
echo " 3×3 cross-corpus committee matrix"
echo "  PYBIN  : ${PYBIN}"
echo "  SCRIPT : ${SCRIPT}"
echo "  ARGS   : $*"
echo "=============================================================="

cd "${REPO}"
"${PYBIN}" "${SCRIPT}" "$@"

echo ""
echo "=============================================================="
echo " DONE"
echo "=============================================================="
