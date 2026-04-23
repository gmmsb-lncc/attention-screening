#!/usr/bin/env bash
# =============================================================================
# Run ConPLex only on the 6 off-diagonal cross-dataset cells.
#
# Wrapper around run_cross_matrix.sh with MODELS=conplex. Use AFTER ConPLex
# has all 5 rep checkpoints trained (trained_{corpus}_rep{0..4}). Writes to
# the same layout as the full cross-matrix, so aggregate.py picks up all
# four models without reconfiguration:
#     results/cross_matrix/conplex/{train}_to_{test}/seed_{s}/raw_predictions.npz
#
# Re-running is idempotent (per-seed raw_predictions.npz acts as skip marker).
#
# Env overrides:
#   SEEDS          seeds (default: 42 123 456 789 1024)
#   OUT_ROOT       output dir (default: results/cross_matrix)
#   CONPLEX_ENV    conda env (default: conplex)
#   SKIP_LEAKAGE_FILTER  1 = reuse unmodified test TSV
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export MODELS="conplex"

echo "=============================================================="
echo "ConPLex-only cross-dataset run"
echo "  delegating to run_cross_matrix.sh with MODELS=conplex"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_cross_matrix.sh" "$@"

echo
echo "Done. To aggregate with the previously-run 3-model diagonal + cross:"
echo "  python3 ${SCRIPT_DIR}/aggregate.py \\"
echo "      --results-root results/cross_matrix \\"
echo "      --out-dir results/cross_matrix/summary \\"
echo "      --diagonal-conplex-human     ConPLex/results_universal/human \\"
echo "      --diagonal-conplex-non_human ConPLex/results_universal/non_human \\"
echo "      --diagonal-conplex-all       ConPLex/results_universal/all \\"
echo "      [... plus the 9 diagonal paths for dtkinase/drugban/graphban ...]"
