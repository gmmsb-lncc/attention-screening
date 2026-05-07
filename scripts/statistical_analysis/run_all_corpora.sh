#!/usr/bin/env bash
# Run run_full_stats.sh over all three corpora sequentially.
#
# Usage:
#   bash scripts/statistical_analysis/run_all_corpora.sh
#
# Forwards N_BOOTSTRAP, PYTHON, MIRROR_TO env vars to the per-corpus runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_full_stats.sh"

for CORPUS in non_human human all; do
    echo ""
    echo "=========================================="
    echo "  corpus = ${CORPUS}"
    echo "=========================================="
    bash "${RUNNER}" "${CORPUS}"
done

echo ""
echo "[run_all_corpora] all three corpora complete."
