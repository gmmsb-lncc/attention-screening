#!/usr/bin/env bash
# Run run_full_stats.sh for "human" and then "all", sequentially.
# Starts "all" as soon as "human" finishes (only if "human" exits 0).
#
# Usage:
#   bash scripts/statistical_analysis/run_human_then_all.sh
#
# Forwards N_BOOTSTRAP, PYTHON, MIRROR_TO env vars to the per-corpus runner.
# Defaults:
#   N_BOOTSTRAP = 10000 (canonical, inherited from run_full_stats.sh)
#   MIRROR_TO   = ${HOME}/PhD/figures   (set MIRROR_TO= to disable mirroring)
#
# Per-corpus stdout+stderr is teed to results/statistical/<corpus>/run.log
# so you can `tail -f` it from another shell.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_full_stats.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MIRROR_TO="${MIRROR_TO-${HOME}/PhD/figures}"

cd "${REPO_ROOT}"

for CORPUS in human all; do
    OUT_DIR="results/statistical/${CORPUS}"
    mkdir -p "${OUT_DIR}"
    LOG="${OUT_DIR}/run.log"
    echo ""
    echo "=========================================="
    echo "  corpus    = ${CORPUS}    started $(date -Iseconds)"
    echo "  log       = ${LOG}"
    echo "  N_BOOT    = ${N_BOOTSTRAP:-10000}"
    echo "  MIRROR_TO = ${MIRROR_TO:-<disabled>}"
    echo "=========================================="
    bash "${RUNNER}" "${CORPUS}" 2>&1 | tee "${LOG}"
    echo "  corpus = ${CORPUS}    finished $(date -Iseconds)"
done

echo ""
echo "[run_human_then_all] human + all complete."
