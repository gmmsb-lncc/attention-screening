#!/usr/bin/env bash
# Dispatch helper for committee_per_seed_poe.py — PoE per-seed metrics.
#
# Usage:
#   # Single corpus (parallel dispatch across machines):
#   ./run_committee_per_seed_poe.sh non_human       # machine A
#   ./run_committee_per_seed_poe.sh human           # machine A (sequential after NH)
#   ./run_committee_per_seed_poe.sh all             # machine B (separate)
#
#   # All 3 corpora sequential (single machine, canonical NH→Human→All):
#   ./run_committee_per_seed_poe.sh
#
# Machine-A profile (NH + Human, low memory):
#   ./run_committee_per_seed_poe.sh non_human && \
#   ./run_committee_per_seed_poe.sh human
#
# Machine-B profile (All, separate to parallelize):
#   ./run_committee_per_seed_poe.sh all
#
# Output: results/inference/committee_per_seed_poe/<corpus>/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY_SCRIPT="$REPO_ROOT/scripts/inference/experiments/committee_per_seed_poe.py"

if [ $# -eq 0 ]; then
    echo "[run] no corpus arg → executing canonical order NH → Human → All"
    cd "$REPO_ROOT" && python3 "$PY_SCRIPT"
    exit $?
fi

CORPUS="$1"
case "$CORPUS" in
    non_human|human|all)
        echo "[run] corpus=$CORPUS"
        cd "$REPO_ROOT" && python3 "$PY_SCRIPT" --corpus "$CORPUS"
        ;;
    *)
        echo "ERROR: corpus must be one of {non_human, human, all}; got '$CORPUS'" >&2
        exit 2
        ;;
esac
