#!/usr/bin/env bash
# =============================================================================
# Human-specialist committee — Pareto-optimal for human kinome screening
# =============================================================================
#
# 3-model committee: DT-Kinase + DrugBAN + ConPLex (drops GraphBAN).
# In-domain human checkpoints. Validated empirically:
#
#   ΔMCC = +0.0074 vs 4-model canonical
#   IC95 = [+0.0014, +0.0136]   (block bootstrap by protein, B = 10000)
#   p_two-sided = 0.022
#
#   Holm-Bonferroni m=9 (vs 3 individuais × 3 corpora):
#     6/9 sobrevive α=0.05 (corpus human + all)
#
# 25% lower compute than canonical (3 forwards vs 4).
#
# Reference: results/inference/committee_no_graphban_holm/REPORT.md
#
# Usage:
#   bash scripts/inference/run_human_specialist.sh "<SMILES>" [output_dir]
#   bash scripts/inference/run_human_specialist.sh imatinib.smi
#   bash scripts/inference/run_human_specialist.sh batch.tsv
# =============================================================================

set -eu
set -o pipefail

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $0 <input> [out_dir]

  input    : SMILES string OR path to .smi/.fa/.csv/.tsv
  out_dir  : optional (default: results/inference/human_specialist_<TS>)
EOF
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${REPO}/env/bin/python"

INPUT="$1"
OUT="${2:-${REPO}/results/inference/human_specialist_$(date +%Y%m%d_%H%M%S)}"

"${PYBIN}" "${REPO}/attention_screening.py" "${INPUT}" \
    --profile human_kinome \
    --out "${OUT}" \
    "${@:3}"
