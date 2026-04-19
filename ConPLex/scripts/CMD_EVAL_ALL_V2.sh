#!/usr/bin/env bash
# CMD_EVAL_ALL_V2.sh — Run eval_conplex_v2.py (MCC-optimal calibration) on the kinase_all corpus.
#
# Prerequisites (run on the production machine where checkpoints live):
#   - Checkpoints trained on kinase_all present at:
#       /data/docktkinase/ConPLex/best_models/trained_all_rep{0..N}/
#         trained_all_rep{0..N}_best_model.pt
#   - Dataset ConPLex/dataset/kinase_all/{train,val,test}.csv
#
# Output goes to ConPLex/results_v2/conplex_v2_all_rep{0..N}/:
#   - val_predictions.csv      raw val scores
#   - test_predictions.csv     raw test scores
#   - results_v2.json          metrics under τ=0.5 AND τ* (val-MCC-optimal)
#   - threshold_sweep.csv      full τ sweep (2000 points on val)
#
# After all reps finish, run:
#   python aggregate_conplex_v2.py --results-dir results_v2
# to get the mean ± σ table ready for Apêndice E Tab E.3.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (edit if paths differ on your machine)
# ---------------------------------------------------------------------------
CONPLEX_DIR="${CONPLEX_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CONPLEX_DIR}/best_models}"
RESULTS_DIR="${RESULTS_DIR:-${CONPLEX_DIR}/results_v2}"
DEVICE="${DEVICE:-0}"

# Which reps to evaluate. Adjust to match canonical seeds {42, 123, 456, 789, 1024}.
# Current trained_all_rep0..2 correspond to the first three canonical seeds;
# rep3/rep4 must be trained on the remaining two seeds before paired-bootstrap
# comparison is valid.
REPS=(0 1 2)  # extend to (0 1 2 3 4) once rep3/rep4 are trained

cd "${CONPLEX_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "========================================================================"
echo "ConPLex v2 evaluation on kinase_all"
echo "  CONPLEX_DIR   = ${CONPLEX_DIR}"
echo "  CHECKPOINT_DIR= ${CHECKPOINT_DIR}"
echo "  RESULTS_DIR   = ${RESULTS_DIR}"
echo "  DEVICE        = ${DEVICE}"
echo "  REPS          = ${REPS[*]}"
echo "========================================================================"

for rep in "${REPS[@]}"; do
    CKPT="${CHECKPOINT_DIR}/trained_all_rep${rep}/trained_all_rep${rep}_best_model.pt"
    EXP_ID="conplex_v2_all_rep${rep}"
    OUT_DIR="${RESULTS_DIR}/${EXP_ID}"

    if [ ! -f "${CKPT}" ]; then
        echo "[skip] checkpoint missing: ${CKPT}"
        continue
    fi

    if [ -f "${OUT_DIR}/results_v2.json" ]; then
        echo "[skip] already evaluated: ${OUT_DIR}/results_v2.json"
        continue
    fi

    echo ""
    echo "--- rep${rep} ---"
    python eval_conplex_v2.py \
        --checkpoint "${CKPT}" \
        --data-dir   dataset/kinase_all \
        --exp-id     "${EXP_ID}" \
        --device     "${DEVICE}"
done

echo ""
echo "========================================================================"
echo "Aggregating results"
echo "========================================================================"
python aggregate_conplex_v2.py --results-dir "${RESULTS_DIR}"

echo ""
echo "Done. Integrate the produced numbers into:"
echo "  - PhD/tex/apendiceE.tex  Tab E.3  row 'All'"
echo "  - PhD/tex/apendiceE.tex  §Posicionamento    (remove 'preliminar' caveat)"
echo "  - PhD/tese_lncc.tex      resumo PT/EN        (optional: ConPLex All number)"
