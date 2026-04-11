#!/usr/bin/env bash
# ============================================================================
#  GraphBAN Pretrained Evaluation — compute node execution script
# ============================================================================
#
# Evaluates GraphBAN with original published weights (BioSNAP, BindingDB, KIBA)
# on thesis universal scaffold test sets (non_human, human, all).
#
# Prerequisites:
#   1. conda activate graphban   (from setup_env.sh)
#   2. GraphBAN/src/ cloned      (setup_env.sh handles this)
#   3. Upstream patches applied   (python patch_upstream.py --src src)
#
# Usage:
#   cd /path/to/docktkinase
#   bash GraphBAN/run_pretrained_eval.sh          # all datasets
#   bash GraphBAN/run_pretrained_eval.sh human     # specific dataset
#
# Seeds: The pretrained weights use the upstream seeds {12,14,16,18,20}.
# These are fixed checkpoint IDs from the published GraphBAN, not the thesis
# canonical seeds {42,123,456,789,1024}. This is correct — we are loading
# published weights, not re-training.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# ── Activate environment ──────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate graphban
    echo "[OK] conda env: graphban"
else
    # Fallback: use local venv if no conda
    if [ -f "${REPO_ROOT}/env/bin/activate" ]; then
        source "${REPO_ROOT}/env/bin/activate"
        echo "[OK] venv: ${REPO_ROOT}/env"
    else
        echo "[ERROR] No conda 'graphban' env or local venv found."
        echo "        Run: bash GraphBAN/setup_env.sh"
        exit 1
    fi
fi

# ── Verify DGL is available ───────────────────────────────────────────────
python3 -c "import dgl; from dgllife.utils import smiles_to_bigraph" 2>/dev/null || {
    echo "[ERROR] DGL or dgllife not installed in this environment."
    echo "        Run: pip install dgl dgllife yacs prettytable"
    exit 1
}

# ── Apply upstream patches (idempotent) ───────────────────────────────────
python3 "${SCRIPT_DIR}/patch_upstream.py" --src "${SCRIPT_DIR}/src"

# ── Determine which datasets to evaluate ──────────────────────────────────
DATASET="${1:-all_three}"  # "non_human", "human", "all", or default "all_three" (run all)
SOURCE_MODELS="biosnap bindingdb kiba"
OUTPUT_DIR="${SCRIPT_DIR}/results/pretrained_evaluation"

echo ""
echo "============================================================"
echo "  GraphBAN Pretrained Zero-Shot Evaluation"
echo "============================================================"
echo "  Repository:     ${REPO_ROOT}"
echo "  Source models:   ${SOURCE_MODELS}"
echo "  Output:          ${OUTPUT_DIR}"
echo ""

run_eval() {
    local ds="$1"
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Evaluating: ${ds}"
    echo "════════════════════════════════════════════════════════"
    python3 "${SCRIPT_DIR}/evaluate_pretrained.py" \
        --dataset "${ds}" \
        --source-models ${SOURCE_MODELS} \
        --batch-size 64 \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/${ds}_eval.log"
    echo ""
    echo "[OK] ${ds} complete. Results: ${OUTPUT_DIR}/${ds}/pretrained_evaluation.json"
}

if [ "${DATASET}" = "all_three" ]; then
    # Run all three datasets sequentially
    # Features are cached per dataset, so human+non_human only extract once
    for ds in non_human human all; do
        run_eval "${ds}"
    done
else
    run_eval "${DATASET}"
fi

echo ""
echo "============================================================"
echo "  All evaluations complete!"
echo "  Results directory: ${OUTPUT_DIR}"
echo "============================================================"
echo ""
echo "Result files:"
find "${OUTPUT_DIR}" -name "pretrained_evaluation.json" | sort
