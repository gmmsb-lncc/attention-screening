#!/usr/bin/env bash
# =============================================================================
# Generate the two missing ConPLex OFF-DIAGONAL cells with train=all:
#   - all_to_human       (train=all, test=human)
#   - all_to_non_human   (train=all, test=non_human)
#
# Estes 2 pares são os únicos faltantes para fechar a matriz 3x3 do ConPLex.
# Quando completos: 9/9 cells (3 diag + 6 off) → heatmap_conplex.pdf preenchido.
#
# Pre-requisitos:
#   - ConPLex/best_models/trained_all_rep{0,1,2}/ checkpoints
#   - scaffolds_splits/output/{human,non_human}_test.tsv split files
#   - ConPLex env com torch funcional (pip install se broken)
#
# Pipeline:
#   1. Verificar checkpoints + split files
#   2. Inferência cross-corpus para all_to_human (3 seeds disponíveis)
#   3. Inferência cross-corpus para all_to_non_human
#   4. Verificar npz outputs
#   5. Re-aggregate cross_matrix
#   6. Regenerar figuras (heatmap_conplex.pdf agora 9/9)
#
# Usage:
#   bash scripts/thesis_followups/run_conplex_all_offdiag.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-42 123 456}"
CONPLEX_ENV="${CONPLEX_ENV:-conplex}"
SLIDES_DIR="${SLIDES_DIR:-results/slides/cross_matrix}"

CKPT_ROOT="ConPLex/best_models"
SPLITS_DIR="scaffolds_splits/output"
OFF_ROOT="results/cross_matrix/conplex"
SUMMARY_DIR="results/cross_matrix/summary"

hr() { printf '%*s\n' 70 '' | tr ' ' '='; }
step() { hr; echo "[$(date +%H:%M:%S)] STAGE $1: $2"; hr; }

# =============================================================================
# Stage 1 — Pre-flight checks
# =============================================================================
step 1 "Pre-flight checks"

# Map seed → rep
declare -A SEED_TO_REP
SEED_TO_REP[42]=0; SEED_TO_REP[123]=1; SEED_TO_REP[456]=2
SEED_TO_REP[789]=3; SEED_TO_REP[1024]=4

AVAILABLE=""
for seed in ${SEEDS}; do
    rep=${SEED_TO_REP[${seed}]:-}
    [ -z "${rep}" ] && { echo "  [skip] seed=${seed} fora canônicas"; continue; }
    if [ -d "${CKPT_ROOT}/trained_all_rep${rep}" ]; then
        AVAILABLE="${AVAILABLE} ${seed}"
        echo "  [ok] checkpoint trained_all_rep${rep} (seed ${seed})"
    else
        echo "  [missing] trained_all_rep${rep}"
    fi
done
AVAILABLE=$(echo ${AVAILABLE} | xargs)
[ -z "${AVAILABLE}" ] && { echo "[fatal] nenhum checkpoint disponível"; exit 1; }

for split in human non_human; do
    f="${SPLITS_DIR}/${split}_test.tsv"
    if [ -f "${f}" ]; then
        n=$(wc -l < "${f}")
        echo "  [ok] ${f} (${n} linhas)"
    else
        echo "[fatal] split TSV faltando: ${f}"; exit 1
    fi
done

# =============================================================================
# Stage 2 — Activate env
# =============================================================================
step 2 "Activate ConPLex env"

set +u
if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONPLEX_ENV}" 2>/dev/null && echo "  [env] ${CONPLEX_ENV} ativado" \
        || echo "  [warn] não consegui ativar '${CONPLEX_ENV}' — usando env atual"
fi
set -u

# Torch sanity check
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA off'; print(f'  torch={torch.__version__}, cuda OK')" \
    || { echo "[fatal] torch broken. Fix: pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1"; exit 2; }

# =============================================================================
# Stage 3+4 — Inferência cross-corpus para all_to_human + all_to_non_human
# =============================================================================
step 3 "Inferência cross-corpus all → {human, non_human}"

for test_corpus in human non_human; do
    pair="all_to_${test_corpus}"
    out_dir="${OFF_ROOT}/${pair}"
    test_tsv="${SPLITS_DIR}/${test_corpus}_test.tsv"

    echo
    echo "--- ${pair} ---"
    echo "  output: ${out_dir}"
    echo "  test_tsv: ${test_tsv}"

    mkdir -p "${out_dir}"
    python3 infer_conplex_universal.py \
        --corpus all \
        --seeds ${AVAILABLE} \
        --test-tsv "${test_tsv}" \
        --output-dir "${out_dir}" \
        || { echo "[fatal] inferência ${pair} falhou"; exit 3; }
done

# =============================================================================
# Stage 5 — Verificar outputs
# =============================================================================
step 5 "Verificar npz por par + seed"

n_total=0
n_present=0
for pair in all_to_human all_to_non_human; do
    for seed in ${AVAILABLE}; do
        n_total=$((n_total + 1))
        npz="${OFF_ROOT}/${pair}/seed_${seed}/raw_predictions.npz"
        if [ -f "${npz}" ]; then
            n_present=$((n_present + 1))
            sz=$(stat -c '%s' "${npz}" 2>/dev/null || stat -f '%z' "${npz}" 2>/dev/null)
            echo "  [ok] ${pair}/seed_${seed} (${sz} bytes)"
        else
            echo "  [missing] ${npz}"
        fi
    done
done
echo
echo "Total: ${n_present}/${n_total} npz"

# =============================================================================
# Stage 6 — Re-aggregate
# =============================================================================
step 6 "Re-aggregate cross_matrix (com todas diagonais ConPLex)"

AGG="scripts/thesis_followups/cross_dataset_matrix/aggregate.py"
DIAG_ARGS=""
for c in human non_human all; do
    if [ -d "ConPLex/results_universal/${c}" ]; then
        DIAG_ARGS="${DIAG_ARGS} --diagonal-conplex-${c} ConPLex/results_universal/${c}"
    fi
done

mkdir -p "${SUMMARY_DIR}"
python3 "${AGG}" ${DIAG_ARGS} --results-root results/cross_matrix --out-dir "${SUMMARY_DIR}" \
    || { echo "[fatal] aggregate.py falhou"; exit 4; }

# =============================================================================
# Stage 7 — Verificar JSON cobertura ConPLex
# =============================================================================
step 7 "Verificar cobertura ConPLex no JSON"

CHK=$(mktemp /tmp/chk_conplex.XXXXX.py)
cat > "${CHK}" <<'PYEOF'
import json
d = json.load(open("results/cross_matrix/summary/cross_matrix.json"))
c = d.get("conplex", {})
nd = sum(1 for tr in c for te in c[tr] if tr == te and c[tr][te].get("aggregate"))
no = sum(1 for tr in c for te in c[tr] if tr != te and c[tr][te].get("aggregate"))
print(f"ConPLex: {nd}/3 diag, {no}/6 off, total {nd+no}/9")
for tr in ("human", "non_human", "all"):
    for te in ("human", "non_human", "all"):
        cell = c.get(tr, {}).get(te, {})
        agg = cell.get("aggregate", {}).get("mcc", {}) if isinstance(cell, dict) else {}
        m, s = agg.get("mean"), agg.get("std")
        mark = "*" if tr == te else " "
        if m is not None:
            print(f"  {mark} train={tr:10} test={te:10} mcc={m:.4f} ± {s:.4f}")
        else:
            print(f"  {mark} train={tr:10} test={te:10} MISSING")
PYEOF
python3 "${CHK}"
rm -f "${CHK}"

# =============================================================================
# Stage 8 — Regenerar figuras
# =============================================================================
step 8 "Regenerar figuras (heatmap_conplex agora 9/9)"

MAKE="scripts/thesis_followups/cross_dataset_matrix/make_slides.py"
DIAG_ONLY="results/cross_matrix/summary_diag_only/cross_matrix.json"
PRIMARY="${SUMMARY_DIR}/cross_matrix.json"
ARGS="--json ${PRIMARY} --out-dir ${SLIDES_DIR}"
[ -f "${DIAG_ONLY}" ] && ARGS="${ARGS} --diag-json ${DIAG_ONLY}"

mkdir -p "${SLIDES_DIR}"
python3 "${MAKE}" ${ARGS} || { echo "[fatal] make_slides.py falhou"; exit 5; }

echo
echo "Figuras em ${SLIDES_DIR}/figures/"
ls -la "${SLIDES_DIR}/figures/"*.pdf 2>/dev/null

hr
echo "[$(date +%H:%M:%S)] PIPELINE COMPLETO."
echo "Próximo: copiar pdfs para ~/PhD/figures/ e recompilar tese."
hr
