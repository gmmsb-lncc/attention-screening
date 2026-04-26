#!/usr/bin/env bash
# =============================================================================
# Generate ConPLex diagonal `all` corpus + re-aggregate cross_matrix + regenerate
# bar/heatmap figures for the thesis Anexo A.
#
# Pre-requisitos (verificados no início):
#   - ConPLex/best_models/trained_all_rep{0,1,2}/  (checkpoints treinados)
#   - ConPLex env Python ativável (conda activate conplex)
#   - infer_conplex_universal.py funcional
#
# Pipeline (5 estágios sequenciais, cada um guarded por verificação):
#   1. Detect available reps + seeds
#   2. Run inference for each available seed (skip if output already exists)
#   3. Verify outputs (raw_predictions.npz por seed)
#   4. Re-aggregate via aggregate.py (inclui diagonal all)
#   5. Re-generate make_slides.py figures (heatmaps + bars)
#
# Output:
#   ConPLex/results_universal/all/seed_{42,123,456}/raw_predictions.npz
#   results/cross_matrix/summary/cross_matrix.json (atualizado)
#   slides/cross_matrix/figures/heatmap_conplex.pdf (TODAS 9 células)
#
# Usage (de qualquer diretório):
#   bash scripts/thesis_followups/run_conplex_all_diag.sh
#
# Env overrides:
#   SEEDS         (default: "42 123 456" — reps 0,1,2 disponíveis em d03)
#   CONPLEX_ENV   (default: "conplex" — conda env)
#   SKIP_INFER    (default: "0" — pular estágio 2 se outputs já existem)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-42 123 456}"
CONPLEX_ENV="${CONPLEX_ENV:-conplex}"
SKIP_INFER="${SKIP_INFER:-0}"

CKPT_ROOT="ConPLex/best_models"
OUT_ROOT="ConPLex/results_universal/all"
SUMMARY_DIR="results/cross_matrix/summary"
SLIDES_DIR="slides/cross_matrix"

# Mapping seed → rep (matches CANONICAL_SEEDS in infer_conplex_universal.py).
declare -A SEED_TO_REP
SEED_TO_REP[42]=0
SEED_TO_REP[123]=1
SEED_TO_REP[456]=2
SEED_TO_REP[789]=3
SEED_TO_REP[1024]=4

# =============================================================================
# Helpers
# =============================================================================

hr() { printf '%*s\n' 70 '' | tr ' ' '='; }

step() {
    hr
    echo "[$(date +%H:%M:%S)] STAGE $1: $2"
    hr
}

# =============================================================================
# Stage 1 — Detect available checkpoints
# =============================================================================
step 1 "Detect available reps for ConPLex 'all' corpus"

AVAILABLE_SEEDS=""
MISSING_SEEDS=""
for seed in ${SEEDS}; do
    rep=${SEED_TO_REP[${seed}]:-}
    if [ -z "${rep}" ]; then
        echo "  [warn] seed=${seed} fora das canônicas (42,123,456,789,1024)"
        continue
    fi
    ckpt_dir="${CKPT_ROOT}/trained_all_rep${rep}"
    if [ -d "${ckpt_dir}" ]; then
        echo "  [ok]   seed=${seed} → rep${rep} → ${ckpt_dir}"
        AVAILABLE_SEEDS="${AVAILABLE_SEEDS} ${seed}"
    else
        echo "  [miss] seed=${seed} → rep${rep} → ${ckpt_dir} (não existe)"
        MISSING_SEEDS="${MISSING_SEEDS} ${seed}"
    fi
done
AVAILABLE_SEEDS=$(echo ${AVAILABLE_SEEDS} | xargs)

if [ -z "${AVAILABLE_SEEDS}" ]; then
    echo "[fatal] nenhum checkpoint disponível em ${CKPT_ROOT} — precisa treinar primeiro"
    exit 1
fi

n_avail=$(echo ${AVAILABLE_SEEDS} | wc -w)
n_total=$(echo ${SEEDS} | wc -w)
echo
echo "Disponíveis: ${n_avail}/${n_total} seeds → '${AVAILABLE_SEEDS}'"
if [ -n "${MISSING_SEEDS}" ]; then
    echo "Faltando:  '${MISSING_SEEDS}' (precisa treinar reps correspondentes para 5-seed canônico)"
fi

# =============================================================================
# Stage 2 — Inferência diagonal
# =============================================================================
step 2 "Inferência ConPLex --corpus all (idempotente: skip outputs existentes)"

# Verifica quantos seeds já têm output salvo
EXISTING=0
for seed in ${AVAILABLE_SEEDS}; do
    if [ -f "${OUT_ROOT}/seed_${seed}/raw_predictions.npz" ]; then
        EXISTING=$((EXISTING + 1))
    fi
done
echo "  Outputs existentes: ${EXISTING}/${n_avail}"

if [ "${SKIP_INFER}" = "1" ]; then
    echo "  SKIP_INFER=1 → pulando estágio 2"
elif [ "${EXISTING}" -eq "${n_avail}" ]; then
    echo "  Todos os outputs já existem → infer pulará automaticamente (idempotente)"
fi

# Activate conda env if available
set +u
if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        if conda activate "${CONPLEX_ENV}" 2>/dev/null; then
            echo "  [env] conda activate ${CONPLEX_ENV} OK"
        else
            echo "  [warn] não consegui ativar '${CONPLEX_ENV}' — continuando com env atual"
        fi
    fi
fi
set -u

if [ "${SKIP_INFER}" != "1" ]; then
    echo "  Executando infer_conplex_universal.py --corpus all --seeds ${AVAILABLE_SEEDS}"
    python3 infer_conplex_universal.py \
        --corpus all \
        --seeds ${AVAILABLE_SEEDS} \
        || { echo "[fatal] inferência falhou"; exit 2; }
fi

# =============================================================================
# Stage 3 — Verificar outputs
# =============================================================================
step 3 "Verificar raw_predictions.npz por seed"

PRESENT=""
ABSENT=""
for seed in ${AVAILABLE_SEEDS}; do
    npz="${OUT_ROOT}/seed_${seed}/raw_predictions.npz"
    if [ -f "${npz}" ]; then
        size=$(stat -c '%s' "${npz}" 2>/dev/null || stat -f '%z' "${npz}" 2>/dev/null)
        echo "  [ok] seed=${seed} → ${npz} (${size} bytes)"
        PRESENT="${PRESENT} ${seed}"
    else
        echo "  [missing] seed=${seed} → ${npz} NÃO EXISTE"
        ABSENT="${ABSENT} ${seed}"
    fi
done

n_present=$(echo ${PRESENT} | wc -w)
echo
echo "Presentes: ${n_present}/${n_avail} seeds"
if [ -n "${ABSENT}" ]; then
    echo "Ausentes: ${ABSENT}"
    echo "[warn] continuando mesmo assim — aggregate.py lida com seeds parciais"
fi

# =============================================================================
# Stage 4 — Re-aggregate cross_matrix
# =============================================================================
step 4 "Re-aggregate via scripts/thesis_followups/cross_dataset_matrix/aggregate.py"

AGG="scripts/thesis_followups/cross_dataset_matrix/aggregate.py"
if [ ! -f "${AGG}" ]; then
    echo "[fatal] aggregate.py não encontrado em ${AGG}"
    exit 3
fi

# Build args dynamicamente: include all 3 ConPLex diagonal corpora (human,
# non_human, all). aggregate.py vai mergir off-diagonal cells de
# results/cross_matrix/conplex/ automaticamente.
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
# Stage 5 — Verificar JSON agregado
# =============================================================================
step 5 "Verificar cobertura ConPLex no JSON agregado"

CHECK_PY=$(mktemp /tmp/check_conplex.XXXXXX.py)
cat > "${CHECK_PY}" <<'PYEOF'
import json, sys
path = "results/cross_matrix/summary/cross_matrix.json"
try:
    d = json.load(open(path))
except Exception as e:
    print(f"[fatal] cannot load {path}: {e}")
    sys.exit(1)
c = d.get("conplex", {})
n_diag = sum(1 for tr in c for te in c[tr] if tr == te and c[tr][te].get("aggregate"))
n_off  = sum(1 for tr in c for te in c[tr] if tr != te and c[tr][te].get("aggregate"))
print(f"ConPLex: {n_diag}/3 diagonais, {n_off}/6 off-diagonais → total {n_diag + n_off}/9")
# Per-cell mcc
for tr in ("human", "non_human", "all"):
    for te in ("human", "non_human", "all"):
        cell = c.get(tr, {}).get(te, {})
        agg = cell.get("aggregate", {}).get("mcc", {}) if isinstance(cell, dict) else {}
        mark = "*" if tr == te else " "
        m = agg.get("mean")
        s = agg.get("std")
        if m is not None:
            print(f"  {mark} train={tr:10} test={te:10} mcc={m:.4f} ± {s:.4f}")
        else:
            print(f"  {mark} train={tr:10} test={te:10} MISSING")
PYEOF
python3 "${CHECK_PY}"
rm -f "${CHECK_PY}"

# =============================================================================
# Stage 6 — Regenerar figuras (heatmaps + bars)
# =============================================================================
step 6 "Regenerar figuras (slides + thesis Anexo A)"

MAKE="scripts/thesis_followups/cross_dataset_matrix/make_slides.py"
DIAG_ONLY="results/cross_matrix/summary_diag_only/cross_matrix.json"
PRIMARY="${SUMMARY_DIR}/cross_matrix.json"

ARGS="--json ${PRIMARY} --out-dir ${SLIDES_DIR}"
if [ -f "${DIAG_ONLY}" ]; then
    ARGS="${ARGS} --diag-json ${DIAG_ONLY}"
    echo "  [info] usando merge --diag-json ${DIAG_ONLY}"
fi

mkdir -p "${SLIDES_DIR}"
python3 "${MAKE}" ${ARGS} || { echo "[fatal] make_slides.py falhou"; exit 5; }

echo
echo "  Figuras geradas em ${SLIDES_DIR}/figures/"
ls -la "${SLIDES_DIR}/figures/"*.pdf 2>/dev/null | awk '{print "    "$NF, "("$5" bytes)"}'

# =============================================================================
# Stage 7 — Próximos passos manuais (cópia para thesis repo)
# =============================================================================
step 7 "Próximos passos (manuais)"

cat <<EOF
  1. Copiar PDFs atualizadas para thesis repo:
       cp ${SLIDES_DIR}/figures/*.pdf ~/PhD/figures/
       cp ${SLIDES_DIR}/figures/*.pdf ~/PhD/cross_matrix/figures/

  2. Recompilar tese:
       cd ~/PhD && pdflatex tese_lncc.tex && pdflatex tese_lncc.tex

  3. Verificar Figura cross_matrix do ConPLex (heatmap_conplex.pdf):
       agora deve mostrar TODAS as 9 células do 3x3 preenchidas.

  4. (Opcional) Treinar reps 3, 4 (seeds 789, 1024) para 5-seed canônico:
       python3 ConPLex/train_DTI.py --dataset all --rep 3
       python3 ConPLex/train_DTI.py --dataset all --rep 4
       Tempo: várias horas por rep em RTX 4090.
EOF

hr
echo "[$(date +%H:%M:%S)] PIPELINE COMPLETO."
hr
