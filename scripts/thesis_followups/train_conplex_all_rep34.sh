#!/usr/bin/env bash
# =============================================================================
# Train ConPLex reps 3 and 4 (seeds 789, 1024) for the `all` corpus.
#
# Por que script dedicado:
#   run_conplex_full_pipeline.sh nomeia checkpoints por rep_idx (0-based array
#   index), NÃO pelo seed real. Se override `SEEDS=(789 1024)`, ele criaria
#   trained_all_rep0 e trained_all_rep1, SOBRESCREVENDO os checkpoints
#   existentes. Este script chama train_DTI.py diretamente com exp_id explícito.
#
# Pre-requisitos:
#   - ConPLex env com torch funcional (pip install se broken)
#   - ConPLex/dataset/kinase_all/ data prontos
#   - ConPLex/configs/ config padrão disponível
#
# Output:
#   ConPLex/best_models/trained_all_rep3/trained_all_rep3_best_model.pt
#   ConPLex/best_models/trained_all_rep4/trained_all_rep4_best_model.pt
#
# Tempo estimado em RTX 4090: ~30-60 min por rep (`all` é 386k pairs, ~2× NH).
#
# Após terminar:
#   bash scripts/thesis_followups/run_conplex_all_diag.sh    # diag all (5 seeds)
#   bash scripts/thesis_followups/run_conplex_all_offdiag.sh # offdiag all_to_*
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/ConPLex"

DEVICE="${CUDA_DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONPLEX_ENV="${CONPLEX_ENV:-conplex}"
CONFIG_FILE="${CONFIG_FILE:-configs/default_config.yaml}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-best_models}"

# Reps a treinar: 3 (seed=789) e 4 (seed=1024)
declare -A REP_TO_SEED
REP_TO_SEED[3]=789
REP_TO_SEED[4]=1024

REPS_TO_TRAIN="${REPS_TO_TRAIN:-3 4}"

hr() { printf '%*s\n' 70 '' | tr ' ' '='; }
step() { hr; echo "[$(date +%H:%M:%S)] $1"; hr; }

# =============================================================================
# Pre-flight
# =============================================================================
step "Pre-flight checks"

if [ ! -f "train_DTI.py" ]; then
    echo "[fatal] train_DTI.py não encontrado em $(pwd)"; exit 1
fi
if [ ! -d "dataset/kinase_all" ]; then
    echo "[fatal] dataset/kinase_all não encontrado"; exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "[warn] ${CONFIG_FILE} não encontrado — usando default do train_DTI.py"
    CONFIG_FILE=""
fi

# Activate env
set +u
if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] && source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONPLEX_ENV}" 2>/dev/null && echo "[env] ${CONPLEX_ENV} ativado" \
        || echo "[warn] não consegui ativar '${CONPLEX_ENV}'"
fi
set -u

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA off'; print(f'[torch] {torch.__version__} cuda OK')" \
    || { echo "[fatal] torch broken. Fix: pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1"; exit 2; }

mkdir -p "${MODEL_SAVE_DIR}"

# =============================================================================
# Train each rep
# =============================================================================
total_start=$(date +%s)

for rep in ${REPS_TO_TRAIN}; do
    seed="${REP_TO_SEED[${rep}]:-}"
    if [ -z "${seed}" ]; then
        echo "[skip] rep=${rep} fora de {3, 4}"; continue
    fi
    exp_id="trained_all_rep${rep}"
    ckpt_dir="${MODEL_SAVE_DIR}/${exp_id}"

    step "Training ${exp_id} (seed=${seed}, ${EPOCHS} epochs, batch=${BATCH_SIZE})"

    if [ -d "${ckpt_dir}" ] && ls "${ckpt_dir}"/${exp_id}_best_model*.pt &>/dev/null; then
        echo "[skip] ${ckpt_dir} já tem checkpoint best — re-treinar com SKIP_EXISTING=0"
        if [ "${SKIP_EXISTING:-1}" = "0" ]; then
            echo "[force] SKIP_EXISTING=0 → removendo e re-treinando"
            rm -rf "${ckpt_dir}"
        else
            continue
        fi
    fi

    rep_start=$(date +%s)
    cmd_args=(
        --exp-id "${exp_id}"
        --task "kinase_all"
        --d "${DEVICE}"
        --r "${seed}"
        --epochs "${EPOCHS}"
        -b "${BATCH_SIZE}"
    )
    [ -n "${CONFIG_FILE}" ] && cmd_args+=(--config "${CONFIG_FILE}")

    python train_DTI.py "${cmd_args[@]}" 2>&1 | tail -20
    rep_end=$(date +%s)

    if ls "${ckpt_dir}"/${exp_id}_best_model*.pt 1>/dev/null 2>&1; then
        latest=$(ls -t "${ckpt_dir}"/${exp_id}_best_model*.pt | head -1)
        echo "[ok] checkpoint: $(basename ${latest}) ($(( (rep_end - rep_start) / 60 ))min)"
    else
        echo "[error] nenhum checkpoint salvo para ${exp_id}"
    fi
done

total_end=$(date +%s)
elapsed=$((total_end - total_start))
hr
echo "[$(date +%H:%M:%S)] DONE — total $((elapsed / 60))min $((elapsed % 60))s"

cat <<EOF

Próximos passos:
  1. Re-rodar diagonais all (incluindo seeds 789, 1024):
       SEEDS="42 123 456 789 1024" bash scripts/thesis_followups/run_conplex_all_diag.sh
  2. Re-rodar offdiagonais all_to_*:
       SEEDS="42 123 456 789 1024" bash scripts/thesis_followups/run_conplex_all_offdiag.sh
  3. Re-aggregate + regenerate figures (ambos scripts já fazem isso).
  4. Copiar PDFs para ~/PhD/figures/ e recompilar tese.
EOF
hr
