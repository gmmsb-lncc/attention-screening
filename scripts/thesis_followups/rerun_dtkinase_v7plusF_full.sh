#!/usr/bin/env bash
# =============================================================================
# Re-execute todos cálculos DT-Kinase usando o canônico atual v7+F LEGACY.
# Substitui as métricas atuais (vindas de configs/v7.yaml) pelas do
# canônico melhorado (configs/v7_plus_F.yaml + ADAPTER_LEGACY=1 default).
#
# Pipeline (3 fases):
#
# FASE 1 - Diagonais (3 corpora × 5 seeds = 15 runs)
#   Treina v7+F em cada corpus, salva checkpoints + raw_predictions.npz
#   Output: results/benchmark_plusF_{corpus}_8M/test/level4_cnn_8M/{corpus}/seed_*/
#
# FASE 2 - Off-diagonais (6 pares × 5 seeds = 30 evals)
#   Usa eval_checkpoint_on_dataset.py para avaliar cada checkpoint
#   diagonal nos OUTROS corpora (transferência cross-corpus).
#   Output: results/cross_matrix/dtkinase_v7plusF/{train}_to_{test}/seed_*/metrics.json
#
# FASE 3 - Re-aggregate + regenerar figuras
#   Aponta aggregate.py para os novos paths v7+F.
#   make_slides.py regenera heatmaps + bars com os valores novos.
#
# Tempo estimado em diamante-01 (cuDNN ON, RTX 4090):
#   Fase 1: ~3-4h (NH 50min + H 30min + All 90-120min, 5 seeds cada)
#   Fase 2: ~2-3h (30 evals, ~5min/eval com checkpoint reuse)
#   Fase 3: ~5min
#   TOTAL: ~6-7h
#
# Env overrides:
#   SEEDS         (default: "42 123 456 789 1024")
#   CORPORA       (default: "non_human human all")
#   SKIP_PHASE1   (default: 0; set 1 para pular treino se já rodou)
#   SKIP_PHASE2   (default: 0; set 1 para pular cross-eval)
#   PRIMARY_HOST  (default: diamante-01)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-42 123 456 789 1024}"
CORPORA="${CORPORA:-non_human human all}"
SKIP_PHASE1="${SKIP_PHASE1:-0}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"
SKIP_PHASE3="${SKIP_PHASE3:-0}"
V7_CONFIG="configs/v7_plus_F.yaml"

CROSS_ROOT="results/cross_matrix/dtkinase_v7plusF"
SLIDES_DIR="results/slides/cross_matrix"
SUMMARY_DIR="results/cross_matrix/summary"

hr() { printf '%*s\n' 70 '' | tr ' ' '='; }
step() { hr; echo "[$(date +%H:%M:%S)] $1"; hr; }

# =============================================================================
# Pre-flight
# =============================================================================
step "Pre-flight"

if [ ! -f "${V7_CONFIG}" ]; then
    echo "[fatal] ${V7_CONFIG} não encontrado"; exit 1
fi
if [ ! -f "scripts/v8/run_v7_yaml.sh" ]; then
    echo "[fatal] scripts/v8/run_v7_yaml.sh não encontrado"; exit 1
fi
if [ ! -f "scripts/thesis_followups/eval_checkpoint_on_dataset.py" ]; then
    echo "[fatal] eval_checkpoint_on_dataset.py não encontrado"; exit 1
fi

echo "Config:    ${V7_CONFIG}"
echo "Corpora:   ${CORPORA}"
echo "Seeds:     ${SEEDS}"
echo "SKIP_P1:   ${SKIP_PHASE1}"
echo "SKIP_P2:   ${SKIP_PHASE2}"
echo "SKIP_P3:   ${SKIP_PHASE3}"

# =============================================================================
# FASE 1 - Diagonais (treina v7+F em cada corpus)
# =============================================================================
if [ "${SKIP_PHASE1}" = "1" ]; then
    echo "[skip] FASE 1 (SKIP_PHASE1=1)"
else
    step "FASE 1 - Diagonais (treino v7+F em ${CORPORA})"
    for corpus in ${CORPORA}; do
        echo
        hr
        echo "[$(date +%H:%M:%S)] FASE 1.${corpus^^}"
        hr
        out_dir="results/benchmark_plusF_${corpus}_8M"
        if [ -f "${out_dir}/test/benchmark_comparison.json" ]; then
            n_seeds_done=$(find "${out_dir}/test/level4_cnn_8M/${corpus}" -name "level4_cnn_results.json" 2>/dev/null | wc -l)
            n_seeds_target=$(echo ${SEEDS} | wc -w)
            if [ "${n_seeds_done}" -eq "${n_seeds_target}" ]; then
                echo "  [skip] ${out_dir} já tem ${n_seeds_done}/${n_seeds_target} seeds"
                continue
            fi
            echo "  [partial] ${out_dir} tem ${n_seeds_done}/${n_seeds_target} — re-rodando completo"
            mv "${out_dir}" "${out_dir}_pre_$(date +%Y%m%d_%H%M)" 2>/dev/null || true
        fi

        SEEDS="${SEEDS}" V7_CONFIG="${V7_CONFIG}" CORPUS="${corpus}" \
            BENCHMARK_LEVEL4CNN_DISABLE_CUDNN=0 \
            bash scripts/v8/run_v7_yaml.sh \
            || { echo "[fatal] FASE 1.${corpus} falhou"; exit 2; }

        echo "[done] FASE 1.${corpus} → ${out_dir}/test/"
    done
fi

# =============================================================================
# FASE 2 - Off-diagonais (eval cross-corpus dos checkpoints)
# =============================================================================
if [ "${SKIP_PHASE2}" = "1" ]; then
    echo "[skip] FASE 2 (SKIP_PHASE2=1)"
else
    step "FASE 2 - Off-diagonais cross-corpus"

    mkdir -p "${CROSS_ROOT}"

    for train_corpus in ${CORPORA}; do
        ckpt_root="results/benchmark_plusF_${train_corpus}_8M/test/level4_cnn_8M/${train_corpus}"
        if [ ! -d "${ckpt_root}" ]; then
            echo "  [skip] ${ckpt_root} não existe (FASE 1 ${train_corpus} falhou?)"
            continue
        fi

        for test_corpus in ${CORPORA}; do
            if [ "${train_corpus}" = "${test_corpus}" ]; then
                continue  # diagonal já existe da FASE 1
            fi
            pair="${train_corpus}_to_${test_corpus}"
            out_pair_dir="${CROSS_ROOT}/${pair}"
            mkdir -p "${out_pair_dir}"

            echo
            echo "  --- ${pair} ---"

            for seed in ${SEEDS}; do
                ckpt="${ckpt_root}/seed_${seed}/level4_cnn_model.pt"
                out_seed_dir="${out_pair_dir}/seed_${seed}"
                metrics="${out_seed_dir}/metrics.json"

                if [ -f "${metrics}" ]; then
                    echo "    [skip] seed=${seed} já tem ${metrics}"
                    continue
                fi
                if [ ! -f "${ckpt}" ]; then
                    echo "    [missing] seed=${seed} ckpt: ${ckpt}"
                    continue
                fi
                mkdir -p "${out_seed_dir}"

                python3 scripts/thesis_followups/eval_checkpoint_on_dataset.py \
                    --checkpoint "${ckpt}" \
                    --train-corpus "${train_corpus}" \
                    --eval-dataset "${test_corpus}" \
                    --output-dir "${out_seed_dir}" \
                    --seed "${seed}" \
                    || { echo "    [warn] seed=${seed} falhou — continuando"; continue; }

                echo "    [ok] seed=${seed}"
            done
        done
    done
fi

# =============================================================================
# FASE 3 - Re-aggregate + regen figuras
# =============================================================================
if [ "${SKIP_PHASE3}" = "1" ]; then
    echo "[skip] FASE 3 (SKIP_PHASE3=1)"
else
    step "FASE 3 - Re-aggregate + regenerar figuras"

    # Aggregate.py args: pega novas diagonais v7+F + cross_matrix de outros
    # modelos (DrugBAN, GraphBAN, ConPLex sem mudança).
    DTK_DIAG_ARGS=""
    for c in ${CORPORA}; do
        d="results/benchmark_plusF_${c}_8M/test/level4_cnn_8M/${c}"
        if [ -d "${d}" ]; then
            DTK_DIAG_ARGS="${DTK_DIAG_ARGS} --diagonal-dtkinase-${c} ${d}"
        fi
    done

    # IMPORTANT: aggregate.py espera resultados em results/cross_matrix/{model}/.
    # Para v7+F, as off-diagonais foram salvas em results/cross_matrix/dtkinase_v7plusF/.
    # Precisa renomear/copiar para que aggregate.py veja (ou renomear o root).
    # Aqui: cria symlink temporário results/cross_matrix/dtkinase → dtkinase_v7plusF
    # PRESERVANDO a v7 original em results/cross_matrix/dtkinase_v7original/.
    if [ -d "results/cross_matrix/dtkinase" ] && [ ! -L "results/cross_matrix/dtkinase" ]; then
        if [ ! -d "results/cross_matrix/dtkinase_v7original" ]; then
            mv results/cross_matrix/dtkinase results/cross_matrix/dtkinase_v7original
            echo "  [backup] results/cross_matrix/dtkinase → dtkinase_v7original"
        fi
    fi
    [ -L "results/cross_matrix/dtkinase" ] && rm results/cross_matrix/dtkinase
    if [ -d "${CROSS_ROOT}" ]; then
        ln -s "$(basename ${CROSS_ROOT})" results/cross_matrix/dtkinase
        echo "  [link] results/cross_matrix/dtkinase → ${CROSS_ROOT}"
    fi

    python3 scripts/thesis_followups/cross_dataset_matrix/aggregate.py \
        ${DTK_DIAG_ARGS} \
        --diagonal-drugban-human        results/semantic-screening-results/drugban/human \
        --diagonal-drugban-non_human    results/semantic-screening-results/drugban/non_human \
        --diagonal-drugban-all          results/semantic-screening-results/drugban/all \
        --diagonal-graphban-human       results/semantic-screening-results/graphban/human \
        --diagonal-graphban-non_human   results/semantic-screening-results/graphban/non_human \
        --diagonal-graphban-all         results/semantic-screening-results/graphban/all \
        --diagonal-conplex-human        ConPLex/results_universal/human \
        --diagonal-conplex-non_human    ConPLex/results_universal/non_human \
        --diagonal-conplex-all          ConPLex/results_universal/all \
        --results-root results/cross_matrix \
        --out-dir "${SUMMARY_DIR}" \
        || { echo "[fatal] aggregate.py falhou"; exit 4; }

    python3 scripts/thesis_followups/cross_dataset_matrix/make_slides.py \
        --json "${SUMMARY_DIR}/cross_matrix.json" \
        --out-dir "${SLIDES_DIR}" \
        || { echo "[fatal] make_slides.py falhou"; exit 5; }

    echo
    echo "  Figuras geradas em ${SLIDES_DIR}/figures/"
    ls -la "${SLIDES_DIR}/figures/"*.pdf 2>/dev/null | awk '{print "    "$NF}'
fi

hr
echo "[$(date +%H:%M:%S)] PIPELINE COMPLETO."
echo "Próximo: copiar PDFs para ~/PhD/figures/ e recompilar tese."
hr
