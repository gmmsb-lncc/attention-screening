#!/usr/bin/env bash
# =============================================================================
# Cross-dataset evaluation matrix (Human x Non-Human x All).
#
# Runs zero-shot inference of each trained model on the 6 off-diagonal cells
# of the 3x3 (train, test) matrix. Checkpoints from the diagonal runs are
# reused — no retraining.
#
# Cells (train -> test):
#     human      -> non_human, all
#     non_human  -> human, all
#     all        -> human, non_human
#
# Per cell, per model, per seed: produces raw_predictions.npz + metrics.json
# under results/cross_matrix/{model}/{train}_to_{test}/seed_{s}/.
#
# Prerequisites:
#   - DT-Kinase v7 diagonal checkpoints under V7_CKPT_ROOT/{corpus}/seed_{s}/level4_cnn_model.pt
#   - DrugBAN diagonal checkpoints under DrugBAN/results/{corpus}/seed_{s}/
#   - GraphBAN diagonal checkpoints under GraphBAN/results/{corpus}/seed_{s}/
#   - ConPLex diagonal checkpoints under ConPLex/best_models/trained_{corpus}_rep{idx}/
#
# Env overrides:
#   SEEDS             seeds to evaluate (default: 42 123 456 789 1024)
#   MODELS            models to run (default: dtkinase drugban graphban conplex)
#   OUT_ROOT          output root (default: results/cross_matrix)
#   V7_CKPT_ROOT      root of v7 diagonal checkpoints (required for dtkinase)
#   V7_ENV            env to activate for dtkinase + python utilities (default: env)
#   DRUGBAN_ENV       conda env for drugban (default: drugban)
#   GRAPHBAN_ENV      conda env for graphban (default: graphban)
#   CONPLEX_ENV       conda env for conplex (default: conplex)
#   SKIP_LEAKAGE_FILTER  set to 1 to skip filter (uses unmodified test TSV)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(${SEEDS:-42 123 456 789 1024})
MODELS=(${MODELS:-dtkinase drugban graphban conplex})
OUT_ROOT="${OUT_ROOT:-results/cross_matrix}"
V7_CKPT_ROOT="${V7_CKPT_ROOT:-results/v7_diagonal}"
V7_ENV="${V7_ENV:-env}"
DRUGBAN_ENV="${DRUGBAN_ENV:-drugban}"
GRAPHBAN_ENV="${GRAPHBAN_ENV:-graphban}"
CONPLEX_ENV="${CONPLEX_ENV:-conplex}"
SKIP_LEAKAGE_FILTER="${SKIP_LEAKAGE_FILTER:-0}"

SPLITS_DIR="scaffolds_splits/output"

corpus_stem() {
    case "$1" in
        human)     echo "human" ;;
        non_human) echo "non_human" ;;
        all)       echo "universal" ;;
    esac
}

pairs=(
    "human non_human"
    "human all"
    "non_human human"
    "non_human all"
    "all human"
    "all non_human"
)

activate_env() {
    # shellcheck disable=SC1091
    if [ -f "${V7_ENV}/bin/activate" ]; then
        source "${V7_ENV}/bin/activate"
    else
        conda activate "$1" 2>/dev/null || echo "[warn] cannot activate env '$1'"
    fi
}

run_filter() {
    local train="$1" test="$2" out_dir="$3"
    mkdir -p "${out_dir}"
    if [ "${SKIP_LEAKAGE_FILTER}" = "1" ]; then
        local stem; stem="$(corpus_stem "${test}")"
        cp "${SPLITS_DIR}/${stem}_test.tsv" "${out_dir}/test_clean.tsv"
        echo "{\"train_corpus\":\"${train}\",\"test_corpus\":\"${test}\",\"skipped\":true,\"out_tsv\":\"${out_dir}/test_clean.tsv\"}" > "${out_dir}/leakage_report.json"
    else
        python3 "${SCRIPT_DIR}/leakage_filter.py" \
            --train-corpus "${train}" --test-corpus "${test}" --out-dir "${out_dir}"
    fi
}

run_dtkinase() {
    local train="$1" test="$2" seed="$3" out="$4"
    mkdir -p "${out}"
    local ckpt="${V7_CKPT_ROOT}/${train}/seed_${seed}/level4_cnn_model.pt"
    if [ ! -f "${ckpt}" ]; then
        echo "[dtkinase ${train}->${test} s=${seed}] checkpoint missing: ${ckpt}"
        return
    fi
    if [ -f "${out}/metrics.json" ]; then
        echo "[dtkinase ${train}->${test} s=${seed}] already done"
        return
    fi
    python3 scripts/thesis_followups/eval_checkpoint_on_dataset.py \
        --checkpoint "${ckpt}" \
        --train-corpus "${train}" \
        --eval-dataset "${test}" \
        --split test \
        --output "${out}/metrics.json" \
        --seed "${seed}" 2>&1 | tee "${out}/eval.log"
}

run_drugban() {
    local train="$1" test="$2" seed="$3" out="$4" test_tsv="$5"
    mkdir -p "${out}"
    if [ -f "${out}/raw_predictions.npz" ]; then
        echo "[drugban ${train}->${test} s=${seed}] already done"
        return
    fi
    conda run -n "${DRUGBAN_ENV}" --no-capture-output \
        python infer_drugban_universal.py \
            --corpus "${train}" \
            --seeds "${seed}" \
            --test-tsv "${test_tsv}" \
            --output-dir "${out%/seed_${seed}}" 2>&1 | tee "${out}/eval.log"
}

run_graphban() {
    local train="$1" test="$2" seed="$3" out="$4" test_tsv="$5"
    mkdir -p "${out}"
    if [ -f "${out}/raw_predictions.npz" ]; then
        echo "[graphban ${train}->${test} s=${seed}] already done"
        return
    fi
    conda run -n "${GRAPHBAN_ENV}" --no-capture-output \
        python infer_graphban_universal.py \
            --corpus "${train}" \
            --seeds "${seed}" \
            --test-tsv "${test_tsv}" \
            --output-dir "${out%/seed_${seed}}" 2>&1 | tee "${out}/eval.log"
}

run_conplex() {
    local train="$1" test="$2" seed="$3" out="$4" test_tsv="$5"
    mkdir -p "${out}"
    if [ -f "${out}/raw_predictions.npz" ]; then
        echo "[conplex ${train}->${test} s=${seed}] already done"
        return
    fi
    conda run -n "${CONPLEX_ENV}" --no-capture-output \
        python infer_conplex_universal.py \
            --corpus "${train}" \
            --seeds "${seed}" \
            --test-tsv "${test_tsv}" \
            --output-dir "${out%/seed_${seed}}" 2>&1 | tee "${out}/eval.log"
}

for pair in "${pairs[@]}"; do
    read -r TRAIN TEST <<<"${pair}"
    label="${TRAIN}_to_${TEST}"
    echo
    echo "================================================================"
    echo "  ${label}"
    echo "================================================================"

    filter_dir="${OUT_ROOT}/filters/${label}"
    run_filter "${TRAIN}" "${TEST}" "${filter_dir}"
    test_tsv="${filter_dir}/test_clean.tsv"

    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            seed_out="${OUT_ROOT}/${model}/${label}/seed_${seed}"
            case "${model}" in
                dtkinase) run_dtkinase "${TRAIN}" "${TEST}" "${seed}" "${seed_out}" ;;
                drugban)  run_drugban  "${TRAIN}" "${TEST}" "${seed}" "${seed_out}" "${test_tsv}" ;;
                graphban) run_graphban "${TRAIN}" "${TEST}" "${seed}" "${seed_out}" "${test_tsv}" ;;
                conplex)  run_conplex  "${TRAIN}" "${TEST}" "${seed}" "${seed_out}" "${test_tsv}" ;;
                *) echo "[skip] unknown model: ${model}" ;;
            esac
        done
    done
done

echo
echo "Done. Aggregate with:"
echo "  python3 ${SCRIPT_DIR}/aggregate.py --results-root ${OUT_ROOT} --out-dir ${OUT_ROOT}/summary"
