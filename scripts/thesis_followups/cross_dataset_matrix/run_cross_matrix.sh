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
# Each model runs in its own conda env (activated inside a subshell) so the
# surrounding shell is never polluted:
#     dtkinase  -> conda activate ${V7_ENV}
#     drugban   -> conda activate ${DRUGBAN_ENV}
#     graphban  -> conda activate ${GRAPHBAN_ENV}
#     conplex   -> conda activate ${CONPLEX_ENV}
#
# Env overrides:
#   SEEDS             seeds to evaluate (default: 42 123 456 789 1024)
#   MODELS            models to run (default: dtkinase drugban graphban conplex)
#   OUT_ROOT          output root (default: results/cross_matrix)
#   V7_CKPT_{HUMAN,NON_HUMAN,ALL}  per-corpus v7 checkpoint dirs (defaults below)
#   V7_ENV            conda env for dtkinase (default: docktkinase)
#   DRUGBAN_ENV       conda env for drugban (default: drugban)
#   GRAPHBAN_ENV      conda env for graphban (default: graphban)
#   CONPLEX_ENV       conda env for conplex (default: conplex)
#   CONDA_BASE        conda install root (autodetected via `conda info --base`)
#   SKIP_LEAKAGE_FILTER  set to 1 to skip filter (uses unmodified test TSV)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS=(${SEEDS:-42 123 456 789 1024})
MODELS=(${MODELS:-dtkinase drugban graphban conplex})
OUT_ROOT="${OUT_ROOT:-results/cross_matrix}"

# DT-Kinase v7 checkpoint paths. Default points at the consolidated
# `semantic-screening-results` bundle (identical in diamante-02 and
# local checkouts), which contains the canonical thesis checkpoints.
# Path ends at .../{corpus}; seed_{s}/level4_cnn_model.pt is appended
# internally. Override with V7_CKPT_{HUMAN,NON_HUMAN,ALL} if you want
# to use the original date-stamped training dirs instead.
V7_CKPT_HUMAN="${V7_CKPT_HUMAN:-results/semantic-screening-results/dt-kinase/benchmark_human_8M_01_04_2026/test/level4_cnn_8M/human}"
V7_CKPT_NON_HUMAN="${V7_CKPT_NON_HUMAN:-results/semantic-screening-results/dt-kinase/benchmark_non_human_8M_13_05_2026/test/level4_cnn_8M/non_human}"
V7_CKPT_ALL="${V7_CKPT_ALL:-results/semantic-screening-results/dt-kinase/benchmark_all_8M_13_04_2026/test/level4_cnn_8M/all}"

V7_ENV="${V7_ENV:-env}"  # DT-Kinase: venv (env/bin/activate) or conda env name
DRUGBAN_ENV="${DRUGBAN_ENV:-drugban}"
GRAPHBAN_ENV="${GRAPHBAN_ENV:-graphban}"
CONPLEX_ENV="${CONPLEX_ENV:-conplex}"
SKIP_LEAKAGE_FILTER="${SKIP_LEAKAGE_FILTER:-0}"

SPLITS_DIR="scaffolds_splits/output"

# ----------------------------------------------------------------------------
# Locate conda and load its shell integration. Each run_* function sources
# this file inside its subshell so `conda activate` works.
# ----------------------------------------------------------------------------
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo)"
fi
if [ -z "${CONDA_BASE}" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[fatal] conda not found. Set CONDA_BASE env var to your conda install root." >&2
    exit 1
fi
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

activate() {
    # Usage: activate <env>  (inside a subshell).
    # If <env> is a path to a venv (has bin/activate), source it;
    # else treat as conda env name. MKL activate scripts in some conda envs
    # reference MKL_INTERFACE_LAYER under `set -u` — disable nounset while
    # sourcing / activating, then restore.
    local env="$1"
    set +u
    if [ -f "${REPO_ROOT}/${env}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/${env}/bin/activate"
    elif [ -f "${env}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${env}/bin/activate"
    else
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
        conda activate "${env}" || {
            echo "[fatal] cannot activate env '${env}' (not a venv path, not a conda env name)" >&2
            exit 1
        }
    fi
    set -u
}

# ----------------------------------------------------------------------------
# Pre-flight: prune models whose upstream source is missing. Baselines
# vendor upstream under <Model>/src (gitignored). On hosts that did not
# run <Model>/setup_env.sh, infer_<model>_universal.py fatal-exits at
# import time. Detect and skip silently instead of repeating the fatal
# once per (pair × seed).
# ----------------------------------------------------------------------------
REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
_pruned=()
for m in "${MODELS[@]}"; do
    case "${m}" in
        drugban)
            if [ ! -f "${REPO_ROOT_ABS}/DrugBAN/src/dataloader.py" ]; then
                echo "[preflight] DrugBAN upstream missing at DrugBAN/src/ — skipping model 'drugban'"
                echo "            install via: bash DrugBAN/setup_env.sh"
                continue
            fi
            ;;
        graphban)
            if [ ! -f "${REPO_ROOT_ABS}/GraphBAN/src/dataloader.py" ] \
               && [ ! -f "${REPO_ROOT_ABS}/GraphBAN/src/case_study/dataloader.py" ]; then
                echo "[preflight] GraphBAN upstream missing at GraphBAN/src/ — skipping model 'graphban'"
                continue
            fi
            ;;
    esac
    _pruned+=("${m}")
done
MODELS=("${_pruned[@]}")

echo "=============================================================="
echo "Cross-dataset matrix"
echo "  CONDA_BASE:    ${CONDA_BASE}"
echo "  V7_ENV:        ${V7_ENV}"
echo "  DRUGBAN_ENV:   ${DRUGBAN_ENV}"
echo "  GRAPHBAN_ENV:  ${GRAPHBAN_ENV}"
echo "  CONPLEX_ENV:   ${CONPLEX_ENV}"
echo "  MODELS:        ${MODELS[*]}"
echo "  SEEDS:         ${SEEDS[*]}"
echo "  OUT_ROOT:      ${OUT_ROOT}"
echo "=============================================================="

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
corpus_stem() {
    case "$1" in
        human)     echo "human" ;;
        non_human) echo "non_human" ;;
        all)       echo "universal" ;;
    esac
}

v7_ckpt_root_for() {
    case "$1" in
        human)     echo "${V7_CKPT_HUMAN}" ;;
        non_human) echo "${V7_CKPT_NON_HUMAN}" ;;
        all)       echo "${V7_CKPT_ALL}" ;;
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

run_filter() {
    local train="$1" test="$2" out_dir="$3"
    mkdir -p "${out_dir}"
    if [ "${SKIP_LEAKAGE_FILTER}" = "1" ]; then
        local stem; stem="$(corpus_stem "${test}")"
        cp "${SPLITS_DIR}/${stem}_test.tsv" "${out_dir}/test_clean.tsv"
        echo "{\"train_corpus\":\"${train}\",\"test_corpus\":\"${test}\",\"skipped\":true,\"out_tsv\":\"${out_dir}/test_clean.tsv\"}" > "${out_dir}/leakage_report.json"
    else
        # leakage_filter.py runs in V7_ENV (rdkit + pandas dependency)
        (
            activate "${V7_ENV}"
            python3 "${SCRIPT_DIR}/leakage_filter.py" \
                --train-corpus "${train}" --test-corpus "${test}" --out-dir "${out_dir}"
        )
    fi
}

run_dtkinase() {
    local train="$1" test="$2" seed="$3" out="$4"
    mkdir -p "${out}"
    local root; root="$(v7_ckpt_root_for "${train}")"
    local ckpt="${root}/seed_${seed}/level4_cnn_model.pt"
    if [ ! -f "${ckpt}" ]; then
        echo "[dtkinase ${train}->${test} s=${seed}] checkpoint missing: ${ckpt}"
        return
    fi
    if [ -f "${out}/metrics.json" ]; then
        echo "[dtkinase ${train}->${test} s=${seed}] already done"
        return
    fi
    (
        activate "${V7_ENV}"
        python3 scripts/thesis_followups/eval_checkpoint_on_dataset.py \
            --checkpoint "${ckpt}" \
            --train-corpus "${train}" \
            --eval-dataset "${test}" \
            --split test \
            --output "${out}/metrics.json" \
            --seed "${seed}"
    ) 2>&1 | tee "${out}/eval.log"
}

run_drugban() {
    local train="$1" test="$2" seed="$3" out="$4" test_tsv="$5"
    mkdir -p "${out}"
    if [ -f "${out}/raw_predictions.npz" ]; then
        echo "[drugban ${train}->${test} s=${seed}] already done"
        return
    fi
    (
        activate "${DRUGBAN_ENV}"
        python infer_drugban_universal.py \
            --corpus "${train}" \
            --seeds "${seed}" \
            --test-tsv "${test_tsv}" \
            --output-dir "${out%/seed_${seed}}"
    ) 2>&1 | tee "${out}/eval.log"
}

run_graphban() {
    local train="$1" test="$2" seed="$3" out="$4" test_tsv="$5"
    mkdir -p "${out}"
    if [ -f "${out}/raw_predictions.npz" ]; then
        echo "[graphban ${train}->${test} s=${seed}] already done"
        return
    fi
    (
        activate "${GRAPHBAN_ENV}"
        python infer_graphban_universal.py \
            --corpus "${train}" \
            --seeds "${seed}" \
            --test-tsv "${test_tsv}" \
            --output-dir "${out%/seed_${seed}}"
    ) 2>&1 | tee "${out}/eval.log"
}

run_conplex() {
    local train="$1" test="$2" seed="$3" out="$4" test_tsv="$5"
    mkdir -p "${out}"
    if [ -f "${out}/raw_predictions.npz" ]; then
        echo "[conplex ${train}->${test} s=${seed}] already done"
        return
    fi
    (
        activate "${CONPLEX_ENV}"
        python infer_conplex_universal.py \
            --corpus "${train}" \
            --seeds "${seed}" \
            --test-tsv "${test_tsv}" \
            --output-dir "${out%/seed_${seed}}"
    ) 2>&1 | tee "${out}/eval.log"
}

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
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
