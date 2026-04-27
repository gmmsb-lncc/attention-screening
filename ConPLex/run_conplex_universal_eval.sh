#!/usr/bin/env bash
# =============================================================================
# ConPLex cross-dataset inference on the universal test split.
#
# For each training corpus in {non_human, human, all}, runs 5-seed inference
# of ConPLex trained on that corpus, evaluating on
#   scaffolds_splits/output/universal_test.tsv  (41 441 rows).
#
# Val probs come from the *training corpus* val split (default behavior of
# scripts/inference/legacy/infer_conplex_universal.py) so that any downstream threshold calibration
# stays pinned to the training distribution.
#
# Outputs:
#   ConPLex/results_crossuniversal/{corpus}_on_universal/{corpus}/seed_{s}/raw_predictions.npz
#
# Pushes GPU + CPU utilization:
#   - auto-detects GPU VRAM; picks BATCH_SIZE (24GB: 4096, 12GB: 2048, 8GB: 1024)
#   - pre-featurizes val+test ONCE per corpus (amortizes ProtBert over 5 seeds)
#   - threads set to all visible cores
#   - AMP enabled on CUDA
#   - ProtBertFeaturizer moved to CUDA when supported
#
# Env overrides (all optional):
#   CORPORA       corpora to process (default: "non_human human all")
#   SEEDS         seeds (default: "42 123 456 789 1024")
#   BATCH_SIZE    inference batch (default: auto from VRAM)
#   CONPLEX_ENV   conda env name (default: conplex)
#   OUT_ROOT      output dir (default: ConPLex/results_crossuniversal)
#   UNIVERSAL_TSV test TSV path (default: scaffolds_splits/output/universal_test.tsv)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CORPORA="${CORPORA:-non_human human all}"
SEEDS="${SEEDS:-42 123 456 789 1024}"
CONPLEX_ENV="${CONPLEX_ENV:-conplex}"
OUT_ROOT="${OUT_ROOT:-ConPLex/results_crossuniversal}"
UNIVERSAL_TSV="${UNIVERSAL_TSV:-scaffolds_splits/output/universal_test.tsv}"

# ----------------------------------------------------------------------------
# GPU / CPU detection
# ----------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
else
    VRAM_MB=0
    GPU_NAME="cpu"
fi

if [ -z "${BATCH_SIZE:-}" ]; then
    if   [ "${VRAM_MB}" -ge 40000 ]; then BATCH_SIZE=8192
    elif [ "${VRAM_MB}" -ge 22000 ]; then BATCH_SIZE=4096
    elif [ "${VRAM_MB}" -ge 11000 ]; then BATCH_SIZE=2048
    elif [ "${VRAM_MB}" -ge  7000 ]; then BATCH_SIZE=1024
    else                                  BATCH_SIZE=512
    fi
fi

N_CPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

# Thread knobs for NumPy / PyTorch / HuggingFace tokenizer
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${N_CPU}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${N_CPU}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${N_CPU}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${N_CPU}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${N_CPU}}"
# Avoid HF Transformers warning and allow async CUDA kernels
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=============================================================="
echo "ConPLex cross-universal inference"
echo "  GPU:         ${GPU_NAME} (${VRAM_MB} MB VRAM)"
echo "  CPUs:        ${N_CPU} logical"
echo "  BATCH_SIZE:  ${BATCH_SIZE}"
echo "  CORPORA:     ${CORPORA}"
echo "  SEEDS:       ${SEEDS}"
echo "  TEST TSV:    ${UNIVERSAL_TSV}"
echo "  OUT_ROOT:    ${OUT_ROOT}"
echo "=============================================================="

if [ ! -f "${UNIVERSAL_TSV}" ]; then
    echo "[fatal] missing universal test TSV: ${UNIVERSAL_TSV}"
    exit 1
fi

# ----------------------------------------------------------------------------
# Activate ConPLex env
# ----------------------------------------------------------------------------
# shellcheck disable=SC1091
if command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONPLEX_ENV}" || {
            echo "[fatal] could not activate conda env '${CONPLEX_ENV}'"; exit 1;
        }
    fi
fi

python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}')" || {
    echo "[fatal] torch not available in env '${CONPLEX_ENV}'"; exit 1;
}

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
mkdir -p "${OUT_ROOT}"
LOG_ROOT="${OUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

total_start=$(date +%s)

for corpus in ${CORPORA}; do
    label="${corpus}_on_universal"
    out_dir="${OUT_ROOT}/${label}"
    log_file="${LOG_ROOT}/${label}.log"

    echo
    echo "---- ${label} ----"

    corpus_start=$(date +%s)
    # Pre-featurization happens once per corpus (all 5 seeds share the same
    # val+test embeddings — ProtBert is the expensive step).
    python scripts/inference/legacy/infer_conplex_universal.py \
        --corpus "${corpus}" \
        --seeds ${SEEDS} \
        --test-tsv "${UNIVERSAL_TSV}" \
        --batch-size "${BATCH_SIZE}" \
        --output-dir "${out_dir}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    corpus_end=$(date +%s)

    if [ "${status}" -ne 0 ]; then
        echo "[warn] ${label} exited with status ${status}"
    fi
    echo "  elapsed: $((corpus_end - corpus_start))s   →  ${out_dir}"
done

total_end=$(date +%s)
echo
echo "=============================================================="
echo "Total elapsed: $((total_end - total_start))s"
echo
echo "Summary of produced files:"
find "${OUT_ROOT}" -name raw_predictions.npz -printf '  %p  (%sB)\n' 2>/dev/null \
    || find "${OUT_ROOT}" -name raw_predictions.npz -exec ls -l {} \;

echo
echo "Next step — aggregate:"
echo "  python3 scripts/thesis_followups/cross_dataset_matrix/aggregate.py \\"
echo "      --results-root ${OUT_ROOT} \\"
echo "      --out-dir ${OUT_ROOT}/summary"
