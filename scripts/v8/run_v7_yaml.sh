#!/usr/bin/env bash
# =============================================================================
# Run canonical configs/v7.yaml via run_from_config.py on diamante-02.
#
# Forces fp32 + cuDNN off via env overrides (diamante-02 driver 12.4 + cuDNN
# 9.x ABI mismatch + user directive: fp32 default for v7). TF32 disabled
# automatically in level4_cnn.py when pure_fp32 path is active.
#
# Env overrides:
#   CORPUS        default: non_human
#   V8ENV         default: v8env
#   EXTRA_ARGS    passed verbatim to run_from_config.py (e.g. "--dry-run")
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-non_human}"
# Default: repo venv at ./env (setup.py). v7/v7+/v7_ban only need v7
# dependencies — no ChemBERTa/BioBERT modules from v8 POC. Override
# with V8ENV=v8env or other conda env as needed.
V8ENV="${V8ENV:-env}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Config path — default v7_plus_F.yaml (CANÔNICO MCC primário, LEGACY adapter,
# 0.5266 ± 0.010 NH 3-seed em diamante-01). Override pra outras variantes:
#   V7_CONFIG=configs/v7.yaml         # baseline original (referência tese)
#   V7_CONFIG=configs/v7_ban.yaml     # BAN (bilinear attention) variant
#   V7_CONFIG=configs/v7_plus.yaml    # A+C apenas (sem F)
V7_CONFIG="${V7_CONFIG:-configs/v7_plus_F.yaml}"
# Single seed default for fast iteration. Expand to "42 123 456 789 1024"
# for full thesis multi-seed protocol once config is validated.
export SEEDS="${SEEDS:-42}"

activate_env() {
    local env="$1"
    set +u
    if [ -f "${REPO_ROOT}/${env}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${REPO_ROOT}/${env}/bin/activate"
    elif command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${env}" || { echo "[fatal] cannot activate '${env}'" >&2; exit 1; }
    else
        echo "[fatal] no venv at ${env} and no conda" >&2
        exit 1
    fi
    set -u
}

N_CPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${N_CPU}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${N_CPU}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${N_CPU}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${N_CPU}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Force fp32 + cuDNN off (diamante-02 constraints). NO_AMP/DOUBLE are
# yaml-controlled (run_from_config.py clobbers external env from yaml)
# so these exports only matter for DISABLE_CUDNN and NO_COMPILE which
# have no yaml mapping. Keep the DOUBLE/NO_AMP exports for documentation.
export BENCHMARK_LEVEL4CNN_DOUBLE="${BENCHMARK_LEVEL4CNN_DOUBLE:-0}"
export BENCHMARK_LEVEL4CNN_NO_AMP="${BENCHMARK_LEVEL4CNN_NO_AMP:-1}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-1}"
# torch.compile produces dynamo symbolic_shapes warnings on dynamic cross-
# attention maps + recompiles frequently. Disable by default.
export BENCHMARK_LEVEL4CNN_NO_COMPILE="${BENCHMARK_LEVEL4CNN_NO_COMPILE:-1}"

echo "=============================================================="
echo "run_from_config.py"
echo "  config:    ${V7_CONFIG}"
echo "  corpus:    ${CORPUS}"
echo "  env:       ${V8ENV}"
echo "  seeds:     ${SEEDS}"
echo "  overrides: DOUBLE=${BENCHMARK_LEVEL4CNN_DOUBLE} NO_AMP=${BENCHMARK_LEVEL4CNN_NO_AMP} DISABLE_CUDNN=${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN}"
echo "=============================================================="

(
    activate_env "${V8ENV}"
    python3 run_from_config.py "${V7_CONFIG}" --dataset "${CORPUS}" ${EXTRA_ARGS}
)
