#!/usr/bin/env bash
# =============================================================================
# v7-baseline runner in v8 env — apples-to-apples control.
#
# Invokes train_v8.py with ablation=v7-baseline: all v8 feature gates OFF,
# so InteractionMapCNNv8 degrades to vanilla InteractionMapCNN (v7). Uses
# the exact same env (cuDNN off, no_amp: true, LR 1e-3, float32) as the
# v8 runs → direct comparison.
#
# Purpose: if v7-baseline recovers MCC ~= 0.506 (v7 reference), v8 collapse
# is due to the injected features. If v7-baseline also collapses, problem
# is in the shared training env on diamante-02.
#
# Env overrides:
#   CORPUS        default: non_human
#   SEEDS         default: 42
#   OUT_ROOT      default: results/v8
#   V8ENV         default: v8env
#   V8_BASE_LR    default: 1e-3 (inherited by train_v8)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-non_human}"
SEEDS=(${SEEDS:-42})
OUT_ROOT="${OUT_ROOT:-results/v8}"
V8ENV="${V8ENV:-v8env}"

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

echo "=============================================================="
echo "v7-baseline control run (v8 infra, all v8 features OFF)"
echo "  corpus:   ${CORPUS}"
echo "  seeds:    ${SEEDS[*]}"
echo "  out_root: ${OUT_ROOT}"
echo "  env:      ${V8ENV}"
echo "=============================================================="

for seed in "${SEEDS[@]}"; do
    out_dir="${OUT_ROOT}/${CORPUS}/v7-baseline/seed_${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[skip] v7-baseline/${CORPUS}/seed_${seed} already done"
        continue
    fi
    mkdir -p "${out_dir}"
    echo
    echo "================================================================"
    echo "  train v7-baseline/${CORPUS}/seed_${seed}"
    echo "================================================================"
    (
        activate_env "${V8ENV}"
        python3 scripts/v8/train_v8.py \
            --config configs/v8.yaml \
            --dataset "${CORPUS}" \
            --seed "${seed}" \
            --ablation v7-baseline \
            --output-dir "${out_dir}"
    ) 2>&1 | tee "${out_dir}/train.log"
done

echo
echo "[done] v7-baseline complete → ${OUT_ROOT}/${CORPUS}/v7-baseline/"
python3 -c "
import json, glob, statistics as st
paths = sorted(glob.glob('${OUT_ROOT}/${CORPUS}/v7-baseline/seed_*/metrics.json'))
if not paths:
    print('  v7-baseline: no results'); exit()
mccs = [json.load(open(p))['test_mcc'] for p in paths]
aurocs = [json.load(open(p))['test_auroc'] for p in paths]
n = len(mccs)
print(f'  v7-baseline MCC={st.mean(mccs):.4f}±{st.stdev(mccs) if n>1 else 0.0:.4f}  '
      f'AUROC={st.mean(aurocs):.4f}  (n={n})')
print(f'  v7 reference (NH): 0.506  →  {\"MATCH\" if st.mean(mccs) >= 0.45 else \"DIVERGE — env problem\"}')
"
