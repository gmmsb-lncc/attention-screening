#!/usr/bin/env bash
# =============================================================================
# v7-side: v7 + per-token side representations (ChemBERTa + BioBERT).
#
# Architecture: MoLFormer + ESM-2 backbones unchanged. ChemBERTa-77M (ligand)
# and BioBERT (protein) injected alongside as per-token matrices (L_c, 768)
# and (L_b, 320), concatenated on the sequence dim before the EmbeddingAdapter.
# Combined sequence flows through the full v7 pipeline (adapter → cross-attn
# 2D map → CNN → hierarchical pool → classifier) with no other changes.
#
# Env overrides:
#   CORPUS        default: non_human
#   SEEDS         default: "42" (single seed for fast iteration)
#   OUT_ROOT      default: results/v7_side
#   V8ENV         default: v8env
#   V8_BASE_LR    default: 1e-4 (matches v7 canonical LR; avoid 1e-3 which
#                 memorized on v8)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-non_human}"
SEEDS=(${SEEDS:-42})
OUT_ROOT="${OUT_ROOT:-results/v7_side}"
V8ENV="${V8ENV:-v8env}"
export V8_BASE_LR="${V8_BASE_LR:-1e-4}"
export V8_DISABLE_CUDNN="${V8_DISABLE_CUDNN:-1}"

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
echo "v7-side: per-token ChemBERTa + BioBERT alongside MoLFormer + ESM-2"
echo "  corpus:   ${CORPUS}"
echo "  seeds:    ${SEEDS[*]}"
echo "  out_root: ${OUT_ROOT}"
echo "  env:      ${V8ENV}"
echo "  base LR:  ${V8_BASE_LR}"
echo "=============================================================="

for seed in "${SEEDS[@]}"; do
    out_dir="${OUT_ROOT}/${CORPUS}/seed_${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[skip] seed_${seed} already done"
        continue
    fi
    mkdir -p "${out_dir}"
    echo
    echo "================================================================"
    echo "  train v7-side/${CORPUS}/seed_${seed}"
    echo "================================================================"
    (
        activate_env "${V8ENV}"
        python3 scripts/v8/train_v8.py \
            --config configs/v7_side.yaml \
            --dataset "${CORPUS}" \
            --seed "${seed}" \
            --ablation v7-side \
            --output-dir "${out_dir}"
    ) 2>&1 | tee "${out_dir}/train.log"
done

echo
echo "[done] v7-side complete → ${OUT_ROOT}/${CORPUS}/"
python3 -c "
import json, glob, statistics as st
paths = sorted(glob.glob('${OUT_ROOT}/${CORPUS}/seed_*/metrics.json'))
if not paths:
    print('  v7-side: no results'); exit()
mccs = [json.load(open(p))['test_mcc'] for p in paths]
aurocs = [json.load(open(p))['test_auroc'] for p in paths]
n = len(mccs)
print(f'  v7-side MCC={st.mean(mccs):.4f}±{st.stdev(mccs) if n>1 else 0.0:.4f}  '
      f'AUROC={st.mean(aurocs):.4f}  (n={n})')
print(f'  v7 ref (NH, 5 seeds): MCC=0.506  AUROC=0.80')
"
