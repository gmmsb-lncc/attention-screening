#!/usr/bin/env bash
# =============================================================================
# DT-Kinase v8 POC orchestrator.
#
# Pipeline:
#   1. Pre-compute all v8 feature caches (ligand + protein) for a corpus
#   2. Train v8-lig and v8-full for 5 canonical seeds on that corpus
#
# Initial POC target: Non-Human only. Human follows after Go/No-Go.
#
# Env overrides:
#   CORPUS        default: non_human
#   SEEDS         default: 42 123 456 789 1024
#   ABLATIONS     default: "v8-lig v8-full"
#   PFAM_HMM      path to Pfam-A.hmm (REQUIRED for v8-full or v8-prot)
#   TAXDUMP       dir with names.dmp + nodes.dmp (REQUIRED for v8-full or v8-prot)
#   CACHE_ROOT    default: data/embeddings/v8
#   OUT_ROOT      default: results/v8
#   V7_ENV        default: env (venv); alternatives: conda env name
#   SKIP_CACHE    1 to skip pre-compute phase (assume caches exist)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-non_human}"
SEEDS=(${SEEDS:-42 123 456 789 1024})
ABLATIONS=(${ABLATIONS:-v8-lig v8-full})
PFAM_HMM="${PFAM_HMM:-}"
TAXDUMP="${TAXDUMP:-}"
CACHE_ROOT="${CACHE_ROOT:-data/embeddings/v8}"
OUT_ROOT="${OUT_ROOT:-results/v8}"
# Two envs required (see scripts/v8/setup_v8env.sh):
#   V8ENV    — training + ChemBERTa/BioBERT/Pfam/Taxonomy/ClassyFire/UniProt
#   ADMETENV — admet_ai precompute only (isolated; admet_ai drags many deps
#              incompatible with the training torch version)
# Allow V7_ENV as a fallback alias for V8ENV for backward compat with earlier
# invocations.
V8ENV="${V8ENV:-${V7_ENV:-v8env}}"
ADMETENV="${ADMETENV:-admetenv}"
SKIP_CACHE="${SKIP_CACHE:-0}"

# ---------------------------------------------------------------------------
# Env activation (venv-path OR conda env name)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# CPU / GPU env vars
# ---------------------------------------------------------------------------
N_CPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${N_CPU}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${N_CPU}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${N_CPU}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${N_CPU}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=============================================================="
echo "DT-Kinase v8 POC"
echo "  corpus:      ${CORPUS}"
echo "  seeds:       ${SEEDS[*]}"
echo "  ablations:   ${ABLATIONS[*]}"
echo "  cache_root:  ${CACHE_ROOT}"
echo "  out_root:    ${OUT_ROOT}"
echo "  CPUs:        ${N_CPU}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Phase 1: pre-compute feature caches
#   Most features run in V8ENV (torch 2.5.1+cu124, GPU).
#   ADMET-AI runs in ADMETENV (isolated; pulls torch/lightning/chemprop
#   that would clobber V8ENV's torch).
# ---------------------------------------------------------------------------
if [ "${SKIP_CACHE}" != "1" ]; then
    (
        activate_env "${V8ENV}"
        echo "--- precompute chemberta (${V8ENV}) ---"
        python3 scripts/v8/precompute_chemberta_ligand.py --corpus "${CORPUS}"

        echo "--- precompute classyfire (${V8ENV}) ---"
        python3 scripts/v8/precompute_classyfire_ligand.py --corpus "${CORPUS}"

        # Protein features only if any ablation uses the protein side
        needs_prot=0
        for abl in "${ABLATIONS[@]}"; do
            if [ "${abl}" = "v8-prot" ] || [ "${abl}" = "v8-full" ]; then
                needs_prot=1
            fi
        done
        if [ "${needs_prot}" = "1" ]; then
            echo "--- fetch UniProt annotations (${V8ENV}) ---"
            python3 scripts/v8/fetch_uniprot_annotations.py --corpus "${CORPUS}"

            echo "--- precompute biobert (${V8ENV}) ---"
            python3 scripts/v8/precompute_biobert_target.py --corpus "${CORPUS}"

            if [ -z "${PFAM_HMM}" ]; then
                echo "[warn] PFAM_HMM not set — skipping Pfam cache (features will be zero-filled)"
            else
                echo "--- precompute pfam (${V8ENV}) ---"
                python3 scripts/v8/precompute_pfam_target.py --corpus "${CORPUS}" --hmm "${PFAM_HMM}"
            fi

            if [ -z "${TAXDUMP}" ]; then
                echo "[warn] TAXDUMP not set — skipping Taxonomy cache (features will be zero-filled)"
            else
                echo "--- precompute taxonomy (${V8ENV}) ---"
                python3 scripts/v8/precompute_taxonomy_target.py --corpus "${CORPUS}" --taxdump "${TAXDUMP}"
            fi
        fi
    )

    # ADMET runs in its dedicated env (CPU only, torch clobber isolated)
    (
        activate_env "${ADMETENV}"
        echo "--- precompute admet-ai (${ADMETENV}) ---"
        python3 scripts/v8/precompute_admet_ligand.py --corpus "${CORPUS}"
    )
fi

# ---------------------------------------------------------------------------
# Phase 2: train v8 configs × seeds
# ---------------------------------------------------------------------------
for ablation in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out_dir="${OUT_ROOT}/${CORPUS}/${ablation}/seed_${seed}"
        if [ -f "${out_dir}/metrics.json" ]; then
            echo "[skip] ${ablation}/${CORPUS}/seed_${seed} already done"
            continue
        fi
        mkdir -p "${out_dir}"
        echo
        echo "================================================================"
        echo "  train ${ablation}/${CORPUS}/seed_${seed}"
        echo "================================================================"
        (
            activate_env "${V8ENV}"
            python3 scripts/v8/train_v8.py \
                --config configs/v8.yaml \
                --dataset "${CORPUS}" \
                --seed "${seed}" \
                --ablation "${ablation}" \
                --output-dir "${out_dir}"
        ) 2>&1 | tee "${out_dir}/train.log"
    done
done

echo
echo "[done] POC v8 complete → ${OUT_ROOT}/${CORPUS}/"
echo "Aggregate MCC summary:"
for ablation in "${ABLATIONS[@]}"; do
    python3 -c "
import json, glob, statistics as st
paths = sorted(glob.glob('${OUT_ROOT}/${CORPUS}/${ablation}/seed_*/metrics.json'))
if not paths:
    print('  ${ablation}: no results'); exit()
mccs = [json.load(open(p))['test_mcc'] for p in paths]
print(f'  ${ablation} MCC={st.mean(mccs):.4f}±{st.stdev(mccs) if len(mccs)>1 else 0.0:.4f} (n={len(mccs)})')
"
done
