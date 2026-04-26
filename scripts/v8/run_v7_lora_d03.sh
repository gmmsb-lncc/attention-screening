#!/usr/bin/env bash
# =============================================================================
# v7+F + LoRA-fine-tuned MoLFormer pipeline (d03 host).
#
# Three sequential stages:
#   1. LoRA fine-tune MoLFormer on train SMILES (MLM, self-supervised)
#   2. Re-cache per-token MoLFormer embeddings using LoRA delta
#   3. Run v7+F benchmark with LoRA cache override
#
# All three are guarded — re-running skips completed stages.
#
# Knobs (env):
#   LORA_RANK          (default 8)
#   LORA_ALPHA         (default 16)
#   LORA_TOP_LAYERS    (default 2)
#   LORA_EPOCHS        (default 10)
#   LORA_BATCH         (default 32)
#   LORA_LR            (default 5e-4)
#   CORPUS             (default non_human)
#   SEEDS              (default "42 123 456")
#   LORA_TAG           (default "v1" — versioning of cache)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CORPUS="${CORPUS:-non_human}"
LORA_TAG="${LORA_TAG:-v1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TOP_LAYERS="${LORA_TOP_LAYERS:-2}"
LORA_EPOCHS="${LORA_EPOCHS:-10}"
LORA_BATCH="${LORA_BATCH:-32}"
LORA_LR="${LORA_LR:-5e-4}"
SEEDS="${SEEDS:-42 123 456}"

LORA_DIR="results/lora/molformer_${CORPUS}_${LORA_TAG}"
CACHE_DIR="results/lora/molformer_cache_${LORA_TAG}/${CORPUS}"
SPLITS_DIR="scaffolds_splits/output"
TRAIN_TSV="${SPLITS_DIR}/${CORPUS}_train.tsv"

if [ ! -f "${TRAIN_TSV}" ]; then
    TRAIN_TSV="${SPLITS_DIR}/scenarios/Sc/universal_train.tsv"
    if [ ! -f "${TRAIN_TSV}" ]; then
        echo "[fatal] no train TSV found at ${SPLITS_DIR}/${CORPUS}_train.tsv or ${TRAIN_TSV}" >&2
        exit 1
    fi
fi

activate_env() {
    set +u
    if [ -f "${REPO_ROOT}/env/bin/activate" ]; then
        source "${REPO_ROOT}/env/bin/activate"
    elif command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base)"
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate env || conda activate v8env || true
    fi
    set -u
}

activate_env

echo "=============================================================="
echo "v7+F + LoRA pipeline"
echo "  corpus:        ${CORPUS}"
echo "  LoRA tag:      ${LORA_TAG}"
echo "  rank/α:        ${LORA_RANK}/${LORA_ALPHA}"
echo "  top-layers:    ${LORA_TOP_LAYERS}"
echo "  epochs/batch:  ${LORA_EPOCHS}/${LORA_BATCH}"
echo "  LR:            ${LORA_LR}"
echo "  train TSV:     ${TRAIN_TSV}"
echo "  LoRA out:      ${LORA_DIR}"
echo "  Cache out:     ${CACHE_DIR}"
echo "  seeds:         ${SEEDS}"
echo "=============================================================="

# ---------- Stage 1: LoRA MLM fine-tune --------------------------------
if [ -f "${LORA_DIR}/lora_state.pt" ]; then
    echo "[stage1] SKIP — found ${LORA_DIR}/lora_state.pt"
else
    echo "[stage1] LoRA fine-tune ..."
    python3 scripts/v8/lora_finetune_molformer.py \
        --train-tsv "${TRAIN_TSV}" \
        --output-dir "${LORA_DIR}" \
        --rank "${LORA_RANK}" --alpha "${LORA_ALPHA}" \
        --top-layers "${LORA_TOP_LAYERS}" \
        --epochs "${LORA_EPOCHS}" \
        --batch-size "${LORA_BATCH}" \
        --lr "${LORA_LR}"
fi

# ---------- Stage 2: re-cache per-token embeddings ---------------------
if [ -f "${CACHE_DIR}/_recache_summary.json" ]; then
    echo "[stage2] SKIP — found ${CACHE_DIR}/_recache_summary.json"
else
    echo "[stage2] Re-cache MoLFormer embeddings via LoRA ..."
    python3 scripts/v8/recache_molformer_lora.py \
        --lora-dir "${LORA_DIR}" \
        --splits-dir "${SPLITS_DIR}" \
        --corpus "${CORPUS}" \
        --out-dir "${CACHE_DIR}" \
        --skip-existing
fi

# ---------- Stage 3: run v7+F benchmark with LoRA cache ----------------
OUT_ROOT="results/benchmark_lora_${LORA_TAG}_${CORPUS}_8M"
echo "[stage3] Run v7+F with cache override → ${OUT_ROOT}"

mv "${OUT_ROOT}" "${OUT_ROOT}_pre_$(date +%Y%m%d_%H%M)" 2>/dev/null || true

# Override only ligand cache (LoRA on MoLFormer only; ESM untouched).
export BENCHMARK_LEVEL4CNN_LIGAND_CACHE_OVERRIDE="${PWD}/${CACHE_DIR}"

# Use a per-run yaml so output_root reflects the LoRA tag.
TMP_CONFIG="$(mktemp --suffix=.yaml)"
cat configs/v7_plus_F.yaml \
    | sed "s|output_root:.*|output_root: ${OUT_ROOT}|" \
    > "${TMP_CONFIG}"

SEEDS="${SEEDS}" V7_CONFIG="${TMP_CONFIG}" CORPUS="${CORPUS}" \
    bash scripts/v8/run_v7_yaml.sh

rm -f "${TMP_CONFIG}"

echo "=============================================================="
echo "DONE. Inspect: ${OUT_ROOT}/test/benchmark_comparison.json"
echo "=============================================================="
