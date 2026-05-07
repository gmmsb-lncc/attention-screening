#!/usr/bin/env bash
# Run the full statistical-analysis panel for one corpus.
#
# Usage:
#   bash scripts/statistical_analysis/run_full_stats.sh non_human
#   bash scripts/statistical_analysis/run_full_stats.sh human
#   bash scripts/statistical_analysis/run_full_stats.sh all
#
# Optional environment variables:
#   N_BOOTSTRAP    bootstrap iterations (default 10000)
#   PYTHON         python interpreter (default env_baseline/bin/python)
#   MIRROR_TO      if set, copy figures to this directory (e.g. ~/PhD/figures)
#
# Outputs all artifacts under results/statistical/<corpus>/.
set -euo pipefail

CORPUS="${1:-non_human}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
PYTHON="${PYTHON:-env_baseline/bin/python}"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: python interpreter not found at ${PYTHON}" >&2
    echo "Set PYTHON env var or fix env_baseline/." >&2
    exit 1
fi

OUT="results/statistical/${CORPUS}"
mkdir -p "${OUT}/figures"
echo "[run_full_stats] corpus=${CORPUS} out=${OUT} B=${N_BOOTSTRAP}"

PY="${PYTHON} -m scripts.statistical_analysis"

${PY}.null_model              --corpus "${CORPUS}" --out "${OUT}/null_model.json" \
                              --n-bootstrap "${N_BOOTSTRAP}"
${PY}.upper_limit             --corpus "${CORPUS}" --out "${OUT}/upper_limit.json" \
                              --n-trials "${N_BOOTSTRAP}"
${PY}.posthoc_classification  --corpus "${CORPUS}" --out "${OUT}/posthoc.json"
${PY}.effect_size             --corpus "${CORPUS}" --out "${OUT}/effect_size.json"
${PY}.anova_tukey             --corpus "${CORPUS}" --out "${OUT}/anova_tukey.json"
${PY}.tost_sensitivity        --corpus "${CORPUS}" --metric mcc \
                              --n-bootstrap "${N_BOOTSTRAP}" \
                              --out "${OUT}/tost.json"

for METRIC in mcc auroc f1 auprc; do
    ${PY}.plot_simultaneous_ci  --corpus "${CORPUS}" --metric "${METRIC}" \
                                --out "${OUT}/figures/sim_ci_${METRIC}.pdf"
    ${PY}.plot_mcsim_heatmap    --corpus "${CORPUS}" --metric "${METRIC}" \
                                --effect-source "${OUT}/effect_size.json" \
                                --pvalue-source "${OUT}/anova_tukey.json" \
                                --out "${OUT}/figures/mcsim_${METRIC}.pdf"
done

MIRROR_FLAG=""
if [ -n "${MIRROR_TO:-}" ]; then
    MIRROR_FLAG="--mirror-to ${MIRROR_TO}"
fi

# shellcheck disable=SC2086
${PY}.aggregate_panel         --corpus "${CORPUS}" --in-dir "${OUT}" \
                              --out-json "${OUT}/panel.json" \
                              --out-tex "${OUT}/panel.tex" \
                              --out-checklist "${OUT}/checklist.md" \
                              ${MIRROR_FLAG}

echo "[run_full_stats] done -> ${OUT}/panel.json + ${OUT}/checklist.md"
