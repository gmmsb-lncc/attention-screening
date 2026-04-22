#!/usr/bin/env bash
# ============================================================================
# run_universal_f1val_baselines.sh
#
# Protocolo completo pós-hoc:
#   1. Inferência DrugBAN/GraphBAN/ConPLex sobre universal_{val,test}.tsv
#   2. Recalibração F1-ótimo na validação (critério nativo BAN; uniformizado)
#   3. Agregação por corpus em JSON consumível pelas tabelas da tese
#
# Uso (em diamante-02):
#     cd /path/to/semantic-screening
#     git checkout cross_attention_lite
#     git pull
#     bash run_universal_f1val_baselines.sh
#
# Variáveis configuráveis via ambiente:
#     SEEDS    (default: "42 123 456 789 1024")
#     CORPORA  (default: "non_human human all")
#     MODELS   (default: "drugban graphban conplex")
#     OUTDIR   (default: "./universal_f1val_results")
#     PYTHON   (default: python)
# ============================================================================

set -euo pipefail

SEEDS="${SEEDS:-42 123 456 789 1024}"
CORPORA="${CORPORA:-non_human human all}"
MODELS="${MODELS:-drugban graphban conplex}"
OUTDIR="${OUTDIR:-./universal_f1val_results}"
PYTHON="${PYTHON:-python}"

# Nomes dos ambientes conda por modelo (conforme configuração em diamante-02)
CONDA_ENV_DRUGBAN="${CONDA_ENV_DRUGBAN:-drugban}"
CONDA_ENV_GRAPHBAN="${CONDA_ENV_GRAPHBAN:-graphban}"
CONDA_ENV_CONPLEX="${CONDA_ENV_CONPLEX:-conplex}"

# Garantir que `conda activate` funcione dentro do script
CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
else
    echo "[erro] conda.sh não encontrado; defina CONDA_SH ou garanta 'conda' no PATH" >&2
    exit 1
fi

mkdir -p "${OUTDIR}/logs"
echo "============================================================"
echo " Universal split inference + F1-val recalibration"
echo " Seeds:    ${SEEDS}"
echo " Corpora:  ${CORPORA}"
echo " Models:   ${MODELS}"
echo " Output:   ${OUTDIR}"
echo "============================================================"

for model in ${MODELS}; do
    case "${model}" in
        drugban)  INFER="infer_drugban_universal.py";  CKPT_ROOT="DrugBAN/results";     OUT_INFER="DrugBAN/results_universal";  ENV_NAME="${CONDA_ENV_DRUGBAN}"  ;;
        graphban) INFER="infer_graphban_universal.py"; CKPT_ROOT="GraphBAN/results";    OUT_INFER="GraphBAN/results_universal"; ENV_NAME="${CONDA_ENV_GRAPHBAN}" ;;
        conplex)  INFER="infer_conplex_universal.py";  CKPT_ROOT="ConPLex/best_models"; OUT_INFER="ConPLex/results_universal";  ENV_NAME="${CONDA_ENV_CONPLEX}"  ;;
        *) echo "[erro] modelo desconhecido: ${model}"; continue ;;
    esac

    echo ">>> [${model}] conda activate ${ENV_NAME}"
    conda activate "${ENV_NAME}" || { echo "[erro] falha ao ativar ${ENV_NAME}"; exit 2; }

    for corpus in ${CORPORA}; do
        LOG="${OUTDIR}/logs/infer_${model}_${corpus}.log"
        echo ">>> [${model}/${corpus}] inferência sobre universal split (log: ${LOG})"
        ${PYTHON} "${INFER}" --corpus "${corpus}" --seeds ${SEEDS} \
            --checkpoint-root "${CKPT_ROOT}" \
            --output-dir "${OUT_INFER}" \
            2>&1 | tee "${LOG}"
    done

    echo ">>> [${model}] recalibração F1-val"
    RECALIB_OUT="${OUTDIR}/recalib_${model}_universal.json"
    ${PYTHON} recalibrate_baselines_f1val.py \
        --results-root "$(dirname "${OUT_INFER}")" \
        --models "$(basename "${OUT_INFER}")" \
        --output "${RECALIB_OUT}" \
        2>&1 | tee "${OUTDIR}/logs/recalib_${model}.log"
    echo "    salvo: ${RECALIB_OUT}"

    echo ">>> [${model}] conda deactivate"
    conda deactivate
done

echo
echo "============================================================"
echo " Protocolo concluído. JSONs em ${OUTDIR}/recalib_*_universal.json"
echo "============================================================"
echo
echo "Resumo agregado:"
for model in ${MODELS}; do
    RECALIB_OUT="${OUTDIR}/recalib_${model}_universal.json"
    if [[ -f "${RECALIB_OUT}" ]]; then
        echo
        echo "--- ${model} ---"
        ${PYTHON} -c "
import json
d = json.load(open('${RECALIB_OUT}'))
for m, cs in d.items():
    for c, v in cs.items():
        a = v['aggregate']
        n = a['n_seeds']
        mcc = a['test_mcc']; f1 = a['test_f1']
        p = a['test_precision']; r = a['test_recall']; au = a['test_auroc']
        print(f'  {c:>10s}: n={n} | MCC={mcc[\"mean\"]:.4f}±{mcc[\"std\"]:.4f}  F1={f1[\"mean\"]:.4f}±{f1[\"std\"]:.4f}  P={p[\"mean\"]:.4f}  R={r[\"mean\"]:.4f}  AUROC={au[\"mean\"]:.4f}')
"
    fi
done

echo
echo "Transfira os JSONs para a máquina local:"
echo "    scp leon@diamante-02:$(pwd)/${OUTDIR}/recalib_*_universal.json ~/PhD/"
