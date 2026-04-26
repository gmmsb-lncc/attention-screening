#!/usr/bin/env bash
# d01 — v7+F + CORAL domain alignment (Sun & Saenko 2016)
# Substitui experimento Morgan FP (regrediu -0.085 MCC).
# IMPORTANTE: corpus DEVE ser 'all' (mistura H+NH em batches)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BENCHMARK_LEVEL4CNN_CORAL_LAMBDA="${BENCHMARK_LEVEL4CNN_CORAL_LAMBDA:-0.1}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

export SEEDS="${SEEDS:-42 123}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_plus_F_coral.yaml}"
export CORPUS="${CORPUS:-all}"

echo "=============================================================="
echo "v7+F + CORAL DA (d01)"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}  (deve ser 'all' para mistura H+NH)"
echo "  seeds:         ${SEEDS}"
echo "  λ_coral:       ${BENCHMARK_LEVEL4CNN_CORAL_LAMBDA}"
echo "=============================================================="

if [ "${CORPUS}" != "all" ]; then
    echo "[warn] CORPUS=${CORPUS} (não 'all') — sinal CORAL será zero"
    echo "       (todos exemplos do mesmo domínio → covariância indefinida)"
fi

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
