#!/usr/bin/env bash
# d02 — v7+F + Adversarial Domain Loss CDAN-style (All 2-seed)
# IMPORTANTE: corpus DEVE ser 'all' (mistura H+NH em batches)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA="${BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA:-0.1}"
export BENCHMARK_LEVEL4CNN_ADVERSARIAL_N_DOMAINS="${BENCHMARK_LEVEL4CNN_ADVERSARIAL_N_DOMAINS:-2}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

export SEEDS="${SEEDS:-42 123}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_plus_F_adv.yaml}"
export CORPUS="${CORPUS:-all}"

echo "=============================================================="
echo "v7+F + Adversarial DA (d02)"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}  (deve ser 'all' para mistura H+NH)"
echo "  seeds:         ${SEEDS}"
echo "  λ_adv max:     ${BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA}"
echo "  n_domains:     ${BENCHMARK_LEVEL4CNN_ADVERSARIAL_N_DOMAINS}"
echo "=============================================================="

if [ "${CORPUS}" != "all" ]; then
    echo "[warn] CORPUS=${CORPUS} (não 'all') — sinal adversarial será zero"
    echo "       (todos exemplos do mesmo domínio → CE loss = 0)"
fi

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
