#!/usr/bin/env bash
# =============================================================================
# Run configs/v7_ban_res.yaml — v7+F + BAN-residual com α-gate aprendível.
# Implementa lição 12 reformulada (licoes_aprendidas §6.4).
#
# Identity-init em t=0 (α_k=0) → modelo idêntico ao v7+F. Gradient flui
# para α via W_k Xavier (anti-cascade). W_k começa treinar quando α > 0.
#
# Custo: +3.93M params trainable (16 × 320 × 768 = 3.93M para W_k).
# Tempo esperado: ~30-40 min (3-seed) em RTX 4090 cuDNN ON;
# +20% sobre v7+F base devido aos params extras.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BENCHMARK_LEVEL4CNN_BAN_RESIDUAL="${BENCHMARK_LEVEL4CNN_BAN_RESIDUAL:-1}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

export SEEDS="${SEEDS:-42 123 456}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_ban_res.yaml}"
export CORPUS="${CORPUS:-non_human}"

echo "=============================================================="
echo "v7+F + BAN-residual (α-gate identity-init, lição 12 reformulada)"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}"
echo "  seeds:         ${SEEDS}"
echo "  BAN_RESIDUAL:  ${BENCHMARK_LEVEL4CNN_BAN_RESIDUAL}"
echo "  cuDNN disable: ${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN}"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
