#!/usr/bin/env bash
# =============================================================================
# Run configs/v7_rope.yaml — v7+F + 2D RoPE per-modality.
#
# RoPE injeta posicionalidade relativa via rotação determinística nos
# head projections antes do produto interno. Zero params adicionais.
# Custo: ~5% extra forward (rotação de senos/cossenos) + risco de leve
# regressão por NÃO ser identity-init em t=0 (mas RoPE é estabelecido
# em literatura — RoFormer, LLaMA, etc.).
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BENCHMARK_LEVEL4CNN_USE_ROPE="${BENCHMARK_LEVEL4CNN_USE_ROPE:-1}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

export SEEDS="${SEEDS:-42 123 456}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_rope.yaml}"
export CORPUS="${CORPUS:-non_human}"

echo "=============================================================="
echo "v7+F + 2D RoPE per-modality"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}"
echo "  seeds:         ${SEEDS}"
echo "  USE_ROPE:      ${BENCHMARK_LEVEL4CNN_USE_ROPE}"
echo "  cuDNN disable: ${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN}"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
