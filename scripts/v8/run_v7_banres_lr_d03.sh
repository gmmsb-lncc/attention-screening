#!/usr/bin/env bash
# d03 — v7+F + BAN-residual α-gate + BAN_LR_MULT=5.0 (NH 2-seed)
# Aplica Lição 19 (atomicidade) ao BAN-residual rejeitado em §6.12
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BENCHMARK_LEVEL4CNN_BAN_RESIDUAL="${BENCHMARK_LEVEL4CNN_BAN_RESIDUAL:-1}"
export BENCHMARK_LEVEL4CNN_BAN_LR_MULT="${BENCHMARK_LEVEL4CNN_BAN_LR_MULT:-5.0}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

export SEEDS="${SEEDS:-42 123}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_ban_res_lr.yaml}"
export CORPUS="${CORPUS:-non_human}"

echo "=============================================================="
echo "v7+F + BAN-residual + BAN_LR_MULT (d03)"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}"
echo "  seeds:         ${SEEDS}"
echo "  BAN_RESIDUAL:  ${BENCHMARK_LEVEL4CNN_BAN_RESIDUAL}"
echo "  BAN_LR_MULT:   ${BENCHMARK_LEVEL4CNN_BAN_LR_MULT}"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
