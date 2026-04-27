#!/usr/bin/env bash
# =============================================================================
# Run configs/v7_plus_F_adapt_v2.yaml — decoupled threshold (F1) and
# selection (MCC) criteria. Implements lesson 15 (docs/01-methodology/licoes_aprendidas.md §6.7).
#
# THRESHOLD: F1-optimal val (matches DrugBAN/GraphBAN reporting).
# SELECTION: MCC-optimal val (preserves DT-Kinase discriminative signal).
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC="${BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC:-f1}"
export BENCHMARK_LEVEL4CNN_SELECTION_METRIC="${BENCHMARK_LEVEL4CNN_SELECTION_METRIC:-mcc}"
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"
export BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS="${BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS:-0.0}"

export SEEDS="${SEEDS:-42 123 456}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_plus_F_adapt_v2.yaml}"
export CORPUS="${CORPUS:-non_human}"

echo "=============================================================="
echo "v7_plus_F_adapt_v2 — decoupled threshold/selection (lesson 15)"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}"
echo "  seeds:         ${SEEDS}"
echo "  THRESHOLD:     ${BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC}"
echo "  SELECTION:     ${BENCHMARK_LEVEL4CNN_SELECTION_METRIC}"
echo "  cuDNN:         disable=${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN}"
echo "  λ_loss:        ${BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS}"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
