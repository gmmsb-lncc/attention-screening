#!/usr/bin/env bash
# =============================================================================
# Run configs/v7_plus_F_adapt.yaml — v7+F + EmbeddingAdapter §6.5 fixes,
# threshold criterion = F1-optimal val (DrugBAN/GraphBAN native criterion).
#
# Defaults to diamante-01 (cuDNN ON). For diamante-02 set:
#   BENCHMARK_LEVEL4CNN_DISABLE_CUDNN=1 bash scripts/v8/run_v7_plus_F_adapt.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# F1-optimal threshold (matches DrugBAN/GraphBAN baseline criterion).
export BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC="${BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC:-f1}"

# diamante-01 default: cuDNN ON for fair comparison with v7+F (3-seed 0.5260).
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"

# Composite selection disabled (λ=0 → pure best-metric selection).
export BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS="${BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS:-0.0}"

# 3-seed default to match v7+F validation set (0.5260 ± 0.0274 reference).
export SEEDS="${SEEDS:-42 123 456}"
export V7_CONFIG="${V7_CONFIG:-configs/v7_plus_F_adapt.yaml}"
export CORPUS="${CORPUS:-non_human}"

echo "=============================================================="
echo "v7_plus_F_adapt — EmbeddingAdapter §6.5 + F1 threshold"
echo "  config:        ${V7_CONFIG}"
echo "  corpus:        ${CORPUS}"
echo "  seeds:         ${SEEDS}"
echo "  threshold:     ${BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC}"
echo "  cuDNN:         disable=${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN}"
echo "  λ_loss:        ${BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS}"
echo "=============================================================="

bash "${SCRIPT_DIR}/run_v7_yaml.sh"
