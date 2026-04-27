#!/usr/bin/env bash
# =============================================================================
# v7-pro multi-seed validation: non_human → human (sequential).
#
# Runs Tier A + C + E + F stack on both corpora, 5 canonical seeds each.
# Human run starts automatically when non_human run finishes. Errors in
# the first phase do not block the second (set +e during runs to keep
# pipeline going; user can review logs).
#
# Estimated time on diamante-01 (cuDNN ON):
#   non_human × 5 seeds ≈ 25 min
#   human × 5 seeds      ≈ 4-6 h
#   total                ≈ 4.5-7 h
#
# Usage:
#   bash scripts/v8/run_v7_pro_validation.sh
#
# Override knobs:
#   SEEDS         default: "42 123 456 789 1024"
#   V7_CONFIG     default: configs/v7_pro.yaml
#   V8ENV         default: env (repo venv)
#   FORCE_RM      "1" to wipe output dirs first (default: skip if metrics.json exists)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

V7_CONFIG_PATH="${V7_CONFIG:-configs/v7_pro.yaml}"
SEEDS_LIST="${SEEDS:-42 123 456 789 1024}"
RUNNER="${SCRIPT_DIR}/run_v7_yaml.sh"
FORCE_RM="${FORCE_RM:-0}"

# diamante-01 defaults: cuDNN ON, fp32, no compile.
export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"
export BENCHMARK_LEVEL4CNN_NO_COMPILE="${BENCHMARK_LEVEL4CNN_NO_COMPILE:-1}"

run_corpus() {
    local corpus="$1"
    # Output dir derived from v7_pro.yaml output_root template
    # results/benchmark_pro_<corpus>_8M
    local out_dir="results/benchmark_pro_${corpus}_8M"

    echo
    echo "=============================================================="
    echo "  CORPUS: ${corpus}"
    echo "  CONFIG: ${V7_CONFIG_PATH}"
    echo "  SEEDS:  ${SEEDS_LIST}"
    echo "  OUT:    ${out_dir}"
    echo "  start:  $(date -Iseconds)"
    echo "=============================================================="

    if [ "${FORCE_RM}" = "1" ] && [ -d "${out_dir}" ]; then
        echo "[forceRM] removing ${out_dir}"
        rm -rf "${out_dir}"
    fi

    SEEDS="${SEEDS_LIST}" \
    V7_CONFIG="${V7_CONFIG_PATH}" \
    CORPUS="${corpus}" \
    bash "${RUNNER}"

    local rc=$?
    echo
    echo "[corpus=${corpus}] runner exit=${rc}  end: $(date -Iseconds)"
    return ${rc}
}

echo "=============================================================="
echo "v7-pro validation pipeline (sequential: non_human → human)"
echo "  config:    ${V7_CONFIG_PATH}"
echo "  seeds:     ${SEEDS_LIST}"
echo "  cuDNN:     ${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN} (0=on, 1=off)"
echo "  force_rm:  ${FORCE_RM}"
echo "  pipeline_start: $(date -Iseconds)"
echo "=============================================================="

# Pull latest before running (assumes no local changes; safe on diamante-01).
echo "[git] pulling latest…"
git pull --ff-only || echo "[warn] git pull failed (continuing anyway)"

# Phase 1: non_human (fast, ~25 min)
run_corpus non_human
nh_rc=$?

# Phase 2: human (slow, ~5h). Runs regardless of phase 1 status.
run_corpus human
hu_rc=$?

echo
echo "=============================================================="
echo "v7-pro validation pipeline complete"
echo "  non_human exit: ${nh_rc}"
echo "  human exit:     ${hu_rc}"
echo "  pipeline_end:   $(date -Iseconds)"
echo "=============================================================="

# Final consolidated summary
for corpus in non_human human; do
    json_path="results/benchmark_pro_${corpus}_8M/test/benchmark_comparison.json"
    if [ -f "${json_path}" ]; then
        echo
        echo "--- ${corpus} ---"
        python3 -c "
import json
d = json.load(open('${json_path}'))
for level, models in d.items():
    for model, metrics in models.items():
        mcc = metrics.get('mcc', 'n/a')
        std = metrics.get('mcc_std', 0.0)
        n = metrics.get('n_seeds', metrics.get('seeds', '?'))
        print(f'  {model:25s}  MCC={mcc}  ±{std}  (n={n})')
" 2>/dev/null || cat "${json_path}"
    else
        echo "[skip] ${corpus} json not found at ${json_path}"
    fi
done

exit 0
