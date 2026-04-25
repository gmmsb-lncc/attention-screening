#!/usr/bin/env bash
# =============================================================================
# Ablation E vs F — isolated tests of Mixup (Tier E) and label smoothing
# (Tier F) on top of v7+ canonical (Tier A+C), 3 seeds each on non_human.
#
# Background: v7-pro (Tier A+C+E+F) regressed in 5-seed NH eval
# (0.4961 ± 0.0245 vs v7+ A+C 0.5143 ± 0.0079). Two hypotheses:
#   (i)  Mixup destabilizes (variance jumped 3x)
#   (ii) E and F do not stack additively
#
# This script isolates each component to identify the culprit:
#   Phase 1: configs/v7_plus_E.yaml  (A + C + E only)
#   Phase 2: configs/v7_plus_F.yaml  (A + C + F only)
#
# 3 seeds (42, 123, 456) for fast screening. 5-seed validation reserved
# for the eventual winner.
#
# Behaviour:
#   - Waits for any active run_from_config.py process to finish first
#     (avoids stomping on a Human run still in progress)
#   - Runs E ablation, then F ablation, sequentially
#   - Prints consolidated MCC summary at the end
#
# Estimated wall-time on diamante-01 (cuDNN ON):
#   Phase 1 E: ~15-20 min
#   Phase 2 F: ~15-20 min
#   Total:     ~30-40 min
#
# Usage (foreground):
#   bash scripts/v8/run_ablation_E_F_3seeds.sh
#
# Usage (background, queues behind any active training):
#   nohup bash scripts/v8/run_ablation_E_F_3seeds.sh \
#       > ablation_E_F.log 2>&1 &
#   echo $! > ablation_E_F.pid
#   disown
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS_LIST="${SEEDS:-42 123 456}"
RUNNER="${SCRIPT_DIR}/run_v7_yaml.sh"
CORPUS="${CORPUS:-non_human}"

export BENCHMARK_LEVEL4CNN_DISABLE_CUDNN="${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN:-0}"
export BENCHMARK_LEVEL4CNN_NO_COMPILE="${BENCHMARK_LEVEL4CNN_NO_COMPILE:-1}"

# ---------------------------------------------------------------------------
# Wait for any active run_from_config.py before starting.
# Polls every 60 s. Bails out cleanly if no active run found within 5 s
# (typical when invoked manually after a finished session).
# ---------------------------------------------------------------------------
echo "[wait] checking for active run_from_config.py …"
sleep 5
while pgrep -f "run_from_config.py" > /dev/null 2>&1; do
    echo "[wait] active training detected; will retry in 60 s ($(date -Iseconds))"
    sleep 60
done
echo "[wait] no active training detected at $(date -Iseconds); proceeding."

echo "=============================================================="
echo "Ablation E vs F (3 seeds each) — corpus: ${CORPUS}"
echo "  seeds:   ${SEEDS_LIST}"
echo "  cuDNN:   ${BENCHMARK_LEVEL4CNN_DISABLE_CUDNN} (0=on, 1=off)"
echo "  start:   $(date -Iseconds)"
echo "=============================================================="

echo "[git] pulling latest…"
git pull --ff-only || echo "[warn] git pull failed; continuing"

run_phase() {
    local label="$1"
    local config="$2"
    local out_root="$3"

    echo
    echo "=============================================================="
    echo "  PHASE: ${label}"
    echo "  config: ${config}"
    echo "  out_root: ${out_root}"
    echo "  start: $(date -Iseconds)"
    echo "=============================================================="

    rm -rf "${out_root}"

    SEEDS="${SEEDS_LIST}" \
    V7_CONFIG="${config}" \
    CORPUS="${CORPUS}" \
    bash "${RUNNER}"

    local rc=$?
    echo "[${label}] runner exit=${rc}  end: $(date -Iseconds)"
    return ${rc}
}

# Phase 1: E (Mixup only)
run_phase "E (Mixup only)" \
    "configs/v7_plus_E.yaml" \
    "results/benchmark_plusE_${CORPUS}_8M"
e_rc=$?

# Phase 2: F (label smoothing only)
run_phase "F (label smoothing only)" \
    "configs/v7_plus_F.yaml" \
    "results/benchmark_plusF_${CORPUS}_8M"
f_rc=$?

echo
echo "=============================================================="
echo "Ablation pipeline complete"
echo "  E exit: ${e_rc}"
echo "  F exit: ${f_rc}"
echo "  end:    $(date -Iseconds)"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Consolidated summary: prints MCC mean ± std from each phase's
# benchmark_comparison.json. Compare against v7+ canonical baseline
# (NH 5-seed: 0.5143 ± 0.0079).
# ---------------------------------------------------------------------------
echo
echo "=== Comparative summary ==="
echo "v7+ canonical (A+C, NH 5-seed):  0.5143 ± 0.0079  (reference)"
echo "v7-pro (A+C+E+F, NH 5-seed):     0.4961 ± 0.0245  (regressed)"
echo

for phase in plusE plusF; do
    json_path="results/benchmark_${phase}_${CORPUS}_8M/test/benchmark_comparison.json"
    if [ -f "${json_path}" ]; then
        python3 -c "
import json
with open('${json_path}') as f:
    d = json.load(f)
res = d.get('results', {})
for model, m in res.items():
    mcc = m.get('mcc', 0.0)
    std = m.get('mcc_std', 0.0)
    n = len(d.get('metadata', {}).get('seeds', []))
    print(f'${phase} (NH ${n}-seed):  {model:25s}  MCC={mcc:.4f} ± {std:.4f}')
" 2>/dev/null || echo "[warn] could not parse ${json_path}"
    else
        echo "[skip] ${phase}: ${json_path} not found"
    fi
done

exit 0
