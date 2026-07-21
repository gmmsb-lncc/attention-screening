#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CMADTI_ENV_NAME:-cmadti-cuda}"
SEEDS="${CMADTI_SEEDS:-42 123 456 789 1024}"
CORPORA="${CMADTI_CORPORA:-all}"

cd "${ROOT_DIR}"
for corpus in ${CORPORA}; do
  echo "[$(date '+%F %T')] Preparing CMA-DTI corpus: ${corpus}"
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -u scripts/cmadti/prepare_universal.py --corpus "${corpus}"
  for seed in ${SEEDS}; do
    output="results/cmadti/cmadti_${corpus}_seed${seed}"
    reuse=()
    if [[ -f "${output}/best_model.pt" ]]; then
      reuse=(--reuse-checkpoint)
    fi
    echo "[$(date '+%F %T')] CMA-DTI corpus=${corpus} seed=${seed}"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -u scripts/cmadti/train_canonical.py \
        --corpus "${corpus}" --seed "${seed}" --output "${output}" "${reuse[@]}"
  done
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -u scripts/cmadti/aggregate_results.py \
      --corpus "${corpus}" --seeds ${SEEDS}
done
