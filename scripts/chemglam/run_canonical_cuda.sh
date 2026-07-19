#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CHEMGLAM_ENV_NAME:-chemglam-cuda}"
SEEDS="${CHEMGLAM_SEEDS:-42 123 456 789 1024}"
CORPORA="${CHEMGLAM_CORPORA:-all}"

cd "${ROOT_DIR}"
bash scripts/chemglam/apply_upstream_patches.sh

for corpus in ${CORPORA}; do
  echo "[$(date '+%F %T')] Preparing corpus: ${corpus}"
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -u scripts/chemglam/prepare_universal.py --corpus "${corpus}"

  for seed in ${SEEDS}; do
    run_name="chemglam_${corpus}_seed${seed}"
    config_dir="results/chemglam/${run_name}/configs"
    echo "[$(date '+%F %T')] Creating configs: ${run_name}"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -u scripts/chemglam/make_run_configs.py \
      --corpus "${corpus}" --seed "${seed}" --output "${config_dir}"

    echo "[$(date '+%F %T')] Starting training: ${run_name}"
    WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}/ChemGLaM" \
      conda run --no-capture-output -n "${ENV_NAME}" \
        python -u ChemGLaM/train.py -c "${config_dir}/train.json"

    for split in val test; do
      echo "[$(date '+%F %T')] Starting prediction: ${run_name} (${split})"
      WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}/ChemGLaM" \
        conda run --no-capture-output -n "${ENV_NAME}" \
          python -u ChemGLaM/predict.py -c "${config_dir}/${split}.json"
    done

    echo "[$(date '+%F %T')] Evaluating: ${run_name}"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -u scripts/chemglam/evaluate_predictions.py \
      --corpus "${corpus}" \
      --val-predictions "logs/${run_name}_val/prediction.csv" \
      --test-predictions "logs/${run_name}_test/prediction.csv" \
      --output "results/chemglam/${run_name}"
    echo "[$(date '+%F %T')] Finished: ${run_name}"
  done
done
