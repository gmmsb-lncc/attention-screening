#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CHEMGLAM_ENV_NAME:-chemglam-cuda}"
SEEDS="${CHEMGLAM_SEEDS:-42 123 456 789 1024}"

cd "${ROOT_DIR}"
conda run -n "${ENV_NAME}" python scripts/chemglam/prepare_universal.py

for seed in ${SEEDS}; do
  run_name="chemglam_universal_seed${seed}"
  config_dir="results/chemglam/${run_name}/configs"
  conda run -n "${ENV_NAME}" python scripts/chemglam/make_run_configs.py \
    --seed "${seed}" --output "${config_dir}"

  WANDB_MODE=disabled PYTHONPATH="${ROOT_DIR}/ChemGLaM" \
    conda run -n "${ENV_NAME}" python ChemGLaM/train.py -c "${config_dir}/train.json"

  for split in val test; do
    WANDB_MODE=disabled PYTHONPATH="${ROOT_DIR}/ChemGLaM" \
      conda run -n "${ENV_NAME}" python ChemGLaM/predict.py -c "${config_dir}/${split}.json"
  done

  conda run -n "${ENV_NAME}" python scripts/chemglam/evaluate_predictions.py \
    --val-predictions "logs/${run_name}_val/prediction.csv" \
    --test-predictions "logs/${run_name}_test/prediction.csv" \
    --output "results/chemglam/${run_name}"
done

