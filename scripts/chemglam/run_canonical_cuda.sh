#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CHEMGLAM_ENV_NAME:-chemglam-cuda}"
SEEDS="${CHEMGLAM_SEEDS:-42 123 456 789 1024}"
CORPORA="${CHEMGLAM_CORPORA:-all}"
SKIP_TRAIN_IF_CHECKPOINT="${CHEMGLAM_SKIP_TRAIN_IF_CHECKPOINT:-1}"

cd "${ROOT_DIR}"
bash scripts/chemglam/apply_upstream_patches.sh

for corpus in ${CORPORA}; do
  echo "[$(date '+%F %T')] Preparing corpus: ${corpus}"
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -u scripts/chemglam/prepare_universal.py --corpus "${corpus}"

  for seed in ${SEEDS}; do
    seed_start=${SECONDS}
    run_name="chemglam_${corpus}_seed${seed}"
    config_dir="results/chemglam/${run_name}/configs"
    checkpoint="logs/${run_name}/best_checkpoint.ckpt"
    echo "[$(date '+%F %T')] Creating configs: ${run_name}"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -u scripts/chemglam/make_run_configs.py \
      --corpus "${corpus}" --seed "${seed}" --output "${config_dir}"

    if [[ "${SKIP_TRAIN_IF_CHECKPOINT}" == "1" && -f "${checkpoint}" ]]; then
      echo "[$(date '+%F %T')] Reusing checkpoint: ${checkpoint}"
    else
      echo "[$(date '+%F %T')] Starting training: ${run_name}"
      WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}/ChemGLaM" \
        conda run --no-capture-output -n "${ENV_NAME}" \
          python -u ChemGLaM/train.py -c "${config_dir}/train.json"
    fi

    # Train predictions are retained for overfit diagnosis and parity with
    # DrugBAN/GraphBAN raw_predictions.npz.  Threshold selection still uses
    # validation only; test remains fully held out.
    for split in train val test; do
      config_name="${split}"
      if [[ "${split}" == "train" ]]; then
        config_name="train_eval"
      fi
      echo "[$(date '+%F %T')] Starting prediction: ${run_name} (${split})"
      WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTHONPATH="${ROOT_DIR}/ChemGLaM" \
        conda run --no-capture-output -n "${ENV_NAME}" \
          python -u ChemGLaM/predict.py -c "${config_dir}/${config_name}.json"
    done

    echo "[$(date '+%F %T')] Evaluating: ${run_name}"
    conda run --no-capture-output -n "${ENV_NAME}" \
      python -u scripts/chemglam/evaluate_predictions.py \
      --corpus "${corpus}" \
      --seed "${seed}" \
      --train-predictions "logs/${run_name}_train/prediction.csv" \
      --val-predictions "logs/${run_name}_val/prediction.csv" \
      --test-predictions "logs/${run_name}_test/prediction.csv" \
      --checkpoint "${checkpoint}" \
      --config "${config_dir}/train.json" \
      --elapsed-seconds "$((SECONDS - seed_start))" \
      --output "results/chemglam/${run_name}"
    echo "[$(date '+%F %T')] Finished: ${run_name}"
  done

  echo "[$(date '+%F %T')] Aggregating seeds: ${corpus}"
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -u scripts/chemglam/aggregate_results.py \
      --corpus "${corpus}" --seeds ${SEEDS} \
      --results-root results/chemglam
done
