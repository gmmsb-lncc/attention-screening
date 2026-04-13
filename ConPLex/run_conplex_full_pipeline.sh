#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_conplex_full_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════════
# Complete pipeline: Train ConPLex from scratch + Evaluate with val-calibrated
# threshold (MCC-optimal, zero test leakage).
#
# Protocol:
#   1. Train 3 reps per dataset with contrastive loss + BCE
#   2. Best model saved per rep (val AUPRC criterion, native ConPLex)
#   3. Evaluate v2: inference(val) → sweep τ* → inference(test) → apply τ*
#
# Usage:
#   bash run_conplex_full_pipeline.sh
#
# Environment variables:
#   CUDA_DEVICE  — GPU device ID (default: 0)
#   EPOCHS       — Number of training epochs (default: 50)
#   BATCH_SIZE   — Training batch size (default: 32)
#   SKIP_TRAIN   — Set to 1 to skip training and only run eval_v2
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${CUDA_DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SEEDS=(42 123 456)
DATASETS=(non_human human)

MODEL_SAVE_DIR="./best_models"
RESULTS_V2_DIR="./results_v2"

# Disable wandb globally — no login required
export WANDB_MODE=disabled

echo "═══════════════════════════════════════════════════════════════════"
echo " ConPLex Full Pipeline — Train + Eval v2 (val-calibrated τ*)"
echo "═══════════════════════════════════════════════════════════════════"
echo "  Device:     cuda:${DEVICE}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Seeds:      ${SEEDS[*]}"
echo "  Datasets:   ${DATASETS[*]}"
echo "  Model dir:  ${MODEL_SAVE_DIR}"
echo "  Results:    ${RESULTS_V2_DIR}"
echo "  wandb:      DISABLED"
echo "═══════════════════════════════════════════════════════════════════"

# ── Generate kinase config (always overwrite to ensure correctness) ────────
CONFIG_FILE="configs/kinase_config.yaml"
echo "  Generating ${CONFIG_FILE}..."
mkdir -p configs
cat > "$CONFIG_FILE" << 'YAML'
task: davis
contrastive_split: within

drug_featurizer: MorganFeaturizer
target_featurizer: ProtBertFeaturizer
model_architecture: SimpleCoembedding
latent_dimension: 1024
latent_distance: "Cosine"

batch_size: 32
contrastive_batch_size: 256
shuffle: True
num_workers: 0

epochs: 50
every_n_val: 1
lr: 1e-4
lr_t0: 10
contrastive: False
clr: 1e-5
clr_t0: 10
margin_fn: 'tanh_decay'
margin_max: 0.25
margin_t0: 10

replicate: 0
device: 0
verbosity: 3

wandb_save: False
log_file: ./logs/training.log
model_save_dir: ./best_models
YAML
echo "  ✓ Config created"

# ── Prepare kinase datasets (auto-detect from benchmark splits) ────────────
echo ""
echo "━━━ Checking kinase datasets ━━━"
DATASETS_OK=true
for dataset in "${DATASETS[@]}"; do
    data_dir="dataset/kinase_${dataset}"
    if [ -f "${data_dir}/train.csv" ] && [ -f "${data_dir}/val.csv" ] && [ -f "${data_dir}/test.csv" ]; then
        n_train=$(wc -l < "${data_dir}/train.csv")
        echo "  ✓ kinase_${dataset}: train=${n_train} lines"
    else
        DATASETS_OK=false
    fi
done

if [ "$DATASETS_OK" = false ]; then
    echo "  → Some datasets missing. Running prepare_kinase_datasets.py..."
    python prepare_kinase_datasets.py --output-dir ./dataset
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Training
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_TRAIN" = "1" ]; then
    echo ""
    echo "⏭  SKIP_TRAIN=1 — Skipping training phase"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " PHASE 1: TRAINING (from scratch)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    mkdir -p logs

    for dataset in "${DATASETS[@]}"; do
        task="kinase_${dataset}"
        echo ""
        echo "  ═══ Dataset: ${dataset} (task=${task}) ═══"
        
        for rep_idx in 0 1 2; do
            seed="${SEEDS[$rep_idx]}"
            exp_id="trained_${dataset}_rep${rep_idx}"
            
            echo ""
            echo "  ▶ ${exp_id} (seed=${seed})"
            echo "    Training ${EPOCHS} epochs..."
            
            python train_DTI.py \
                --exp-id "$exp_id" \
                --config "$CONFIG_FILE" \
                --task "$task" \
                --d "$DEVICE" \
                --r "$seed" \
                --epochs "$EPOCHS" \
                -b "$BATCH_SIZE" 2>&1 | tail -10
            
            # Verify checkpoint was saved
            if ls ${MODEL_SAVE_DIR}/${exp_id}/${exp_id}_best_model*.pt 1>/dev/null 2>&1; then
                n_ckpts=$(ls ${MODEL_SAVE_DIR}/${exp_id}/${exp_id}_best_model*.pt | wc -l)
                latest=$(ls -t ${MODEL_SAVE_DIR}/${exp_id}/${exp_id}_best_model*.pt | head -1)
                echo "    ✓ ${n_ckpts} checkpoint(s). Latest: $(basename $latest)"
            else
                echo "    ✗ ERROR: No checkpoint found for ${exp_id}!"
                ls -la "${MODEL_SAVE_DIR}/${exp_id}/" 2>/dev/null || echo "      (directory does not exist)"
            fi
        done
    done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Evaluation v2 (val-calibrated threshold)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PHASE 2: EVALUATION v2 (val-calibrated threshold, zero leakage)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "  ═══ Dataset: ${dataset} ═══"
    
    for rep_idx in 0 1 2; do
        exp_id="trained_${dataset}_rep${rep_idx}"
        v2_exp_id="conplex_v2_${dataset}_rep${rep_idx}"
        data_dir="dataset/kinase_${dataset}"
        
        # Find the best model checkpoint
        ckpt_dir="${MODEL_SAVE_DIR}/${exp_id}"
        
        # Prefer _best_model.pt (final), fallback to latest epoch checkpoint
        ckpt=""
        if [ -f "${ckpt_dir}/${exp_id}_best_model.pt" ]; then
            ckpt="${ckpt_dir}/${exp_id}_best_model.pt"
        else
            ckpt=$(ls -t ${ckpt_dir}/${exp_id}_best_model_epoch*.pt 2>/dev/null | head -1)
        fi
        
        if [ -z "$ckpt" ]; then
            echo "  ⚠ No checkpoint found for ${exp_id} — skipping eval"
            continue
        fi
        
        echo ""
        echo "  ▶ ${v2_exp_id}"
        echo "    Checkpoint: $(basename $ckpt)"
        
        python eval_conplex_v2.py \
            --checkpoint "$ckpt" \
            --data-dir "$data_dir" \
            --exp-id "$v2_exp_id" \
            --device "$DEVICE" \
            --batch-size 256 \
            --output-dir "$RESULTS_V2_DIR" \
            --n-thresholds 2000
    done
done

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Aggregate results
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PHASE 3: AGGREGATE RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python aggregate_conplex_v2.py --results-dir "$RESULTS_V2_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo " Models:  ${MODEL_SAVE_DIR}/"
echo " Results: ${RESULTS_V2_DIR}/"
echo " Run 'python aggregate_conplex_v2.py' to re-aggregate anytime."
echo "═══════════════════════════════════════════════════════════════════"
