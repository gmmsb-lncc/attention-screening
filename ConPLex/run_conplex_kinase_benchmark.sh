#!/usr/bin/env bash
# ===========================================================================
# run_conplex_kinase_benchmark.sh
#
# Unified benchmark for ConPLex on DT-Kinase universal datasets.
# Optimized for RTX 4090 (24 GB VRAM) + multi-core CPU.
#
# PROTOCOL 1: Evaluate pretrained ConPLex (original weights) on kinase test
#             sets to measure native kinase prediction performance.
#
# PROTOCOL 2: Train ConPLex from scratch on each kinase dataset to create
#             kinase-specialist models, then evaluate on the same test sets.
#
# Usage:
#   bash run_conplex_kinase_benchmark.sh              # full run
#   PROTOCOL=1 bash run_conplex_kinase_benchmark.sh   # pretrained eval only
#   PROTOCOL=2 bash run_conplex_kinase_benchmark.sh   # train-from-scratch only
#   DATASET=non_human bash run_conplex_kinase_benchmark.sh  # single dataset
#
# Requirements:
#   - conda environment 'conplex' (created by setup_env.sh)
#   - kinase datasets in scaffolds_splits/output/{non_human,human}_*.tsv
#
# Output:
#   - Protocol 1: results/pretrained_{dataset}/results.json
#   - Protocol 2: results/trained_{dataset}_rep{N}/results.json
#                  best_models/trained_{dataset}_rep{N}/
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

# ── Configuration (override via env vars) ──────────────────────────────────
PROTOCOL="${PROTOCOL:-all}"                    # 1 | 2 | all
DATASET="${DATASET:-all}"                      # non_human | human | all | all_datasets
GPU="${GPU:-0}"                                # CUDA device ID
EPOCHS="${EPOCHS:-50}"                         # ConPLex paper uses 50
SEEDS="${SEEDS:-0 1 2}"                        # 3 replicatas (paper uses 5)
CONTRASTIVE="${CONTRASTIVE:-False}"            # Disable DUD-E contrastive by default
                                               # (kinase datasets don't have DUD-E decoys)
DRUG_FEAT="${DRUG_FEAT:-MorganFeaturizer}"
TARGET_FEAT="${TARGET_FEAT:-ProtBertFeaturizer}"
CHECKPOINT="${CHECKPOINT:-models/protbert_epoch3_state_dict.pt}"

# Source datasets: thesis scaffold splits (TSV format in scaffolds_splits/output/)
KINASE_DATA_ROOT="${KINASE_DATA_ROOT:-${REPO_ROOT}/scaffolds_splits/output}"

# ── Hardware-aware auto-tuning ─────────────────────────────────────────────
# Detect CPU count for DataLoader workers
N_CPUS=$(python3 -c "import os; print(os.cpu_count() or 4)" 2>/dev/null || echo 4)
# Use half the CPUs for workers (leave some for main process + OS)
NUM_WORKERS="${NUM_WORKERS:-$(( N_CPUS / 2 > 8 ? 8 : N_CPUS / 2 ))}"
# Clamp to at least 2
[ "${NUM_WORKERS}" -lt 2 ] && NUM_WORKERS=2

# Detect GPU VRAM for batch size auto-tuning
VRAM_MB=$(python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(${GPU})
    print(props.total_mem // (1024*1024))
else:
    print(0)
" 2>/dev/null || echo 0)

# Auto-tune batch size based on VRAM
# ConPLex model is very lightweight (~4M params), bottleneck is features
# RTX 4090 (24GB) → batch 256; RTX 3090 (24GB) → batch 256; RTX 3060 (12GB) → batch 128
if [ "${VRAM_MB}" -ge 20000 ]; then
    BATCH_SIZE="${BATCH_SIZE:-256}"
elif [ "${VRAM_MB}" -ge 10000 ]; then
    BATCH_SIZE="${BATCH_SIZE:-128}"
elif [ "${VRAM_MB}" -ge 6000 ]; then
    BATCH_SIZE="${BATCH_SIZE:-64}"
else
    BATCH_SIZE="${BATCH_SIZE:-32}"
fi

echo "================================================================="
echo " ConPLex × DT-Kinase Universal Benchmark"
echo "================================================================="
echo " Protocol:       ${PROTOCOL}"
echo " Datasets:       ${DATASET}"
echo " GPU:            ${GPU}  (${VRAM_MB} MB VRAM)"
echo " CPUs:           ${N_CPUS} total → ${NUM_WORKERS} DataLoader workers"
echo " Batch size:     ${BATCH_SIZE} (auto-tuned for VRAM)"
echo " Epochs:         ${EPOCHS}"
echo " Seeds:          ${SEEDS}"
echo " Contrastive:    ${CONTRASTIVE}"
echo " Drug feat:      ${DRUG_FEAT}"
echo " Target feat:    ${TARGET_FEAT}"
echo " Checkpoint:     ${CHECKPOINT}"
echo " Kinase data:    ${KINASE_DATA_ROOT}"
echo " ConPLex dir:    ${SCRIPT_DIR}"
echo "================================================================="
echo ""

# ── Performance env vars ───────────────────────────────────────────────────
# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
# Use all CPU cores for intra-op parallelism (PyTorch default = all cores)
export OMP_NUM_THREADS="${N_CPUS}"
export MKL_NUM_THREADS="${N_CPUS}"
# Disable tokenizer warnings in multiprocess
export TOKENIZERS_PARALLELISM=false
# Force GPU
export CUDA_VISIBLE_DEVICES="${GPU}"

# ── Activate conda env ─────────────────────────────────────────────────────
CONDA_SH=""
for p in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"; do
  [ -f "$p" ] && { CONDA_SH="$p"; break; }
done
[ -n "${CONDA_SH}" ] && source "${CONDA_SH}"
conda activate conplex 2>/dev/null || echo "[WARN] Could not activate 'conplex'"

cd "${SCRIPT_DIR}"

# ── Determine dataset list ─────────────────────────────────────────────────
if [ "${DATASET}" = "all" ] || [ "${DATASET}" = "all_datasets" ]; then
    DATASETS=(non_human human all)
else
    DATASETS=(${DATASET})
fi

# ── Step 0: Convert kinase CSVs to ConPLex format ──────────────────────────
echo "[STEP 0] Converting kinase datasets to ConPLex format..."
echo ""

for ds in "${DATASETS[@]}"; do
    DST_DIR="${SCRIPT_DIR}/dataset/kinase_${ds}"
    mkdir -p "${DST_DIR}"

    # Map split names to thesis TSV files
    # Thesis files: {non_human,human}_{train,val,test}.tsv
    # 'all' dataset: uses canonical universal_{train,val,test}.tsv
    for split in train val test; do
        DST="${DST_DIR}/${split}.csv"

        if [ "${ds}" = "all" ]; then
            SRC="${KINASE_DATA_ROOT}/universal_${split}.tsv"
        else
            SRC="${KINASE_DATA_ROOT}/${ds}_${split}.tsv"
        fi

        if [ ! -f "${SRC}" ]; then
            echo "[ERROR] Source not found: ${SRC}"
            exit 1
        fi

        if [ -f "${DST}" ] && [ "${DST}" -nt "${SRC}" ]; then
            echo "  [SKIP] ${ds}/${split}.csv (already converted)"
            continue
        fi

        python3 -c "
import pandas as pd
split_name = '${split}'
df = pd.read_csv('${SRC}', sep='\t')

# Convert thesis format → ConPLex format
out = pd.DataFrame()
out['SMILES'] = df['canonical_smiles']
out['Target Sequence'] = df['seq']
out['Label'] = df['label'].astype(int)
out = out.dropna().reset_index(drop=True)
out.to_csv('${DST}', index=True)
n_pos = int(out['Label'].sum())
n_neg = len(out) - n_pos
print(f'  [{split_name:5s}] {len(out):7d} pairs | pos={n_pos:6d} neg={n_neg:6d} | → ${DST}')
"
    done
done

echo ""

# ── PROTOCOL 1: Evaluate pretrained model on kinase test sets ──────────────
if [ "${PROTOCOL}" = "1" ] || [ "${PROTOCOL}" = "all" ]; then
    echo "================================================================="
    echo " PROTOCOL 1: Pretrained ConPLex → Kinase Test Sets"
    echo "================================================================="
    echo ""

    for ds in "${DATASETS[@]}"; do
        DATA_DIR="${SCRIPT_DIR}/dataset/kinase_${ds}"
        EXP_ID="pretrained_${ds}"

        echo "[P1] Evaluating on ${ds}..."
        python3 eval_conplex.py \
            --checkpoint "${CHECKPOINT}" \
            --data-dir "${DATA_DIR}" \
            --exp-id "${EXP_ID}" \
            --drug-featurizer "${DRUG_FEAT}" \
            --target-featurizer "${TARGET_FEAT}" \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${NUM_WORKERS}" \
            --device "${GPU}" \
            --output-dir "./results"
        echo ""
    done

    echo "[P1] Done. Results:"
    for ds in "${DATASETS[@]}"; do
        RES="./results/pretrained_${ds}/results.json"
        if [ -f "${RES}" ]; then
            echo "  ${ds}: $(cat "${RES}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'AUROC={r[\"AUROC\"]:.4f}  AUPRC={r[\"AUPRC\"]:.4f}  n={r[\"n_test_samples\"]}')
")"
        fi
    done
    echo ""
fi

# ── PROTOCOL 2: Train from scratch on each kinase dataset ─────────────────
if [ "${PROTOCOL}" = "2" ] || [ "${PROTOCOL}" = "all" ]; then
    echo "================================================================="
    echo " PROTOCOL 2: Train ConPLex from Scratch on Kinase Datasets"
    echo "================================================================="
    echo ""

    for ds in "${DATASETS[@]}"; do
        DATA_DIR="${SCRIPT_DIR}/dataset/kinase_${ds}"

        for seed in ${SEEDS}; do
            EXP_ID="trained_${ds}_rep${seed}"

            echo "[P2] Training on ${ds} (seed=${seed})..."

            # Generate per-dataset config
            CONFIG_TMP="${SCRIPT_DIR}/configs/kinase_${ds}_config.yaml"
            cat > "${CONFIG_TMP}" << YAML
task: kinase_${ds}
contrastive_split: within

drug_featurizer: ${DRUG_FEAT}
target_featurizer: ${TARGET_FEAT}
model_architecture: SimpleCoembeddingNoSigmoid
latent_dimension: 1024
latent_distance: "Cosine"

batch_size: ${BATCH_SIZE}
contrastive_batch_size: 256
shuffle: True
num_workers: ${NUM_WORKERS}

epochs: ${EPOCHS}
every_n_val: 1
lr: 1e-4
lr_t0: 10
contrastive: ${CONTRASTIVE}
clr: 1e-5
clr_t0: 10
margin_fn: 'tanh_decay'
margin_max: 0.25
margin_t0: 10

replicate: ${seed}
device: ${GPU}
verbosity: 2

wandb_save: False
log_file: ./logs/${EXP_ID}.log
model_save_dir: ./best_models
YAML

            # Train with monkey-patched get_task_dir
            WANDB_MODE=disabled python3 -c "
import sys, os
os.chdir('${SCRIPT_DIR}')
sys.path.insert(0, '.')

# Monkey-patch get_task_dir to include kinase datasets
import src.data as data_module
from pathlib import Path
_orig_get_task_dir = data_module.get_task_dir
def _patched_get_task_dir(task_name):
    if task_name.startswith('kinase_'):
        return Path('./dataset/' + task_name).resolve()
    return _orig_get_task_dir(task_name)
data_module.get_task_dir = _patched_get_task_dir

# Enable CUDA optimizations
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from train_DTI import main, parser

sys.argv = [
    'train_DTI.py',
    '--exp-id', '${EXP_ID}',
    '--config', '${CONFIG_TMP}',
    '--d', '${GPU}',
    '--r', '${seed}',
    '--epochs', '${EPOCHS}',
    '--batch-size', '${BATCH_SIZE}',
]

main()
"

            echo "[P2] Training ${ds} rep${seed} done."

            # Evaluate on test set using best model
            BEST_MODEL=$(ls -t ./best_models/${EXP_ID}/${EXP_ID}_best_model*.pt 2>/dev/null | head -1 || true)
            if [ -n "${BEST_MODEL}" ]; then
                echo "[P2] Evaluating best model: ${BEST_MODEL}"
                python3 eval_conplex.py \
                    --checkpoint "${BEST_MODEL}" \
                    --data-dir "${DATA_DIR}" \
                    --exp-id "${EXP_ID}" \
                    --drug-featurizer "${DRUG_FEAT}" \
                    --target-featurizer "${TARGET_FEAT}" \
                    --batch-size "${BATCH_SIZE}" \
                    --num-workers "${NUM_WORKERS}" \
                    --device "${GPU}" \
                    --output-dir "./results"
            else
                echo "[WARN] No best model found for ${EXP_ID}"
            fi
            echo ""
        done
    done

    echo "[P2] Done. Results:"
    for ds in "${DATASETS[@]}"; do
        for seed in ${SEEDS}; do
            RES="./results/trained_${ds}_rep${seed}/results.json"
            if [ -f "${RES}" ]; then
                echo "  ${ds} rep${seed}: $(cat "${RES}" | python3 -c "
import sys, json
r = json.load(sys.stdin)
print(f'AUROC={r[\"AUROC\"]:.4f}  AUPRC={r[\"AUPRC\"]:.4f}')
")"
            fi
        done
    done
    echo ""
fi

# ── Final Summary ──────────────────────────────────────────────────────────
echo "================================================================="
echo " SUMMARY"
echo "================================================================="
echo ""
echo " Results directory: ${SCRIPT_DIR}/results/"
echo ""

python3 -c "
import json, os, glob
from pathlib import Path

results_dir = Path('${SCRIPT_DIR}/results')
rows = []
for f in sorted(results_dir.glob('*/results.json')):
    with open(f) as fh:
        r = json.load(fh)
    exp = f.parent.name
    protocol = 'P1-pretrained' if 'pretrained' in exp else 'P2-trained'
    dataset = exp.replace('pretrained_', '').replace('trained_', '').rsplit('_rep', 1)[0]
    seed = exp.rsplit('_rep', 1)[1] if '_rep' in exp else '-'
    rows.append((protocol, dataset, seed, r.get('AUROC', 0), r.get('AUPRC', 0), r.get('n_test_samples', 0)))

if rows:
    print(f' {\"Protocol\":<15s} {\"Dataset\":<12s} {\"Seed\":<5s} {\"AUROC\":<8s} {\"AUPRC\":<8s} {\"N_test\":<8s}')
    print(f' {\"-\"*15} {\"-\"*12} {\"-\"*5} {\"-\"*8} {\"-\"*8} {\"-\"*8}')
    for p, d, s, au, ap, n in rows:
        print(f' {p:<15s} {d:<12s} {s:<5s} {au:<8.4f} {ap:<8.4f} {n:<8d}')
else:
    print(' No results found yet.')
print()
"

echo "================================================================="
echo " Benchmark complete."
echo "================================================================="
