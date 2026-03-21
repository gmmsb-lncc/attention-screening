# DrugBAN Baseline Integration

External baseline using [DrugBAN](https://github.com/peizhenbai/DrugBAN) (NeurIPS, 2022) for fair comparison with DT-Kinase.

DrugBAN is a **bilinear attention network** for drug-target interaction prediction: GCN encodes molecular graphs, CNN encodes protein sequences, and bilinear attention captures pairwise interactions with optional domain adaptation (CDAN).

## Quick Start

```bash
# 1. Setup environment
bash setup_env.sh

# 2. Activate
conda activate drugban

# 3. Run baseline (auto-prepares data)
python run_baseline.py --dataset non_human

# For quick testing
python run_baseline.py --dataset non_human --max-epoch 5 --seeds 42
```

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Creates conda env, installs deps, clones DrugBAN |
| `prepare_data.py` | Converts thesis scaffold splits to `SMILES,Protein,Y` CSV |
| `run_baseline.py` | Full training wrapper with fair evaluation protocol |
| `configs/kinase.yaml` | Hyperparameters for kinase datasets |

## Pipeline

```
1. prepare_data.py (auto-called by run_baseline.py if needed)
   scaffolds_splits/output/ → datasets/kinase/{dataset}/scaffold/{train,val,test}.csv

2. run_baseline.py (per seed):
   a. Build molecular graphs (GCN, 75-d atom features → 128-d)
   b. Encode protein sequences (CNN, kernels [3,6,9])
   c. Train DrugBAN (bilinear attention + optional CDAN domain adaptation)
   d. Model selection by validation AUROC
   e. Fair evaluation: MCC-optimal threshold on validation set
   f. Apply threshold to train/val/test → metrics for all splits

3. Aggregate across 5 canonical seeds → drugban_results.json
```

## Fair Evaluation Protocol

| Aspect | DrugBAN Original | Our Wrapper (Fair) |
|--------|-----------------|-------------------|
| Threshold | Test-set F1-optimal | Val-set MCC-optimal |
| Model selection | Test AUROC | Val AUROC |
| Comparison | Potentially inflated | Fair comparison with DT-Kinase |

Both protocols are recorded in the output JSON for transparency.

## Architecture

```
DrugBAN
├── Drug branch:   SMILES → GCN(75→128³) → molecular graph features
│
├── Protein branch: Sequence → integer encoding → CNN(128, [3,6,9]) → protein features
│
├── Interaction:    BAN(drug, protein, heads=2) → 256-d
│
├── Domain Adapt:   CDAN with random layer (align source/target distributions)
│
└── Decoder:        MLP(256→512→128→2) → 2-class classification
```

## Output

Results saved to `results/{dataset}/drugban_results.json`:

```json
{
  "model": "DrugBAN",
  "dataset": "non_human",
  "split": "scaffold",
  "aggregate": {
    "train": { "mcc": {"mean": ..., "std": ...}, "auroc": {...}, ... },
    "val":   { "mcc": {"mean": ..., "std": ...}, "auroc": {...}, ... },
    "test":  { "mcc": {"mean": ..., "std": ...}, "auroc": {...}, ... }
  },
  "per_seed": [ ... ]
}
```

## Citation

```bibtex
@inproceedings{bai2023drugban,
  title     = {Interpretable bilinear attention network with domain adaptation
               improves drug-target prediction},
  author    = {Bai, Peizhen and Miljkovi{\'c}, Filip and John, Bino and Lu, Haiping},
  booktitle = {Nature Machine Intelligence},
  year      = {2023},
}
```
