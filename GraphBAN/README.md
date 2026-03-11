# GraphBAN Baseline Integration

External baseline using [GraphBAN](https://github.com/HamidHadipour/GraphBAN) (Nature Communications, 2025) for fair comparison with DT-Kinase.

GraphBAN is an **inductive graph-based approach** for compound-protein interaction (CPI) prediction using knowledge distillation: a teacher GAE on the bipartite CPI graph produces structural embeddings that guide a student model combining GCN (drug), CNN (protein), ESM-1b, ChemBERTa, and bilinear attention with domain adaptation.

## Quick Start

```bash
# 1. Setup environment
bash setup_env.sh

# 2. Activate
conda activate graphban

# 3. Run baseline (auto-prepares data)
python run_baseline.py --dataset non_human

# For quick testing
python run_baseline.py --dataset non_human --max-epoch 5 --seeds 42
```

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Creates conda env, installs deps, clones GraphBAN |
| `prepare_data.py` | Converts thesis scaffold splits to `SMILES,Protein,Y` CSV |
| `run_baseline.py` | Full training wrapper with fair evaluation protocol |
| `dgl_compat.py` | DGL graphbolt compatibility shim |
| `configs/kinase_inductive.yaml` | Hyperparameters for kinase datasets |

## Pipeline

```
1. prepare_data.py
   scaffolds_splits/output/ → datasets/kinase/{dataset}/scaffold/{train,val,test}.csv

2. run_baseline.py (per seed):
   a. Extract ESM-1b protein features (1280-d, mean pool) — cached
   b. Extract ChemBERTa drug features (384-d, CLS token) — cached
   c. teacher_gae.py → teacher_embeddings.parquet (GAE on bipartite CPI graph)
   d. Train GraphBAN (inductive + domain adaptation)
   e. Fair evaluation: MCC-optimal threshold on validation set
   f. Apply to test → final metrics

3. Aggregate across 5 canonical seeds → graphban_results.json
```

## Fair Evaluation Protocol

This wrapper fixes a test-leakage issue in GraphBAN's original inductive mode:

| Aspect | GraphBAN Original | Our Wrapper (Fair) |
|--------|------------------|--------------------|
| Val dataloader | `test_dataset` (leakage!) | `val_dataset` (proper) |
| Model selection | Test AUROC | Val AUROC |
| Threshold | Test-set F1-optimal | Val-set MCC-optimal |
| Comparison | Inflated metrics | Fair comparison with DT-Kinase |

Both protocols are recorded in the output JSON for transparency.

## Architecture

```
GraphBAN
├── Drug branch:   SMILES → GCN(75→128³) → mol graph features
│                  SMILES → ChemBERTa-77M-MTR → 384-d CLS token → Linear(384→128)
│                  Fusion: molFusion(GCN, ChemBERTa)
│
├── Protein branch: Sequence → integer encoding → CNN(128, [3,6,9]) → protein features
│                   Sequence → ESM-1b → 1280-d mean pool → Linear(1280→128)
│                   Fusion: proFusion(CNN, ESM)
│
├── Interaction:    BAN(drug_fused, protein_fused, heads=2) → 256-d
│
├── Teacher KD:     Bipartite CPI graph → GAE(SAGEConv³) → 256-d link embeddings
│                   MSE loss between BAN output and teacher embeddings
│
├── Domain Adapt:   CDAN with random layer (align source/target distributions)
│
└── Decoder:        MLP(256→512→128→2) → 2-class classification
```

## Output

Results saved to `results/{dataset}/graphban_results.json`:

```json
{
  "model": "GraphBAN",
  "dataset": "non_human",
  "split": "scaffold",
  "methodology": { ... },
  "aggregate": {
    "mcc": {"mean": 0.XXX, "std": 0.XXX},
    "auroc": {"mean": 0.XXX, "std": 0.XXX},
    ...
  },
  "per_seed": [ ... ]
}
```

## Citation

```bibtex
@article{Hadipour2025graphban,
  title   = {GraphBAN: An Inductive Graph-Based Approach for Enhanced Prediction
             of Compound-Protein Interactions},
  author  = {Hadipour, Hamid and Li, Yan Yi and Sun, Yan and Deng, Chutong
             and Lac, Leann and Davis, Rebecca and Cardona, Silvia T and Hu, Pingzhao},
  journal = {Nature Communications},
  year    = {2025},
}
```
