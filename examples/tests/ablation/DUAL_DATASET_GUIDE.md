# Dual-Dataset Ablation Study

This directory now supports running ablation studies on **two datasets**:

## Datasets

### 1. Non-Human Kinases (`results_non_human`)
- **TSV**: `${PROJECT_ROOT}/tests/datasets/kinase_non_human_compounds.tsv`
- **Embeddings**: `${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2/`
- **Size**: 15,616 interactions, 299 proteins, 8,131 ligands
- **Status**: ✅ Complete (baseline results)

### 2. Human Kinases (`results_human`)
- **TSV**: `/data/docktkinase/datasets/kinase_human_compounds.tsv`
- **Embeddings**: `/data/docktkinase/results/protein_model_benchmark_human_v2/`
- **Size**: To be determined
- **Status**: 🔄 Ready to run

## Directory Structure

```
ablation/
├── README.md                          # Main documentation
├── DUAL_DATASET_GUIDE.md             # This file
├── run_ablation_study.py             # Orchestrator script
│
├── classification/
│   ├── data/
│   │   ├── results_non_human/        # Non-human data & results
│   │   │   ├── processed/            # Extracted proteins, ligands, interactions
│   │   │   ├── embeddings/           # Morgan FP, One-Hot encodings
│   │   │   └── combinations/         # C1-C4 combined features
│   │   └── results_human/            # Human data & results (same structure)
│   │
│   ├── scripts/                      # All scripts support --results-suffix
│   │   ├── 01_extract_data.py
│   │   ├── 02_generate_morgan_fingerprints.py
│   │   ├── 03_generate_aac_dpc_encoding.py
│   │   ├── 04_create_combinations.py
│   │   ├── 05_run_classification.py
│   │   └── 06_visualize_results.py
│   │
│   ├── results_non_human/            # Non-human classification results
│   │   ├── classification_results.json
│   │   ├── classification_summary.csv
│   │   └── figures/
│   │
│   └── results_human/                # Human classification results
│       ├── classification_results.json
│       ├── classification_summary.csv
│       └── figures/
│
└── regression/
    ├── data/
    │   ├── results_non_human/        # Non-human regression data
    │   └── results_human/            # Human regression data
    │
    ├── scripts/                      # All scripts support --results-suffix
    │   ├── 01_extract_data_regression.py
    │   ├── 02_run_regression.py
    │   ├── 03_visualize_regression_results.py
    │   └── consolidate_checkpoints.py
    │
    ├── results_non_human/            # Non-human regression results
    │   ├── regression_results.json
    │   ├── regression_summary.csv
    │   └── figures/
    │
    └── results_human/                # Human regression results
        ├── regression_results.json
        ├── regression_summary.csv
        └── figures/
```

## Usage

### Method 1: Orchestrator Script (Recommended)

Run both pipelines for a specific dataset:

```bash
# Non-human dataset (default, already complete)
python run_ablation_study.py --dataset non_human

# Human dataset (new)
python run_ablation_study.py --dataset human

# Both datasets sequentially
python run_ablation_study.py --dataset both
```

Run specific tasks:

```bash
# Only classification for human
python run_ablation_study.py --dataset human --task classification

# Only regression for both
python run_ablation_study.py --dataset both --task regression
```

### Method 2: Manual Script Execution

All scripts now accept `--results-suffix` to separate datasets:

#### Classification Pipeline

```bash
cd ablation/classification/scripts

# Non-human (default)
python 01_extract_data.py --results-suffix results_non_human
python 02_generate_morgan_fingerprints.py --results-suffix results_non_human
python 03_generate_aac_dpc_encoding.py --results-suffix results_non_human
python 04_create_combinations.py --results-suffix results_non_human \
    --embeddings-dir ${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2
python 05_run_classification.py --results-suffix results_non_human
python 06_visualize_results.py --results-suffix results_non_human

# Human (new)
python 01_extract_data.py \
    --tsv-path /data/docktkinase/datasets/kinase_human_compounds.tsv \
    --results-suffix results_human
python 02_generate_morgan_fingerprints.py --results-suffix results_human
python 03_generate_aac_dpc_encoding.py --results-suffix results_human
python 04_create_combinations.py --results-suffix results_human \
    --embeddings-dir /data/docktkinase/results/protein_model_benchmark_human_v2
python 05_run_classification.py --results-suffix results_human
python 06_visualize_results.py --results-suffix results_human
```

#### Regression Pipeline

```bash
cd ablation/regression/scripts

# Non-human
python 01_extract_data_regression.py --results-suffix results_non_human
nohup python -u 02_run_regression.py --results-suffix results_non_human \
    --embeddings-dir ${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2 \
    > ../results_non_human/regression.log 2>&1 &
python consolidate_checkpoints.py --results-suffix results_non_human
python 03_visualize_regression_results.py --results-suffix results_non_human

# Human
python 01_extract_data_regression.py \
    --tsv-path /data/docktkinase/datasets/kinase_human_compounds.tsv \
    --results-suffix results_human
nohup python -u 02_run_regression.py --results-suffix results_human \
    --embeddings-dir /data/docktkinase/results/protein_model_benchmark_human_v2 \
    > ../results_human/regression.log 2>&1 &
python consolidate_checkpoints.py --results-suffix results_human
python 03_visualize_regression_results.py --results-suffix results_human
```

## Command-Line Arguments

All scripts accept these arguments:

- `--tsv-path`: Path to input TSV file (default: non_human)
- `--embeddings-dir`: Path to ESM-2/SMI-TED embeddings (default: non_human_v2)
- `--results-suffix`: Directory suffix for results (`results_non_human` or `results_human`)

## Configuration Summary

| Parameter | Non-Human | Human |
|-----------|-----------|-------|
| TSV Path | `${PROJECT_ROOT}/tests/datasets/kinase_non_human_compounds.tsv` | `/data/docktkinase/datasets/kinase_human_compounds.tsv` |
| Embeddings | `${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2/` | `/data/docktkinase/results/protein_model_benchmark_human_v2/` |
| Results Suffix | `results_non_human` | `results_human` |

## Comparison Analysis

After running both datasets, you can compare results:

```python
import pandas as pd

# Load both classification results
df_nh = pd.read_csv('classification/results_non_human/classification_summary.csv')
df_h = pd.read_csv('classification/results_human/classification_summary.csv')

# Compare mean ROC-AUC
print("Non-Human:", df_nh['test_auc'].mean())
print("Human:", df_h['test_auc'].mean())

# Load regression results
df_reg_nh = pd.read_csv('regression/results_non_human/regression_summary.csv')
df_reg_h = pd.read_csv('regression/results_human/regression_summary.csv')

# Compare R²
print("Non-Human R²:", df_reg_nh['test_r2'].mean())
print("Human R²:", df_reg_h['test_r2'].mean())
```

## Expected Differences

### Non-Human vs Human

- **Dataset Size**: Human typically larger
- **Performance**: May differ due to:
  - Protein sequence diversity
  - Ligand distribution
  - Binding affinity ranges
  - Evolutionary distance from training data

### Research Questions

1. Do learned representations (ESM-2) generalize better to human kinases?
2. Are handcrafted features more robust across species?
3. Which representation shows less performance drop?
4. Are certain ESM-2 model sizes better for cross-species transfer?

## Status Tracking

### Non-Human Dataset
- ✅ Classification: Complete (50 experiments)
- 🔄 Regression: In progress (PID 3228787)

### Human Dataset
- ⏳ Classification: Ready to run
- ⏳ Regression: Ready to run

## Notes

- All scripts maintain backward compatibility (default to `results_non_human`)
- Results are completely isolated between datasets
- Same random seeds used for reproducibility
- Visualization scripts create dataset-specific figures

---

**Last Updated**: January 17, 2026  
**Status**: Dual-dataset support implemented ✅
