# Regression Ablation Study

## Overview

This directory contains the regression ablation study for predicting **pChEMBL values** (binding affinity) from protein-ligand embeddings. This study parallels the classification ablation study but uses continuous target values instead of binary labels.

## Task

**Regression Task**: Predict pChEMBL value (continuous, typically 4-10 range) given:
- Protein embedding (ESM-2: 8M, 150M, 3B)
- Ligand embedding (concatenated with protein)

## Methodology

### Data Split
- **Split Strategy**: Random 80/10/10
- **Training**: 80% of data
- **Validation**: 10% of data
- **Test**: 10% of data
- **Random Seed**: 420 (same as baseline)
- **Multiple Seeds**: [42, 123, 456, 789, 1024] for statistical significance

### Models
1. **KNN Regressor**
   - k=5 neighbors
   - Distance-weighted
   - Cosine metric
   
2. **MLP Regressor**
   - Architecture: (256, 128, 64) hidden layers
   - Activation: ReLU
   - Optimizer: Adam
   - Early stopping: 20 epochs patience
   - Max iterations: 500

### ESM-2 Models Evaluated
| Model | Parameters | Embedding Dim |
|-------|------------|---------------|
| esm2_t6_8M_UR50D | 8M | 320 |
| esm2_t30_150M_UR50D | 150M | 640 |
| esm2_t36_3B_UR50D | 3B | 2560 |

### Evaluation Metrics
- **R² (Coefficient of Determination)**: Primary metric (0-1, higher is better)
- **RMSE (Root Mean Squared Error)**: Error magnitude in original scale
- **MAE (Mean Absolute Error)**: Average absolute error
- **Pearson Correlation**: Linear correlation between predictions and true values
- **Spearman Correlation**: Rank correlation (robust to outliers)
- **CCC (Concordance Correlation Coefficient)**: MCC equivalent for regression
  - Lin's Concordance: measures accuracy AND precision
  - Range: -1 to 1 (1 = perfect agreement)
  - Formula: CCC = 2·cov(y_true, y_pred) / (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))²)

## Directory Structure

```
regression/
├── data/
│   └── processed/
│       ├── proteins.csv           # Unique proteins
│       ├── ligands.csv            # Unique ligands
│       ├── interactions_regression.csv  # Interactions with pChEMBL values
│       └── index_mapping.json     # ID mappings
├── scripts/
│   ├── 01_extract_data_regression.py   # Data extraction
│   ├── 02_run_regression.py            # Main experiment runner
│   └── 03_visualize_regression_results.py  # Visualization
├── results/
│   ├── regression_results.json    # Detailed results
│   └── regression_summary.csv     # Summary table
├── figures/
│   ├── regression_metrics_comparison.png
│   ├── regression_r2_focus.png
│   ├── regression_correlation_comparison.png
│   ├── regression_error_metrics.png
│   ├── regression_heatmap_summary.png
│   └── regression_summary_table.csv
└── README.md
```

## Running the Experiments

```bash
# Activate virtual environment
cd ${PROJECT_ROOT}
source env/bin/activate

# 1. Extract data (reuses classification data if available)
python ablation/regression/scripts/01_extract_data_regression.py

# 2. Run regression experiments
python ablation/regression/scripts/02_run_regression.py

# 3. Generate visualizations
python ablation/regression/scripts/03_visualize_regression_results.py
```

## Expected Results

Based on classification results, we expect:
- **R² scores**: 0.4-0.7 (regression is harder than classification)
- **ESM-2 3B**: Best performance due to richer embeddings
- **MLP**: Better than KNN for regression (can learn non-linear relationships)
- **RMSE**: ~0.7-1.0 pChEMBL units

## Comparison with Classification

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Target | Binary (active/inactive) | Continuous (pChEMBL) |
| Threshold | pChEMBL ≥ 6.5 = active | N/A |
| Main Metric | ROC-AUC | R² |
| Expected Performance | ~0.95 AUC | ~0.5-0.7 R² |

## Notes

- Regression is inherently harder than classification because it requires predicting exact values
- The same embeddings are used for both tasks, allowing direct comparison
- pChEMBL values follow approximately normal distribution (mean ~6.5)
