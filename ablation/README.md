# Ablation Studies for Protein-Ligand Binding Prediction

## Overview

This directory contains comprehensive ablation studies designed to systematically evaluate the impact of different representation choices on protein-ligand binding prediction tasks. The studies compare learned representations (ESM-2, SMI-TED) against traditional handcrafted features (One-Hot encoding, Morgan fingerprints) for both classification and regression tasks.

## Motivation

Understanding which molecular representations contribute most to predictive performance is crucial for:
- **Model Selection**: Identifying optimal representation strategies for different scenarios
- **Computational Efficiency**: Balancing accuracy with computational cost
- **Interpretability**: Understanding what information drives predictions
- **Generalization**: Evaluating robustness across different data splits and random seeds

## Study Design

### Common Elements

Both classification and regression studies share:
- **Dataset**: 15,616 non-human kinase protein-ligand interactions
  - 299 unique proteins
  - 8,131 unique ligands
  - pChEMBL values ranging from 3.95 to 11.00 (mean: 6.46)
- **Data Split**: Random 80/10/10 (train/validation/test)
- **Statistical Rigor**: 5 random seeds [42, 123, 456, 789, 1024] for significance testing
- **Models**: KNN and MLP (shallow vs deep learning)
- **ESM-2 Variants**: 8M (320-dim), 150M (640-dim), 3B (2560-dim) parameters

### Representation Combinations

We evaluate 4 systematic combinations for classification:

| Code | Protein Representation | Ligand Representation | Complexity | Total Dimensions |
|------|------------------------|------------------------|------------|------------------|
| **C1** | ESM-2 (learned) | SMI-TED (learned) | Highest | 1088-3328 |
| **C2** | ESM-2 (learned) | Morgan FP (handcrafted) | Mixed | 2368-4608 |
| **C3** | One-Hot (simple) | SMI-TED (learned) | Mixed | 1088 |
| **C4** | One-Hot (simple) | Morgan FP (handcrafted) | Lowest | 2347 |

**Hypothesis**: Learned representations (C1) should outperform handcrafted features (C4), with mixed approaches (C2, C3) showing intermediate performance.

## Directory Structure

```
ablation/
├── README.md                          # This file (overview)
├── classification/                    # Binary classification study
│   ├── README.md                      # Classification-specific documentation
│   ├── data/
│   │   ├── processed/                 # Extracted proteins, ligands, interactions
│   │   ├── embeddings/                # Morgan FP, One-Hot encodings
│   │   └── combinations/              # C1-C4 combined feature matrices
│   ├── scripts/
│   │   ├── 01_extract_data.py        # Data extraction from TSV
│   │   ├── 02_generate_morgan_fingerprints.py  # RDKit Morgan FP generation
│   │   ├── 03_generate_aac_dpc_encoding.py     # One-Hot encoding
│   │   ├── 04_create_combinations.py  # Combine protein+ligand features
│   │   ├── 05_run_classification.py   # Run KNN/MLP experiments
│   │   ├── 06_visualize_results.py    # Generate plots and tables
│   │   └── add_mcc_to_results.py      # Retroactively add MCC metric
│   ├── results/
│   │   ├── classification_results.json  # Detailed results with confusion matrices
│   │   └── classification_summary.csv   # Aggregated metrics
│   └── figures/
│       ├── ablation_comparison_all_metrics.png  # 2×3 grid: 6 metrics
│       ├── ablation_auc_comparison.png          # ROC-AUC focus
│       ├── protein_ligand_contribution_heatmap.png  # 2×2 heatmap
│       └── ablation_summary_table.csv           # Mean ± std table
│
└── regression/                        # Continuous pChEMBL prediction study
    ├── README.md                      # Regression-specific documentation
    ├── CHANGELOG.md                   # Version history and improvements
    ├── data/
    │   └── processed/                 # Same proteins/ligands, pChEMBL targets
    ├── scripts/
    │   ├── 01_extract_data_regression.py      # Data extraction
    │   ├── 02_run_regression.py               # Run KNN/MLP regressors
    │   ├── 03_visualize_regression_results.py # Generate plots
    │   └── consolidate_checkpoints.py         # Merge checkpoint files
    ├── results/
    │   ├── regression_results.json            # Detailed results
    │   ├── regression_summary.csv             # Aggregated metrics
    │   └── regression_results_*_seed*.json    # Checkpoints per seed
    └── figures/
        ├── regression_metrics_comparison.png  # 2×3 grid: R², RMSE, MAE, etc.
        ├── regression_r2_focus.png            # R² detailed comparison
        ├── regression_correlation_comparison.png  # Pearson vs Spearman
        ├── regression_error_metrics.png       # RMSE vs MAE
        └── regression_heatmap_summary.png     # Performance heatmap
```

## Tasks

### 1. Classification Task

**Objective**: Predict binary binding activity (active/inactive)
- **Threshold**: pChEMBL ≥ 6.5 = active, < 6.5 = inactive
- **Class Balance**: ~60% active, ~40% inactive

**Metrics** (all range 0-1):
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area Under Receiver Operating Characteristic curve
- **MCC**: Matthews Correlation Coefficient (robust to class imbalance)

**Expected Performance**: ROC-AUC ~0.93-0.96

### 2. Regression Task

**Objective**: Predict continuous pChEMBL values (binding affinity)
- **Range**: 3.95-11.00 (higher = stronger binding)
- **Distribution**: Approximately normal, mean 6.46, std 1.53

**Metrics**:
- **R²**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Squared Error (pChEMBL units, lower is better)
- **MAE**: Mean Absolute Error (pChEMBL units, lower is better)
- **Pearson r**: Linear correlation (-1 to 1)
- **Spearman r**: Rank correlation (-1 to 1)
- **CCC**: Concordance Correlation Coefficient (Lin's formula, -1 to 1)
  - **MCC equivalent for regression**
  - Formula: `CCC = 2·cov / (var_true + var_pred + (mean_true - mean_pred)²)`
  - Measures both accuracy (correlation) and precision (agreement)

**Expected Performance**: R² ~0.73-0.80 (regression is harder than classification)

## Key Features

### Statistical Rigor
- **Multiple Seeds**: 5 random seeds ensure reproducibility and statistical significance
- **Confidence Intervals**: Mean ± standard deviation reported for all metrics
- **Fair Comparison**: Identical data splits across all representation combinations

### Robustness Improvements
- **Checkpoint System** (regression): Results saved after each seed to prevent data loss
- **Real-time Logging**: `sys.stdout.flush()` for immediate output visibility
- **Incremental Saves**: JSON and CSV checkpoints for long-running experiments

### Advanced Metrics
- **MCC (Classification)**: Added retroactively from confusion matrices
  - More robust than accuracy for imbalanced datasets
  - Range: -1 (total disagreement) to 1 (perfect prediction)
- **CCC (Regression)**: Lin's Concordance Correlation Coefficient
  - Regression equivalent of MCC
  - Penalizes systematic bias unlike R²

### Visualization Standards
- **Consistent Layout**: 2×3 grid (6 metrics) for both studies
- **Normalized Y-axis**: 0.0-1.0 for all normalized metrics
- **Error Bars**: Standard deviation across 5 seeds
- **Color Coding**: Consistent colors for KNN (blue) vs MLP (red)

## Expected Results

### Classification Findings
| Combination | Expected ROC-AUC | Expected MCC | Rationale |
|-------------|------------------|--------------|-----------|
| C1 (ESM+SMITED) | **0.94-0.96** | **0.76-0.78** | Both learned, richest information |
| C2 (ESM+Morgan) | 0.94-0.96 | 0.75-0.77 | Learned protein, simple ligand |
| C3 (OneHot+SMITED) | 0.94-0.95 | 0.76-0.77 | Simple protein, learned ligand |
| C4 (OneHot+Morgan) | 0.93-0.95 | 0.74-0.76 | Both simple, baseline |

**Key Insight**: All combinations perform well (ROC-AUC >0.93), suggesting:
- Protein-ligand binding signal is strong
- Even simple representations capture essential information
- ESM-2 provides marginal but consistent improvement

### Regression Findings
| Model | Expected R² (KNN) | Expected R² (MLP) | Expected CCC |
|-------|-------------------|-------------------|--------------|
| ESM-2 8M | 0.73-0.76 | 0.73-0.78 | 0.70-0.75 |
| ESM-2 150M | 0.75-0.78 | 0.76-0.80 | 0.72-0.77 |
| ESM-2 3B | 0.77-0.80 | 0.78-0.82 | 0.74-0.79 |

**Key Insight**: Larger ESM-2 models improve regression performance more than classification, likely due to richer embedding space capturing subtle affinity differences.

## Comparison: Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Task** | Binary (active/inactive) | Continuous (pChEMBL) |
| **Threshold** | pChEMBL ≥ 6.5 | N/A |
| **Difficulty** | Easier (binary decision) | Harder (exact values) |
| **Main Metric** | ROC-AUC | R² |
| **Robustness Metric** | MCC | CCC |
| **Performance** | ~0.95 AUC | ~0.75 R² |
| **Model Preference** | MLP ≈ KNN | MLP > KNN |

**Why Regression is Harder**:
- Requires predicting exact continuous values, not just class boundaries
- More sensitive to outliers and noise
- Evaluation metrics (R², RMSE) are stricter than classification metrics

## Running the Studies

### Prerequisites
```bash
# Activate virtual environment
cd /media/leon/ssd2tb/docktkinase
source env/bin/activate

# Verify ESM-2 embeddings exist
ls results/protein_model_benchmark_non_human_v2/esm2_*/build/
```

### Classification Pipeline
```bash
cd ablation/classification/scripts

# 1. Extract data (proteins, ligands, interactions)
python 01_extract_data.py

# 2. Generate Morgan fingerprints
python 02_generate_morgan_fingerprints.py

# 3. Generate One-Hot protein encodings
python 03_generate_aac_dpc_encoding.py

# 4. Create all C1-C4 combinations
python 04_create_combinations.py

# 5. Run experiments (KNN + MLP, 5 seeds × 10 combinations = 100 experiments)
python 05_run_classification.py

# 6. (Optional) Add MCC if not present
python add_mcc_to_results.py

# 7. Generate visualizations
python 06_visualize_results.py
```

### Regression Pipeline
```bash
cd ablation/regression/scripts

# 1. Extract data (reuses classification data + pChEMBL targets)
python 01_extract_data_regression.py

# 2. Run experiments with checkpoints (3 models × 5 seeds × 2 regressors = 30 experiments)
# Use nohup for long runs (~3-5 hours)
nohup python -u 02_run_regression.py > ../regression.log 2>&1 &

# 3. Monitor progress in real-time
tail -f ../regression.log

# 4. Check process status
ps aux | grep "python.*regression"

# 5. Consolidate checkpoints (optional, useful if interrupted)
python consolidate_checkpoints.py

# 6. Generate visualizations (after completion)
python 03_visualize_regression_results.py
```

## Output Files

### Classification Results
- **JSON** (`classification_results.json`): Full details with confusion matrices, predictions
- **CSV** (`classification_summary.csv`): Aggregated table (100 rows: 10 combos × 5 seeds × 2 classifiers)
- **Figures**:
  - `ablation_comparison_all_metrics.png`: 2×3 grid showing all 6 metrics
  - `ablation_auc_comparison.png`: Focused ROC-AUC comparison
  - `protein_ligand_contribution_heatmap.png`: 2×2 heatmap showing protein/ligand impact
  - `ablation_summary_table.csv`: Mean ± std for each combination

### Regression Results
- **JSON** (`regression_results.json`): Full details with predictions
- **CSV** (`regression_summary.csv`): Aggregated table (30 rows: 3 models × 5 seeds × 2 regressors)
- **Checkpoints** (`regression_results_*_seed*.json`): Incremental saves per seed
- **Figures**:
  - `regression_metrics_comparison.png`: 2×3 grid (R², RMSE, MAE, Pearson, Spearman, CCC)
  - `regression_r2_focus.png`: Detailed R² comparison with error bars
  - `regression_correlation_comparison.png`: Pearson vs Spearman side-by-side
  - `regression_error_metrics.png`: RMSE vs MAE comparison
  - `regression_heatmap_summary.png`: Color-coded performance matrix

## Interpretation Guide

### How to Read Results

**Classification**:
- **ROC-AUC > 0.95**: Excellent discrimination
- **MCC > 0.75**: Strong agreement (accounts for class imbalance)
- **Low std (<0.01)**: Stable across seeds (reproducible)
- **C1 > C4**: Learned representations outperform handcrafted

**Regression**:
- **R² > 0.70**: Good variance explained
- **CCC > 0.70**: Good concordance (accuracy + precision)
- **RMSE < 0.8**: Predictions within ±0.8 pChEMBL units
- **Larger models**: Better performance (ESM-2 3B > 150M > 8M)

### Common Pitfalls to Avoid

1. **Confusing Metrics**:
   - Accuracy can be misleading for imbalanced data → Use MCC
   - R² alone doesn't capture systematic bias → Use CCC

2. **Overfitting**:
   - Test metrics >> Validation metrics → Model not generalizing
   - Check consistency across 5 seeds

3. **Data Leakage**:
   - Ensure same protein/ligand never appears in both train and test
   - Our random split may allow this (use stratified split for stricter separation)

## Future Extensions

### Potential Improvements
- **Stratified Split by Protein**: Ensure protein-level separation
- **Cold-Start Evaluation**: Test on unseen proteins/ligands
- **Temporal Split**: Use assay date if available
- **Ensemble Methods**: Combine multiple representations
- **Attention Mechanisms**: Learn optimal representation weighting

### Additional Analyses
- **Error Analysis**: Identify systematic prediction failures
- **Feature Importance**: SHAP values for interpretation
- **Molecular Similarity**: Cluster analysis of errors
- **Cross-Dataset Validation**: Generalization to other kinase families

## References

### Methods
- **ESM-2**: Lin et al. (2022) "Language models of protein sequences at the scale of evolution"
- **SMI-TED**: Honda et al. (2019) "SMILES Transformer Encoder for Drug Design"
- **Morgan Fingerprints**: Rogers & Hahn (2010) "Extended-connectivity fingerprints"
- **MCC**: Matthews (1975) "Comparison of the predicted and observed secondary structure of T4 phage lysozyme"
- **CCC**: Lin (1989) "A concordance correlation coefficient to evaluate reproducibility"

### Software
- **ESM**: Meta AI (facebook/esm)
- **RDKit**: Open-source cheminformatics
- **Scikit-learn**: Machine learning library
- **Matplotlib/Seaborn**: Visualization

## Citation

If you use these ablation studies, please cite:
```bibtex
@software{docktkinase_ablation_2026,
  title = {Ablation Studies for Protein-Ligand Binding Prediction},
  author = {DockTKinase Team},
  year = {2026},
  url = {https://github.com/gmmsb-lncc/docktkinase},
  note = {Systematic evaluation of learned vs handcrafted molecular representations}
}
```

## License

See main repository LICENSE file.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Last Updated**: January 17, 2026  
**Status**: Classification complete ✅ | Regression in progress 🔄
