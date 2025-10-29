# Regression Pipeline

**Last Updated**: October 28, 2025  
**Section**: Chapter 02 - User Guide  
**Audience**: End Users

---

## Overview

The regression pipeline predicts binding affinity values (Ki, Kd, IC50) using 11 different regression models. It provides quantitative predictions for protein-ligand interactions.

## Available Models (11 Total)

### Linear Models (4)
1. **Linear Regression**: Simple linear relationship
2. **Ridge**: Linear with L2 regularization
3. **Lasso**: Linear with L1 regularization (feature selection)
4. **ElasticNet**: Combined L1 + L2 regularization

### Tree-Based Models (4)
5. **Decision Tree**: Simple decision tree regressor
6. **Random Forest**: Ensemble of decision trees
7. **Gradient Boosting**: Sequential tree boosting
8. **XGBoost**: Optimized gradient boosting

### Other Models (3)
9. **Support Vector Regressor (SVR)**: Kernel-based regression
10. **K-Nearest Neighbors (KNN)**: Instance-based regression
11. **Multi-Layer Perceptron (MLP)**: Neural network regression

---

## Usage

### Basic Usage

```bash
# Run complete regression pipeline
python run_regression_pipeline.py
```

### With Options

```bash
python run_regression_pipeline.py \
    --dataset data/kinase_data.tsv \
    --activity-type ki \
    --models linear ridge xgboost \
    --output-dir results/regression
```

### Programmatic Usage

```python
from src.regression import RegressionTrainer

trainer = RegressionTrainer(
    data_path='results/matrix/embedding_matrix.npz',
    output_dir='results/regression',
    activity_type='Ki'  # or 'Kd', 'IC50'
)

# Train all models
trainer.train_all_models()

# Train specific models
trainer.train_models(['LinearRegression', 'XGBoost', 'RandomForest'])
```

---

## Input Format

TSV file with required columns:
- `Ligand_SMILES`: SMILES notation
- `Target_Seq`: Protein sequence
- `Ki` or `Kd` or `IC50`: Binding affinity value (nM)

Activity priority: **Ki > Kd > IC50** (scientific order)

Example:
```
Ligand_SMILES	Target_Seq	Ki	Kd	IC50
CCO	MKVLW...	10.5	-	-
CCCO	MKALT...	-	25.3	-
```

---

## Output

Results saved in `results/<dataset>/regression/`:
- `models/` - Trained regression models (.pkl)
- `metrics/` - R², MAE, RMSE metrics
- `predictions/` - Predicted vs actual values
- `plots/` - Scatter plots, residual plots

---

## Performance Metrics

Models evaluated using:
- **R² (R-squared)**: Coefficient of determination (0-1, higher is better)
- **MAE (Mean Absolute Error)**: Average absolute difference
- **RMSE (Root Mean Squared Error)**: Root mean squared error
- **Pearson Correlation**: Linear correlation coefficient

---

## Configuration

Create `config.json`:

```json
{
  "data_path": "results/matrix/embedding_matrix.npz",
  "output_dir": "results/regression",
  "activity_type": "Ki",
  "test_size": 0.2,
  "random_state": 42,
  "n_jobs": -1,
  "models_to_train": [
    "LinearRegression",
    "Ridge",
    "XGBoost",
    "RandomForest"
  ]
}
```

Run with:
```bash
python run_regression_pipeline.py --config config.json
```

---

## Related Documentation

- **Classification Pipeline**: [Classification Workflow](classification-pipeline.md)
- **Visualization**: [Visualizing Results](visualization.md)
- **Module Documentation**: [Chapter 04: Modules](../04-modules/README.md)

---

**Previous**: [← Classification Pipeline](classification-pipeline.md) | **Next**: [Model Comparison →](model-comparison.md)
