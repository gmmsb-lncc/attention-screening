# Classification Pipeline

**Last Updated**: October 28, 2025  
**Section**: Chapter 02 - User Guide  
**Audience**: End Users

---

## Overview

The classification pipeline predicts kinase families using 6 different machine learning models. It processes protein sequences and ligand structures to generate binary predictions (active/inactive).

## Available Models

### 1. Random Forest (RF)
- **Type**: Ensemble method
- **Best for**: Balanced datasets
- **Pros**: Fast training, good interpretability
- **Hyperparameters**: `n_estimators`, `max_depth`, `min_samples_split`

### 2. XGBoost
- **Type**: Gradient boosting
- **Best for**: Large datasets
- **Pros**: High accuracy, handles missing values
- **Hyperparameters**: `learning_rate`, `max_depth`, `n_estimators`

### 3. Gradient Boosting
- **Type**: Ensemble method
- **Best for**: Complex patterns
- **Pros**: Good performance, feature importance
- **Hyperparameters**: `learning_rate`, `n_estimators`, `max_depth`

### 4. Support Vector Machine (SVM)
- **Type**: Kernel-based
- **Best for**: High-dimensional data
- **Pros**: Effective in high dimensions
- **Hyperparameters**: `C`, `kernel`, `gamma`

### 5. K-Nearest Neighbors (KNN)
- **Type**: Instance-based
- **Best for**: Small to medium datasets
- **Pros**: Simple, no training phase
- **Hyperparameters**: `n_neighbors`, `weights`, `metric`

### 6. Multi-Layer Perceptron (MLP)
- **Type**: Neural network
- **Best for**: Complex non-linear patterns
- **Pros**: Can learn complex relationships
- **Hyperparameters**: `hidden_layer_sizes`, `activation`, `learning_rate`

---

## Usage

### Basic Usage

```bash
# Run complete classification pipeline
python run_complete_pipeline.py
```

### Programmatic Usage

```python
from src.classifier import ClassifierPipeline

pipeline = ClassifierPipeline(
    input_tsv='src/database/kinase_data.tsv',
    output_dir='results/classification'
)

# Run all models
results = pipeline.run_all_models()

# Run specific model
results = pipeline.run_model('rf')  # rf, xgboost, gb, svm, knn, mlp
```

---

## Input Format

TSV file with required columns:
- `Ligand_SMILES`: SMILES notation of ligand
- `Target_Seq`: Protein sequence
- `Y`: Binary label (0=inactive, 1=active)

Example:
```
Ligand_SMILES	Target_Seq	Y
CCO	MKVLW...	1
CCCO	MKALT...	0
```

---

## Output

Results saved in `results/<dataset>/`:
- `models/` - Trained model files (.pkl)
- `metrics/` - Performance metrics (accuracy, precision, recall, F1)
- `predictions/` - Prediction outputs
- `plots/` - Confusion matrices, ROC curves

---

## Performance Metrics

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

---

## Related Documentation

- **Model Comparison**: [Compare Classifiers](model-comparison.md)
- **Execution Guide**: [How to Run](execution-guide.md)
- **Regression Pipeline**: [Regression Workflow](regression-pipeline.md)

---

**Previous**: [← Execution Guide](execution-guide.md) | **Next**: [Regression Pipeline →](regression-pipeline.md)
