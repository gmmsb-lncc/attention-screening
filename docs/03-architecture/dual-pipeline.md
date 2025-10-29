# Dual Pipeline System

**Last Updated**: October 28, 2025  
**Section**: Chapter 03 - Architecture  
**Audience**: Developers & Advanced Users

---

## Overview

DockTKinase implements a dual pipeline system supporting both classification and regression tasks for protein-ligand interaction prediction.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT DATA (TSV)                          │
│  • Ligand_SMILES • Target_Seq • Y • Ki/Kd/IC50              │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│   PROTEIN    │         │   LIGAND     │
│  EMBEDDINGS  │         │  EMBEDDINGS  │
│   (ESM-2)    │         │  (FM4M/SMI)  │
│  1280-dim    │         │   Varies     │
└──────┬───────┘         └──────┬───────┘
       │                        │
       └────────┬───────────────┘
                │
                ▼
        ┌──────────────┐
        │   COMBINED   │
        │   FEATURES   │
        │    MATRIX    │
        └──────┬───────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌─────────────┐ ┌─────────────┐
│ CLASSIFICATION│ │  REGRESSION │
│  PIPELINE    │ │   PIPELINE  │
│  (6 models)  │ │ (11 models) │
└──────┬───────┘ └──────┬──────┘
       │                │
       ▼                ▼
┌─────────────┐ ┌─────────────┐
│   BINARY    │ │ QUANTITATIVE│
│ PREDICTIONS │ │  PREDICTIONS│
│ Active/Inact│ │ Ki/Kd/IC50  │
└─────────────┘ └─────────────┘
```

---

## Classification Pipeline

### Purpose
Predict binary classification (active/inactive) for kinase families.

### Models (6 Total)
1. Random Forest (RF)
2. XGBoost
3. Gradient Boosting (GB)
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Multi-Layer Perceptron (MLP)

### Output
- Binary predictions (0/1)
- Probability scores
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### Usage
```bash
python run_complete_pipeline.py
```

---

## Regression Pipeline

### Purpose
Predict quantitative binding affinity values (Ki, Kd, IC50 in nM).

### Models (11 Total)

**Linear Models:**
1. Linear Regression
2. Ridge
3. Lasso
4. ElasticNet

**Tree-Based:**
5. Decision Tree
6. Random Forest
7. Gradient Boosting
8. XGBoost

**Other:**
9. SVR
10. KNN
11. MLP

### Output
- Continuous predictions (binding affinity in nM)
- Metrics: R², MAE, RMSE, Pearson correlation

### Usage
```bash
python run_regression_pipeline.py
```

---

## Shared Components

### 1. Data Loading
- Common TSV parser
- Validation and preprocessing
- Missing value handling

### 2. Embeddings
- **Protein**: ESM-2 (1280-dim)
- **Ligand**: FM4M or SMI-TED
- Cached for reuse

### 3. Stratification
- Train/test splitting
- Stratified sampling
- Class balance (classification)

### 4. Utilities
- Configuration management
- Device management (GPU/CPU)
- Logging
- Metrics computation

---

## Pipeline Orchestration

### Sequential Execution
```bash
# Run both pipelines in sequence
python run_complete_pipeline.py      # Classification
python run_regression_pipeline.py    # Regression
```

### Programmatic Control
```python
from src.build import BuildPipeline
from src.classifier import ClassifierPipeline
from src.regression import RegressionTrainer

# Build features
build = BuildPipeline(input_tsv='data.tsv', output_dir='results')
build.run()

# Classification
classifier = ClassifierPipeline(output_dir='results')
classifier.run_all_models()

# Regression
regressor = RegressionTrainer(
    data_path='results/matrix/embedding_matrix.npz',
    output_dir='results/regression'
)
regressor.train_all_models()
```

---

## Data Flow

### Input Requirements
```
Required Columns:
- Ligand_SMILES (string): SMILES notation
- Target_Seq (string): Amino acid sequence
- Y (int): Binary label for classification (0/1)
- Ki or Kd or IC50 (float): Binding affinity for regression (nM)
```

### Processing Steps
1. **Load Data** → TSV parsing
2. **Generate Embeddings** → ESM-2 + FM4M
3. **Build Matrix** → Combined feature matrix
4. **Stratify** → Train/test split
5. **Train Models** → Classification and/or Regression
6. **Evaluate** → Metrics and visualizations

---

## Performance Comparison

| Pipeline | Models | Output Type | Metrics | Training Time* |
|----------|--------|-------------|---------|----------------|
| Classification | 6 | Binary (0/1) | Acc, F1, AUC | ~10 min |
| Regression | 11 | Continuous (nM) | R², MAE, RMSE | ~15 min |

*For 10,000 samples on GPU

---

## Related Documentation

- **Classification Details**: [Chapter 02: Classification Pipeline](../02-user-guide/classification-pipeline.md)
- **Regression Details**: [Chapter 02: Regression Pipeline](../02-user-guide/regression-pipeline.md)
- **Build System**: [Build Architecture](build-system.md)
- **Modules**: [Chapter 04: Modules](../04-modules/README.md)

---

**Previous**: [← Build System](build-system.md) | **Next**: [Data Flow →](data-flow.md)
