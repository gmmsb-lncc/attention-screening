# 🚀 DockTKinase Execution Guide

**Comprehensive step-by-step guide to execute classification and regression pipelines.**

---

## 📋 **Workflow Overview**

DockTKinase provides **dual pipelines** for molecular activity prediction:

```
┌──────────────┐
│   Setup      │  python setup.py
└──────┬───────┘
       │
┌──────▼───────────┐
│ Data Preparation │  Place TSV in src/database/
└──────┬───────────┘
       │
       ├─────────────────────┬─────────────────────┐
       │                     │                     │
┌──────▼──────────┐  ┌──────▼──────────┐  ┌──────▼──────────┐
│ Classification  │  │   Regression    │  │  Both Pipelines │
│   Pipeline      │  │    Pipeline     │  │   (Recommended) │
│                 │  │                 │  │                 │
│ Binary          │  │ Quantitative    │  │ Complete        │
│ (Active/        │  │ (Ki, Kd, IC50)  │  │ Analysis        │
│  Inactive)      │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 🔧 **STAGE 1: Setup and Installation**

### **1.1 Automatic Installation (RECOMMENDED)**
```bash
# Clone repository
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Create virtual environment
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Automatic installation
python setup.py
```

### **1.2 Verification**
```bash
# Run tests
python -m pytest tests/ -v

# Expected: 19/19 tests passing ✅
```

---

## 📊 **STAGE 2: Data Preparation**

### **2.1 TSV File Format**

**Required columns for CLASSIFICATION:**
- `Ligand_SMILES` - SMILES representation
- `Target_Seq` - Protein sequence
- `Y` - Binary label (0=inactive, 1=active)

**Additional columns for REGRESSION:**
- `Ki` - Ki value in nM (nanomolar)
- `Kd` - Kd value in nM
- `IC50` - IC50 value in nM

**Example:**
```tsv
Ligand_SMILES	Target_Seq	Y	Ki	Kd	IC50
CC(=O)Oc1ccccc1C(=O)O	MKTAYIAK...	1	10.5	25.3	100.0
```

---

## 🎯 **STAGE 3A: Classification Pipeline**

### **3A.1 Complete Pipeline**
```bash
python scripts/run_complete_pipeline.py \
    --input src/database/your_file.tsv \
    --output results/my_experiment \
    --protein-model esm2_t36_3B_UR50D \
    --device cuda
```

**Steps executed:**
1. ✅ Protein embeddings (ESM-2/Boltz-2)
2. ✅ Ligand embeddings (FM4M)
3. ✅ Matrix construction
4. ✅ Binary labels
5. ✅ **Adaptive Stratification** (auto-threshold detection)
6. ✅ Train classification models

### **3A.2 Stratification Options**

```bash
# Auto threshold (default - target method)
python scripts/run_complete_pipeline.py \
    --input data.tsv \
    --output results/auto_strat

# Manual threshold
python scripts/run_complete_pipeline.py \
    --input data.tsv \
    --output results/manual_strat \
    --stratifier-threshold 0.95

# Custom auto-threshold method
python scripts/run_complete_pipeline.py \
    --input data.tsv \
    --output results/silhouette_strat \
    --stratifier-method silhouette
```

**Stratification Methods:**
| Method | Description |
|--------|-------------|
| `target` | Binary search for ~1% clusters (default) |
| `silhouette` | Maximize silhouette score |
| `elbow` | Elbow method for K-means |
| `percentile` | Use similarity percentile |
| `manual` | Use --stratifier-threshold value |

### **3A.3 Classification Models (10)**
- RandomForest
- XGBoost
- GradientBoosting
- LightGBM
- CatBoost
- SVM
- KNN
- MLP
- LogisticRegression
- ExtraTrees

---

## 📈 **STAGE 3B: Regression Pipeline**

### **3B.1 Complete Regression Pipeline**
```bash
python scripts/run_complete_pipeline.py \
    --input src/database/your_file.tsv \
    --output results/my_experiment \
    --no-classification
```

**Steps executed:**
1. ✅ Load matrix (or build if needed)
2. ✅ Extract activity (Ki/Kd/IC50)
3. ✅ Data validation (10+ checks)
4. ✅ Train 11 regression models
5. ✅ Evaluate (R², MAE, RMSE)
6. ✅ Generate visualizations

### **3B.2 Regression Models (11)**

**Linear:**
- LinearRegression
- Ridge
- Lasso
- ElasticNet

**Tree-based:**
- RandomForest
- GradientBoosting
- XGBoost
- DecisionTree

**Others:**
- SVR
- KNN
- MLP

### **3B.3 Activity Priority**
**Ki > Kd > IC50**

---

## 🔄 **QUICK START EXAMPLES**

### **Beginners (Complete Workflow)**
```bash
# 1. Setup
python3 -m venv env
source env/bin/activate
python setup.py

# 2. Classification + Regression (Complete)
python scripts/run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/exp1
```

### **Advanced (Custom Configuration)**
```bash
# Classification with GPU
python scripts/run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/exp1 \
    --protein-model esm2_t36_3B_UR50D \
    --device cuda \
    --batch-size 8

# Regression with specific models
python scripts/run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/exp1 \
    --no-classification \
    --regression-models RandomForest XGBoost MLP
```

### **Python API (Maximum Control)**
```python
from src.build import BuildPipeline
from src.regression import RegressionTrainer, RegressionConfig

# 1. Embeddings
pipeline = BuildPipeline(
    input_tsv='src/database/data.tsv',
    output_dir='results/custom'
)
pipeline.run()

# 2. Regression
config = RegressionConfig(
    data_path='results/custom/matrix/embedding_matrix.npz',
    output_dir='results/custom/regression',
    activity_type='Ki'
)
trainer = RegressionTrainer(config)
trainer.train_all_models()
```

---

## 📁 **Output Structure**

```
results/my_experiment/
├── build/
│   ├── embeddings/
│   │   ├── protein_embeddings.npy  # Protein embeddings
│   │   └── ligand_embeddings.npy   # FM4M embeddings
│   ├── embedding_matrix.npy        # Combined matrix
│   └── stratification/
│       ├── stratification_clustering_metrics.json  # Clustering metrics
│       ├── stratification_split_info.json          # Split statistics
│       └── cluster_labels.npy                      # Cluster assignments
├── splits/
│   ├── stratified_splits.npz       # Train/val/test indices
│   ├── train_indices.npy
│   ├── val_indices.npy
│   └── test_indices.npy
├── classifier/                     # Classification results
│   ├── models/
│   │   ├── RandomForest.pkl
│   │   └── XGBoost.pkl
│   └── metrics/
│       └── test_metrics.json
└── regression/                     # Regression results
    ├── models/
    │   ├── LinearRegression.pkl
    │   └── RandomForest.pkl
    ├── evaluations/
    │   └── metrics_summary.json
    └── visualizations/
        ├── scatter_plots.png
        └── residual_plots.png
```

---

## 🚨 **Troubleshooting**

### **CUDA out of memory**
```bash
# Reduce batch size
python scripts/run_complete_pipeline.py --batch-size 4

# Or use CPU
python scripts/run_complete_pipeline.py --device cpu
```

### **Module not found**
```bash
# Activate environment
source env/bin/activate

# Reinstall
python setup.py
```

### **No activity values**
```bash
# Check columns
head -1 src/database/your_file.tsv

# Ensure Ki, Kd, or IC50 column exists
```

---

## 📊 **Performance**

| Dataset | Embeddings | Training | Total |
|---------|-----------|----------|-------|
| 1K samples | ~5 min | ~1 min | ~6 min |
| 10K samples | ~30 min | ~5 min | ~35 min |
| 100K samples | ~4 hours | ~20 min | ~4.5h |

---

## 📚 **See Also**

- [QUICK_START.md](QUICK_START.md) - Quick start
- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Installation
- [USER_GUIDE.md](USER_GUIDE.md) - User manual
- [../src/regression/README.md](../src/regression/README.md) - Regression docs

---

**Last updated**: October 28, 2025 | **Branch**: regression
