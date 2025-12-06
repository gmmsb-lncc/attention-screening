# 🧬 DockTKinase - Complete User Guide

**Comprehensive manual for classification and regression pipelines.**

---

## 📋 Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Classification Pipeline](#classification-pipeline)
4. [Regression Pipeline](#regression-pipeline)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Automatic Installation (Recommended)
```bash
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

python3 -m venv env
source env/bin/activate

python setup.py
```

### Verify Installation
```bash
python -m pytest tests/ -v
# Expected: 19/19 tests passing ✅
```

---

## ⚡ Quick Start

### Classification (Binary Prediction)
```bash
source env/bin/activate

python run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/exp1
```

### Regression (Quantitative Prediction)
```bash
python run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/exp1 \
    --no-classification
```

---

## 🎯 Classification Pipeline

### Command Line Interface

#### Basic Usage
```bash
python run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/experiment1
```

#### Advanced Options
```bash
python run_complete_pipeline.py \
    --input src/database/data.tsv \
    --output results/experiment1 \
    --protein-model esm2_t36_3B_UR50D \
    --device cuda \
    --seed 42
```

### Python API

#### Basic Pipeline
```python
from src.build import BuildPipeline

pipeline = BuildPipeline(
    input_tsv='src/database/data.tsv',
    output_dir='results/exp1'
)
results = pipeline.run()
```

#### Custom Configuration
```python
from src.build import BuildConfig, BuildPipeline

config = BuildConfig({
    'input_tsv': 'src/database/data.tsv',
    'output_dir': 'results/custom',
    'esm_model': 'esm2_t36_3B_UR50D',
    'device': 'cuda',
    'batch_size': 8,
    'threshold': 1000,
    'stratification': True
})

pipeline = BuildPipeline(config)
pipeline.run()
```

### Available Models (6)

| Model | Type | Best For |
|-------|------|----------|
| RandomForest | Ensemble | Robustness |
| XGBoost | Boosting | Accuracy |
| GradientBoosting | Boosting | Complex patterns |
| SVM | Kernel | High dimensions |
| KNN | Instance-based | Similarity |
| MLP | Neural Network | Non-linearity |

---

## 📈 Regression Pipeline

### Command Line Interface

#### Basic Usage
```bash
python run_regression_pipeline.py \
    --data results/exp1/matrix/embedding_matrix.npz \
    --output results/exp1/regression \
    --activity-type Ki
```

#### Advanced Options
```bash
python run_regression_pipeline.py \
    --data results/exp1/matrix/embedding_matrix.npz \
    --output results/exp1/regression \
    --activity-type Ki \
    --models RandomForest,XGBoost,MLP \
    --test-size 0.2 \
    --n-jobs -1
```

### Python API

#### Basic Training
```python
from src.regression import RegressionTrainer, RegressionConfig

config = RegressionConfig(
    data_path='results/exp1/matrix/embedding_matrix.npz',
    output_dir='results/exp1/regression',
    activity_type='Ki'
)

trainer = RegressionTrainer(config)
results = trainer.train_all_models()
```

#### Custom Models
```python
from src.regression import RegressionConfig, RegressionTrainer

config = RegressionConfig(
    data_path='results/exp1/matrix/embedding_matrix.npz',
    output_dir='results/exp1/regression',
    activity_type='Ki',
    models_to_train=['RandomForest', 'XGBoost', 'MLP'],
    test_size=0.2,
    random_state=42,
    n_jobs=-1
)

trainer = RegressionTrainer(config)
results = trainer.train_all_models()

# Access results
for model_name, model_results in results.items():
    print(f"{model_name}:")
    print(f"  R²: {model_results['metrics']['r2']:.3f}")
    print(f"  MAE: {model_results['metrics']['mae']:.3f} nM")
```

### Available Models (11)

#### Linear Models
- **LinearRegression** - Simple linear regression
- **Ridge** - L2 regularization
- **Lasso** - L1 regularization (feature selection)
- **ElasticNet** - L1 + L2 regularization

#### Tree-Based Models
- **RandomForest** - Ensemble of decision trees
- **GradientBoosting** - Sequential boosting
- **XGBoost** - Optimized gradient boosting
- **DecisionTree** - Single interpretable tree

#### Other Models
- **SVR** - Support Vector Regression
- **KNN** - K-Nearest Neighbors
- **MLP** - Multi-Layer Perceptron (Neural Network)

### Activity Types

#### Ki (Inhibition Constant)
```python
config = RegressionConfig(
    activity_type='Ki',
    # ...
)
```

#### Kd (Dissociation Constant)
```python
config = RegressionConfig(
    activity_type='Kd',
    # ...
)
```

#### IC50 (Half Maximal Inhibitory Concentration)
```python
config = RegressionConfig(
    activity_type='IC50',
    # ...
)
```

**Priority:** Ki > Kd > IC50

---

## ⚙️ Configuration

### Build Configuration (Classification)

```python
from src.build import BuildConfig

config = BuildConfig({
    # Input/Output
    'input_tsv': 'src/database/data.tsv',
    'output_dir': 'results/experiment',
    
    # Embeddings
    'esm_model': 'esm2_t36_3B_UR50D',  # ESM-2 model
    'device': 'cuda',                   # cuda or cpu
    'batch_size': 8,
    
    # Labels
    'threshold': 1000,                  # IC50 threshold (nM)
    
    # Stratification
    'stratification': True,
    'test_size': 0.1,
    'val_size': 0.1,
    'random_state': 42
})
```

### Regression Configuration

```python
from src.regression import RegressionConfig

config = RegressionConfig(
    # Data
    data_path='results/exp1/matrix/embedding_matrix.npz',
    output_dir='results/exp1/regression',
    
    # Activity
    activity_type='Ki',  # Ki, Kd, or IC50
    
    # Models
    models_to_train=[
        'LinearRegression',
        'RandomForest',
        'XGBoost'
    ],
    
    # Training
    test_size=0.2,
    random_state=42,
    n_jobs=-1,  # Use all cores
    
    # Validation
    cv_folds=5,
    
    # Output
    save_models=True,
    generate_visualizations=True
)
```

---

## 🔧 API Reference

### BuildPipeline Class

```python
from src.build import BuildPipeline

pipeline = BuildPipeline(
    input_tsv='path/to/data.tsv',
    output_dir='path/to/output',
    config=None  # Optional BuildConfig
)

# Methods
pipeline.run()                    # Run complete pipeline
pipeline.generate_embeddings()    # Generate embeddings only
pipeline.build_matrix()           # Build matrix only
pipeline.generate_labels()        # Generate labels only
```

### RegressionTrainer Class

```python
from src.regression import RegressionTrainer

trainer = RegressionTrainer(config)

# Methods
results = trainer.train_all_models()        # Train all
results = trainer.train_model('XGBoost')    # Train specific
trainer.save_models()                       # Save trained models
trainer.load_models()                       # Load saved models
```

### RegressionEvaluator Class

```python
from src.regression import RegressionEvaluator

evaluator = RegressionEvaluator(config)

# Methods
metrics = evaluator.evaluate(model, X_test, y_test)
evaluator.plot_predictions(y_true, y_pred)
evaluator.plot_residuals(y_true, y_pred)
evaluator.generate_report()
```

### Utility Functions

```python
from src.utils import safe_get, safe_get_numeric

# Safe dictionary access
value = safe_get(data, 'key', default='N/A')

# Safe numeric conversion
number = safe_get_numeric(data, 'Ki', default=0.0)
```

---

## 🚨 Troubleshooting

### Common Errors

#### 1. ModuleNotFoundError
```bash
# Solution: Activate environment
source env/bin/activate
python setup.py
```

#### 2. CUDA out of memory
```bash
# Solution 1: Reduce batch size
python run_complete_pipeline.py --batch-size 4

# Solution 2: Use CPU
python run_complete_pipeline.py --device cpu
```

#### 3. No activity values found
```bash
# Check TSV columns
head -1 src/database/data.tsv

# Ensure Ki, Kd, or IC50 column exists
```

#### 4. File not found
```python
# Use absolute paths
import os
data_path = os.path.abspath('src/database/data.tsv')
```

### Performance Optimization

#### GPU Memory
```python
# Reduce batch size for large models
config = BuildConfig({
    'batch_size': 4,  # Default: 8
    'device': 'cuda'
})
```

#### CPU Parallelization
```python
# Use all available cores
config = RegressionConfig(
    n_jobs=-1  # Use all cores
)
```

#### Selective Model Training
```python
# Train only fast models
config = RegressionConfig(
    models_to_train=['LinearRegression', 'Ridge', 'Lasso']
)
```

---

## 📊 Examples

### Example 1: Basic Classification
```python
from src.build import BuildPipeline

pipeline = BuildPipeline(
    input_tsv='src/database/kinase_data.tsv',
    output_dir='results/kinase_exp'
)
pipeline.run()
```

### Example 2: Regression with Multiple Activities
```python
from src.regression import RegressionConfig, RegressionTrainer

# Try all activities
for activity in ['Ki', 'Kd', 'IC50']:
    config = RegressionConfig(
        data_path='results/kinase_exp/matrix/embedding_matrix.npz',
        output_dir=f'results/kinase_exp/regression_{activity}',
        activity_type=activity
    )
    
    trainer = RegressionTrainer(config)
    results = trainer.train_all_models()
    
    print(f"\n{activity} Results:")
    for model, res in results.items():
        print(f"  {model}: R² = {res['metrics']['r2']:.3f}")
```

### Example 3: Cross-Validation
```python
from src.regression import RegressionConfig, RegressionTrainer
from sklearn.model_selection import cross_val_score

config = RegressionConfig(
    data_path='results/exp/matrix/embedding_matrix.npz',
    output_dir='results/exp/regression',
    activity_type='Ki',
    cv_folds=5
)

trainer = RegressionTrainer(config)
results = trainer.train_all_models()

# Cross-validation scores
for model_name, model_data in results.items():
    cv_scores = model_data.get('cv_scores', [])
    if cv_scores:
        print(f"{model_name}:")
        print(f"  CV R²: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

---

## 📚 Additional Resources

- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Detailed installation
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Execution workflows
- **[../src/regression/README.md](../src/regression/README.md)** - Regression module docs
- **[../src/utils/README.md](../src/utils/README.md)** - Utilities docs
- **[../README.md](../README.md)** - Main project README

---

## 🤝 Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/gmmsb-lncc/docktkinase/issues
- **Documentation**: See `docs/` folder
- **Examples**: See `examples/` folder

---

**Last updated**: October 28, 2025 | **Branch**: regression
