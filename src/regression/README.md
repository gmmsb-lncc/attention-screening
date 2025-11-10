# Regression Module

**Version**: 1.0.0  
**Python**: 3.8+  
**Scikit-learn**: 1.3+

Modular regression pipeline with comprehensive model comparison, cross-validation, and evaluation capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Module Structure](#module-structure)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)

---

## 🎯 Overview

Complete regression pipeline supporting 9 algorithms with train/val/test stratified splitting, cross-validation, and comprehensive metrics (MAE, RMSE, R², MSE).

**Key Components**:
- **Stratified Data Split**: 70/10/20 (train/val/test)
- **9 Regression Models**: Ridge, Lasso, ElasticNet, RF, GB, SVR, KNN, MLP, XGBoost
- **K-Fold Cross-Validation**: Professional CV with fold-level metrics
- **Model Comparison**: Automatic ranking and selection
- **Visualization**: Plots for predictions, residuals, and model comparison

---

## ✨ Features

- ✅ **9 Regression Algorithms** ready to use
- ✅ **Stratified Splitting** (70/10/20) for robust validation
- ✅ **Cross-Validation** (K-Fold) with comprehensive statistics
- ✅ **4 Core Metrics**: MAE, RMSE, R², MSE
- ✅ **Model Comparison** with automatic ranking
- ✅ **Visualization Suite** for analysis
- ✅ **Reproducibility** via random seeds
- ✅ **CSV Export** for results

---

## 🚀 Quick Start

### Basic Usage

```python
from regression.modular_pipeline import RegressionPipeline
import numpy as np

# Load data
X = np.load("features.npy")
y = np.load("targets.npy")

# Create pipeline
pipeline = RegressionPipeline(X, y, test_size=0.2, val_size=0.1)

# Train models
results = pipeline.train_models(['Ridge', 'RandomForest', 'XGBoost'])

# Best model
best = results['best_model']
print(f"Best: {best} (MAE: {results['best_mae']:.2f})")

# Cross-validation
cv_results = pipeline.cross_validate(['Ridge', 'Lasso'], n_splits=5)
```

### Cross-Validation Only

```python
from regression.core import quick_cross_validate
import numpy as np

X = np.load("features.npy")
y = np.load("targets.npy")

# Quick CV with 3 models
results = quick_cross_validate(
    X, y,
    model_names=['Ridge', 'Lasso', 'ElasticNet'],
    n_splits=5
)

# Results
for model, result in results.items():
    print(f"{model}: MAE = {result.get_mean_metric('mae'):.2f} ± {result.get_std_metric('mae'):.2f}")
```

### Command Line

```bash
# Train multiple models
python -m regression.modular_pipeline \
    --X_path features.npy \
    --y_path targets.npy \
    --models Ridge Lasso RandomForest \
    --output_dir results/

# With cross-validation
python -m regression.modular_pipeline \
    --X_path features.npy \
    --y_path targets.npy \
    --models Ridge XGBoost \
    --cv_folds 5 \
    --output_dir results/
```

---

## 📁 Module Structure

```
src/regression/
├── __init__.py                    # Package initialization
├── modular_pipeline.py            # Main pipeline (RegressionPipeline)
├── config.py                      # RegressionConfig
├── logger.py                      # RegressionLogger
├── visualizer.py                  # RegressionVisualizer
│
├── core/                          # Core modules
│   ├── __init__.py
│   ├── data_loader.py            # DataManager (stratified split)
│   ├── trainer.py                # RegressionTrainer
│   ├── evaluator.py              # RegressionEvaluator (metrics)
│   └── cross_validator.py        # RegressionCrossValidator
│
└── models/                        # Model definitions
    ├── __init__.py
    └── models.py                 # RegressionModels (9 algorithms)
```

---

## 📚 API Reference

### RegressionPipeline
**Module**: `regression.modular_pipeline`

Main pipeline for complete regression workflow.

```python
class RegressionPipeline:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    )
    
    def train_models(
        self,
        model_names: List[str],
        verbose: bool = True
    ) -> Dict[str, Any]
    
    def cross_validate(
        self,
        model_names: List[str],
        n_splits: int = 5,
        verbose: bool = True
    ) -> Dict[str, CrossValidationResults]
    
    def predict(
        self,
        model_name: str,
        X_new: np.ndarray
    ) -> np.ndarray
    
    def save_results(
        self,
        output_dir: str,
        format: str = "csv"
    ) -> None
```

**Example**:
```python
pipeline = RegressionPipeline(X, y, test_size=0.2)
results = pipeline.train_models(['Ridge', 'Lasso'])
pipeline.save_results("results/")
```

---

### RegressionCrossValidator
**Module**: `regression.core.cross_validator`

K-Fold cross-validation with comprehensive metrics.

```python
class RegressionCrossValidator:
    def __init__(
        self,
        config: CrossValidationConfig = None
    )
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models_dict: Dict[str, Any],
        model_names: List[str]
    ) -> Dict[str, CrossValidationResults]
    
    def get_best_model(
        self,
        metric: str = 'mae'
    ) -> str
    
    def compare_models(self) -> pd.DataFrame

# Convenience function
def quick_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_names: List[str],
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, CrossValidationResults]
```

**Example**:
```python
from regression.core import quick_cross_validate

results = quick_cross_validate(X, y, ['Ridge', 'Lasso'], n_splits=5)
best = results['Ridge'].get_mean_metric('mae')
print(f"Ridge MAE: {best:.2f}")
```

---

### RegressionEvaluator
**Module**: `regression.core.evaluator`

Compute and compare regression metrics.

```python
class RegressionEvaluator:
    def __init__(self)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame
    
    def export_to_csv(
        self,
        results: Dict[str, Dict[str, float]],
        filepath: str
    ) -> None
```

**Metrics**: MAE, RMSE, R², MSE

**Example**:
```python
from regression.core import RegressionEvaluator

evaluator = RegressionEvaluator()
metrics = evaluator.evaluate(y_true, y_pred, "Ridge")
print(f"MAE: {metrics['mae']:.2f}")
```

---

### RegressionModels
**Module**: `regression.models.models`

Factory for 9 regression algorithms.

```python
class RegressionModels:
    @staticmethod
    def get_model(model_name: str) -> Any:
        """
        Available models:
        - Ridge
        - Lasso
        - ElasticNet
        - RandomForest
        - GradientBoosting
        - SVR
        - KNN
        - MLP
        - XGBoost
        """
    
    @staticmethod
    def get_all_models() -> Dict[str, Any]
    
    @staticmethod
    def get_model_names() -> List[str]
```

**Example**:
```python
from regression.models import RegressionModels

# Single model
ridge = RegressionModels.get_model('Ridge')
ridge.fit(X_train, y_train)

# All models
all_models = RegressionModels.get_all_models()
```

---

### Configuration Classes

#### CrossValidationConfig

```python
@dataclass
class CrossValidationConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = 42
    verbose: bool = True
```

#### CrossValidationResults

```python
@dataclass
class CrossValidationResults:
    model_name: str
    fold_metrics: List[FoldMetrics]
    summary_statistics: Dict[str, Dict[str, float]]
    best_fold: int
    config: Optional[CrossValidationConfig] = None
    
    def get_mean_metric(self, metric_name: str) -> float
    def get_std_metric(self, metric_name: str) -> float
```

---

## 💡 Examples

### Example 1: Train and Compare Models

```python
from regression.modular_pipeline import RegressionPipeline
import numpy as np

# Data
X = np.load("features.npy")
y = np.load("targets.npy")

# Pipeline
pipeline = RegressionPipeline(X, y, test_size=0.2, val_size=0.1)

# Train 5 models
results = pipeline.train_models([
    'Ridge', 'Lasso', 'ElasticNet',
    'RandomForest', 'GradientBoosting'
])

# Results
print(f"Best model: {results['best_model']}")
print(f"Best MAE: {results['best_mae']:.2f}")
print(f"Test R²: {results['models'][results['best_model']]['test_metrics']['r2']:.4f}")
```

### Example 2: Cross-Validation Comparison

```python
from regression.core import quick_cross_validate
import numpy as np

X = np.load("features.npy")
y = np.load("targets.npy")

# CV with 5 folds
results = quick_cross_validate(
    X, y,
    model_names=['Ridge', 'Lasso', 'RandomForest', 'XGBoost'],
    n_splits=5
)

# Compare
for model, result in results.items():
    mae_mean = result.get_mean_metric('mae')
    mae_std = result.get_std_metric('mae')
    r2_mean = result.get_mean_metric('r2')
    
    print(f"{model:15s} MAE: {mae_mean:6.2f} ± {mae_std:5.2f}  R²: {r2_mean:.4f}")
```

### Example 3: Manual Cross-Validation

```python
from regression.core import RegressionCrossValidator, CrossValidationConfig
from regression.models import RegressionModels
import numpy as np

# Data
X = np.load("features.npy")
y = np.load("targets.npy")

# Config
config = CrossValidationConfig(
    n_splits=10,
    shuffle=True,
    random_state=999
)

# Models
models = {
    'Ridge': RegressionModels.get_model('Ridge'),
    'Lasso': RegressionModels.get_model('Lasso'),
    'RF': RegressionModels.get_model('RandomForest')
}

# CV
cv = RegressionCrossValidator(config)
results = cv.cross_validate(X, y, models, list(models.keys()))

# Best model
best = cv.get_best_model(metric='r2')
print(f"Best by R²: {best}")

# Comparison DataFrame
df = cv.compare_models()
print(df)
```

### Example 4: Predictions and Visualization

```python
from regression.modular_pipeline import RegressionPipeline
import numpy as np

# Pipeline
pipeline = RegressionPipeline(X, y)
results = pipeline.train_models(['Ridge', 'RandomForest'])

# Best model predictions
best_model = results['best_model']
y_pred = pipeline.predict(best_model, X_test)

# Visualize
from regression.visualizer import RegressionVisualizer

viz = RegressionVisualizer()
viz.plot_predictions(y_test, y_pred, title=f"{best_model} Predictions")
viz.plot_residuals(y_test, y_pred, title=f"{best_model} Residuals")
viz.save_all("figures/")
```

### Example 5: Export Results

```python
from regression.modular_pipeline import RegressionPipeline

pipeline = RegressionPipeline(X, y)
results = pipeline.train_models(['Ridge', 'Lasso', 'RandomForest'])

# Export to CSV
pipeline.save_results("results/", format="csv")

# Files created:
# - results/train_metrics.csv
# - results/val_metrics.csv
# - results/test_metrics.csv
# - results/summary.csv
```

---

## 🧪 Testing

Complete test suite with **66 tests** across 9 levels (100% passing).

### Run All Tests

```bash
cd tests/regression_test/
python -m pytest -v
```

### Run Specific Levels

```bash
# Data loading tests
python test_1_1_data_loader.py

# Model tests
python test_2_1_models_factory.py

# Training tests
python test_3_1_trainer.py

# Cross-validation tests
python test_9_cross_validation.py
```

### Test Coverage

| Level | Description | Tests | Status |
|-------|-------------|-------|--------|
| 1 | Data Loading & Preprocessing | 10 | ✅ 100% |
| 2 | Feature Engineering | 6 | ✅ 100% |
| 3 | Model Training | 9 | ✅ 100% |
| 4 | Model Evaluation | 9 | ✅ 100% |
| 5 | Hyperparameter Optimization | 7 | ✅ 100% |
| 6 | Predictions & Inference | 7 | ✅ 100% |
| 7 | Visualization | 6 | ✅ 100% |
| 8 | Error Handling | 8 | ✅ 100% |
| 9 | Cross-Validation | 4 | ✅ 100% |

**Total**: 66 tests, 100% passing

See [REGRESSION_MODULE_COMPLETE_SUMMARY.md](../../docs/REGRESSION_MODULE_COMPLETE_SUMMARY.md) for details.

---

## 📈 Supported Models

| Model | Type | Best For |
|-------|------|----------|
| Ridge | Linear | High dimensionality, regularized |
| Lasso | Linear | Feature selection, sparse |
| ElasticNet | Linear | Combined L1/L2 regularization |
| Random Forest | Ensemble | Non-linear, robust |
| Gradient Boosting | Ensemble | High accuracy, tunable |
| SVR | Kernel | Non-linear patterns |
| KNN | Instance | Local patterns |
| MLP | Neural Network | Complex non-linear |
| XGBoost | Ensemble | Competition-grade performance |

---

## 📊 Metrics

All models evaluated with 4 core metrics:

- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (Coefficient of Determination): Variance explained
- **MSE** (Mean Squared Error): Squared error average

---

## 📄 License

This module is part of the DockTKinase project.

---

## 🤝 Contributing

### Guidelines

1. **Keep it simple**: Avoid over-engineering
2. **Type hints**: Use annotations for clarity
3. **Tests**: Maintain 100% coverage
4. **Documentation**: Clear docstrings
5. **Consistency**: Follow existing patterns

---

## 📞 Support

- **Repository**: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- **Tests**: See `tests/regression_test/` for examples
- **Docs**: `docs/REGRESSION_MODULE_COMPLETE_SUMMARY.md`

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-10  
**Status**: Production Ready ✅
