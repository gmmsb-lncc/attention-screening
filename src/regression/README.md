# Regression Module

**Version**: 1.0.0  
**Python**: 3.8+  
**Scikit-learn**: 1.3+

Modular regression pipeline for pKi prediction with 9 algorithms, stratified validation, and cross-validation support.

---

## 🎯 Overview

Complete pipeline for regression tasks supporting 9 machine learning algorithms with professional validation strategies:

- **Stratified Split**: 70/10/20 (train/val/test) maintaining target distribution
- **9 Models**: Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, SVR, KNN, MLP, XGBoost
- **K-Fold CV**: Professional cross-validation with fold-level statistics
- **4 Metrics**: MAE, RMSE, R², MSE
- **Reproducible**: Fixed random seeds for consistent results

---

## 📦 Installation

```bash
cd /path/to/docktkinase
pip install -e .
```

**Core Dependencies**:
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- joblib >= 1.3.0

---

## 🚀 Quick Start

### Basic Usage

```python
from regression.modular_pipeline import RegressionPipeline

# Initialize pipeline
pipeline = RegressionPipeline(
    embeddings_path="protein_embeddings.npy",
    targets_path="pki_values.npy",
    output_dir="results/",
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

# Load and split data
pipeline.load_data()

# Train all models (or specify subset)
results = pipeline.train_all_models()  # or train_all_models(['Ridge', 'XGBoost'])

# Best model
print(f"Best Model: {results['best_model']}")
print(f"Best MAE: {results['best_mae']:.3f}")

# Results saved automatically in output_dir/
```

### Cross-Validation

```python
from regression.core import quick_cross_validate

# Quick CV with default settings
cv_results = quick_cross_validate(
    X, y,
    model_names=['Ridge', 'Lasso', 'RandomForest'],
    n_splits=5,
    random_state=42
)

# Analyze results
for model_name, result in cv_results.items():
    mae_mean = result.get_mean_metric('mae')
    mae_std = result.get_std_metric('mae')
    print(f"{model_name}: MAE = {mae_mean:.3f} ± {mae_std:.3f}")
```

### Command Line

```bash
# Train with all default models
python -m regression.modular_regression \
    --embeddings protein_embeddings.npy \
    --targets pki_values.npy \
    --output results/

# Train specific models
python -m regression.modular_regression \
    --embeddings protein_embeddings.npy \
    --targets pki_values.npy \
    --models Ridge Lasso XGBoost \
    --output results/

# With cross-validation
python -m regression.modular_regression \
    --embeddings protein_embeddings.npy \
    --targets pki_values.npy \
    --cv-folds 5 \
    --output results/
```

---

## 📁 Module Structure

```
src/regression/
├── __init__.py                    # Package initialization
├── modular_pipeline.py            # RegressionPipeline (main)
├── modular_regression.py          # CLI entry point
├── config.py                      # RegressionConfig
├── logger.py                      # RegressionLogger
├── visualizer.py                  # RegressionVisualizer
├── validation.py                  # Data validation utilities
├── utils.py                       # Helper functions
│
├── core/                          # Core modules
│   ├── __init__.py
│   ├── data_loader.py            # DataManager (stratified split)
│   ├── trainer.py                # RegressionTrainer
│   ├── evaluator.py              # RegressionEvaluator
│   └── cross_validator.py        # RegressionCrossValidator
│
└── models/                        # Model definitions
    ├── __init__.py
    └── models.py                 # RegressionModels factory
```

---

## 📚 API Reference

### Core Classes

#### RegressionPipeline
**Module**: `regression.modular_pipeline`

Main pipeline for complete regression workflow.

```python
class RegressionPipeline:
    def __init__(
        self,
        embeddings_path: str,
        targets_path: str,
        output_dir: str = 'results/regression',
        models_to_train: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        verbose: bool = True
    )
    
    def load_data(self) -> None:
        """Load and split data (stratified 70/10/20)"""
    
    def train_all_models(
        self,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train multiple models and compare results"""
    
    def train_single_model(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """Train a single model"""
    
    def save_results(self) -> None:
        """Save metrics, predictions, and visualizations"""
```

**Available Models**:
- `Ridge` - L2 regularized linear regression
- `Lasso` - L1 regularized linear regression
- `ElasticNet` - Combined L1/L2 regularization
- `RandomForest` - Ensemble of decision trees
- `GradientBoosting` - Sequential boosted trees
- `SVR` - Support Vector Regression
- `KNN` - K-Nearest Neighbors
- `MLP` - Multi-Layer Perceptron
- `XGBoost` - Gradient boosting (competition-grade)

**Example**:
```python
pipeline = RegressionPipeline(
    embeddings_path="embeddings.npy",
    targets_path="pki.npy",
    output_dir="results/",
    models_to_train=['Ridge', 'XGBoost'],  # Optional: train subset
    random_state=42
)

pipeline.load_data()
results = pipeline.train_all_models()
pipeline.save_results()
```

---

#### RegressionCrossValidator
**Module**: `regression.core.cross_validator`

K-Fold cross-validation with comprehensive statistics.

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
    ) -> Dict[str, CrossValidationResults]:
        """Run K-Fold CV for multiple models"""
    
    def get_best_model(
        self,
        metric: str = 'mae'
    ) -> str:
        """Get best model by specified metric"""
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all models in DataFrame"""

# Convenience function
def quick_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_names: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, CrossValidationResults]
```

**CrossValidationResults**:
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

**Example**:
```python
from regression.core import RegressionCrossValidator, CrossValidationConfig
from regression.models import RegressionModels

# Configuration
config = CrossValidationConfig(
    n_splits=5,
    shuffle=True,
    random_state=42,
    verbose=True
)

# Models
models = {
    'Ridge': RegressionModels.get_model('Ridge'),
    'XGBoost': RegressionModels.get_model('XGBoost')
}

# Run CV
cv = RegressionCrossValidator(config)
results = cv.cross_validate(X, y, models, list(models.keys()))

# Best model
best = cv.get_best_model(metric='mae')
print(f"Best model: {best}")

# Compare
df = cv.compare_models()
print(df[['model', 'mae_mean', 'mae_std', 'r2_mean']])
```

---

#### RegressionEvaluator
**Module**: `regression.core.evaluator`

Compute and compare regression metrics.

```python
class RegressionEvaluator:
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """Evaluate predictions with 4 core metrics"""
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Compare multiple models in DataFrame"""
    
    def export_to_csv(
        self,
        results: Dict[str, Dict[str, float]],
        filepath: str
    ) -> None:
        """Export results to CSV"""
```

**Metrics**:
- **MAE** (Mean Absolute Error): Average absolute difference
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **R²** (Coefficient of Determination): Variance explained (0-1)
- **MSE** (Mean Squared Error): Average squared difference

**Example**:
```python
from regression.core import RegressionEvaluator

evaluator = RegressionEvaluator()

# Single model
metrics = evaluator.evaluate(y_true, y_pred, "Ridge")
print(f"MAE: {metrics['mae']:.3f}")
print(f"R²: {metrics['r2']:.4f}")

# Multiple models comparison
results = {
    'Ridge': evaluator.evaluate(y_true, y_pred_ridge, "Ridge"),
    'XGBoost': evaluator.evaluate(y_true, y_pred_xgb, "XGBoost")
}

df = evaluator.compare_models(results)
evaluator.export_to_csv(results, "model_comparison.csv")
```

---

#### DataManager
**Module**: `regression.core.data_loader`

Data loading with stratified splitting.

```python
class DataManager:
    def __init__(
        self,
        embeddings_path: str,
        targets_path: str
    )
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings and targets"""
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """Stratified train/val/test split (70/10/20)"""
```

**Stratification**: Splits data maintaining target value distribution across all sets.

**Example**:
```python
from regression.core import DataManager

dm = DataManager("embeddings.npy", "targets.npy")
X, y = dm.load_data()

X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(
    X, y, test_size=0.2, val_size=0.1, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

---

#### RegressionModels
**Module**: `regression.models.models`

Factory for 9 regression algorithms with sensible defaults.

```python
class RegressionModels:
    @staticmethod
    def get_model(model_name: str, **kwargs) -> Any:
        """Get single model by name"""
    
    @staticmethod
    def get_all_models() -> Dict[str, Any]:
        """Get all 9 models"""
    
    @staticmethod
    def get_model_names() -> List[str]:
        """List available model names"""
```

**Models & Hyperparameters**:

| Model | Key Parameters | Best For |
|-------|---------------|----------|
| Ridge | alpha=1.0 | High-dimensional, correlated features |
| Lasso | alpha=0.1 | Feature selection, sparse solutions |
| ElasticNet | alpha=0.1, l1_ratio=0.5 | Combined L1/L2 regularization |
| RandomForest | n_estimators=100 | Non-linear, robust to outliers |
| GradientBoosting | n_estimators=100, lr=0.1 | High accuracy, tunable |
| SVR | kernel='rbf', C=1.0 | Non-linear patterns |
| KNN | n_neighbors=5 | Local patterns, simple |
| MLP | hidden_layers=(100,50) | Complex non-linear |
| XGBoost | n_estimators=100, lr=0.1 | Competition-grade performance |

**Example**:
```python
from regression.models import RegressionModels

# Single model
ridge = RegressionModels.get_model('Ridge', alpha=2.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

# All models
all_models = RegressionModels.get_all_models()
for name, model in all_models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: R² = {score:.4f}")

# List available
print(RegressionModels.get_model_names())
```

---

### Configuration

#### RegressionConfig
**Module**: `regression.config`

Configuration management with presets.

```python
class RegressionConfig:
    # Data settings
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # Training settings
    models_to_train: List[str] = field(default_factory=list)
    cv_folds: int = 5
    
    # Output settings
    output_dir: str = "results/regression"
    save_models: bool = True
    save_predictions: bool = True
    
    # Visualization
    create_plots: bool = True
    plot_format: str = "png"
    
    # Verbosity
    verbose: bool = True

# Presets
def get_fast_config() -> RegressionConfig:
    """Fast iteration, fewer models"""

def get_production_config() -> RegressionConfig:
    """Production settings, all models"""

def get_debug_config() -> RegressionConfig:
    """Debug settings, minimal"""
```

**Example**:
```python
from regression.config import RegressionConfig, get_production_config

# Custom config
config = RegressionConfig(
    test_size=0.15,
    val_size=0.15,
    models_to_train=['Ridge', 'XGBoost'],
    cv_folds=10,
    verbose=True
)

# Production preset
config = get_production_config()
```

---

## 💡 Examples

### Example 1: Complete Pipeline

```python
from regression.modular_pipeline import RegressionPipeline

# Initialize
pipeline = RegressionPipeline(
    embeddings_path="data/protein_embeddings.npy",
    targets_path="data/pki_values.npy",
    output_dir="results/pki_prediction",
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

# Load data
pipeline.load_data()
print(f"Data loaded: {len(pipeline.X_train)} train samples")

# Train all models
results = pipeline.train_all_models()

# Results
print(f"\nBest Model: {results['best_model']}")
print(f"Best MAE: {results['best_mae']:.3f}")
print(f"Best R²: {results['best_r2']:.4f}")

# Save everything (metrics, predictions, plots)
pipeline.save_results()
print(f"\nResults saved to: {pipeline.output_dir}")
```

### Example 2: Train Specific Models

```python
from regression.modular_pipeline import RegressionPipeline

pipeline = RegressionPipeline(
    embeddings_path="embeddings.npy",
    targets_path="pki.npy",
    output_dir="results/",
    models_to_train=['Ridge', 'Lasso', 'XGBoost'],  # Only these 3
    random_state=42
)

pipeline.load_data()
results = pipeline.train_all_models()

# Access individual model results
for model_name in ['Ridge', 'Lasso', 'XGBoost']:
    metrics = results['models'][model_name]['test_metrics']
    print(f"{model_name}:")
    print(f"  MAE: {metrics['mae']:.3f}")
    print(f"  R²: {metrics['r2']:.4f}")
```

### Example 3: Cross-Validation

```python
from regression.core import quick_cross_validate
import numpy as np

# Load data
X = np.load("embeddings.npy")
y = np.load("pki.npy")

# Quick CV
cv_results = quick_cross_validate(
    X, y,
    model_names=['Ridge', 'RandomForest', 'XGBoost'],
    n_splits=5,
    random_state=42
)

# Analyze results
print("Cross-Validation Results (5-fold):")
print("-" * 50)

for model_name, result in cv_results.items():
    mae_mean = result.get_mean_metric('mae')
    mae_std = result.get_std_metric('mae')
    r2_mean = result.get_mean_metric('r2')
    r2_std = result.get_std_metric('r2')
    
    print(f"\n{model_name}:")
    print(f"  MAE: {mae_mean:.3f} ± {mae_std:.3f}")
    print(f"  R²:  {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"  Best fold: {result.best_fold}")
```

### Example 4: Manual Workflow

```python
from regression.core import DataManager, RegressionTrainer, RegressionEvaluator
from regression.models import RegressionModels

# 1. Load and split data
dm = DataManager("embeddings.npy", "pki.npy")
X, y = dm.load_data()
X_train, X_val, X_test, y_train, y_val, y_test = dm.split_data(
    X, y, test_size=0.2, val_size=0.1, random_state=42
)

# 2. Get models
models = {
    'Ridge': RegressionModels.get_model('Ridge'),
    'XGBoost': RegressionModels.get_model('XGBoost')
}

# 3. Train
trainer = RegressionTrainer()
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    trained_model = trainer.train(model, X_train, y_train)
    trained_models[name] = trained_model

# 4. Evaluate
evaluator = RegressionEvaluator()
results = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred, name)
    results[name] = metrics

# 5. Compare
df = evaluator.compare_models(results)
print("\nModel Comparison:")
print(df)
```

### Example 5: Custom Configuration

```python
from regression.config import RegressionConfig
from regression.modular_pipeline import run_regression_pipeline

# Custom config
config = RegressionConfig(
    # Data
    test_size=0.15,
    val_size=0.15,
    random_state=999,
    
    # Models
    models_to_train=['Ridge', 'Lasso', 'ElasticNet', 'XGBoost'],
    cv_folds=10,
    
    # Output
    output_dir="results/custom_run",
    save_models=True,
    save_predictions=True,
    create_plots=True,
    
    # Verbosity
    verbose=True
)

# Run pipeline with config
results = run_regression_pipeline(
    embeddings_path="embeddings.npy",
    targets_path="pki.npy",
    config=config
)

print(f"Best model: {results['best_model']}")
print(f"Best MAE: {results['best_mae']:.3f}")
```

---

## 🧪 Testing

Complete test suite with **66 tests** (100% passing).

### Run All Tests

```bash
cd tests/regression_test/
python -m pytest -v
```

### Test Levels

| Level | Tests | Description |
|-------|-------|-------------|
| 1 | 10 | Data loading & preprocessing |
| 2 | 6 | Feature engineering |
| 3 | 9 | Model training |
| 4 | 9 | Model evaluation |
| 5 | 7 | Hyperparameter optimization |
| 6 | 7 | Predictions & inference |
| 7 | 6 | Visualization |
| 8 | 8 | Error handling |
| 9 | 4 | Cross-validation |

**Total**: 66 tests, 100% passing ✅

---

## 📊 Output Structure

```
results/regression/
├── metrics/
│   ├── train_metrics.csv
│   ├── val_metrics.csv
│   ├── test_metrics.csv
│   └── summary.json
├── predictions/
│   ├── ridge_predictions.npy
│   ├── xgboost_predictions.npy
│   └── ...
├── models/
│   ├── ridge_model.joblib
│   ├── xgboost_model.joblib
│   └── ...
└── plots/
    ├── predictions_comparison.png
    ├── residuals.png
    └── model_ranking.png
```

---

## 🎯 Best Practices

1. **Always use stratified splitting** for robust validation
2. **Set random_state** for reproducibility
3. **Start with fast models** (Ridge, Lasso) before slower ones (XGBoost, MLP)
4. **Use cross-validation** for final model selection
5. **Check R²** - values < 0.3 suggest poor fit
6. **Monitor MAE** - main metric for regression performance

---

## 📄 License

Part of DockTKinase project.

---

## 📞 Support

- **Repository**: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- **Tests**: See `tests/regression_test/`
- **Issues**: GitHub Issues

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-10  
**Status**: Production Ready ✅
