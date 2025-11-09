# Classifier Module

**Version**: 1.0.0  
**Python**: 3.12+  
**PyTorch**: 2.6+

A production-ready, modular MLP-based binary classification system with comprehensive training, evaluation, and hyperparameter optimization capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Structure](#module-structure)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Performance](#performance)

---

## 🎯 Overview

The Classifier module provides a complete machine learning pipeline for binary classification using Multi-Layer Perceptron (MLP) neural networks. It includes:

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, training, evaluation, and utilities
- **Production-Ready**: Comprehensive error handling, validation, and logging
- **Flexible Configuration**: Multiple configuration options via files, CLI, or programmatic API
- **Performance Optimized**: Supports mixed precision (AMP), gradient clipping, and device management
- **Hyperparameter Optimization**: Integrated Optuna support for automated tuning
- **Cross-Validation**: Stratified K-Fold cross-validation with comprehensive metrics
- **Robust Data Handling**: Built-in data validation, stratified splitting, and imbalance detection

---

## ✨ Features

### Core Features
- ✅ **MLP Binary Classifier** with customizable architecture
- ✅ **Automated Training Pipeline** with early stopping
- ✅ **Cross-Validation** (StratifiedKFold)
- ✅ **Hyperparameter Optimization** (Optuna)
- ✅ **Mixed Precision Training** (AMP)
- ✅ **Gradient Clipping** for stability
- ✅ **Learning Rate Scheduling** (ReduceLROnPlateau)
- ✅ **Comprehensive Metrics** (Accuracy, Precision, Recall, F1, ROC-AUC, MCC)
- ✅ **Model Serialization** (save/load checkpoints)
- ✅ **Device Management** (CPU/GPU auto-detection)
- ✅ **Data Validation** with quality reports
- ✅ **Stratified Train/Test Split** with statistical validation

### Advanced Features
- ✅ **Chi-Square Testing** for split validation
- ✅ **Imbalance Detection** and reporting
- ✅ **Optional Dependencies** with graceful degradation
- ✅ **Robust Import System** for flexible deployment
- ✅ **CLI Interface** with comprehensive arguments
- ✅ **Configuration Management** with templates
- ✅ **PySpark Integration** for metrics tracking

---

## 📦 Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Optional dependencies
optuna>=3.0.0  # For hyperparameter optimization
scipy>=1.11.0  # For statistical tests
pyspark>=3.4.0  # For Spark-based metrics
```

### Install from Source

```bash
cd /path/to/docktkinase
pip install -e .
```

---

## 🚀 Quick Start

### 1. Command Line Interface

```bash
# Manual training with specified hyperparameters
python -m classifier.modular_classifier \
    embeddings.npy labels.npy \
    --mode manual \
    --lr 0.001 \
    --batch_size 64 \
    --epochs 50 \
    --early_stopping_patience 5

# Hyperparameter optimization with Optuna
python -m classifier.modular_classifier \
    embeddings.npy labels.npy \
    --mode optuna \
    --trials 100 \
    --cv_folds 5

# Advanced pipeline with configuration
python -m classifier.main \
    --data_path data.csv \
    --config_path config.json \
    --mode full \
    --output_dir results/
```

### 2. Python API

```python
from classifier.modular_pipeline import MLPEmbeddingPipeline

# Create pipeline
pipeline = MLPEmbeddingPipeline(
    embeddings_path="embeddings.npy",
    labels_path="labels.npy",
    batch_size=64,
    lr=0.001,
    epochs=50,
    early_stopping_patience=5
)

# Load data
pipeline.load_data()

# Train model
val_loss = pipeline.train()

# Cross-validation
cv_results = pipeline.cross_validate(k=5)

print(f"Validation Loss: {val_loss:.4f}")
print(f"CV Mean ROC-AUC: {cv_results['mean_roc_auc']:.4f}")
```

### 3. Advanced Pipeline API

```python
from classifier.main import MLPPipeline
from pathlib import Path

# Initialize with auto-configuration
pipeline = MLPPipeline(
    config_template="production",  # or "development", "research"
    auto_configure=True
)

# Load data (auto-configures based on data size)
pipeline.load_data(
    Path("data.csv"),
    target_column="target"
)

# Run full pipeline
pipeline.run_hyperparameter_optimization(n_trials=100)
pipeline.run_cross_validation(n_folds=5)
pipeline.train_final_model(train_ratio=0.8)

# Save results
pipeline.save_results(Path("results/"))
```

---

## 📁 Module Structure

```
src/classifier/
├── __init__.py                    # Package initialization
├── classifier.py                  # Legacy monolithic script
├── modular_classifier.py          # Modern CLI interface
├── modular_pipeline.py            # Modular pipeline implementation
├── main.py                        # Advanced pipeline entry point
│
├── config/                        # Configuration management
│   ├── __init__.py
│   └── mlp_config.py             # MLPConfig class
│
├── core/                          # Core training and evaluation
│   ├── __init__.py
│   ├── data_loader.py            # DataManager class
│   ├── data_manager.py           # Dataset & DataManager classes
│   ├── trainer.py                # ModelTrainer & TrainingConfig
│   ├── evaluator.py              # ModelEvaluator & metrics
│   ├── cross_validator.py        # CrossValidator & K-Fold CV
│   └── hyperopt.py               # HyperparameterOptimizer (Optuna)
│
├── models/                        # Neural network models
│   ├── __init__.py
│   ├── base_model.py             # BaseClassifier abstract class
│   ├── mlp.py                    # MLPEmbeddingClassifier (main)
│   └── mlp_classifier.py         # MLPEmbeddingClassifier (legacy)
│
├── optional/                      # Optional features
│   ├── __init__.py
│   └── hyperopt.py               # Hyperparameter optimization (alias)
│
└── utils/                         # Utility modules
    ├── __init__.py
    ├── config_manager.py         # SimpleConfig & templates
    ├── data_validation.py        # DataValidator & quality checks
    ├── device_manager.py         # SmartDeviceManager (CPU/GPU)
    ├── import_utils.py           # Robust import system
    ├── metrics.py                # MetricsCalculator & aggregation
    ├── optional_deps.py          # Dependency management
    ├── train_test_split.py       # TrainTestSplitter
    └── robust_train_test_split.py # RobustTrainTestSplitter
```

---

## 📚 API Reference

### Core Classes

#### `MLPEmbeddingPipeline`
**Module**: `classifier.modular_pipeline`

Complete pipeline for MLP training with Spark integration.

```python
class MLPEmbeddingPipeline:
    def __init__(
        self,
        embeddings_path: str,
        labels_path: str,
        batch_size: int = 64,
        lr: float = 0.001,
        epochs: int = 50,
        test_split: float = 0.1,
        val_split: float = 0.1,
        early_stopping_patience: int = 5,
        model_output: str = "mlp_model.pth",
        metrics_output: str = "training_metrics.json"
    )
    
    def load_data(
        self,
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None
    ) -> None
    
    def train(
        self,
        train_idx: Optional[np.ndarray] = None,
        val_idx: Optional[np.ndarray] = None,
        test_idx: Optional[np.ndarray] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> float
    
    def cross_validate(self, k: int = 5) -> Dict[str, Any]
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, Any]
```

**Example**:
```python
pipeline = MLPEmbeddingPipeline(
    "embeddings.npy", "labels.npy",
    batch_size=64, lr=0.001, epochs=50
)
pipeline.load_data()
val_loss = pipeline.train()
```

---

#### `MLPPipeline`
**Module**: `classifier.main`

Advanced pipeline with auto-configuration and comprehensive features.

```python
class MLPPipeline:
    def __init__(
        self,
        config_template: Optional[str] = None,
        device_requirement: Optional[str] = None,
        enable_benchmarking: Optional[bool] = None,
        min_gpu_memory_gb: Optional[float] = None,
        **config_overrides
    )
    
    def load_data(
        self,
        data_path: Path,
        target_column: str = "target",
        feature_columns: Optional[list] = None,
        batch_size: Optional[int] = None
    ) -> None
    
    def auto_configure_for_data(
        self,
        n_samples: int,
        n_features: int,
        n_classes: Optional[int] = None
    ) -> None
    
    def run_cross_validation(
        self,
        n_folds: int = 5
    ) -> Dict[str, Any]
    
    def run_hyperparameter_optimization(
        self,
        n_trials: int = 100,
        cv_folds: int = 3
    ) -> Tuple[MLPConfig, TrainingConfig]
    
    def train_final_model(
        self,
        train_ratio: float = 0.8,
        use_robust_split: bool = True
    ) -> Dict[str, Any]
    
    def save_results(self, output_dir: Path) -> None
    
    def save_config(
        self,
        path: Union[str, Path],
        format: str = "json"
    ) -> None
    
    def load_config_file(self, path: Union[str, Path]) -> None
    
    def validate_device_status(self) -> bool
```

**Example**:
```python
pipeline = MLPPipeline(config_template="production")
pipeline.load_data(Path("data.csv"), target_column="label")
pipeline.run_hyperparameter_optimization(n_trials=50)
pipeline.train_final_model()
pipeline.save_results(Path("output/"))
```

---

### Model Classes

#### `MLPEmbeddingClassifier`
**Module**: `classifier.models.mlp`

Multi-Layer Perceptron for binary classification.

```python
class MLPEmbeddingClassifier(BaseClassifier):
    def __init__(
        self,
        config: MLPConfig,
        input_size: Optional[int] = None
    )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        layer_idx: int = -2
    ) -> torch.Tensor
    
    def count_parameters(self) -> int
```

**Architecture**:
- Input layer (configurable)
- Hidden layers with dropout and batch normalization
- Output layer (1 unit for binary classification)
- BCEWithLogitsLoss for training

**Example**:
```python
from classifier.models.mlp import MLPEmbeddingClassifier
from classifier.config.mlp_config import MLPConfig

config = MLPConfig(
    hidden_layers=[256, 128, 64],
    dropout_rate=0.3,
    learning_rate=0.001
)
model = MLPEmbeddingClassifier(config, input_size=512)
```

---

### Training & Evaluation

#### `ModelTrainer`
**Module**: `classifier.core.trainer`

Handles model training with early stopping and advanced features.

```python
class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device
    )
    
    def setup_training(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> TrainingHistory
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None
    
    def load_checkpoint(
        self,
        path: Union[str, Path]
    ) -> Dict[str, Any]
```

**Features**:
- Early stopping with patience
- Mixed precision (AMP) training
- Gradient clipping
- Learning rate scheduling
- Checkpoint management

**Example**:
```python
from classifier.core.trainer import ModelTrainer, TrainingConfig

config = TrainingConfig(
    max_epochs=100,
    early_stopping_patience=10,
    gradient_clip_val=1.0,
    amp_enabled=True
)

trainer = ModelTrainer(model, config, device)
trainer.setup_training(optimizer=torch.optim.Adam(model.parameters()))
history = trainer.train(train_loader, val_loader)
```

---

#### `ModelEvaluator`
**Module**: `classifier.core.evaluator`

Computes comprehensive classification metrics.

```python
class ModelEvaluator:
    def __init__(self, device: torch.device)
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, Any]
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]
```

**Metrics Computed**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

**Example**:
```python
from classifier.core.evaluator import ModelEvaluator

evaluator = ModelEvaluator(device)
metrics = evaluator.evaluate(model, test_loader)
print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
```

---

#### `CrossValidator`
**Module**: `classifier.core.cross_validator`

Stratified K-Fold cross-validation.

```python
class CrossValidator:
    def __init__(
        self,
        model_config: MLPConfig,
        cv_config: CrossValidationConfig,
        device: torch.device
    )
    
    def run(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, Any]
    
    def get_best_fold(self) -> int
    
    def get_fold_metrics(self, fold_idx: int) -> Dict[str, float]

# Convenience function
def quick_cross_validate(
    model_config: MLPConfig,
    X: torch.Tensor,
    y: torch.Tensor,
    n_splits: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, Any]
```

**Example**:
```python
from classifier.core.cross_validator import quick_cross_validate
from classifier.config.mlp_config import MLPConfig

config = MLPConfig(hidden_layers=[128, 64])
results = quick_cross_validate(config, X, y, n_splits=5)

print(f"Mean ROC-AUC: {results['summary_statistics']['roc_auc']['mean']:.4f}")
print(f"Std ROC-AUC: {results['summary_statistics']['roc_auc']['std']:.4f}")
```

---

### Hyperparameter Optimization

#### `HyperparameterOptimizer`
**Module**: `classifier.core.hyperopt`

Automated hyperparameter tuning with Optuna.

```python
class HyperparameterOptimizer:
    def __init__(
        self,
        base_config: MLPConfig,
        optimization_config: OptimizationConfig,
        device: torch.device
    )
    
    def optimize(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_trials: int = 100
    ) -> Tuple[MLPConfig, TrainingConfig, optuna.Study]
    
    def create_search_space(self, trial: optuna.Trial) -> Dict[str, Any]

# Convenience function
def quick_hyperparameter_search(
    X: torch.Tensor,
    y: torch.Tensor,
    base_config: MLPConfig,
    n_trials: int = 100,
    cv_folds: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[MLPConfig, TrainingConfig, optuna.Study]
```

**Search Space**:
- Learning rate: [1e-5, 1e-2] (log scale)
- Batch size: [16, 32, 64, 128, 256]
- Hidden dimensions: [32-512]
- Dropout rate: [0.0, 0.5]
- Number of layers: [1-4]

**Example**:
```python
from classifier.core.hyperopt import quick_hyperparameter_search
from classifier.config.mlp_config import create_default_config

base_config = create_default_config()
best_model_cfg, best_train_cfg, study = quick_hyperparameter_search(
    X, y, base_config, n_trials=100, cv_folds=5
)

print(f"Best trial ROC-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

### Utility Classes

#### `SmartDeviceManager`
**Module**: `classifier.utils.device_manager`

Intelligent CPU/GPU device selection and management.

```python
class SmartDeviceManager(DeviceManager):
    def __init__(
        self,
        enable_benchmarking: bool = False,
        min_gpu_memory_gb: float = 1.0,
        prefer_gpu: bool = True
    )
    
    def get_device(
        self,
        requirement: str = "auto"
    ) -> torch.device
    
    def get_device_info(self) -> Optional[DeviceInfo]
    
    def validate_current_device(self) -> bool
    
    def benchmark_devices(self) -> Dict[str, float]

# Convenience function
def get_best_device(
    requirement: str = "auto",
    min_gpu_memory_gb: float = 1.0,
    enable_benchmarking: bool = False
) -> torch.device
```

**Device Requirements**:
- `"auto"`: Best available device
- `"gpu_only"`: Force GPU (error if unavailable)
- `"cpu_only"`: Force CPU
- `"fastest"`: Benchmark and select fastest

**Example**:
```python
from classifier.utils.device_manager import SmartDeviceManager

device_mgr = SmartDeviceManager(
    enable_benchmarking=True,
    min_gpu_memory_gb=2.0
)
device = device_mgr.get_device("auto")
print(f"Selected device: {device}")
```

---

#### `DataValidator`
**Module**: `classifier.utils.data_validation`

Data quality checks and validation.

```python
class DataValidator:
    def __init__(self)
    
    def validate_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> DataQualityReport
    
    def validate_files(
        self,
        embeddings_path: str,
        labels_path: str
    ) -> DataQualityReport
    
    def check_class_balance(
        self,
        y: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, Any]

# Convenience functions
def quick_data_check(
    embeddings_path: str,
    labels_path: str
) -> bool

def get_data_statistics(
    embeddings_path: str,
    labels_path: str
) -> Dict[str, Any]
```

**Checks**:
- NaN/Inf detection
- Data type validation
- Shape compatibility
- Class balance
- Value range analysis

**Example**:
```python
from classifier.utils.data_validation import DataValidator

validator = DataValidator()
report = validator.validate_arrays(X, y)

if report.is_valid:
    print("✅ Data validation passed")
else:
    print("❌ Issues found:")
    for issue in report.issues:
        print(f"  - {issue}")
```

---

#### `TrainTestSplitter`
**Module**: `classifier.utils.train_test_split`

Stratified train/test splitting with validation.

```python
class TrainTestSplitter:
    def __init__(
        self,
        random_state: Optional[int] = None,
        verbose: bool = False
    )
    
    def split(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        test_size: float = 0.2,
        stratify: bool = True,
        min_samples_per_class: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    
    def validate_split(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> SplitValidationReport

# Convenience function
def robust_train_test_split(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True,
    validate_split: bool = True
) -> Union[Tuple, Tuple[..., SplitValidationReport]]
```

**Features**:
- Stratified splitting
- Chi-square validation
- Imbalance detection
- Distribution analysis

**Example**:
```python
from classifier.utils.train_test_split import TrainTestSplitter

splitter = TrainTestSplitter(random_state=42, verbose=True)
X_train, X_test, y_train, y_test = splitter.split(
    X, y, test_size=0.2, stratify=True
)

report = splitter.validate_split(y_train, y_test)
print(f"Chi-square p-value: {report.chi_square_p_value:.4f}")
print(f"Imbalance ratio: {report.imbalance_ratio:.2f}")
```

---

#### `MetricsCalculator`
**Module**: `classifier.utils.metrics`

Advanced metrics computation and aggregation.

```python
class MetricsCalculator:
    def __init__(self, device: torch.device)
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        amp_enabled: bool = False
    ) -> ClassificationMetrics
    
    def compute_batch_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> Dict[str, float]

class MetricsAggregator:
    def __init__(self)
    
    def add_fold_metrics(
        self,
        fold_idx: int,
        metrics: Dict[str, float]
    ) -> None
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]
    
    def get_best_fold(self, metric: str = "roc_auc") -> int

# Utility functions
def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]
```

**Example**:
```python
from classifier.utils.metrics import MetricsCalculator, find_optimal_threshold

calc = MetricsCalculator(device)
metrics = calc.evaluate_model(model, test_loader)

# Find optimal threshold
best_threshold, best_f1 = find_optimal_threshold(
    y_true, y_prob, metric="f1"
)
print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
```

---

#### `SimpleConfig`
**Module**: `classifier.utils.config_manager`

Configuration management with templates.

```python
class SimpleConfig:
    def __init__(
        self,
        model: Optional[Dict] = None,
        training: Optional[Dict] = None,
        data: Optional[Dict] = None,
        device: Optional[Dict] = None
    )
    
    def to_dict(self) -> Dict[str, Any]
    
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimpleConfig'
    
    def save(
        self,
        path: Union[str, Path],
        format: str = "json"
    ) -> None
    
    def load(
        cls,
        path: Union[str, Path]
    ) -> 'SimpleConfig'

def create_default_config() -> SimpleConfig
```

**Templates**:
- `"development"`: Fast iteration, small models
- `"production"`: Balanced performance
- `"research"`: Large models, extensive tuning

**Example**:
```python
from classifier.utils.config_manager import SimpleConfig

config = SimpleConfig()
config.model.hidden_layers = [256, 128, 64]
config.training.max_epochs = 100
config.save("config.json")

# Load config
loaded_config = SimpleConfig.load("config.json")
```

---

### Configuration Classes

#### `MLPConfig`
**Module**: `classifier.config.mlp_config`

Model architecture configuration.

```python
class MLPConfig:
    input_size: Optional[int] = None
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    output_size: int = 1
    dropout_rate: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

def create_default_config() -> MLPConfig
```

---

#### `TrainingConfig`
**Module**: `classifier.core.trainer`

Training configuration.

```python
class TrainingConfig:
    max_epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    gradient_clip_val: Optional[float] = 1.0
    amp_enabled: bool = False
    log_interval: int = 10
    validation_interval: int = 1
    checkpoint_dir: Optional[Path] = None
```

---

#### `CrossValidationConfig`
**Module**: `classifier.core.cross_validator`

Cross-validation configuration.

```python
class CrossValidationConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = 42
    stratify: bool = True
    verbose: bool = True
```

---

#### `OptimizationConfig`
**Module**: `classifier.core.hyperopt`

Hyperparameter optimization configuration.

```python
class OptimizationConfig:
    n_trials: int = 100
    sampler: str = "tpe"  # "tpe", "random", "cmaes"
    direction: str = "minimize"
    cv_folds: int = 3
    metric: str = "roc_auc"
    timeout: Optional[int] = None
    pruner: Optional[str] = "median"
```

---

## 💡 Examples

### Example 1: Basic Training

```python
from classifier.modular_pipeline import MLPEmbeddingPipeline

# Initialize pipeline
pipeline = MLPEmbeddingPipeline(
    embeddings_path="data/embeddings.npy",
    labels_path="data/labels.npy",
    batch_size=64,
    lr=0.001,
    epochs=50
)

# Train
pipeline.load_data()
val_loss = pipeline.train()
print(f"Final validation loss: {val_loss:.4f}")
```

### Example 2: Cross-Validation

```python
from classifier.core.cross_validator import quick_cross_validate
from classifier.config.mlp_config import create_default_config
import torch

# Load data
X = torch.load("embeddings.pt")
y = torch.load("labels.pt")

# Configure model
config = create_default_config()
config.hidden_layers = [256, 128, 64, 32]
config.dropout_rate = 0.4

# Run CV
results = quick_cross_validate(config, X, y, n_splits=5)

# Print results
print(f"Mean ROC-AUC: {results['summary_statistics']['roc_auc']['mean']:.4f}")
print(f"Std ROC-AUC: {results['summary_statistics']['roc_auc']['std']:.4f}")
print(f"Best fold: {results['best_fold']}")
```

### Example 3: Hyperparameter Optimization

```python
from classifier.core.hyperopt import quick_hyperparameter_search
from classifier.config.mlp_config import create_default_config
import torch

# Load data
X = torch.load("embeddings.pt")
y = torch.load("labels.pt")

# Base configuration
base_config = create_default_config()

# Optimize
best_model_cfg, best_train_cfg, study = quick_hyperparameter_search(
    X, y,
    base_config=base_config,
    n_trials=100,
    cv_folds=5
)

# Best parameters
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print(f"\nBest ROC-AUC: {study.best_value:.4f}")
```

### Example 4: Advanced Pipeline

```python
from classifier.main import MLPPipeline
from pathlib import Path

# Initialize with production template
pipeline = MLPPipeline(
    config_template="production",
    enable_benchmarking=True,
    min_gpu_memory_gb=2.0
)

# Load data (auto-configures)
pipeline.load_data(
    Path("data/features.csv"),
    target_column="is_active",
    batch_size=128
)

# Full workflow
pipeline.run_hyperparameter_optimization(n_trials=50, cv_folds=5)
pipeline.run_cross_validation(n_folds=10)
final_results = pipeline.train_final_model(train_ratio=0.8)

# Save everything
pipeline.save_results(Path("results/run_001/"))
pipeline.save_config(Path("results/run_001/config.json"))

# Results
print(f"Test ROC-AUC: {final_results['test_metrics'].roc_auc:.4f}")
print(f"Test Accuracy: {final_results['test_metrics'].accuracy:.4f}")
```

### Example 5: Custom Training Loop

```python
from classifier.models.mlp import MLPEmbeddingClassifier
from classifier.core.trainer import ModelTrainer, TrainingConfig
from classifier.config.mlp_config import MLPConfig
import torch

# Create model
config = MLPConfig(
    hidden_layers=[512, 256, 128],
    dropout_rate=0.3,
    learning_rate=0.001
)
model = MLPEmbeddingClassifier(config, input_size=1024)

# Training configuration
train_config = TrainingConfig(
    max_epochs=100,
    early_stopping_patience=15,
    gradient_clip_val=1.0,
    amp_enabled=True
)

# Setup trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = ModelTrainer(model, train_config, device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)
trainer.setup_training(optimizer)

# Train
history = trainer.train(train_loader, val_loader)

# Results
print(f"Best epoch: {history.best_epoch}")
print(f"Best val loss: {history.best_val_loss:.4f}")

# Save checkpoint
trainer.save_checkpoint(
    "best_model.pt",
    epoch=history.best_epoch,
    loss=history.best_val_loss,
    metrics=history.get_summary()
)
```

### Example 6: Data Validation

```python
from classifier.utils.data_validation import DataValidator, quick_data_check
import numpy as np

# Quick check
is_valid = quick_data_check("embeddings.npy", "labels.npy")
print(f"Data valid: {is_valid}")

# Detailed validation
validator = DataValidator()
X = np.load("embeddings.npy")
y = np.load("labels.npy")

report = validator.validate_arrays(X, y)

print(f"\nValidation Report:")
print(f"  Valid: {report.is_valid}")
print(f"  Samples: {report.n_samples}")
print(f"  Features: {report.n_features}")
print(f"  Classes: {report.n_classes}")
print(f"  Has NaN: {report.has_nan}")
print(f"  Has Inf: {report.has_inf}")

if report.issues:
    print("\nIssues:")
    for issue in report.issues:
        print(f"  ⚠️  {issue}")

if report.warnings:
    print("\nWarnings:")
    for warning in report.warnings:
        print(f"  ⚠️  {warning}")

# Check class balance
balance_info = validator.check_class_balance(y)
print(f"\nClass balance: {balance_info['is_balanced']}")
print(f"Imbalance ratio: {balance_info['imbalance_ratio']:.2f}")
```

---

## 🧪 Testing

The classifier module has **100% test coverage** with 93 tests across 12 levels.

### Run All Tests

```bash
cd tests/classifier_test/
python run_all_tests.py
```

### Run Specific Test Levels

```bash
# Level 1: Foundation
python run_level1_unit.py

# Level 2: Models
python run_level2_model.py

# Level 3: Training
python run_level3_training.py

# Individual tests
python test_6_performance.py
python test_9_hyperparameter_optimization.py
python test_12_cli_integration.py
```

### Test Coverage

| Level | Description | Tests | Status |
|-------|-------------|-------|--------|
| 1 | Foundation & Utils | 7 | ✅ 100% |
| 2 | Models | 3 | ✅ 100% |
| 3 | Training & Evaluation | 42 | ✅ 100% |
| 4 | Integration | 14 | ✅ 100% |
| 5 | Edge Cases | 7 | ✅ 100% |
| 6 | Performance | 3 | ✅ 100% |
| 7 | Serialization | 2 | ✅ 100% |
| 8 | End-to-End | 2 | ✅ 100% |
| 9 | Hyperparameter Opt | 3 | ✅ 100% |
| 10 | Optional Dependencies | 5 | ✅ 100% |
| 11 | Robust Split | 3 | ✅ 100% |
| 12 | CLI Integration | 2 | ✅ 100% |

**Total**: 93 tests, 100% passing

See [COMPLETE_TEST_REPORT.md](../../tests/classifier_test/COMPLETE_TEST_REPORT.md) for details.

---

## ⚡ Performance

### Benchmarks (CPU - Apple M1)

| Metric | Value |
|--------|-------|
| **Training Speed** | 10,000 - 34,000 samples/sec |
| **Inference Speed** | 29,000 - 5,700,000 samples/sec |
| **Memory Usage** | 0.00 - 0.64 MB |
| **Scaling Factor** | 1.24x (10x data increase) |

### Optimization Features

- ✅ **Mixed Precision (AMP)**: 2x faster training on GPU
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Batch Processing**: Efficient DataLoader with num_workers
- ✅ **Early Stopping**: Prevents unnecessary computation
- ✅ **Smart Device Selection**: Auto CPU/GPU with benchmarking

---

## 📄 License

This module is part of the DockTKinase project.

---

## 🤝 Contributing

### Code Structure Guidelines

1. **Modular Design**: Keep modules focused and independent
2. **Type Hints**: Use type annotations for all functions
3. **Documentation**: Docstrings for all public APIs
4. **Error Handling**: Comprehensive exception handling
5. **Testing**: 100% test coverage for new features
6. **Logging**: Use logging module, not print statements

### Adding New Features

1. Create feature in appropriate module (core/models/utils)
2. Add comprehensive tests (target: 100% coverage)
3. Update API documentation in README
4. Add examples demonstrating usage
5. Ensure backward compatibility

---

## 📞 Support

For issues, questions, or contributions:

- **Repository**: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- **Tests**: See `tests/classifier_test/` for comprehensive examples
- **Documentation**: This README and inline docstrings

---

## 🗺️ Roadmap

### Version 1.1 (Planned)
- [ ] Multi-class classification support
- [ ] Additional model architectures (CNN, Transformer)
- [ ] Distributed training (DDP)
- [ ] Enhanced visualization tools
- [ ] REST API interface

### Version 2.0 (Future)
- [ ] AutoML integration
- [ ] Model compression (pruning, quantization)
- [ ] ONNX export
- [ ] Cloud deployment templates
- [ ] Real-time inference server

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-08  
**Status**: Production Ready ✅
