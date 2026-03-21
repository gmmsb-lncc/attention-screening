# 🧪 Classifier Module - Test Suite

## 📋 Overview

This directory contains comprehensive tests for the **Classifier Module** (`src/classifier/`). The classifier is responsible for binary classification of protein-ligand interactions using MLP (Multi-Layer Perceptron) models.

## 🏗️ Module Structure

The classifier module has the following architecture:

```
src/classifier/
├── core/                        # Core classification logic (7 componentes)
│   ├── data_loader.py          # DataManager: Load .npy embeddings/labels
│   ├── data_manager.py         # Alternative data management (existe?)
│   ├── trainer.py              # ModelTrainer: Training with AMP, gradient clipping
│   ├── evaluator.py            # ModelEvaluator: 16 metrics calculation
│   ├── cross_validator.py      # CrossValidator: StratifiedKFold (corrigido)
│   └── hyperopt.py             # Hyperparameter optimization (Optuna - existe?)
│
├── models/                      # Model definitions (3 modelos)
│   ├── base_model.py           # BaseClassifier: Abstract interface
│   ├── mlp.py                  # MLP basic (existe?)
│   └── mlp_classifier.py       # MLPEmbeddingClassifier: 3-layer MLP
│                               #   Architecture: input → hidden → hidden//2 → 1
│                               #   BatchNorm1d handling for batch_size=1
│
├── utils/                       # Utilities (10 componentes)
│   ├── config_manager.py       # ConfigManager: Configuration handling
│   ├── device_manager.py       # DeviceManager: CPU/GPU auto-detection
│   ├── metrics.py              # MetricsCalculator: ClassificationMetrics dataclass
│   ├── data_validation.py      # DataValidator: Input validation
│   ├── train_test_split.py     # Stratified split (80/10/10)
│   ├── robust_train_test_split.py  # Robust splitting with edge cases
│   ├── simple_config.py        # Simple configuration (existe?)
│   ├── optional_deps.py        # Optional dependencies handling
│   └── import_utils.py         # Import utilities
│
├── config/                      # Configuration (existe?)
│   └── mlp_config.py           # MLPConfig dataclass
│
├── modular_pipeline.py          # MLPEmbeddingPipeline: Main orchestrator
│   └── Métodos: __init__, train(), cross_validate(), evaluate()
│   └── PySpark integration para métricas
│
├── modular_classifier.py        # CLI interface (argparse)
├── classifier.py                # Legacy classifier (backward compatibility)
└── test_modular.py             # Teste existente (simplificado)
```

## 🎯 Test Strategy

### Test Levels

Similar to the embeddings module, we'll organize tests in progressive levels based on **actual implementation**:

#### **Level 1: Unit Tests - Core Components** (~7 tests, ~45s)
- `test_1_1_data_loader.py`: DataManager (load_data, create_data_loaders, stratified split)
- `test_1_2_device_manager.py`: DeviceManager (CPU/GPU auto-detection)
- `test_1_3_metrics_calculator.py`: MetricsCalculator (16 metrics)
- `test_1_4_config_manager.py`: ConfigManager (configuration loading)
- `test_1_5_data_validation.py`: DataValidator (input validation)
- `test_1_6_import_utils.py`: Import utilities and optional dependencies
- `test_1_7_train_test_split.py`: Stratified split functions

#### **Level 2: Model Tests** (~4 tests, ~1min)
- `test_2_1_mlp_classifier.py`: MLPEmbeddingClassifier architecture
  - 3 layers: input_dim → hidden_dim → hidden_dim//2 → 1
  - BatchNorm1d with batch_size=1 handling
  - Forward pass correctness
- `test_2_2_base_classifier.py`: BaseClassifier interface (se existe)
- `test_2_3_model_forward.py`: Forward pass and predictions (sigmoid output)
- `test_2_4_model_gradients.py`: Gradient flow and backward pass

#### **Level 3: Training and Evaluation** (~6 tests, ~3min)
- `test_3_1_trainer.py`: ModelTrainer functionality
  - TrainingConfig dataclass
  - AMP (automatic mixed precision)
  - Gradient clipping
  - Learning rate scheduler
- `test_3_2_evaluator.py`: ModelEvaluator (16 metrics)
  - Accuracy, Precision, Recall, F1
  - ROC_AUC, Average Precision, Brier Score
  - MCC (Matthews Correlation Coefficient)
  - Confusion matrix (TP, TN, FP, FN)
  - Specificity, Fbeta (0.5, 2.0)
- `test_3_3_cross_validator.py`: CrossValidator
  - StratifiedKFold (n_splits=5)
  - FoldResult dataclass
  - No data leakage
- `test_3_4_early_stopping.py`: Early stopping mechanism (patience, min_delta)
- `test_3_5_checkpointing.py`: Model checkpointing
- `test_3_6_metrics_aggregation.py`: MetricsAggregator for CV

#### **Level 4: Integration Tests** (~5 tests, ~4min)
- `test_4_1_pipeline_basic.py`: Basic pipeline execution
  - Load data → Train → Evaluate
  - DataManager + ModelTrainer + ModelEvaluator
- `test_4_2_pipeline_cross_val.py`: Pipeline with cross-validation
  - CrossValidator integration
  - Fold results aggregation
- `test_4_3_pipeline_spark.py`: PySpark integration (metrics output)
- `test_4_4_cli_interface.py`: modular_classifier.py CLI testing
  - Argparse arguments
  - Mode: train, cross-validate, optuna
- `test_4_5_backward_compatibility.py`: Compare with classifier.py original

#### **Level 5: Edge Cases and Robustness** (~8 tests, ~2min)
- `test_5_1_edge_small_dataset.py`: Very small datasets (<20 samples)
- `test_5_2_edge_imbalanced.py`: Highly imbalanced (99:1, 95:5)
- `test_5_3_edge_batch_size_one.py`: Batch size = 1 (BatchNorm issue)
- `test_5_4_edge_nan_inf.py`: NaN/Inf in embeddings
- `test_5_5_edge_all_same_class.py`: All labels same class (edge for metrics)
- `test_5_6_edge_perfect_predictions.py`: All predictions correct
- `test_5_7_edge_gpu_unavailable.py`: GPU unavailable fallback
- `test_5_8_edge_memory_pressure.py`: Large batch sizes / OOM scenarios

#### **Level 6: Performance and Optimization** (~4 tests, ~3min)
- `test_6_1_training_throughput.py`: Training speed (samples/second)
- `test_6_2_memory_profiling.py`: Memory usage tracking
- `test_6_3_amp_effectiveness.py`: AMP speedup and accuracy
- `test_6_4_convergence_analysis.py`: Loss convergence patterns

#### **Level 7: Serialization and I/O** (~4 tests, ~2min)
- `test_7_1_model_save_load.py`: Model checkpoint save/load
- `test_7_2_metrics_json_export.py`: Metrics to JSON (DataTypeConverter)
- `test_7_3_spark_dataframe_output.py`: PySpark DataFrame output
- `test_7_4_config_persistence.py`: Configuration save/load

#### **Level 8: End-to-End Scenarios** (~4 tests, ~6min)
- `test_8_1_e2e_full_pipeline.py`: Complete pipeline from scratch
  - Synthetic data → Train → Cross-validate → Export
- `test_8_2_e2e_production_scenario.py`: Production-like scenario
  - Real data format
  - Complete metrics output
  - Model persistence
- `test_8_3_e2e_comparison.py`: Compare modular vs. original classifier
- `test_8_4_e2e_regression_suite.py`: Full regression test suite

### Total: ~42 tests across 8 levels (~22 minutes)

## 🚀 Quick Start

### Run All Tests
```bash
cd tests/classifier_test
python run_all_tests.py
```

### Run Specific Level
```bash
# Level 1: Unit tests
python run_level1_unit.py

# Level 4: Integration tests
python run_level4_integration.py

# Level 8: End-to-end tests
python run_level8_e2e.py
```

### Run Individual Test
```bash
python test_1_data_loader.py
python test_3_trainer.py
python test_8_e2e_full_pipeline.py
```

## 📊 Test Data Requirements

The tests will use:
- **Synthetic data**: Generated embeddings and labels for unit tests
- **Small real data**: Subset of actual embeddings for integration tests
- **Full dataset**: Complete pipeline tests (optional, if available)

### Data Generation
```python
# Synthetic embeddings for testing
import numpy as np

# Generate test embeddings (100 samples, 256 features)
embeddings = np.random.randn(100, 256).astype(np.float32)
labels = np.random.randint(0, 2, 100).astype(np.int64)

# Save for tests
np.save('test_embeddings.npy', embeddings)
np.save('test_labels.npy', labels)
```

## ✅ Success Criteria

Each test should verify:

1. **Correctness**: Outputs match expected behavior
2. **Robustness**: Handles edge cases gracefully
3. **Performance**: Meets speed/memory requirements
4. **Compatibility**: Works with existing APIs
5. **Stability**: Produces consistent results

## 📈 Test Metrics to Track

- **Accuracy metrics**: Accuracy, Precision, Recall, F1
- **Training metrics**: Loss convergence, epoch time
- **Memory metrics**: Peak memory usage, memory leaks
- **Speed metrics**: Samples/second, epochs/minute
- **Robustness**: Error handling, edge case coverage

## 🎯 Key Areas to Test (Based on Actual Implementation)

### 1. Data Loading and Validation
- ✅ Load `.npy` files correctly (allow_pickle=True)
- ✅ Stratified train/val/test split (80/10/10)
- ✅ Handle labels flattening (`.flatten()`)
- ✅ Cache mechanism in DataManager (`_embeddings`, `_labels`, `_dataset`)
- ✅ Validate embedding dimensions
- ✅ DataValidator for input validation

### 2. Model Architecture (MLPEmbeddingClassifier)
- ✅ MLPEmbeddingClassifier initialization
- ✅ Three-layer architecture: input_dim → hidden_dim → hidden_dim//2 → 1
- ✅ Forward pass correctness
- ✅ **Critical**: BatchNorm1d behavior with batch_size=1
  ```python
  if x.size(0) > 1:
      x = self.bn1(x)  # Skip BatchNorm for single samples
  ```
- ✅ Dropout application (default: 0.3)
- ✅ Output sigmoid activation
- ✅ create_mlp_model() factory function

### 3. Training Process (ModelTrainer)
- ✅ TrainingConfig dataclass (max_epochs, patience, min_delta)
- ✅ Loss calculation (BCELoss)
- ✅ Optimizer (Adam) updates
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ **AMP (Automatic Mixed Precision)** support
  - amp_enabled, amp_dtype (float16/bfloat16)
  - GradScaler for gradient scaling
- ✅ Gradient clipping (gradient_clip_value, gradient_clip_norm)
- ✅ Early stopping triggers (monitor_metric, monitor_mode)
- ✅ TrainingHistory dataclass (tracking losses, metrics, epochs)

### 4. Evaluation System (ModelEvaluator)
- ✅ All 16 metrics calculated correctly:
  1. Loss
  2. Accuracy
  3. Precision
  4. Recall
  5. F1
  6. ROC_AUC
  7. True_Positives
  8. True_Negatives
  9. False_Positives
  10. False_Negatives
  11. Specificity
  12. Fbeta_0.5
  13. Fbeta_2
  14. MCC (Matthews Correlation Coefficient)
  15. Average_Precision
  16. Brier_Score
- ✅ ClassificationMetrics dataclass
- ✅ MetricsCalculator class
- ✅ Confusion matrix accuracy
- ✅ Handle edge cases: all same class, perfect predictions
- ✅ DataTypeConverter for JSON serialization (np.float64 → float)

### 5. Cross-Validation (CrossValidator)
- ✅ StratifiedKFold implementation
- ✅ CrossValidationConfig dataclass (n_splits, shuffle, random_state)
- ✅ FoldResult dataclass per fold
- ✅ **Critical**: No data leakage (correct index handling)
- ✅ Stratification maintained across folds
- ✅ MetricsAggregator for fold aggregation (mean, std)
- ✅ Fold independence validation

### 6. Pipeline Integration (MLPEmbeddingPipeline)
- ✅ PySpark integration (SparkSession)
- ✅ Methods: __init__, train(), cross_validate(), evaluate()
- ✅ Parameters match original: batch_size, lr, epochs, patience, etc.
- ✅ Early stopping integration
- ✅ Model save/load (model_output, metrics_output)
- ✅ Backwards compatibility with classifier.py

### 7. Edge Cases (Critical for Robustness)
- ✅ Single sample batch (batch_size=1) - BatchNorm skip
- ✅ All same class labels (metrics calculation)
- ✅ Perfect predictions (all correct) - avoid division by zero
- ✅ Random predictions (all wrong)
- ✅ NaN/Inf in embeddings - validation
- ✅ GPU unavailable fallback to CPU
- ✅ Highly imbalanced datasets (99:1 ratio)
- ✅ Very small datasets (<20 samples)

### 8. Configuration Management
- ✅ TrainingConfig dataclass validation
- ✅ CrossValidationConfig dataclass
- ✅ ConfigManager (if exists)
- ✅ Simple configuration handling

## 🔧 Test Configuration

```python
# Default test configuration
TEST_CONFIG = {
    'embedding_dim': 256,
    'hidden_dim': 128,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 3,  # Small for fast tests
    'patience': 2,
    'use_gpu': False,  # CPU for reproducibility
    'random_seed': 42
}
```

## 📝 Test Documentation Format

Each test file should include:
```python
"""
Test: <Test Name>
Level: <1-8>
Duration: ~<X seconds>
Description: <What is being tested>

Tests:
1. <Test case 1>
2. <Test case 2>
...

Requirements:
- <Dependency 1>
- <Dependency 2>
"""
```

## 🚨 Known Issues and Limitations

Document any known issues here:
- [ ] Batch size = 1 requires special BatchNorm handling
- [ ] GPU tests require CUDA-enabled device
- [ ] Optuna tests can be non-deterministic
- [ ] Memory profiling requires `memory_profiler`

## 🔄 Continuous Integration

These tests should be run:
- Before merging to main
- After any classifier module changes
- Weekly for regression testing
- Before releases

## 📚 Additional Resources

- **Module Documentation**: `src/classifier/README_MODULAR.md`
- **Original Classifier**: `src/classifier/classifier.py`
- **Modular Pipeline**: `src/classifier/modular_pipeline.py`
- **Example Usage**: `src/classifier/modular_classifier.py`

---

**Created**: November 2025  
**Author**: semantic-screening Team  
**Branch**: modules-testing  
**Status**: 🚧 In Development
