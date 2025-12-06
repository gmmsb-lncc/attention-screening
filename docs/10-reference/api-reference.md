# API Reference

**Last Updated**: December 6, 2025  
**Version**: 2.1

This document provides a high-level reference for the core Python classes and interfaces in DockTKinase.

---

## 🏗️ Build Module (`src.build`)

The Build module handles embedding generation, matrix construction, and data preparation.

### `BuildConfig`
**Location**: `src.build.core.config.py`

Configuration manager for the build process. Automatically synchronizes model dimensions.

```python
class BuildConfig:
    def __init__(self, config_data: dict = None, **kwargs):
        """
        Initialize build configuration.
        
        Args:
            config_data: Dictionary with configuration overrides
            **kwargs: Individual configuration overrides
        """
    
    def get_model_dimensions(self) -> dict:
        """Returns {'protein_dim': int, 'ligand_dim': int, 'total_dim': int}"""
```

### `BuildPipeline`
**Location**: `src.build.pipeline.build_pipeline.py`

Main orchestrator for the build phase.

```python
class BuildPipeline:
    def __init__(self, input_tsv: str, output_dir: str, config: BuildConfig = None):
        """
        Initialize build pipeline.
        
        Args:
            input_tsv: Path to input data
            output_dir: Directory for results
            config: Optional BuildConfig instance
        """
        
    def run(self) -> dict:
        """Executes the full build pipeline and returns paths to generated files."""
```

---

## 🎯 Classifier Module (`src.classifier`)

Handles binary classification (Active/Inactive).

### `MultiModelPipeline`
**Location**: `src.classifier.multi_model_pipeline.py`

Trains and evaluates multiple classification models simultaneously.

```python
class MultiModelPipeline:
    def __init__(self, input_data: str, output_dir: str, models: list = None):
        """
        Args:
            input_data: Path to embedding matrix (.npz)
            output_dir: Directory for results
            models: List of model names (default: all 10)
        """
```

### Supported Models
- `RandomForest`, `XGBoost`, `GradientBoosting`, `LightGBM`, `CatBoost`
- `SVM`, `KNN`, `MLP`, `LogisticRegression`, `ExtraTrees`

---

## 📊 Regression Module (`src.regression`)

Handles quantitative prediction (Ki, Kd, IC50).

### `RegressionTrainer`
**Location**: `src.regression.modular_regression.py`

Main class for training regression models.

```python
class RegressionTrainer:
    def __init__(self, config: RegressionConfig):
        """
        Args:
            config: RegressionConfig object
        """
        
    def train_all_models(self):
        """Trains all configured regression models."""
```

### `RegressionConfig`
**Location**: `src.regression.config.py`

```python
class RegressionConfig:
    def __init__(self, data_path: str, output_dir: str, activity_type: str = 'Ki'):
        """
        Args:
            data_path: Path to embedding matrix
            output_dir: Output directory
            activity_type: 'Ki', 'Kd', or 'IC50'
        """
```

---

## 🧠 Attention Matrix (`src.attention_matrix`)

Deep learning models for affinity prediction.

### `CrossAttentionModel`
**Location**: `src.attention_matrix.model.py`

Standard CNN + Cross-Attention architecture.

```python
class CrossAttentionModel(nn.Module):
    def __init__(self, protein_dim=320, ligand_dim=768, hidden_dim=128, ...):
        """
        Args:
            protein_dim: Input dimension (e.g., 320 for ESM-2 8M)
            ligand_dim: Input dimension (e.g., 768 for SMI-TED)
        """
```

### `ImprovedCrossAttentionModel`
**Location**: `src.attention_matrix.model.py`

Enhanced architecture with deeper projections and multiple attention layers.

```python
class ImprovedCrossAttentionModel(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=256, ...):
        """
        Advanced model with stacked attention layers.
        """
```

### `VisionTransformerModel`
**Location**: `src.attention_matrix.model.py`

ViT-style architecture treating protein-ligand pairs as a single sequence.

```python
class VisionTransformerModel(nn.Module):
    def __init__(self, max_tokens=2048, ...):
        """
        Transformer encoder with [CLS] token pooling.
        """
```
