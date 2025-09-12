# DockTKinase Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgr### Parallelization
- **Optuna**: Automatic parallel trials
- **DataLoader**: Parallel loading with `num_workers`
- **GPU**: Automatic vectorized operations

## 🐛 Troubleshooting

### Common Issues

#### Error: "CUDA out of memory"
```bash
# Reduce batch size
python main.py --config_path config_small_batch.json
```

#### Error: "File not found"advanced molecular classification system based on Multi-Layer Perceptron (MLP) for predicting kinase binding properties. The classifier utilizes high-dimensional molecular embeddings to perform binary predictions with high accuracy and scientific robustness.

## 🎯 Key Features

- **Flexible MLP Architecture**: Dynamic neural layer configuration
- **Hyperparameter Optimization**: Optuna integration for automated search
- **Rigorous Cross-Validation**: Scientific validation with multiple folds
- **Complete Pipeline**: From data loading to evaluation
- **Production-Ready**: Robust system tested in real scenarios
- **Modular & Extensible**: Clean architecture for easy maintenance

## 📊 Performance

- **Average ROC-AUC**: 0.8496 ± 0.0131 (3-fold CV)
- **Training Time**: ~2-5 seconds per fold
- **GPU Support**: Automatic CUDA acceleration
- **Mixed Precision**: Optimized training with AMP

## 🏗️ System Architecture

```
src/classifier/
├── config/                 # Model configurations
│   ├── mlp_config.py      # Configuration dataclasses
│   └── mlp_config_new.py  # Extended configurations
├── core/                  # Core modules
│   ├── trainer.py         # Training system
│   ├── cross_validator.py # Cross-validation
│   └── hyperopt.py        # Hyperparameter optimization
├── models/                # Model architectures
│   ├── mlp.py            # MLP implementation
│   └── base_model.py     # Abstract base model
├── utils/                 # Utilities
│   ├── data_validation.py # Data validation
│   ├── metrics.py        # Metrics calculation
│   └── visualization.py  # Plotting and charts
├── tests/                # Unit tests
├── main.py              # Main CLI interface
└── README.md           # This documentation
```

## 🚀 Installation

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.8 (opcional, para GPU)
```

### Dependencies

```bash
pip install torch>=2.0.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install optuna>=4.0.0
pip install pyyaml
```

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/sulfierry/docktkinase.git
cd docktkinase/src/classifier

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"
```

## 💡 Quick Start

### Basic Example

```python
from pathlib import Path
from main import MLPPipeline

# Initialize pipeline
pipeline = MLPPipeline()

# Load data
pipeline.load_data("data.csv", target_column="target")
pipeline.load_config()  # Default configuration

# Simple training
results = pipeline.train_final_model(train_ratio=0.8)
print(f"ROC-AUC: {results['test_metrics'].roc_auc:.4f}")
```

### Command Line Interface

#### Basic Training
```bash
python main.py --data_path data.csv --target_column target --mode train
```

#### Cross-Validation
```bash
python main.py --data_path data.csv --mode cv --n_folds 5
```

#### Hyperparameter Optimization
```bash
python main.py --data_path data.csv --mode hyperopt --n_trials 100
```

#### Complete Pipeline
```bash
python main.py --data_path data.csv --mode full --n_trials 50 --n_folds 3
```

## 🔧 Configuration

### Model Configuration

```python
from config.mlp_config import MLPConfig

config = MLPConfig(
    input_size=1024,           # Embedding size
    hidden_layers=[512, 256],  # Hidden layer architecture
    activation="ReLU",         # Activation function
    dropout_rate=0.3,          # Dropout rate
    use_batch_norm=True,       # Batch normalization
    learning_rate=1e-3,        # Learning rate
    weight_decay=1e-4          # L2 regularization
)
```

### Training Configuration

```python
from core.trainer import TrainingConfig

training_config = TrainingConfig(
    max_epochs=50,             # Maximum epochs
    patience=10,               # Early stopping patience
    batch_size=64,             # Batch size
    amp_enabled=False,         # Mixed precision
    scheduler_factor=0.5,      # Scheduler factor
    scheduler_patience=5       # Scheduler patience
)
```

## 📈 Execution Modes

### 1. Simple Training (`train`)
Trains a single model with train/test split:
```bash
python main.py --data_path data.csv --mode train --train_ratio 0.8
```

**Outputs:**
- Trained model
- Test metrics
- Training history

### 2. Cross-Validation (`cv`)
Scientifically rigorous cross-validation:
```bash
python main.py --data_path data.csv --mode cv --n_folds 5
```

**Outputs:**
- Per-fold metrics: `accuracy ± std`
- Aggregated statistics
- Best fold identified

### 3. Hyperparameter Optimization (`hyperopt`)
Automatic search for best hyperparameters:
```bash
python main.py --data_path data.csv --mode hyperopt --n_trials 100
```

**Search space:**
- `hidden_layers`: [[64], [128], [128,64], [256,128], [512,256]]
- `dropout_rate`: [0.0, 0.6]
- `activation`: ["ReLU", "GELU", "LeakyReLU", "ELU"]
- `use_batch_norm`: [True, False]
- `learning_rate`: [1e-5, 1e-2] (log)
- `weight_decay`: [1e-6, 1e-3] (log)

### 4. Complete Pipeline (`full`)
Full execution: optimization + validation + training:
```bash
python main.py --data_path data.csv --mode full
```

## 📊 Data Format

### Input CSV File
```csv
feature_1,feature_2,...,feature_n,target
0.123,0.456,...,0.789,1
0.321,0.654,...,0.987,0
...
```

**Requirements:**
- CSV file with header
- Numeric features (float32)
- Binary target (0/1)
- Minimum 10 samples per class

### Automatic Validation
The system performs automatic validations:
- ✅ Missing values (NaN/inf)
- ✅ Class balance
- ✅ Adequate dimensionality
- ✅ Correct data types

## 📋 System Outputs

### Results Structure
```
results/run_YYYYMMDD_HHMMSS/
├── config.json           # Used configurations
├── results.json          # Detailed metrics
├── final_model.pt        # Trained model (PyTorch)
└── plots/               # Performance plots
    ├── training_curves.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

### Calculated Metrics
- **ROC-AUC**: Area Under ROC Curve
- **Accuracy**: Overall accuracy
- **Precision**: Precision per class
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean
- **Average Precision**: AP score
- **Brier Score**: Probabilistic calibration
- **Matthews Correlation**: MCC
- **Specificity**: True negative rate

## 🧪 Advanced Examples

### Custom Configuration

```python
# Create custom configuration
custom_config = MLPConfig(
    input_size=2048,
    hidden_layers=[1024, 512, 256, 128],
    activation="GELU",
    dropout_rate=0.4,
    use_batch_norm=True,
    learning_rate=5e-4,
    weight_decay=1e-5
)

# Save configuration
import json
with open("custom_config.json", "w") as f:
    json.dump({"model": custom_config.__dict__}, f, indent=2)

# Use configuration
python main.py --config_path custom_config.json --data_path data.csv
```

### Results Analysis

```python
import json
import torch

# Load results
with open("results/run_20250912_153615/results.json", "r") as f:
    results = json.load(f)

# Cross-validation metrics
cv_results = results["cross_validation"]["summary_statistics"]
print(f"ROC-AUC: {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")

# Load trained model
checkpoint = torch.load("results/run_20250912_153615/final_model.pt")
model_state = checkpoint["model_state_dict"]
config = checkpoint["model_config"]
```

### Prediction on New Data

```python
# Load trained model
from models.mlp import MLPEmbeddingClassifier

checkpoint = torch.load("results/run_20250912_153615/final_model.pt")
model = MLPEmbeddingClassifier(checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Make predictions
import pandas as pd
new_data = pd.read_csv("new_molecules.csv")
X_new = torch.tensor(new_data.values, dtype=torch.float32)

with torch.no_grad():
    predictions = model(X_new)
    probabilities = torch.sigmoid(predictions)

print(f"Predictions: {probabilities.numpy()}")
```

## 🔬 System Validation

### Automated Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_mlp.py::test_forward_pass
```

### Integrity Check
```bash
# Check imports
python -c "from main import MLPPipeline; print('✅ Imports OK')"

# Test with synthetic data
python main.py --data_path synthetic_data.csv --mode cv --n_folds 3
```

## ⚡ Performance Optimization

### GPU vs CPU
```bash
# Force CPU
python main.py --device cpu --data_path data.csv

# Force GPU (if available)
python main.py --device cuda --data_path data.csv

# Auto-detection (default)
python main.py --device auto --data_path data.csv
```

### Mixed Precision Training
```json
{
  "training": {
    "amp_enabled": true,
    "batch_size": 128
  }
}
```

### Paralelização
- **Otuna**: Trials paralelos automáticos
- **DataLoader**: Carregamento paralelo com `num_workers`
- **GPU**: Operações vetorizadas automáticas

## 🐛 Solução de Problemas

### Problemas Comuns

#### Erro: "CUDA out of memory"
```bash
# Reduzir batch size
python main.py --config_path config_small_batch.json
```

#### Erro: "File not found"
```bash
# Check path
ls -la data.csv
python main.py --data_path ./data/molecules.csv
```

#### Warning: "Dataset too small"
```bash
# System works, but consider increasing data
# Minimum recommended: 100 samples per class
```

### Detailed Logs
```bash
# Enable verbose logging
export PYTHONPATH=/path/to/docktkinase
python -u main.py --data_path data.csv --mode full 2>&1 | tee pipeline.log
```

## 📚 Scientific References

### Architecture
- **MLP Design**: Goodfellow et al., "Deep Learning" (2016)
- **Dropout**: Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
- **Batch Normalization**: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (2015)

### Optimization
- **Adam Optimizer**: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
- **Learning Rate Scheduling**: Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)
- **Hyperparameter Optimization**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)

### Validation
- **Cross-Validation**: Kohavi, "A Study of Cross-Validation and Bootstrap for Accuracy Estimation" (1995)
- **ROC Analysis**: Fawcett, "An Introduction to ROC Analysis" (2006)

## 🤝 Contributing

### Development
```bash
# Fork the repository
git clone https://github.com/your-username/docktkinase.git
cd docktkinase/src/classifier

# Create branch
git checkout -b feature/new-functionality

# Develop and test
python -m pytest tests/

# Commit and push
git commit -m "feat: add new functionality"
git push origin feature/new-functionality
```

### Code Standards
- **Type Hints**: Required for public functions
- **Docstrings**: Google style for documentation
- **Testing**: pytest for unit tests
- **Linting**: flake8 for code quality

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sulfierry** - *Lead Developer* - [@sulfierry](https://github.com/sulfierry)

## 🙏 Acknowledgments

- PyTorch team for the exceptional framework
- Optuna developers for hyperparameter optimization
- Scikit-learn for API inspiration
- Open-source community for continuous collaboration

---

## 🔄 Version History

### v1.0.0 - Production-Ready System
- ✅ Complete MLP implementation
- ✅ Cross-validation pipeline
- ✅ Hyperparameter optimization
- ✅ Robust CLI interface
- ✅ Automated testing
- ✅ Complete documentation

### Future Versions
- [ ] Regression support
- [ ] Ensemble models
- [ ] Web interface
- [ ] Containerized deployment

---

**Status**: 🚀 **Production-Ready** - Fully tested and validated system

For more information, check the [technical documentation](docs/) or open an [issue](https://github.com/sulfierry/docktkinase/issues).
