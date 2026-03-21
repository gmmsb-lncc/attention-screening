# Encoder Architecture Comparison Framework

A modular framework for comparing different encoder architectures in protein-ligand affinity prediction tasks.

## 📁 Project Structure

```
encoder_compare/
├── __init__.py
├── config.py                    # Configuration and constants
├── experiment.py                # Main experiment logic
├── models/
│   ├── __init__.py
│   └── flexible_model.py        # FlexibleCrossAttentionModel
├── data/
│   ├── __init__.py
│   └── splits.py                # Data splitting strategies
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training utilities
│   └── evaluator.py             # Evaluation metrics
├── utils/
│   ├── __init__.py
│   ├── checkpoints.py           # Checkpoint management
│   └── device.py                # Device utilities
└── visualization/
    ├── __init__.py
    ├── plots.py                 # Plotting functions
    └── summary.py               # Result summaries
```

## 🎯 Features

### Encoder Types
- **Linear**: Simple linear projection baseline
- **CNN**: Convolutional encoder with local feature extraction
- **CNN+Attention**: Hybrid architecture with CNN + self-attention

### Evaluation Scenarios

Ordered from hardest to easiest:

1. **New Compound + New Kinase** (True Generalization)
   - Both compounds and kinases in test are completely unseen
   - Hardest scenario, evaluates real-world generalization
   - May use ~60-70% of data due to strict splitting

2. **Split by Compound** (No Compound Leakage)
   - Compounds in test are unseen
   - Kinases may overlap between train/test
   - Uses 100% of data

3. **Random Split** (With Leakage)
   - Baseline scenario
   - Both compounds and kinases may appear in train/test
   - Uses 100% of data

### Primary Metric: MCC (Matthews Correlation Coefficient)

- More robust than accuracy for imbalanced datasets
- Ranges from -1 (total disagreement) to +1 (perfect)
- 0 indicates random prediction
- Takes all confusion matrix elements into account

## 🚀 Usage

### Basic Usage

```bash
# Single seed (fastest)
python compare_encoder_architectures.py --embedding 150M --dataset non_human

# Multiple seeds for robust statistics
python compare_encoder_architectures.py --embedding 150M --dataset non_human --seeds 42 123 456

# All embeddings
python compare_encoder_architectures.py --run_all --dataset non_human
```

### Advanced Options

```bash
# Custom number of epochs
python compare_encoder_architectures.py --embedding 150M --dataset non_human --epochs 100

# Start fresh (ignore checkpoints)
python compare_encoder_architectures.py --embedding 150M --dataset non_human --no-resume

# Custom output directory
python compare_encoder_architectures.py --embedding 150M --dataset non_human --output_dir ./my_results
```

## 🔄 Checkpoints & Resuming

The framework automatically saves checkpoints after each seed completes, allowing you to:
- Resume from interruptions (Ctrl+C, crashes, timeouts)
- Skip already completed experiments
- Continue long-running experiments across sessions

**Checkpoint Location**: `<output_dir>/checkpoints/<dataset>_<embedding>_checkpoint.pt`

**Resume Behavior (Default)**:
- Automatically loads existing checkpoint
- Skips completed experiments
- Continues from last incomplete seed

**Fresh Start**:
```bash
python compare_encoder_architectures.py --no-resume
```

## 📊 Output Files

### Results JSON
```
<output_dir>/<dataset>_<embedding>_encoder_comparison.json
```

Contains:
- Metadata (timestamp, embedding, dataset)
- Results for each encoder, scenario, and seed
- Mean and standard deviation across seeds

### Comparison Plot
```
<output_dir>/<dataset>_<embedding>_encoder_comparison.png
```

Two subplots:
1. MCC comparison across scenarios (with error bars)
2. MCC degradation from easy to hard scenario

## 🏗️ Architecture & Design Principles

### SOLID Principles

#### Single Responsibility (S)
- Each module has one clear purpose
- `trainer.py`: Training logic only
- `evaluator.py`: Evaluation metrics only
- `plots.py`: Visualization only

#### Open/Closed (O)
- Easy to add new encoder types via `create_encoder()`
- New split strategies added to `splits.py`
- Extensible configuration in `config.py`

#### Liskov Substitution (L)
- All split functions follow same interface
- Consistent return types across modules

#### Interface Segregation (I)
- Small, focused interfaces
- No forcing of unused dependencies

#### Dependency Inversion (D)
- High-level modules (experiment) depend on abstractions
- Low-level modules (models, data) are independent

### KISS (Keep It Simple, Stupid)

- Clear function names
- Short, focused functions
- Minimal nesting
- Explicit over implicit

### Clean Code

- Descriptive variable names
- Docstrings for all public functions
- Type hints for clarity
- No magic numbers (use config)

## 🧪 Adding New Features

### New Encoder Type

1. Add encoder to `src/classifier/models/encoder_variants.py`
2. Update `ENCODER_TYPES` in `config.py`
3. Framework automatically picks it up!

### New Split Strategy

```python
# In encoder_compare/data/splits.py

def split_custom(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Your custom split logic."""
    # ... implementation
    return train_idx, val_idx, test_idx

# In get_scenarios()
def get_scenarios() -> list:
    return [
        ('New Comp + New Kinase', split_new_compound_new_kinase),
        ('Split by Compound', split_by_compound),
        ('Random Split', split_random),
        ('Custom Split', split_custom)  # Add here
    ]
```

### New Metric

```python
# In encoder_compare/training/evaluator.py

def evaluate(model, loader, device):
    # ... existing code
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        # ... existing metrics
        'your_metric': calculate_your_metric(all_labels, all_preds)  # Add here
    }
```

## 📈 Progress Monitoring

During training, you'll see progress every 10 epochs:

```
Epoch  10/200: loss=0.4234, val_mcc=0.5876, val_acc=0.7654, val_auc=0.8234
Epoch  20/200: loss=0.3876, val_mcc=0.6123, val_acc=0.7765, val_auc=0.8456
Epoch  30/200: loss=0.3567, val_mcc=0.6543, val_acc=0.7876, val_auc=0.8654 🏆 NEW BEST!
```

After training:
```
✓ Best model from epoch 130 (val_mcc=0.6987)
Test: MCC=0.6987, AUC=0.8876, F1=0.7234, Acc=0.8234
💾 Checkpoint saved
```

## 🔧 Dependencies

- PyTorch >= 1.9
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Custom modules from `src/classifier/`

## 📝 Example Output

```
SUMMARY - MCC (PRIMARY METRIC) - mean ± std across seeds
====================================================================================================
Encoder            New Comp + New Kinase       Split by Compound           Random Split            MCC Drop%
----------------------------------------------------------------------------------------------------
LINEAR             0.6234±0.0123              0.7123±0.0098              0.7876±0.0087              20.8±1.9%
CNN                0.6543±0.0145              0.7456±0.0112              0.8123±0.0095              19.4±2.1%
CNN_ATTENTION      0.6789±0.0134              0.7678±0.0108              0.8234±0.0089              17.5±1.8%
----------------------------------------------------------------------------------------------------
BEST ENCODERS:
  🏆 Best on True Generalization (New Comp + New Kinase): CNN_ATTENTION
      MCC = 0.6789 ± 0.0134
  📉 Smallest MCC Drop (best robustness): CNN_ATTENTION
```

## 🤝 Contributing

When adding new features:
1. Follow the existing module structure
2. Add docstrings to all functions
3. Use type hints
4. Keep functions small and focused
5. Update this README

## 📄 License

Part of the DockTKinase project.
