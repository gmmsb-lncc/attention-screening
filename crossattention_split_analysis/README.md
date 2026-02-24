# CrossAttention Split Analysis

This module evaluates the CNN + Cross-Attention model for protein-ligand affinity prediction using three rigorous evaluation scenarios that progressively test generalization capabilities.

## Model Guides

- [CrossAttention Lite: Detailed Visual Guide](./CROSS_ATTENTION_LITE.md)

## Overview

The CrossAttention Split Analysis module implements a scientifically rigorous evaluation framework that assesses model generalization across three increasingly challenging scenarios:

1. **Random Split** - Traditional train/validation/test split allowing data leakage (upper bound)
2. **Split by Compound** - No compound overlap between splits (compound generalization)
3. **New Compound + New Kinase** - No compound or kinase overlap (true generalization)

## Scientific Rationale

### Evaluation Scenarios

Each scenario addresses a specific aspect of model generalization:

- **Random Split**: Establishes the performance upper bound when data leakage is allowed. This represents the optimistic scenario often seen in literature but unrealistic in real-world deployment.

- **Split by Compound**: Tests the model's ability to predict affinities for novel compounds against known kinases. This simulates the common drug discovery scenario of screening new compounds against established targets.

- **New Compound + New Kinase**: Provides the most realistic assessment of true generalization capability, simulating the challenge of predicting interactions between entirely novel compounds and kinases.

### Methodology

The module uses per-token embeddings from ESM-2 protein language models combined with SMI-TED ligand embeddings, processed through a dual-encoder architecture with cross-attention mechanisms.

## Architecture

```
crossattention_split_analysis/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration classes and constants
├── experiment.py               # Main experiment runner
├── data/                       # Data loading and splitting
│   ├── __init__.py
│   ├── datasets.py             # Dataset classes
│   └── splits.py               # Splitting strategies
├── training/                   # Training and evaluation
│   ├── __init__.py
│   ├── trainer.py              # Training loop implementation
│   └── evaluator.py            # Evaluation utilities
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── checkpoints.py          # Checkpoint management
│   └── device.py               # Device detection and reproducibility
└── visualization/              # Result plotting and reporting
    ├── __init__.py
    ├── plots.py                # Visualization functions
    └── summary.py              # Result aggregation and summary
```

## Key Features

### Reproducibility
- Deterministic training with configurable random seeds
- Comprehensive environment logging
- Checkpoint-based resumption of experiments

### Robustness
- Explicit handling of numerical instabilities (NaN/Inf values)
- Gradient clipping for stable training
- Early stopping based on validation MCC

### Scientific Rigor
- Multiple random seeds for statistical significance
- Confidence intervals for reported metrics
- Detailed environment and configuration logging

## Usage

### Command Line Interface

```bash
# Basic usage with 150M embeddings on non-human dataset
python crossattention_split_analysis.py --embedding 150M --dataset non_human

# Using attention matrices instead of embeddings
python crossattention_split_analysis.py --embedding 150M --dataset non_human --use-attention

# Run all supported embeddings
python crossattention_split_analysis.py --run_all --dataset non_human

# Force recalculation even if results exist
python crossattention_split_analysis.py --embedding 150M --dataset non_human --force
```

### Programmatic Usage

```python
from crossattention_split_analysis import run_crossattention_analysis, TrainingConfig

# Configure experiment
config = TrainingConfig(
    protein_dim=640,  # For 150M ESM-2
    num_epochs=500,
    batch_size=32
)

# Run analysis
results, stats = run_crossattention_analysis(
    embedding_name='150M',
    dataset_type='non_human',
    output_dir='./results',
    config=config,
    seeds=[42, 123, 456]  # Multiple seeds for statistics
)
```

## Configuration

### Training Configuration

The `TrainingConfig` dataclass allows fine-grained control over model architecture and training hyperparameters:

- `protein_dim`: Dimension of protein embeddings (320, 640, or 1280)
- `ligand_dim`: Dimension of ligand embeddings (fixed at 768 for SMI-TED)
- `hidden_dim`: Hidden dimension for all layers
- `num_cnn_layers`: Number of CNN encoder layers
- `num_cross_attn_layers`: Number of cross-attention layers
- `num_heads`: Number of attention heads
- `ff_dim`: Feed-forward layer dimension
- `dropout`: Dropout rate
- Training parameters (batch size, learning rate, epochs, etc.)

### Affinity Threshold

The default threshold uses pChEMBL >= 6.0 (logarithmic scale), which corresponds to 1000 nM (1 μM) in the original scale. This follows standard practice in kinase drug discovery for distinguishing active from inactive compounds. Using logarithmic scale (pChEMBL values) provides better numerical stability and is the standard representation in medicinal chemistry.

## Output Format

Results are saved in JSON format with:
- Environment information (Python, PyTorch, CUDA versions)
- Git commit hash and branch information
- Training configuration
- Per-scenario metrics with means and standard deviations across seeds
- Split statistics (sample counts, unique compounds/kinases)

Plots are generated showing accuracy and MCC comparisons across scenarios.

## Reproducibility Information

For reproducible results, the module:
- Sets all random seeds (Python, NumPy, PyTorch)
- Uses deterministic algorithms when possible
- Logs complete environment information
- Supports checkpoint-based resumption

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy
- TQDM

## References

The evaluation methodology follows best practices established in computational biology literature for assessing model generalization in molecular property prediction tasks.
