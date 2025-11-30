# DockTKinase 🧬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-gmmsb--lncc%2Fdocktkinase-green.svg)](https://github.com/gmmsb-lncc/docktkinase)

**Integrated Pipeline for Protein-Ligand Property Prediction using Deep Learning**

An end-to-end machine learning pipeline that generates protein and ligand embeddings, constructs interaction matrices, and trains classifiers/regressors for drug-target activity prediction.

## Overview

DockTKinase combines state-of-the-art protein language models with molecular embeddings to predict compound activity against kinase targets. The pipeline supports:

- **Multiple protein embedding models**: ESM-2 (6 sizes), ESM-C (3 variants), Boltz-2
- **Ligand embeddings**: FM4M (SMI-TED) - 768 dimensions
- **Cross-Attention modeling**: CNN + Cross-Attention for interaction learning
- **ML pipelines**: 12 classifiers + 12 regressors + neural networks
- **Adaptive stratification**: Intelligent clustering-based train/val/test splitting

## Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Multi-Model Protein Embeddings** | ESM-2 (8M-15B params), ESM-C, Boltz-2 (384-dim) |
| 🔬 **Ligand Embeddings** | FM4M SMI-TED (768-dim) |
| 🎯 **Cross-Attention Module** | CNN + Cross-Attention for protein-ligand interactions |
| 📊 **ML Classifiers** | XGBoost, LightGBM, CatBoost, Random Forest, SVM, etc. |
| 📈 **ML Regressors** | Gradient Boosting, Ridge, Lasso, Neural Networks |
| 🔀 **Adaptive Stratification** | Cluster-based splits maintaining data distribution |
| ⚡ **GPU Acceleration** | CUDA, MPS (Apple Silicon), or CPU |
| 💾 **Smart Caching** | Incremental embedding generation with caching |

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU)
- 8GB+ RAM (16GB+ recommended)

### Quick Install

```bash
# Clone repository
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Create conda environment
conda env create -f environment.yml
conda activate docktkinase

# Install dependencies
python scripts/post_install.py
```

### Platform-Specific Requirements

```bash
# macOS (Apple Silicon)
pip install -r requirements-mac.txt

# Linux with CUDA
pip install -r requirements-cuda.txt

# CPU only
pip install -r requirements.txt
```

## Quick Start

### Basic Pipeline

```bash
# Run complete pipeline (embeddings → stratification → classification → regression)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/my_run \
    --protein-model esm2_t33_650M_UR50D \
    --seed 42
```

### Cross-Attention Training

Train a CNN + Cross-Attention model on pre-computed embeddings:

```bash
# Enable cross-attention training
python attention_matrix.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --embedding-matrix concatenated_embeddings/embedding_matrix.npy \
    --output results/attention_run \
    --attention-matrix on \
    --epochs 50 \
    --batch-size 64 \
    --device mps

# Disable cross-attention (skip training)
python attention_matrix.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --attention-matrix off
```

### Input Data Format

TSV file with required columns:
```
compound_smiles    target_sequence    pchembl_value
CCO...             MKVLW...           7.5
```

- `compound_smiles`: Valid SMILES string
- `target_sequence`: Amino acid sequence
- `pchembl_value`: Activity value (pChEMBL scale)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_complete_pipeline.py                     │
│                       (Main CLI Entry Point)                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      IntegratedPipeline                          │
│                    (End-to-End Orchestration)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   BUILD PHASE   │ │  CLASSIFIER     │ │  REGRESSION     │
│                 │ │     PHASE       │ │     PHASE       │
│ • Protein Emb.  │ │                 │ │                 │
│ • Ligand Emb.   │ │ • 12 Models     │ │ • 12 Models     │
│ • Matrix Build  │ │ • MLP           │ │ • Neural Net    │
│ • Stratification│ │ • Cross-Val     │ │ • Metrics       │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ATTENTION MATRIX MODULE                        │
│                  (attention_matrix.py CLI)                       │
│                                                                  │
│  • CNN + Cross-Attention Model (~1.8M parameters)                │
│  • Pre-computed embedding matrix support                         │
│  • Classification (accuracy, ROC-AUC, F1, MCC)                   │
│  • Regression (R², Pearson, Spearman, RMSE)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Embedding Matrix Processing

The pipeline generates a concatenated embedding matrix combining protein and ligand representations:

### Matrix Structure
```
embedding_matrix.npy: (N, 1088)
├── Protein embeddings: columns [0:320]   (ESM-2 t6_8M: 320-dim)
└── Ligand embeddings:  columns [320:1088] (SMI-TED: 768-dim)
```

### Generation Process

```python
# 1. Generate protein embeddings
from src.build.embeddings.protein_embedding import ProteinEmbedding

embedder = ProteinEmbedding(model_name='esm2_t6_8M_UR50D')
protein_emb = embedder.generate_batch_embeddings(sequences)  # (N, 320)

# 2. Generate ligand embeddings
from src.build.embeddings.ligand_embedding import LigandEmbedding

ligand_embedder = LigandEmbedding(model_name='fm4m')
ligand_emb = ligand_embedder.generate_batch_embeddings(smiles)  # (N, 768)

# 3. Concatenate into matrix
import numpy as np
embedding_matrix = np.concatenate([protein_emb, ligand_emb], axis=1)  # (N, 1088)
np.save('concatenated_embeddings/embedding_matrix.npy', embedding_matrix)
```

### Cross-Attention Model Architecture

The `ImprovedCrossAttentionModel` processes the embedding matrix:

```
Input: (batch, 1088)
    │
    ▼
┌──────────────────┐
│ Reshape + Expand │ → (batch, 1, 1088)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CNN Layers      │
│  Conv1d(64→128)  │
│  BatchNorm + ReLU│
│  MaxPool1d       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Cross-Attention  │
│  MultiheadAttn   │
│  (8 heads)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Output Heads    │
├──────────────────┤
│ Classification:1 │ → Binary (active/inactive)
│ Regression: 1    │ → pChEMBL prediction
└──────────────────┘
```

## Supported Models

### Protein Embedding Models

| Model | Parameters | Embedding Dim | Memory | Speed |
|-------|------------|---------------|--------|-------|
| `esm2_t6_8M_UR50D` | 8M | 320 | ~1 GB | Fast |
| `esm2_t12_35M_UR50D` | 35M | 480 | ~2 GB | Fast |
| `esm2_t30_150M_UR50D` | 150M | 640 | ~4 GB | Medium |
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~6 GB | Medium |
| `esm2_t36_3B_UR50D` | 3B | 2560 | ~12 GB | Slow |
| `esm2_t48_15B_UR50D` | 15B | 5120 | ~48 GB | Very Slow |
| `esmc_300m` | 300M | 960 | ~3 GB | Fast |
| `esmc_600m` | 600M | 1152 | ~5 GB | Medium |
| `esmc_6b` | 6B | 4096 | ~24 GB | Slow |
| `boltz2` | ~400M | 384 | ~4 GB | Medium |

### Ligand Embedding Model

| Model | Embedding Dim | Description |
|-------|---------------|-------------|
| `fm4m` (SMI-TED) | 768 | Molecular foundation model from IBM Research |

## CLI Reference

### run_complete_pipeline.py

```bash
python run_complete_pipeline.py \
    --input INPUT_TSV \
    --output OUTPUT_DIR \
    --protein-model MODEL_NAME \
    --ligand-model fm4m \
    --batch-size 32 \
    --device cuda \
    --seed 42 \
    --stratifier-method auto \
    --stratifier-threshold 0.9 \
    --skip-classification \
    --skip-regression
```

### attention_matrix.py

```bash
python attention_matrix.py \
    --input INPUT_TSV \
    --embedding-matrix MATRIX_NPY \
    --output OUTPUT_DIR \
    --attention-matrix on|off \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --device cuda|mps|cpu \
    --seed 42
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input TSV file with compound data | Required |
| `--embedding-matrix` | Pre-computed embedding matrix (.npy) | Required |
| `--output` | Output directory for results | `results/attention_matrix` |
| `--attention-matrix` | Enable/disable training (`on`/`off`) | `on` |
| `--epochs` | Number of training epochs | `50` |
| `--batch-size` | Training batch size | `64` |
| `--learning-rate` | Optimizer learning rate | `0.0005` |
| `--device` | Compute device | Auto-detect |
| `--seed` | Random seed for reproducibility | `42` |

## Project Structure

```
docktkinase/
├── run_complete_pipeline.py    # Main pipeline CLI
├── attention_matrix.py         # Cross-Attention training CLI
│
├── src/
│   ├── integrated_pipeline.py  # Pipeline orchestration
│   │
│   ├── attention_matrix/       # Cross-Attention module
│   │   ├── config.py           # AttentionMatrixConfig
│   │   ├── model.py            # CrossAttentionModel, ImprovedCrossAttentionModel
│   │   ├── pipeline.py         # AttentionMatrixPipeline
│   │   ├── data_loader.py      # EmbeddingDataLoader
│   │   ├── dataset.py          # EmbeddingDataset
│   │   ├── trainer.py          # ModelTrainer
│   │   ├── evaluator.py        # ModelEvaluator
│   │   ├── metrics.py          # Classification/Regression metrics
│   │   ├── splitter.py         # DataSplitter
│   │   ├── attention_analyzer.py  # Attention weight analysis
│   │   └── __init__.py
│   │
│   ├── build/                  # Embedding generation & matrices
│   │   ├── pipeline/           # Build orchestration
│   │   ├── embeddings/         # Protein & ligand embeddings
│   │   │   ├── strategies/     # Model implementations
│   │   │   │   ├── esm2_strategy.py
│   │   │   │   ├── esmc_strategy.py
│   │   │   │   └── boltz_strategy.py
│   │   │   └── core/
│   │   ├── matrix/             # Matrix construction
│   │   └── stratification/     # Adaptive clustering
│   │
│   ├── classifier/             # Classification pipeline
│   │   ├── models/             # 12 sklearn + MLP
│   │   └── core/               # Training, evaluation
│   │
│   └── regression/             # Regression pipeline
│       ├── models/             # 12 sklearn + neural
│       └── core/
│
├── tests/                      # Integration tests
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── scripts/                    # Setup scripts
│
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata
└── LICENSE                     # MIT License
```

## Results & Performance

### Classification Metrics

| Model | Accuracy | ROC-AUC | F1-Score | MCC |
|-------|----------|---------|----------|-----|
| XGBoost | 0.85 | 0.91 | 0.88 | 0.68 |
| LightGBM | 0.84 | 0.90 | 0.87 | 0.66 |
| CatBoost | 0.84 | 0.90 | 0.87 | 0.66 |
| Random Forest | 0.82 | 0.88 | 0.85 | 0.62 |
| **Cross-Attention** | **0.80** | **0.80** | **0.86** | **0.51** |

### Regression Metrics

| Model | R² | Pearson | Spearman | RMSE |
|-------|----|---------|---------| -----|
| XGBoost | 0.52 | 0.74 | 0.69 | 0.89 |
| LightGBM | 0.51 | 0.73 | 0.68 | 0.91 |
| CatBoost | 0.50 | 0.72 | 0.67 | 0.92 |
| **Cross-Attention** | **0.47** | **0.70** | **0.54** | **0.98** |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_attention_matrix.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- **[Getting Started](docs/01-getting-started/)** - Installation and prerequisites
- **[User Guide](docs/02-user-guide/)** - Pipeline usage and workflows
- **[Architecture](docs/03-architecture/)** - System design and patterns
- **[Modules](docs/04-modules/)** - Component documentation
  - [Boltz-2 Strategy](docs/04-modules/BOLTZ_STRATEGY_GUIDE.md)
  - [Adaptive Clustering](docs/04-modules/ADAPTIVE_CLUSTERING_GUIDE.md)
- **[Development](docs/05-development/)** - Contributing guidelines
- **[Troubleshooting](docs/07-troubleshooting/)** - Common issues

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

### Foundation Models
- **[ESM-2](https://github.com/facebookresearch/esm)** - Meta AI
- **[ESM-C](https://github.com/evolutionaryscale/esm)** - EvolutionaryScale
- **[Boltz-2](https://github.com/jwohlwend/boltz)** - MIT AI Lab
- **[FM4M](https://github.com/IBM/materials)** - IBM Research

### Core Dependencies
- PyTorch, scikit-learn, NumPy, Transformers

## Citation

```bibtex
@software{docktkinase2025,
  title = {DockTKinase: Integrated Pipeline for Protein-Ligand Property Prediction},
  author = {DockTKinase Development Team},
  year = {2025},
  url = {https://github.com/gmmsb-lncc/docktkinase},
  version = {2.0}
}
```

## Contact

- **Repository**: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- **Issues**: [Bug reports & features](https://github.com/gmmsb-lncc/docktkinase/issues)
- **Discussions**: [Q&A](https://github.com/gmmsb-lncc/docktkinase/discussions)

---

**Status**: ✅ Production Ready | **Version**: 2.0 | **Last Updated**: November 2025
