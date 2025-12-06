# DockTKinase 🧬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-gmmsb--lncc%2Fdocktkinase-green.svg)](https://github.com/gmmsb-lncc/docktkinase)

**Integrated Pipeline for Protein-Ligand Property Prediction using Deep Learning**

DockTKinase combines state-of-the-art protein language models (ESM-2, ESM-C, Boltz-2) with molecular embeddings (SMI-TED) to predict compound activity against kinase targets. It features a hybrid CNN + Cross-Attention architecture for interaction modeling and a rigorous K-means++ stratification strategy to prevent data leakage.

---

## 📚 Documentation

For detailed information, please refer to the full documentation:

- **[Methodology & Theory](docs/methodology.md)** - Comprehensive scientific background.
- **[User Guide](docs/02-user-guide/)** - Detailed usage instructions and workflows.
- **[Architecture](docs/03-architecture/)** - System design patterns and diagrams.
- **[Modules](docs/04-modules/)** - Deep dive into specific components.
- **[Stratification Guide](docs/02-user-guide/stratification-methodology.md)** - Details on the K-means++ splitting strategy.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Multi-Model Protein Embeddings** | ESM-2 (8M-15B params), ESM-C, Boltz-2 (384-dim) |
| 🔬 **Ligand Embeddings** | FM4M SMI-TED (768-dim) |
| 🎯 **Cross-Attention Module** | CNN + Cross-Attention for protein-ligand interactions |
| 📊 **ML Classifiers** | XGBoost, LightGBM, CatBoost, Random Forest, SVM, etc. |
| 📈 **ML Regressors** | Gradient Boosting, Ridge, Lasso, Neural Networks |
| 🔀 **K-means++ Stratification** | Cluster-aware train/val/test splitting with cosine similarity validation |
| ⚡ **GPU Acceleration** | CUDA, MPS (Apple Silicon), or CPU |

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

## CNN + Cross-Attention Model

The deep learning component implements a **hybrid CNN-Transformer architecture** that leverages convolutional neural networks for local feature extraction and cross-attention mechanisms for modeling non-local protein-ligand interactions.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Per-token Embedding Matrices                                        │
│  ├── Protein: [batch, seq_prot, d_prot]  (ESM-2 per-residue embeddings)    │
│  └── Ligand:  [batch, seq_lig, d_lig]    (SMI-TED per-atom embeddings)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CNN ENCODERS (Local Feature Extraction)                           │
│  • Multi-Scale Conv1D (kernels 3, 5, 7)                                     │
│  • Residual Connections & LayerNorm                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: BIDIRECTIONAL CROSS-ATTENTION (Interaction Modeling)              │
│  • Protein → Ligand Attention                                               │
│  • Ligand → Protein Attention                                               │
│  • Multi-Head (8 heads) to capture different interaction types              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: MULTI-TASK PREDICTION                                             │
│  • Classification Head (Active/Inactive)                                    │
│  • Regression Head (pChEMBL Affinity)                                       │
│  • Uncertainty-weighted Multi-Task Loss                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Biological Interpretation:**
The attention matrix $A_{ij}$ represents how much protein residue $i$ "attends to" ligand atom $j$. High attention weights correlate with physical interactions (H-bonds, hydrophobic contacts) in the binding pocket, enabling "semantic docking" without explicit 3D coordinates.

## Project Structure

```
docktkinase/
├── run_complete_pipeline.py    # Main pipeline CLI
├── attention_matrix.py         # Cross-Attention training CLI
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── integrated_pipeline.py  # Pipeline orchestration
│   │
│   ├── attention_matrix/       # Cross-Attention Deep Learning Module
│   │   ├── model.py            # CNN + Cross-Attention Architecture
│   │   └── ...
│   │
│   ├── build/                  # Data Ingestion & Embedding Generation
│   │   ├── embeddings/         # ESM-2, Boltz-2, SMI-TED wrappers
│   │   ├── matrix/             # Matrix construction
│   │   └── stratification/     # K-means++ Stratification logic
│   │
│   ├── classifier/             # Classical ML Classification (XGBoost, etc.)
│   │
│   └── regression/             # Classical ML Regression
│
├── docs/                       # Comprehensive Documentation
├── examples/                   # Usage Examples
└── tests/                      # Unit and Integration Tests
```

## Quick Start

### Installation

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

### Running the Pipeline

```bash
# Run complete pipeline (embeddings → stratification → classification → regression)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/my_run \
    --protein-model esm2_t33_650M_UR50D \
    --seed 42
```

### Training Cross-Attention Model

```bash
python attention_matrix.py \
    --input data/kinase_compounds.tsv \
    --embedding-matrix concatenated_embeddings/embedding_matrix.npy \
    --output results/attention_run \
    --attention-matrix on
```

## CLI Reference

| Script | Description | Key Arguments |
|--------|-------------|---------------|
| `run_complete_pipeline.py` | End-to-end workflow | `--input`, `--output`, `--protein-model`, `--stratifier-method` |
| `attention_matrix.py` | Train DL model | `--input`, `--embedding-matrix`, `--epochs`, `--batch-size` |

See [User Guide](docs/02-user-guide/) for full parameter lists.

## Citation

```bibtex
@software{docktkinase2025,
  title = {DockTKinase: Integrated Pipeline for Protein-Ligand Property Prediction},
  author = {DockTKinase Development Team},
  year = {2025},
  url = {https://github.com/gmmsb-lncc/docktkinase},
  version = {2.1}
}
```

## Contact

- **Repository**: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- **Issues**: [Bug reports & features](https://github.com/gmmsb-lncc/docktkinase/issues)

---

**Status**: ✅ Production Ready | **Version**: 2.1 | **Last Updated**: December 2025
