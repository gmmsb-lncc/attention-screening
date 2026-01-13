# semantic-screening 🧬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-gmmsb--lncc%2Fsemantic--screening-green.svg)](https://github.com/gmmsb-lncc/semantic-screening)

**Open-source platform for semantic screening of protein-ligand interactions using deep learning**

semantic-screening is an extensible platform combining state-of-the-art protein language models (ESM-2, ESM-C, Boltz-2) with molecular embeddings (SMI-TED) to predict compound activity against protein targets. It implements the **DT-Kinase** architecture—a hybrid CNN + Cross-Attention neural network for interaction modeling—alongside classical ML pipelines and rigorous stratification to prevent data leakage.

### 🔬 Scientific Context & Motivation

**Kinases** comprise ~2% of the human proteome (518 genes) but regulate ~30% of all cellular proteins through phosphorylation. Dysregulation drives oncogenic transformation and other diseases. The central pharmacological challenge is achieving **selectivity across a highly conserved catalytic domain**—all 518 kinases share >85% structural similarity in their ATP-binding pocket, making it notoriously difficult to design selective inhibitors without off-target toxicity or emergent drug resistance mutations.

**Traditional approaches** fail the simultaneity test:
- **Molecular docking**: Accurate but computationally expensive, requires 3D structures (~40% of kinases lack experimental structures)
- **Experimental panels**: High-throughput but expensive and slow
- **Early ML methods**: Limited by one-hot encoding and shallow networks

**The semantic-screening hypothesis** proposes a paradigm shift: abandon geometric representations and operate directly on **primary sequence information interpreted through contextual embeddings from Protein Language Models (PLMs)**. This reformulation answers the selectivity question through semantic compatibility in latent space rather than geometric fit in 3D space.

**The DT-Kinase solution** (implemented within semantic-screening) validates this hypothesis empirically by:
1. Using ESM-2/ESM-C to encode evolutionary and structural information implicitly in sequence
2. Modeling interaction patterns through CNN + Cross-Attention mechanisms
3. Achieving universal applicability (any protein with known sequence, no structure required)
4. Demonstrating superior selectivity prediction compared to structure-based and first-generation ML approaches

---

## 📚 Documentation

**Start here to understand the concepts**:

- **[Concepts: semantic-screening vs DT-Kinase](docs/CONCEPTS.md)** ⭐ **START HERE** - Clarifies the distinction between the platform and the neural architecture

For detailed information:

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

## The DT-Kinase Architecture

**DT-Kinase** is the neural network architecture implemented within semantic-screening that solves protein-ligand interaction prediction through semantic embeddings. It combines:

1. **Protein Encoding**: Per-residue embeddings from Protein Language Models (ESM-2, ESM-C, Boltz-2) capture evolutionary constraints and implicit structural information
2. **Ligand Encoding**: Per-atom embeddings from chemical foundation models (FM4M SMI-TED) encode molecular properties and SMILES syntax
3. **Local Feature Extraction**: Multi-scale CNN encoders capture local patterns in protein sequences and molecular structures
4. **Semantic Interaction Modeling**: Bidirectional cross-attention mechanisms model protein-ligand compatibility by learning which residues interact with which atoms
5. **Multi-Task Prediction**: Simultaneous classification (active/inactive) and regression (affinity quantification) with uncertainty-weighted loss

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Contextual Embeddings                                               │
│  ├── Protein: ESM-2/ESM-C per-residue embeddings [seq_len, d_protein]      │
│  └── Ligand:  SMI-TED per-atom embeddings [mol_len, d_ligand]              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: MULTI-SCALE CNN ENCODERS (Local Feature Extraction)               │
│  • Conv1D with kernels {3, 5, 7} for multi-scale pattern recognition       │
│  • Residual connections preserve feature hierarchy                          │
│  • LayerNorm for training stability                                         │
│  └─→ Output: [seq_len/pool, hidden_dim]                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: BIDIRECTIONAL CROSS-ATTENTION (Semantic Interaction Modeling)     │
│  • Query: Protein residues; Key/Value: Ligand atoms → Protein perspectives  │
│  • Query: Ligand atoms; Key/Value: Protein residues → Ligand perspectives   │
│  • Multi-Head Attention (8 heads) for diverse interaction types             │
│  └─→ Output: Learned attention weights indicating residue-atom affinities   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: MULTI-TASK PREDICTION HEAD                                        │
│  • Classification: {Active, Inactive} (binary logits)                       │
│  • Regression: Affinity value in pChEMBL scale (continuous)                 │
│  • Joint optimization with task-weighted loss                               │
│  └─→ Outputs: Classification logits + Regression value + Uncertainty       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **Primacy of Sequence**: No 3D coordinates required—information is encoded in primary sequence via PLM embeddings
- **Contextuality**: Transformer self-attention captures long-range dependencies and global sequence context
- **Semantic Compatibility**: Answers "How compatible are these latent representations?" rather than "How well does this geometrically fit?"
- **Scalability**: Inference is pure neural network forward pass, enabling trillion-compound screening against entire proteome
- **Universality**: Applicable to any protein with known sequence, including those without experimental structures

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
git clone https://github.com/gmmsb-lncc/semantic-screening.git
cd semantic-screening

# Create conda environment
conda env create -f environment.yml
conda activate semantic-screening

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
@software{semanticscreening2025,
  title = {semantic-screening: Platform for semantic screening of protein-ligand interactions},
  author = {semantic-screening Development Team},
  year = {2025},
  url = {https://github.com/gmmsb-lncc/semantic-screening},
  version = {2.1}
}
```

## Contact

- **Repository**: [gmmsb-lncc/semantic-screening](https://github.com/gmmsb-lncc/semantic-screening)
- **Issues**: [Bug reports & features](https://github.com/gmmsb-lncc/semantic-screening/issues)

---

**Status**: ✅ Production Ready | **Version**: 2.1 | **Last Updated**: December 2025
