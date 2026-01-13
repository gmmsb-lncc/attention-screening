# semantic-screening 🧬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-gmmsb--lncc%2Fsemantic--screening-green.svg)](https://github.com/gmmsb-lncc/semantic-screening)

**Open-source platform for semantic screening of protein-ligand interactions using deep learning**

semantic-screening is an extensible platform combining state-of-the-art protein language models (ESM-2, ESM-C, Boltz-2) with molecular embeddings (SMI-TED) to predict compound activity against protein targets. It implements the **DT-Kinase** architecture—a hybrid CNN + Cross-Attention neural network for interaction modeling—alongside classical ML pipelines and rigorous stratification to prevent data leakage.

### 🔬 Scientific Context & Motivation

**Protein kinases** constitute approximately 2% of the human proteome (518 genes per Manning et al. taxonomy) but regulate an estimated 30% of all cellular proteins through reversible phosphorylation of serine, threonine, and tyrosine residues. This topological centrality in cell signaling networks establishes kinases as **control nodes** whose activation state determines fundamental cellular decisions: proliferation vs. quiescence, differentiation vs. pluripotency maintenance, survival vs. apoptosis.

Kinase dysregulation—through activating mutations, gene amplification, chromosomal fusions, or loss of negative regulators—constitutes a driver oncogenic event across a broad spectrum of human malignancies. As of 2024, 72 small-molecule kinase inhibitors have obtained regulatory approval, generating global revenues exceeding $80 billion annually.

**The Selectivity Paradox**: Despite clinical success, kinase pharmacology faces a fundamental paradox: how to achieve therapeutic selectivity across 518 enzymes sharing highly evolutionarily conserved catalytic architecture? The catalytic kinase domain presents extraordinary structural conservation, with RMSD < 2Å between ATP-binding sites of evolutionarily distant kinases. Systematic selectivity profiling reveals that clinically approved inhibitors typically modulate dozens of kinases beyond their primary target—resulting polypharmacology manifests as dose-limiting toxicities that narrow therapeutic windows.

**Traditional approaches fail simultaneous criteria**:
- **Molecular docking**: Scoring function error (±2-3 kcal/mol) exceeds the 1.4 kcal/mol difference required for 10-fold selectivity discrimination; ~40% of kinases lack experimental 3D structures
- **Experimental panels**: Cost ~$75-100 per kinase-compound pair ($30-40K for full kinome profiling per compound); coverage limited to ~60% of kinome
- **First-generation ML (DeepDTA)**: Late fusion via concatenation fails to explicitly model position-specific interactions; limited representations from raw sequences

**The semantic-screening hypothesis** proposes a fundamental paradigm shift: abandon dependence on 3D geometric representations and operate directly on **information space encoded in primary sequences, interpreted through foundation deep learning models**. This reformulation substitutes the geometric question "How well does this ligand fit this site?" with the semantic question "How compatible are the latent representations of this protein and ligand in shared vector space?"

**The DT-Kinase architecture** (implemented within semantic-screening) validates this hypothesis by:
1. **Contextual encoding**: ESM-2/ESM-C embeddings integrate global sequence context via self-attention—residues in the binding site incorporate information from regulatory domains, activation loops, and distal regions
2. **Explicit interaction modeling**: Cross-attention learns position-specific correspondences between protein residues and ligand atoms
3. **Universal coverage**: Applicable to any protein with known sequence, including the ~40% of kinases without experimental structures
4. **Scalability**: Throughput > 10⁶ predictions/hour enables ultralarge library (10⁹ compounds) screening against complete kinome

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

### Key Design Principles (from PhD Thesis Ch. 2)

- **Structural Independence**: System must operate exclusively from primary sequences, eliminating dependence on 3D coordinates and guaranteeing universal applicability to the entire kinome
- **Rich Semantic Representations**: Proteins and ligands encoded through pre-trained contextual embeddings capturing latent structural and functional information (ESM-2 for proteins, SMI-TED for ligands)
- **Explicit Interaction Modeling**: Cross-attention mechanism between protein and ligand representations enables learning which regions of each entity are relevant for specific affinity prediction
- **Computational Scalability**: Throughput > 10⁶ predictions/hour, enabling ultralarge chemical library screening against complete kinome; linear scaling with multi-GPU clusters
- **Multi-Task Framework**: Joint prediction of binary bioactivity and quantitative affinity through shared training objective improves generalization via learning transferable representations

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

If you use semantic-screening or the DT-Kinase architecture in your research, please cite:

```bibtex
@phdthesis{dtkinase2026,
  title = {DT-Kinase: Semantic Screening of Protein-Ligand Interactions via Cross-Attention over Protein Language Model Embeddings},
  author = {Leon Sulfierry},
  year = {2026},
  school = {Laboratório Nacional de Computação Científica (LNCC)},
  type = {PhD Thesis}
}

@software{semanticscreening2025,
  title = {semantic-screening: Open Platform for Semantic Protein-Ligand Interaction Screening},
  author = {GMMSB-LNCC Team},
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
