# CLAUDE.md - Project Context for Future Sessions

## Project Overview

**semantic-screening** is an open-source platform for semantic screening of protein-ligand interactions using deep learning. It implements the **DT-Kinase** architecture—a hybrid CNN + Cross-Attention neural network—alongside classical ML pipelines for predicting compound activity against protein targets.

**Repository**: [gmmsb-lncc/semantic-screening](https://github.com/gmmsb-lncc/semantic-screening)
**Author**: Leon Sulfierry (GMMSB-LNCC)
**Version**: 2.1
**License**: MIT

---

## Scientific Context

### The Problem
Kinases represent ~2% of the human proteome (518 genes) but regulate ~30% of all cellular proteins. The central pharmacological challenge is achieving **selectivity across a highly conserved ATP-binding pocket** (>85% structural similarity across all 518 kinases).

### The Solution
Instead of geometric fitting in 3D space (molecular docking), semantic-screening operates on **primary sequence information interpreted through contextual embeddings from Protein Language Models (PLMs)**. This enables "semantic docking" in latent space without requiring 3D structures.

---

## Core Architecture

### Main Entry Points

| Script | Purpose |
|--------|---------|
| `run_complete_pipeline.py` | End-to-end pipeline: embeddings → stratification → classification → regression |
| `scripts/attention_matrix.py` | Cross-Attention model training CLI |
| `crossattention_split_analysis_main.py` | Split analysis with different data leakage scenarios |

### Module Structure

```
semantic-screening/
├── src/
│   ├── integrated_pipeline.py      # Pipeline orchestration
│   ├── attention_matrix/           # CNN + Cross-Attention deep learning
│   │   ├── model.py                # Architecture definition
│   │   ├── pipeline.py             # Training pipeline
│   │   ├── data_loader.py          # Embedding data loading
│   │   └── trainer.py              # Training loop
│   ├── build/                      # Data ingestion & embedding generation
│   │   ├── embeddings/             # ESM-2, ESM-C, SMI-TED, MoLFormer wrappers
│   │   ├── stratification/         # K-means++ stratification logic
│   │   └── matrix/                 # Matrix construction
│   ├── classifier/                 # Classical ML classification (XGBoost, etc.)
│   └── regression/                 # Classical ML regression
├── crossattention_split_analysis/  # Split mode analysis module
│   ├── config.py                   # Configuration constants
│   ├── experiment.py               # Experiment runner
│   └── training/                   # Training utilities
└── tests/datasets/                 # Test datasets (TSV format)
```

---

## Supported Models

### Protein Embedding Models

#### ESM-2 (Meta AI)
Bidirectional transformer with MLM training on UniRef50.

| Model | Parameters | Embedding Dim |
|-------|------------|---------------|
| `esm2_t6_8M_UR50D` | 8M | 320 |
| `esm2_t12_35M_UR50D` | 35M | 480 |
| `esm2_t30_150M_UR50D` | 150M | 640 |
| `esm2_t33_650M_UR50D` | 650M | 1280 |
| `esm2_t36_3B_UR50D` | 3B | 2560 |
| `esm2_t48_15B_UR50D` | 15B | 5120 |

#### ESM-C / ESM-3 (EvolutionaryScale)
Causal transformer with Next Token Prediction.

| Model | Parameters | Embedding Dim | Notes |
|-------|------------|---------------|-------|
| `esmc-300m-2024-12` | 300M | 960 | Local |
| `esmc-600m-2024-12` | 600M | 1152 | Local |
| `esmc-6b-2024-12` | 6B | 3072 | **Requires ESM_API_KEY** |

### Ligand Embedding Models

| Model | Embedding Dim | Output Type | Description |
|-------|---------------|-------------|-------------|
| **SMI-TED** | 768 | Vector/Matrix | IBM Foundation Models for Molecules |
| **MoLFormer** | 768 | Matrix | Per-token embeddings for cross-attention |

---

## Key Concepts

### Data Split Strategies (Preventing Data Leakage)

The platform supports three split modes defined in `crossattention_split_analysis/config.py`:

| Split Mode | CLI Flag | Description | Difficulty |
|------------|----------|-------------|------------|
| **Random** | `random` | Random split (baseline with potential leakage) | Easiest |
| **Compound-Only** | `compound` | Compound unseen in test, kinase may overlap | Medium |
| **New Compound + New Kinase** | `new_compound_new_kinase` | Both compound AND kinase unseen in test | Hardest |

These splits are critical for evaluating model generalization:
- `random`: May overestimate performance due to data leakage
- `compound`: Tests generalization to new molecules
- `new_compound_new_kinase`: Tests true generalization to novel protein-ligand pairs

### Affinity Threshold

Default threshold: `pChEMBL >= 6.0` (equivalent to IC50 <= 1000 nM)
- Active: pChEMBL >= 6.0
- Inactive: pChEMBL < 6.0

### Embedding Modes

- **Vector mode**: Mean-pooled embeddings (classical ML)
- **Matrix mode**: Per-residue/per-token embeddings (Cross-Attention)

---

## CLI Usage Examples

### Complete Pipeline
```bash
# Basic run with ESM-2 650M
python run_complete_pipeline.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --output results/my_experiment \
    --protein-model esm2_t33_650M_UR50D \
    --seed 42

# With ESM-C 6B (requires API key)
ESM_API_KEY="your_key" python run_complete_pipeline.py \
    --input data.tsv \
    --output results/esmc_6b_test \
    --protein-model esmc-6b-2024-12

# Skip classification, only regression
python run_complete_pipeline.py \
    --input data.tsv \
    --no-classification
```

### Cross-Attention Model
```bash
# Train with pre-computed matrices
python scripts/attention_matrix.py \
    --attention-matrix on \
    --input data.tsv \
    --build results/build_output \
    --mode matrix

# Generate matrices and train
python scripts/attention_matrix.py \
    --attention-matrix on \
    --input data.tsv \
    --build results/ \
    --generate-matrices
```

### Split Analysis
```bash
# Single embedding analysis
python crossattention_split_analysis_main.py \
    --embedding 150M \
    --dataset non_human \
    --scenarios random,compound,new_compound_new_kinase

# All embeddings (8M, 150M, 650M)
python crossattention_split_analysis_main.py \
    --run_all \
    --dataset non_human

# Custom training parameters
python crossattention_split_analysis_main.py \
    --embedding 150M \
    --dataset non_human \
    --epochs 300 \
    --patience 20 \
    --batch_size 64 \
    --seeds 42 123 456 789 1024
```

---

## Important Files

| File | Description |
|------|-------------|
| `src/integrated_pipeline.py` | Main pipeline orchestrator |
| `src/attention_matrix/model.py` | CNN + Cross-Attention architecture |
| `src/attention_matrix/pipeline.py` | Cross-Attention training pipeline |
| `crossattention_split_analysis/config.py` | Split modes and training configuration |
| `crossattention_split_analysis/experiment.py` | Experiment runner for split analysis |
| `docs/methodology.md` | Comprehensive scientific documentation |

---

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate semantic-screening

# Install dependencies
python scripts/post_install.py

# For ESM-2, local repository is required at llm/ESM/
git clone https://github.com/facebookresearch/esm.git llm/ESM
```

### Key Dependencies
- PyTorch 2.0+
- Python 3.9+
- esm (local from llm/ESM)
- transformers
- scikit-learn
- XGBoost, LightGBM, CatBoost

---

## Dataset Format

Input TSV files must contain:
- `seq_id`: Protein identifier
- `seq`: Protein sequence (amino acids)
- `molecule_chembl_id` or `chembl_id`: Ligand identifier
- `canonical_smiles` or `smiles`: SMILES string
- `pchembl_value`: Binding affinity (pChEMBL scale)
- `standard_value`: Activity value in nM (optional, used for regression)
- `target_kinase`: Target protein name (optional)

**Note**: The test datasets (`kinase_all_compounds.tsv`, `kinase_human_compounds.tsv`, `kinase_non_human_compounds.tsv`) are NOT in the Git repository due to their size (~415 MB total). They must be downloaded separately or generated via scripts in `src/database/`.

---

## Output Structure

```
results/
├── build/
│   ├── protein_embeddings/        # Per-protein .npy files
│   ├── ligand_embeddings/         # Per-ligand .npy files
│   ├── protein_matrices/          # Per-residue matrices (for cross-attention)
│   ├── ligand_matrices/           # Per-token matrices (for cross-attention)
│   └── concatenated_embeddings/   # Combined embedding matrix
├── classifier/                    # Classification results
├── regression/                    # Regression results
└── attention_matrix/              # Cross-attention model outputs
```

---

## Architecture Details

### DT-Kinase (CNN + Cross-Attention)

```
INPUT: Protein embeddings [seq_len, d_protein] + Ligand embeddings [mol_len, d_ligand]
    │
    ▼
STAGE 1: Multi-Scale CNN Encoders
    - Conv1D kernels {3, 5, 7}
    - Residual connections
    - LayerNorm
    │
    ▼
STAGE 2: Bidirectional Cross-Attention
    - Protein → Ligand attention
    - Ligand → Protein attention
    - Multi-Head (8 heads)
    │
    ▼
STAGE 3: Multi-Task Prediction
    - Classification: Active/Inactive (binary)
    - Regression: pChEMBL value (continuous)
```

### Classical ML Pipeline

10 models for both classification and regression:
- Random Forest, Gradient Boosting
- XGBoost, LightGBM, CatBoost
- SVM, KNN, Ridge, Lasso
- Multi-Layer Perceptron

---

## Common Tasks

### Adding a New Protein Model
1. Add model name to choices in `run_complete_pipeline.py`
2. Add dimension mapping in `protein_dims` dict
3. Implement strategy in `src/build/embeddings/strategies/`

### Running with Pre-computed Embeddings
```bash
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-embeddings-dir /path/to/protein_embeddings \
    --ligand-embeddings-dir /path/to/ligand_embeddings
```

### Debugging
- Use `--debug` flag in `crossattention_split_analysis_main.py`
- Check `src/__init__.py` for ESM loading issues
- Verify local ESM at `llm/ESM/`

---

## Notes for Development

- ESM must be loaded from local repository (`llm/ESM/`) to avoid segfaults
- Checkpoint system enabled by default (disable with `--no-checkpoints`)
- Multi-seed experiments recommended (5 seeds) for statistical rigor
- Default seeds: `[42, 123, 456, 789, 1024]`

---

## Related Documentation

- `docs/methodology.md` - Scientific background and theory
- `docs/01-methodology/` - Methodology review
- `docs/02-user-guide/` - Detailed usage instructions
- `docs/03-architecture/` - System design
- `docs/04-modules/` - Module-specific documentation
- `ablation/` - Ablation study scripts
- `encoder_compare/` - Encoder comparison experiments
