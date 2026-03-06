# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**semantic-screening** is an open-source platform for predicting protein-ligand interactions using deep learning. It implements the **DT-Kinase** architecture (CNN + Cross-Attention) and classical ML pipelines. The scientific goal is "semantic docking" — predicting compound activity against kinase targets using PLM embeddings of primary sequences, without requiring 3D structures.

**Repository**: gmmsb-lncc/semantic-screening | **License**: MIT | **Python**: 3.9+ (env uses 3.12) | **PyTorch**: 2.0+

## Environment Setup

```bash
# Option A: Conda
conda env create -f environment.yml && conda activate docktkinase

# Option B: venv (used in this repo)
python setup.py              # Creates env/, installs deps, downloads models
source activate_env.sh       # Activates venv + sets PYTHONPATH

# ESM-2 must be loaded from local repo (pip versions cause segfaults)
git clone https://github.com/facebookresearch/esm.git llm/ESM
```

The virtual environment lives in `env/`. Activate with `source env/bin/activate`.

## Running Tests

```bash
# Pytest (configured in pyproject.toml)
pytest                                    # All tests
pytest -m unit                            # Unit tests only
pytest -m "not slow"                      # Skip slow tests
pytest tests/classifier_test/             # Classifier module tests
pytest tests/regression_test/             # Regression module tests
pytest tests/embeddings_modular_test/     # Embedding module tests
pytest tests/test_cross_attention_model.py  # Single test file

# Custom test runners (older, use subprocess)
python tests/run_all_tests.py
python tests/classifier_test/run_all_tests.py
```

Markers defined: `slow`, `integration`, `unit`, `regression`, `classifier`, `build`, `requires_gpu`, `requires_data`.

## Main Entry Points

| Command | Purpose |
|---------|---------|
| `python scripts/run_complete_pipeline.py --input data.tsv --output results/ --protein-model esm2_t33_650M_UR50D` | Full pipeline: embeddings → stratification → classification → regression |
| `python scripts/attention_matrix.py --attention-matrix on --input data.tsv --build results/` | Cross-Attention model training |
| `python semantic_screening_models.py --dataset non_human --embedding 8M --levels 1a 1b 1c 2 3 4` | Benchmark: 6-level model comparison |

## Architecture

### Two Parallel Pipelines

1. **Classical ML Pipeline** (`src/integrated_pipeline.py`): Generates mean-pooled vector embeddings → trains 10 models (XGBoost, LightGBM, CatBoost, RF, SVM, KNN, Ridge, Lasso, MLP, GB) for both classification and regression.

2. **DT-Kinase Deep Learning Pipeline** (`src/attention_matrix/`, `crossattention_split_analysis/`): Uses per-token matrix embeddings → CNN multi-scale encoders (kernels {3,5,7}) → bidirectional cross-attention → multi-task prediction (classification + regression jointly).

### Module Dependency Flow

```
Input TSV (seq_id, seq, chembl_id, smiles, pchembl_value)
    │
    ├─→ src/build/embeddings/strategies/     # ESM-2, ESM-C, SMI-TED, MoLFormer
    │       ├─→ protein_matrices/ [seq_len, protein_dim]   (per-residue)
    │       ├─→ ligand_matrices/  [mol_len, 768]           (SMI-TED per-token)
    │       ├─→ molformer_matrix/ [mol_len, 768]           (MoLFormer per-token)
    │       └─→ protein_embeddings/ / ligand_embeddings/   (mean-pooled vectors)
    │
    ├─→ src/classifier/ + src/regression/    # Classical ML (uses vectors)
    │
    └─→ src/attention_matrix/model.py        # DT-Kinase (uses matrices)
            CrossAttentionModel              # Basic: single cross-attn layer
            ImprovedCrossAttentionModel      # Deep: multi-layer + FFN + GELU
        src/classifier/models/cross_attention_model.py
            CrossAttentionAffinityModel      # Full DT-Kinase with CNN encoders
            MultiTaskLoss                    # Joint classification + regression loss
```

### ESM Loading (Critical)

`src/__init__.py` adds `llm/ESM/` to `sys.path` at import time and pre-imports ESM to lock in the local version. Many modules also do their own `sys.path.insert` for ESM. **Never install ESM via pip** (`fair-esm` or `esm` packages) — they conflict with the local version.

### Embedding Dimensions

| Model | Shorthand | Protein Dim |
|-------|-----------|-------------|
| `esm2_t6_8M_UR50D` | `8M` | 320 |
| `esm2_t30_150M_UR50D` | `150M` | 640 |
| `esm2_t33_650M_UR50D` | `650M` | 1280 |

Ligand dim is always 768 (both SMI-TED and MoLFormer).

## Data Layout

**Input datasets** (not in git, ~415 MB total):
- `tests/datasets/kinase_human_compounds.tsv`
- `tests/datasets/kinase_non_human_compounds.tsv`
- `tests/datasets/kinase_all_compounds.tsv`

**Pre-computed embeddings** are stored at:
`./results/protein_model_benchmark_{human|non_human}_v2/{embedding_name}/build/`
with subdirs: `protein_matrices/`, `ligand_matrices/`, `molformer_matrix/`, `attention_matrices/`.

**File naming**: `{seq_id}_matrix.npy`, `{chembl_id}_matrix.npy`, `{chembl_id}_molformer_matrix.npy`, `{seq_id}_attention.npy`.

## Split Analysis Module (`crossattention_split_analysis/`)

This is the most actively developed module. Key files:

- `config.py` — `TrainingConfig` dataclass, `SUPPORTED_EMBEDDINGS`, `DATASET_PATHS`, `AVAILABLE_SCENARIOS`, thresholds
- `experiment.py` — `run_single_analysis()` (CLI entry), `run_crossattention_analysis()` (orchestrator), `run_scenario()` (single train/eval)
- `data/splits.py` — Three split strategies: `split_random`, `split_by_compound`, `split_new_compound_new_kinase`
- `data/datasets.py` — `AttentionMatrixDataset`, `collate_attention_batch` (padding + masks)
- `training/trainer.py` — `train_model()` with AdamW + CosineAnnealingLR, early stopping on val MCC
- `training/evaluator.py` — `evaluate()` returns metrics: accuracy, f1, mcc, auc

**Three data split scenarios** (hardest → easiest):
1. `new_compound_new_kinase` — both compound AND kinase unseen in test (true generalization)
2. `compound` — compound unseen, kinase may overlap
3. `random` — random 80/10/10 split (baseline, allows data leakage)

**Affinity threshold**: pChEMBL >= 6.0 (IC50 <= 1000 nM) → active.

## Key Development Notes

- Default seeds for multi-seed experiments: `[42, 123, 456, 789, 1024]`
- Model selection (early stopping) uses **validation MCC**, not loss
- `MultiTaskLoss` weights: classification=1.0, regression=0.5
- Checkpoint system uses atomic writes (temp file + rename) to prevent corruption
- The `--use_attention` flag switches protein input from per-residue embeddings to attention matrices `[seq_len, seq_len]`
- The `--molformer_ligand` flag switches ligand input from SMI-TED to MoLFormer matrices
- Dataset `all` combines human + non_human by loading from both embedding directories

## Adding a New Protein Model

1. Add model name to choices in `scripts/run_complete_pipeline.py`
2. Add dimension mapping in `protein_dims` dict
3. Implement embedding strategy in `src/build/embeddings/strategies/`
