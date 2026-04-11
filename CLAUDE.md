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
*Note: Some baseline models (like ConPLex, DrugBAN) require their own isolated conda environments. See `setup_env.sh` inside their respective directories.*

## Running Tests

```bash
# Pytest (configured in pyproject.toml)
pytest                                    # All tests
pytest -m unit                            # Unit tests only
pytest -m "not slow"                      # Skip slow tests
pytest tests/classifier_test/             # Classifier module tests
pytest tests/test_cross_attention_model.py  # Single test file
```

Markers defined: `slow`, `integration`, `unit`, `regression`, `classifier`, `build`, `requires_gpu`, `requires_data`.

## Architecture: DT-Kinase

### Parallel Pipelines

1. **Classical ML Pipeline** (`src/integrated_pipeline.py`): Generates mean-pooled vector embeddings → trains 10 models (XGBoost, LightGBM, CatBoost, RF, SVM, etc) for both classification and regression.
2. **DT-Kinase Deep Learning Pipeline** (`src/attention_matrix/`, `crossattention_split_analysis/`): Uses per-token matrix embeddings → CNN multi-scale encoders (kernels {3,5,7}) → bidirectional cross-attention → multi-task prediction (classification + regression jointly).

### Module Dependency Flow

```
Input TSV (seq_id, seq, chembl_id, smiles, pchembl_value)
    │
    ├─→ src/build/embeddings/strategies/     # ESM-2, ESM-C, SMI-TED, MoLFormer
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

## Baseline Models Integration

To rigorously evaluate DT-Kinase, the repository integrates several State-of-the-Art (SOTA) baseline models. Each resides in its own root directory with isolated environments to avoid dependency conflicts:

1. **DrugBAN** (`DrugBAN/`): Deep Bilinear Attention Network.
2. **GraphBAN** (`GraphBAN/`): Graph-based Bilinear Attention Network.
3. **DeepDTAGen** (`DeepDTAGen/`): Generative DTI and Multi-task architecture.
4. **ConPLex** (`ConPLex/`): Contrastive PLM-based exploration (structure-free, co-embedding with distance metrics).

**Evaluation Workflow**:
- We use unified **scaffold splits** (train/val/test) for fair comparison.
- Baseline datasets are usually located in `DrugBAN/datasets/kinase/{dataset}/scaffold/`.
- Training and evaluation are run via dedicated shell scripts and python wrappers in each baseline directory (e.g., `ConPLex/run_conplex_kinase_benchmark.sh`, `DrugBAN/run_dtkinase_drugban.sh`).

## Data Layout

**Input datasets** (not in git, ~415 MB total):
- `tests/datasets/kinase_human_compounds.tsv`
- `tests/datasets/kinase_non_human_compounds.tsv`
- `tests/datasets/kinase_all_compounds.tsv`

**Kinase Scaffold Splits** (used for standard benchmarking):
- `DrugBAN/datasets/kinase/{non_human|human|all}/scaffold/{train|val|test}.csv`

**Pre-computed embeddings** are stored at:
`./results/protein_model_benchmark_{human|non_human}_v2/{embedding_name}/build/`
with subdirs: `protein_matrices/`, `ligand_matrices/`, `molformer_matrix/`, `attention_matrices/`.

## Split Analysis Module (`crossattention_split_analysis/`)

This is the most actively developed module. Key files:

- `config.py` — `TrainingConfig` dataclass, `SUPPORTED_EMBEDDINGS`, `DATASET_PATHS`, `AVAILABLE_SCENARIOS`, thresholds
- `experiment.py` — `run_single_analysis()` (CLI entry), `run_crossattention_analysis()` (orchestrator), `run_scenario()` (single train/eval)
- `data/splits.py` — Three split strategies: `split_random`, `split_by_compound`, `split_new_compound_new_kinase`

**Three data split scenarios** (hardest → easiest):
1. `new_compound_new_kinase` — both compound AND kinase unseen in test (true generalization)
2. `compound` — compound unseen, kinase may overlap
3. `random` — random 80/10/10 split (baseline, allows data leakage)

**Affinity threshold**: pChEMBL >= 6.0 (IC50 <= 1000 nM) → active.

## Key Development Notes

- Default seeds for multi-seed experiments: `[42, 123, 456, 789, 1024]`
- Model selection (early stopping) uses **validation MCC** or **validation AUPR/AUROC** depending on the pipeline, not loss.
- `MultiTaskLoss` weights: classification=1.0, regression=0.5
- ESM loading: `src/__init__.py` adds `llm/ESM/` to `sys.path`. **Never install ESM via pip**.
- The `--use_attention` flag switches protein input from per-residue embeddings to attention matrices `[seq_len, seq_len]`.
- The dataset `all` combines `human` + `non_human` by loading from both embedding directories.

## Adding a New Protein Model

1. Add model name to choices in `scripts/run_complete_pipeline.py`
2. Add dimension mapping in `protein_dims` dict
3. Implement embedding strategy in `src/build/embeddings/strategies/`
