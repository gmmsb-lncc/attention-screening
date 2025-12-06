# CLI Reference

**Last Updated**: December 6, 2025  
**Version**: 2.1

Reference for the Command Line Interface (CLI) tools, primarily `run_complete_pipeline.py`.

---

## `run_complete_pipeline.py`

The main entry point for executing the end-to-end pipeline.

### Usage
```bash
python run_complete_pipeline.py --input <FILE> [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input` | Path to the input TSV file containing SMILES and Sequences. |

### Model Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--protein-model` | `esm2_t6_8M_UR50D` | Selects the protein embedding model. Options: ESM-2 variants, ESM-C variants, `boltz2`. |
| `--ligand-model` | `SMI-TED` | Selects the ligand embedding model. |
| `--protein-dim` | Auto | Manually override protein dimension (not recommended). |
| `--api` | None | API Key for remote models (e.g., `esmc-6b`). |

### Pipeline Control

| Argument | Description |
|----------|-------------|
| `--no-classification` | Skip the binary classification phase. |
| `--no-regression` | Skip the quantitative regression phase. |
| `--classification-models` | Specify list of models to train (e.g., `RandomForest XGBoost`). |
| `--regression-models` | Specify list of models to train. |
| `--no-checkpoints` | Force recalculation of all steps, ignoring cached files. |

### Execution Environment

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `results/pipeline_output` | Directory to save all results. |
| `--device` | `auto` | Hardware acceleration: `auto`, `cpu`, `cuda`, `mps` (Apple Silicon). |
| `--seed` | `42` | Random seed for reproducibility. |
| `--batch-size` | `32` | Batch size for embedding generation. |

### Data Splitting

| Argument | Default | Description |
|----------|---------|-------------|
| `--test-size` | `0.1` | Proportion of data for testing (0.0-1.0). |
| `--val-size` | `0.1` | Proportion of data for validation (0.0-1.0). |
| `--stratifier-threshold` | Auto | Manual threshold for clustering stratification. |

---

## `scripts/post_install.py`

Helper script for environment setup.

### Usage
```bash
python scripts/post_install.py
```

**Actions:**
1. Installs ML dependencies (`xgboost`, `lightgbm`, `catboost`).
2. Installs `accelerate` for large model support.
3. Downloads FM4M model files (SMI-TED).
4. Downloads/Caches default ESM model.
