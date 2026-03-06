# Benchmark Package

Modular framework for comparing protein–ligand interaction prediction models across four representation levels, using scaffold-based data splits for rigorous evaluation.

This package is a SOLID-compliant rewrite of the monolithic `semantic_screening_models_beta.py` (2 837 lines) into 17 focused modules totalling ~2 900 lines.

## Quick Start

```bash
# Activate the project virtual environment
source env/bin/activate

# Run all four levels on the human dataset with default seeds
python semantic_screening_models.py --dataset human --embedding 8M --levels 1 2 3 4

# Run only levels 1 and 2 with custom seeds
python semantic_screening_models.py --dataset human --embedding 650M --levels 1 2 --seeds 42 123

# Include ESM-2 / MolFormer fine-tuning before level runs
python semantic_screening_models.py --dataset human --embedding 8M --levels 1 2 3 --finetune
```

## The Four Levels

Each level represents a progressively richer molecular representation:

| Level | Input Representation | Model | Module |
|-------|---------------------|-------|--------|
| **1** | Morgan fingerprints (no PLM) | KNN + MLP | `levels/level1.py` |
| **2** | Mean-pooled ESM-2 protein + attention-pooled MoLFormer ligand **vectors** | KNN + MLP | `levels/level2.py` |
| **3** | Per-residue ESM-2 protein + per-token MoLFormer ligand **matrices** → mean pooling | KNN + MLP | `levels/level3.py` |
| **4** | Per-residue/per-token matrices → CNN encoders + bidirectional cross-attention (DT-Kinase) | KNN + MLP | `levels/level4.py` |

All levels use **scaffold splits** (structurally dissimilar compounds in test set) and run across **5 seeds** by default (`[42, 123, 456, 789, 1024]`) for robust mean ± std reporting.

## Architecture

```
semantic_screening_models.py          ← Thin CLI entry point (33 lines)
│
└── benchmark/
    ├── __init__.py                   ← Public API exports
    ├── config.py                     ← BenchmarkConfig (frozen dataclass), constants
    ├── cli.py                        ← argparse → BenchmarkConfig
    ├── orchestrator.py               ← Facade: coordinates the full pipeline
    ├── progress.py                   ← tqdm-based step tracker
    ├── splits.py                     ← Scaffold split verification / generation
    ├── embeddings.py                 ← Attention pooling for ligand vectors
    ├── finetuning.py                 ← ESM-2 + MolFormer fine-tuning
    ├── metrics.py                    ← Multi-level metric aggregation
    ├── reporting.py                  ← Terminal table + JSON export
    ├── visualization.py              ← 5 plot types (bar, radar, heatmap, ranking, strip)
    └── levels/
        ├── __init__.py               ← Re-exports all runners
        ├── base.py                   ← BaseLevelRunner ABC (Template Method)
        ├── level1.py                 ← Fingerprint baseline
        ├── level2.py                 ← Embedding vectors
        ├── level3.py                 ← Matrix mean-pooling
        └── level4.py                 ← Cross-attention (DT-Kinase)
```

## Design Principles

### SOLID Compliance

- **Single Responsibility (SRP):** Each module owns exactly one concern — config, CLI, splits, a single level, metrics, plotting, etc.
- **Open/Closed (OCP):** Adding a new level (e.g. Level 5) requires only a new file in `levels/` inheriting from `BaseLevelRunner`. The orchestrator discovers runners from the config without modification.
- **Liskov Substitution (LSP):** All level runners implement the `BaseLevelRunner` interface and are interchangeable.
- **Interface Segregation (ISP):** Modules expose narrow, focused APIs — `ensure_scaffold_splits()`, `ensure_ligand_vectors()`, `generate_all()`, etc.
- **Dependency Inversion (DIP):** The orchestrator depends on the `BaseLevelRunner` abstraction, not concrete level implementations.

### Patterns Used

- **Template Method** — `BaseLevelRunner.run()` defines the multi-seed loop; subclasses implement only `run_single_seed()`.
- **Facade** — `BenchmarkOrchestrator.run()` hides all pipeline complexity behind a single call.
- **Frozen Dataclass** — `BenchmarkConfig` is immutable after construction, preventing accidental side effects.

## Module Reference

### `config.py`
Defines `BenchmarkConfig` (frozen dataclass) and all shared constants:
- `SUPPORTED_EMBEDDINGS` — Maps shorthands (`8M`, `150M`, `650M`) to full ESM-2 model names.
- `EMBEDDING_BASE_PATH` — Template for embedding directories.
- `METRICS_ORDER` — Canonical metric ordering: accuracy, mcc, f1, precision, recall, auc.
- `LEVEL_LABELS`, `LEVEL_COLORS` — Display metadata for tables and plots.
- `PCHEMBL_ACTIVITY_THRESHOLD` — 6.0 (IC₅₀ ≤ 1000 nM → active).

### `cli.py`
- `build_parser()` — Returns a fully configured `ArgumentParser`.
- `parse_levels()` — Accepts both `--levels 1 2 3` and `--levels 1,2,3` formats.
- `config_from_args()` — Converts `argparse.Namespace` into a validated `BenchmarkConfig`.

### `orchestrator.py`
Coordinates the pipeline in order:
1. Verify/generate scaffold splits
2. Extract ligand vectors (if Level 2 requested)
3. Fine-tuning (if `--finetune` flag)
4. Run requested levels (each via its `BaseLevelRunner`)
5. Aggregate metrics, print table, save JSON
6. Generate visualizations

### `levels/base.py`
Abstract base class with:
- `level_tag` (abstract property) — e.g. `"level1_fingerprint"`.
- `run_single_seed()` (abstract) — Trains and evaluates for one seed.
- `run()` — Template method: loops over seeds, accumulates results, aggregates mean ± std.
- `_load_cached_results()` — Avoids redundant computation when `--force` is not set.

### `levels/level1.py` – `level4.py`
Concrete runners implementing `run_single_seed()`. Each delegates to the appropriate downstream engine:
- **Level 1/2**: `split_comparison_analysis.run_single_dataset()` with `feature_type="fingerprint"` or `"embedding"`.
- **Level 3**: Custom `_MatrixDataset` + sklearn KNN/MLP with mean-pooled matrices.
- **Level 4**: `crossattention_split_analysis.experiment.run_single_analysis()` for the full DT-Kinase model.

### `finetuning.py`
Optional step that:
1. Fine-tunes ESM-2 on kinase classification data.
2. Fine-tunes MolFormer on kinase-relevant SMILES.
3. Regenerates protein and ligand embeddings using fine-tuned weights.

### `metrics.py`
`aggregate_benchmark_metrics()` — Collects per-level dicts and unifies them into a flat `{model_key: {metric: value}}` suitable for reporting.

### `reporting.py`
- `print_comparison_table()` — Renders an aligned ASCII table to stdout.
- `save_benchmark_json()` — Writes structured JSON with metadata + results.

### `visualization.py`
Five plot functions plus `generate_all()`:
- `plot_grouped_bar()` — Side-by-side bar chart of all metrics per model.
- `plot_radar()` — Radar/spider chart for multi-metric comparison.
- `plot_heatmap()` — Color-coded matrix of models × metrics.
- `plot_mcc_ranking()` — Horizontal bar chart sorted by MCC.
- `plot_strip()` — Strip plot showing per-seed variance.

### `splits.py`
`ensure_scaffold_splits()` — Checks for pre-existing scaffold split TSVs; falls back to running `scaffold_split.py` if missing.

### `embeddings.py`
- `AttentionPooling` — A learned attention-based pooling layer (`nn.Module`).
- `ensure_ligand_vectors()` — Extracts fixed-size vectors from per-token MoLFormer matrices.

### `progress.py`
`BenchmarkProgress` — Wraps `tqdm` for step-level tracking with timing summaries.

## Adding a New Level

1. Create `benchmark/levels/level5.py`:
   ```python
   from benchmark.levels.base import BaseLevelRunner

   class Level5Runner(BaseLevelRunner):
       @property
       def level_tag(self) -> str:
           return "level5_my_method"

       def run_single_seed(self, seed, output_dir, **kwargs):
           # Your training/evaluation logic here
           return {"Split by Scaffold": {"KNN": {...}, "MLP": {...}}}
   ```

2. Register it in `benchmark/levels/__init__.py`:
   ```python
   from benchmark.levels.level5 import Level5Runner
   ```

3. Add level `5` to `VALID_LEVELS` in `config.py` and add label/color entries.

4. Add a `if 5 in config.levels:` block in `orchestrator.py:_build_runners()`.

No other files need modification.

## CLI Reference

```
python semantic_screening_models.py \
    --dataset {human,non_human,all}       # Required
    --embedding {8M,150M,650M}            # Default: 8M
    --levels 1 2 3 4                      # Default: 1,2,3,4
    --seeds 42 123 456                    # Default: [42, 123, 456, 789, 1024]
    --output_dir ./my_results             # Default: ./results/benchmark_{dataset}_{embedding}
    --scaffold_split_dir path/to/splits   # Default: scaffolds_splits/output
    --epochs 500                          # Max DL epochs (default: 500)
    --batch_size 32                       # DL batch size (default: 32)
    --patience 5                          # Early stopping (default: 5, 0=disable)
    --learning_rate 1e-4                  # DL learning rate (default: 1e-4)
    --force                               # Recompute all levels
    --force_split                         # Regenerate scaffold splits
    --finetune                            # Fine-tune ESM-2 + MolFormer first
    --use_finetuned                       # Use existing fine-tuned embeddings
    --debug                               # Verbose output
```

## Output Structure

```
results/benchmark_human_8M/
├── benchmark_comparison.json             # Full results with metadata
├── benchmark_grouped_bar.png             # Grouped bar chart
├── benchmark_radar.png                   # Radar chart
├── benchmark_heatmap.png                 # Models × metrics heatmap
├── benchmark_mcc_ranking.png             # MCC ranking bar chart
├── benchmark_strip.png                   # Per-seed variance strip plot
├── level1_fingerprint/human/             # Level 1 per-seed outputs
│   ├── seed_42/
│   ├── seed_123/
│   └── ...
├── level2_embedding_8M/human/            # Level 2 per-seed outputs
├── level3_mat_8M/human/                  # Level 3 per-seed outputs
└── level4_crossatt_8M/                   # Level 4 per-seed outputs
```

## Dependencies

Core dependencies (must be installed in `env`):
- `torch` ≥ 2.1
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `tqdm`

Level-specific (imported lazily):
- `split_comparison_analysis` — Levels 1 & 2
- `crossattention_split_analysis` — Level 4
- `src.finetuning` — Fine-tuning step
