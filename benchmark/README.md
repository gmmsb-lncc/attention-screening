# Benchmark Package

Modular framework for comparing protein–ligand interaction prediction models across a **six-level hierarchical representation benchmark**, using universal scaffold-based data splits for rigorous evaluation.

> SOLID-compliant package with ~15 focused modules. All levels share the same canonical classifiers, scaffold split, activity threshold, metrics, and multi-seed protocol.

## Scientific Design

### Core Principle: Single Independent Variable

The **only factor** that varies across levels is the **molecular representation**. All levels 1a–3 use the exact same canonical KNN and MLP classifiers (`benchmark/classifiers.py`), the same scaffold split, the same activity threshold (pChEMBL ≥ 6.0), and the same evaluation metrics (MCC primary). Level 4 trains end-to-end but is evaluated under the same scaffold split and metrics.

### Classifier Pipeline (Levels 1a–3)

| Component | Specification |
|-----------|--------------|
| **KNN** | FAISS inner-product on L2-normalised features (cosine similarity), *k* = 5, distance-weighted voting |
| **MLP** | `sklearn.neural_network.MLPClassifier` with hidden layers `(256, 128)`, ReLU activation, Adam solver, adaptive learning rate, α = 10⁻³, early stopping (patience = 20, 10% validation), max 2000 iterations, decision threshold = 0.5 |
| **Scaler** | `StandardScaler` (fit on reference partition, applied to all partitions) |

### Evaluation Protocol

```
Scaffold Split:  Train (~80%)  /  Val (~10%)  /  Test (~10%)
                       │               │               │
  Level 1a:            │     FP(val) ──┤    FP(test) ──┤
  Level 1b:            │   Mean(val) ──┤   Mean(test) ─┤
  Level 1c:  train AttnPool →  pool(val)├    pool(test) ┤
  Level 2:             │  Mean₂(val) ──┤  Mean₂(test) ─┤
  Level 3:   train AttnPool₂→  pool(val)├    pool(test) ┤
  Level 4:   train CNN 2D  →  end-to-end on test        │
                       │               │               │
                       │         ┌─────┘        ┌──────┘
                       │         ▼              ▼
                       │    KNN/MLP.fit()   KNN/MLP.predict()
                       │    (on val feats)  (on test feats)
```

**Two-phase protocol**: `--train` fits classifiers on training data (80%) and evaluates on validation (10%); `--test` fits on validation and evaluates on held-out test (10%). The test set is **never loaded** during training. The MLP selection from the train phase is **frozen** and reused in the test phase.

## The Six Levels

| Level | Input | Aggregation | Trainable Params | Module |
|-------|-------|-------------|-----------------|--------|
| **1a** | Morgan FP (1024-bit, r=2) — ligand only | — | 0 | `levels/level1.py` |
| **1b** | MoLFormer per-token embeddings — ligand only | Masked mean pooling (zero params) | 0 | `levels/level1b.py` |
| **1c** | MoLFormer per-token embeddings — ligand only | Proj (Linear→LN→GELU→Drop) + attention pooling (1q, 8h) | ~264K | `levels/level1c.py` |
| **2** | ESM-2 protein + MoLFormer ligand embeddings | Masked mean pooling per modality + concatenation | 0 | `levels/level2.py` |
| **3** | ESM-2 protein + MoLFormer ligand matrices | Proj + attention pooling (1q, 8h) per modality + concat | ~528K | `levels/level3.py` |
| **4** | ESM-2 protein + MoLFormer ligand matrices | Multi-head projection → 2D interaction maps → CNN 2D → hierarchical attn pooling | ~550K | `levels/level4_cnn.py` |

### Level Architecture Details

**Levels 1a, 1b, 2**: No trainable parameters. Level 1a uses Morgan fingerprints. Levels 1b and 2 apply masked mean pooling directly over raw foundation model embeddings — no projection, no learned components.

**Level 1c**: Single-layer projection (`Linear → LayerNorm → GELU → Dropout`, 768 → 256) followed by attention pooling with a single learned query vector (q ∈ ℝ^256) and multi-head attention (8 heads). Produces feature vectors z_L ∈ ℝ^256.

**Level 3**: Same backbone as Level 1c, replicated independently for protein and ligand modalities. Each modality has its own projection and attention pooling module with an independent learned query. Outputs are concatenated: z^(3) = [z_P ∥ z_L] ∈ ℝ^512. No cross-modal interaction is modeled — concatenation is the sole fusion point.

**Level 4 (CNN)**: K=8 multi-head linear projections → scaled dot-product interaction maps ∈ ℝ^(K×n×m) → 4-layer CNN 2D (including dilated convolution) → hierarchical attention pooling (first along ligand axis, then along protein axis) → linear classifier (end-to-end).

### Training Protocol (Levels 1c, 3)

- **Optimizer**: AdamW (η = 10⁻⁴, weight decay λ = 0.01)
- **LR schedule**: CosineAnnealingLR (T = 500 epochs)
- **Gradient clipping**: ‖∇‖₂ ≤ 1.0
- **Early stopping**: patience = 5 (monitoring validation loss)
- **Auxiliary head**: Binary classification head (BCEWithLogitsLoss) discarded after training
- **Weight initialization**: Xavier uniform (projection), std = 0.02 (attention query)

### Data Flow per Level

- **Level 1a**: Scaffold val/test TSV → Morgan FP → `StandardScaler` → KNN/MLP
- **Level 1b**: Scaffold val/test TSV → load MoLFormer matrices → mean pool → `StandardScaler` → KNN/MLP
- **Level 1c**: Train projection + attn pooling on train (early stop on val) → extract features from val/test → `StandardScaler` → KNN/MLP
- **Level 2**: Scaffold val/test TSV → load ESM-2 + MoLFormer matrices → mean pool per modality → concatenate → `StandardScaler` → KNN/MLP
- **Level 3**: Train bimodal projection + attn pooling on train (early stop on val) → extract features from val/test → `StandardScaler` → KNN/MLP
- **Level 4**: Train CNN 2D end-to-end on train (early stop on val MCC) → classify test directly

## Architecture

```
attention_screening_models.py          ← Thin CLI entry point
│
└── benchmark/
    ├── __init__.py                   ← Public API exports
    ├── config.py                     ← BenchmarkConfig (frozen dataclass), constants
    ├── cli.py                        ← argparse → BenchmarkConfig
    ├── orchestrator.py               ← Facade: coordinates the full pipeline
    ├── classifiers.py                ← Canonical KNN + MLP (traditional classifiers)
    ├── progress.py                   ← tqdm-based step tracker
    ├── splits.py                     ← Scaffold split verification / generation
    ├── embeddings.py                 ← AttentionPooling module
    ├── finetuning.py                 ← Optional ESM-2 / MoLFormer fine-tuning
    ├── metrics.py                    ← Multi-level metric aggregation
    ├── reporting.py                  ← Terminal table + JSON export
    ├── visualization.py              ← Plot generation (bar, radar, heatmap, ranking, strip)
    └── levels/
        ├── __init__.py               ← Re-exports all runners
        ├── base.py                   ← BaseLevelRunner ABC (Template Method)
        ├── matrix_utils.py           ← Shared matrix loading, padding, mean-pool
        ├── level1.py                 ← Level 1a: Morgan fingerprints
        ├── level1b.py                ← Level 1b: MoLFormer mean pooling
        ├── level1c.py                ← Level 1c: MoLFormer attention pooling
        ├── level2.py                 ← Level 2: Bimodal mean pooling
        ├── level3.py                 ← Level 3: Bimodal attention pooling
        ├── level4_cnn.py             ← Level 4: CNN 2D interaction maps
        ├── level4_crossatt.py        ← Level 4 variant: Cross-attention (experimental)
        ├── level4.py                 ← Level 4 variant: LoRA fine-tuning (experimental)
        ├── level5.py                 ← Level 5a: Domain adaptation + GRL (experimental)
        ├── level5b.py                ← Level 5b: AttnPool + GRL (experimental)
        ├── level6a.py                ← Level 6a: BAN + GRL (experimental)
        └── level6b.py                ← Level 6b: AttnPool + BAN + GRL (experimental)
```

## Design Principles

### SOLID Compliance

- **Single Responsibility**: Each module owns one concern — config, CLI, splits, a single level, metrics, plotting
- **Open/Closed**: Adding a new level requires only a new file in `levels/` inheriting `BaseLevelRunner`
- **Liskov Substitution**: All runners implement the `BaseLevelRunner` interface
- **Interface Segregation**: Narrow, focused APIs: `ensure_scaffold_splits()`, `train_knn_mlp()`, `generate_all()`
- **Dependency Inversion**: Orchestrator depends on `BaseLevelRunner` abstraction, not concrete implementations

### Patterns

- **Template Method** — `BaseLevelRunner.run()` defines the multi-seed loop; subclasses implement `run_single_seed()`
- **Facade** — `BenchmarkOrchestrator.run()` hides all pipeline complexity
- **Frozen Dataclass** — `BenchmarkConfig` is immutable after construction

## Quick Start

```bash
# Full benchmark: core 6 levels (non-human dataset, ESM-2 8M)
# Train
python attention_screening_models.py --dataset non_human --embedding 8M \
    --levels 1a 1b 1c 2 3 4cnn --train

# Test
python attention_screening_models.py --dataset non_human --embedding 8M \
    --levels 1a 1b 1c 2 3 4cnn --test

# Or use the automated script:
DATASET=non_human EMBEDDING=8M LEVELS_CSV=1a,1b,1c,2,3,4cnn bash run_benchmark.sh
```

## CLI Reference

```
python attention_screening_models.py \
    --dataset {human,non_human,all}       # Required
    --embedding {8M,150M,650M}            # Default: 8M
    --levels 1a 1b 1c 2 3 4cnn           # Default: all available
    --seeds 42 123 456                    # Default: [42, 123, 456, 789, 1024]
    --output_dir ./my_results             # Default: ./results/benchmark_{dataset}_{embedding}
    --scaffold_split_dir path/to/splits   # Default: scaffolds_splits/output
    --epochs 500                          # Max DL epochs (default: 500)
    --batch_size 32                       # DL batch size (default: 32)
    --learning_rate 1e-4                  # DL learning rate (default: 1e-4)
    --model_selection_metric {val_loss,mcc} # Early stop metric (default: val_loss)
    --train | --test                      # Execution mode (mutually exclusive)
    --force                               # Recompute all levels
```

## Adding a New Level

1. Create `benchmark/levels/level_new.py`:
   ```python
   from benchmark.levels.base import BaseLevelRunner

   class NewLevelRunner(BaseLevelRunner):
       @property
       def level_tag(self) -> str:
           return "level_new_method"

       def run_single_seed(self, seed, output_dir, **kwargs):
           # Your training/evaluation logic
           return {"Split by Scaffold": {"KNN": {...}, "MLP": {...}}}
   ```

2. Register it in `benchmark/levels/__init__.py`.

3. Add level key to `VALID_LEVELS` in `config.py` and add label/color entries.

4. Add a runner instantiation block in `orchestrator.py:_build_runners()`.

## Dependencies

Core:
- `torch` ≥ 2.1, `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`, `tqdm`
- `faiss-cpu` (or `faiss-gpu`) — KNN classification

Level-specific (imported lazily):
- `rdkit` — Level 1a (Morgan fingerprints)
