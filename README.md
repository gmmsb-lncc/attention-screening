# semantic-screening 🧬

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-gmmsb--lncc%2Fsemantic--screening-green.svg)](https://github.com/gmmsb-lncc/semantic-screening)

**Hierarchical benchmark framework for semantic screening of protein–ligand interactions using foundation language models.**

semantic-screening is an extensible platform that combines frozen protein language models ([ESM-2](https://github.com/facebookresearch/esm)) with molecular language models ([MoLFormer](https://github.com/IBM/molformer)) to predict compound bioactivity against kinase targets. It implements a **five-level hierarchical benchmark** that decomposes the sources of predictive gain — from classical fingerprints to learned bimodal attention pooling and 2D interaction maps — under a single, rigorously controlled experimental protocol with scaffold-based splitting and multi-seed evaluation.

---

## 🔬 Scientific Motivation

**Kinases** comprise ~2% of the human proteome (518 genes) but regulate ~30% of all cellular proteins through phosphorylation. Dysregulation drives oncogenic transformation, inflammatory disease, and antimicrobial resistance. The central pharmacological challenge is achieving **selectivity** across a highly conserved catalytic domain — all 518 human kinases share >85% structural similarity in their ATP-binding pocket.

**The semantic-screening hypothesis**: abandon geometric representations and operate directly on **primary sequence information** interpreted through contextual embeddings from foundation language models. This reformulation answers the selectivity question through *semantic compatibility in latent space* rather than geometric fit in 3D space — with universal applicability to any protein with a known sequence, including the ~40% of kinases lacking experimental structures (the "dark kinome").

---

## 🏗️ Five-Level Hierarchical Benchmark

The framework evaluates five levels of increasing representational complexity, all under the **same scaffold split**, **same metrics**, and **same multi-seed protocol**. The only variable across levels is the **representation strategy**; classifiers are held constant.

| Level | Representation | Protein? | Aggregation | Feature Dim | Trainable Params | Isolated Variable |
|-------|---------------|----------|-------------|-------------|-----------------|------------------|
| **1a** | Morgan FP (1024-bit, r=2) | No | — | 1024 | 0 | Baseline |
| **1b** | MoLFormer embeddings | No | Mean pooling | 768 | 0 | Semantic repr. vs. classical |
| **1c** | MoLFormer embeddings | No | ResProj + MQAttn (4q, 8h) | 256 | ~461K | + Learned aggregation |
| **3** | ESM-2 + MoLFormer | Yes | ResProj + MQAttn + InterFeat + Aux | 1282 | ~543K | + Protein modality |
| **4** | ESM-2 + MoLFormer | Yes | CNN 2D + Hierarchical Attn | 64 | ~337K | + Spatial interactions |

> **Parameter counts** for ESM-2 8M (`d_P = 320`). Levels 3 and 4 scale with the chosen ESM-2 variant (8M / 150M / 650M).

### Key Transitions

- **1a → 1b**: Value of semantic pre-trained representations vs. classical fingerprints
- **1b → 1c**: Value of learned projection + selective attention vs. uniform mean pooling
- **1c → 3**: Value of protein information + bimodal interaction features
- **3 → 4**: Value of explicit spatial residue–atom interaction modeling (end-to-end)

---

## 🧪 Evaluation Protocol

### Scaffold Split (No Data Leakage)

All levels are evaluated under **Bemis–Murcko scaffold splitting**, ensuring that no chemical scaffold appears in both training and test sets. A **universal partition** is computed once over the combined Human + Non-Human corpus, guaranteeing inter-corpus scaffold isolation.

```
Scaffold Split:  Train (~80%)  /  Val (~10%)  /  Test (~10%)
                       │               │              │
  Level 1a:            │    FP(val)  ──┤   FP(test) ──┤
  Level 1b:            │    Mean(val) ─┤   Mean(test) ┤
  Level 1c:  train AttnPool → AttnPool(val)├ AttnPool(test)┤
  Level 3:   train AttnPool → AttnPool(val)├ AttnPool(test)┤
  Level 4:   train CNN 2D → end-to-end evaluation on test  │
                       │               │              │
                       │         ┌─────┘       ┌──────┘
                       │         ▼             ▼
                       │    KNN/MLP.fit()  KNN/MLP.predict()
                       │    (on val feats) (on test feats)
```

### Canonical Classifiers (Levels 1a–3)

All non-end-to-end levels share the **exact same** classifier pipeline:

| Component | Specification |
|-----------|--------------|
| **KNN** | FAISS cosine similarity, *k* = 5, distance-weighted voting |
| **MLP** | 9-candidate topology search (128–512 neurons), 5-fold CV, 3 restarts, ensemble of 5, OOF threshold refinement |
| **Scaler** | `StandardScaler` (fit on reference partition, applied to all) |
| **Metric** | MCC-optimized threshold selection with stability penalty |

### Multi-Seed Protocol

Every experiment runs across 5 independent seeds: `{42, 123, 456, 789, 1024}`. Metrics are reported as mean ± std, separating optimization variance from genuine performance differences.

### Primary Metric: MCC

The **Matthews Correlation Coefficient** (MCC) is the sole criterion for model comparison — the only binary classification metric invariant to class proportion and with complete statistical interpretation as a Pearson correlation.

---

## 📊 Datasets

Built from **ChEMBL 35** with rigorous curation: direct biochemical assays only (IC₅₀, Kᵢ, K_d), PAINS filtering, IQR-based outlier removal, and monotonic profile filtering.

| Dataset | Samples | Compounds | Kinases | Active % |
|---------|---------|-----------|---------|----------|
| **Human** | 473,760 | 136,003 | 517 | ~42% |
| **Non-Human** | 14,080 | 7,428 | 114 | ~40% |
| **All** (combined) | 487,840 | — | — | — |
| **After monotonic filtering** | 386,099 | — | 642 | ~43% |

**Monotonic filtering** removes kinases and compounds with trivially predictable bioactivity profiles (all-active or all-inactive), reducing the Non-Human corpus by 30.8% and the Human corpus by 20.4%.

---

## 🧬 Foundation Models

Both models operate as **frozen feature extractors** — weights are never updated during benchmark training.

### Protein Encoder: ESM-2

| Model | Parameters | Embedding Dim | Layers |
|-------|-----------|---------------|--------|
| `esm2_t6_8M_UR50D` | 8M | 320 | 6 |
| `esm2_t30_150M_UR50D` | 150M | 640 | 30 |
| `esm2_t33_650M_UR50D` | 650M | 1280 | 33 |

### Ligand Encoder: MoLFormer

| Model | Parameters | Embedding Dim | Pre-training |
|-------|-----------|---------------|-------------|
| MoLFormer-XL | 47M | 768 | MLM on 1.1B molecules (ZINC + PubChem) |

Embeddings are **pre-computed once** and stored as `.npy` matrices, amortizing inference cost across all epochs, seeds, and levels.

---

## 📂 Project Structure

```
semantic-screening/
├── semantic_screening_models.py        # CLI entry point (thin wrapper)
├── run_benchmark.sh                    # Automated train→test pipeline
├── benchmark/                          # Core benchmark package
│   ├── cli.py                          # CLI parser → BenchmarkConfig
│   ├── config.py                       # Frozen config, constants, paths
│   ├── orchestrator.py                 # Multi-seed pipeline coordinator
│   ├── classifiers.py                  # Canonical KNN/MLP (9-candidate CV)
│   ├── splits.py                       # Scaffold split verification
│   ├── metrics.py                      # Metric aggregation
│   ├── reporting.py                    # JSON export + terminal tables
│   ├── visualization.py               # Plot generation (bar, radar, heatmap)
│   ├── embeddings.py                   # AttentionPooling module
│   ├── finetuning.py                   # Optional ESM-2/MoLFormer fine-tuning
│   ├── progress.py                     # tqdm step tracker
│   └── levels/                         # Level runners (Template Method)
│       ├── base.py                     # BaseLevelRunner ABC
│       ├── matrix_utils.py             # Matrix loading, padding, pooling
│       ├── level1.py                   # Level 1a: Fingerprints
│       ├── level1b.py                  # Level 1b: MoLFormer mean pooling
│       ├── level1c.py                  # Level 1c: MoLFormer attention pooling
│       ├── level3.py                   # Level 3: Bimodal attention pooling
│       ├── level4_cnn.py               # Level 4: CNN 2D interaction maps
│       └── ...                         # Experimental levels (4crossatt, 5, 6)
│
├── scaffolds_splits/                   # Scaffold split logic & output
│   └── output/                         # Pre-computed universal partitions
│
├── scripts/                            # Utility scripts
│   ├── scaffold_split.py               # Universal scaffold split generation
│   └── ...
│
├── src/                                # Legacy source & embedding generation
│   └── build/embeddings/strategies/    # ESM-2, MoLFormer extractors
│
├── docs/                               # Extended documentation
└── tests/                              # Unit and integration tests
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/gmmsb-lncc/semantic-screening.git
cd semantic-screening

# Create conda environment
conda env create -f environment.yml
conda activate docktkinase

# Install post-install dependencies
python scripts/post_install.py
```

### Running the Benchmark

The benchmark operates in two phases: **train** (fit on train, evaluate on val) and **test** (fit on val, evaluate on held-out test). The test set is never loaded during training.

```bash
# Automated pipeline (recommended): train + test with rigor safeguards
bash run_benchmark.sh

# Manual: specify dataset, embedding, and levels
# Train phase
python semantic_screening_models.py \
    --dataset non_human --embedding 8M \
    --levels 1a 1b 1c 3 4cnn \
    --epochs 500 --train

# Test phase (reuses frozen train-phase MLP selection)
python semantic_screening_models.py \
    --dataset non_human --embedding 8M \
    --levels 1a 1b 1c 3 4cnn \
    --test

# Quick baseline: Level 1a only (fingerprint, no GPU needed)
python semantic_screening_models.py --dataset non_human --embedding 8M --levels 1a --train

# Human dataset with ESM-2 150M
python semantic_screening_models.py --dataset human --embedding 150M --levels 1a 1b 1c 3 --train
```

### Output Structure

```
results/benchmark_{dataset}_{embedding}/
├── train/
│   ├── benchmark_comparison.json       # Full results with metadata
│   ├── benchmark_*.png                 # Visualization plots
│   ├── level1a_fingerprint/            # Per-level, per-seed outputs
│   ├── level1b_ligmean_{emb}/
│   ├── level1c_ligattn_{emb}/
│   ├── level3_attnpool_{emb}/
│   └── level4_cnn_{emb}/
└── test/
    ├── benchmark_comparison.json
    └── ...
```

---

## ⚙️ Environment Variables

The benchmark is configured through environment variables for reproducibility:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_MLP_USE_CV` | `1` | Enable stratified CV for MLP selection |
| `BENCHMARK_MLP_FOLDS` | `5` | Number of CV folds |
| `BENCHMARK_MLP_CAL_RESTARTS` | `3` | Restarts per candidate for threshold calibration |
| `BENCHMARK_MLP_ENSEMBLE` | `5` | Ensemble size for final MLP prediction |
| `BENCHMARK_MLP_OOF_THRESHOLD` | `1` | OOF threshold refinement (recommended) |
| `BENCHMARK_MLP_FULL_REFIT` | `0` | Disable early stopping in final refit |
| `BENCHMARK_LEVEL3_INTERACTION_FEATURES` | `0` | Enable interaction features (prod, diff, cos) |
| `BENCHMARK_LEVEL3_AUX_INTERACTIONS` | `0` | Train aux head with interaction features |
| `BENCHMARK_LEVEL3_MULTILAYER_LAYERS` | `` | Multi-layer MoLFormer (e.g., `"4,5,6"`) |
| `BENCHMARK_LEVEL3_HIDDEN_DIM` | auto | Override hidden dim (auto-scaled per ESM-2) |
| `BENCHMARK_REQUIRE_TRAIN_SELECTION` | `1` | Require frozen train-phase MLP selection for test |
| `BENCHMARK_STRICT_LEVEL_COMPLETENESS` | `1` | Require all requested levels to complete |

---

## 🔒 Anti-Leakage Protocol

The framework enforces strict separation between model selection and final evaluation:

1. **Train mode** (`--train`): Classifiers trained on train split (80%), evaluated on validation (10%). Test set is **never loaded**.
2. **Test mode** (`--test`): Classifiers trained on validation (10%), evaluated on held-out test (10%). MLP configuration is **frozen** from train phase — no re-selection allowed.
3. **Scaffold isolation**: Universal partition guarantees zero scaffold overlap between train/val/test, including inter-corpus isolation.
4. **Monotonic filtering**: Removes trivially predictable entities (all-active/all-inactive kinases and compounds).

---

## 📖 CLI Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | `human`, `non_human`, or `all` | Required |
| `--embedding` | ESM-2 variant: `8M`, `150M`, `650M` | `8M` |
| `--levels` | Levels to run (e.g., `1a 1b 1c 3 4cnn`) | All |
| `--train` / `--test` | Execution mode (mutually exclusive) | `--train` |
| `--epochs` | Max training epochs for learned levels | `500` |
| `--batch_size` | Batch size for learned levels | `32` |
| `--learning_rate` | Learning rate for learned levels | `1e-4` |
| `--model_selection_metric` | `val_loss` or `mcc` | `val_loss` |
| `--seeds` | Custom seeds (space-separated) | `42 123 456 789 1024` |
| `--force` | Force recalculation of all levels | Off |
| `--output_dir` | Custom output directory | Auto |

---

## 📚 Further Documentation

- **[Benchmark Package README](benchmark/README.md)** — Architecture, module reference, design patterns
- **[Concepts Guide](docs/CONCEPTS.md)** — Platform vs. architecture distinction
- **[Methodology](docs/methodology.md)** — Comprehensive scientific background
- **[User Guide](docs/02-user-guide/)** — Detailed usage instructions
- **[Architecture](docs/03-architecture/)** — System design patterns

---

## Citation

```bibtex
@software{semanticscreening2026,
  title   = {semantic-screening: Hierarchical benchmark for semantic screening
             of protein-ligand interactions},
  author  = {Sulfierry, Leon and GMMSB-LNCC},
  year    = {2026},
  url     = {https://github.com/gmmsb-lncc/semantic-screening},
  version = {4.0}
}
```

## Contact

- **Repository**: [gmmsb-lncc/semantic-screening](https://github.com/gmmsb-lncc/semantic-screening)
- **Issues**: [Bug reports & features](https://github.com/gmmsb-lncc/semantic-screening/issues)

---

**Status**: Production Ready | **Version**: 4.0 | **Last Updated**: March 2026
