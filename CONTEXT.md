# CONTEXT.md — Project Context for Developers and LLMs

This document provides the full context needed to understand, modify, or extend the
**semantic-screening** project. It is designed to be self-contained: any developer or LLM
reading only this file should be able to orient themselves in the codebase.

---

## 1. What This Project Does

**semantic-screening** predicts whether a small molecule (drug candidate) will bind to a
protein target (kinase), and how strongly. It does this entirely from **primary sequences**
(amino acids for proteins, SMILES for molecules) — no 3D crystal structures required.

The core idea: Protein Language Models (ESM-2) and Chemical Language Models (MoLFormer)
encode proteins and molecules into dense vector representations. These representations are
then compared using various ML and deep learning models to predict interaction.

**Key terms:**
- **pChEMBL**: `-log10(IC50 in M)`. Higher = stronger binding. Threshold: **pChEMBL >= 6.0** (IC50 <= 1000 nM) = **active**.
- **DT-Kinase**: The CNN + Cross-Attention neural architecture implemented here.
- **Scaffold**: The Murcko core ring system of a molecule (chemical backbone).

---

## 2. The 4-Level Benchmark

The project implements a unified benchmark (`semantic_screening_models_beta.py`) that
evaluates models at 4 levels of increasing complexity:

```
Level 1: Fingerprint + KNN/MLP        — Classical cheminformatics baseline
Level 2: Embedding vectors + KNN/MLP  — Value of PLM representations
Level 3: CNN on embedding matrices     — Value of per-residue/per-atom context
Level 4: CNN + Cross-Attention         — Value of explicit interaction modeling
```

Each level answers a scientific question:
- **L1 vs L2**: Do PLM embeddings improve over hand-crafted fingerprints?
- **L2 vs L3**: Does preserving per-token context (matrices) beat pooled vectors?
- **L3 vs L4**: Does cross-attention (explicit interaction modeling) help beyond CNN?

**All levels use the same scaffold split** for fair comparison.

### Running the benchmark

```bash
# Default: Levels 1, 2, 3
python semantic_screening_models_beta.py --dataset non_human --embedding 8M

# All 4 levels
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 1,2,3,4

# Quick test (only baseline levels)
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 1,2
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | `human`, `non_human`, or `all` |
| `--embedding` | `8M` | ESM-2 model: `8M`, `150M`, `650M` |
| `--levels` | `1,2,3` | Which levels to run |
| `--seeds` | `[42,123,456,789,1024]` | Seeds for Level 3/4 multi-seed evaluation |
| `--epochs` | `500` | Max training epochs for Level 3/4 |
| `--patience` | `5` | Early stopping patience |
| `--batch_size` | `32` | Batch size for Level 3/4 |
| `--force` | `False` | Recalculate even if cached |

---

## 3. Scaffold Split — The Core Methodology

### Why scaffold splits?

Drug discovery data has **chemical series**: many compounds share the same ring backbone
(Murcko scaffold). A random train/test split allows compounds from the same series to
appear in both, inflating performance through memorization rather than generalization.

### How it works

**Implementation**: `scaffold_split.py` + `scaffolds_splits/scenario_splitter.py`

```
Step 1: Extract Murcko scaffold for every compound (RDKit)
Step 2: Select ~10% of scaffolds as TEST scaffolds (shared across human/non_human)
        - Optimized via random restarts to balance class distribution
Step 3: Remaining scaffolds → split into TRAIN (~80%) and VAL (~10%)
        - Scaffold-disjoint: no scaffold appears in more than one split
Step 4: Validate disjointness constraints automatically
```

### Guarantees

1. **No scaffold overlap**: `scaffolds_train ∩ scaffolds_val = ∅`, `scaffolds_train ∩ scaffolds_test = ∅`
2. **Fixed test set**: Test scaffolds are shared across human/non_human datasets
3. **Class balance**: Optimization minimizes class-rate deviation
4. **Reproducible**: Deterministic given seed

### File structure

```
scaffolds_splits/output/
    manifest.json                               # Full metadata
    universal_scaffolds.json                    # Scaffold assignments
    {dataset}_test.tsv.gz                       # Fixed test set
    {dataset}_train.tsv.gz                      # Default train (Sc scenario)
    {dataset}_val.tsv.gz                        # Default val (Sc scenario)
    scenarios/
        Sc/                                     # Scaffold-disjoint (primary)
            {dataset}_train.tsv.gz
            {dataset}_val.tsv.gz
        S1/                                     # Random (baseline with leakage)
        S2/                                     # Compound-disjoint
        S3/                                     # Kinase-disjoint
        S4/                                     # Double-disjoint (compound + kinase)
    split_class_distribution_summary.csv
```

### Scenario codes

| Code | Name | Splitting unit | Use |
|------|------|----------------|-----|
| **Sc** | **Scaffold** | **Murcko scaffolds** | **Default for all benchmarks** |
| S1 | Random | Individual rows | Baseline (allows leakage) |
| S2 | Compound | Unique compounds | No compound leakage |
| S3 | Kinase | Unique kinases | No kinase leakage |
| S4 | New Comp.+New Kinase | Both | Double disjointness |

### Loading splits in code

All levels use `_load_precomputed_scaffold_splits()` from
`crossattention_split_analysis/experiment.py`, which reads `.tsv` or `.tsv.gz`
transparently.

---

## 4. Data Layout

### Input datasets (not in git)

```
tests/datasets/
    kinase_human_compounds.tsv          # ~476K rows
    kinase_non_human_compounds.tsv      # ~15.6K rows
    kinase_all_compounds.tsv            # Combined
```

**Columns**: `seq_id`, `target_sequence`, `target_kinase`, `chembl_id`, `canonical_smiles`, `pchembl_value`

### Pre-computed embeddings

```
results/protein_model_benchmark_{human|non_human}_v2/
    esm2_t6_8M_UR50D/build/
        protein_matrices/       # {seq_id}_matrix.npy       [seq_len, 320]
        molformer_matrix/       # {chembl_id}_matrix.npy    [mol_len, 768]
        ligand_embeddings/      # {chembl_id}_embedding.npy [768] (mean-pooled)
    esm2_t30_150M_UR50D/build/
        ...  (same structure, protein dim = 640)
    esm2_t33_650M_UR50D/build/
        ...  (same structure, protein dim = 1280)
```

### Embedding dimensions

| Model | Shorthand | Protein dim | Ligand dim |
|-------|-----------|-------------|------------|
| `esm2_t6_8M_UR50D` | `8M` | 320 | 768 |
| `esm2_t30_150M_UR50D` | `150M` | 640 | 768 |
| `esm2_t33_650M_UR50D` | `650M` | 1280 | 768 |

Ligand dim is always **768** (MoLFormer). Ligand vectors (`ligand_embeddings/`) are
mean-pooled from MoLFormer per-token matrices.

---

## 5. Benchmark Output

```
results/benchmark_{dataset}_{embedding}/
    level1_fingerprint/{dataset}/           # Level 1 results
        split_comparison_results.json
    level2_embedding_{emb}/{dataset}/       # Level 2 results
        split_comparison_results.json
    level3_cnn_{emb}/                       # Level 3 results
        *_crossattention_analysis_results.json
    level4_cnn_ca_{emb}/                    # Level 4 results (if --levels includes 4)
        *_crossattention_analysis_results.json
    benchmark_comparison.json               # Unified metrics table
    benchmark_grouped_bar.png               # Comparative bar chart
    benchmark_radar.png                     # Radar chart
    benchmark_heatmap.png                   # Performance heatmap
    benchmark_mcc_ranking.png               # MCC ranking
    benchmark_per_metric.png                # Per-metric strip chart
```

### Metrics collected

| Metric | Description | Role |
|--------|-------------|------|
| **MCC** | Matthews Correlation Coefficient | **Primary selection metric** |
| AUC | Area Under ROC Curve | Ranking quality |
| F1 | Harmonic mean of Precision and Recall | Balance |
| Accuracy | Overall correctness | General |
| Precision | True positives / predicted positives | FP control |
| Recall | True positives / actual positives | FN control |

Level 3/4 report **mean ± std** across 5 seeds.

---

## 6. Test Set Integrity

The test set is **never** used during training or threshold optimization in any level:

- **Threshold calibration**: Always on validation set (maximizing MCC)
- **Scaler fit**: Always on training set only
- **Model training**: Only train + val (val for early stopping in Level 3/4)
- **Test evaluation**: Happens after training is complete, with val-calibrated threshold

All 4 levels use the **same test set** from the same scaffold split files.

---

## 7. Key Files Map

### Orchestration

| File | Purpose |
|------|---------|
| `semantic_screening_models_beta.py` | **Unified benchmark orchestrator** (main entry point) |
| `scaffold_split.py` | Generate scaffold splits |
| `split_comparison_analysis.py` | Level 1 & 2 (KNN/MLP with fingerprints or embeddings) |
| `crossattention_split_analysis/experiment.py` | Level 3 & 4 (CNN / CNN+CrossAttention) |

### Configuration

| File | Purpose |
|------|---------|
| `crossattention_split_analysis/config.py` | `TrainingConfig`, `SUPPORTED_EMBEDDINGS`, `DATASET_PATHS`, `DEFAULT_SEEDS` |
| `scaffolds_splits/scenario_splitter.py` | Scenario-specific splitting logic |
| `scaffolds_splits/validation.py` | Split disjointness validation |

### Models

| File | Purpose |
|------|---------|
| `src/classifier/models/cross_attention_model.py` | `CrossAttentionAffinityModel` (DT-Kinase) |
| `src/classifier/models/diffusion_model.py` | Diffusion variant |
| `crossattention_split_analysis/training/trainer.py` | Training loop with tqdm, early stopping |
| `crossattention_split_analysis/training/evaluator.py` | Evaluation, threshold optimization |

### Data

| File | Purpose |
|------|---------|
| `crossattention_split_analysis/data/datasets.py` | `AttentionMatrixDataset`, collate function |
| `scripts/extract_ligand_vectors.py` | Mean-pool MoLFormer matrices → vectors |

---

## 8. Architecture Details

### Level 3: CNN-only (`num_cross_attn_layers=0`)

```
Protein matrix [L, d_prot] → Linear → CNN encoder (kernels 3,5,7) → LayerNorm → Mean pool → [hidden_dim]
Ligand matrix  [M, d_lig]  → Linear → CNN encoder (kernels 3,5,7) → LayerNorm → Mean pool → [hidden_dim]
                                                                                    ↓
                                                                        Concat [2 * hidden_dim]
                                                                                    ↓
                                                                        Multi-task head → classification + regression
```

### Level 4: CNN + Cross-Attention (`num_cross_attn_layers=2`)

Same as Level 3, but between CNN encoding and pooling, adds:

```
                    ┌─────────────────────────────────────────┐
                    │ Bidirectional Cross-Attention (×2 layers) │
                    │   Protein → Ligand: Q=prot, K=V=lig     │
                    │   Ligand → Protein: Q=lig, K=V=prot     │
                    │   8 heads, FFN(4×hidden), GELU, dropout  │
                    └─────────────────────────────────────────┘
```

### Default hyperparameters

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 256 |
| `num_cnn_layers` | 3 |
| `kernel_sizes` | (3, 5, 7) |
| `num_heads` | 8 |
| `ff_dim` | 1024 |
| `dropout` | 0.1 |
| `batch_size` | 32 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 0.01 |
| `patience` | 5 |
| `optimizer` | AdamW |
| `scheduler` | CosineAnnealingLR |

---

## 9. Reproducibility

- **Seeds**: `[42, 123, 456, 789, 1024]` (5 seeds, configurable)
- **Early stopping**: Best model selected by **validation MCC**
- **Checkpointing**: Atomic writes (temp file + rename) to prevent corruption
- **Caching**: Each level caches results as JSON; use `--force` to recalculate
- **Multi-task loss**: `L = 1.0 * BCE_classification + 0.5 * MSE_regression`

---

## 10. Environment

```bash
# Activate
source env/bin/activate    # or: conda activate docktkinase

# ESM-2 must be from local repo (pip versions segfault)
# Located at llm/ESM/ — src/__init__.py adds it to sys.path

# Python 3.9+ (env uses 3.12), PyTorch 2.0+
```

**Critical**: Never install `fair-esm` or `esm` via pip. The local `llm/ESM/` version
is required and is auto-loaded by `src/__init__.py`.

---

## 11. Dataset Types

| Dataset | Description | Rows | Kinases | Compounds |
|---------|-------------|------|---------|-----------|
| `human` | Human kinase compounds from ChEMBL | ~476K | ~590 | ~136K |
| `non_human` | Non-human kinase compounds | ~15.6K | ~231 | ~8K |
| `all` | Union of human + non_human | ~492K | ~821 | ~144K |

When `dataset=all`, the code loads splits from both human and non_human directories
and concatenates them. Embeddings are loaded from both
`protein_model_benchmark_human_v2/` and `protein_model_benchmark_non_human_v2/`.

---

## 12. Common Tasks

### Add a new embedding model

1. Add to `SUPPORTED_EMBEDDINGS` and `PROTEIN_DIMS` in `crossattention_split_analysis/config.py`
2. Generate embeddings: protein matrices in `protein_matrices/`, MoLFormer matrices in `molformer_matrix/`
3. Extract ligand vectors: `python scripts/extract_ligand_vectors.py --embedding-dir {build_dir}`

### Regenerate scaffold splits

```bash
python scaffold_split.py --output-dir scaffolds_splits/output --scenarios Sc
```

### Run a single level

```bash
# Level 1 only
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 1

# Level 3 with custom epochs
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 3 --epochs 100
```

### Force recalculation

```bash
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --force
```
