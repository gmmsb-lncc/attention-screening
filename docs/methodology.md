# semantic-screening: Methodology & Theory

**Author**: Leon Sulfierry (GMMSB-LNCC)
**Date**: March 2026
**Version**: 4.0 (Five-Level Hierarchical Benchmark)

---

## Abstract

semantic-screening is a modular, scalable platform for predicting kinase–ligand bioactivity using foundation language models. It implements a **five-level hierarchical benchmark** that decomposes the sources of predictive gain — from classical Morgan fingerprints to learned bimodal attention pooling and CNN 2D interaction maps — under a **single independent variable** protocol. All non-end-to-end levels share the same canonical KNN/MLP classifiers, scaffold-based splits (Bemis–Murcko), monotonic profile filtering, and multi-seed evaluation (5 seeds). The primary metric is MCC (Matthews Correlation Coefficient). This document details the theoretical foundations, architectural decisions, and evaluation methodology.

---

## Chapter 1: Introduction

### 1.1 The Kinase Drug Discovery Challenge

Protein kinases constitute ~2% of the human proteome (518 genes) but regulate ~30% of all cellular proteins through reversible phosphorylation. The central pharmacological challenge lies in the **Selectivity Paradox**: the ATP-binding pocket is highly conserved across the >500 human kinases (RMSD < 2Å), requiring ΔΔG differences of only 1.4 kcal/mol for 10-fold selectivity — below the systematic error of scoring functions.

Beyond human kinases, bacterial kinases of the Hanks type represent critical targets for combating antimicrobial resistance, a WHO-designated global health priority.

### 1.2 The Semantic Hypothesis

semantic-screening proposes a paradigm shift: abandon geometric representations and operate directly on **primary sequence information** interpreted through contextual embeddings from foundation language models. The central hypothesis: **sequence determines structure, determines function** — and this mapping is computationally recoverable via pre-trained Transformers.

### 1.3 Problem Formulation

Given a protein sequence $P = \{a_1, a_2, \ldots, a_L\}$ and a ligand SMILES string $L = \{s_1, s_2, \ldots, s_M\}$, predict the binary bioactivity label $y \in \{0, 1\}$ defined by:

$$y = \mathbb{1}[\text{pChEMBL} \geq 6.0] \quad \Leftrightarrow \quad IC_{50} \leq 1000\,\text{nM}$$

---

## Chapter 2: Foundation Models

Both models operate as **frozen feature extractors** — weights are never updated during benchmark training.

### 2.1 Protein Encoder: ESM-2

ESM-2 (Lin et al., 2023) is a bidirectional Transformer trained with Masked Language Modeling (MLM) on UniRef50. For a protein sequence $P$ of length $L$, the model outputs per-residue embeddings:

$$\Phi_P: \mathcal{A}^L \to \mathbb{R}^{L \times d_P}$$

| Model | Parameters | $d_P$ | Layers |
|-------|-----------|-------|--------|
| `esm2_t6_8M_UR50D` | 8M | 320 | 6 |
| `esm2_t30_150M_UR50D` | 150M | 640 | 30 |
| `esm2_t33_650M_UR50D` | 650M | 1280 | 33 |

The MLM objective:
$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(a_i | a_{\setminus \mathcal{M}})$$

The attention matrix encodes implicit contact maps: $A_{ij}$ captures whether residues $i$ and $j$ are spatially proximal, without explicit 3D coordinate supervision.

### 2.2 Ligand Encoder: MoLFormer

MoLFormer (Ross et al., 2022) is a Transformer encoder with **linear attention** (replacing quadratic $O(M^2)$ with $O(M)$ complexity via kernel decomposition), pre-trained on 1.1 billion molecules (ZINC + PubChem) using MLM:

$$\Phi_L: \mathcal{S}^M \to \mathbb{R}^{M \times 768}$$

Per-token embeddings encode molecular properties and SMILES syntax at each position.

### 2.3 Pre-Computation Strategy

Embeddings are computed **once** per unique protein/ligand and stored as `.npy` matrices. This amortizes inference cost across all epochs, seeds, and levels, and enables the benchmark to focus exclusively on representation strategy without re-running the expensive foundation models.

---

## Chapter 3: Data Curation & Splitting

### 3.1 Data Source

Building from **ChEMBL 35**, the curation pipeline applies:

1. **Direct assays only**: IC₅₀, Kᵢ, K_d (no cell-based or functional assays)
2. **pChEMBL normalisation**: $\text{pChEMBL} = -\log_{10}([\text{Aff}]_{\text{mol/L}})$
3. **PAINS filtering**: Remove pan-assay interference compounds
4. **IQR outlier removal**: Remove statistical outliers per kinase
5. **Monotonic filtering**: Remove trivially predictable profiles (see §3.3)

### 3.2 Scaffold-Based Splitting

All levels use **Bemis–Murcko scaffold decomposition** to prevent chemical series leakage. A **universal partition** is computed once over the combined Human + Non-Human corpus:

$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

Test scaffolds are selected via constrained optimisation (random restarts) to balance:
- Target test fraction (~10% of unique compounds)
- Class distribution preservation
- Cross-dataset proportionality

### 3.3 Monotonic Filtering

Entities with trivially predictable bioactivity profiles are removed:

| Category | Description | Impact |
|----------|-------------|--------|
| **Monotonic kinase** | All tested compounds are active (or all inactive) | Model memorises "kinase X → always active" |
| **Monotonic compound** | Active against all tested kinases (pan-active) or none | Model memorises "compound Y → always active" |

Removing these entities forces the model to learn genuine selectivity patterns rather than exploiting trivial baselines. The Non-Human corpus is reduced by ~30.8%, the Human corpus by ~20.4%.

### 3.4 Dataset Statistics (Post-Curation)

| Dataset | Samples | Compounds | Kinases | Active % |
|---------|---------|-----------|---------|----------|
| **Human** | 473,760 | 136,003 | 517 | ~42% |
| **Non-Human** | 14,080 | 7,428 | 114 | ~40% |
| **After monotonic filtering** | 386,099 | — | 642 | ~43% |

---

## Chapter 4: Five-Level Hierarchical Benchmark

### 4.1 Design Principle: Single Independent Variable

The only factor that varies across levels is the **molecular representation**. All levels 1a–3 use the exact same canonical KNN/MLP classifiers, the same scaffold split, the same activity threshold, and the same evaluation metrics.

### 4.2 Level 1a — Fingerprint Baseline

Morgan fingerprints (ECFP, 1024-bit, radius 2) computed via RDKit. Ligand-only, no protein information. **Zero trainable parameters.** This establishes the baseline for classical cheminformatics descriptors:

$$\mathbf{x}^{(1a)} = \text{MorganFP}(L) \in \{0, 1\}^{1024}$$

### 4.3 Level 1b — MoLFormer Mean Pooling

Per-token MoLFormer embeddings aggregated via uniform mean pooling. **Zero trainable parameters:**

$$\mathbf{x}^{(1b)} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{h}^L_i \in \mathbb{R}^{768}$$

### 4.4 Level 1c — MoLFormer Attention Pooling

Introduces learned aggregation via a two-layer residual projection followed by multi-query attention pooling:

1. **Residual Projection**: $\text{Proj}(h) = h + \text{GELU}(\text{LN}(W_2 \cdot \text{GELU}(\text{LN}(W_1 h + b_1))) + b_2)$
2. **Multi-Query Attention Pooling** (4 learned queries, 8 heads): collapses variable-length sequence into 4 fixed-size vectors, concatenated into $\mathbb{R}^{4 \times 64} = \mathbb{R}^{256}$
3. **Auxiliary head** (training-only): Binary classification head for gradient signal

**~461K trainable parameters.** Feature vector: $\mathbf{x}^{(1c)} \in \mathbb{R}^{256}$.

### 4.5 Level 3 — Bimodal Attention Pooling

Extends Level 1c to the bimodal (protein + ligand) setting:

1. **Dual residual projections**: One per modality (ESM-2 → $d$, MoLFormer → $d$)
2. **Dual attention pooling**: Multi-query attention (4 queries, 8 heads) per modality
3. **Explicit interaction features**:
   - Element-wise product: $z_{\text{prod}} = z_P \odot z_L$
   - Absolute difference: $z_{\text{diff}} = |z_P - z_L|$
   - Cosine similarity: $z_{\cos} = \frac{z_P \cdot z_L}{\|z_P\| \|z_L\|}$
4. **Auxiliary MLP** (2-layer): Trained on the full interaction-enriched vector
5. **Feature extraction**: Concatenation of pooled vectors + interaction features + aux head hidden layer

**~543K trainable parameters** (ESM-2 8M). Feature vector: $\mathbf{x}^{(3)} \in \mathbb{R}^{1282}$.

### 4.6 Level 4 — CNN 2D Interaction Maps

End-to-end architecture without downstream KNN/MLP:

1. **K=8 multi-head projection**: Linear projections of protein and ligand embeddings
2. **Interaction maps**: $\mathbf{I}_k = \frac{\mathbf{Z}_P^{(k)} (\mathbf{Z}_L^{(k)})^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$, stacked into $\mathbb{R}^{K \times n \times m}$
3. **4-layer CNN 2D**: Including dilated convolution (dilation=2) for expanded receptive field
4. **Hierarchical attention pooling**: First along ligand axis, then along protein axis
5. **Linear classifier**: Binary output with sigmoid

**~337K trainable parameters** (ESM-2 8M). Trained end-to-end with binary cross-entropy.

### 4.7 Level Summary

| Level | Representation | Protein? | Params (8M) | Isolated Variable |
|-------|---------------|----------|-------------|-------------------|
| 1a | Morgan FP (1024-bit) | No | 0 | Baseline |
| 1b | MoLFormer mean pool | No | 0 | Semantic repr. vs. classical |
| 1c | MoLFormer attn pool | No | ~461K | + Learned aggregation |
| 3 | ESM-2 + MoLFormer attn pool | Yes | ~543K | + Protein modality |
| 4 | ESM-2 + MoLFormer CNN 2D | Yes | ~337K | + Spatial interactions |

---

## Chapter 5: Canonical Classifier Pipeline

### 5.1 KNN (FAISS)

FAISS inner-product index on L2-normalised features (equivalent to cosine similarity), $k = 5$, distance-weighted voting:

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbb{1}[y_i = c], \quad w_i = \max(\text{sim}(\mathbf{x}, \mathbf{x}_i), 0)$$

### 5.2 MLP Selection

Nine candidate topologies are evaluated via 5-fold stratified CV with 3 restarts per candidate:

| # | Topology | α | η₀ | tol |
|---|----------|---|----|----|
| 1 | (512) | 10⁻² | 10⁻³ | 10⁻⁵ |
| 2 | (512, 256) | 5×10⁻³ | 8×10⁻⁴ | 8×10⁻⁶ |
| 3 | (256, 128) | 5×10⁻³ | 10⁻³ | 10⁻⁵ |
| 4 | (256) | 10⁻² | 10⁻³ | 10⁻⁵ |
| 5 | (128, 64) | 5×10⁻³ | 8×10⁻⁴ | 8×10⁻⁶ |
| 6 | (256, 128) | 5×10⁻² | 5×10⁻⁴ | 10⁻⁵ |
| 7 | (128) | 10⁻¹ | 10⁻³ | 10⁻⁵ |
| 8 | (512, 256, 128) | 10⁻² | 5×10⁻⁴ | 8×10⁻⁶ |
| 9 | (384, 192) | 10⁻² | 8×10⁻⁴ | 10⁻⁵ |

All use ReLU, Adam (adaptive LR), `max_iter = 2000`, patience = 60.

**Selection objective** (stability-adjusted MCC):
$$\text{score}(c) = \overline{\text{MCC}}_c - 0.10 \cdot \sigma_c$$

### 5.3 MLP Ensemble

Best candidate retrained as ensemble of $M = 5$ members with independent seeds:
$$\hat{p}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} \hat{p}_m(\mathbf{x})$$

### 5.4 OOF Threshold Refinement

Decision threshold optimised via out-of-fold predictions on training data, maximising MCC across a dense grid of thresholds.

### 5.5 Two-Phase Protocol

- **Train mode**: Classifiers trained on training split (80%), evaluated on validation (10%). Test set **never loaded**.
- **Test mode**: Classifiers trained on validation (10%), evaluated on held-out test (10%). MLP configuration **frozen** from train phase.

---

## Chapter 6: Evaluation Metrics

### 6.1 Primary Metric: MCC

The **Matthews Correlation Coefficient** is the sole criterion for model comparison:

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Properties: invariant to class proportion, complete statistical interpretation as Pearson correlation between observed and predicted labels, range $[-1, +1]$.

### 6.2 Supporting Metrics

| Metric | Formula |
|--------|---------|
| F1 | $2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| Precision | $\frac{TP}{TP + FP}$ |
| Recall | $\frac{TP}{TP + FN}$ |
| AUC-ROC | Area under the ROC curve |

### 6.3 Multi-Seed Protocol

Every experiment runs across 5 independent seeds: `{42, 123, 456, 789, 1024}`. Results are reported as mean ± std, separating optimisation variance from genuine performance differences.

---

## Chapter 7: Implementation

### 7.1 Software Architecture

- **Entry point**: `semantic_screening_models.py` (thin CLI wrapper)
- **Package**: `benchmark/` (SOLID-compliant, 15 modules)
- **Design patterns**: Template Method, Facade, Frozen Dataclass

### 7.2 Reproducibility

All hyperparameters are configurable via environment variables (see `run_benchmark.sh`). Key defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_MLP_ENSEMBLE` | 5 | Ensemble members |
| `BENCHMARK_MLP_CAL_RESTARTS` | 3 | CV restarts |
| `BENCHMARK_MLP_FOLDS` | 5 | CV folds |
| `BENCHMARK_LEVEL3_HIDDEN_DIM` | auto | Hidden dim (per ESM-2 variant) |

### 7.3 Hardware Requirements

- **Levels 1a–1b**: CPU only (no GPU required)
- **Levels 1c–4**: Single GPU (CUDA or MPS)
- **Embedding generation**: GPU recommended for ESM-2 150M+ models

---

**Last updated**: March 2026 | **Version**: 4.0
