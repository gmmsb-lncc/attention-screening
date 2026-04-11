# semantic-screening: Methodology & Theory

**Author**: Leon Sulfierry (GMMSB-LNCC)
**Date**: March 2026
**Version**: 4.0 (Six-Level Hierarchical Benchmark)

---

## Abstract

semantic-screening is a modular, scalable platform for predicting kinase–ligand bioactivity using foundation language models. It implements a **six-level hierarchical benchmark** that decomposes the sources of predictive gain — from classical Morgan fingerprints to learned bimodal attention pooling and CNN 2D interaction maps — under a **single independent variable** protocol. All non-end-to-end levels share the same canonical KNN/MLP classifiers, scaffold-based splits (Bemis–Murcko), monotonic profile filtering, and multi-seed evaluation (5 seeds). The primary metric is MCC (Matthews Correlation Coefficient). This document details the theoretical foundations, architectural decisions, and evaluation methodology.

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

## Chapter 4: Six-Level Hierarchical Benchmark

### 4.1 Design Principle: Single Independent Variable

The only factor that varies across levels is the **molecular representation**. All levels 1a–3 use the exact same canonical KNN/MLP classifiers, the same scaffold split, the same activity threshold, and the same evaluation metrics.

### 4.2 Level 1a — Fingerprint Baseline

Morgan fingerprints (ECFP, 1024-bit, radius 2) computed via RDKit. Ligand-only, no protein information. **Zero trainable parameters.** This establishes the baseline for classical cheminformatics descriptors:

$$\mathbf{x}^{(1a)} = \text{MorganFP}(L) \in \{0, 1\}^{1024}$$

### 4.3 Level 1b — MoLFormer Mean Pooling

Per-token MoLFormer embeddings aggregated via masked mean pooling. **Zero trainable parameters:**

$$\mathbf{x}^{(1b)} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{h}^L_i \in \mathbb{R}^{768}$$

where $\mathcal{V}$ is the set of valid (non-padding) token indices.

### 4.4 Level 1c — MoLFormer Attention Pooling

Introduces learned aggregation via a single-layer linear projection followed by attention pooling:

1. **Projection**: `Linear(768, 256) → LayerNorm → GELU → Dropout(0.1)` — projects per-token embeddings from 768d to 256d
2. **Attention Pooling** (1 learned query, 8 heads): A single learned query vector $\mathbf{q} \in \mathbb{R}^{256}$ attends over all projected tokens via `MultiheadAttention(embed_dim=256, num_heads=8)`, collapsing the variable-length sequence into a fixed-size summary vector
3. **Auxiliary head** (training-only): Binary classification head (`Linear(2 × 256, 1)`) for gradient signal, discarded after training

**~264K trainable parameters.** Feature vector: $\mathbf{x}^{(1c)} = z_L \in \mathbb{R}^{256}$.

### 4.5 Level 2 — Bimodal Mean Pooling

Per-token ESM-2 and MoLFormer embeddings aggregated independently via masked mean pooling, then concatenated. **Zero trainable parameters:**

$$\mathbf{x}^{(2)} = [\bar{h}_P \| \bar{h}_L] \in \mathbb{R}^{d_P + 768}$$

For ESM-2 8M ($d_P = 320$), this produces a 1088-dimensional vector. This level isolates the value of adding protein information while using the same parameter-free aggregation strategy as Level 1b.

### 4.6 Level 3 — Bimodal Attention Pooling

Same backbone as Level 1c, replicated independently for protein and ligand modalities:

1. **Dual projections**: One per modality — `Linear(d_P, 256) → LayerNorm → GELU → Dropout` for protein, `Linear(768, 256) → LayerNorm → GELU → Dropout` for ligand
2. **Dual attention pooling**: Independent attention pooling (1 learned query, 8 heads) per modality — each modality has its own learned query vector
3. **Auxiliary head** (training-only): Binary classification head (`Linear(2 × 256, 1)`) for gradient signal, discarded after training
4. **Concatenation**: $z^{(3)} = [z_P \| z_L] \in \mathbb{R}^{512}$

No cross-modal interaction features are computed. Concatenation is the sole fusion point.

**~528K trainable parameters** (ESM-2 8M). Feature vector: $\mathbf{x}^{(3)} \in \mathbb{R}^{512}$.

### 4.7 Level 4 — CNN 2D Interaction Maps

End-to-end architecture without downstream KNN/MLP:

1. **K=8 multi-head projection**: Linear projections of protein and ligand embeddings
2. **Interaction maps**: $\mathbf{I}_k = \frac{\mathbf{Z}_P^{(k)} (\mathbf{Z}_L^{(k)})^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$, stacked into $\mathbb{R}^{K \times n \times m}$
3. **4-layer CNN 2D**: Including dilated convolution (dilation=2) for expanded receptive field
4. **Hierarchical attention pooling**: First along ligand axis, then along protein axis
5. **Linear classifier**: Binary output with sigmoid

**~550K trainable parameters** (ESM-2 8M). Trained end-to-end with binary cross-entropy.

### 4.8 Level Summary

| Level | Representation | Protein? | Params (8M) | Feature Dim | Isolated Variable |
|-------|---------------|----------|-------------|-------------|-------------------|
| 1a | Morgan FP (1024-bit) | No | 0 | 1024 | Baseline |
| 1b | MoLFormer mean pool | No | 0 | 768 | Semantic repr. vs. classical |
| 1c | MoLFormer attn pool | No | ~264K | 256 | + Learned aggregation |
| 2 | ESM-2 + MoLFormer mean pool | Yes | 0 | d_P + 768 | + Protein modality |
| 3 | ESM-2 + MoLFormer attn pool | Yes | ~528K | 512 | Bimodal selective aggregation |
| 4 | ESM-2 + MoLFormer CNN 2D | Yes | ~550K | 64 | + Spatial interactions |

### 4.9 Training Protocol (Levels 1c, 3)

- **Optimizer**: AdamW (η = 10⁻⁴, weight decay λ = 0.01)
- **LR schedule**: CosineAnnealingLR (T = 500 epochs)
- **Gradient clipping**: ‖∇‖₂ ≤ 1.0
- **Early stopping**: patience = 5 (monitoring validation loss)
- **Weight initialization**: Xavier uniform (projection), std = 0.02 (attention query)

---

## Chapter 5: Canonical Classifier Pipeline

### 5.1 KNN (FAISS)

FAISS inner-product index on L2-normalised features (equivalent to cosine similarity), $k = 5$, distance-weighted voting:

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbb{1}[y_i = c], \quad w_i = \max(\text{sim}(\mathbf{x}, \mathbf{x}_i), 0)$$

### 5.2 MLP Classifier

A traditional `sklearn.neural_network.MLPClassifier` with fixed architecture:

| Parameter | Value |
|-----------|-------|
| Hidden layers | (256, 128) |
| Activation | ReLU |
| Solver | Adam |
| Learning rate | Adaptive (initial η = 10⁻³) |
| L2 regularisation (α) | 10⁻³ |
| Max iterations | 2000 |
| Early stopping | Yes (patience = 20, 10% validation split) |
| Decision threshold | 0.5 (fixed) |

### 5.3 Two-Phase Protocol

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

### 7.2 Hardware Requirements

- **Levels 1a–1b, 2**: CPU only (no GPU required, no trainable parameters)
- **Levels 1c, 3, 4**: Single GPU (CUDA or MPS)
- **Embedding generation**: GPU recommended for ESM-2 150M+ models

---

## Chapter 8: Comparison with Baseline Architectures

This chapter provides a formal comparison between the DT-Kinase hierarchical benchmark (this work) and two established baselines for compound–protein interaction (CPI) prediction: **DrugBAN** (Bai et al., 2023) and **GraphBAN** (Bai et al., 2023). Both baselines are evaluated under the same experimental protocol (scaffold split, multi-seed evaluation, MCC-calibrated threshold) to ensure methodological parity.

### 8.1 Architectural Overview

All three models address the same binary classification task — predicting whether a compound is active against a target kinase — but differ fundamentally in how they encode, interact, and classify protein–ligand pairs.

| Component | DrugBAN | GraphBAN | DT-Kinase (Level 4) |
|-----------|---------|----------|---------------------|
| **Protein encoder** | CNN 1D (trained from scratch) | CNN 1D + **ESM-1b** (frozen, 650M) | **ESM-2** (frozen, 8M/150M/650M) |
| **Ligand encoder** | GCN (trained from scratch) | GCN + **ChemBERTa** (frozen) | **MoLFormer** (frozen) |
| **Interaction module** | Bilinear Attention Network (BAN) | Graph-based BAN | Scaled dot-product interaction maps |
| **Classifier** | MLP (256→512→128→2) | MLP (256→512→128→2) | Linear (64→1) |
| **Domain adaptation** | CDAN (optional) | CDAN (optional) | Not used |
| **Output** | 2-class softmax | 2-class softmax | Sigmoid (BCE loss) |

### 8.2 Protein and Ligand Encoders

#### 8.2.1 DrugBAN — Task-Specific Encoders

DrugBAN encodes both modalities **from scratch** during training:

- **Protein**: A 1D convolutional network operates on integer-encoded amino acid sequences, with kernels of sizes [3, 6, 9] and 128 filters each, capturing local sequence motifs of varying lengths.
- **Ligand**: A Graph Convolutional Network (GCN) with 3 hidden layers (128-d each) operates on the molecular graph, where atoms are nodes (75-d atomic features) and bonds are edges.

No external pre-trained representations are used. The learned representations are entirely conditioned on the downstream classification objective and the training data.

#### 8.2.2 GraphBAN — Hybrid Encoders with Foundation Models

GraphBAN introduces a critical extension: **fusion of task-specific encoders with frozen foundation language models**:

- **Protein**: The CNN 1D features are fused with **ESM-1b** (Rives et al., 2021; 650M parameters) embeddings via a learned projection $\text{Linear}(1280 \to 128)$ followed by a fusion module (`proFusion`). ESM-1b provides contextualised per-residue embeddings trained on 250M protein sequences via Masked Language Modelling (MLM).
- **Ligand**: The GCN features are fused with **ChemBERTa** (Chithrananda et al., 2020) embeddings, a RoBERTa-based model pre-trained on SMILES strings.

This hybrid approach allows GraphBAN to leverage the transferable knowledge encoded in foundation models while retaining the task-specific inductive biases of GCN and CNN architectures.

#### 8.2.3 DT-Kinase (Level 4) — Foundation Models as Sole Encoders

DT-Kinase employs foundation language models as the **exclusive** source of molecular representations:

- **Protein**: ESM-2 (Lin et al., 2023) provides per-residue embeddings $\mathbf{H}_P \in \mathbb{R}^{n \times d_P}$, where $n$ is the sequence length and $d_P \in \{320, 640, 1280\}$ depends on the model scale.
- **Ligand**: MoLFormer (Ross et al., 2022) provides per-token embeddings $\mathbf{H}_L \in \mathbb{R}^{m \times 768}$, where $m$ is the SMILES token count.

Both models are kept **frozen** — no gradients flow through them. This isolates the representation quality from downstream training dynamics and enables fair comparison across embedding scales (8M, 150M, 650M).

### 8.3 Interaction Mechanisms

The most fundamental architectural difference lies in how each model captures protein–ligand interactions.

#### 8.3.1 Bilinear Attention Network (BAN) — DrugBAN and GraphBAN

Both DrugBAN and GraphBAN employ the Bilinear Attention Network (Kim et al., 2018) to model interactions. Given drug features $\mathbf{d} \in \mathbb{R}^{D}$ and protein features $\mathbf{p} \in \mathbb{R}^{D}$, BAN computes:

$$\mathbf{f}_k = \sigma(\mathbf{d}^\top \mathbf{A}_k \mathbf{p}), \quad k = 1, \ldots, K$$

where $\mathbf{A}_k \in \mathbb{R}^{D \times D}$ is a learned bilinear matrix for each of $K$ attention heads, and $\sigma$ is a non-linear activation. The outputs are concatenated to form a fixed-size interaction vector $\mathbf{f} = [\mathbf{f}_1 \| \ldots \| \mathbf{f}_K] \in \mathbb{R}^{256}$.

This produces a **global summary** of the interaction — a single vector per protein–ligand pair. The spatial (residue-level) information is lost during pooling.

#### 8.3.2 Scaled Dot-Product Interaction Maps — DT-Kinase

Level 4 preserves full **residue-level spatial resolution**. For each of $K=8$ heads, independent linear projections map both modalities into a shared $d_k$-dimensional space:

$$\mathbf{Z}_P^{(k)} = \mathbf{H}_P \mathbf{W}_P^{(k)} \in \mathbb{R}^{n \times d_k}, \quad \mathbf{Z}_L^{(k)} = \mathbf{H}_L \mathbf{W}_L^{(k)} \in \mathbb{R}^{m \times d_k}$$

The interaction map for head $k$ is the scaled dot product:

$$\mathbf{I}_k = \frac{\mathbf{Z}_P^{(k)} (\mathbf{Z}_L^{(k)})^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$$

The $K$ maps are stacked to form a multi-channel 2D tensor $\mathbf{I} \in \mathbb{R}^{K \times n \times m}$, where each entry $(k, i, j)$ quantifies the semantic compatibility between residue $i$ and token $j$ under the $k$-th projection head.

This representation is analogous to a multi-channel image, where:
- The spatial dimensions $(n \times m)$ encode all-vs-all residue–token comparisons
- The channel dimension $(K)$ encodes $K$ learned "perspectives" on the interaction

### 8.4 Classification Architecture

#### 8.4.1 MLP Classifier — DrugBAN and GraphBAN

Both baselines pass the BAN interaction vector through a multi-layer perceptron:

$$\hat{y} = \text{softmax}(\text{MLP}(\mathbf{f}))$$

where the MLP has architecture: $\text{Linear}(256 \to 512) \to \text{ReLU} \to \text{Linear}(512 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 2)$.

The MLP has $\sim$200K trainable parameters dedicated solely to classification — representing significant additional capacity beyond the interaction module.

#### 8.4.2 CNN 2D + Linear Classifier — DT-Kinase

Level 4 processes the interaction maps through a 4-layer 2D CNN that extracts local spatial patterns:

$$\mathbf{F} = \text{CNN}_\text{2D}(\mathbf{I}) \in \mathbb{R}^{C \times n \times m}, \quad C = 64$$

The CNN includes dilated convolution (layer 3, dilation=2) to expand the effective receptive field without increasing parameter count. Feature maps are then aggregated via **hierarchical attention pooling**:

1. **Ligand axis**: For each protein position $i$, attention-weighted pooling across $m$ ligand positions → $\mathbb{R}^{n \times C}$
2. **Protein axis**: Attention-weighted pooling across $n$ protein positions → $\mathbb{R}^C$

The final classifier is a single linear layer:

$$\hat{y} = \sigma(\mathbf{w}^\top \mathbf{z} + b), \quad \mathbf{w} \in \mathbb{R}^{64}, \; b \in \mathbb{R}$$

This represents the minimal possible classifier — a single neuron — in contrast to the deep MLP used by DrugBAN and GraphBAN. The classificatory capacity is delegated to the CNN feature extraction and attention pooling, rather than to the final decision layer.

### 8.5 Domain Adaptation

Both DrugBAN and GraphBAN employ **Conditional Domain Adversarial Networks (CDAN)** (Long et al., 2018) to improve generalisation under distribution shift between training and test scaffolds:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{cls} + \lambda \cdot \mathcal{L}_\text{domain}$$

where $\mathcal{L}_\text{domain}$ is the cross-entropy of a domain discriminator (MLP) trained to distinguish source (training) from target (validation) representations. A **Gradient Reversal Layer (GRL)** inverts the gradient of $\mathcal{L}_\text{domain}$ during backpropagation, forcing the encoder to learn domain-invariant features:

$$\text{GRL}(x) = x \quad \text{(forward)}, \qquad \frac{\partial \text{GRL}}{\partial x} = -\lambda \cdot \mathbf{I} \quad \text{(backward)}$$

Shared configuration: `RANDOM_LAYER=True`, `RANDOM_DIM=256`, `INIT_EPOCH=10`, $\lambda = 1.0$.

DT-Kinase (Level 4) does **not** employ domain adaptation; generalisation is addressed exclusively through scaffold-based splitting and the transferable representations of frozen foundation models.

### 8.6 Summary of Key Differentiators

| Aspect | DrugBAN / GraphBAN | DT-Kinase (Level 4) |
|--------|-------------------|---------------------|
| **Representation source** | Task-specific (DrugBAN) or hybrid (GraphBAN) | Foundation models only |
| **Interaction granularity** | Global vector (one per pair) | Residue × token matrix (all-vs-all) |
| **Classifier complexity** | Deep MLP (~200K params) | Single linear layer (65 params) |
| **Spatial information** | Lost during BAN pooling | Preserved through CNN 2D |
| **Domain adaptation** | CDAN adversarial training | Not used |
| **Interpretability** | BAN attention weights | Interaction maps visualisable as heatmaps |

The DT-Kinase architecture is deliberately designed to shift classification capacity from the final decision layer (MLP) to the intermediate representation (interaction maps + CNN). This design enables direct visualisation of which protein residues interact with which ligand fragments — providing mechanistic interpretability that is absent in MLP-based classifiers.

---

**Last updated**: March 2026 | **Version**: 4.1
