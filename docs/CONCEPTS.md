# Fundamental Concepts: semantic-screening

## 📌 What is semantic-screening?

**semantic-screening** is an open-source platform for predicting protein–ligand bioactivity using foundation language models. It implements a **five-level hierarchical benchmark** that decomposes the sources of predictive gain — from classical fingerprints to learned bimodal interaction modeling — under a single, rigorously controlled experimental protocol.

**Core hypothesis**: Sequence determines structure, which determines function — and this mapping is computationally recoverable via pre-trained language models. By leveraging the contextual embeddings from protein (ESM-2) and molecular (MoLFormer) language models, semantic-screening predicts interaction compatibility in latent space rather than through geometric fitting.

---

## 🏗️ The Five-Level Benchmark

The benchmark evaluates five levels of increasing representational complexity. The **only variable** across levels is the molecular representation; classifiers (KNN/MLP) are held constant.

```
Complexity increases monotonically:

  Level 1a:  Morgan Fingerprints (1024-bit)          → KNN/MLP    [0 params]
  Level 1b:  MoLFormer mean pooling                  → KNN/MLP    [0 params]
  Level 1c:  MoLFormer + ResProj + attention pooling  → KNN/MLP    [~461K params]
  Level 3:   ESM-2 + MoLFormer + bimodal attention    → KNN/MLP    [~543K params]
  Level 4:   ESM-2 + MoLFormer + CNN 2D interaction   → end-to-end [~337K params]
```

### Key Scientific Questions Answered

| Transition | Question |
|-----------|----------|
| **1a → 1b** | Do semantic pre-trained representations outperform classical fingerprints? |
| **1b → 1c** | Does learned selective aggregation (attention) outperform uniform mean pooling? |
| **1c → 3** | Does adding protein modality and interaction features improve prediction? |
| **3 → 4** | Does explicit spatial residue–atom interaction modeling help? |

---

## 🧬 Foundation Models Used

Both models operate as **frozen feature extractors** — weights are never updated during benchmark training.

### Protein Encoder: ESM-2 (Meta AI)
Bidirectional transformer trained with Masked Language Modeling (MLM) on UniRef50.

| Variant | Params | Embedding Dim |
|---------|--------|---------------|
| `esm2_t6_8M_UR50D` | 8M | 320 |
| `esm2_t30_150M_UR50D` | 150M | 640 |
| `esm2_t33_650M_UR50D` | 650M | 1280 |

### Ligand Encoder: MoLFormer (IBM Research)
Transformer encoder with linear attention, pre-trained on 1.1B molecules (ZINC + PubChem).

| Variant | Params | Embedding Dim |
|---------|--------|---------------|
| MoLFormer-XL | 47M | 768 |

Embeddings are pre-computed once and stored as `.npy` matrices, amortized across all epochs, seeds, and levels.

---

## 🔬 Canonical Classifier Pipeline

All non-end-to-end levels (1a–3) share the **exact same** classifier pipeline:

- **KNN**: FAISS cosine similarity, *k* = 5, distance-weighted voting
- **MLP**: 9-candidate topology search via 5-fold stratified CV × 3 restarts, stability-adjusted MCC objective, ensemble of 5 members, OOF threshold refinement
- **Metric**: MCC (Matthews Correlation Coefficient) — the primary evaluation metric
- **Multi-seed**: 5 seeds `{42, 123, 456, 789, 1024}` → mean ± std

---

## 🔒 Data Integrity

- **Scaffold split**: Bemis–Murcko scaffold decomposition prevents chemical series leakage
- **Universal partition**: Computed once over the combined Human + Non-Human corpus
- **Monotonic filtering**: Removes trivially predictable entities (all-active/all-inactive)
- **Two-phase protocol**: Train → Test with frozen MLP selection (no information leakage from test to model selection)

---

## 📊 Architecture Levels in Detail

### Level 1a — Fingerprint Baseline
Morgan fingerprints (1024-bit, radius 2). Ligand-only, no protein information. Zero trainable parameters.

### Level 1b — MoLFormer Mean Pooling
Per-token MoLFormer embeddings → mean pooling → 768-dim vector. Tests whether semantic representations outperform fingerprints.

### Level 1c — MoLFormer Attention Pooling
Per-token MoLFormer embeddings → 2-layer residual projection → multi-query attention pooling (4 learned queries, 8 heads) → 256-dim vector. Tests whether selective aggregation beats uniform pooling.

### Level 3 — Bimodal Attention Pooling
ESM-2 protein + MoLFormer ligand → residual projection per modality → multi-query attention pooling per modality → explicit interaction features (element-wise product, absolute difference, cosine similarity) → 2-layer auxiliary MLP → 1282-dim feature vector. Tests the value of protein information.

### Level 4 — CNN 2D Interaction Maps
K=8 multi-head linear projections → scaled dot-product interaction maps → 4-layer CNN 2D (including dilated convolution) → hierarchical attention pooling → linear classifier. End-to-end training. Tests explicit spatial interaction modeling.

---

## 📚 Relationship to the Thesis

- **Chapter 2**: State of the art — kinase selectivity, experimental panels, computational methods
- **Chapter 3**: Theoretical foundations — Transformers, ESM-2, MoLFormer, attention pooling, scaffold splitting, evaluation metrics
- **Chapter 4**: Methodology — five-level benchmark, data curation, model architectures, training protocols

---

**Last updated**: March 2026
**Status**: Reference for all documentation
