# Fundamental Concepts: attention-screening

## 📌 What is attention-screening?

**attention-screening** is an open-source platform for predicting protein–ligand bioactivity using foundation language models. It implements a **six-level hierarchical benchmark** that decomposes the sources of predictive gain — from classical fingerprints to learned bimodal attention pooling and 2D interaction maps — under a single, rigorously controlled experimental protocol.

**Core hypothesis**: Sequence determines structure, which determines function — and this mapping is computationally recoverable via pre-trained language models. By leveraging the contextual embeddings from protein (ESM-2) and molecular (MoLFormer) language models, attention-screening predicts interaction compatibility in latent space rather than through geometric fitting.

---

## 🏗️ The Six-Level Benchmark

The benchmark evaluates six levels of increasing representational complexity. The **only variable** across levels is the molecular representation; classifiers (KNN/MLP) are held constant.

```
Complexity increases monotonically:

  Level 1a:  Morgan Fingerprints (1024-bit)          → KNN/MLP    [0 params]
  Level 1b:  MoLFormer mean pooling                  → KNN/MLP    [0 params]
  Level 1c:  MoLFormer + Proj + attention pooling     → KNN/MLP    [~264K params]
  Level 2:   ESM-2 + MoLFormer mean pooling           → KNN/MLP    [0 params]
  Level 3:   ESM-2 + MoLFormer + bimodal attn pool    → KNN/MLP    [~528K params]
  Level 4:   ESM-2 + MoLFormer + CNN 2D interaction   → end-to-end [~550K params]
```

### Key Scientific Questions Answered

| Transition | Question |
|-----------|----------|
| **1a → 1b** | Do semantic pre-trained representations outperform classical fingerprints? |
| **1b → 1c** | Does learned selective aggregation (attention) outperform uniform mean pooling? |
| **1b → 2** | Does adding protein information improve prediction (same aggregation strategy)? |
| **2 → 3** | Does learned bimodal aggregation outperform raw bimodal mean pooling? |
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
- **MLP**: `MLPClassifier(256, 128)`, ReLU activation, Adam solver (η=10⁻³), α=10⁻³, adaptive LR, early stopping (patience=20), max 2000 iterations, decision threshold = 0.5
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
Per-token MoLFormer embeddings → masked mean pooling → 768-dim vector. Tests whether semantic representations outperform fingerprints.

### Level 1c — MoLFormer Attention Pooling
Per-token MoLFormer embeddings → linear projection (`Linear → LayerNorm → GELU → Dropout`, 768 → 256) → attention pooling with a single learned query and 8 attention heads → 256-dim vector. Tests whether selective aggregation beats uniform pooling.

### Level 2 — Bimodal Mean Pooling
ESM-2 protein + MoLFormer ligand → masked mean pooling per modality → concatenation → (d_P + 768)-dim vector. Zero trainable parameters. Isolates the value of adding protein information with the same aggregation strategy as Level 1b.

### Level 3 — Bimodal Attention Pooling
ESM-2 protein + MoLFormer ligand → linear projection per modality (to 256d) → independent attention pooling per modality (1 learned query, 8 heads) → concatenation → 512-dim vector. Same backbone as Level 1c, replicated for both modalities. No cross-modal interaction features.

### Level 4 — CNN 2D Interaction Maps
K=8 multi-head linear projections → scaled dot-product interaction maps → 4-layer CNN 2D (including dilated convolution) → hierarchical attention pooling → linear classifier. End-to-end training. Tests explicit spatial interaction modeling.

---

## 📚 Relationship to the Thesis

- **Chapter 2**: State of the art — kinase selectivity, experimental panels, computational methods
- **Chapter 3**: Theoretical foundations — Transformers, ESM-2, MoLFormer, attention pooling, scaffold splitting, evaluation metrics
- **Chapter 4**: Methodology — six-level benchmark, data curation, model architectures, training protocols

---

**Last updated**: March 2026
**Status**: Reference for all documentation
