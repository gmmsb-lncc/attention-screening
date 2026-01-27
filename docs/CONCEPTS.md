# Fundamental Concepts: semantic-screening and DT-Kinase

## 📌 Definitions

### semantic-screening
**Open and extensible platform** for predicting protein-ligand interaction properties using deep learning based on protein language models.

**Scope**:
- Complete pipeline implementation: embeddings → processing → prediction
- Support for multiple strategies:
  - **Classical models**: 12 ML algorithms (classification + regression)
  - **Neural architecture**: DT-Kinase (CNN + Cross-Attention)
  - **Embedding models**: ESM-2, ESM-3/ESM-C (protein), SMI-TED, MoLFormer (ligand)
- Robust stratification with leakage validation
- Modular and reusable for new datasets, models, and approaches

**Analogy**: semantic-screening is like a "toolkit" – it provides reusable components and patterns for building screening solutions.

---

### DT-Kinase
**Specific neural architecture** implemented in the semantic-screening platform that solves the kinase selectivity prediction problem through semantic reformulation.

**Components**:
1. **Protein Encoding**: Per-residue contextual embeddings from protein language models (ESM-2, ESM-3/ESM-C)
   - Capture implicit evolutionary information in sequence
   - Do not require experimental 3D structure

2. **Ligand Encoding**: Per-atom embeddings from chemical foundation models (SMI-TED, MoLFormer)
   - Capture molecular properties and SMILES syntax
   - Encode 2D/semantic structure patterns

3. **Local Feature Extraction**: Multi-scale CNN encoders
   - Kernels {3, 5, 7} to capture patterns at multiple scales
   - Residual connections preserve feature hierarchy

4. **Semantic Interaction Modeling**: Bidirectional Cross-Attention mechanisms
   - Protein → Ligand: "Which residues bind to which atoms?"
   - Ligand → Protein: "Which atoms interact with which residues?"
   - Multi-Head (8 heads) to capture different interaction types

5. **Multi-Task Prediction**:
   - **Classification**: Active/Inactive (binary logits)
   - **Regression**: Affinity value in pChEMBL scale (continuous)
   - Joint optimization with task weighting

**Analogy**: DT-Kinase is a specific architecture – just as "AlexNet" is a specific CNN architecture within the broader deep learning ecosystem.

---

## 🔗 Conceptual Relationship

```
┌────────────────────────────────────────────────────────────────┐
│                 SEMANTIC-SCREENING                             │
│        (Open semantic screening platform)                      │
│                                                                │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  ML CLASSICALS   │         │   DT-KINASE      │             │
│  │  12 algorithms   │         │   (CNN + Cross-  │             │
│  │  (RF, XGB, etc.) │         │    Attention)    │             │
│  └──────────────────┘         └──────────────────┘             │
│                                                                │
│  ┌──────────────────────────────────────────────────┐          │
│  │  EMBEDDING INFRASTRUCTURE                        │          │
│  │  • ESM-2 / ESM-3 (ESM-C) (Protein)              │          │
│  │  • SMI-TED / MoLFormer (Ligand)                 │          │
│  │  • Cached embeddings & validation                │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                │
│  ┌──────────────────────────────────────────────────┐          │
│  │  STRATIFICATION & VALIDATION                     │          │
│  │  • Agglomerative clustering                      │          │
│  │  • Cosine similarity validation                  │          │
│  │  • Train/Val/Test splitting                      │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 When to Use Each

### semantic-screening
Use when you want:
- A **complete modular platform** for protein-ligand interaction screening
- **Explore multiple approaches**: compare classical ML vs deep learning
- **Customization**: add new models, embeddings, or strategies
- **Production**: validated scalability and robustness
- **Investigation**: understand which components affect performance

### DT-Kinase (within semantic-screening)
Use when you want:
- **Leverage semantic information** from proteins and molecules via PLMs
- **Model explainable interactions** with attention mechanisms
- **Optimized performance**: CNN captures local, Cross-Attention captures interactions
- **No 3D structure**: applicable to any protein with known sequence
- **Multi-task**: simultaneous classification and regression with uncertainty

---

## 📊 Use Cases

### Example 1: Bacterial Kinase Inhibitor Discovery
**Scenario**: You have a dataset of 15K molecules against 42 bacterial kinases with affinity data.

**semantic-screening + DT-Kinase approach**:
1. Generate ESM-C embeddings for 42 kinases (once)
2. Generate MoLFormer embeddings for 15K molecules
3. Train DT-Kinase for multi-task prediction
4. Use classical ML models as baselines
5. Compare: DT-Kinase vs 12 ML algorithms
6. Robustly stratify with leakage validation

**Result**: Neural architecture specialized for your problem + validation against multiple baselines.

---

### Example 2: Ultra-Large Chemical Library Screening
**Scenario**: You have 1B molecules and want to predict activity against 100 proteins.

**semantic-screening + DT-Kinase approach**:
1. Train DT-Kinase once on benchmark dataset
2. Generate embeddings for 100 proteins (reusable cache)
3. Process 1B molecules in batches (pure forward pass = fast)
4. Rank candidates by predicted affinity score

**Result**: Screening of billions of compounds in hours, no 3D structure, with uncertainties.

---

### Example 3: New Target Without Crystal Structure
**Scenario**: New target with annotated sequence but no PDB structure.

**semantic-screening + DT-Kinase approach**:
1. PLM (ESM-2) reconstructs local structure implicitly in embeddings
2. DT-Kinase doesn't need explicit 3D
3. Immediately applicable to "orphaned" targets

**Result**: Functional screening without crystallography.

---

## 🔬 Theoretical Foundation

### Why semantic-screening?
- **Sequence contains structural information**: Demonstrated by AlphaFold2, ESMFold, ProtBERT
- **PLMs learn semantics**: Via self-attention on hundreds of millions of sequences
- **Problem reformulation**: Not "What is the geometry?" but "How compatible is the semantics?"
- **Universality**: Applicable to any protein with sequence, structure or not

### Why DT-Kinase as a specific architecture?
- **CNN**: Captures local patterns in sequences and molecules
- **Cross-Attention**: Models semantic compatibility between protein and ligand
- **Multi-Task**: Classification + Regression with task weighting
- **Scalable**: Pure forward pass, no geometric bottleneck

---

## 📚 References in Thesis

- **Chapter 1, Section 1.3**: "From Docking to Language Modeling: The DockTKinase Philosophy"
  - Explains why to abandon 3D representations
  - Establishes PLM usage

- **Chapter 1, Section 1.5**: "DT-Kinase: Objectives and Contributions"
  - Defines DT-Kinase as proposed architecture
  - Specifies components: embeddings + CNN + Cross-Attention

- **Chapter 2**: State of the art
  - Docking limitations
  - Experimental panel limitations
  - Need for semantic approach

- **Chapter 3**: Theoretical foundations
  - Mathematical formulation of PLMs
  - Cross-attention
  - Proposed architecture

---

## ✅ Conceptual Checklist

- [ ] semantic-screening = Open and modular platform
- [ ] DT-Kinase = Specific neural architecture within semantic-screening
- [ ] semantic-screening implements multiple approaches (classical ML + DL)
- [ ] DT-Kinase is optimized for kinases and protein-ligand interactions
- [ ] Both operate without requiring 3D structure
- [ ] semantic-screening provides embedding infrastructure, validation, stratification
- [ ] DT-Kinase provides CNN + Cross-Attention architecture for interactions

---

**Last updated**: January 2026
**Document**: Conceptual clarification
**Status**: Reference for all documentation
