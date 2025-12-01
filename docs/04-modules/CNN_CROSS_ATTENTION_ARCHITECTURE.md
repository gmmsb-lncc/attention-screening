# CNN + Cross-Attention Architecture for Protein-Ligand Affinity Prediction

## Overview

This document describes the deep learning architecture used in DockTKinase for protein-ligand binding affinity prediction. The architecture combines **Convolutional Neural Networks (CNN)** with **Cross-Attention mechanisms** to model interactions between protein sequences and small molecule ligands.

## Table of Contents

1. [Architecture Rationale](#architecture-rationale)
2. [CNN Encoder](#cnn-encoder)
3. [Cross-Attention Mechanism](#cross-attention-mechanism)
4. [Optimized CNN Encoder](#optimized-cnn-encoder)
5. [Scientific References](#scientific-references)

---

## Architecture Rationale

### Why CNN + Cross-Attention?

Protein-ligand binding prediction requires understanding **two types of patterns**:

| Pattern Type | What it Captures | Best Approach |
|--------------|------------------|---------------|
| **Local patterns** | Sequence motifs, functional groups | CNN |
| **Interaction patterns** | Which residues bind to which atoms | Cross-Attention |

### The Complementary Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Pre-computed Embeddings from Foundation Models                      │
│                                                                             │
│  Protein: ESM-2 embeddings [batch, seq_len, 2560]                          │
│           • Already captures: evolutionary conservation, structure          │
│           • Already captures: long-range dependencies via self-attention    │
│                                                                             │
│  Ligand: SMI-TED embeddings [batch, seq_len, 768]                          │
│           • Already captures: molecular properties, functional groups       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CNN ENCODER: Extract Complementary Local Features                          │
│                                                                             │
│  Purpose: Detect LOCAL MOTIFS with translation invariance                  │
│                                                                             │
│  Why CNN and not more Transformers?                                        │
│  1. ESM-2 already provides global context via self-attention               │
│  2. CNNs have strong inductive bias for local pattern detection            │
│  3. CNNs are more parameter-efficient for local feature extraction         │
│  4. Translation invariance: same motif detected anywhere in sequence       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CROSS-ATTENTION: Model Protein-Ligand Interactions                        │
│                                                                             │
│  Purpose: Learn WHICH protein residues interact with WHICH ligand atoms    │
│                                                                             │
│  Key insight: Binding is inherently a cross-modal interaction              │
│  • Protein and ligand are different entities                               │
│  • Cross-attention explicitly models their relationship                    │
│  • Attention weights are interpretable (show binding site predictions)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MULTI-TASK HEAD: Classification + Regression                               │
│                                                                             │
│  • Classification: Active vs Inactive (binary)                             │
│  • Regression: pChEMBL value prediction (continuous)                       │
│                                                                             │
│  Multi-task learning provides implicit regularization                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CNN Encoder

### Standard Architecture

The CNN encoder processes per-token embeddings to extract local patterns:

```
Input: [batch, seq_len, embed_dim]
            │
            ▼
    ┌───────────────────┐
    │  Linear Projection│  embed_dim → hidden_dim (256)
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  Conv1D Block 1   │  kernel=3, receptive field: 5
    │  + Residual       │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  Conv1D Block 2   │  kernel=5, receptive field: 13
    │  + Residual       │
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │  Conv1D Block 3   │  kernel=7, receptive field: 25
    │  + Residual       │
    └───────────────────┘
            │
            ▼
Output: [batch, seq_len, hidden_dim]
```

### Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Multi-scale kernels (3,5,7)** | Progressive receptive field | Captures motifs of different sizes |
| **Residual connections** | Gradient flow | Enables deeper networks (He et al., 2016) |
| **BatchNorm** | Training stability | Normalizes activations per batch |
| **GELU activation** | Smooth gradients | Better than ReLU for NLP (Hendrycks & Gimpel, 2016) |
| **2 convs per block** | VGG-style depth | More expressive without excessive parameters |

### What Local Patterns Are Captured?

For proteins:
- **Dipeptide motifs** (kernel=3): Adjacent amino acid preferences
- **Secondary structure elements** (kernel=5-7): Alpha helix turns, beta sheet strands
- **Local binding pockets** (kernel=7+): Short loop conformations

For ligands:
- **Functional groups** (kernel=3): Hydroxyl, amino, carbonyl
- **Ring systems** (kernel=5): Aromatic rings, heterocycles
- **Pharmacophore features** (kernel=7): H-bond donors/acceptors patterns

---

## Cross-Attention Mechanism

### How Cross-Attention Works

Cross-attention allows one sequence (protein) to "look at" another sequence (ligand):

```
Protein Features: [batch, protein_len, hidden_dim] ─┐
                                                    │
                                                    ▼
                              ┌────────────────────────────────────────┐
                              │           CROSS-ATTENTION              │
                              │                                        │
                              │  Query (Q) = Protein features          │
                              │  Key (K)   = Ligand features           │
                              │  Value (V) = Ligand features           │
                              │                                        │
                              │         Q × Kᵀ                         │
                              │  Attn = ─────── → softmax              │
                              │          √d                            │
                              │                                        │
                              │  Output = Attn × V                     │
                              │                                        │
                              │  "Which ligand atoms matter for        │
                              │   each protein residue?"               │
                              │                                        │
                              └────────────────────────────────────────┘
                                                    │
Ligand Features: [batch, ligand_len, hidden_dim] ───┘
```

### Bidirectional Cross-Attention

We use **bidirectional** cross-attention in each block:

1. **Protein → Ligand**: Each protein residue attends to ligand atoms
   - "Which ligand atoms are relevant to this residue?"
   
2. **Ligand → Protein**: Each ligand atom attends to protein residues
   - "Which protein residues interact with this atom?"

### Multi-Head Attention

We use **8 attention heads**, allowing the model to capture different interaction types:

| Head | Potential Specialization |
|------|--------------------------|
| 1-2 | Hydrogen bonding patterns |
| 3-4 | Hydrophobic contacts |
| 5-6 | Electrostatic interactions |
| 7-8 | Shape complementarity |

*Note: Head specialization emerges during training; the above is illustrative.*

### Attention Visualization

The attention weights can be visualized to understand model predictions:

```
                    Ligand Atoms
                 1  2  3  4  5  6  7  8  9  10
              ┌─────────────────────────────────┐
           45 │ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │
           50 │ ·  ·  ·  ▓  ▓  █  █  ▓  ·  ·  │ ← Active site
Protein    55 │ ·  ·  ▓  █  █  █  █  █  ▓  ·  │    residues
Residues   60 │ ·  ·  ·  ▓  ▓  █  █  ▓  ·  ·  │
           65 │ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │
           70 │ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │
              └─────────────────────────────────┘
                          ↑
                    Key functional group

█ = high attention  ▓ = medium  · = low
```

---

## Optimized CNN Encoder

### Motivation for Optimization

The standard CNN encoder works well but has room for improvement:

| Issue | Impact | Solution |
|-------|--------|----------|
| High parameter count | Overfitting on small datasets | Depthwise Separable Conv |
| Fixed receptive field | May miss some patterns | Dilated Convolutions |
| Uniform channel weights | Suboptimal feature selection | Squeeze-and-Excitation |
| Post-LayerNorm | Training instability | Pre-LayerNorm |

### Optimized Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  OPTIMIZED CNN ENCODER                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [batch, seq_len, embed_dim]                                        │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Input Projection + LayerNorm + GELU                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OptimizedConv1DBlock (dilation=1)                                  │   │
│  │  ├── Pre-LayerNorm (Xiong et al., 2020)                             │   │
│  │  ├── Depthwise Separable Conv (Chollet, 2017)                       │   │
│  │  ├── Squeeze-and-Excitation (Hu et al., 2018)                       │   │
│  │  └── Residual Connection                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OptimizedConv1DBlock (dilation=2)                                  │   │
│  │  Cumulative receptive field: 9                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OptimizedConv1DBlock (dilation=4)                                  │   │
│  │  Cumulative receptive field: 17                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                      │                                                      │
│                      ▼                                                      │
│  Output: [batch, seq_len, hidden_dim]                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Depthwise Separable Convolution (Chollet, 2017)

Factorizes standard convolution into two steps:

```
Standard Conv:              Depthwise Separable:
C_in × C_out × K            C_in × 1 × K (spatial)
= 196,608 params            + C_in × C_out × 1 (channel mixing)
                            = 66,304 params

Parameter reduction: ~3x
```

**Why it works for embeddings:** The dimensions of ESM-2 embeddings encode different biological properties (structure, conservation, etc.). Depthwise separable convolution processes spatial patterns first, then mixes channels—matching the structure of the data.

#### 2. Squeeze-and-Excitation (Hu et al., 2018)

Learns channel importance adaptively:

```
Input: [batch, channels, seq_len]
           │
           ▼
    ┌──────────────┐
    │ Global Pool  │  Squeeze: aggregate spatial info
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  FC → ReLU   │  Excitation: learn importance
    │  FC → Sigmoid│
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   Scale      │  Multiply input by importance weights
    └──────────────┘
```

**Why it works for embeddings:** Different dimensions of ESM-2 encode different properties. SE blocks learn which dimensions are most predictive for binding affinity—effectively performing feature selection.

#### 3. Pre-LayerNorm (Xiong et al., 2020)

Places normalization before the transformation:

```
Post-norm (original):        Pre-norm (improved):
    x ────────────┐              x ────────────┐
    │             │              │             │
    ▼             │              ▼             │
  Conv            │           LayerNorm        │
    │             │              │             │
    ▼             │              ▼             │
 LayerNorm        │            Conv            │
    │             │              │             │
    └───── + ◄────┘              └───── + ◄────┘
```

**Why it matters:** With pre-norm, gradients flow directly through the residual path without transformation. This enables:
- Larger learning rates
- Faster convergence
- More stable training

### Performance Comparison

| Metric | Standard CNN | Optimized CNN | Improvement |
|--------|--------------|---------------|-------------|
| **Parameters** | 2,626,816 | 1,155,840 | **-56%** |
| **Receptive Field** | 25 | 29 | **+16%** |
| **Training Stability** | Good | Better | Pre-LayerNorm |
| **Channel Adaptation** | None | SE blocks | Automatic |

---

## Scientific References

### CNN Architecture

1. **He, K., et al. (2016)**. *Deep Residual Learning for Image Recognition*. CVPR.
   - Foundation for residual connections in deep networks

2. **Chollet, F. (2017)**. *Xception: Deep Learning with Depthwise Separable Convolutions*. CVPR.
   - Depthwise separable convolutions for efficiency

3. **Hu, J., et al. (2018)**. *Squeeze-and-Excitation Networks*. CVPR.
   - Channel recalibration via SE blocks

4. **Xiong, R., et al. (2020)**. *On Layer Normalization in the Transformer Architecture*. ICML.
   - Pre-LayerNorm for training stability

### Cross-Attention

5. **Vaswani, A., et al. (2017)**. *Attention Is All You Need*. NeurIPS.
   - Foundation for attention mechanisms

6. **Tsai, Y.-H. H., et al. (2019)**. *Multimodal Transformer for Unaligned Multimodal Language Sequences*. ACL.
   - Cross-modal attention for different input types

### Protein-Ligand Modeling

7. **Rives, A., et al. (2021)**. *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. PNAS.
   - ESM-2 protein language model

8. **Rao, R., et al. (2019)**. *Evaluating Protein Transfer Learning with TAPE*. NeurIPS.
   - Transfer learning for protein representations

---

## Usage

### Standard CNN Encoder

```python
from src.classifier.models.cnn_encoder import CNNEncoder

encoder = CNNEncoder(
    input_dim=2560,      # ESM-2 3B dimension
    hidden_dim=256,
    num_layers=3,
    kernel_sizes=(3, 5, 7),
    dropout=0.1
)

# Forward pass
protein_emb = torch.randn(32, 500, 2560)  # [batch, seq_len, dim]
features = encoder(protein_emb)            # [32, 500, 256]
```

### Optimized CNN Encoder

```python
from src.classifier.models.cnn_encoder import OptimizedCNNEncoder

encoder = OptimizedCNNEncoder(
    input_dim=2560,
    hidden_dim=256,
    num_layers=3,
    kernel_size=3,
    dilations=(1, 2, 4),
    dropout=0.1,
    use_se=True
)

# Same interface
features = encoder(protein_emb)  # [32, 500, 256]

# Check configuration
print(f"Parameters: {encoder.count_parameters():,}")
print(f"Receptive field: {encoder.receptive_field}")
```

### Factory Function

```python
from src.classifier.models.cnn_encoder import create_encoder

# Standard encoder
encoder = create_encoder(input_dim=2560, optimized=False)

# Optimized encoder
encoder = create_encoder(input_dim=2560, optimized=True)
```

---

## Summary

The CNN + Cross-Attention architecture is designed for protein-ligand affinity prediction with:

1. **CNN Encoders**: Extract local patterns (motifs) from pre-trained embeddings
2. **Cross-Attention**: Model bidirectional protein-ligand interactions
3. **Multi-Task Head**: Predict both classification and regression targets

The **optimized encoder** reduces parameters by 56% while improving receptive field and training stability, making it suitable for smaller datasets common in drug discovery.
