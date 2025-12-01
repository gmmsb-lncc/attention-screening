# CNN + Cross-Attention Architecture for Protein-Ligand Affinity Prediction

**Authors**: DockTKinase Development Team  
**Last Updated**: December 2025  
**Version**: 2.1

## Abstract

This document describes the deep learning architecture used in DockTKinase for protein-ligand binding affinity prediction. We present a hybrid **Convolutional Neural Network (CNN) + Cross-Attention** architecture that combines the local feature extraction capabilities of CNNs with the relational modeling power of attention mechanisms. The architecture processes per-token embeddings from protein language models (ESM-2; Lin et al., 2023) and molecular encoders (SMI-TED; Ross et al., 2022), producing binding affinity predictions through learned protein-ligand interaction patterns.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Sequence Length Handling and Positional Encoding](#sequence-length-handling-and-positional-encoding)
3. [Architecture Overview](#architecture-overview)
4. [CNN Encoder](#cnn-encoder)
5. [Cross-Attention Mechanism](#cross-attention-mechanism)
6. [Multi-Task Learning](#multi-task-learning)
7. [Optimized CNN Encoder](#optimized-cnn-encoder)
8. [Implementation Details](#implementation-details)
9. [Scientific References](#scientific-references)

---

## Theoretical Background

### The Protein-Ligand Binding Problem

Protein-ligand binding is governed by the free energy of binding, which can be expressed as:

$$\Delta G_{bind} = \Delta H - T\Delta S = -RT \ln K_d$$

where $K_d$ is the dissociation constant. In the pChEMBL scale used for our regression target:

$$pChEMBL = -\log_{10}(K_d / 1M) = -\log_{10}(K_d) + 9$$

The binding interaction can be decomposed into pairwise atomic contributions (Eldridge et al., 1997):

$$\Delta G_{bind} \approx \sum_{i \in P} \sum_{j \in L} w_{ij} \cdot f(r_i, l_j, d_{ij})$$

where $r_i$ represents protein residue $i$, $l_j$ represents ligand atom $j$, $d_{ij}$ is their distance, and $f(\cdot)$ captures interaction features. Our cross-attention mechanism learns an approximation of the weighting function $w_{ij}$ without explicit structural information.

### Why CNN + Cross-Attention?

The architecture addresses two complementary aspects of binding prediction:

| Pattern Type | Biological Basis | Computational Approach |
|--------------|------------------|----------------------|
| **Local patterns** | Binding pocket motifs, functional groups | CNN with defined receptive field |
| **Interaction patterns** | Protein-ligand contacts, binding pose | Cross-attention mechanism |

The key insight is that **binding is inherently a bipartite interaction problem**: we must model relationships between two distinct entities (protein and ligand) rather than patterns within a single entity.

### Inductive Biases

Our architecture incorporates several inductive biases appropriate for molecular modeling:

1. **Translation equivariance** (CNN): The same binding motif should be recognized regardless of sequence position
2. **Permutation invariance** (Attention pooling): The order of residue-atom pairs should not affect the final prediction
3. **Locality** (CNN kernel size): Binding interactions are primarily local in sequence space
4. **Sparsity** (Attention softmax): Most residue-atom pairs do not interact directly

---

## Sequence Length Handling and Positional Encoding

### Understanding the Two-Stage Pipeline

The sequence length handling in DockTKinase occurs at **two distinct stages**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: EMBEDDING GENERATION (ESM/ESM-C Models)                           │
│                                                                             │
│  ⚠️  Limitation: ESM models have FIXED maximum sequence lengths            │
│                                                                             │
│  If protein length > max_len:                                              │
│      → Sequence is TRUNCATED before embedding                              │
│      → Information from truncated residues is LOST                         │
│                                                                             │
│  This truncation happens BEFORE our CNN+CrossAttention model               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CNN + CROSS-ATTENTION (Our Model)                                 │
│                                                                             │
│  ✅  With RoPE: Can process embeddings of ANY length                        │
│  ⚠️  With Sinusoidal: Limited by pre-defined max_len                       │
│                                                                             │
│  Note: Our model can only process what ESM provides                        │
│  If ESM truncated the sequence, we cannot recover that information         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ESM Model Sequence Limits

The following table shows the maximum sequence lengths supported by each protein embedding model:

| Model | Max Tokens | Embedding Dim | Recommended Use |
|-------|------------|---------------|-----------------|
| **esm2_t6_8M_UR50D** | 1024 | 320 | Quick testing, short proteins |
| **esm2_t12_35M_UR50D** | 1024 | 480 | Fast inference |
| **esm2_t30_150M_UR50D** | 1024 | 640 | Balanced |
| **esm2_t33_650M_UR50D** | 1024 | 1280 | Good accuracy |
| **esm2_t36_3B_UR50D** | **4096** | 2560 | Long proteins ✅ |
| **esm2_t48_15B_UR50D** | **5120** | 5120 | Very long proteins ✅ |
| **esmc-300m-2024-12** | 2048 | 960 | ESM Cambrian |
| **esmc-600m-2024-12** | 2048 | 1152 | ESM Cambrian |
| **esmc-6b-2024-12** | 2048 | 3072 | Via Forge API |

**Key insight**: For kinases (typically 250-500 aa), all models are sufficient. For large proteins (>1024 aa), use `esm2_t36_3B_UR50D` or larger.

### Positional Encoding Strategies

Once embeddings are generated, our CNN+CrossAttention model needs positional information. We support two strategies:

#### 1. Sinusoidal Positional Encoding (Vaswani et al., 2017)

The classical approach from "Attention Is All You Need":

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Characteristics**:
- ✅ Simple and well-understood
- ✅ Fixed computation (pre-computed buffer)
- ❌ Requires pre-defined `max_len`
- ❌ Positions beyond `max_len` cannot be encoded

**Usage**:
```python
model = CrossAttentionAffinityModel(
    positional_encoding_type='sinusoidal',  # default
    max_protein_len=2048,  # must be set appropriately
    max_ligand_len=512
)
```

#### 2. Rotary Position Embedding (RoPE) (Su et al., 2021)

Modern approach used by LLaMA, Mistral, and ESM-2 internally:

$$\text{RoPE}(x, pos) = x \cdot \cos(\theta_{pos}) + \text{rotate\_half}(x) \cdot \sin(\theta_{pos})$$

where:
$$\theta_i = \frac{pos}{10000^{2i/d}}$$

**Characteristics**:
- ✅ **No maximum sequence length** (extends dynamically)
- ✅ Preserves relative position information
- ✅ Applied to Q and K only (not V)
- ✅ Used by state-of-the-art models (LLaMA, Mistral)
- ⚠️ Slightly more compute per forward pass

**Mathematical Intuition**: RoPE rotates the query and key vectors in 2D subspaces by angles proportional to their positions. When computing attention scores $q \cdot k$, the rotation difference encodes the relative position:

$$\text{RoPE}(q, m) \cdot \text{RoPE}(k, n) = f(q, k, m-n)$$

This means attention naturally captures relative rather than absolute positions.

**Usage**:
```python
model = CrossAttentionAffinityModel(
    positional_encoding_type='rope',  # unlimited length
    # max_protein_len is now just initial cache size
)
```

### Practical Recommendations

| Scenario | ESM Model | Positional Encoding |
|----------|-----------|---------------------|
| Quick testing | esm2_t6_8M | sinusoidal |
| Standard kinases (<500 aa) | esm2_t33_650M | sinusoidal |
| Long proteins (500-2000 aa) | esm2_t36_3B | RoPE |
| Very long proteins (>2000 aa) | esm2_t36_3B or esm2_t48_15B | RoPE |
| Variable-length batches | any | RoPE |

### Code Example: Handling Long Sequences

```python
from src.classifier.models import CrossAttentionAffinityModel

# For standard kinases (< 1024 aa)
model_standard = CrossAttentionAffinityModel(
    protein_dim=1280,  # esm2_t33_650M
    positional_encoding_type='sinusoidal'
)

# For variable-length proteins (any size)
model_flexible = CrossAttentionAffinityModel(
    protein_dim=2560,  # esm2_t36_3B (max_len=4096)
    positional_encoding_type='rope'  # handles any length
)

# The model will process whatever ESM provides
# ESM truncation is the bottleneck, not our model
```

### References

- Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
- Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv:2104.09864.

---

## Architecture Overview

### Why CNN + Cross-Attention?

Protein-ligand binding prediction requires understanding **two types of patterns**:

| Pattern Type | What it Captures | Best Approach |
|--------------|------------------|---------------|
| **Local patterns** | Sequence motifs, functional groups | CNN |
| **Interaction patterns** | Which residues bind to which atoms | Cross-Attention |

### Model Architecture

The complete model follows a staged processing pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 0: INPUT REPRESENTATIONS                                             │
│                                                                             │
│  Protein: ESM-2 per-residue embeddings                                     │
│           P ∈ ℝ^(L_p × d_p)  where L_p = protein length, d_p = 1280        │
│           Source: Lin et al. (2023), Science 379, 1123-1130                │
│                                                                             │
│  Ligand: SMI-TED per-atom embeddings                                       │
│          L ∈ ℝ^(L_l × d_l)  where L_l = ligand length, d_l = 768           │
│          Source: Ross et al. (2022), Nature Machine Intelligence           │
│                                                                             │
│  Note: These embeddings already capture:                                    │
│  • Evolutionary conservation (ESM-2 trained on 250M sequences)             │
│  • Structural information (ESM-2 predicts contact maps)                    │
│  • Molecular properties (SMI-TED trained on molecular datasets)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CNN ENCODING (Local Feature Extraction)                           │
│                                                                             │
│  For modality m ∈ {protein, ligand}:                                       │
│                                                                             │
│  1. Input projection:  h⁰_m = ReLU(W_proj · x_m + b_proj)                  │
│  2. For each layer l = 1...L:                                              │
│     h^l_m = Conv1DBlock(h^(l-1)_m)                                         │
│                                                                             │
│  Output: h_P ∈ ℝ^(L_p × d_h), h_L ∈ ℝ^(L_l × d_h)  where d_h = 256        │
│                                                                             │
│  Positional encoding added: h_m ← h_m + PE(pos)                            │
│  PE(pos, 2i) = sin(pos / 10000^(2i/d))                                     │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                                   │
│  Source: Vaswani et al. (2017), NeurIPS                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CROSS-ATTENTION (Interaction Modeling)                            │
│                                                                             │
│  Bidirectional cross-attention (×N_layers):                                │
│                                                                             │
│  (A) Protein attending to Ligand:                                          │
│      Q_P = h_P · W_Q,  K_L = h_L · W_K,  V_L = h_L · W_V                   │
│      A_PL = softmax(Q_P · K_L^T / √d_k) · V_L                              │
│                                                                             │
│  (B) Ligand attending to Protein:                                          │
│      Q_L = h_L · W_Q,  K_P = h_P · W_K,  V_P = h_P · W_V                   │
│      A_LP = softmax(Q_L · K_P^T / √d_k) · V_P                              │
│                                                                             │
│  Multi-head attention with H=8 heads:                                      │
│  MultiHead(Q,K,V) = Concat(head_1,...,head_H) · W_O                        │
│  head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)                             │
│                                                                             │
│  Source: Vaswani et al. (2017), NeurIPS                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: POOLING AND PREDICTION                                            │
│                                                                             │
│  Global average pooling:                                                   │
│  z_P = (1/L_p) Σ_i h_P[i]  ∈ ℝ^d_h                                        │
│  z_L = (1/L_l) Σ_j h_L[j]  ∈ ℝ^d_h                                        │
│                                                                             │
│  Concatenation: z = [z_P; z_L] ∈ ℝ^(2·d_h)                                 │
│                                                                             │
│  Multi-task heads:                                                         │
│  ŷ_cls = σ(MLP_cls(z))     # Classification: P(active)                    │
│  ŷ_reg = MLP_reg(z)        # Regression: pChEMBL                          │
│                                                                             │
│  Source: Kendall et al. (2018), CVPR - Multi-task uncertainty             │
└─────────────────────────────────────────────────────────────────────────────┘
```

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

### Mathematical Formulation

Cross-attention extends self-attention (Vaswani et al., 2017) to operate across two different modalities. Given protein representation $P \in \mathbb{R}^{L_p \times d}$ and ligand representation $L \in \mathbb{R}^{L_l \times d}$:

#### Scaled Dot-Product Attention

For protein attending to ligand:

$$\text{Attention}(Q_P, K_L, V_L) = \text{softmax}\left(\frac{Q_P K_L^T}{\sqrt{d_k}}\right) V_L$$

where:
- $Q_P = P \cdot W_Q \in \mathbb{R}^{L_p \times d_k}$ (protein queries)
- $K_L = L \cdot W_K \in \mathbb{R}^{L_l \times d_k}$ (ligand keys)
- $V_L = L \cdot W_V \in \mathbb{R}^{L_l \times d_v}$ (ligand values)
- The scaling factor $\sqrt{d_k}$ prevents dot products from becoming too large, which would push softmax into regions with extremely small gradients (Vaswani et al., 2017).

#### Multi-Head Attention

We use $H = 8$ attention heads to allow the model to jointly attend to information from different representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) W^O$$

where $\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$

Each head operates on a subspace of dimension $d_k = d_{model} / H = 256 / 8 = 32$.

#### Biological Interpretation of Attention Weights

The attention matrix $A \in \mathbb{R}^{L_p \times L_l}$ has a direct biological interpretation:

$$A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}$$

- $A_{ij}$ represents the "importance" of ligand atom $j$ for protein residue $i$
- High $A_{ij}$ values indicate potential interaction sites
- The softmax ensures each residue distributes attention across all ligand atoms (sums to 1)

This can be compared to contact maps from molecular docking, where $A_{ij}$ approximates $P(\text{contact} | r_i, l_j)$.

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
                    Ligand Atoms (pharmacophore groups)
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
                  (hydrogen bond acceptor)

█ = A_ij > 0.3 (strong predicted interaction)
▓ = 0.1 < A_ij < 0.3 (moderate)
· = A_ij < 0.1 (weak/no interaction)
```

---

## Multi-Task Learning

### Joint Classification and Regression

The model simultaneously predicts:
1. **Classification**: Binary active/inactive label
2. **Regression**: Continuous pChEMBL value

This multi-task formulation provides several benefits (Caruana, 1997):
- **Implicit regularization**: Shared representations prevent overfitting
- **Data augmentation**: Each task provides additional training signal
- **Feature learning**: Tasks help discover relevant features

### Uncertainty-Weighted Loss

We use homoscedastic uncertainty weighting (Kendall et al., 2018) to automatically balance the classification and regression losses:

$$\mathcal{L}_{total} = \frac{1}{2\sigma_1^2}\mathcal{L}_{cls} + \frac{1}{2\sigma_2^2}\mathcal{L}_{reg} + \log\sigma_1 + \log\sigma_2$$

where:
- $\mathcal{L}_{cls}$ = Binary Cross-Entropy loss for classification
- $\mathcal{L}_{reg}$ = Mean Squared Error loss for regression  
- $\sigma_1, \sigma_2$ are learnable task-specific uncertainty parameters

#### Mathematical Derivation

Starting from the Gaussian likelihood for regression:

$$p(y | f(x)) = \mathcal{N}(f(x), \sigma^2)$$

The negative log-likelihood gives:

$$-\log p(y | f(x)) = \frac{1}{2\sigma^2}||y - f(x)||^2 + \log\sigma + const$$

For multi-task learning with tasks $t_1, t_2$:

$$\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log\sigma_i$$

The $\log\sigma$ terms prevent the trivial solution where $\sigma \to \infty$ (which would drive the loss to zero).

### Implementation

```python
class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable uncertainty weighting.
    Reference: Kendall et al. (2018), CVPR
    """
    def __init__(self):
        super().__init__()
        # Log variance (initialized to 0 → σ = 1)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))
    
    def forward(self, loss_cls, loss_reg):
        # Precision = 1/σ² = exp(-log_var)
        precision_cls = torch.exp(-self.log_var_cls)
        precision_reg = torch.exp(-self.log_var_reg)
        
        # Weighted losses + regularization
        total = precision_cls * loss_cls + self.log_var_cls
        total += precision_reg * loss_reg + self.log_var_reg
        
        return 0.5 * total
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

#### 4. Dilated Convolutions (Yu & Koltun, 2016)

Dilated (atrous) convolutions increase receptive field without increasing parameters:

```
Standard Conv (dilation=1):    Dilated Conv (dilation=2):
  [■ ■ ■]                        [■ · ■ · ■]
  kernel touches 3 positions     kernel touches 3 positions
  receptive field = 3            receptive field = 5

With exponential dilation (1, 2, 4, 8):
RF = 1 + Σᵢ 2 × (k-1) × dᵢ = 1 + 2 × 2 × (1+2+4+8) = 61 positions
```

**Why it works for proteins:** Protein binding sites often span 15-30 residues. Dilated convolutions capture these long-range patterns without the parameter cost of large kernels.

### Performance Comparison

| Metric | Standard CNN | Optimized CNN | Improvement |
|--------|--------------|---------------|-------------|
| **Parameters** | 2,626,816 | 1,155,840 | **-56%** |
| **Receptive Field** | 25 | 29 | **+16%** |
| **Training Stability** | Good | Better | Pre-LayerNorm |
| **Channel Adaptation** | None | SE blocks | Automatic |
| **Memory Usage** | Baseline | -40% | Fewer parameters |

---

## Implementation Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden dimension | 256 | Balance between capacity and efficiency |
| CNN layers | 3-4 | Sufficient receptive field without overfitting |
| Attention heads | 8 | Standard in transformers (Vaswani et al., 2017) |
| Cross-attention layers | 2 | Sufficient for protein-ligand modeling |
| Dropout | 0.1-0.2 | Regularization for small datasets |
| Learning rate | 5e-4 | Adam optimizer default |
| Batch size | 32-64 | Depends on GPU memory |

### Training Protocol

1. **Initialization**: Xavier/Glorot for linear layers, Kaiming for convolutions
2. **Optimizer**: AdamW with weight decay 0.01
3. **Learning rate schedule**: Cosine annealing with warmup
4. **Early stopping**: Patience of 10 epochs on validation loss
5. **Gradient clipping**: Max norm 1.0

---

## Scientific References

### Protein Language Models

1. **Lin, Z., et al. (2023)**. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science 379, 1123-1130. https://doi.org/10.1126/science.ade2574
   - ESM-2 protein language model used for per-residue embeddings
   - Demonstrates that language models capture protein structure
   - Foundation for our protein representation pipeline

2. **Rives, A., et al. (2021)**. *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. PNAS 118(15), e2016239118. https://doi.org/10.1073/pnas.2016239118
   - Original ESM work showing learned protein representations
   - Demonstrates transfer learning for computational biology

### Molecular Representation

3. **Ross, J., et al. (2022)**. *Large-scale chemical language representations capture molecular structure and properties*. Nature Machine Intelligence 4, 1256-1264. https://doi.org/10.1038/s42256-022-00580-7
   - SMI-TED (FM4M) molecular representations
   - Foundation for our ligand embeddings

### Deep Learning Architecture

4. **Vaswani, A., et al. (2017)**. *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS) 30.
   - Multi-head attention mechanism
   - Scaled dot-product attention formulation
   - Positional encoding scheme

5. **He, K., Zhang, X., Ren, S., & Sun, J. (2016)**. *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
   - Residual connections for deep networks
   - Skip connections enabling gradient flow

6. **Chollet, F. (2017)**. *Xception: Deep Learning with Depthwise Separable Convolutions*. IEEE CVPR, 1251-1258.
   - Depthwise separable convolutions
   - Parameter-efficient convolution design

7. **Hu, J., Shen, L., & Sun, G. (2018)**. *Squeeze-and-Excitation Networks*. IEEE CVPR, 7132-7141.
   - Channel attention mechanism
   - Adaptive feature recalibration

8. **Xiong, R., et al. (2020)**. *On Layer Normalization in the Transformer Architecture*. International Conference on Machine Learning (ICML), 10524-10533.
   - Pre-LayerNorm vs Post-LayerNorm analysis
   - Improved training stability

9. **Yu, F., & Koltun, V. (2016)**. *Multi-Scale Context Aggregation by Dilated Convolutions*. International Conference on Learning Representations (ICLR).
   - Dilated/atrous convolutions
   - Exponentially growing receptive field

10. **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016)**. *Layer Normalization*. arXiv:1607.06450.
    - Layer normalization technique
    - Alternative to batch normalization for sequence models

### Multi-Task Learning

11. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. IEEE CVPR, 7482-7491.
    - Homoscedastic uncertainty for task weighting
    - Automatic loss balancing in multi-task learning

12. **Caruana, R. (1997)**. *Multitask Learning*. Machine Learning 28, 41-75.
    - Foundational work on multi-task learning
    - Theoretical basis for shared representations

### Cross-Modal Attention

13. **Tsai, Y.-H. H., et al. (2019)**. *Multimodal Transformer for Unaligned Multimodal Language Sequences*. Annual Meeting of the Association for Computational Linguistics (ACL), 6558-6569.
    - Cross-modal attention for different input types
    - Fusion of heterogeneous modalities

### Protein-Ligand Binding

14. **Eldridge, M. D., et al. (1997)**. *Empirical scoring functions: I. The development of a fast empirical scoring function to estimate the binding affinity of ligands in receptor complexes*. Journal of Computer-Aided Molecular Design 11, 425-445.
    - Empirical scoring functions for binding
    - Pairwise atomic interaction decomposition

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
    num_layers=4,
    kernel_size=3,
    dilations=(1, 2, 4, 8),
    dropout=0.1,
    use_se=True
)

# Same interface
features = encoder(protein_emb)  # [32, 500, 256]

# Check configuration
print(f"Parameters: {encoder.count_parameters():,}")  # ~1.15M
print(f"Config: {encoder.get_config()}")
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

### Key Contributions

The CNN + Cross-Attention architecture addresses protein-ligand affinity prediction with a principled design:

1. **Theoretical grounding**: The architecture is motivated by the bipartite nature of protein-ligand binding, using cross-attention to model pairwise interactions without explicit structural information.

2. **Complementary components**:
   - **CNN Encoders**: Translation-equivariant local feature extraction with defined receptive fields
   - **Cross-Attention**: Learning which protein residues attend to which ligand atoms
   - **Multi-Task Head**: Joint classification and regression with uncertainty weighting

3. **Efficient optimization**: The optimized encoder reduces parameters by 56% (from 2.6M to 1.15M) while increasing receptive field by 16% (from 25 to 29 tokens), using:
   - Depthwise separable convolutions (Chollet, 2017)
   - Squeeze-and-Excitation blocks (Hu et al., 2018)
   - Pre-LayerNorm (Xiong et al., 2020)
   - Dilated convolutions (Yu & Koltun, 2016)

4. **Scientific rigor**: Every design choice is grounded in peer-reviewed literature, with 14 primary references covering protein language models, attention mechanisms, CNN architectures, and multi-task learning.

### Comparison with Related Work

| Method | Structure Required | Learned Interactions | Multi-Task |
|--------|-------------------|---------------------|------------|
| Traditional Docking | Yes (3D) | No (physics-based) | No |
| DeepDTA | No | CNN features | No |
| AttentionDTA | No | Self-attention | No |
| **Ours (CNN + Cross-Attention)** | **No** | **Bidirectional cross-attention** | **Yes** |

### Future Directions

1. **3D structure integration**: Incorporate predicted structures from ESMFold or AlphaFold
2. **Explainability**: Extract and validate attention maps against known binding sites
3. **Pre-training**: Self-supervised pre-training on large unlabeled protein-ligand datasets
4. **Ensemble methods**: Combine with traditional ML models for improved robustness

---

**Document Information**:
- **Authors**: DockTKinase Development Team
- **Last Updated**: December 2025
- **Version**: 2.1
- **License**: MIT
