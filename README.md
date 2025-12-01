# DockTKinase 🧬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-gmmsb--lncc%2Fdocktkinase-green.svg)](https://github.com/gmmsb-lncc/docktkinase)

**Integrated Pipeline for Protein-Ligand Property Prediction using Deep Learning**

An end-to-end machine learning pipeline that generates protein and ligand embeddings, constructs interaction matrices, and trains classifiers/regressors for drug-target activity prediction.

## Overview

DockTKinase combines state-of-the-art protein language models with molecular embeddings to predict compound activity against kinase targets. The pipeline supports:

- **Multiple protein embedding models**: ESM-2 (6 sizes), ESM-C (3 variants), Boltz-2
- **Ligand embeddings**: FM4M (SMI-TED) - 768 dimensions
- **Cross-Attention modeling**: CNN + Cross-Attention for interaction learning
- **ML pipelines**: 12 classifiers + 12 regressors + neural networks
- **Adaptive stratification**: Intelligent clustering-based train/val/test splitting

## Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Multi-Model Protein Embeddings** | ESM-2 (8M-15B params), ESM-C, Boltz-2 (384-dim) |
| 🔬 **Ligand Embeddings** | FM4M SMI-TED (768-dim) |
| 🎯 **Cross-Attention Module** | CNN + Cross-Attention for protein-ligand interactions |
| 📊 **ML Classifiers** | XGBoost, LightGBM, CatBoost, Random Forest, SVM, etc. |
| 📈 **ML Regressors** | Gradient Boosting, Ridge, Lasso, Neural Networks |
| 🔀 **K-means++ Stratification** | Cluster-aware train/val/test splitting with cosine similarity validation |
| ⚡ **GPU Acceleration** | CUDA, MPS (Apple Silicon), or CPU |
| 💾 **Smart Caching** | Incremental embedding generation with caching |

## Scalable Stratification with K-means++

### The Stratification Challenge

Machine learning models for drug discovery require careful dataset splitting to prevent **data leakage**. Simply random splitting can place highly similar protein-ligand pairs in different sets, leading to:

- **Overly optimistic performance estimates** during validation
- **Poor generalization** to truly novel compounds
- **Unreliable model selection**

Our stratification strategy addresses this by clustering similar samples together before splitting, ensuring that chemically/structurally related samples remain in the same subset.

### K-means++ Algorithm

We implement **K-means++** (Arthur & Vassilvitskii, 2007), which provides theoretical guarantees for clustering quality:

**Mathematical Foundation:**

The K-means++ initialization selects initial centroids with probability proportional to squared distance from existing centroids:

$$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$

where $D(x)$ is the distance to the nearest existing centroid.

**Key Properties:**
- **O(log k) competitive ratio**: Expected cost is at most $O(\log k)$ times the optimal k-means cost
- **Deterministic given seed**: Reproducible results for scientific experiments
- **Faster convergence**: Reduces Lloyd's algorithm iterations by 2-5x

**Implementation:**

```python
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    init='k-means++',      # Arthur & Vassilvitskii initialization
    n_init=10,             # Multiple initializations for robustness
    batch_size=1024,       # Memory-efficient mini-batch processing
    random_state=42        # Reproducibility
)
cluster_labels = kmeans.fit_predict(L2_normalized_embeddings)
```

### How K-means++ and Cosine Similarity Complement Each Other

Our stratification pipeline combines **K-means++ clustering** with **cosine similarity analysis** in a complementary fashion:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATIFICATION WORKFLOW: K-means++ + Cosine Similarity                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: EMBEDDING NORMALIZATION                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ • L2-normalize protein embeddings: ||p|| = 1                          │  │
│  │ • L2-normalize ligand embeddings:  ||l|| = 1                          │  │
│  │ • Concatenate: [p | l] → combined embedding                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  STEP 2: K-MEANS++ CLUSTERING (Primary: Cluster Formation)                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ • K-means++ on L2-normalized vectors uses Euclidean distance          │  │
│  │ • For normalized vectors: d²(a,b) = 2(1 - cos(θ))                     │  │
│  │ • Therefore: minimizing Euclidean ≡ maximizing cosine similarity      │  │
│  │ • Result: clusters of semantically similar protein-ligand pairs       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  STEP 3: GREEDY CLUSTER ASSIGNMENT (Split: 80/10/10)                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ • Sort clusters by size (descending)                                  │  │
│  │ • Assign each cluster to train/val/test based on current proportions  │  │
│  │ • Target: ~80% train, ~10% validation, ~10% test (±3% tolerance)      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  STEP 4: COSINE SIMILARITY VALIDATION (Secondary: Quality Assurance)       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ • Compute pairwise cosine similarity between splits                   │  │
│  │ • Verify low inter-split similarity (no data leakage)                 │  │
│  │ • Verify high intra-split similarity (coherent clusters)              │  │
│  │ • Generate quality reports and statistics                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why This Combination Works:**

| Component | Role | Metric |
|-----------|------|--------|
| **K-means++** | Forms clusters of similar samples | Euclidean on L2-normalized (≡ cosine) |
| **Cosine Similarity** | Validates split quality | Direct similarity measurement |
| **Greedy Assignment** | Ensures proportions | Sample count ratios |

**Mathematical Equivalence:**

For L2-normalized vectors $\vec{a}$ and $\vec{b}$ where $||\vec{a}|| = ||\vec{b}|| = 1$:

$$d_{euclidean}^2(\vec{a}, \vec{b}) = ||\vec{a} - \vec{b}||^2 = ||\vec{a}||^2 + ||\vec{b}||^2 - 2\vec{a} \cdot \vec{b} = 2(1 - \cos\theta)$$

Therefore, K-means++ on normalized embeddings **implicitly optimizes for cosine similarity**, while the explicit cosine similarity analysis provides **interpretable validation metrics**.

### Scalability for Large Datasets

Standard clustering requires computing a pairwise distance matrix $D \in \mathbb{R}^{n \times n}$, which has memory complexity $O(n^2)$:

| Dataset Size | Memory Required |
|--------------|-----------------|
| 10,000 samples | 0.4 GB |
| 100,000 samples | 40 GB |
| 500,000 samples | **1,000 GB** |
| 1,000,000 samples | **4,000 GB** |

For large-scale drug discovery datasets (100k+ compound-target pairs), this becomes computationally infeasible.

### Scalable Solution: Representative Sampling + K-means++

We implement a scalable clustering algorithm that combines K-means++ with representative sampling:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SCALABLE STRATIFICATION ALGORITHM                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. REPRESENTATIVE SAMPLING (PCA-stratified)                                │
│     • Sample size: min(50k, √n × 50)                                        │
│     • 50% random + 50% PCA-stratified → covers embedding space              │
│                                                                             │
│  2. K-MEANS++ CLUSTERING ON SAMPLE                                          │
│     • MiniBatchKMeans with init='k-means++' on sample                       │
│     • Memory: O(sample²) instead of O(n²)                                   │
│     • Theoretical guarantee: O(log k) competitive ratio                     │
│                                                                             │
│  3. CENTROID COMPUTATION                                                    │
│     • Compute mean embedding per cluster                                    │
│     • Normalize centroids for cosine similarity                             │
│                                                                             │
│  4. LABEL PROPAGATION                                                       │
│     • Assign all n points to nearest centroid                               │
│     • Batch processing (10k samples) → constant memory                      │
│     • Complexity: O(n × k) where k = number of clusters                     │
│                                                                             │
│  5. GREEDY SPLIT ASSIGNMENT                                                 │
│     • Assign clusters to train/val/test (80/10/10)                          │
│     • Ensures balanced stratification with ±3% tolerance                    │
│                                                                             │
│  6. COSINE SIMILARITY VALIDATION                                            │
│     • Verify inter-split dissimilarity (no leakage)                         │
│     • Report intra-cluster cohesion statistics                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complexity Analysis

| Metric | Standard Approach | Our Approach |
|--------|-------------------|--------------|
| **Memory** | $O(n^2)$ | $O(s^2 + n \cdot k)$ where $s$ = sample size |
| **Time** | $O(n^2 \log n)$ | $O(s^2 \log s + n \cdot k)$ |
| **Example (500k samples)** | 1,000 GB | ~10 GB |

### Scientific References

This approach is grounded in established research:

1. **Arthur, D., & Vassilvitskii, S. (2007)**. *k-means++: The advantages of careful seeding*. Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '07).
   - Foundation for K-means++ initialization with O(log k) competitive guarantee
   - Core algorithm used in our stratification pipeline

2. **Sculley, D. (2010)**. *Web-scale k-means clustering*. Proceedings of the 19th International Conference on World Wide Web (WWW '10). ACM.
   - Demonstrates that MiniBatchKMeans efficiently handles large-scale datasets
   - Basis for our scalable implementation

3. **Kaufman, L., & Rousseeuw, P. J. (1990)**. *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley Series in Probability and Statistics.
   - Theoretical basis for cluster validation and silhouette analysis

4. **Nguyen, X., Epps, J., & Bailey, J. (2010)**. *Information theoretic measures for clusterings comparison*. Journal of Machine Learning Research, 11, 2837-2854.
   - Framework for evaluating clustering quality in stratification

### Usage

The K-means++ stratification is automatically used by the pipeline:

```python
# Automatic - pipeline uses K-means++ with cosine similarity validation
python run_complete_pipeline.py \
    --input dataset.tsv \
    --output results/ \
    --seed 42

# Logs will show:
# [INFO] Clustering with K-means++ (Arthur & Vassilvitskii, 2007)
# [INFO] Using 10 initializations for robustness
# [INFO] Clustering complete: 45 clusters formed, inertia=1234.56
# [INFO] Split proportions: train=80.2%, val=9.8%, test=10.0%
```

For large datasets (>40k samples), scalable mode is automatically activated:

```python
# Large dataset - automatic scalable mode
python run_complete_pipeline.py \
    --input large_dataset.tsv \  # 500k+ samples
    --output results/ \
    --seed 42

# Logs will show:
# [INFO] Dataset too large for full distance matrix (500000 samples would require ~931.3 GiB)
# [INFO] Using scalable representative sampling approach
# [INFO] Scalable clustering: 500000 samples → 35355 sample size
# [INFO] K-means++ on sample, then label propagation to all points
# [INFO] Scalable clustering complete: 127 clusters, silhouette=0.3421
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU)
- 8GB+ RAM (16GB+ recommended)

### Quick Install

```bash
# Clone repository
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Create conda environment
conda env create -f environment.yml
conda activate docktkinase

# Install dependencies
python scripts/post_install.py
```

### Platform-Specific Requirements

```bash
# macOS (Apple Silicon)
pip install -r requirements-mac.txt

# Linux with CUDA
pip install -r requirements-cuda.txt

# CPU only
pip install -r requirements.txt
```

## Quick Start

### Basic Pipeline

```bash
# Run complete pipeline (embeddings → stratification → classification → regression)
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/my_run \
    --protein-model esm2_t33_650M_UR50D \
    --seed 42
```

### Cross-Attention Training

Train a CNN + Cross-Attention model on pre-computed embeddings:

```bash
# Enable cross-attention training
python attention_matrix.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --embedding-matrix concatenated_embeddings/embedding_matrix.npy \
    --output results/attention_run \
    --attention-matrix on \
    --epochs 50 \
    --batch-size 64 \
    --device mps

# Disable cross-attention (skip training)
python attention_matrix.py \
    --input tests/datasets/kinase_non_human_compounds.tsv \
    --attention-matrix off
```

### Input Data Format

TSV file with required columns:
```
compound_smiles    target_sequence    pchembl_value
CCO...             MKVLW...           7.5
```

- `compound_smiles`: Valid SMILES string
- `target_sequence`: Amino acid sequence
- `pchembl_value`: Activity value (pChEMBL scale)

## System Architecture

DockTKinase provides a **reproducible, documented pipeline** that ensures consistent train/validation/test splits across all experiments. The same stratification is used for both traditional ML models and deep learning approaches.

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_complete_pipeline.py                     │
│                       (Main CLI Entry Point)                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      IntegratedPipeline                          │
│                    (End-to-End Orchestration)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   BUILD PHASE   │ │  CLASSIFIER     │ │  REGRESSION     │
│                 │ │     PHASE       │ │     PHASE       │
│ • Protein Emb.  │ │                 │ │                 │
│ • Ligand Emb.   │ │ • 12 ML Models  │ │ • 12 ML Models  │
│ • Matrix Build  │ │ • MLP           │ │ • Neural Net    │
│ • Stratification│ │ • Cross-Val     │ │ • Metrics       │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         │         ┌─────────┴─────────┐         │
         │         │  SAME SPLITS USED │         │
         │         │  ACROSS ALL MODELS│         │
         │         └─────────┬─────────┘         │
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             CNN + CROSS-ATTENTION MODULE                         │
│              (Deep Learning for Interactions)                    │
│                                                                  │
│  Uses the SAME stratified splits for fair comparison with ML    │
└─────────────────────────────────────────────────────────────────┘
```

### Reproducible Stratification

All models (ML and DL) receive the **exact same data splits**, ensuring:

| Guarantee | Description |
|-----------|-------------|
| **Reproducibility** | Same random seed → identical splits every run |
| **Fair Comparison** | All 12 classifiers + 12 regressors + CNN use same train/val/test |
| **No Data Leakage** | K-means++ clustering keeps similar samples together |
| **Documented** | Split indices saved in JSON for audit and reproducibility |

```python
# Splits are saved and can be reloaded
results/
├── stratification/
│   ├── split_indices.json      # Exact sample indices for train/val/test
│   ├── split_metadata.json     # Clustering parameters, proportions
│   └── cluster_assignments.npy # Which cluster each sample belongs to
```

## CNN + Cross-Attention Model

The deep learning component implements a **hybrid CNN-Transformer architecture** that leverages convolutional neural networks for local feature extraction and cross-attention mechanisms for modeling non-local protein-ligand interactions. This design is motivated by the complementary strengths of both paradigms in biological sequence modeling.

### Theoretical Foundation

#### The Protein-Ligand Binding Problem

Predicting binding affinity between proteins and small molecules is a fundamental challenge in computational drug discovery. The binding free energy $\Delta G_{bind}$ can be decomposed as:

$$\Delta G_{bind} = \Delta H - T\Delta S \approx \sum_{i,j} w_{ij} \cdot \phi(r_i, l_j)$$

where $r_i$ represents protein residue $i$, $l_j$ represents ligand atom $j$, and $\phi(\cdot)$ captures pairwise interaction features. Our model learns this interaction function through attention mechanisms.

#### Why CNN + Cross-Attention?

| Component | Biological Motivation | Computational Role |
|-----------|----------------------|-------------------|
| **CNN Encoder** | Protein motifs (α-helices, β-sheets) and ligand functional groups are **local patterns** | Extracts position-invariant features with defined receptive fields |
| **Cross-Attention** | Binding involves **non-local interactions** between distant residues and ligand atoms | Models which residues attend to which atoms, learning implicit docking poses |
| **Multi-Head Attention** | Multiple interaction types exist simultaneously (H-bonds, hydrophobic, electrostatic) | Each head can specialize in different interaction modes |

### Architecture

The model processes per-token embeddings from protein language models (ESM-2) and molecular encoders (SMI-TED), producing binding affinity predictions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Per-token Embedding Matrices                                        │
│  ├── Protein: [batch, seq_prot, d_prot]  (ESM-2 per-residue embeddings)    │
│  └── Ligand:  [batch, seq_lig, d_lig]    (SMI-TED per-atom embeddings)     │
│                                                                             │
│  Typical dimensions:                                                        │
│  • ESM-2 650M: d_prot = 1280, seq_prot ≤ 1024                              │
│  • SMI-TED:    d_lig = 768,  seq_lig ≤ 512                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CNN ENCODERS (Local Feature Extraction)                           │
│                                                                             │
│  For each modality (protein/ligand):                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Input Projection: Linear(d_input → d_hidden)                       │ │
│  │  2. Multi-Scale Conv1D Stack:                                          │ │
│  │     • Layer 1: kernel=3, captures local patterns (3 tokens)            │ │
│  │     • Layer 2: kernel=5, medium context (7 tokens cumulative)          │ │
│  │     • Layer 3: kernel=7, wider context (13 tokens cumulative)          │ │
│  │  3. Residual Connections: x_out = Conv(x) + x (He et al., 2016)        │ │
│  │  4. Layer Normalization: stabilizes training (Ba et al., 2016)         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Output: [batch, seq_len, d_hidden] for each modality                       │
│                                                                             │
│  + Positional Encoding (sinusoidal, Vaswani et al., 2017)                   │
│    PE(pos, 2i) = sin(pos / 10000^{2i/d})                                    │
│    PE(pos, 2i+1) = cos(pos / 10000^{2i/d})                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: BIDIRECTIONAL CROSS-ATTENTION (Interaction Modeling)              │
│                                                                             │
│  Cross-Attention Layer (×N_layers, default N=2):                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  (A) Protein → Ligand Attention:                                       │ │
│  │      Q = W_Q · protein_repr                                            │ │
│  │      K = W_K · ligand_repr                                             │ │
│  │      V = W_V · ligand_repr                                             │ │
│  │      Attention(Q,K,V) = softmax(QK^T / √d_k) · V                       │ │
│  │                                                                        │ │
│  │  (B) Ligand → Protein Attention:                                       │ │
│  │      Q = W_Q · ligand_repr                                             │ │
│  │      K = W_K · protein_repr                                            │ │
│  │      V = W_V · protein_repr                                            │ │
│  │                                                                        │ │
│  │  Multi-Head: 8 heads, each with d_k = d_hidden / 8                     │ │
│  │  • Different heads learn different interaction types                   │ │
│  │  • Head 1-2: may learn hydrogen bonding patterns                       │ │
│  │  • Head 3-4: may learn hydrophobic contacts                            │ │
│  │  • Head 5-8: may learn electrostatic, π-stacking, etc.                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Each layer includes:                                                       │
│  • Pre-LayerNorm (Xiong et al., 2020): more stable gradients               │
│  • Residual connection: enables deep stacking                              │
│  • Feed-forward network: expands representation capacity                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: POOLING + MULTI-TASK PREDICTION                                   │
│                                                                             │
│  Global Average Pooling: [batch, seq, d] → [batch, d]                       │
│  Concatenation: [protein_pooled | ligand_pooled] → [batch, 2·d_hidden]     │
│                                                                             │
│  Multi-Task Heads:                                                          │
│  ┌─────────────────┐   ┌─────────────────┐                                  │
│  │  Classification │   │   Regression    │                                  │
│  │  MLP → sigmoid  │   │   MLP → linear  │                                  │
│  │  P(active)      │   │   pChEMBL       │                                  │
│  └─────────────────┘   └─────────────────┘                                  │
│                                                                             │
│  Multi-Task Loss (Kendall et al., 2018):                                    │
│  L = (1/2σ₁²)·L_cls + (1/2σ₂²)·L_reg + log(σ₁) + log(σ₂)                   │
│  • σ₁, σ₂ are learnable uncertainty parameters                             │
│  • Automatically balances classification vs regression                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CNN Encoder: Design Rationale

The CNN encoder transforms variable-length sequences into fixed-dimensional representations while preserving positional information through the convolutional structure.

#### Standard CNN Encoder

The baseline implementation uses cascaded 1D convolutions:

```python
class Conv1DBlock(nn.Module):
    """
    Standard 1D Convolutional block with residual connection.
    
    Architecture:
        Input → Conv1D → LayerNorm → GELU → Dropout → Conv1D → + Input → Output
                                                                ↑
                                                         Residual connection
    
    The residual connection enables training of deeper networks by providing
    gradient highways (He et al., 2016).
    """
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        # Two convolutions with residual skip
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = gelu(self.norm(x.transpose(-1,-2)).transpose(-1,-2))
        x = self.conv2(x)
        return x + residual  # Residual connection
```

**Receptive Field Calculation:**

For a stack of $L$ convolutional layers with kernel size $k$:

$$RF = 1 + L \cdot (k - 1)$$

With our default configuration (3 layers, kernel sizes 3, 5, 7):

$$RF = 1 + 2 + 4 + 6 = 13 \text{ tokens (standard encoder)}$$

#### Optimized CNN Encoder

We provide an optimized variant incorporating modern architectural improvements:

| Optimization | Reference | Benefit |
|--------------|-----------|---------|
| **Depthwise Separable Conv** | Chollet (2017), Xception | ~3× fewer parameters per layer |
| **Squeeze-and-Excitation** | Hu et al. (2018), SENet | Adaptive channel weighting |
| **Pre-LayerNorm** | Xiong et al. (2020) | More stable training dynamics |
| **Dilated Convolutions** | Yu & Koltun (2016) | Larger receptive field without parameter increase |

```python
class OptimizedConv1DBlock(nn.Module):
    """
    Optimized 1D Convolutional block with:
    - Pre-LayerNorm: More stable training (Xiong et al., 2020)
    - Depthwise Separable Conv: Efficient (Chollet, 2017)
    - Squeeze-and-Excitation: Channel attention (Hu et al., 2018)
    - Dilated convolutions: Larger receptive field (Yu & Koltun, 2016)
    """
    def __init__(self, channels, kernel_size=3, dilation=1, use_se=True):
        self.pre_norm = nn.LayerNorm(channels)  # Pre-norm for stability
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                    padding='same', groups=channels, dilation=dilation)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
```

**Parameter Comparison:**

| Architecture | Parameters | Receptive Field | Memory |
|--------------|------------|-----------------|--------|
| Standard (3 layers) | ~2.6M | 13 tokens | Baseline |
| Optimized (4 layers, dilated) | ~1.15M | 29 tokens | -40% |
| **Reduction** | **-56%** | **+123%** | **-40%** |

### Cross-Attention Mechanism

Cross-attention enables the model to learn which protein residues are relevant to which ligand atoms, effectively learning an implicit "soft docking" without explicit 3D structural information.

#### Mathematical Formulation

Given protein representation $P \in \mathbb{R}^{L_p \times d}$ and ligand representation $L \in \mathbb{R}^{L_l \times d}$:

**Protein attending to Ligand:**
$$\text{Attention}(Q_P, K_L, V_L) = \text{softmax}\left(\frac{Q_P K_L^T}{\sqrt{d_k}}\right) V_L$$

where:
- $Q_P = P \cdot W_Q$ (protein queries)
- $K_L = L \cdot W_K$ (ligand keys)  
- $V_L = L \cdot W_V$ (ligand values)

**Biological Interpretation:**

The attention matrix $A \in \mathbb{R}^{L_p \times L_l}$ where $A_{ij} = \text{softmax}(q_i \cdot k_j^T / \sqrt{d_k})$ represents:

- $A_{ij}$ = how much residue $i$ "attends to" ligand atom $j$
- High $A_{ij}$ suggests residue $i$ interacts with atom $j$
- The learned attention patterns correlate with known binding site residues

```
Cross-Attention Visualization:

                    Ligand Atoms (pharmacophore groups)
                 1  2  3  4  5  6  7  8  9  10
              ┌─────────────────────────────────┐
           45 │ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │
           50 │ ·  ·  ·  ▓  ▓  █  █  ▓  ·  ·  │ ← Binding pocket
Protein    55 │ ·  ·  ▓  █  █  █  █  █  ▓  ·  │    residues show high
Residues   60 │ ·  ·  ·  ▓  ▓  █  █  ▓  ·  ·  │    attention to key
           65 │ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │    functional groups
           70 │ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │

█ = A_ij > 0.3 (strong interaction)
▓ = 0.1 < A_ij < 0.3 (moderate)
· = A_ij < 0.1 (weak/no interaction)
```

### Multi-Task Learning

The model jointly predicts classification (active/inactive) and regression (pChEMBL value) targets using uncertainty-weighted multi-task learning:

$$\mathcal{L}_{total} = \frac{1}{2\sigma_1^2}\mathcal{L}_{cls} + \frac{1}{2\sigma_2^2}\mathcal{L}_{reg} + \log\sigma_1 + \log\sigma_2$$

where $\sigma_1, \sigma_2$ are learnable task-specific uncertainty parameters (Kendall et al., 2018). This formulation:

1. Automatically balances tasks based on homoscedastic uncertainty
2. Prevents one task from dominating the gradient updates
3. The $\log\sigma$ terms prevent trivial solutions where $\sigma \to \infty$

### Scientific References

**CNN Architecture:**

1. **He, K., Zhang, X., Ren, S., & Sun, J. (2016)**. *Deep Residual Learning for Image Recognition*. CVPR 2016.
   - Residual connections enabling deeper networks
   - Gradient highway for stable training

2. **Chollet, F. (2017)**. *Xception: Deep Learning with Depthwise Separable Convolutions*. CVPR 2017.
   - Depthwise separable convolutions
   - ~3× parameter reduction with maintained performance

3. **Hu, J., Shen, L., & Sun, G. (2018)**. *Squeeze-and-Excitation Networks*. CVPR 2018.
   - Channel attention mechanism
   - Adaptive feature recalibration

4. **Yu, F., & Koltun, V. (2016)**. *Multi-Scale Context Aggregation by Dilated Convolutions*. ICLR 2016.
   - Dilated/atrous convolutions
   - Exponentially growing receptive field

**Transformer/Attention:**

5. **Vaswani, A., et al. (2017)**. *Attention Is All You Need*. NeurIPS 2017.
   - Multi-head self-attention mechanism
   - Scaled dot-product attention formulation
   - Sinusoidal positional encoding

6. **Xiong, R., et al. (2020)**. *On Layer Normalization in the Transformer Architecture*. ICML 2020.
   - Pre-LayerNorm vs Post-LayerNorm analysis
   - Improved training stability with Pre-LN

**Multi-Task Learning:**

7. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR 2018.
   - Homoscedastic uncertainty for task weighting
   - Automatic loss balancing

**Protein-Ligand Modeling:**

8. **Lin, Z., et al. (2023)**. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science 379, 1123-1130.
   - ESM-2 protein language model
   - Per-residue representations for downstream tasks

9. **Rao, R., et al. (2021)**. *MSA Transformer*. ICML 2021.
   - Attention mechanisms for protein modeling
   - Cross-row attention for evolutionary information

### Usage

Train the CNN + Cross-Attention model:

```bash
# Enable cross-attention training with pre-computed embeddings
python attention_matrix.py \
    --input data/kinase_compounds.tsv \
    --embedding-matrix concatenated_embeddings/embedding_matrix.npy \
    --output results/attention_run \
    --attention-matrix on \
    --epochs 50 \
    --batch-size 64 \
    --device cuda
```

Using the optimized encoder (56% fewer parameters):

```python
from src.classifier.models import create_encoder, OptimizedCNNEncoder

# Factory function for encoder selection
encoder = create_encoder(
    input_dim=1280,       # ESM-2 dimension
    hidden_dim=256,
    optimized=True        # Use optimized architecture
)

# Or directly instantiate
encoder = OptimizedCNNEncoder(
    input_dim=1280,
    hidden_dim=256,
    num_layers=4,
    dilations=(1, 2, 4, 8),  # Exponentially growing receptive field
    use_se=True              # Enable Squeeze-and-Excitation
)

print(f"Parameters: {encoder.count_parameters():,}")  # ~1.15M vs 2.6M
print(f"Receptive field: 29 tokens")  # vs 13 tokens
```

## Embedding Matrix Processing

The pipeline generates a concatenated embedding matrix combining protein and ligand representations:

### Matrix Structure
```
embedding_matrix.npy: (N, 1088)
├── Protein embeddings: columns [0:320]   (ESM-2 t6_8M: 320-dim)
└── Ligand embeddings:  columns [320:1088] (SMI-TED: 768-dim)
```

### Generation Process

```python
# 1. Generate protein embeddings
from src.build.embeddings.protein_embedding import ProteinEmbedding

embedder = ProteinEmbedding(model_name='esm2_t6_8M_UR50D')
protein_emb = embedder.generate_batch_embeddings(sequences)  # (N, 320)

# 2. Generate ligand embeddings
from src.build.embeddings.ligand_embedding import LigandEmbedding

ligand_embedder = LigandEmbedding(model_name='fm4m')
ligand_emb = ligand_embedder.generate_batch_embeddings(smiles)  # (N, 768)

# 3. Concatenate into matrix
import numpy as np
embedding_matrix = np.concatenate([protein_emb, ligand_emb], axis=1)  # (N, 1088)
np.save('concatenated_embeddings/embedding_matrix.npy', embedding_matrix)
```

## Supported Models

### Protein Embedding Models

| Model | Parameters | Embedding Dim | Memory | Speed |
|-------|------------|---------------|--------|-------|
| `esm2_t6_8M_UR50D` | 8M | 320 | ~1 GB | Fast |
| `esm2_t12_35M_UR50D` | 35M | 480 | ~2 GB | Fast |
| `esm2_t30_150M_UR50D` | 150M | 640 | ~4 GB | Medium |
| `esm2_t33_650M_UR50D` | 650M | 1280 | ~6 GB | Medium |
| `esm2_t36_3B_UR50D` | 3B | 2560 | ~12 GB | Slow |
| `esm2_t48_15B_UR50D` | 15B | 5120 | ~48 GB | Very Slow |
| `esmc_300m` | 300M | 960 | ~3 GB | Fast |
| `esmc_600m` | 600M | 1152 | ~5 GB | Medium |
| `esmc_6b` | 6B | 4096 | ~24 GB | Slow |
| `boltz2` | ~400M | 384 | ~4 GB | Medium |

### Ligand Embedding Model

| Model | Embedding Dim | Description |
|-------|---------------|-------------|
| `fm4m` (SMI-TED) | 768 | Molecular foundation model from IBM Research |

## CLI Reference

### run_complete_pipeline.py

```bash
python run_complete_pipeline.py \
    --input INPUT_TSV \
    --output OUTPUT_DIR \
    --protein-model MODEL_NAME \
    --ligand-model fm4m \
    --batch-size 32 \
    --device cuda \
    --seed 42 \
    --stratifier-method auto \
    --stratifier-threshold 0.9 \
    --skip-classification \
    --skip-regression
```

### attention_matrix.py

```bash
python attention_matrix.py \
    --input INPUT_TSV \
    --embedding-matrix MATRIX_NPY \
    --output OUTPUT_DIR \
    --attention-matrix on|off \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --device cuda|mps|cpu \
    --seed 42
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input TSV file with compound data | Required |
| `--embedding-matrix` | Pre-computed embedding matrix (.npy) | Required |
| `--output` | Output directory for results | `results/attention_matrix` |
| `--attention-matrix` | Enable/disable training (`on`/`off`) | `on` |
| `--epochs` | Number of training epochs | `50` |
| `--batch-size` | Training batch size | `64` |
| `--learning-rate` | Optimizer learning rate | `0.0005` |
| `--device` | Compute device | Auto-detect |
| `--seed` | Random seed for reproducibility | `42` |

## Project Structure

```
docktkinase/
├── run_complete_pipeline.py    # Main pipeline CLI
├── attention_matrix.py         # Cross-Attention training CLI
│
├── src/
│   ├── integrated_pipeline.py  # Pipeline orchestration
│   │
│   ├── attention_matrix/       # Cross-Attention module
│   │   ├── config.py           # AttentionMatrixConfig
│   │   ├── model.py            # CrossAttentionModel, ImprovedCrossAttentionModel
│   │   ├── pipeline.py         # AttentionMatrixPipeline
│   │   ├── data_loader.py      # EmbeddingDataLoader
│   │   ├── dataset.py          # EmbeddingDataset
│   │   ├── trainer.py          # ModelTrainer
│   │   ├── evaluator.py        # ModelEvaluator
│   │   ├── metrics.py          # Classification/Regression metrics
│   │   ├── splitter.py         # DataSplitter
│   │   ├── attention_analyzer.py  # Attention weight analysis
│   │   └── __init__.py
│   │
│   ├── build/                  # Embedding generation & matrices
│   │   ├── pipeline/           # Build orchestration
│   │   ├── embeddings/         # Protein & ligand embeddings
│   │   │   ├── strategies/     # Model implementations
│   │   │   │   ├── esm2_strategy.py
│   │   │   │   ├── esmc_strategy.py
│   │   │   │   └── boltz_strategy.py
│   │   │   └── core/
│   │   ├── matrix/             # Matrix construction
│   │   └── stratification/     # Adaptive clustering
│   │
│   ├── classifier/             # Classification pipeline
│   │   ├── models/             # 12 sklearn + MLP
│   │   └── core/               # Training, evaluation
│   │
│   └── regression/             # Regression pipeline
│       ├── models/             # 12 sklearn + neural
│       └── core/
│
├── tests/                      # Integration tests
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── scripts/                    # Setup scripts
│
├── environment.yml             # Conda environment
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata
└── LICENSE                     # MIT License
```

## Results & Performance

### Classification Metrics

| Model | Accuracy | ROC-AUC | F1-Score | MCC |
|-------|----------|---------|----------|-----|
| XGBoost | 0.85 | 0.91 | 0.88 | 0.68 |
| LightGBM | 0.84 | 0.90 | 0.87 | 0.66 |
| CatBoost | 0.84 | 0.90 | 0.87 | 0.66 |
| Random Forest | 0.82 | 0.88 | 0.85 | 0.62 |
| **Cross-Attention** | **0.80** | **0.80** | **0.86** | **0.51** |

### Regression Metrics

| Model | R² | Pearson | Spearman | RMSE |
|-------|----|---------|---------| -----|
| XGBoost | 0.52 | 0.74 | 0.69 | 0.89 |
| LightGBM | 0.51 | 0.73 | 0.68 | 0.91 |
| CatBoost | 0.50 | 0.72 | 0.67 | 0.92 |
| **Cross-Attention** | **0.47** | **0.70** | **0.54** | **0.98** |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_attention_matrix.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- **[Getting Started](docs/01-getting-started/)** - Installation and prerequisites
- **[User Guide](docs/02-user-guide/)** - Pipeline usage and workflows
- **[Architecture](docs/03-architecture/)** - System design and patterns
- **[Modules](docs/04-modules/)** - Component documentation
  - [Boltz-2 Strategy](docs/04-modules/BOLTZ_STRATEGY_GUIDE.md)
  - [Adaptive Clustering](docs/04-modules/ADAPTIVE_CLUSTERING_GUIDE.md)
  - [CNN + Cross-Attention Architecture](docs/04-modules/CNN_CROSS_ATTENTION_ARCHITECTURE.md)
- **[Development](docs/05-development/)** - Contributing guidelines
- **[Troubleshooting](docs/07-troubleshooting/)** - Common issues

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Scientific References

### Protein Language Models

1. **Lin, Z., et al. (2023)**. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science 379, 1123-1130.
   - ESM-2 protein language model used for per-residue embeddings
   - Foundation for our protein representation pipeline

2. **Rives, A., et al. (2021)**. *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. PNAS 118(15), e2016239118.
   - Original ESM work demonstrating learned protein representations
   - Basis for transfer learning in computational biology

### Deep Learning Architecture

3. **Vaswani, A., et al. (2017)**. *Attention Is All You Need*. NeurIPS 2017.
   - Multi-head attention mechanism used in our cross-attention module
   - Scaled dot-product attention formulation

4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016)**. *Deep Residual Learning for Image Recognition*. CVPR 2016.
   - Residual connections enabling deep CNN training
   - Foundation for our convolutional encoder architecture

5. **Chollet, F. (2017)**. *Xception: Deep Learning with Depthwise Separable Convolutions*. CVPR 2017.
   - Depthwise separable convolutions for parameter efficiency
   - Used in our optimized CNN encoder

6. **Hu, J., Shen, L., & Sun, G. (2018)**. *Squeeze-and-Excitation Networks*. CVPR 2018.
   - Channel attention for adaptive feature recalibration
   - Implemented in OptimizedConv1DBlock

7. **Xiong, R., et al. (2020)**. *On Layer Normalization in the Transformer Architecture*. ICML 2020.
   - Pre-LayerNorm for stable training dynamics
   - Applied in our optimized encoder

8. **Yu, F., & Koltun, V. (2016)**. *Multi-Scale Context Aggregation by Dilated Convolutions*. ICLR 2016.
   - Dilated convolutions for larger receptive field
   - Enables capturing longer-range dependencies

### Multi-Task Learning

9. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. *Multi-Task Learning Using Uncertainty to Weigh Losses*. CVPR 2018.
   - Homoscedastic uncertainty for automatic loss balancing
   - Used in our MultiTaskLoss implementation

### Clustering & Stratification

10. **Arthur, D., & Vassilvitskii, S. (2007)**. *k-means++: The advantages of careful seeding*. SODA 2007.
    - K-means++ initialization with O(log k) competitive guarantee
    - Core algorithm for our stratification pipeline

11. **Sculley, D. (2010)**. *Web-scale k-means clustering*. WWW 2010.
    - MiniBatchKMeans for large-scale clustering
    - Enables stratification of 500k+ samples

### Molecular Representation

12. **Ross, J., et al. (2022)**. *Large-scale chemical language representations capture molecular structure and properties*. Nature Machine Intelligence 4, 1256-1264.
    - SMI-TED molecular representations (FM4M)
    - Used for ligand embeddings in our pipeline

## Acknowledgments

### Foundation Models
- **[ESM-2](https://github.com/facebookresearch/esm)** - Meta AI (Lin et al., 2023)
- **[ESM-C](https://github.com/evolutionaryscale/esm)** - EvolutionaryScale
- **[Boltz-2](https://github.com/jwohlwend/boltz)** - MIT AI Lab
- **[FM4M/SMI-TED](https://github.com/IBM/materials)** - IBM Research (Ross et al., 2022)

### Core Dependencies
- PyTorch, scikit-learn, NumPy, Transformers

## Citation

```bibtex
@software{docktkinase2025,
  title = {DockTKinase: Integrated Pipeline for Protein-Ligand Property Prediction},
  author = {DockTKinase Development Team},
  year = {2025},
  url = {https://github.com/gmmsb-lncc/docktkinase},
  version = {2.1}
}
```

## Contact

- **Repository**: [gmmsb-lncc/docktkinase](https://github.com/gmmsb-lncc/docktkinase)
- **Issues**: [Bug reports & features](https://github.com/gmmsb-lncc/docktkinase/issues)
- **Discussions**: [Q&A](https://github.com/gmmsb-lncc/docktkinase/discussions)

---

**Status**: ✅ Production Ready | **Version**: 2.1 | **Last Updated**: December 2025
