# semantic-screening: Semantic Interaction Prediction via Multi-Modal Foundation Models

**Author**: Leon Sulfierry (GMMSB-LNCC)
**Date**: February 2026
**Version**: 3.0 (Scaffold splits + Unified Benchmark)

---

## Abstract

The accurate identification of potent kinase inhibitors is a cornerstone of modern drug discovery, yet it remains a computationally challenging problem due to the high dimensionality of biological space and the scarcity of labeled structural data. This document presents **semantic-screening**, a modular, scalable, and scientifically rigorous deep learning platform designed to address these challenges. By integrating state-of-the-art Protein Language Models (ESM-2, ESM-3/ESM-C) and Chemical Foundation Models (SMI-TED, MoLFormer) within the novel **DT-Kinase** architecture—a Cross-Attention Convolutional neural network—semantic-screening learns to predict both **binary bioactivity** (active/inactive) and **binding affinity** ($K_d, IC_{50}$) directly from sequence and SMILES representations. This approach enables high-throughput **candidate prioritization** by bypassing the need for explicit 3D co-crystal structures during inference, effectively performing "semantic docking" in a latent space. We introduce a **scaffold-based splitting methodology** using Murcko scaffold decomposition to rigorously evaluate generalization capabilities and prevent chemical series leakage. This document details the theoretical foundations, architectural decisions, and implementation strategies that define the semantic-screening platform and DT-Kinase architecture.

---

## Chapter 1: Introduction

### 1.1 The Kinase Drug Discovery Challenge

Protein kinases constitute approximately 2% of the human proteome (518 genes per Manning et al. taxonomy) but regulate an estimated 30% of all cellular proteins through reversible phosphorylation. This topological centrality in cell signaling networks establishes kinases as **control nodes** whose activation state determines fundamental cellular decisions. Kinase dysregulation—through activating mutations, gene amplification, chromosomal fusions, or loss of negative regulators—constitutes a driver oncogenic event across a broad spectrum of human malignancies. As of 2024, 72 small-molecule kinase inhibitors have obtained regulatory approval (FDA/EMA), generating global revenues exceeding $80 billion annually.

The central pharmacological challenge lies in the **Selectivity Paradox**: the ATP-binding pocket is highly conserved across the >500 human kinases (RMSD < 2Å between structurally distant kinases). This structural similarity makes it notoriously difficult to design **selective inhibitors** that target a specific kinase without causing off-target toxicity. Discriminating 10-fold selectivity requires ΔG differences of only 1.4 kcal/mol—below the systematic error of scoring functions (±2-3 kcal/mol).

semantic-screening addresses this paradox by treating the interaction problem as a **multi-modal representation learning task**, aiming to capture subtle sequence variations that dictate selectivity through semantic compatibility in latent space rather than geometric fitting.

### 1.2 Defining the Prediction Tasks

To effectively prioritize drug candidates, semantic-screening solves two distinct but complementary problems:

1.  **Binary Bioactivity Prediction (Classification)**:
    *   **Goal**: Filter the vast chemical space to identify "Active" compounds.
    *   **Definition**: A compound is labeled $y=1$ (Active) if its affinity exceeds a threshold (e.g., $pChEMBL \ge 6.0$ or $IC_{50} \le 1000nM$), and $y=0$ otherwise.
    *   **Role**: High-recall screening to reduce the search space.

2.  **Binding Affinity Prediction (Regression)**:
    *   **Goal**: Quantify the strength of the interaction for active candidates.
    *   **Definition**: Predict the precise thermodynamic value, typically represented as $pChEMBL = -\log_{10}(IC_{50}/K_i/K_d)$.
    *   **Role**: High-precision ranking for lead optimization.

### 1.3 From Docking to Language Modeling: The semantic-screening Philosophy

The platform name **semantic-screening** reflects the fundamental paradigm shift proposed in this work. While traditional approaches like **DockThor** (the molecular docking platform developed by GMMSB-LNCC) rely on physics-based simulations and explicit 3D coordinate sampling to find optimal binding poses, semantic-screening adopts a purely **semantic** approach based on Protein Language Models.

We treat biological interaction not as a geometric puzzle, but as a **semantic compatibility problem** between the "language" of protein sequences (amino acids) and the "language" of chemical structures (SMILES). The central hypothesis (PhD Thesis, Chapter 1, Section 1.4) states: **sequence determines structure, which determines function**—and this mapping is computationally recoverable via PLMs. By leveraging the attention mechanisms of Transformer models, semantic-screening infers interaction patterns from the evolutionary and chemical context embedded in these sequences.

The **DT-Kinase** architecture, implemented within semantic-screening, operationalizes this hypothesis through bidirectional cross-attention that learns position-specific correspondences between protein residues and ligand atoms, effectively performing "semantic docking" in a high-dimensional latent space.

### 1.4 Formal Design Criteria (from PhD Thesis, Chapter 2)

semantic-screening was developed to satisfy five simultaneous requirements:

1.  **Structural Independence**: The system must operate exclusively from primary sequences, eliminating dependence on 3D coordinates and guaranteeing universal applicability to the entire kinome (~40% of kinases lack experimental structures).
2.  **Rich Semantic Representations**: Proteins and ligands must be encoded through pre-trained contextual embeddings capturing latent structural and functional information (ESM-2 for proteins, SMI-TED for ligands).
3.  **Explicit Interaction Modeling**: Cross-attention mechanism between protein and ligand representations enables learning which regions of each entity are relevant for specific affinity prediction.
4.  **Computational Scalability**: Throughput > 10⁶ predictions/hour, enabling ultralarge chemical library screening (10⁹ compounds) against complete kinome.
5.  **Multi-Task Framework**: Joint prediction of binary bioactivity (classification) and quantitative affinity (regression) through shared training objective improves generalization via learning transferable representations.

---

## Chapter 2: Theoretical Foundations

Before detailing the specific architecture of semantic-screening, it is essential to establish the theoretical framework that underpins our approach. This chapter introduces the core concepts of biological language modeling, explaining how proteins and molecules can be treated as linguistic entities and how modern deep learning techniques can extract meaningful representations from them.

### 2.1 The Language of Life: Proteins as Sequences

Proteins are the molecular machines of life, performing a vast array of functions from catalysis (enzymes) to signaling (receptors). Structurally, a protein is a linear polymer composed of a specific sequence of small molecules called **amino acids**. There are 20 standard amino acids, each represented by a single letter (e.g., 'A' for Alanine, 'K' for Lysine).

$$ P = \{a_1, a_2, ..., a_L\} \quad \text{where} \quad a_i \in \mathcal{A} = \{A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y\} $$

This linear structure is remarkably similar to human language, where a sentence is a sequence of words. Just as the meaning of a word depends on its context within a sentence, the function of an amino acid depends on its neighbors and its position in the 3D structure.

#### 2.1.1 Protein Language Models (pLMs)
Traditional bioinformatics relied on alignment-based methods (like BLAST) to find evolutionary relationships. However, recent advances in NLP have given rise to **Protein Language Models (pLMs)**. These models, typically based on the **Transformer** architecture, are trained on billions of protein sequences to predict missing or masked amino acids.

By learning to predict the next amino acid in a sequence, the model implicitly learns the "grammar" of protein folding and function. The internal representation (embedding) generated by the model captures physicochemical properties (charge, hydrophobicity) and structural contacts without ever being explicitly trained on 3D coordinates.

### 2.2 The Language of Chemistry: Molecules as Graphs and Strings

Small molecule drugs (ligands) are fundamentally different from proteins. They are not linear polymers but defined by graph structures where atoms are nodes and chemical bonds are edges. To process these molecules with language models, we use linear string representations, most notably **SMILES** (Simplified Molecular Input Line Entry System).

A SMILES string encodes the molecular graph into a sequence of characters. For example, Benzene is represented as `c1ccccc1`.

$$ L = \{s_1, s_2, \dots, s_M\} \quad \text{where} \quad s_i \in \mathcal{S} = \{C, N, O, =, (, ), [, ], @, \#, \dots\} $$

#### 2.2.2 Chemical Foundation Models
Similar to pLMs, **Chemical Foundation Models** are trained on massive databases of chemical structures (like PubChem or ChEMBL). They learn to understand chemical syntax and semantics, generating vector representations that capture molecular properties such as solubility, toxicity, and binding potential.

### 2.3 The Interaction Problem: From Lock-and-Key to Induced Fit

The classical view of protein-ligand interaction is the **"Lock and Key"** model, where a rigid protein pocket (lock) perfectly accommodates a specific ligand (key). However, biological reality is more complex. Proteins are dynamic; they breathe and change shape upon binding, a phenomenon known as **"Induced Fit"**.

Computational methods must account for this flexibility.
1.  **Molecular Docking**: Simulates the physical process of binding, exploring thousands of orientations (poses) and conformations. It is accurate but computationally expensive ($O(N^3)$ or worse).
2.  **Machine Learning**: Attempts to learn a function $f(Protein, Ligand) \to Affinity$ from data.

semantic-screening represents the next evolution of ML approaches. Unlike earlier models (e.g., DeepDTA) that relied on simple CNNs over one-hot encodings, we use **contextual embeddings** from Foundation Models. By combining the "protein understanding" of pLMs with the "chemical understanding" of chemical models, we aim to predict interaction compatibility directly from the learned latent spaces, approximating the thermodynamics of induced fit without explicit simulation.

### 2.4 Mathematical Formulation of Representation Learning

To bridge the gap between biology and computation, we formalize the representation learning process.

#### 2.4.1 The Embedding Function
Let $\mathcal{V}$ be the vocabulary of amino acids (for proteins) or atoms (for ligands). A sequence $S$ of length $N$ is a tuple $(x_1, \dots, x_N)$ where $x_i \in \mathcal{V}$.
A Foundation Model $\Phi$ acts as a function mapping this discrete sequence to a sequence of continuous vectors in $\mathbb{R}^d$:

$$ \Phi: \mathcal{V}^N \to \mathbb{R}^{N \times d} $$
$$ \mathbf{H} = \Phi(S) = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N]^T $$

Where $\mathbf{h}_i \in \mathbb{R}^d$ represents the contextual embedding of the $i$-th token.

#### 2.4.2 The Self-Attention Mechanism
The core engine of $\Phi$ (e.g., ESM-2, SMI-TED) is the **Self-Attention** mechanism. For a given input matrix $\mathbf{X}$, the model computes three projections: Queries ($\mathbf{Q}$), Keys ($\mathbf{K}$), and Values ($\mathbf{V}$):

$$ \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V $$

The attention weights $\mathbf{A}$ represent the relevance of token $j$ to token $i$:

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} $$

Where $d_k$ is the dimension of the attention head, serving as a scaling factor to prevent vanishing gradients in the softmax function. Biologically, this allows the model to "attend" to distant residues that are close in 3D space (contact prediction) or functionally coupled (co-evolution), effectively learning the protein's contact map without explicit supervision.

---

## Chapter 3: Computational Framework & Architecture

### 3.1 Modular Design Philosophy

The semantic-screening system adheres to the **Separation of Concerns** principle, dividing the complex workflow of affinity prediction into three distinct, loosely coupled modules. This modularity ensures maintainability, testability, and the flexibility to upgrade individual components without systemic disruption.

The architecture is composed of the following core subsystems:

1.  **Build Module (`src.build`)**: Responsible for data ingestion, embedding generation, and matrix construction. It acts as the ETL (Extract, Transform, Load) layer of the pipeline.
2.  **Classifier Module (`src.classifier`)**: A multi-model benchmarking suite designed for the **Binary Bioactivity Prediction** task (see 1.2). It serves as a high-recall filter to identify potential binders.
3.  **Regression Module (`src.regression`)**: A precision-focused module that predicts quantitative **Binding Affinity** (see 1.2) for the candidates identified by the classifier.

### 3.2 The Strategy Pattern for Model Integration

A critical architectural challenge in modern bioinformatics is the rapid pace of model evolution. To accommodate this, semantic-screening employs the **Strategy Design Pattern** for embedding generation. This allows the system to switch between different protein language models (e.g., ESM-2, ESM-3/ESM-C) and ligand encoders (e.g., SMI-TED, MoLFormer) at runtime, while maintaining a consistent API for the downstream pipelines.

#### 3.2.1 Protein Embedding Strategy

The `ProteinEmbedding` class acts as the context, delegating the actual computation to concrete strategy implementations derived from `BaseProteinStrategy`.

*   **Context**: `src.build.embeddings.protein_embedding.ProteinEmbedding`
*   **Interface**: `src.build.embeddings.strategies.base_protein_strategy.BaseProteinStrategy`
*   **Concrete Strategies**:
    *   `ESM2Strategy`: Wraps Meta AI's `fair-esm` library for models ranging from 8M to 15B parameters.
    *   `ESMCStrategy`: Integrates EvolutionaryScale's generative models (300M, 600M, 6B).

This design allows researchers to experiment with cutting-edge models simply by changing a configuration string (e.g., `--protein-model esmc-600m-2024-12`), without modifying the core pipeline code.

### 3.3 Pipeline Orchestration

The `IntegratedPipeline` class (`src.integrated_pipeline.py`) serves as the master orchestrator. It manages the data flow between modules, handles checkpointing, and ensures that the output of the Build module (embedding matrices) is correctly formatted for the Classifier and Regression modules.

$$ \text{Raw Data} \xrightarrow{\text{Build}} \text{Embeddings} \xrightarrow{\text{Scaffold Split}} \text{Train/Val/Test} \xrightarrow{\text{3-Level Benchmark}} \text{Comparison} $$

This linear flow is augmented by a robust **Checkpoint System**, which caches intermediate results (e.g., `embedding_matrix.npy`) to prevent redundant computations—a crucial feature when working with large-scale biological datasets that can take days to process.

---

## Chapter 4: Data Representation & Embeddings

The efficacy of any deep learning model is fundamentally limited by the quality of its input representations. semantic-screening eschews manual feature engineering (e.g., molecular fingerprints, physicochemical descriptors) in favor of learned representations from large-scale foundation models.

### 4.1 Protein Representation: The Language of Life

Proteins are treated as sequences of amino acids, analogous to sentences in natural language. We utilize **Protein Language Models (pLMs)** trained on billions of sequences (e.g., UniRef50) to extract embeddings that capture deep evolutionary and structural context.

#### 4.1.1 ESM-2 (Evolutionary Scale Modeling)
ESM-2 (Lin et al., 2023) is a BERT-style transformer trained with a Masked Language Modeling (MLM) objective. For a protein sequence $S = \{x_1, x_2, ..., x_L\}$, the model outputs a matrix $\mathbf{H}_P \in \mathbb{R}^{L \times D}$, where $D$ is the embedding dimension.

*   **Architecture**: Transformer Encoder with Rotary Position Embeddings (RoPE).
*   **Scale**: We support the full range of ESM-2 models, from 8M parameters ($D=320$) to 15B parameters ($D=5120$).
*   **Training Objective**: Masked Language Modeling (MLM). The model randomly masks 15% of amino acids and learns to predict them based on the surrounding context.

$$ \mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\setminus \mathcal{M}}) $$

*   **Usage**: We extract the per-residue representations from the final hidden layer, providing a granular view of the protein surface.

#### 4.1.2 ESM-C (Generative Modeling)
ESM-C (Hayes et al., 2024) represents a shift towards generative modeling. Unlike ESM-2's bidirectional attention, ESM-C uses causal masking, allowing it to model the probability distribution of the next amino acid. This is particularly useful for capturing long-range dependencies and functional motifs that define binding pockets.

*   **Architecture**: Causal Transformer Decoder.
*   **Training Objective**: Next Token Prediction (NTP). The model predicts the amino acid at position $t$ given the sequence $x_{1:t-1}$.

$$ P(x) = \prod_{t=1}^L P(x_t | x_{<t}) $$

*   **Advantage**: While MLM excels at understanding static structure, NTP captures the generative grammar of evolution, potentially offering better representations for mutational effects.

### 4.2 Ligand Representation: Chemical Foundation Models

Small molecules are represented using SMILES (Simplified Molecular Input Line Entry System) strings. semantic-screening supports two chemical foundation models:

#### 4.2.1 SMI-TED (SMILES-based Transformer Encoder-Decoder)
A model from the FM4M (Foundation Models for Molecules) suite by IBM Research.

*   **Tokenization**: SMILES strings are tokenized into chemical atoms and bond symbols (e.g., `C`, `N`, `=`, `(`).
*   **Architecture**: A Transformer encoder trained on large chemical databases (PubChem, ChEMBL).
*   **Output**: A sequence of vectors $\mathbf{H}_L \in \mathbb{R}^{M \times 768}$, where $M$ is the number of atoms/tokens.

#### 4.2.2 MoLFormer
An alternative chemical foundation model optimized for molecular property prediction.

*   **Architecture**: Transformer encoder with linear attention for efficiency.
*   **Output**: Per-token embeddings $\mathbf{H}_L \in \mathbb{R}^{M \times 768}$.
*   **Advantage**: Better suited for cross-attention models due to richer per-token representations.

### 4.3 From Sequences to Vectors: Pooling Strategies

The Foundation Models described above output **sequence embeddings** (matrices of shape $L \times D$). While Deep Learning architectures (Chapter 6) can process these sequences directly, Classical Machine Learning models (Chapter 5) typically require fixed-size **vector inputs** (shape $1 \times D$).

To bridge this gap, we employ **Pooling Strategies** to aggregate the sequence information into a single global representation:

1.  **Mean Pooling**: Calculates the element-wise average of all token embeddings along the sequence dimension.
    $$ \mathbf{v} = \frac{1}{L} \sum_{i=1}^L \mathbf{h}_i $$
    This captures the "average" physicochemical state of the protein or molecule.

2.  **CLS Token Pooling**: Uses the embedding of the special classification token (`[CLS]` or `<sos>`) prepended to the sequence. In BERT-like models (ESM-2), this token is trained to aggregate global context.

3.  **Max Pooling**: Takes the maximum value across the sequence dimension for each feature. This is effective for detecting the presence of specific motifs (e.g., a specific binding site residue) regardless of its position.

These aggregated vectors serve as the input features for the Classical Machine Learning Suite described in the next chapter.

---

## Chapter 5: Classical Machine Learning Suite

While the Deep Learning module focuses on end-to-end representation learning, semantic-screening incorporates a robust **Classical Machine Learning Suite** (`src.classifier`) to serve as a high-recall filter and baseline comparator. This module implements 12 distinct algorithms, ranging from probabilistic models to state-of-the-art gradient boosting machines.

**Input**: These models operate on the **fixed-size aggregated vectors** derived from the Foundation Models (as described in Section 4.3), effectively treating the embeddings as high-quality, pre-computed feature vectors.

### 5.1 Probabilistic & Linear Models

#### 5.1.1 Naive Bayes (GaussianNB)
The simplest baseline, based on applying Bayes' theorem with the "naive" assumption of conditional independence between features.
$$ P(y|x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y) $$
Despite the independence assumption being violated in dense embeddings, it serves as an ultra-fast baseline (~2s training time).

#### 5.1.2 Logistic Regression
A linear model that models the probability of the positive class using the sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$.
$$ P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) $$
We employ the L-BFGS solver with L2 regularization to handle the high-dimensionality of the embedding space.

#### 5.1.3 Linear SVC (Support Vector Classifier)
Finds the optimal hyperplane that maximizes the margin between classes.
$$ \min_{\mathbf{w}, b} \frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b)) $$
It is particularly effective in high-dimensional spaces where data is often linearly separable.

### 5.2 Tree-Based Ensembles

#### 5.2.1 Decision Trees & Extra Trees
Decision Trees recursively partition the feature space to maximize Information Gain (or minimize Gini Impurity). **Extra Trees (Extremely Randomized Trees)** introduce further randomness by selecting cut-points completely at random, which reduces variance and computational cost compared to standard Random Forests.

#### 5.2.2 Random Forest
An ensemble method that uses **Bagging (Bootstrap Aggregating)**. It trains multiple decision trees on random subsets of the data and features, averaging their predictions to prevent overfitting.
$$ \hat{y} = \frac{1}{K} \sum_{k=1}^K f_k(\mathbf{x}) $$

### 5.3 Gradient Boosting Machines

Boosting algorithms build an ensemble sequentially, where each new model attempts to correct the errors of the previous ones.

#### 5.3.1 AdaBoost (Adaptive Boosting)
The first practical boosting algorithm. It adjusts the weights of training instances, focusing subsequent learners on hard-to-classify examples.

#### 5.3.2 Gradient Boosting & XGBoost
These methods generalize boosting by optimizing an arbitrary differentiable loss function using gradient descent in function space.
$$ F_{m}(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x}) $$
**XGBoost (Extreme Gradient Boosting)** includes regularization terms in the objective function to control model complexity, making it the state-of-the-art for tabular data (and by extension, fixed embeddings).

#### 5.3.3 LightGBM
A highly efficient implementation of gradient boosting that uses Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to speed up training on large datasets without compromising accuracy.

### 5.4 Instance-Based & Neural Models

#### 5.4.1 K-Nearest Neighbors (KNN)
A non-parametric method that classifies a sample based on the majority vote of its neighbors in the embedding space. We use the Euclidean distance metric:
$$ d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=1}^D (x_i - z_i)^2} $$
Due to the "Curse of Dimensionality", KNN benefits significantly from the dimensionality reduction properties of the Foundation Models.

#### 5.4.2 Multi-Layer Perceptron (MLP)
A feedforward artificial neural network. While simpler than our Cross-Attention architecture, the sklearn-based MLP serves as a bridge between classical and deep learning approaches within the ensemble.

---

## Chapter 6: Deep Learning Architectures for Affinity Prediction

semantic-screening introduces a specialized neural architecture designed to model the physical interaction between a protein target and a ligand molecule. Unlike previous approaches (e.g., DeepDTA, GraphDTA) that rely on shallow representations or fixed descriptors, our model leverages the rich, contextual embeddings from Foundation Models to learn a **bipartite interaction function**.

**Input**: Unlike the classical models, this architecture processes the **full sequence embeddings** ($L \times D$) from Chapter 4, preserving the spatial and sequential context of every residue and atom.

### 6.1 The Cross-Attention Mechanism

The core of our architecture (`src.classifier.models.cross_attention_model.CrossAttentionAffinityModel`) is the Cross-Attention mechanism (Vaswani et al., 2017). Unlike self-attention, which models relationships within a sequence, cross-attention models relationships *between* two distinct sequences.

Let $\mathbf{H}_P \in \mathbb{R}^{L \times d}$ be the protein embedding sequence (acting as **Queries**) and $\mathbf{H}_L \in \mathbb{R}^{M \times d}$ be the ligand embedding sequence (acting as **Keys** and **Values**), projected to a common hidden dimension $d$. The attention weights $\mathbf{A} \in \mathbb{R}^{L \times M}$ are computed as:

$$ \mathbf{Q} = \mathbf{H}_P \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}_L \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}_L \mathbf{W}_V $$

$$ A_{ij} = \frac{\exp(\mathbf{Q}_i \cdot \mathbf{K}_j^T / \sqrt{d_k})}{\sum_{k=1}^M \exp(\mathbf{Q}_i \cdot \mathbf{K}_k^T / \sqrt{d_k})} $$

Where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$ are learnable projection matrices. The output context matrix $\mathbf{C}_P$ for the protein is then:

$$ \mathbf{C}_P = \mathbf{A} \mathbf{V} $$

**Biological Interpretation**: The attention weight $A_{ij}$ represents the learned probability of interaction between protein residue $i$ and ligand atom $j$. A high weight implies that the model considers this specific atom-residue pair critical for binding, effectively performing "soft docking" in latent space.

### 6.2 The DT-Kinase Architecture: CNN + Cross-Attention

The **DT-Kinase** architecture combines multi-scale Convolutional Neural Networks (CNNs) for local feature extraction with bidirectional Cross-Attention for global interaction modeling.

#### 6.2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Per-Token Embeddings                                                │
│  ├── Protein: ESM-2/ESM-C per-residue embeddings [batch, L, d_protein]     │
│  └── Ligand:  SMI-TED/MoLFormer per-token embeddings [batch, M, d_ligand]  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: LINEAR PROJECTION                                                 │
│  ├── Protein: W_p ∈ ℝ^(d_protein × hidden_dim)                             │
│  └── Ligand:  W_l ∈ ℝ^(d_ligand × hidden_dim)                              │
│  └─→ Output: [batch, seq_len, hidden_dim]                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: MULTI-SCALE CNN ENCODERS (Local Feature Extraction)               │
│                                                                             │
│  For each input (protein and ligand separately):                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Conv1DBlock(kernel=3) ──┐                                          │   │
│  │  Conv1DBlock(kernel=5) ──┼── Concatenate → Fusion → Residual        │   │
│  │  Conv1DBlock(kernel=7) ──┘                                          │   │
│  │                                                                      │   │
│  │  Conv1DBlock = Conv1D → BatchNorm → GELU → Dropout → Conv1D → BN   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  • Residual connections preserve feature hierarchy                          │
│  • LayerNorm for training stability                                         │
│  └─→ Output: [batch, seq_len, hidden_dim]                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: POSITIONAL ENCODING (Optional)                                    │
│                                                                             │
│  Option A: Sinusoidal (Vaswani et al., 2017)                               │
│    PE(pos, 2i) = sin(pos / 10000^(2i/d))                                   │
│    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))                                 │
│                                                                             │
│  Option B: RoPE (Rotary Position Embedding)                                 │
│    RoPE(x, m)_{2i} = x_{2i}cos(mθ_i) - x_{2i+1}sin(mθ_i)                  │
│    RoPE(x, m)_{2i+1} = x_{2i}sin(mθ_i) + x_{2i+1}cos(mθ_i)                │
│    where θ_i = 10000^(-2i/d)                                               │
│                                                                             │
│  RoPE Advantages: Unlimited sequence length, relative position awareness    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: BIDIRECTIONAL CROSS-ATTENTION (N=2 layers)                        │
│                                                                             │
│  For each layer:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  a) Protein → Ligand Attention                                      │   │
│  │     protein' = LayerNorm(protein + MultiHeadAttn(Q=protein,K=V=lig))│   │
│  │     protein' = LayerNorm(protein' + FFN(protein'))                  │   │
│  │                                                                      │   │
│  │  b) Ligand → Protein Attention                                      │   │
│  │     ligand' = LayerNorm(ligand + MultiHeadAttn(Q=ligand,K=V=prot)) │   │
│  │     ligand' = LayerNorm(ligand' + FFN(ligand'))                    │   │
│  │                                                                      │   │
│  │  FFN = Linear(d→4d) → GELU → Dropout → Linear(4d→d)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  • Multi-Head Attention (8 heads) for diverse interaction patterns          │
│  └─→ Output: Learned attention weights indicating residue-atom affinities   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: GLOBAL POOLING                                                    │
│                                                                             │
│  protein_pooled = mean(protein', dim=1)  → [batch, hidden_dim]             │
│  ligand_pooled = mean(ligand', dim=1)    → [batch, hidden_dim]             │
│  combined = concat(protein_pooled, ligand_pooled) → [batch, 2*hidden_dim]  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: MULTI-TASK PREDICTION HEAD                                        │
│                                                                             │
│  Shared: Linear(2*hidden → hidden) → LayerNorm → GELU → Dropout            │
│                        │                                                    │
│         ┌──────────────┴──────────────┐                                    │
│         ▼                             ▼                                    │
│  Classification Head            Regression Head                            │
│  Linear(hidden → hidden/2)      Linear(hidden → hidden/2)                  │
│  → GELU → Dropout               → GELU → Dropout                           │
│  → Linear(hidden/2 → 1)         → Linear(hidden/2 → 1)                     │
│  → Sigmoid                      → Identity                                 │
│         │                             │                                    │
│         ▼                             ▼                                    │
│  P(active) ∈ [0,1]              pChEMBL ∈ ℝ                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 CNN Encoder Mathematical Formulation

The CNN encoder extracts local patterns at multiple scales:

$$ \mathbf{H}^{(k)} = \text{Conv1D}(\mathbf{X}, \mathbf{W}^{(k)}, \text{kernel}=k) \quad \text{for } k \in \{3, 5, 7\} $$

Where each Conv1D operation with kernel size $k$ is:

$$ \mathbf{H}^{(k)}_i = \sigma\left(\sum_{j=0}^{k-1} \mathbf{W}^{(k)}_j \cdot \mathbf{X}_{i+j-\lfloor k/2 \rfloor} + \mathbf{b}^{(k)}\right) $$

The multi-scale features are fused:

$$ \mathbf{H}_{fused} = \text{Fusion}\left([\mathbf{H}^{(3)}; \mathbf{H}^{(5)}; \mathbf{H}^{(7)}]\right) + \mathbf{X} $$

Where the residual connection preserves the original information.

### 6.3 Key Design Principles

- **Primacy of Sequence**: No 3D coordinates required—information is encoded in primary sequence via PLM embeddings
- **Contextuality**: Transformer self-attention captures long-range dependencies and global sequence context
- **Semantic Compatibility**: Answers "How compatible are these latent representations?" rather than "How well does this geometrically fit?"
- **Scalability**: Inference is pure neural network forward pass, enabling trillion-compound screening against entire proteome
- **Universality**: Applicable to any protein with known sequence, including those without experimental structures

### 6.4 The Dual-Task Strategy: Classification & Regression

semantic-screening is designed to solve two distinct but related problems simultaneously: identifying *active* compounds (Classification) and predicting their *potency* (Regression).

#### 6.4.1 Binary Bioactivity Prediction
The primary goal is to filter the vast chemical space for potential hits. We define a binary label $y_{cls} \in \{0, 1\}$ based on a threshold (typically $pChEMBL \ge 6.0$ or $IC_{50} \le 1000nM$).
The model outputs a probability $p = \sigma(z_{cls})$ using a sigmoid activation.

**Handling Class Imbalance**: To address the inherent imbalance (inactives $\gg$ actives) typical of screening libraries, we employ a **Weighted Binary Cross-Entropy (BCE)** loss, assigning a higher penalty to false negatives:
$$ \mathcal{L}_{BCE} = - \frac{1}{N} \sum_{i=1}^N [w_{pos} \cdot y_i \log(p_i) + (1-y_i) \log(1-p_i)] $$

#### 6.4.2 Affinity Regression (Potency Prediction)
For active compounds, we need to rank them by potency. The target variable $y_{reg}$ is the $pChEMBL$ value ($-\log_{10}(IC_{50}/K_i)$).
The model outputs a continuous scalar $\hat{y}_{reg}$. We minimize the **Mean Squared Error (MSE)** loss:
$$ \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 $$

#### 6.4.3 Multi-Task Learning (MTL)
By training on both tasks simultaneously, the model learns a shared representation that captures features relevant to both binding (binary) and binding strength (continuous). This acts as a powerful regularizer.
$$ \mathcal{L}_{total} = \lambda_{cls} \mathcal{L}_{BCE} + \lambda_{reg} \mathcal{L}_{MSE} $$
Where $\lambda_{cls}=1.0$ and $\lambda_{reg}=0.5$ are the default task weights.

### 6.5 Training Strategy & Hyperparameters

The architecture is defined by a set of hyperparameters optimized for the kinase interaction task. These parameters balance model capacity with computational efficiency and regularization needs.

#### 6.5.1 Optimization Dynamics
*   **Optimizer**: We use **AdamW** (Adam with Decoupled Weight Decay) to prevent overfitting in the high-dimensional parameter space.
*   **Scheduler**: A **Linear Warmup** (first 10% of steps) followed by **Cosine Decay** is used to ensure stable convergence and escape local minima.
*   **Regularization**: In addition to Dropout ($p=0.1$), we apply Weight Decay ($1e-2$) to the projection layers.
*   **Gradient Clipping**: Maximum gradient norm of 1.0 to prevent exploding gradients.

#### 6.5.2 Hyperparameter Configuration (Default)

| Component | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Inputs** | Protein Dim | 640 | Dimension of ESM-2 (150M) embeddings |
| | Ligand Dim | 768 | Dimension of SMI-TED/MoLFormer embeddings |
| **Projections** | Hidden Dim | 256 | Common latent space dimension |
| | Activation | GELU | Gaussian Error Linear Unit |
| **CNN Encoder** | Layers | 3 | Number of convolutional blocks |
| | Kernels | (3, 5, 7) | Multi-scale kernel sizes |
| | Filters | 256 | Number of output channels per block |
| **Attention** | Heads | 8 | Number of parallel attention heads |
| | Layers | 2 | Number of stacked cross-attention blocks |
| | FF Dim | 1024 | Feed-forward hidden dimension (4× hidden) |
| | Dropout | 0.1 | Regularization rate |
| **Training** | Batch Size | 32 | Samples per batch |
| | Learning Rate | 1e-4 | Initial learning rate |
| | Weight Decay | 0.01 | L2 regularization |
| | Epochs | 500 | Maximum training epochs |
| | Patience | 30 | Early stopping patience |

This configuration results in a model with approximately **1.5M trainable parameters** (excluding the frozen foundation models), which is small enough to train on a single GPU yet expressive enough to capture complex interaction patterns.

---

## Chapter 7: Stratification & Validation Methodology

A pervasive issue in machine learning for biology is **data leakage** caused by molecular similarity. If similar compounds or proteins are distributed across training and test sets, a model can achieve high accuracy simply by "memorizing" chemical scaffolds rather than learning the physics of binding.

### 7.1 The Data Leakage Problem

Standard random splitting assumes independent and identically distributed (i.i.d.) data. However, drug discovery data is structured:
- **Compound families**: Many compounds share the same **Murcko scaffold** (core ring system)
- **Protein families**: Kinases share high sequence similarity (>85% in ATP-binding pocket)
- **Cross-contamination**: A random split might place Compound A in training and its close analog Compound B in test, inflating performance

### 7.2 Scaffold-Based Splitting (Primary Methodology)

**Implementation**: `scaffold_split.py` + `scaffolds_splits/scenario_splitter.py`

semantic-screening uses **Murcko scaffold decomposition** as the primary splitting strategy. This ensures that compounds sharing the same chemical backbone are never split across train/val/test sets, providing the most chemically meaningful separation.

#### 7.2.1 Murcko Scaffolds

A **Murcko scaffold** is the core ring system of a molecule after removing all side chains. For example, all benzodiazepines share the same scaffold regardless of their substituents. This captures the medicinal chemistry concept of "chemical series".

$$ \text{scaffold}: \mathcal{C} \to \mathcal{S}, \quad \text{scaffold}(c) = \text{MurckoScaffold}(\text{SMILES}(c)) $$

#### 7.2.2 Fixed Test Set with Shared Scaffolds

The splitting proceeds in two phases:

**Phase 1 — Test scaffold selection** (shared across human and non-human datasets):
1. Compute Murcko scaffolds for all compounds across both datasets
2. Select test scaffolds via optimization with random restarts to balance:
   - Target test fraction (~10% of unique compounds)
   - Class distribution preservation (active/inactive ratio)
   - Cross-dataset proportionality
3. All rows belonging to test scaffolds form the **fixed test set**

**Phase 2 — Scenario-specific train/val partitioning** (from the remainder pool):

| Scenario | Code | Splitting Unit | Disjointness Guarantee |
|----------|------|----------------|------------------------|
| **Scaffold** | `Sc` | Scaffold groups | No scaffold overlap between train and val |
| Random | `S1` | Individual rows | Stratified random (baseline) |
| Compound | `S2` | Unique compounds | No compound overlap |
| Kinase | `S3` | Unique kinases | No kinase overlap |
| New Comp. + New Kinase | `S4` | Both compounds and kinases | Double disjointness |

The **Scaffold (Sc)** scenario is the default and recommended split for all benchmarks. It provides the most chemically meaningful evaluation of generalization.

#### 7.2.3 Mathematical Formulation

Let $s(c)$ denote the Murcko scaffold of compound $c$, and let $s(x_i)$ denote the scaffold of the compound in sample $x_i$. The scaffold split ensures:

$$\mathcal{S}_{train} \cap \mathcal{S}_{val} = \emptyset, \quad \mathcal{S}_{train} \cap \mathcal{S}_{test} = \emptyset, \quad \mathcal{S}_{val} \cap \mathcal{S}_{test} = \emptyset$$

Where $\mathcal{S}_k = \{s(x_i) : x_i \in \mathcal{D}_k\}$ for $k \in \{train, val, test\}$.

The training set contains all samples whose scaffolds belong to the training scaffold set:

$$\mathcal{D}_{train} = \{(x_i, y_i) : s(x_i) \in \mathcal{S}_{train}\}$$

This guarantees that the model cannot exploit scaffold-level memorization during evaluation.

#### 7.2.4 Split Output Structure

```
scaffolds_splits/output/
    manifest.json                          # Full split metadata
    universal_scaffolds.json               # Scaffold assignments
    universal_test.tsv                     # Combined test set
    human_test.tsv                         # Human-specific test
    non_human_test.tsv                     # Non-human-specific test
    human_train.tsv / human_val.tsv        # Default (Sc) train/val
    non_human_train.tsv / non_human_val.tsv
    scenarios/
        Sc/                                # Scaffold-disjoint
            {dataset}_train.tsv
            {dataset}_val.tsv
        S1/ ... S4/                        # Other scenarios
    split_class_distribution_summary.csv   # Class balance report
```

#### 7.2.5 Validation of Split Integrity

The splitting system validates disjointness constraints automatically:
- **Sc**: No scaffold overlap between train and val
- **S2**: No compound overlap
- **S3**: No kinase overlap
- **S4**: No compound or kinase overlap

Class distribution is monitored across all splits with optimization to minimize class-rate deviation from the overall dataset.

### 7.3 Why Scaffold Splits Over Other Methods

| Aspect | Random | Compound | Scaffold | Comp.+Kinase |
|--------|--------|----------|----------|--------------|
| **Chemical series leakage** | Yes | Partial | **No** | **No** |
| **Compound leakage** | Yes | **No** | **No** | **No** |
| **Kinase leakage** | Yes | Yes | Yes | **No** |
| **Medicinal chemistry relevance** | Low | Medium | **High** | High |
| **Test set stability** | Variable | Variable | **Fixed** | Variable |
| **Recommended for benchmarks** | No | No | **Yes** | Ablation only |

The scaffold split strikes the optimal balance: it prevents the most common form of data leakage in drug discovery (same chemical series in train/test) while maintaining enough data in each split for reliable evaluation. The fixed shared test set across datasets also enables fair cross-dataset comparison.

### 7.5 Validation Metrics

We evaluate model performance using a comprehensive suite of metrics:

*   **Classification**:
    *   **AUC-ROC**: Area Under the Receiver Operating Characteristic curve.
    *   **F1-Score**: Harmonic mean of precision and recall.
    *   **MCC (Matthews Correlation Coefficient)**: A robust metric for imbalanced datasets, defined as:

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP) \times (TP+FN) \times (TN+FP) \times (TN+FN)}}$$

*   **Regression**:
    *   **RMSE (Root Mean Squared Error)**: Measures the average magnitude of error in $pChEMBL$ units.
    *   **Pearson Correlation ($r$)**: Measures linear correlation between predicted and actual affinity.
    *   **Concordance Index (CI)**: Measures the probability that the predicted order of affinities matches the true order.

### 7.6 Statistical Rigor: Multi-Seed Evaluation

To ensure reproducibility and statistical significance, semantic-screening uses multiple random seeds:

```python
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds
MIN_SEEDS_FOR_STATISTICS = 3
RECOMMENDED_SEEDS_FOR_PUBLICATION = 5
```

All results are reported as mean ± standard deviation across seeds.

---

## Chapter 8: Implementation & Engineering

semantic-screening is engineered not just as a research prototype, but as a scalable production system capable of handling industrial-scale datasets.

### 8.1 Scalability and Resource Management

#### 8.1.1 CPU Offloading
Large protein language models (e.g., ESM-2 15B) require VRAM far exceeding the capacity of standard consumer GPUs. We implement **CPU Offloading** using the `accelerate` library, which dynamically moves model layers between CPU RAM and GPU VRAM during the forward pass. This allows 15B parameter models to run on hardware with limited VRAM (e.g., 16GB), trading inference speed for accessibility.

#### 8.1.2 Distributed Processing with Spark
For data preprocessing and matrix construction, we utilize **Apache Spark** (via PySpark). This allows for parallel processing of millions of protein-ligand pairs, distributing the workload across available CPU cores. The system automatically configures Spark memory allocation based on the host environment (`src.build.core.constants.SPARK_CONFIG`).

### 8.2 Checkpointing and Caching

Given the computational cost of embedding generation, semantic-screening implements a granular checkpointing system.
*   **Embedding Cache**: Embeddings for unique proteins and ligands are cached on disk (`.npy` format). If a sequence reappears in a new dataset, its embedding is retrieved rather than recomputed.

### 8.3 Dynamic Dimension Synchronization

A significant engineering challenge in multi-modal learning is handling the varying dimensionality of upstream models. A 15B parameter protein model outputs 5120-dimensional vectors, while a standard ligand model outputs 768 dimensions.

semantic-screening implements a **Dynamic Dimension Synchronization** system within `src.build.core.config.BuildConfig`. This system automatically detects the selected model configuration and adjusts the input layers of the downstream neural networks accordingly.

```python
# Model dimension mapping (run_complete_pipeline.py)
protein_dims = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t33_650M_UR50D': 1280,
    'esm2_t36_3B_UR50D': 2560,
    'esm2_t48_15B_UR50D': 5120,
    'esmc-300m-2024-12': 960,
    'esmc-600m-2024-12': 1152,
    'esmc-6b-2024-12': 3072,
}
```

This ensures that the architecture is agnostic to the specific foundation model being used, facilitating rapid benchmarking and ablation studies.

---

## Chapter 9: Model Inventory & Complexity Analysis

To provide a comprehensive overview of the computational landscape within semantic-screening, we present a consolidated inventory of all machine learning models and transformers utilized. The models are categorized by their role (Representation vs. Prediction) and ordered by architectural complexity.

### 9.1 Foundation Models (Transformers)

These models serve as the feature extraction engine, transforming raw biological sequences into high-dimensional vector representations.

| Model | Variant | Parameters | Layers | Embedding Dim ($d$) | Architecture | Training Objective |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SMI-TED** | Standard | ~100M | 12 | 768 | Transformer Enc-Dec | Masked Language Modeling |
| **MoLFormer** | Standard | ~47M | 12 | 768 | Transformer Encoder | Masked Language Modeling |
| **ESM-2** | t6_8M | 8M | 6 | 320 | Transformer Encoder | Masked Language Modeling |
| **ESM-2** | t12_35M | 35M | 12 | 480 | Transformer Encoder | Masked Language Modeling |
| **ESM-2** | t30_150M | 150M | 30 | 640 | Transformer Encoder | Masked Language Modeling |
| **ESM-C** | 300M | 300M | 30 | 960 | Causal Decoder | Next Token Prediction |
| **ESM-2** | t33_650M | 650M | 33 | 1280 | Transformer Encoder | Masked Language Modeling |
| **ESM-C** | 600M | 600M | 36 | 1152 | Causal Decoder | Next Token Prediction |
| **ESM-2** | t36_3B | 3B | 36 | 2560 | Transformer Encoder | Masked Language Modeling |
| **ESM-C** | 6B | 6B | 56 | 3072 | Causal Decoder | Next Token Prediction |
| **ESM-2** | t48_15B | 15B | 48 | 5120 | Transformer Encoder | Masked Language Modeling |

### 9.2 Predictive Models (Classical & Deep)

These models consume the embeddings generated by the Foundation Models to perform the downstream tasks of classification and regression.

| Model | Type | Complexity | Key Hyperparameters | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | Probabilistic | $O(N)$ | `var_smoothing=1e-9` | Baseline probabilistic classifier. |
| **Decision Tree** | Tree | Low | `max_depth=20`, `class_weight='balanced'` | Single tree with interpretable splits. |
| **Logistic Regression** | Linear | Low | `C=1.0`, `solver='lbfgs'`, `penalty='l2'` | Linear decision boundary. |
| **Linear SVC** | Linear | Low | `C=1.0`, `loss='squared_hinge'` | Max-margin hyperplane. |
| **LightGBM** | Boosting | Medium | `n_estimators=100`, `max_depth=6`, `lr=0.1` | Gradient boosting with GOSS. |
| **XGBoost** | Boosting | Medium | `n_estimators=100`, `max_depth=6`, `lr=0.1` | Regularized gradient boosting. |
| **Extra Trees** | Ensemble | Medium | `n_estimators=100`, `max_depth=20` | Randomized decision trees. |
| **Random Forest** | Ensemble | High | `n_estimators=100`, `max_depth=20` | Bagging of decision trees. |
| **AdaBoost** | Boosting | High | `n_estimators=100`, `lr=0.5` | Adaptive boosting. |
| **KNN** | Instance-based | High | `n_neighbors=5`, `weights='distance'` | Proximity-based classification. |
| **Gradient Boosting** | Boosting | Very High | `n_estimators=100`, `max_depth=5` | Sequential error correction. |
| **MLP** | Neural Network | Very High | `hidden=(100, 50)`, `max_iter=50` | Multi-layer perceptron (2 layers). |
| **DT-Kinase** | Hybrid Deep Learning | **Extreme** | `heads=8`, `layers=2`, `hidden=256` | **End-to-End**: CNN (Local) + Cross-Attention (Global). |

---

## Chapter 10: Generative Lead Optimization (The Semantic Designer)

The ultimate goal of drug discovery is not merely to *identify* active compounds, but to *design* better ones. While the previous chapters focused on **Discriminative Modeling** (predicting $y$ given $x$), this chapter introduces **Generative Modeling** (generating $x'$ that maximizes $y$).

We introduce the **Generative Optimizer**, a module that leverages the bidirectional context of SMI-TED to perform **Semantic In-painting** on ligand structures.

### 10.1 The "Diagnose-and-Repair" Paradigm

Traditional lead optimization relies on medicinal chemists' intuition to modify functional groups. semantic-screening automates this via a two-step process:

1.  **Diagnosis (Attention-Guided Masking)**: Identifying the "weak links" in the protein-ligand interaction.
2.  **Repair (Masked Token Prediction)**: Using a chemical language model to suggest semantically valid replacements that improve affinity.

### 10.2 Step 1: Attention-Guided Masking

We utilize the **Cross-Attention Matrix** ($\mathbf{A} \in \mathbb{R}^{L \times M}$) generated by the architecture described in Chapter 6.
Let $A_{ij}$ be the attention weight between protein residue $i$ and ligand atom $j$. We compute the **Ligand Relevance Score** $R_j$ for each atom $j$:

$$ R_j = \max_{i} A_{ij} $$

Atoms with low $R_j$ scores contribute little to the binding interface. We define a mask $M$ where:

$$ M_j = \begin{cases} \text{[MASK]} & \text{if } R_j < \tau \text{ (Threshold)} \\ \text{Atom}_j & \text{otherwise} \end{cases} $$

This effectively "erases" the parts of the molecule that are not interacting effectively with the protein.

### 10.3 Step 2: Semantic In-painting with SMI-TED

SMI-TED is a **Masked Language Model (MLM)** based on the BERT architecture. Unlike Causal models (e.g., GPT) that generate text left-to-right, SMI-TED uses bidirectional context, making it ideal for **In-painting** (filling in gaps).

Given the masked SMILES sequence $S_{masked}$, the model predicts the probability distribution over the vocabulary $\mathcal{V}$ for each masked position:

$$ P(x_j | S_{masked}) = \text{SMI-TED}(S_{masked}) $$

We sample $K$ candidate tokens from this distribution, generating a set of new molecular variants $\{L'_1, L'_2, \dots, L'_K\}$.

### 10.4 Step 3: The Critic Loop

The generated variants are not guaranteed to be better binders. To verify this, we pass each candidate $L'_k$ back through the **Regression Module** (Chapter 6.4.2) to predict its affinity $\hat{y}_k$.

$$ L_{optimized} = \underset{L'_k}{\text{argmax}} \, \text{Regressor}(P, L'_k) $$

This closes the loop, creating a self-improving cycle where the model diagnoses its own weak interactions and proposes specific chemical modifications to fix them.

---

## Chapter 11: Unified Benchmark Pipeline

To facilitate reproducible and comprehensive model comparison, semantic-screening provides a **unified benchmark orchestrator** that coordinates all evaluation levels through a single entry point.

### 8.1 Three-Level Model Hierarchy

The benchmark evaluates models at three levels of increasing complexity and representational richness:

| Level | Input Representation | Models | Description |
|-------|---------------------|--------|-------------|
| **Level 1** | Molecular Fingerprints (ECFP) | KNN, MLP | Baseline using classical cheminformatics descriptors |
| **Level 2** | Mean-pooled Embedding Vectors | KNN, MLP | PLM-based fixed-size representations |
| **Level 3** | Per-token Embedding Matrices | CNN+CrossAttention | Full DT-Kinase architecture with sequence-level context |

This hierarchy answers a fundamental scientific question: **how much does each level of representation contribute to predictive performance?**

- **Level 1 vs Level 2**: Measures the value of PLM embeddings over hand-crafted fingerprints
- **Level 2 vs Level 3**: Measures the value of preserving per-residue/per-atom context (sequence-level vs pooled)

### 8.2 Benchmark Execution

**Entry point**: `semantic_screening_models_beta.py`

```bash
# Full benchmark (all 3 levels)
python semantic_screening_models_beta.py --dataset non_human --embedding 8M

# Specific levels
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 1,2

# Level 3 with custom hyperparameters
python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 3 --epochs 100
```

### 8.3 Benchmark Pipeline Flow

```
Step 0:  Verify/generate scaffold splits (scaffold_split.py)
Step 0b: Verify/extract ligand vectors (mean-pool MoLFormer matrices)
Step 1:  Level 1 — Fingerprint + KNN/MLP
Step 2:  Level 2 — Embedding vectors + KNN/MLP
Step 3:  Level 3 — Per-token matrices + CNN+CrossAttention (multi-seed)
Step 4:  Comparative report + visualizations
```

All levels use the **same scaffold split** to ensure fair comparison. Level 3 uses multi-seed evaluation (default: 5 seeds) for statistical robustness.

### 8.4 Comparative Visualizations

The benchmark generates five comparative plots:

1. **Grouped Bar Chart** (`benchmark_grouped_bar.png`): All metrics side-by-side per model, with error bars for multi-seed std
2. **Radar Chart** (`benchmark_radar.png`): Each model as a polygon overlaid for quick visual comparison of relative strengths
3. **Heatmap** (`benchmark_heatmap.png`): Color-coded models (rows) x metrics (columns), with $\pm$ std annotations
4. **MCC Ranking** (`benchmark_mcc_ranking.png`): Horizontal bars ranking models by MCC (the primary selection metric)
5. **Per-Metric Strip** (`benchmark_per_metric.png`): One panel per metric showing exact values per model

### 8.5 Output Structure

```
results/benchmark_{dataset}_{embedding}/
    level1_fingerprint/{dataset}/          # Fingerprint baseline results
    level2_embedding_{emb}/{dataset}/      # PLM vector results
    level3_cnn_crossattn_{emb}/            # DT-Kinase results
    benchmark_comparison.json              # Unified metrics table
    benchmark_grouped_bar.png              # Comparative plots
    benchmark_radar.png
    benchmark_heatmap.png
    benchmark_mcc_ranking.png
    benchmark_per_metric.png
```

---

## Conclusion

semantic-screening represents a holistic approach to the protein-ligand affinity prediction problem, implementing the theoretical framework developed in the PhD thesis "DT-Kinase: Semantic Screening of Protein-Ligand Interactions via Cross-Attention over Protein Language Model Embeddings". By synthesizing the representational power of foundation models (ESM-2, ESM-3/ESM-C, SMI-TED, MoLFormer) with the physics-inspired DT-Kinase Cross-Attention architecture and a rigorous scaffold-based validation methodology, it offers a robust platform for computational drug discovery that resolves the selectivity paradox through semantic compatibility in latent space rather than geometric fitting in 3D space. The three-level benchmark pipeline enables systematic evaluation of representation quality—from classical fingerprints through PLM vectors to full per-token cross-attention—providing clear scientific evidence for the contribution of each component. The modular design ensures that as the field advances—whether through better language models or novel attention mechanisms—semantic-screening can evolve, serving as a flexible platform for future research.

---

## References

1.  **Lin, Z., et al. (2023)**. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 379(6637), 1123-1130. (ESM-2)
2.  **Hayes, T., et al. (2024)**. *Simulating 500 million years of evolution with a language model*. bioRxiv. (ESM-3/ESM-C)
3.  **Vaswani, A., et al. (2017)**. *Attention is all you need*. Advances in neural information processing systems, 30.
4.  **Ross, J., et al. (2022)**. *Large-scale chemical language representations capture molecular structure and properties*. Nature Machine Intelligence, 4(12), 1256-1264. (SMI-TED/FM4M)
5.  **Eldridge, M. D., et al. (1997)**. *Empirical scoring functions: I. The development of a fast empirical scoring function to estimate the binding affinity of ligands in receptor complexes*. Journal of Computer-Aided Molecular Design, 11, 425-445.
6.  **Dosovitskiy, A., et al. (2020)**. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR. (ViT)
7.  **Wu, Z., et al. (2021)**. *MolFormer: Large-scale chemical language representations capture molecular structure and properties*. arXiv:2106.09553. (MoLFormer)
8.  **Su, J., et al. (2024)**. *RoFormer: Enhanced transformer with rotary position embedding*. Neurocomputing, 568, 127063. (RoPE)
