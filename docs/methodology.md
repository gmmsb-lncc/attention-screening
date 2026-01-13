# semantic-screening: Semantic Interaction Prediction via Multi-Modal Foundation Models

**Author**: Leon Sulfierry (GMMSB-LNCC)  
**Date**: January 2026  
**Version**: 2.0 (Aligned with PhD Thesis)

---

## Abstract

The accurate identification of potent kinase inhibitors is a cornerstone of modern drug discovery, yet it remains a computationally challenging problem due to the high dimensionality of biological space and the scarcity of labeled structural data. This document presents **semantic-screening**, a modular, scalable, and scientifically rigorous deep learning platform designed to address these challenges. By integrating state-of-the-art Protein Language Models (ESM-2, ESM-C) and Chemical Foundation Models (SMI-TED) within the novel **DT-Kinase** architecture—a Cross-Attention Convolutional neural network—semantic-screening learns to predict both **binary bioactivity** (active/inactive) and **binding affinity** ($K_d, IC_{50}$) directly from sequence and SMILES representations. This approach enables high-throughput **candidate prioritization** by bypassing the need for explicit 3D co-crystal structures during inference, effectively performing "semantic docking" in a latent space. We introduce a mathematically grounded stratification methodology to mitigate data leakage, ensuring that performance metrics reflect true generalization capabilities across the kinaseome. This document details the theoretical foundations, architectural decisions, and implementation strategies that define the semantic-screening platform and DT-Kinase architecture.

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
    *   **Definition**: A compound is labeled $y=1$ (Active) if its affinity exceeds a threshold (e.g., $pChEMBL \ge 7.0$ or $IC_{50} \le 100nM$), and $y=0$ otherwise.
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

$$ P = \{a_1, a_2, ..., a_L\} \quad \text{where} \quad a_i \in \mathcal{A} = \{A, C, D, E, ...\} $$

This linear structure is remarkably similar to human language, where a sentence is a sequence of words. Just as the meaning of a word depends on its context within a sentence, the function of an amino acid depends on its neighbors and its position in the 3D structure.

#### 2.1.1 Protein Language Models (pLMs)
Traditional bioinformatics relied on alignment-based methods (like BLAST) to find evolutionary relationships. However, recent advances in NLP have given rise to **Protein Language Models (pLMs)**. These models, typically based on the **Transformer** architecture, are trained on billions of protein sequences to predict missing or masked amino acids.

By learning to predict the next amino acid in a sequence, the model implicitly learns the "grammar" of protein folding and function. The internal representation (embedding) generated by the model captures physicochemical properties (charge, hydrophobicity) and structural contacts without ever being explicitly trained on 3D coordinates.

### 2.2 The Language of Chemistry: Molecules as Graphs and Strings

Small molecule drugs (ligands) are fundamentally different from proteins. They are not linear polymers but defined by graph structures where atoms are nodes and chemical bonds are edges. To process these molecules with language models, we use linear string representations, most notably **SMILES** (Simplified Molecular Input Line Entry System).

A SMILES string encodes the molecular graph into a sequence of characters. For example, Benzene is represented as `c1ccccc1`.

$$ L = \{s_1, s_2, \dots, s_M\} \quad \text{where} \quad s_i \in \mathcal{S} = \{C, N, O, =, (, ), \dots\} $$

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

A critical architectural challenge in modern bioinformatics is the rapid pace of model evolution. To accommodate this, semantic-screening employs the **Strategy Design Pattern** for embedding generation. This allows the system to switch between different protein language models (e.g., ESM-2, ESM-C, Boltz-2) and ligand encoders (e.g., SMI-TED) at runtime, while maintaining a consistent API for the downstream pipelines.

#### 3.2.1 Protein Embedding Strategy

The `ProteinEmbedding` class acts as the context, delegating the actual computation to concrete strategy implementations derived from `BaseProteinStrategy`.

*   **Context**: `src.build.embeddings.protein_embedding.ProteinEmbedding`
*   **Interface**: `src.build.embeddings.strategies.base_protein_strategy.BaseProteinStrategy`
*   **Concrete Strategies**:
    *   `ESM2Strategy`: Wraps Meta AI's `fair-esm` library for models ranging from 8M to 15B parameters.
    *   `ESMCStrategy`: Integrates EvolutionaryScale's generative models (300M, 600M, 6B).
    *   `BoltzStrategy`: Adapts the Boltz-2 foundation model for structure-aware embeddings.

This design allows researchers to experiment with cutting-edge models simply by changing a configuration string (e.g., `--protein-model boltz2`), without modifying the core pipeline code.

### 3.3 Pipeline Orchestration

The `IntegratedPipeline` class (`src.integrated_pipeline.py`) serves as the master orchestrator. It manages the data flow between modules, handles checkpointing, and ensures that the output of the Build module (embedding matrices) is correctly formatted for the Classifier and Regression modules.

$$ \text{Raw Data} \xrightarrow{\text{Build}} \text{Embeddings} \xrightarrow{\text{Stratification}} \text{Splits} \xrightarrow{\text{Classifier}} \text{Candidates} \xrightarrow{\text{Regression}} \text{Predictions} $$

This linear flow is augmented by a robust **Checkpoint System**, which caches intermediate results (e.g., `embedding_matrix.npy`) to prevent redundant computations—a crucial feature when working with large-scale biological datasets that can take days to process.

---

## Chapter 4: Data Representation & Embeddings

The efficacy of any deep learning model is fundamentally limited by the quality of its input representations. semantic-screening eschews manual feature engineering (e.g., molecular fingerprints, physicochemical descriptors) in favor of learned representations from large-scale foundation models.

### 4.1 Protein Representation: The Language of Life

Proteins are treated as sequences of amino acids, analogous to sentences in natural language. We utilize **Protein Language Models (pLMs)** trained on billions of sequences (e.g., UniRef50) to extract embeddings that capture deep evolutionary and structural context.

#### 4.1.1 ESM-2 (Evolutionary Scale Modeling)
ESM-2 (Lin et al., 2023) is a BERT-style transformer trained with a Masked Language Modeling (MLM) objective. For a protein sequence $S = \{x_1, x_2, ..., x_L\}$, the model outputs a matrix $E \in \mathbb{R}^{L \times D}$, where $D$ is the embedding dimension.

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

#### 4.1.3 Boltz-2 (Structure-Aware)
While ESM models are sequence-based, Boltz-2 (Wohlwend et al., 2024) is a foundation model explicitly trained to predict 3D structures. By extracting embeddings from the **Pairformer** blocks, we obtain representations that are implicitly aware of spatial proximity ($d_{ij}$), even without explicit coordinate input. This provides a critical inductive bias for binding affinity prediction.

*   **Architecture**: Pairformer (Triangle Multiplicative Update + Axial Attention).
*   **Invariant Point Attention (IPA)**: A specialized attention mechanism that operates on 3D coordinates, ensuring invariance to rotation and translation (SE(3) invariance).
*   **Output**: Unlike standard Transformers that output a sequence $L \times D$, Boltz produces both single representations ($L \times D$) and pair representations ($L \times L \times C$), encoding the distance map directly.

### 4.2 Ligand Representation: Chemical Foundation Models

Small molecules are represented using SMILES (Simplified Molecular Input Line Entry System) strings. To process these, we employ **SMI-TED** (SMILES-based Transformer Encoder-Decoder), a model from the FM4M (Foundation Models for Molecules) suite.

*   **Tokenization**: SMILES strings are tokenized into chemical atoms and bond symbols (e.g., `C`, `N`, `=`, `(`).
*   **Architecture**: A Transformer encoder trained on large chemical databases (PubChem, ChEMBL).
*   **Output**: A sequence of vectors $L \in \mathbb{R}^{M \times 768}$, where $M$ is the number of atoms/tokens.

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

The core of our architecture (`src.attention_matrix.model.CrossAttentionModel`) is the Cross-Attention mechanism (Vaswani et al., 2017). Unlike self-attention, which models relationships within a sequence, cross-attention models relationships *between* two distinct sequences.

Let $H_P \in \mathbb{R}^{L \times d}$ be the protein embedding sequence (acting as **Queries**) and $H_L \in \mathbb{R}^{M \times d}$ be the ligand embedding sequence (acting as **Keys** and **Values**), projected to a common hidden dimension $d$. The attention weights $A \in \mathbb{R}^{L \times M}$ are computed as:

$$ Q = H_P W_Q, \quad K = H_L W_K, \quad V = H_L W_V $$

$$ A_{ij} = \frac{\exp(Q_i \cdot K_j^T / \sqrt{d_k})}{\sum_{k=1}^M \exp(Q_i \cdot K_k^T / \sqrt{d_k})} $$

Where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are learnable projection matrices. The output context matrix $C_P$ for the protein is then:

$$ C_P = A V $$

**Biological Interpretation**: The attention weight $A_{ij}$ represents the learned probability of interaction between protein residue $i$ and ligand atom $j$. A high weight implies that the model considers this specific atom-residue pair critical for binding, effectively performing "soft docking" in latent space.

### 6.2 The Hybrid CNN-Attention Architecture

While Transformers excel at capturing global dependencies, Convolutional Neural Networks (CNNs) are superior at extracting local features. In biological sequences, local motifs (e.g., binding sites, functional domains) are critical.

**Why CNNs on top of Transformers?**
While ESM-2 captures global evolutionary context, its embeddings are trained on the objective of *masked language modeling*, not binding. The CNN layers serve a critical dual purpose:
1.  **Task Adaptation**: They project the general-purpose evolutionary features into a binding-specific latent space.
2.  **Local Motif Enhancement**: They explicitly emphasize local physicochemical motifs (e.g., hydrophobic patches, charge clusters) that drive the initial stages of molecular recognition, filtering out evolutionary noise that is irrelevant to the specific binding task.

semantic-screening employs a **Hybrid Architecture** that combines the best of both worlds:

1.  **Local Feature Extraction (CNN)**: The raw embeddings from the Foundation Models are first passed through a multi-scale 1D-CNN encoder (`src.classifier.models.cnn_encoder`).
    *   **Kernels**: We use varying kernel sizes (3, 5, 7) to capture motifs of different lengths.
    *   **Length-Preserving Convolutions**: To ensure that the learned features can be mapped back to specific residues for interpretability, we employ padding strategies (e.g., `padding='same'`) that preserve the original sequence length $L$.
    *   **Residual Connections**: To facilitate gradient flow across deep networks.

$$ H_{local} = \text{CNN}(H_{raw}) $$

2.  **Global Interaction Modeling (Cross-Attention)**: The locally enriched features $H_{local}$ are then fed into the Cross-Attention mechanism described above.

### 6.3 Architecture Variants

#### 6.3.1 Improved Cross-Attention Model
To improve gradient flow and capacity, the `ImprovedCrossAttentionModel` incorporates:
*   **Deep Projections**: Multi-layer perceptrons (MLPs) with GELU activations before the attention block.
*   **Stacked Layers**: Multiple cross-attention blocks ($N=2$) to model higher-order interactions.
*   **Feed-Forward Networks (FFN)**: Transformer-style FFNs after each attention block.
*   **Layer Normalization**: Applied pre- and post-attention (Pre-LN) for training stability.

#### 6.3.2 Vision Transformer (ViT) Adaptation
We also explore a global context approach (`VisionTransformerModel`) where the protein and ligand sequences are concatenated into a single sequence $S_{joint} = [H_P; H_L]$. A learnable `[CLS]` token is prepended, and the entire sequence is processed by a standard Transformer Encoder. This allows for bidirectional information flow (Protein $\leftrightarrow$ Ligand) via self-attention, offering an alternative inductive bias.

### 6.4 The Dual-Task Strategy: Classification & Regression

semantic-screening is designed to solve two distinct but related problems simultaneously: identifying *active* compounds (Classification) and predicting their *potency* (Regression).

#### 6.4.1 Binary Bioactivity Prediction
The primary goal is to filter the vast chemical space for potential hits. We define a binary label $y_{cls} \in \{0, 1\}$ based on a threshold (typically $pChEMBL \ge 7.0$ or $IC_{50} \le 100nM$).
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
Where $\lambda_{cls}$ and $\lambda_{reg}$ are hyperparameters balancing the tasks (typically 1.0).

### 6.5 Training Strategy & Hyperparameters

The architecture is defined by a set of hyperparameters optimized for the kinase interaction task. These parameters balance model capacity with computational efficiency and regularization needs.

#### 6.5.1 Optimization Dynamics
*   **Optimizer**: We use **AdamW** (Adam with Decoupled Weight Decay) to prevent overfitting in the high-dimensional parameter space.
*   **Scheduler**: A **Linear Warmup** (first 10% of steps) followed by **Cosine Decay** is used to ensure stable convergence and escape local minima.
*   **Regularization**: In addition to Dropout ($p=0.2$), we apply Weight Decay ($1e-2$) to the projection layers.

#### 6.5.2 Hyperparameter Configuration (Base Model)

| Component | Parameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Inputs** | Protein Dim | 320 | Dimension of ESM-2 (8M) embeddings (Configurable) |
| | Ligand Dim | 768 | Dimension of SMI-TED embeddings |
| **Projections** | Hidden Dim | 256 | Common latent space dimension |
| | Activation | GELU | Gaussian Error Linear Unit |
| **Attention** | Heads | 8 | Number of parallel attention heads |
| | Layers | 2 | Number of stacked cross-attention blocks |
| | Dropout | 0.2 | Regularization rate |
| **CNN Encoder** | Layers | 3 | Number of convolutional blocks |
| | Kernels | (3, 5, 7) | Multi-scale kernel sizes |
| | Filters | 256 | Number of output channels per block |

This configuration results in a model with approximately **1.5M trainable parameters** (excluding the frozen foundation models), which is small enough to train on a single GPU yet expressive enough to capture complex interaction patterns.

---

## Chapter 7: Stratification & Validation Methodology

A pervasive issue in machine learning for biology is **data leakage** caused by evolutionary homology. Proteins often share high sequence similarity; if homologous proteins are distributed across training and test sets, a model can achieve high accuracy simply by "memorizing" the family rather than learning the physics of binding.

### 7.1 The Homology Problem

Standard random splitting assumes independent and identically distributed (i.i.d.) data. However, biological data is structured into families. A random split might place Kinase A in the training set and its close homolog Kinase B in the test set. Since they share 90% sequence identity and likely bind similar ligands, the test performance will be optimistically biased.

### 7.2 Adaptive Clustering Stratification

To address this, semantic-screening implements a rigorous **Clustering-based Stratification** strategy. The goal is to ensure that no cluster of similar proteins spans across the train/test boundary.

#### 7.2.1 Algorithm
We employ a rigorous **Clustering-based Stratification** strategy using **MiniBatchKMeans** with **k-means++** initialization. This approach ensures robust cluster centers and computational efficiency ($O(n)$) for large datasets.

1.  **Multi-Modal Embedding Integration**: We construct a unified representation vector for each interaction pair by concatenating the protein embedding ($E_P$) and ligand embedding ($E_L$), weighted by hyperparameters $\alpha$ and $\beta$ (typically 0.6 and 0.4):
    $$ V_{joint} = [\alpha \cdot E_P ; \beta \cdot E_L] $$

2.  **Cosine Similarity Approximation**: To cluster based on directional similarity (cosine similarity) rather than magnitude, we apply **L2-normalization** to the joint vectors:
    $$ \hat{V}_{joint} = \frac{V_{joint}}{||V_{joint}||_2} $$
    Clustering these normalized vectors with K-means is mathematically equivalent to clustering based on cosine similarity.

3.  **Adaptive Clustering**: We determine the optimal number of clusters $k$ adaptively based on dataset size ($k \approx \sqrt{N}$), bounded between 10 and 1000. This ensures that clusters are neither too coarse (high variance) nor too fine (overfitting).

4.  **Greedy Cluster Assignment**: To populate the Train, Validation, and Test sets, we employ a **Greedy Assignment Strategy**:
    *   Clusters are sorted by size (number of samples) in descending order.
    *   Iterating through the sorted clusters, we assign each *entire* cluster to the split (Test, Validation, or Train) that is currently furthest below its target quota (e.g., 10%, 10%, 80%).
    *   **Constraint**: A cluster is never split. All samples belonging to Cluster $C_i$ are assigned to the same set.

$$ \forall x \in \text{Train}, \forall y \in \text{Test}, \text{Cluster}(x) \neq \text{Cluster}(y) $$

This methodology guarantees that chemically and biologically similar instances are strictly separated, preventing the model from "memorizing" molecular families and ensuring that performance metrics reflect true generalization to novel chemical space.

### 7.3 Validation Metrics

We evaluate model performance using a comprehensive suite of metrics:

*   **Classification**:
    *   **AUC-ROC**: Area Under the Receiver Operating Characteristic curve.
    *   **F1-Score**: Harmonic mean of precision and recall.
    *   **MCC (Matthews Correlation Coefficient)**: A robust metric for imbalanced datasets.

*   **Regression**:
    *   **RMSE (Root Mean Squared Error)**: Measures the average magnitude of error in $pChEMBL$ units.
    *   **Pearson Correlation ($r$)**: Measures linear correlation between predicted and actual affinity.
    *   **Concordance Index (CI)**: Measures the probability that the predicted order of affinities matches the true order.

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
# Conceptual Logic
if model == 'esm2_t36_3B_UR50D':
    protein_dim = 2560
elif model == 'boltz2':
    protein_dim = 384
    
# Downstream Projection
self.protein_proj = nn.Linear(protein_dim, hidden_dim)
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
| **ESM-2** | t6_8M | 8M | 6 | 320 | Transformer Encoder | Masked Language Modeling |
| **ESM-2** | t12_35M | 35M | 12 | 480 | Transformer Encoder | Masked Language Modeling |
| **ESM-2** | t30_150M | 150M | 30 | 640 | Transformer Encoder | Masked Language Modeling |
| **ESM-C** | 300M | 300M | 30 | 960 | Causal Decoder | Next Token Prediction |
| **ESM-2** | t33_650M | 650M | 33 | 1280 | Transformer Encoder | Masked Language Modeling |
| **ESM-C** | 600M | 600M | 36 | 1152 | Causal Decoder | Next Token Prediction |
| **ESM-2** | t36_3B | 3B | 36 | 2560 | Transformer Encoder | Masked Language Modeling |
| **ESM-C** | 6B | 6B | 56 | 3072 | Causal Decoder | Next Token Prediction |
| **ESM-2** | t48_15B | 15B | 48 | 5120 | Transformer Encoder | Masked Language Modeling |
| **Boltz-2** | Standard | ~200M | 64 | 384 | Pairformer | Structure Prediction |

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

We utilize the **Cross-Attention Matrix** ($A \in \mathbb{R}^{L \times M}$) generated by the architecture described in Chapter 6.
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

$$ L_{optimized} = \operatorname*{argmax}_{L'_k} \text{Regressor}(P, L'_k) $$

This closes the loop, creating a self-improving cycle where the model diagnoses its own weak interactions and proposes specific chemical modifications to fix them.

---

## Conclusion

semantic-screening represents a holistic approach to the protein-ligand affinity prediction problem, implementing the theoretical framework developed in the PhD thesis "DT-Kinase: Semantic Screening of Protein-Ligand Interactions via Cross-Attention over Protein Language Model Embeddings". By synthesizing the representational power of foundation models (ESM-2, ESM-C, Boltz-2, SMI-TED) with the physics-inspired DT-Kinase Cross-Attention architecture and a rigorous validation methodology, it offers a robust platform for computational drug discovery that resolves the selectivity paradox through semantic compatibility in latent space rather than geometric fitting in 3D space. The modular design ensures that as the field advances—whether through better language models or novel attention mechanisms—semantic-screening can evolve, serving as a flexible platform for future research.

---

## References

1.  **Lin, Z., et al. (2023)**. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 379(6637), 1123-1130. (ESM-2)
2.  **Hayes, T., et al. (2024)**. *Simulating 500 million years of evolution with a language model*. bioRxiv. (ESM-C)
3.  **Wohlwend, J., et al. (2024)**. *Boltz-1: Democratizing Biomolecular Interaction Modeling*. arXiv:2411.00001. (Boltz)
4.  **Vaswani, A., et al. (2017)**. *Attention is all you need*. Advances in neural information processing systems, 30.
5.  **Ross, J., et al. (2022)**. *Large-scale chemical language representations capture molecular structure and properties*. Nature Machine Intelligence, 4(12), 1256-1264. (SMI-TED/FM4M)
6.  **Eldridge, M. D., et al. (1997)**. *Empirical scoring functions: I. The development of a fast empirical scoring function to estimate the binding affinity of ligands in receptor complexes*. Journal of Computer-Aided Molecular Design, 11, 425-445.
7.  **Dosovitskiy, A., et al. (2020)**. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR. (ViT)
