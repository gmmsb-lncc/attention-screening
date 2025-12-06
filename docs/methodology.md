# DockTKinase: Semantic Interaction Prediction via Multi-Modal Foundation Models

**Author**: DockTKinase Development Team  
**Date**: December 6, 2025  
**Version**: 1.0

---

## Abstract

The accurate identification of potent kinase inhibitors is a cornerstone of modern drug discovery, yet it remains a computationally challenging problem due to the high dimensionality of biological space and the scarcity of labeled structural data. This dissertation presents **DockTKinase**, a modular, scalable, and scientifically rigorous deep learning framework designed to address these challenges. By integrating state-of-the-art Protein Language Models (ESM-2, ESM-C) and Chemical Foundation Models (SMI-TED) within a novel Cross-Attention Convolutional architecture, DockTKinase learns to predict both **binding affinity** and **binary activity** directly from sequence and SMILES representations. This approach enables high-throughput **candidate prioritization** by bypassing the need for explicit 3D co-crystal structures during inference, effectively performing "semantic docking" in a latent space. We introduce a mathematically grounded stratification methodology to mitigate data leakage, ensuring that performance metrics reflect true generalization capabilities across the kinaseome. This document details the theoretical foundations, architectural decisions, and implementation strategies that define the DockTKinase system.

---

## Chapter 1: Introduction

### 1.1 The Kinase Drug Discovery Challenge

Protein kinases are enzymes that catalyze the transfer of a phosphate group from ATP to specific substrates (phosphorylation). This process acts as a molecular 'on/off' switch for cellular pathways. Dysregulation of kinases is a primary driver of cancer. The challenge is that the ATP-binding pocket is highly conserved across the >500 human kinases, making it difficult to design inhibitors that bind to just one (selectivity).

DockTKinase addresses this by treating the interaction problem as a **multi-modal representation learning task**.

### 1.2 The Binding Affinity Problem

Protein-ligand binding is governed by the laws of thermodynamics, specifically the Gibbs free energy of binding ($\Delta G_{bind}$), which relates to the dissociation constant ($K_d$) via the equation:

$$ \Delta G_{bind} = -RT \ln K_d $$

Where $R$ is the ideal gas constant and $T$ is the temperature. In computational drug discovery, the objective is to approximate this function $f(P, L) \to \mathbb{R}$, where $P$ represents the protein target and $L$ represents the ligand molecule.

Traditional approaches, such as molecular docking, rely on physics-based scoring functions that estimate enthalpic and entropic contributions based on 3D poses. While interpretable, these methods are computationally expensive and sensitive to structural inaccuracies. Conversely, "black-box" machine learning models often fail to generalize to novel protein families due to data leakage and inadequate representation learning.

### 1.3 From Docking to Language Modeling: The DockTKinase Philosophy

The name **DockTKinase** pays homage to **DockThor**, the renowned molecular docking platform developed by the GMMSB-LNCC group. However, a fundamental distinction exists in their methodologies. While DockThor relies on physics-based simulations and explicit 3D coordinate sampling to find optimal binding poses, DockTKinase adopts a purely **Natural Language Processing (NLP)** approach.

We treat biological interaction not as a geometric puzzle, but as a **semantic compatibility problem** between the "language" of protein sequences (amino acids) and the "language" of chemical structures (SMILES). By leveraging the attention mechanisms of Transformer models, DockTKinase infers interaction patterns from the evolutionary and chemical context embedded in these sequences, effectively performing "semantic docking" in a high-dimensional latent semantic space rather than in Euclidean space.

### 1.4 Motivation and Contribution

DockTKinase was developed to bridge the gap between high-throughput sequence data and structural insight. Our primary contributions are:

1.  **Unified Embedding Space**: We leverage large-scale pre-trained transformers (ESM-2, Boltz-2, SMI-TED) to map biological entities into dense vector spaces that capture evolutionary and physicochemical properties.
2.  **Bipartite Interaction Modeling**: We propose a hybrid CNN-Cross-Attention architecture that explicitly models the pairwise interactions between protein residues and ligand atoms, mimicking the physical reality of binding interfaces.
3.  **Rigorous Validation**: We implement an adaptive clustering-based stratification strategy (using DBSCAN and K-means) to enforce strict separation of homologous sequences between training and evaluation sets.
4.  **Scalable Engineering**: The system is built on a modular architecture supporting CPU offloading, dynamic dimension synchronization, and multi-GPU training, enabling the processing of datasets with millions of interactions.

---

## Chapter 2: Theoretical Foundations

Before detailing the specific architecture of DockTKinase, it is essential to establish the theoretical framework that underpins our approach. This chapter introduces the core concepts of biological language modeling, explaining how proteins and molecules can be treated as linguistic entities and how modern deep learning techniques can extract meaningful representations from them.

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

$$ L = \{s_1, s_2, \dots, s_M\} \quad \text{where} \quad s_i \in \mathcal{S} = \{C, N, O, =, \text{\#}, (, ), \dots\} $$

#### 2.2.2 Chemical Foundation Models
Similar to pLMs, **Chemical Foundation Models** are trained on massive databases of chemical structures (like PubChem or ChEMBL). They learn to understand chemical syntax and semantics, generating vector representations that capture molecular properties such as solubility, toxicity, and binding potential.

### 2.3 The Interaction Problem: From Lock-and-Key to Induced Fit

The classical view of protein-ligand interaction is the **"Lock and Key"** model, where a rigid protein pocket (lock) perfectly accommodates a specific ligand (key). However, biological reality is more complex. Proteins are dynamic; they breathe and change shape upon binding, a phenomenon known as **"Induced Fit"**.

Computational methods must account for this flexibility.
1.  **Molecular Docking**: Simulates the physical process of binding, exploring thousands of orientations (poses) and conformations. It is accurate but computationally expensive ($O(N^3)$ or worse).
2.  **Machine Learning**: Attempts to learn a function $f(Protein, Ligand) \to Affinity$ from data.

DockTKinase represents the next evolution of ML approaches. Instead of using fixed descriptors (hand-engineered features), we use the **contextual embeddings** from Foundation Models. By combining the "protein understanding" of pLMs with the "chemical understanding" of chemical models, we aim to predict interaction compatibility directly from the learned latent spaces.

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

Biologically, this allows the model to "attend" to distant residues that are close in 3D space (contact prediction) or functionally coupled (co-evolution), effectively learning the protein's contact map without explicit supervision.

#### 2.4.3 The Training Objective (Masked Language Modeling)
The models are trained by minimizing the negative log-likelihood of predicting masked tokens $\tilde{x}$:

$$ \mathcal{L}_{MLM} = - \sum_{i \in \mathcal{M}} \log P(x_i | S_{\setminus \mathcal{M}}; \theta) $$

Where $\mathcal{M}$ is the set of masked indices. This forces the model to learn the underlying probability distribution of protein sequences evolutionarily.

---

## Chapter 3: Computational Framework & Architecture

### 3.1 Modular Design Philosophy

The DockTKinase system adheres to the **Separation of Concerns** principle, dividing the complex workflow of affinity prediction into three distinct, loosely coupled modules. This modularity ensures maintainability, testability, and the flexibility to upgrade individual components without systemic disruption.

The architecture is composed of the following core subsystems:

1.  **Build Module (`src.build`)**: Responsible for data ingestion, embedding generation, and matrix construction. It acts as the ETL (Extract, Transform, Load) layer of the pipeline.
2.  **Classifier Module (`src.classifier`)**: A multi-model ensemble system designed for the binary classification task (Active vs. Inactive). It serves as a high-recall filter to identify potential binders.
3.  **Regression Module (`src.regression`)**: A precision-focused module that predicts quantitative affinity metrics ($K_i, K_d, IC_{50}$) for the candidates identified by the classifier.

### 3.2 The Strategy Pattern for Model Integration

A critical architectural challenge in modern bioinformatics is the rapid pace of model evolution. To accommodate this, DockTKinase employs the **Strategy Design Pattern** for embedding generation. This allows the system to switch between different protein language models (e.g., ESM-2, ESM-C, Boltz-2) and ligand encoders (e.g., SMI-TED) at runtime, while maintaining a consistent API for the downstream pipelines.

#### 3.2.1 Protein Embedding Strategy

The `ProteinEmbedding` class acts as the context, delegating the actual computation to concrete strategy implementations derived from `BaseProteinStrategy`.

*   **Context**: `src.build.embeddings.protein_embedding.ProteinEmbedding`
*   **Interface**: `src.build.embeddings.strategies.base_protein_strategy.BaseProteinStrategy`
*   **Concrete Strategies**:
    *   `ESM2Strategy`: Wraps Meta AI's `fair-esm` library for models ranging from 8M to 15B parameters.
    *   `ESMCStrategy`: Integrates EvolutionaryScale's generative models (300M, 600M, 6B).
    *   `BoltzStrategy`: Adapts the Boltz-2 foundation model for structure-aware embeddings.
    *   `OpenFoldStrategy`: Extracts representations from the OpenFold3 trunk.

This design allows researchers to experiment with cutting-edge models simply by changing a configuration string (e.g., `--protein-model boltz2`), without modifying the core pipeline code.

### 3.3 Pipeline Orchestration

The `IntegratedPipeline` class (`src.integrated_pipeline.py`) serves as the master orchestrator. It manages the data flow between modules, handles checkpointing, and ensures that the output of the Build module (embedding matrices) is correctly formatted for the Classifier and Regression modules.

$$ \text{Raw Data} \xrightarrow{\text{Build}} \text{Embeddings} \xrightarrow{\text{Stratification}} \text{Splits} \xrightarrow{\text{Classifier}} \text{Candidates} \xrightarrow{\text{Regression}} \text{Predictions} $$

This linear flow is augmented by a robust **Checkpoint System**, which caches intermediate results (e.g., `embedding_matrix.npy`) to prevent redundant computations—a crucial feature when working with large-scale biological datasets that can take days to process.

---

## Chapter 4: Data Representation & Embeddings

The efficacy of any deep learning model is fundamentally limited by the quality of its input representations. DockTKinase eschews manual feature engineering (e.g., molecular fingerprints, physicochemical descriptors) in favor of learned representations from large-scale foundation models.

### 3.1 Protein Representation: The Language of Life

Proteins are treated as sequences of amino acids, analogous to sentences in natural language. We utilize **Protein Language Models (pLMs)** trained on billions of sequences (e.g., UniRef50) to extract embeddings that capture deep evolutionary and structural context.

#### 3.1.1 ESM-2 (Evolutionary Scale Modeling)
ESM-2 (Lin et al., 2023) is a BERT-style transformer trained with a Masked Language Modeling (MLM) objective. For a protein sequence $S = \{x_1, x_2, ..., x_L\}$, the model outputs a matrix $E \in \mathbb{R}^{L \times D}$, where $D$ is the embedding dimension.

*   **Architecture**: Transformer Encoder with Rotary Position Embeddings (RoPE).
*   **Scale**: We support the full range of ESM-2 models, from 8M parameters ($D=320$) to 15B parameters ($D=5120$).
*   **Usage**: We extract the per-residue representations from the final hidden layer, providing a granular view of the protein surface.

#### 3.1.2 ESM-C (Generative Modeling)
ESM-C (Hayes et al., 2024) represents a shift towards generative modeling. Unlike ESM-2's bidirectional attention, ESM-C uses causal masking, allowing it to model the probability distribution of the next amino acid. This is particularly useful for capturing long-range dependencies and functional motifs that define binding pockets.

#### 3.1.3 Boltz-2 (Structure-Aware)
While ESM models are sequence-based, Boltz-2 (Wohlwend et al., 2024) is a foundation model explicitly trained to predict 3D structures. By extracting embeddings from the **Pairformer** blocks, we obtain representations that are implicitly aware of spatial proximity ($d_{ij}$), even without explicit coordinate input. This provides a critical inductive bias for binding affinity prediction.

### 3.2 Ligand Representation: Chemical Foundation Models

Small molecules are represented using SMILES (Simplified Molecular Input Line Entry System) strings. To process these, we employ **SMI-TED** (SMILES-based Transformer Encoder-Decoder), a model from the FM4M (Foundation Models for Molecules) suite.

*   **Tokenization**: SMILES strings are tokenized into chemical atoms and bond symbols (e.g., `C`, `N`, `=`, `(`).
*   **Architecture**: A Transformer encoder trained on large chemical databases (PubChem, ChEMBL).
*   **Output**: A sequence of vectors $L \in \mathbb{R}^{M \times 768}$, where $M$ is the number of atoms/tokens.

### 3.3 Dynamic Dimension Synchronization

A significant engineering challenge in multi-modal learning is handling the varying dimensionality of upstream models. A 15B parameter protein model outputs 5120-dimensional vectors, while a standard ligand model outputs 768 dimensions.

DockTKinase implements a **Dynamic Dimension Synchronization** system within `src.build.core.config.BuildConfig`. This system automatically detects the selected model configuration and adjusts the input layers of the downstream neural networks accordingly.

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

## Chapter 4: Deep Learning Architectures for Affinity Prediction

DockTKinase introduces a specialized neural architecture designed to model the physical interaction between a protein target and a ligand molecule. We frame this as a **bipartite interaction problem**, where the goal is to learn a weighting function $w_{ij}$ representing the contribution of protein residue $i$ and ligand atom $j$ to the total binding energy.

### 4.1 The Cross-Attention Mechanism

The core of our architecture (`src.attention_matrix.model.CrossAttentionModel`) is the Cross-Attention mechanism (Vaswani et al., 2017). Unlike self-attention, which models relationships within a sequence, cross-attention models relationships *between* two distinct sequences.

Let $H_P \in \mathbb{R}^{L \times d}$ be the protein embedding sequence and $H_L \in \mathbb{R}^{M \times d}$ be the ligand embedding sequence, projected to a common hidden dimension $d$. The attention weights $A \in \mathbb{R}^{L \times M}$ are computed as:

$$ A = \text{softmax}\left(\frac{(H_P W_Q)(H_L W_K)^T}{\sqrt{d_k}}\right) $$

Where $W_Q$ and $W_K$ are learnable projection matrices. The output context matrix $C_P$ for the protein is then:

$$ C_P = A (H_L W_V) $$

Intuitively, for each residue in the protein, the model computes a weighted sum of ligand features, where the weights represent the "relevance" or "interaction strength" of that ligand atom to the residue. This mimics the physical process of binding, where specific residues (the binding pocket) interact strongly with specific pharmacophores on the ligand.

### 4.2 Architecture Variants

#### 4.2.1 Improved Cross-Attention Model
To improve gradient flow and capacity, the `ImprovedCrossAttentionModel` incorporates:
*   **Deep Projections**: Multi-layer perceptrons (MLPs) with GELU activations before the attention block.
*   **Stacked Layers**: Multiple cross-attention blocks ($N=2$) to model higher-order interactions.
*   **Feed-Forward Networks (FFN)**: Transformer-style FFNs after each attention block.
*   **Layer Normalization**: Applied pre- and post-attention (Pre-LN) for training stability.

#### 4.2.2 Vision Transformer (ViT) Adaptation
We also explore a global context approach (`VisionTransformerModel`) where the protein and ligand sequences are concatenated into a single sequence $S_{joint} = [H_P; H_L]$. A learnable `[CLS]` token is prepended, and the entire sequence is processed by a standard Transformer Encoder. This allows for bidirectional information flow (Protein $\leftrightarrow$ Ligand) via self-attention, offering an alternative inductive bias.

### 4.3 Multi-Task Learning Head

The model is trained with a multi-task objective to simultaneously predict binary activity (Active/Inactive) and quantitative affinity ($pChEMBL$).

$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{BCE}(y_{cls}, \hat{y}_{cls}) + \lambda_2 \mathcal{L}_{MSE}(y_{reg}, \hat{y}_{reg}) $$

Where:
*   $\mathcal{L}_{BCE}$ is the Binary Cross-Entropy loss for classification.
*   $\mathcal{L}_{MSE}$ is the Mean Squared Error loss for regression.
*   $\lambda_1, \lambda_2$ are hyperparameters balancing the tasks.

This multi-task approach acts as a regularizer, forcing the model to learn representations that are robust enough to capture both the general distinction between binders/non-binders and the subtle differences in binding strength.

---

## Chapter 5: Stratification & Validation Methodology

A pervasive issue in machine learning for biology is **data leakage** caused by evolutionary homology. Proteins often share high sequence similarity; if homologous proteins are distributed across training and test sets, a model can achieve high accuracy simply by "memorizing" the family rather than learning the physics of binding.

### 5.1 The Homology Problem

Standard random splitting assumes independent and identically distributed (i.i.d.) data. However, biological data is structured into families. A random split might place Kinase A in the training set and its close homolog Kinase B in the test set. Since they share 90% sequence identity and likely bind similar ligands, the test performance will be optimistically biased.

### 5.2 Adaptive Clustering Stratification

To address this, DockTKinase implements a rigorous **Clustering-based Stratification** strategy. The goal is to ensure that no cluster of similar proteins spans across the train/test boundary.

#### 5.2.1 Algorithm
We employ unsupervised clustering algorithms (DBSCAN or K-means) on the protein embedding space to identify families.

1.  **Embedding**: Compute protein embeddings $E_P$ using ESM-2.
2.  **Dimensionality Reduction**: Apply PCA or UMAP to reduce noise.
3.  **Clustering**: Group proteins into clusters $C = \{c_1, c_2, ..., c_k\}$ such that proteins within a cluster share high similarity.
4.  **Splitting**: Assign entire clusters to either Train, Validation, or Test sets.

$$ \forall x \in \text{Train}, \forall y \in \text{Test}, \text{Cluster}(x) \neq \text{Cluster}(y) $$

This forces the model to generalize to unseen protein families, providing a realistic estimate of its performance in drug discovery scenarios where novel targets are common.

### 5.3 Validation Metrics

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

## Chapter 6: Implementation & Engineering

DockTKinase is engineered not just as a research prototype, but as a scalable production system capable of handling industrial-scale datasets.

### 6.1 Scalability and Resource Management

#### 6.1.1 CPU Offloading
Large protein language models (e.g., ESM-2 15B) require VRAM far exceeding the capacity of standard consumer GPUs. We implement **CPU Offloading** using the `accelerate` library, which dynamically moves model layers between CPU RAM and GPU VRAM during the forward pass. This allows 15B parameter models to run on hardware with limited VRAM (e.g., 16GB), trading inference speed for accessibility.

#### 6.1.2 Distributed Processing with Spark
For data preprocessing and matrix construction, we utilize **Apache Spark** (via PySpark). This allows for parallel processing of millions of protein-ligand pairs, distributing the workload across available CPU cores. The system automatically configures Spark memory allocation based on the host environment (`src.build.core.constants.SPARK_CONFIG`).

### 6.2 Checkpointing and Caching

Given the computational cost of embedding generation, DockTKinase implements a granular checkpointing system.
*   **Embedding Cache**: Embeddings for unique proteins and ligands are cached on disk (`.npy` format). If a sequence reappears in a new dataset, its embedding is retrieved rather than recomputed.
*   **Pipeline State**: The `IntegratedPipeline` saves its state after each major phase (Build, Classify, Regress). In the event of a failure, execution can resume from the last successful checkpoint.

---

## Conclusion

DockTKinase represents a holistic approach to the protein-ligand affinity prediction problem. By synthesizing the representational power of foundation models (ESM, Boltz, SMI-TED) with a physics-inspired Cross-Attention architecture and a rigorous validation methodology, it offers a robust tool for computational drug discovery. The modular design ensures that as the field advances—whether through better language models or novel attention mechanisms—DockTKinase can evolve, serving as a flexible platform for future research.

---

## References

1.  **Lin, Z., et al. (2023)**. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 379(6637), 1123-1130. (ESM-2)
2.  **Hayes, T., et al. (2024)**. *Simulating 500 million years of evolution with a language model*. bioRxiv. (ESM-C)
3.  **Wohlwend, J., et al. (2024)**. *Boltz-1: Democratizing Biomolecular Interaction Modeling*. arXiv:2411.00001. (Boltz)
4.  **Vaswani, A., et al. (2017)**. *Attention is all you need*. Advances in neural information processing systems, 30.
5.  **Ross, J., et al. (2022)**. *Large-scale chemical language representations capture molecular structure and properties*. Nature Machine Intelligence, 4(12), 1256-1264. (SMI-TED/FM4M)
6.  **Eldridge, M. D., et al. (1997)**. *Empirical scoring functions: I. The development of a fast empirical scoring function to estimate the binding affinity of ligands in receptor complexes*. Journal of Computer-Aided Molecular Design, 11, 425-445.
7.  **Dosovitskiy, A., et al. (2020)**. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR. (ViT)
