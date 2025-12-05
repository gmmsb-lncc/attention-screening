# Stratification Methodology: K-means Clustering and Cosine Similarity

This document provides a comprehensive explanation of the stratification methodology used in DockTKinase for splitting datasets into train, validation, and test sets while preventing data leakage.

## Table of Contents

1. [Overview](#overview)
2. [The Data Leakage Problem](#the-data-leakage-problem)
3. [Cosine Similarity](#cosine-similarity)
4. [Dynamic Threshold Selection](#dynamic-threshold-selection)
5. [K-means++ Clustering](#k-means-clustering)
6. [Cluster Assignment to Train/Val/Test](#cluster-assignment-to-trainvaltest)
7. [Mathematical Formulation](#mathematical-formulation)
8. [Implementation Details](#implementation-details)

---

## Overview

DockTKinase uses a **cluster-based stratification** approach to ensure that chemically similar molecules never appear in different splits (e.g., train and test). This prevents **data leakage** and produces realistic performance estimates.

The pipeline consists of three main steps:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. EMBEDDING GENERATION                                                 │
│     Protein (ESM-2) + Ligand (ChemBERTa) → Combined embedding           │
│                              │                                           │
│                              ▼                                           │
│  2. K-MEANS++ CLUSTERING                                                 │
│     Group similar samples into clusters using cosine similarity          │
│                              │                                           │
│                              ▼                                           │
│  3. CLUSTER-AWARE SPLITTING                                              │
│     Assign ENTIRE clusters to train/val/test (never split clusters)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Data Leakage Problem

### What is Data Leakage?

Data leakage occurs when information from the test set influences the training process, leading to overly optimistic performance estimates.

### Example in Drug Discovery

```
WITHOUT cluster-based splitting (WRONG):
─────────────────────────────────────────
Train set: Molecule A (IC50 = 10 nM, similar to B)
Test set:  Molecule B (IC50 = 12 nM, similar to A)

The model "memorizes" that embeddings like A → active
In test, it sees nearly identical embedding B → correctly predicts active
Result: AUC = 0.95 on test, but AUC = 0.60 on truly novel compounds


WITH cluster-based splitting (CORRECT):
────────────────────────────────────────
Train set: Molecules A, B, C (all similar, same cluster)
Test set:  Molecules X, Y, Z (structurally DIFFERENT cluster)

The model must GENERALIZE to unseen chemical scaffolds
Result: AUC reflects real-world performance on novel compounds
```

---

## Cosine Similarity

### Definition

Cosine similarity measures the angle between two embedding vectors:

$$\text{cosine\_similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \cdot ||\vec{B}||} = \cos(\theta)$$

### Properties

| Value | Interpretation |
|-------|----------------|
| 1.0 | Identical vectors (parallel) |
| 0.0 | Orthogonal vectors (unrelated) |
| -1.0 | Opposite vectors |

### Why Cosine Similarity for Embeddings?

1. **Scale invariance**: Compares direction (structural pattern), not magnitude
2. **High-dimensional suitability**: Works well with 768-1280 dimensional embeddings
3. **Computational efficiency**: O(d) where d is embedding dimension

### Conversion to Distance

For clustering algorithms that require distances:

$$d(\vec{A}, \vec{B}) = 1 - \text{cosine\_similarity}(\vec{A}, \vec{B})$$

---

## Dynamic Threshold Selection

### The Problem with Fixed Thresholds

A fixed similarity threshold (e.g., 0.7) fails because datasets have different similarity distributions:

```
Scenario 1: Homogeneous data (all kinases)
─────────────────────────────────────────
All proteins are kinases → very similar embeddings
Typical similarities: 0.88 to 0.99

With fixed threshold = 0.7:
→ ALL pairs have similarity > 0.7
→ Result: 1 giant cluster (useless for splitting)


Scenario 2: Diverse data (kinases + GPCRs + enzymes)
────────────────────────────────────────────────────
Different protein families → varied embeddings
Typical similarities: 0.30 to 0.85

With fixed threshold = 0.95:
→ Almost no pairs have similarity > 0.95
→ Result: Each sample is its own cluster (trivial)
```

### Adaptive Threshold Algorithm

The system analyzes the similarity distribution and selects an appropriate threshold:

#### Step 1: Analyze Similarity Distribution

```python
# Compute pairwise similarities for a sample
sim_matrix = cosine_similarity(embeddings)
similarities = sim_matrix[upper_triangle]  # Exclude diagonal

# Calculate statistics
stats = {
    'min': np.min(similarities),
    'max': np.max(similarities),
    'mean': np.mean(similarities),
    'p25': np.percentile(similarities, 25),
    'p50': np.percentile(similarities, 50),  # median
    'p75': np.percentile(similarities, 75),
    'p90': np.percentile(similarities, 90),
    'p95': np.percentile(similarities, 95),
    'p99': np.percentile(similarities, 99),
}
```

#### Step 2: Classify Data Homogeneity

```python
def classify_homogeneity(min_similarity):
    if min_similarity > 0.9:
        return 'very_high'  # All samples very similar
    elif min_similarity > 0.7:
        return 'high'       # Most samples similar
    elif min_similarity > 0.5:
        return 'moderate'   # Moderate diversity
    else:
        return 'low'        # High diversity
```

#### Step 3: Define Search Range

| Homogeneity | Search Range | Rationale |
|-------------|--------------|-----------|
| `very_high` | [p50, p99] | Need high threshold to separate similar samples |
| `high` | [p25, p95] | Moderate threshold range |
| `moderate` / `low` | [0.5, 0.95] | Standard range |

#### Step 4: Optimize Using Silhouette Score

The **Silhouette Score** measures clustering quality:

For each sample $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean intra-cluster distance (cohesion)
- $b(i)$ = mean distance to nearest cluster (separation)

**Interpretation:**
- $s(i) \approx +1$: Sample is well-clustered
- $s(i) \approx 0$: Sample is on cluster boundary
- $s(i) \approx -1$: Sample is likely in wrong cluster

**Global Silhouette Score:**

$$S = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Optimization Problem:**

$$\tau^* = \arg\max_{\tau \in [\tau_{min}, \tau_{max}]} S(\tau)$$

Subject to:
- $k(\tau) \geq k_{min}$ (minimum number of clusters)
- $|C_j| \geq n_{min}$ for all clusters $C_j$ (minimum cluster size)

### Example Threshold Selection

```
Dataset: 1000 protein embeddings (dim=1280)

Step 1: Similarity Statistics
─────────────────────────────
min  = 0.72    ← Used to classify homogeneity
mean = 0.84
max  = 0.99
p25  = 0.78
p75  = 0.89
p95  = 0.95

Step 2: Homogeneity Classification
──────────────────────────────────
min = 0.72 > 0.70 → homogeneity = 'high'

Step 3: Search Range
────────────────────
range = (p25, p95) = (0.78, 0.95)

Step 4: Grid Search with Silhouette Optimization
────────────────────────────────────────────────
┌───────────┬───────────┬─────────────┬───────────────┐
│ Threshold │ Distance  │ Num Clusters│ Silhouette    │
│    (τ)    │  (1-τ)    │             │ Score         │
├───────────┼───────────┼─────────────┼───────────────┤
│   0.78    │   0.22    │      8      │    0.42       │
│   0.80    │   0.20    │     12      │    0.51       │
│   0.82    │   0.18    │     18      │    0.58       │
│   0.84    │   0.16    │     25      │    0.64       │
│   0.86    │   0.14    │     35      │    0.71  ◄── BEST │
│   0.88    │   0.12    │     52      │    0.68       │
│   0.90    │   0.10    │     78      │    0.61       │
│   0.92    │   0.08    │    124      │    0.53       │
│   0.94    │   0.06    │    215      │    0.41       │
└───────────┴───────────┴─────────────┴───────────────┘

Result: τ* = 0.86 (Silhouette = 0.71)
```

---

## K-means++ Clustering

### Why K-means++?

K-means++ (Arthur & Vassilvitskii, 2007) provides:

1. **Theoretical guarantees**: O(log k) competitive ratio for centroid initialization
2. **Reproducibility**: Deterministic given random seed
3. **Wide adoption**: Standard in ML literature
4. **Better convergence**: Fewer iterations than random initialization

### Algorithm

#### Step 1: Combine Embeddings

```python
# Concatenate protein and ligand embeddings with weights
combined = np.concatenate([
    protein_embeddings * protein_weight,  # Default: 1.0
    ligand_embeddings * ligand_weight      # Default: 1.0
], axis=1)

# L2 normalize → cosine similarity behavior
combined = normalize(combined, norm='l2', axis=1)
```

#### Step 2: Determine Number of Clusters

```python
# Adaptive cluster count: sqrt(n), bounded [10, 1000]
n_clusters = min(1000, max(10, int(np.sqrt(n_samples))))
```

| Dataset Size | Number of Clusters |
|--------------|-------------------|
| 100 | 10 |
| 1,000 | 32 |
| 10,000 | 100 |
| 100,000 | 316 |
| 1,000,000 | 1,000 |

#### Step 3: Run MiniBatchKMeans

```python
kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    init='k-means++',           # Smart initialization
    batch_size=min(1024, n_samples),
    n_init='auto',
    max_iter=100,
    random_state=42             # Reproducibility
)
cluster_labels = kmeans.fit_predict(combined)
```

### K-means++ Initialization

Standard K-means uses random initialization, which can lead to poor convergence. K-means++ uses a smarter approach:

1. Choose first centroid uniformly at random
2. For each subsequent centroid:
   - Compute distance $D(x)$ from each point to nearest existing centroid
   - Choose next centroid with probability proportional to $D(x)^2$
3. Repeat until k centroids are chosen

This ensures centroids are spread out, leading to better clustering.

---

## Cluster Assignment to Train/Val/Test

### Key Principle

**Entire clusters are assigned to a single split — clusters are NEVER divided.**

This ensures that similar molecules (same cluster) always stay together.

### Algorithm: Greedy Assignment with Rebalancing

#### Step 1: Sort Clusters by Size

```python
# Sort clusters from largest to smallest
sorted_clusters = sorted(unique_clusters, key=lambda c: sizes[c], reverse=True)
```

**Rationale**: Assigning large clusters first gives better control over final proportions.

#### Step 2: Calculate Targets

```python
# For 80/10/10 split with 10,000 samples:
target_train = 8000  # 80%
target_val   = 1000  # 10%
target_test  = 1000  # 10%
```

#### Step 3: Greedy Assignment

For each cluster (largest to smallest):

```python
for cluster in sorted_clusters:
    size = cluster_sizes[cluster]
    
    # Check if TEST needs samples AND cluster fits (≤120% of target)
    if current_test < target_test and current_test + size <= target_test * 1.2:
        test_clusters.append(cluster)
        current_test += size
    
    # Check if VAL needs samples AND cluster fits
    elif current_val < target_val and current_val + size <= target_val * 1.2:
        val_clusters.append(cluster)
        current_val += size
    
    # Default: assign to TRAIN
    else:
        train_clusters.append(cluster)
        current_train += size
```

#### The "Needs AND Fits" Criterion

Two conditions must be satisfied:

**Condition 1: "Needs"** — The split hasn't reached its target yet
```python
current_test < target_test  # Test still needs more samples
```

**Condition 2: "Fits"** — Adding this cluster won't exceed 120% of target
```python
current_test + size <= target_test * 1.2
```

### Why 120% Tolerance?

Clusters are **indivisible** — you cannot take "half" a cluster.

```
WITHOUT tolerance (100% exact limit):
─────────────────────────────────────
target_test = 1000

Cluster A (450): 0 + 450 = 450   ≤ 1000? ✓ → TEST
Cluster B (380): 450 + 380 = 830  ≤ 1000? ✓ → TEST
Cluster C (320): 830 + 320 = 1150 ≤ 1000? ✗ → REJECTED!

Result: Test gets only 830 samples (83% of target)
        170 samples "missing" can never be filled


WITH tolerance (120% limit = 1200):
───────────────────────────────────
target_test = 1000, limit = 1200

Cluster A (450): 0 + 450 = 450   ≤ 1200? ✓ → TEST
Cluster B (380): 450 + 380 = 830  ≤ 1200? ✓ → TEST
Cluster C (320): 830 + 320 = 1150 ≤ 1200? ✓ → TEST

Result: Test gets 1150 samples (115% of target)
        Much closer to ideal!
```

### Visual Example

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ITERATION 1: Cluster C1 (450 samples)                                  │
│  ─────────────────────────────────────                                  │
│  Current state:                                                         │
│    Train: 0/8000    Val: 0/1000    Test: 0/1000                        │
│                                                                         │
│  Question: Who needs AND fits?                                          │
│    • Test needs 1000, 0+450=450 ≤ 1200 ✓                               │
│                                                                         │
│  Decision: C1 → TEST                                                    │
│  New state: Test: 450/1000                                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  ITERATION 5: Cluster C5 (250 samples)                                  │
│  ─────────────────────────────────────                                  │
│  Current state:                                                         │
│    Train: 0/8000    Val: 320/1000    Test: 1120/1000                   │
│                                                                         │
│  Question: Who needs AND fits?                                          │
│    • Test: 1120+250=1370 > 1200 ✗ (doesn't fit!)                       │
│    • Val: 320+250=570 ≤ 1200 ✓                                         │
│                                                                         │
│  Decision: C5 → VAL                                                     │
│  New state: Val: 570/1000                                               │
└─────────────────────────────────────────────────────────────────────────┘

... continues until all clusters assigned ...

┌─────────────────────────────────────────────────────────────────────────┐
│  FINAL STATE                                                            │
│  ───────────                                                            │
│    Train: 7830 samples (78.3%) ← 78 clusters                           │
│    Val:   1050 samples (10.5%) ← 12 clusters                           │
│    Test:  1120 samples (11.2%) ← 10 clusters                           │
│                                                                         │
│  ✓ Proportions: ~78/10.5/11.2 ≈ 80/10/10 (within tolerance)           │
│  ✓ All clusters INTACT in a single split                               │
│  ✓ No similar molecules divided between train and test                 │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Step 4: Rebalancing (if needed)

If proportions deviate >5% from target, move smaller clusters between splits:

```python
def rebalance_clusters(test_clusters, val_clusters, train_clusters, sizes):
    for _ in range(50):  # Max iterations
        # Calculate current proportions
        test_ratio = sum(sizes[c] for c in test_clusters) / total
        val_ratio = sum(sizes[c] for c in val_clusters) / total
        
        # Check if within 5% tolerance
        if abs(test_ratio - 0.10) <= 0.05 and abs(val_ratio - 0.10) <= 0.05:
            break  # Good enough
        
        # Move smallest cluster from over-represented to under-represented
        if test_ratio > 0.15:  # Test has too many
            smallest = min(test_clusters, key=lambda c: sizes[c])
            test_clusters.remove(smallest)
            val_clusters.append(smallest)  # or train_clusters
```

---

## Mathematical Formulation

### Complete Optimization Problem

**Objective:** Maximize clustering quality (Silhouette Score) while achieving balanced splits.

**Variables:**
- $\tau$ = similarity threshold (for hierarchical clustering) or $k$ = number of clusters (for K-means)
- $C = \{C_1, C_2, ..., C_k\}$ = resulting clusters
- $\pi: C \rightarrow \{train, val, test\}$ = cluster assignment function

**Objective Function:**

$$\max_{\tau, \pi} S(\tau) \cdot B(\pi)$$

Where:
- $S(\tau)$ = Silhouette Score for clustering with threshold $\tau$
- $B(\pi)$ = Balance score for assignment $\pi$

**Constraints:**

1. **Minimum clusters**: $k(\tau) \geq k_{min}$
2. **Minimum cluster size**: $|C_j| \geq n_{min}$ for all $j$
3. **Split proportions**: 
   - $0.76 \leq \frac{|\{i: \pi(C_i) = train\}|}{n} \leq 0.84$
   - $0.08 \leq \frac{|\{i: \pi(C_i) = val\}|}{n} \leq 0.12$
   - $0.08 \leq \frac{|\{i: \pi(C_i) = test\}|}{n} \leq 0.12$
4. **Cluster integrity**: Each cluster assigned to exactly one split

### Silhouette Score Computation

For sample $i$ in cluster $C_a$:

$$a(i) = \frac{1}{|C_a| - 1} \sum_{j \in C_a, j \neq i} d(i, j)$$

$$b(i) = \min_{C_b \neq C_a} \frac{1}{|C_b|} \sum_{j \in C_b} d(i, j)$$

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

$$S = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

---

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `src/build/pipeline/stratification_manager.py` | Main orchestrator |
| `src/build/stratification/similarity_analysis.py` | Similarity distribution analysis |
| `src/build/stratification/threshold_optimization.py` | Dynamic threshold selection |
| `src/build/stratification/adaptive_clustering.py` | Clustering algorithms |
| `src/build/stratification/scalable_clustering.py` | Memory-efficient clustering for large datasets |

### Configuration Parameters

```python
from src.build.pipeline.stratification_manager import StratificationManager

manager = StratificationManager(
    protein_weight=1.0,      # Weight for protein embeddings
    ligand_weight=1.0,       # Weight for ligand embeddings
    random_state=42,         # For reproducibility
)

splits = manager.stratify(
    protein_embeddings=protein_emb,  # (N, 1280)
    ligand_embeddings=ligand_emb,    # (N, 768)
    labels=activity_labels,           # (N,) or (N, cols)
    test_size=0.1,                    # 10% test
    val_size=0.1                      # 10% validation
)
```

### Memory Considerations

For datasets > 40,000 samples, full distance matrix computation is infeasible (O(n²) memory). The system automatically switches to:

1. **Representative sampling**: Select ~5,000 representative samples
2. **Sample clustering**: Cluster the sample
3. **Label propagation**: Assign all points to nearest centroid (O(n·k))

---

## References

1. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA '07*.
2. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*.
3. Sculley, D. (2010). Web-scale k-means clustering. *WWW '10*.

---

## Summary

| Component | Method | Rationale |
|-----------|--------|-----------|
| **Similarity metric** | Cosine similarity | Scale-invariant, suitable for high-dimensional embeddings |
| **Threshold** | Dynamic (Silhouette-optimized) | Adapts to dataset homogeneity |
| **Clustering** | K-means++ | Theoretical guarantees, reproducibility |
| **Cluster count** | √n (bounded 10-1000) | Scales with dataset size |
| **Split assignment** | Greedy with 120% tolerance | Achieves target proportions with indivisible clusters |
| **Cluster integrity** | Never split clusters | Prevents data leakage |
