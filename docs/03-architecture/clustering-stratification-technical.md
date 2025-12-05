# Clustering and Stratification: Technical Deep Dive

This document provides a detailed technical explanation of the clustering and stratification algorithms used in DockTKinase, with emphasis on the mathematical foundations and implementation choices.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Cosine Similarity in High-Dimensional Spaces](#cosine-similarity-in-high-dimensional-spaces)
3. [Dynamic Threshold Optimization](#dynamic-threshold-optimization)
4. [Silhouette Score Analysis](#silhouette-score-analysis)
5. [K-means++ Theoretical Guarantees](#k-means-theoretical-guarantees)
6. [Greedy Cluster Assignment Algorithm](#greedy-cluster-assignment-algorithm)
7. [Complexity Analysis](#complexity-analysis)
8. [Edge Cases and Robustness](#edge-cases-and-robustness)

---

## Mathematical Foundations

### Embedding Space

Protein and ligand embeddings are represented as vectors in high-dimensional spaces:

- **Protein embeddings**: $\vec{p} \in \mathbb{R}^{d_p}$ where $d_p = 1280$ (ESM-2)
- **Ligand embeddings**: $\vec{l} \in \mathbb{R}^{d_l}$ where $d_l = 768$ (ChemBERTa)

### Combined Embedding

The combined embedding is formed by weighted concatenation:

$$\vec{x} = [\alpha_p \cdot \vec{p} \; || \; \alpha_l \cdot \vec{l}] \in \mathbb{R}^{d_p + d_l}$$

Where:
- $\alpha_p$ = protein weight (default: 1.0)
- $\alpha_l$ = ligand weight (default: 1.0)
- $||$ denotes concatenation

After L2 normalization:

$$\hat{\vec{x}} = \frac{\vec{x}}{||\vec{x}||_2}$$

---

## Cosine Similarity in High-Dimensional Spaces

### Definition

For two vectors $\vec{a}, \vec{b} \in \mathbb{R}^d$:

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{||\vec{a}||_2 \cdot ||\vec{b}||_2} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \cdot \sqrt{\sum_{i=1}^{d} b_i^2}}$$

### Properties

1. **Bounded range**: $\cos(\theta) \in [-1, 1]$
2. **Symmetry**: $\cos(\vec{a}, \vec{b}) = \cos(\vec{b}, \vec{a})$
3. **Scale invariance**: $\cos(k\vec{a}, \vec{b}) = \cos(\vec{a}, \vec{b})$ for $k > 0$

### Conversion to Distance

For clustering algorithms requiring a distance metric:

$$d(\vec{a}, \vec{b}) = 1 - \cos(\vec{a}, \vec{b})$$

This yields:
- $d \in [0, 2]$
- $d = 0$ when vectors are identical
- $d = 1$ when vectors are orthogonal
- $d = 2$ when vectors are opposite

### Why Not Euclidean Distance?

In high-dimensional spaces, Euclidean distance suffers from the **curse of dimensionality**:

$$\lim_{d \to \infty} \frac{d_{max} - d_{min}}{d_{min}} \to 0$$

All points become approximately equidistant. Cosine similarity focuses on **angular difference**, which remains meaningful in high dimensions.

---

## Dynamic Threshold Optimization

### Motivation

The similarity threshold $\tau$ determines cluster granularity:
- **High $\tau$** (e.g., 0.95): Many small, tight clusters
- **Low $\tau$** (e.g., 0.50): Few large, loose clusters

The optimal $\tau$ depends on data distribution.

### Similarity Distribution Analysis

Given embeddings $X = \{\vec{x}_1, ..., \vec{x}_n\}$, compute:

$$S_{ij} = \cos(\vec{x}_i, \vec{x}_j) \quad \forall i < j$$

This yields $\frac{n(n-1)}{2}$ pairwise similarities.

For large $n$, sample $m$ points ($m \approx 2000$):

$$\hat{S} = \{S_{ij} : i, j \in \text{sample}, i < j\}$$

### Statistical Measures

From $\hat{S}$, compute:

| Statistic | Formula | Purpose |
|-----------|---------|---------|
| Minimum | $S_{min} = \min(\hat{S})$ | Homogeneity classification |
| Mean | $\bar{S} = \frac{1}{|\hat{S}|}\sum_{s \in \hat{S}} s$ | Central tendency |
| Percentiles | $P_q$ such that $q\%$ of $\hat{S} \leq P_q$ | Threshold candidates |

### Homogeneity Classification

```
┌────────────────────────────────────────────────────────────────────┐
│                    S_min values and classification                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  0.0        0.5        0.7        0.9        1.0                   │
│   │──────────│──────────│──────────│──────────│                    │
│   │          │          │          │          │                    │
│   │   LOW    │ MODERATE │   HIGH   │VERY HIGH │                    │
│   │          │          │          │          │                    │
│   │  τ: 0.4-0.7   τ: 0.5-0.9   τ: P25-P95  τ: P50-P99             │
│   │          │          │          │          │                    │
└────────────────────────────────────────────────────────────────────┘
```

### Search Range Definition

| Homogeneity | $\tau_{min}$ | $\tau_{max}$ | Rationale |
|-------------|--------------|--------------|-----------|
| `very_high` | $P_{50}$ | $P_{99}$ | Need high threshold to create any separation |
| `high` | $P_{25}$ | $P_{95}$ | Moderate range |
| `moderate` | 0.50 | 0.95 | Standard range |
| `low` | 0.40 | 0.90 | Lower thresholds work |

---

## Silhouette Score Analysis

### Definition

For sample $i$ assigned to cluster $C_a$:

**Intra-cluster distance (cohesion):**
$$a(i) = \frac{1}{|C_a| - 1} \sum_{j \in C_a, j \neq i} d(i, j)$$

**Inter-cluster distance (separation):**
$$b(i) = \min_{C_b \neq C_a} \left[ \frac{1}{|C_b|} \sum_{j \in C_b} d(i, j) \right]$$

**Silhouette coefficient:**
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Global score:**
$$S = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

### Interpretation

| $s(i)$ value | Interpretation |
|--------------|----------------|
| $\approx +1$ | Sample is far from neighboring clusters (well-clustered) |
| $\approx 0$ | Sample is on cluster boundary |
| $\approx -1$ | Sample is closer to another cluster (misclassified) |

### Optimization Algorithm

```python
def optimize_threshold(embeddings, threshold_candidates):
    """
    Find τ* that maximizes Silhouette Score.
    
    Time complexity: O(k × n² × d)
    where k = number of candidates, n = samples, d = dimensions
    """
    best_score = -1
    best_threshold = None
    
    # Precompute distance matrix (O(n² × d))
    D = pairwise_distances(embeddings, metric='cosine')
    
    for τ in threshold_candidates:  # O(k) iterations
        # Cluster with threshold τ
        labels = agglomerative_clustering(D, distance_threshold=1-τ)
        
        # Compute Silhouette Score (O(n²))
        score = silhouette_score(D, labels, metric='precomputed')
        
        # Validate constraints
        n_clusters = len(set(labels) - {-1})
        if n_clusters >= min_clusters and score > best_score:
            best_score = score
            best_threshold = τ
    
    return best_threshold
```

### Silhouette Score vs Threshold

Typical relationship:

```
Silhouette
Score
   ▲
1.0│
   │                    ●
   │                 ●     ●
0.5│              ●           ●
   │           ●                 ●
   │        ●                       ●
0.0│     ●                             ●
   │  ●                                   ●
-0.5│
   └──────────────────────────────────────────► Threshold (τ)
     0.5   0.6   0.7   0.8   0.9   1.0
                       ↑
                 Optimal τ*
```

- **Too low τ**: Few large clusters → poor cohesion → low $S$
- **Too high τ**: Many singleton clusters → poor separation metric → low $S$
- **Optimal τ**: Balance between cohesion and separation → maximum $S$

---

## K-means++ Theoretical Guarantees

### Standard K-means Objective

Minimize within-cluster sum of squares (WCSS):

$$\min_{\mu_1, ..., \mu_k} \sum_{j=1}^{k} \sum_{\vec{x} \in C_j} ||\vec{x} - \mu_j||^2$$

Where $\mu_j$ is the centroid of cluster $C_j$.

### K-means++ Initialization

**Standard K-means** uses random initialization, which can lead to arbitrarily bad clusterings.

**K-means++** provides probabilistic guarantees:

1. Choose first centroid $\mu_1$ uniformly at random from $X$
2. For $j = 2, ..., k$:
   - Compute $D(\vec{x}) = \min_{i<j} ||\vec{x} - \mu_i||^2$ for each $\vec{x} \in X$
   - Choose $\mu_j = \vec{x}$ with probability $\frac{D(\vec{x})}{\sum_{\vec{y}} D(\vec{y})}$

### Theoretical Result (Arthur & Vassilvitskii, 2007)

Let $\phi_{OPT}$ be the optimal WCSS. K-means++ initialization gives expected WCSS:

$$\mathbb{E}[\phi] \leq 8(\ln k + 2) \cdot \phi_{OPT}$$

This is $O(\log k)$ competitive ratio.

### Why MiniBatchKMeans?

For large datasets, standard K-means is $O(n \cdot k \cdot d \cdot I)$ where $I$ = iterations.

MiniBatchKMeans uses stochastic updates:
1. Sample mini-batch $B$ of size $b$ (e.g., 1024)
2. Update centroids using only $B$
3. Repeat

Time complexity: $O(b \cdot k \cdot d \cdot I)$ — independent of $n$.

---

## Greedy Cluster Assignment Algorithm

### Problem Formulation

**Given:**
- Clusters $C = \{C_1, ..., C_k\}$ with sizes $|C_1|, ..., |C_k|$
- Target proportions: $p_{train} = 0.80$, $p_{val} = 0.10$, $p_{test} = 0.10$
- Total samples: $n = \sum_j |C_j|$

**Find:**
- Assignment $\pi: C \to \{train, val, test\}$
- Minimizing deviation from target proportions
- Subject to: each cluster assigned to exactly one split

### Greedy Algorithm

```python
def greedy_assignment(clusters, sizes, targets, tolerance=1.2):
    """
    Greedy cluster assignment with tolerance.
    
    Args:
        clusters: List of cluster IDs
        sizes: Dict mapping cluster_id -> size
        targets: Dict {train: 8000, val: 1000, test: 1000}
        tolerance: Maximum overfill ratio (default 1.2 = 120%)
    
    Returns:
        assignment: Dict {train: [...], val: [...], test: [...]}
    """
    # Sort by size descending
    sorted_clusters = sorted(clusters, key=lambda c: sizes[c], reverse=True)
    
    assignment = {split: [] for split in ['train', 'val', 'test']}
    current = {split: 0 for split in ['train', 'val', 'test']}
    limits = {split: targets[split] * tolerance for split in targets}
    
    for cluster in sorted_clusters:
        size = sizes[cluster]
        assigned = False
        
        # Priority: test → val → train
        for split in ['test', 'val', 'train']:
            needs = current[split] < targets[split]
            fits = current[split] + size <= limits[split]
            
            if needs and fits:
                assignment[split].append(cluster)
                current[split] += size
                assigned = True
                break
        
        # Default to train if no other split works
        if not assigned:
            assignment['train'].append(cluster)
            current['train'] += size
    
    return assignment
```

### Why Tolerance = 120%?

**Problem**: Clusters are indivisible. Strict adherence to 100% limits leaves gaps.

**Example:**

```
Target: test = 1000 samples
Cluster sizes: [450, 380, 320, ...]

Without tolerance (limit = 1000):
  C1 (450): 0 + 450 = 450 ≤ 1000 ✓ → test = 450
  C2 (380): 450 + 380 = 830 ≤ 1000 ✓ → test = 830
  C3 (320): 830 + 320 = 1150 > 1000 ✗ → REJECTED
  
  Result: test = 830 (83% of target, 17% short)

With tolerance (limit = 1200):
  C1 (450): 0 + 450 = 450 ≤ 1200 ✓ → test = 450
  C2 (380): 450 + 380 = 830 ≤ 1200 ✓ → test = 830
  C3 (320): 830 + 320 = 1150 ≤ 1200 ✓ → test = 1150
  
  Result: test = 1150 (115% of target, acceptable deviation)
```

### Tolerance Analysis

| Tolerance | Max Overfill | Gap Risk | Balance |
|-----------|--------------|----------|---------|
| 100% | 0% | High (15-20%) | Poor |
| 110% | 10% | Medium (5-10%) | Moderate |
| **120%** | 20% | Low (2-5%) | **Good** |
| 150% | 50% | Very Low | Over-tolerance |

120% provides the best trade-off.

### Rebalancing Step

If initial assignment is too unbalanced (>5% deviation):

```python
def rebalance(assignment, sizes, targets, max_iterations=50):
    """
    Move small clusters from over-represented to under-represented splits.
    """
    for _ in range(max_iterations):
        # Compute current proportions
        totals = {s: sum(sizes[c] for c in assignment[s]) for s in assignment}
        total = sum(totals.values())
        
        # Find most over- and under-represented
        deviations = {s: totals[s]/total - targets[s]/sum(targets.values()) 
                      for s in assignment}
        
        over = max(deviations, key=deviations.get)
        under = min(deviations, key=deviations.get)
        
        # Check if within tolerance
        if abs(deviations[over]) <= 0.05 and abs(deviations[under]) <= 0.05:
            break
        
        # Move smallest cluster from over to under
        if assignment[over]:
            smallest = min(assignment[over], key=lambda c: sizes[c])
            assignment[over].remove(smallest)
            assignment[under].append(smallest)
    
    return assignment
```

---

## Complexity Analysis

### Time Complexity

| Step | Complexity | Notes |
|------|------------|-------|
| Similarity matrix | $O(m^2 \cdot d)$ | $m$ = sample size (≤5000), $d$ = embedding dim |
| Threshold search | $O(k \cdot m^2)$ | $k$ = threshold candidates (≈10) |
| K-means++ | $O(n \cdot c \cdot d \cdot I)$ | $c$ = clusters, $I$ = iterations |
| Cluster assignment | $O(c \log c)$ | Sorting + linear scan |
| Index mapping | $O(n)$ | Map clusters to sample indices |

**Overall**: $O(n \cdot c \cdot d)$ — linear in dataset size.

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Embeddings | $O(n \cdot d)$ | Input data |
| Sample similarity matrix | $O(m^2)$ | For threshold optimization |
| Cluster centroids | $O(c \cdot d)$ | K-means output |
| Cluster labels | $O(n)$ | Assignment output |

**Overall**: $O(n \cdot d + m^2)$ — dominated by input embeddings.

### Scalability for Large Datasets

For $n > 40,000$:

1. **Threshold optimization**: Sample $m = 2000-5000$ points
2. **K-means**: Use MiniBatchKMeans (independent of $n$)
3. **Cluster assignment**: Process clusters, not individual samples

Tested on datasets up to 1M samples.

---

## Edge Cases and Robustness

### Highly Homogeneous Data

**Problem**: All similarities > 0.95 → single giant cluster

**Solution**: 
- Detect via `homogeneity = 'very_high'`
- Search in range [P50, P99]
- Use K-means with fixed k = √n if agglomerative fails

### Highly Heterogeneous Data

**Problem**: Few pairs have similarity > 0.5 → too many clusters

**Solution**:
- Detect via `homogeneity = 'low'`
- Use lower thresholds (0.4-0.7)
- Increase minimum cluster size

### Unbalanced Class Distribution

**Problem**: Most samples are inactive (class imbalance)

**Solution**:
- Stratify within each cluster by class label (if available)
- Ensure each split has representative class distribution

### Small Datasets

**Problem**: n < 100 → too few samples per split

**Solution**:
- Reduce minimum cluster size
- Use simpler stratification (label-based)
- Warn user about statistical limitations

### Memory Constraints

**Problem**: n > 50,000 → distance matrix too large

**Solution**:
- Automatic switch to representative sampling
- Sample 5000 points for threshold optimization
- Use MiniBatchKMeans instead of standard K-means

---

## Summary

| Aspect | Choice | Justification |
|--------|--------|---------------|
| **Similarity metric** | Cosine similarity | Scale-invariant, meaningful in high dimensions |
| **Threshold** | Dynamic (Silhouette-optimized) | Adapts to data distribution |
| **Clustering** | K-means++ with MiniBatch | O(log k) guarantees, scalable |
| **Cluster count** | √n bounded [10, 1000] | Empirically validated balance |
| **Assignment** | Greedy with 120% tolerance | Handles indivisible clusters |
| **Rebalancing** | Iterative smallest-cluster moves | Fine-tunes proportions |
