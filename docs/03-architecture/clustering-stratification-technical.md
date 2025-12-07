# Clustering and Stratification: Technical Deep Dive

This document provides a detailed technical explanation of the clustering and stratification algorithms used in DockTKinase, with emphasis on the mathematical foundations and implementation choices.

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Cosine Similarity in High-Dimensional Spaces](#cosine-similarity-in-high-dimensional-spaces)
3. [MiniBatchKMeans with k-means++](#minibatchkmeans-with-k-means)
4. [Silhouette Score Analysis](#silhouette-score-analysis)
5. [K-means++ Theoretical Guarantees](#k-means-theoretical-guarantees)
6. [Greedy Cluster Assignment Algorithm](#greedy-cluster-assignment-algorithm)
7. [Complexity Analysis](#complexity-analysis)
8. [Edge Cases and Robustness](#edge-cases-and-robustness)

---

## Mathematical Foundations

### Embedding Space

Protein and ligand embeddings are represented as vectors in high-dimensional spaces:

- **Protein embeddings**: `p ∈ ℝ^d_p` where `d_p = 1280` (ESM-2)
- **Ligand embeddings**: `l ∈ ℝ^d_l` where `d_l = 768` (SMI-TED)

### Combined Embedding

The combined embedding is formed by weighted concatenation:

```
x = [α_p × p || α_l × l] ∈ ℝ^(d_p + d_l)
```

Where:
- `α_p` = protein weight (default: 1.0)
- `α_l` = ligand weight (default: 1.0)
- `||` denotes concatenation

After L2 normalization:

```
x̂ = x / ||x||₂
```

---

## Cosine Similarity in High-Dimensional Spaces

### Definition

For two vectors `a, b ∈ ℝ^d`:

```
                    a · b           Σᵢ aᵢbᵢ
cos(θ) = ───────────────────── = ─────────────────
          ||a||₂ × ||b||₂      √(Σᵢaᵢ²) × √(Σᵢbᵢ²)
```

### Properties

1. **Bounded range**: `cos(θ) ∈ [-1, 1]`
2. **Symmetry**: `cos(a, b) = cos(b, a)`
3. **Scale invariance**: `cos(k×a, b) = cos(a, b)` for `k > 0`

### Conversion to Distance

For clustering algorithms requiring a distance metric:

```
d(a, b) = 1 - cos(a, b)
```

This yields:
- `d ∈ [0, 2]`
- `d = 0` when vectors are identical
- `d = 1` when vectors are orthogonal
- `d = 2` when vectors are opposite

### Why Not Euclidean Distance?

In high-dimensional spaces, Euclidean distance suffers from the **curse of dimensionality**:

```
         d_max - d_min
lim      ───────────── → 0
d→∞         d_min
```

All points become approximately equidistant. Cosine similarity focuses on **angular difference**, which remains meaningful in high dimensions.

---

## MiniBatchKMeans with k-means++

### Current pipeline (cnn-optm)

- **Embedding prep**: Protein (ESM-2) and ligand (SMI-TED) embeddings are weighted, concatenated, and L2-normalized so Euclidean distance matches cosine geometry.
- **Clustering**: MiniBatchKMeans with k-means++ seeding; `k ≈ sqrt(n)` bounded to `[10, 1000]` to keep runtimes predictable.
- **No similarity threshold**: The former threshold-optimization flow is deprecated; there is no τ tuning in this branch.
- **Batching**: `batch_size = min(1024, n_samples)` to bound memory while maintaining centroid quality.

---

## Silhouette Score Analysis

### Definition

For sample `i` assigned to cluster `Cₐ`:

**Intra-cluster distance (cohesion):**

```
              1
a(i) = ─────────────  Σ  d(i, j)
        |Cₐ| - 1    j∈Cₐ, j≠i
```

**Inter-cluster distance (separation):**

```
b(i) = min   [ 1/|Cᵦ|  Σ  d(i, j) ]
      Cᵦ≠Cₐ          j∈Cᵦ
```

**Silhouette coefficient:**

```
        b(i) - a(i)
s(i) = ─────────────
       max(a(i), b(i))
```

**Global score:**

```
      1   n
S = ───  Σ  s(i)
      n  i=1
```

### Interpretation

| s(i) value | Interpretation |
|------------|----------------|
| ≈ +1 | Sample is far from neighboring clusters (well-clustered) |
| ≈ 0 | Sample is on cluster boundary |
| ≈ -1 | Sample is closer to another cluster (misclassified) |

### Optimization Heuristic (for k)

In the current flow, Silhouette is used to **validate** the chosen `k` (≈√n) rather than to sweep thresholds. A light heuristic is to compute Silhouette for a few nearby `k` values (e.g., `k-5, k, k+5`) on a sample to confirm separation without over-fragmenting clusters.

### Silhouette Score vs Cluster Granularity

Typical relationship (sampling nearby k values):

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
    └──────────────────────────────────────────► k (cluster count)
          small           medium             large
```

- **Too small k**: Few coarse clusters → poor cohesion → low S
- **Too large k**: Many tiny clusters → poor separation metric → low S
- **Good k range**: Balance between cohesion and separation → higher S

---

## K-means++ Theoretical Guarantees

### Standard K-means Objective

Minimize within-cluster sum of squares (WCSS):

```
         k
min     Σ      Σ    ||x - μⱼ||²
μ₁...μₖ j=1  x∈Cⱼ
```

Where `μⱼ` is the centroid of cluster `Cⱼ`.

### K-means++ Initialization

**Standard K-means** uses random initialization, which can lead to arbitrarily bad clusterings.

**K-means++** provides probabilistic guarantees:

1. Choose first centroid `μ₁` uniformly at random from X
2. For `j = 2, ..., k`:
   - Compute `D(x) = min_{i<j} ||x - μᵢ||²` for each `x ∈ X`
   - Choose `μⱼ = x` with probability `D(x) / Σ_y D(y)`

### Theoretical Result (Arthur & Vassilvitskii, 2007)

Let `φ_OPT` be the optimal WCSS. K-means++ initialization gives expected WCSS:

```
E[φ] ≤ 8(ln k + 2) × φ_OPT
```

This is `O(log k)` competitive ratio.

### Why MiniBatchKMeans?

For large datasets, standard K-means is `O(n × k × d × I)` where I = iterations.

MiniBatchKMeans uses stochastic updates:
1. Sample mini-batch B of size b (e.g., 1024)
2. Update centroids using only B
3. Repeat

Time complexity: `O(b × k × d × I)` — independent of n.

---

## Greedy Cluster Assignment Algorithm

### Problem Formulation

**Given:**
- Clusters `C = {C₁, ..., Cₖ}` with sizes `|C₁|, ..., |Cₖ|`
- Target proportions: `p_train = 0.80`, `p_val = 0.10`, `p_test = 0.10`
- Total samples: `n = Σⱼ |Cⱼ|`

**Find:**
- Assignment `π: C → {train, val, test}`
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
| Similarity matrix | `O(m² × d)` | m = sample size (≤5000), d = embedding dim |
| Threshold search | `O(k × m²)` | k = threshold candidates (≈10) |
| K-means++ | `O(n × c × d × I)` | c = clusters, I = iterations |
| Cluster assignment | `O(c log c)` | Sorting + linear scan |
| Index mapping | `O(n)` | Map clusters to sample indices |

**Overall**: `O(n × c × d)` — linear in dataset size.

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Embeddings | `O(n × d)` | Input data |
| Sample similarity matrix | `O(m²)` | For threshold optimization |
| Cluster centroids | `O(c × d)` | K-means output |
| Cluster labels | `O(n)` | Assignment output |

**Overall**: `O(n × d + m²)` — dominated by input embeddings.

### Scalability for Large Datasets

For n > 40,000:

1. **Threshold optimization**: Sample m = 2000-5000 points
2. **K-means**: Use MiniBatchKMeans (independent of n)
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
