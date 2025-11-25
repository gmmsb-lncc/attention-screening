# Adaptive Clustering Guide

**Last Updated**: November 25, 2025  
**Module**: `src/build/stratification/adaptive_clustering.py`

---

## 📋 Overview

The Adaptive Clustering module automatically determines optimal clustering parameters for molecular embeddings. It solves the problem of **homogeneous embeddings** where fixed thresholds fail to produce meaningful clusters.

### The Problem

When embeddings are highly similar (e.g., all pairwise similarities > 0.9), a fixed threshold like 0.7 results in a **single cluster** containing all samples. This defeats the purpose of stratified splitting.

### The Solution

Adaptive Clustering analyzes the similarity distribution and automatically selects an appropriate threshold that produces meaningful clusters.

---

## 🎯 Optimization Methods

### 1. Target Method (Default)

**Binary search to achieve target cluster count.**

```python
target_clusters = n_samples * target_cluster_ratio  # Default: 1%
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_cluster_ratio` | `0.01` | Target clusters as ratio of samples |
| `min_clusters` | `5` | Minimum number of clusters |
| `max_clusters` | `100` | Maximum number of clusters |

**Example**: 10,000 samples → ~100 clusters

**Best For**: Most datasets, provides consistent cluster counts.

---

### 2. Silhouette Method

**Maximize silhouette score via grid search.**

The silhouette score measures how similar samples are to their own cluster compared to other clusters.

```
silhouette_score ∈ [-1, 1]
- 1.0: Perfect clustering
- 0.0: Overlapping clusters
- -1.0: Wrong cluster assignments
```

**Best For**: Optimizing cluster quality, well-separated data.

---

### 3. Elbow Method

**Find optimal k for K-means using curvature analysis.**

Plots inertia vs. number of clusters and finds the "elbow" point where adding more clusters provides diminishing returns.

**Best For**: When cluster count is important, K-means clustering.

---

### 4. Percentile Method

**Use similarity percentile based on data homogeneity.**

| Homogeneity | Threshold Percentile |
|-------------|---------------------|
| Very High (min > 0.9) | P75 |
| High (min > 0.7) | P50 |
| Moderate | 0.7 (fixed) |

**Best For**: Highly homogeneous data, quick approximation.

---

### 5. Manual Method

**Use user-specified threshold with validation warnings.**

```bash
python run_complete_pipeline.py \
    --input data.tsv \
    --stratifier-threshold 0.95
```

The system will warn if the threshold is inappropriate for the data:

```
Warning: Manual threshold 0.70 is below P25 (0.94) for highly homogeneous data.
This may result in very few clusters.
```

**Best For**: When you know the optimal threshold for your dataset.

---

## 📊 CLI Usage

### Basic Usage (Auto Threshold)

```bash
# Uses 'target' method by default
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/auto_strat
```

### Custom Auto-Threshold Method

```bash
# Use silhouette optimization
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/silhouette_strat \
    --stratifier-method silhouette

# Use elbow method
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/elbow_strat \
    --stratifier-method elbow
```

### Manual Threshold

```bash
# Specify exact threshold
python run_complete_pipeline.py \
    --input data/kinase_compounds.tsv \
    --output results/manual_strat \
    --stratifier-threshold 0.95
```

---

## 🐍 Python API

### Using AdaptiveClustering Directly

```python
from src.build.stratification import AdaptiveClustering
import numpy as np

# Load embeddings
embeddings = np.load('embeddings.npy')

# Create adaptive clustering with target method
clustering = AdaptiveClustering(
    method='target',
    target_cluster_ratio=0.01,  # 1% of samples
    min_clusters=5,
    max_clusters=100,
    min_cluster_size=3
)

# Cluster embeddings
labels = clustering.cluster(embeddings)

# Access metrics
print(f"Clusters: {clustering.metrics.n_clusters}")
print(f"Threshold: {clustering.metrics.threshold_used}")
print(f"Silhouette: {clustering.metrics.silhouette_score}")

# Save metrics to JSON
clustering.save_metrics('output/', prefix='my_clustering')
```

### Using Manual Threshold

```python
from src.build.stratification import AdaptiveClustering

# Create with manual threshold
clustering = AdaptiveClustering(
    method='manual',
    manual_threshold=0.95
)

labels = clustering.cluster(embeddings)
```

### Using Stratifier with Adaptive Clustering

```python
from src.build.stratification import Stratifier
from src.build.core import BuildConfig

config = BuildConfig(...)

# Stratifier with adaptive clustering (default)
stratifier = Stratifier(
    config=config,
    clustering_algorithm='adaptive',
    adaptive_method='target',
    target_cluster_ratio=0.01
)

# Perform stratified split
train_idx, val_idx, test_idx = stratifier.stratified_split(
    embeddings=embeddings,
    labels=labels,
    test_size=0.2,
    val_size=0.1,
    output_dir='results/stratification'
)

# Access clustering info
print(stratifier.get_cluster_info())
```

### Multi-View Stratification with Adaptive Clustering

```python
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_embeddings=protein_emb,
    ligand_embeddings=ligand_emb,
    labels=labels,
    test_size=0.2,
    val_size=0.1,
    protein_weight=0.6,
    ligand_weight=0.4,
    output_dir='results/stratification'
)
```

---

## 📁 Output Files

### clustering_metrics.json

```json
{
  "n_clusters": 134,
  "n_samples": 13427,
  "n_noise": 42,
  "silhouette_score": 0.3215,
  "calinski_harabasz_score": 1542.8,
  "davies_bouldin_score": 1.23,
  "threshold_used": 0.9908,
  "method": "target",
  "similarity_stats": {
    "min": 0.9412,
    "max": 0.9999,
    "mean": 0.9823,
    "std": 0.0089,
    "p50": 0.9845,
    "p75": 0.9889,
    "p95": 0.9956,
    "homogeneity": "very_high"
  },
  "cluster_sizes": {"0": 245, "1": 189, "2": 156, ...},
  "threshold_search_history": [...]
}
```

### stratification_split_info.json

```json
{
  "train": {
    "n_samples": 9398,
    "label_distribution": {"0": 4532, "1": 4866},
    "mean_label": 0.518,
    "std_label": 0.499
  },
  "val": {
    "n_samples": 1343,
    "label_distribution": {"0": 645, "1": 698},
    "mean_label": 0.520,
    "std_label": 0.500
  },
  "test": {
    "n_samples": 2686,
    "label_distribution": {"0": 1298, "1": 1388},
    "mean_label": 0.517,
    "std_label": 0.500
  },
  "total_samples": 13427,
  "clustering_algorithm": "adaptive",
  "adaptive_method": "target",
  "n_clusters": 134,
  "n_noise": 42,
  "clustering_metrics": {
    "silhouette_score": 0.3215,
    "threshold_used": 0.9908
  }
}
```

---

## 📈 Visualization

Use the PCA visualization script to analyze clusters:

```bash
python scripts/visualize_cluster_pca.py \
    --embeddings results/build/embedding_matrix.npy \
    --labels results/build/labels.npy \
    --output results/visualization/cluster_pca.png \
    --threshold auto
```

### Output (2x2 Layout)

1. **Cluster Visualization**: PCA with cluster coloring
2. **Activity Distribution**: PCA with active/inactive coloring
3. **Cluster Size Histogram**: Distribution of cluster sizes
4. **Cluster Composition**: Active/inactive ratio per cluster

---

## ⚙️ Configuration Options

### IntegratedConfig Parameters

```python
from src.integrated_pipeline import IntegratedConfig

config = IntegratedConfig(
    input_tsv="data.tsv",
    output_dir="results/",
    
    # Stratification settings
    stratifier_auto_threshold=True,      # Use auto detection (default)
    stratifier_threshold=None,           # Manual threshold (overrides auto)
    stratifier_method='target',          # Auto method: target, silhouette, elbow, percentile
    
    # Other settings...
)
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stratifier_auto_threshold` | `True` | Enable automatic threshold detection |
| `stratifier_threshold` | `None` | Manual threshold (0.0-1.0) |
| `stratifier_method` | `'target'` | Auto-threshold optimization method |
| `target_cluster_ratio` | `0.01` | 1% of samples as clusters |
| `min_clusters` | `5` | Minimum clusters |
| `max_clusters` | `100` | Maximum clusters |
| `min_cluster_size` | `3` | Minimum points per cluster |

---

## 🔬 Algorithm Details

### Similarity Analysis

Before clustering, the system analyzes pairwise cosine similarities:

```python
sim_matrix = cosine_similarity(embeddings)
stats = {
    'min': np.min(similarities),
    'max': np.max(similarities),
    'mean': np.mean(similarities),
    'std': np.std(similarities),
    'percentiles': [p5, p10, p25, p50, p75, p90, p95, p99]
}
```

### Homogeneity Classification

| Level | Condition | Typical Action |
|-------|-----------|----------------|
| Very High | min > 0.9 | Use P75-P99 threshold range |
| High | min > 0.7 | Use P25-P95 threshold range |
| Moderate | min > 0.5 | Use 0.5-0.95 threshold range |
| Low | min ≤ 0.5 | Fixed threshold works |

### Target Method Algorithm

```
1. Calculate target_clusters = n_samples * target_cluster_ratio
2. Binary search:
   - low = min_similarity
   - high = max_similarity
   - For each iteration:
     a. mid = (low + high) / 2
     b. Cluster with threshold = mid
     c. If n_clusters < target: low = mid (need more clusters)
     d. If n_clusters > target: high = mid (need fewer clusters)
     e. Stop when converged or target reached
3. Return best threshold found
```

---

## 🎓 Best Practices

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| General use | `target` (default) |
| Optimizing cluster quality | `silhouette` |
| Fixed cluster count needed | `elbow` |
| Quick approximation | `percentile` |
| Known optimal threshold | `manual` |

### Handling Highly Homogeneous Data

If all similarities are > 0.9:

1. **Use adaptive methods** - they automatically adjust
2. **Consider `target` method** - guarantees reasonable cluster count
3. **Check metrics JSON** - verify silhouette score is acceptable (> 0.2)

### Production Tips

1. **Save metrics** - Always use `output_dir` to save JSON for reproducibility
2. **Validate splits** - Check `split_info.json` for label balance
3. **Visualize** - Use PCA script to verify cluster quality
4. **Compare methods** - Run with different methods and compare silhouette scores

---

## 🔗 Related Documentation

- [Multi-View Stratification](MULTI_VIEW_STRATIFICATION.md) - Protein/ligand weight balancing
- [Cluster PCA Visualization](../../scripts/visualize_cluster_pca.py) - Visualization script
- [Stratification Index](STRATIFICATION_INDEX.md) - Complete stratification overview

---

**Module**: `src/build/stratification/adaptive_clustering.py`  
**Author**: DockTKinase Team  
**Version**: 1.0
