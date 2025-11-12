# Multi-View Stratification Module

## Overview
This module implements **SOLID-based stratified splitting** for train/validation/test sets using multi-view clustering on molecular embeddings. Designed for protein-ligand interaction prediction, it ensures balanced and representative data splits that prevent data leakage and maintain proper generalization.

## ✨ Key Features

- 🎯 **Multi-view clustering**: Combines protein and ligand similarity with configurable weights
- 🏗️ **SOLID architecture**: Modular, extensible, and maintainable code
- 📊 **Multiple algorithms**: DBSCAN, Hierarchical, K-Means, and Random clustering
- 🎨 **Rich visualizations**: PCA, t-SNE, and UMAP plots with automatic optimization
- ⚡ **Performance optimized**: Handles millions of samples with automatic downsampling
- 🔍 **Quality validation**: Comprehensive metrics and split analysis

## 🏗️ Architecture (SOLID Principles)

### Core Components

```
stratification/
├── stratifier.py                    # Main coordinator (SRP, DIP)
├── clustering.py                    # Strategy pattern (OCP, LSP)
├── cluster_splitter.py              # Split logic (SRP)
├── cosine_similarity_calculator.py  # Similarity computation
├── visualization.py                 # Plotting tools (optimized)
├── validator.py                     # Split quality validation
└── cluster_analyzer.py              # Clustering analysis
```

### Design Patterns

1. **Stratifier** (Coordinator)
   - Single Responsibility: Orchestrates the stratification process
   - Dependency Inversion: Depends on abstractions (ClusteringStrategy)

2. **ClusteringStrategy** (Strategy Pattern)
   - Open/Closed: Easy to add new algorithms without modifying existing code
   - Liskov Substitution: All strategies are interchangeable
   - Interface Segregation: Small, focused interface

3. **ClusterSplitter** (Single Responsibility)
   - Handles only the splitting logic
   - Supports edge cases (1, 2, 3+ samples per cluster)

4. **StratificationVisualizer** (Performance + Quality)
   - Automatic downsampling for large datasets
   - IncrementalPCA for memory efficiency
   - Multiple dimensionality reduction methods

## 📊 Embedding Dimensions

The module automatically uses correct dimensions from configuration:

- **Protein embeddings**: 2560 dimensions (ESM-2 t36 3B model)
- **Ligand embeddings**: 768 dimensions (FM4M model)
- **Combined embeddings**: 3328 dimensions (concatenated)

Each point in visualizations represents a **protein-ligand pair** with concatenated embeddings.

## 🚀 Quick Start

### Basic Single-View Stratification

```python
from build.stratification import Stratifier

# Initialize stratifier
stratifier = Stratifier(
    clustering_algorithm='kmeans',  # 'dbscan', 'hierarchical', 'kmeans', 'random'
    similarity_threshold=0.7
)

# Perform stratified split
train_idx, val_idx, test_idx = stratifier.stratified_split(
    embeddings,  # Combined protein-ligand embeddings (n_samples, 3328)
    labels,      # Binary interaction labels (n_samples,)
    test_size=0.2,
    val_size=0.1
)
```

### Multi-View Stratification (Recommended)

```python
# Multi-view considers both protein AND ligand similarity
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_embeddings,  # (n_samples, 2560)
    ligand_embeddings,   # (n_samples, 768)
    labels,              # (n_samples,)
    test_size=0.2,
    val_size=0.1,
    protein_weight=0.6,  # Weight for protein similarity
    ligand_weight=0.4    # Weight for ligand similarity
)
```

### Visualization

```python
from build.stratification.visualization import StratificationVisualizer

# Concatenate embeddings
combined = np.concatenate([protein_emb, ligand_emb], axis=1)

# Create visualizer
viz = StratificationVisualizer(
    method='pca',           # 'pca', 'tsne', or 'umap'
    max_samples=50000,      # Auto-downsample if larger
    use_incremental_pca=True
)

# Generate split visualization
viz.plot_split_visualization(
    embeddings=combined,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    cluster_labels=stratifier.cluster_labels,
    title="Stratification Results",
    save_path="results/stratification.png",
    show=True,
    dpi=150
)

# Compare multiple views
viz.plot_multi_view_comparison(
    protein_embeddings=protein_emb,
    ligand_embeddings=ligand_emb,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    save_path="results/multiview.png"
)
```

## ⚙️ Configuration Options

### Clustering Algorithms

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **kmeans** | Balanced datasets, benchmarks | Fast, predictable K clusters | Assumes spherical clusters |
| **hierarchical** | Natural structure | Detects hierarchy | May form 1 large cluster |
| **dbscan** | Noisy data | Detects outliers | Sensitive to parameters |
| **random** | Baseline comparison | Fast | No similarity consideration |

### Parameters

```python
stratifier = Stratifier(
    clustering_algorithm='kmeans',    # Algorithm choice
    similarity_threshold=0.7,         # Distance threshold (hierarchical)
    cluster_min_size=5,               # Minimum cluster size
    random_state=42                   # Reproducibility
)
```

### Visualization Methods

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **PCA** | ⚡⚡⚡ Fast | Good | Large datasets (>100k) |
| **t-SNE** | 🐌 Slow | Excellent | Small datasets (<10k) |
| **UMAP** | ⚡⚡ Medium | Excellent | Medium datasets (10-100k) |

## 🎯 Performance Optimization

### Automatic Downsampling

For large datasets, visualization automatically downsamples while preserving:
- Train/val/test proportions
- Cluster distributions
- Label balance

```python
# Handles 1M samples by downsampling to 50k
viz = StratificationVisualizer(
    method='pca',
    max_samples=50000  # Configurable threshold
)
```

### IncrementalPCA for Large Data

For datasets >100k samples, uses batch processing:
```python
viz = StratificationVisualizer(
    method='pca',
    use_incremental_pca=True  # Enables batch processing
)
```

### Performance Benchmarks

| Samples | Stratification | Visualization | Total | Memory |
|---------|----------------|---------------|-------|--------|
| 1k      | ~0.5s          | ~1s           | ~1.5s | ~15MB  |
| 10k     | ~2s            | ~3s           | ~5s   | ~150MB |
| 100k    | ~15s           | ~10s          | ~25s  | ~1.5GB |
| 1M      | ~3min          | ~30s (downsampled) | ~4min | ~15GB |

## 🔍 Validation & Metrics

```python
from build.stratification.validator import StratificationValidator

validator = StratificationValidator()

metrics = validator.validate_split(
    train_idx, val_idx, test_idx,
    labels, cluster_labels
)

print(f"Label balance: {metrics['label_balance']}")
print(f"Cluster distribution: {metrics['cluster_distribution']}")
print(f"Split sizes: {metrics['split_sizes']}")
```

## 📚 API Reference

### Stratifier Class

```python
class Stratifier:
    def __init__(
        self,
        clustering_algorithm: str = 'hierarchical',
        similarity_threshold: float = 0.7,
        cluster_min_size: int = 5,
        protein_weight: float = 0.6,
        ligand_weight: float = 0.4,
        random_state: int = 42
    )
    
    def stratified_split(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    
    def multi_view_stratified_split(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        protein_weight: float = 0.6,
        ligand_weight: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

### StratificationVisualizer Class

```python
class StratificationVisualizer:
    def __init__(
        self,
        method: str = 'pca',
        max_samples: int = 50000,
        use_incremental_pca: bool = True
    )
    
    def plot_split_visualization(
        self,
        embeddings: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None,
        title: str = "Stratification Visualization",
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: int = 150
    ) -> Figure
    
    def plot_multi_view_comparison(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        title: str = "Multi-View Comparison",
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: int = 150
    ) -> Figure
```

## 🔧 Integration with Pipeline

### Using in BuildPipeline

```python
from build.pipeline import BuildPipeline

# The pipeline automatically uses stratification if available
pipeline = BuildPipeline(
    input_tsv='data/compounds.tsv',
    output_dir='results/',
    use_stratification=True
)

# Run with stratified splits
results = pipeline.run()
```

### Custom Integration

```python
from build.stratification import Stratifier
from build.embeddings import EmbeddingGenerator

# Generate embeddings
emb_gen = EmbeddingGenerator()
protein_emb = emb_gen.generate_protein_embeddings(sequences)
ligand_emb = emb_gen.generate_ligand_embeddings(smiles)

# Stratified split
stratifier = Stratifier(clustering_algorithm='kmeans')
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_emb, ligand_emb, labels
)

# Use indices for model training
X_train = features[train_idx]
y_train = labels[train_idx]
# ... etc
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: All samples go to train set (0 val, 0 test)
- **Cause**: Similarity threshold too low, forming 1 large cluster
- **Solution**: Increase `similarity_threshold` or use `kmeans` algorithm

**Problem**: Memory error with large datasets
- **Cause**: Trying to compute full similarity matrix
- **Solution**: Enable `use_incremental_pca=True` or reduce `max_samples`

**Problem**: Visualization is very slow
- **Cause**: Too many samples or using t-SNE
- **Solution**: Use PCA for large datasets or enable auto-downsampling

**Problem**: Index type error: "arrays used as indices must be integer"
- **Cause**: Indices returned as float
- **Solution**: Convert to int: `train_idx = train_idx.astype(np.int32)`

### Performance Tips

1. **For large datasets (>100k)**:
   ```python
   viz = StratificationVisualizer(
       method='pca',              # PCA is fastest
       max_samples=50000,         # Limit samples
       use_incremental_pca=True   # Batch processing
   )
   ```

2. **For best clustering**:
   ```python
   stratifier = Stratifier(
       clustering_algorithm='kmeans',  # Most reliable
       random_state=42                 # Reproducibility
   )
   ```

3. **For quality visualization**:
   ```python
   viz.plot_split_visualization(
       ...,
       dpi=300,  # High quality (larger file)
       show=False  # Don't display, just save
   )
   ```

## 📖 Related Documentation

- [STRATIFIER_REFACTORING.md](../../../docs/04-modules/STRATIFIER_REFACTORING.md) - Detailed refactoring documentation
- [MULTI_VIEW_STRATIFICATION.md](../../../docs/04-modules/MULTI_VIEW_STRATIFICATION.md) - Multi-view algorithm details
- [PERFORMANCE_OPTIMIZATIONS.md](../../../docs/04-modules/PERFORMANCE_OPTIMIZATIONS.md) - Performance guide

## 📝 Testing

Run the test suite:
```bash
# Quick test (500 samples)
python tests/test_benchmark_quick.py

# Full multi-view tests
python tests/test_multi_view_stratification.py

# Comprehensive benchmark
python tests/benchmark_visualization.py
```

## 🤝 Contributing

When adding new clustering algorithms:

1. Create new strategy class in `clustering.py`:
   ```python
   class MyClusteringStrategy(ClusteringStrategy):
       def cluster(self, embeddings: np.ndarray) -> np.ndarray:
           # Your implementation
           pass
   ```

2. Register in stratifier factory:
   ```python
   strategies = {
       'myclustering': MyClusteringStrategy(...)
   }
   ```

3. Add tests and documentation

## 📄 License

Part of the DockTKinase project. See main LICENSE file.

## 👥 Authors

- LNCC Bioinformatics Team
- For questions: Contact repository maintainers