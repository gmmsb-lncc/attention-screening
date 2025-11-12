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

The module **automatically syncs dimensions** with the selected models:

### Dynamic Dimension Synchronization

```python
from build.core import BuildConfig

# Dimensions automatically match the model
config = BuildConfig(esm_model='esm2_t36_3B_UR50D')
dims = config.get_model_dimensions()

print(f"Protein: {dims['protein_dim']} dims")  # 2560
print(f"Ligand: {dims['ligand_dim']} dims")    # 768
print(f"Total: {dims['total_dim']} dims")      # 3328
```

### Supported Models

#### ESM Models (Proteins)

| Model | Dimensions | Parameters | Use Case |
|-------|-----------|------------|----------|
| `esm2_t6_8M_UR50D` | 320 | 8M | Quick tests |
| `esm2_t12_35M_UR50D` | 480 | 35M | Small datasets |
| `esm2_t30_150M_UR50D` | 640 | 150M | Medium datasets |
| `esm2_t33_650M_UR50D` | 1280 | 650M | Balanced performance |
| `esm2_t36_3B_UR50D` | **2560** | 3B | **Production (default)** |
| `esm2_t48_15B_UR50D` | 5120 | 15B | Maximum accuracy |

#### FM4M Models (Ligands)

| Model | Dimensions | Type |
|-------|-----------|------|
| `SMI-TED` | **768** | Transformer **(default)** |
| `SELFIES-TED` | 768 | Transformer |
| `SMI-SSED` | 768 | Encoder |
| `MHG` | 768 | Graph |
| `MOL-MOE` | 768 | Mixture |

### What Each Point Represents

Each point in visualizations represents a **protein-ligand pair**:
- Protein embedding: 2560 dimensions (ESM-2 t36 3B default)
- Ligand embedding: 768 dimensions (SMI-TED default)
- Combined: 3328 dimensions (concatenated)
- Reduced to: 2 dimensions (PCA/t-SNE/UMAP for visualization)

**Important**: Dimensions automatically update when you change models!

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

- [STRATIFIER_REFACTORING.md](../../../docs/04-modules/STRATIFIER_REFACTORING.md) - Detailed refactoring documentation with SOLID principles
- [MULTI_VIEW_STRATIFICATION.md](../../../docs/04-modules/MULTI_VIEW_STRATIFICATION.md) - Multi-view algorithm details and theory
- [PERFORMANCE_OPTIMIZATIONS.md](../../../docs/04-modules/PERFORMANCE_OPTIMIZATIONS.md) - Performance guide and benchmarks
- [DYNAMIC_DIMENSIONS.md](../../../docs/04-modules/DYNAMIC_DIMENSIONS.md) - Model dimension synchronization
- [STRATIFICATION_REORGANIZATION.md](../../../docs/04-modules/STRATIFICATION_REORGANIZATION.md) - Module reorganization summary

## 🎓 Key Concepts

### SOLID Principles Applied

1. **Single Responsibility Principle (SRP)**
   - `Stratifier`: Orchestrates the process
   - `ClusterSplitter`: Handles splitting logic
   - `StratificationVisualizer`: Manages visualizations

2. **Open/Closed Principle (OCP)**
   - Easy to add new clustering algorithms via `ClusteringStrategy`
   - Extend functionality without modifying existing code

3. **Liskov Substitution Principle (LSP)**
   - All clustering strategies are interchangeable
   - `DBSCANClustering`, `HierarchicalClustering`, `KMeansClustering`, `RandomClustering`

4. **Interface Segregation Principle (ISP)**
   - Small, focused interfaces
   - Components depend only on what they need

5. **Dependency Inversion Principle (DIP)**
   - Depends on abstractions (`ClusteringStrategy`)
   - Not on concrete implementations

### Multi-View Clustering

Combines protein and ligand similarities with configurable weights:

```
combined_similarity = w_p × protein_similarity + w_l × ligand_similarity
```

Where:
- `w_p` = protein_weight (default: 0.6)
- `w_l` = ligand_weight (default: 0.4)
- `w_p + w_l = 1.0`

This ensures both views are considered for creating balanced splits.

## 📝 Testing

Run the test suite:

```bash
# Navigate to project root
cd /path/to/docktkinase

# Activate environment
source env/bin/activate

# Quick test (500 samples, ~2s)
python tests/test_benchmark_quick.py

# Full multi-view tests (200 samples, 5 tests)
python tests/test_multi_view_stratification.py

# Comprehensive benchmark (1k-1M samples)
python tests/benchmark_visualization.py
```

### Test Coverage

| Test File | Purpose | Duration | Status |
|-----------|---------|----------|--------|
| `test_benchmark_quick.py` | Quick validation | ~2s | ✅ Passing |
| `test_multi_view_stratification.py` | Full suite (5 tests) | ~30s | ✅ Passing |
| `benchmark_visualization.py` | Performance testing | Variable | ✅ Passing |
| `test_stratification.py` | Legacy tests | ~10s | ⏭️ To verify |

### Expected Test Results

```
✅ test_cosine_similarity - Basic similarity calculations
✅ test_multi_view_similarity - Multi-view weighted similarity  
✅ test_stratified_split - Split integrity and balance
✅ test_weight_variations - Different weight configurations
✅ test_visualization - Generate visualization files
```

## 🚀 Quick Start Examples

### Example 1: Basic Stratification

```python
import numpy as np
from build.stratification import Stratifier

# Load your data
embeddings = np.load('combined_embeddings.npy')  # (n_samples, 3328)
labels = np.load('labels.npy')                    # (n_samples,)

# Create stratifier
stratifier = Stratifier(clustering_algorithm='kmeans')

# Split data
train_idx, val_idx, test_idx = stratifier.stratified_split(
    embeddings, labels, test_size=0.2, val_size=0.1
)

# Use splits
X_train = embeddings[train_idx]
y_train = labels[train_idx]
```

### Example 2: Multi-View with Visualization

```python
import numpy as np
from build.stratification import Stratifier
from build.stratification.visualization import StratificationVisualizer

# Load separate embeddings
protein_emb = np.load('protein_embeddings.npy')  # (n_samples, 2560)
ligand_emb = np.load('ligand_embeddings.npy')    # (n_samples, 768)
labels = np.load('labels.npy')                    # (n_samples,)

# Multi-view stratification
stratifier = Stratifier(clustering_algorithm='kmeans')
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_emb, ligand_emb, labels,
    protein_weight=0.6,
    ligand_weight=0.4
)

# Visualize results
combined = np.concatenate([protein_emb, ligand_emb], axis=1)
viz = StratificationVisualizer(method='pca')
viz.plot_split_visualization(
    combined, train_idx, val_idx, test_idx,
    cluster_labels=stratifier.cluster_labels,
    save_path='results/stratification.png'
)
```

### Example 3: Large Dataset Optimization

```python
from build.stratification import Stratifier
from build.stratification.visualization import StratificationVisualizer

# For large datasets (>100k samples)
stratifier = Stratifier(clustering_algorithm='kmeans')
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_emb, ligand_emb, labels
)

# Optimized visualization
viz = StratificationVisualizer(
    method='pca',              # Fastest method
    max_samples=50000,         # Auto-downsample
    use_incremental_pca=True   # Memory efficient
)

viz.plot_split_visualization(
    combined, train_idx, val_idx, test_idx,
    save_path='results/large_dataset.png',
    dpi=150,    # Lower DPI for smaller file
    show=False  # Don't display, just save
)
```

## 📈 Version History

### Current Version (November 2025)
- ✅ SOLID architecture refactoring
- ✅ Performance optimizations for large datasets
- ✅ Dynamic dimension synchronization
- ✅ Comprehensive visualization system
- ✅ Bug fixes (index duplication, type errors)
- ✅ Complete documentation suite

### Previous Version (Legacy)
- Basic stratification with single algorithm
- Manual dimension configuration
- Limited visualization support
- Monolithic 498-line implementation

See [STRATIFICATION_REORGANIZATION.md](../../../docs/04-modules/STRATIFICATION_REORGANIZATION.md) for detailed changelog.

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

## ❓ FAQ (Frequently Asked Questions)

### Q: Which clustering algorithm should I use?

**A**: For most cases, use **`kmeans`**:
- ✅ Fast and predictable
- ✅ Always creates multiple clusters
- ✅ Works well for benchmarking

Use `hierarchical` when you need to detect natural structure, or `dbscan` for noisy data with outliers.

### Q: Why are my dimensions different from expected?

**A**: Dimensions are **automatically synced** with your model choice. Check your config:
```python
config = BuildConfig()
print(config.get('esm_model'))  # Check which model is selected
dims = config.get_model_dimensions()
print(dims)  # See current dimensions
```

### Q: How do I handle very large datasets (>1M samples)?

**A**: Use these optimizations:
1. Enable automatic downsampling: `max_samples=50000`
2. Use IncrementalPCA: `use_incremental_pca=True`
3. Choose PCA over t-SNE/UMAP for visualization
4. Lower DPI for smaller file sizes: `dpi=150`

### Q: What does "multi-view" mean?

**A**: Multi-view considers **both protein AND ligand** similarities when clustering:
- Single-view: Clusters based on combined embeddings
- Multi-view: Weights protein similarity (0.6) and ligand similarity (0.4) separately
- Result: Better representation of both molecular spaces

### Q: Can I change the train/val/test split ratios?

**A**: Yes! Adjust `test_size` and `val_size`:
```python
train_idx, val_idx, test_idx = stratifier.stratified_split(
    embeddings, labels,
    test_size=0.15,  # 15% test
    val_size=0.15    # 15% validation
)
# Remaining 70% goes to train
```

### Q: How do I ensure reproducibility?

**A**: Set `random_state`:
```python
stratifier = Stratifier(
    clustering_algorithm='kmeans',
    random_state=42  # Fixed seed
)
```

### Q: What's the difference between stratifier.py and stratifier_legacy.py?

**A**: 
- `stratifier.py`: New SOLID architecture (200 lines, modular, tested)
- `stratifier_legacy.py`: Old monolithic version (498 lines, backup only)

Always use `stratifier.py` (the default import).

### Q: Why do I get "arrays used as indices must be integer" error?

**A**: Convert indices to int:
```python
train_idx = train_idx.astype(np.int32)
val_idx = val_idx.astype(np.int32)
test_idx = test_idx.astype(np.int32)
```

This is now handled automatically in the latest version.

### Q: How do I visualize only specific clusters?

**A**: Filter the data before visualization:
```python
# Get cluster 0 samples
cluster_0_mask = stratifier.cluster_labels == 0
cluster_0_emb = combined[cluster_0_mask]
cluster_0_train = train_idx[np.isin(train_idx, np.where(cluster_0_mask)[0])]
# ... then visualize
```

### Q: Can I use custom weights for multi-view?

**A**: Yes!
```python
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_emb, ligand_emb, labels,
    protein_weight=0.7,  # 70% protein importance
    ligand_weight=0.3    # 30% ligand importance
)
```

## 🆘 Support

For issues, questions, or contributions:
- 📧 Email: LNCC Bioinformatics Team
- 🐛 Issues: GitHub Issues
- 📚 Docs: See [Related Documentation](#-related-documentation)

## 👥 Authors

- LNCC Bioinformatics Team
- For questions: Contact repository maintainers

---

**Last Updated**: November 11, 2025  
**Version**: 2.0 (SOLID Refactoring)  
**Branch**: stratifier