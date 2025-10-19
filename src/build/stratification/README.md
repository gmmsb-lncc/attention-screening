# Cosine Similarity-Based Stratification Module

## Overview
This module implements stratification methods for train/test/validation splits using cosine similarity on molecular embeddings. It's designed to create balanced and representative data splits for protein-ligand interaction prediction models, ensuring that similar compounds and targets are appropriately distributed across splits.

## Features
- Cosine similarity-based clustering for molecular embeddings
- Support for both protein and ligand similarity-based stratification
- Multi-view stratification considering both protein and ligand spaces
- Integration with existing pipeline architecture
- Configurable similarity thresholds and clustering parameters

## Architecture

### Core Components
1. `cosine_similarity_calculator.py` - Efficient cosine similarity computation
2. `stratifier.py` - Main stratification logic and clustering algorithms
3. `validator.py` - Split quality validation and metrics
4. `cluster_analyzer.py` - Analysis of clustering results

### Multi-Strategy Approach
- **Ligand-based clustering**: Groups compounds by structural/chemical similarity
- **Protein-based clustering**: Groups proteins by sequence/structural similarity
- **Dual-view clustering**: Combines both views for comprehensive stratification
- **Fallback methods**: Random stratification when similarity-based methods fail

## Technical Implementation

### Cosine Similarity Computation
- Optimized for high-dimensional embeddings (both protein and ligand spaces)
- Memory-efficient batch processing for large datasets
- Support for normalized and unnormalized embeddings
- Handles concatenated embeddings (protein + ligand) with proper normalization

### Clustering Algorithms
- DBSCAN/HDBSCAN for density-based clustering (handles variable cluster sizes)
- Hierarchical clustering with multiple linkage options
- K-means as baseline for comparison
- Automatic parameter selection based on dataset characteristics

### Split Strategies
- **Consensus Clustering**: Combines multiple clustering results for robustness
- **Constrained Clustering**: Ensures balanced label distribution across clusters
- **Novelty-Aware Splitting**: Ensures test set contains truly novel compounds/targets
- **Multi-constraint Optimization**: Balances similarity clusters AND label distribution

## Usage

### Basic Usage
```python
from build.stratification import Stratifier
from build.core import BuildConfig

config = BuildConfig()
stratifier = Stratifier(config)

# Perform stratified split
train_idx, val_idx, test_idx = stratifier.stratified_split(
    embeddings,  # concatenated protein-ligand embeddings
    labels,      # interaction labels
    test_size=0.2,
    val_size=0.1
)
```

### Advanced Usage
```python
# Configure specific parameters
config = BuildConfig({
    'stratification': {
        'similarity_threshold': 0.8,
        'clustering_algorithm': 'dbscan',  # 'dbscan', 'hierarchical', 'kmeans'
        'cluster_min_size': 5,
        'stratify_by': 'both'  # 'ligand', 'protein', 'both'
    }
})

stratifier = Stratifier(config)

# Perform multi-view stratification
splits = stratifier.multi_view_stratified_split(
    protein_embeddings,
    ligand_embeddings, 
    labels,
    weights={'protein': 0.6, 'ligand': 0.4}  # Weighting for dual-view clustering
)
```

## Configuration Options

### Main Parameters
- `similarity_threshold`: Threshold for grouping similar samples (default: 0.8)
- `clustering_algorithm`: Algorithm to use ('dbscan', 'hierarchical', 'kmeans', 'random')
- `cluster_min_size`: Minimum cluster size for stratification (default: 5)
- `stratify_by`: Which view to stratify by ('ligand', 'protein', 'both')

### Clustering-Specific Parameters
- `dbscan_eps`: Epsilon parameter for DBSCAN clustering
- `dbscan_min_samples`: Minimum samples parameter for DBSCAN
- `hierarchical_linkage`: Linkage method for hierarchical clustering
- `kmeans_n_clusters`: Number of clusters for k-means (if specified)

## Integration with Existing Pipeline

The stratification module integrates seamlessly with the existing BuildPipeline:

```python
from build.pipeline import BuildPipeline
from build.core import BuildConfig

config = BuildConfig({
    'stratification_enabled': True,
    'stratification_params': {
        'similarity_threshold': 0.8,
        'clustering_algorithm': 'dbscan'
    }
})

pipeline = BuildPipeline(config)
results = pipeline.run_complete_pipeline(
    input_tsv_path='path/to/data.tsv',
    output_dir='path/to/output',
    stratify_splits=True  # Enable stratified splits
)
```

## Validation Metrics

The module includes comprehensive validation of split quality:

- **Cluster Quality**: Silhouette score, within-cluster sum of squares
- **Label Distribution**: Balance of labels across train/test/validation splits
- **Similarity Analysis**: Assessment of compound/target novelty
- **Split Difficulty**: Evaluation of test set challenge level

## Algorithms

### Cosine Similarity Formula
For vectors A and B:
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)
```

### Dual-View Clustering
Combines protein and ligand similarities with configurable weights:
```
combined_similarity = w_p * protein_sim + w_l * ligand_sim
```

## Performance Considerations

### Large Dataset Handling
- Memory-efficient batch processing
- Approximate similarity computation for datasets > 100K samples
- Parallel processing support

### Computational Complexity
- All-pairs similarity: O(n²) - optimized with batch processing
- DBSCAN clustering: O(n log n) average case
- Hierarchical clustering: O(n²) - with optimized implementations for large datasets

## Best Practices

1. **Dataset Size Considerations**:
   - Small datasets (<10K): Use exact similarity computation
   - Medium datasets (10K-100K): Use optimized implementations
   - Large datasets (>100K): Use approximate methods with sampling

2. **Parameter Tuning**:
   - Adjust similarity threshold based on molecular diversity in your dataset
   - Consider biological relevance when choosing clustering algorithm
   - Validate that splits maintain desired label distribution

3. **Validation**:
   - Always validate split quality before model training
   - Check for temporal or batch effects if temporal information is available
   - Assess whether test set difficulty is representative

## Troubleshooting

### Common Issues
- **Memory errors**: Use batch processing for large datasets
- **Slow computation**: Consider using approximate methods for large datasets
- **Poor clustering**: Adjust similarity threshold or try different clustering algorithm
- **Unbalanced splits**: Use constrained clustering to maintain label distribution

### Performance Tips
- Normalize embeddings before computing similarity
- Use appropriate clustering algorithm for your dataset size
- Consider protein and ligand spaces separately if their scales differ significantly