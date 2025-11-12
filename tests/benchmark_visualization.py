#!/usr/bin/env python
"""
Benchmark test for stratification visualization with large datasets.

Tests performance with datasets of varying sizes:
- Small: 1,000 samples
- Medium: 10,000 samples  
- Large: 100,000 samples
- Very Large: 1,000,000 samples
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from build.stratification.stratifier_v2 import Stratifier
from build.stratification.visualization import StratificationVisualizer
from build.core.constants import DEFAULT_PROTEIN_DIM, DEFAULT_LIGAND_DIM


def generate_synthetic_data(n_samples: int, 
                            protein_dim: int = DEFAULT_PROTEIN_DIM,
                            ligand_dim: int = DEFAULT_LIGAND_DIM):
    """
    Generate synthetic dataset with dimensions from config.
    
    Args:
        n_samples: Number of samples to generate
        protein_dim: Protein embedding dimension (default from constants)
        ligand_dim: Ligand embedding dimension (default from constants)
    
    Returns:
        Tuple of (protein_embeddings, ligand_embeddings, labels)
    """
    print(f"  Generating {n_samples:,} samples (protein_dim={protein_dim}, ligand_dim={ligand_dim})...")
    
    # Generate embeddings with appropriate dtypes
    protein_embeddings = np.random.randn(n_samples, protein_dim).astype(np.float32)
    ligand_embeddings = np.random.randn(n_samples, ligand_dim).astype(np.float32)
    labels = np.random.randint(0, 2, size=n_samples, dtype=np.int32)
    
    return protein_embeddings, ligand_embeddings, labels


def benchmark_stratification(n_samples, max_samples=50000):
    """Benchmark stratification with given sample size."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {n_samples:,} samples")
    print(f"{'='*80}")
    
    # Generate data
    start = time.time()
    protein_emb, ligand_emb, labels = generate_synthetic_data(n_samples)
    gen_time = time.time() - start
    print(f"  ✓ Data generation: {gen_time:.2f}s")
    
    # Stratification
    start = time.time()
    stratifier = Stratifier(
        clustering_algorithm='hierarchical',
        similarity_threshold=0.7
    )
    
    train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
        protein_embeddings=protein_emb,
        ligand_embeddings=ligand_emb,
        labels=labels,
        test_size=0.2,
        val_size=0.1
    )
    
    # Convert indices to int (important!)
    train_idx = train_idx.astype(np.int32)
    val_idx = val_idx.astype(np.int32)
    test_idx = test_idx.astype(np.int32)
    
    strat_time = time.time() - start
    print(f"  ✓ Stratification: {strat_time:.2f}s")
    print(f"    - Train: {len(train_idx):,} samples")
    print(f"    - Val:   {len(val_idx):,} samples")
    print(f"    - Test:  {len(test_idx):,} samples")
    
    # Combined embeddings for visualization
    combined = np.concatenate([protein_emb, ligand_emb], axis=1)
    print(f"  ℹ Combined embedding dimension: {combined.shape[1]:,}")
    
    # PCA visualization (fast) - use smaller max_samples for benchmark
    # to avoid very slow dimensionality reduction
    effective_max_samples = min(max_samples, 10_000)  # Limit for benchmark
    if n_samples > effective_max_samples:
        print(f"  ⚠️  Will downsample from {n_samples:,} to {effective_max_samples:,} for visualization")
    
    start = time.time()
    viz_pca = StratificationVisualizer(
        method='pca',
        max_samples=effective_max_samples,
        use_incremental_pca=True
    )
    
    output_path = f"results/benchmark_{n_samples}_pca.png"
    viz_pca.plot_split_visualization(
        embeddings=combined,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cluster_labels=stratifier.cluster_labels,
        title=f"Benchmark: {n_samples:,} samples (PCA)",
        save_path=output_path,
        show=False,
        dpi=100  # Lower DPI for faster save
    )
    viz_time = time.time() - start
    print(f"  ✓ PCA visualization: {viz_time:.2f}s")
    print(f"    - Saved to: {output_path}")
    
    # Memory estimate
    mem_mb = (protein_emb.nbytes + ligand_emb.nbytes + labels.nbytes) / 1024 / 1024
    print(f"  ℹ Memory used: ~{mem_mb:.1f} MB")
    
    total_time = gen_time + strat_time + viz_time
    print(f"  ✅ Total time: {total_time:.2f}s")
    
    return {
        'n_samples': n_samples,
        'gen_time': gen_time,
        'strat_time': strat_time,
        'viz_time': viz_time,
        'total_time': total_time,
        'memory_mb': mem_mb
    }


def main():
    """Run benchmarks."""
    print("\n" + "="*80)
    print("🚀 STRATIFICATION VISUALIZATION BENCHMARK")
    print("="*80)
    print("\nTesting performance with different dataset sizes...")
    
    results = []
    
    # Small dataset
    try:
        result = benchmark_stratification(1_000, max_samples=1_000)
        results.append(result)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Medium dataset
    try:
        result = benchmark_stratification(10_000, max_samples=10_000)
        results.append(result)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Large dataset
    try:
        result = benchmark_stratification(100_000, max_samples=50_000)
        results.append(result)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Very large dataset (optional - skip if too slow)
    try:
        print("\n⚠️  Very large dataset test (1M samples) - this may take several minutes...")
        response = input("Run 1M sample test? (y/n): ").strip().lower()
        
        if response == 'y':
            result = benchmark_stratification(1_000_000, max_samples=50_000)
            results.append(result)
        else:
            print("  ⏭️  Skipped 1M sample test")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Samples':<15} {'Stratify':<12} {'Visualize':<12} {'Total':<12} {'Memory':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['n_samples']:>12,}   {r['strat_time']:>8.2f}s    {r['viz_time']:>8.2f}s    "
              f"{r['total_time']:>8.2f}s    {r['memory_mb']:>6.1f}MB")
    
    print("\n" + "="*80)
    print("✅ Benchmark completed!")
    print(f"Check results/ directory for generated plots.")
    print("="*80)
    
    # Performance notes
    print("\n💡 Performance Notes:")
    print("  - PCA is fastest (linear time)")
    print("  - UMAP is good for up to 100k samples")
    print("  - t-SNE is limited to ~10k samples (quadratic time)")
    print("  - Automatic downsampling preserves split proportions")
    print("  - IncrementalPCA used for >100k samples")
    print("  - Rasterization helps with large plots")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
