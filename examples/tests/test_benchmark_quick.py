#!/usr/bin/env python
"""Quick benchmark test with small dataset."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from build.stratification.stratifier_v2 import Stratifier
from build.stratification.visualization import StratificationVisualizer
from build.core.constants import DEFAULT_PROTEIN_DIM, DEFAULT_LIGAND_DIM

print("\n" + "="*80)
print("🚀 QUICK BENCHMARK TEST")
print("="*80)

print(f"\nDimensions from config:")
print(f"  - Protein: {DEFAULT_PROTEIN_DIM}")
print(f"  - Ligand: {DEFAULT_LIGAND_DIM}")
print(f"  - Total: {DEFAULT_PROTEIN_DIM + DEFAULT_LIGAND_DIM}")

# Generate small dataset
n_samples = 500
print(f"\nGenerating {n_samples} synthetic samples...")
protein_emb = np.random.randn(n_samples, DEFAULT_PROTEIN_DIM).astype(np.float32)
ligand_emb = np.random.randn(n_samples, DEFAULT_LIGAND_DIM).astype(np.float32)
labels = np.random.randint(0, 2, size=n_samples, dtype=np.int32)

print(f"  ✓ Protein shape: {protein_emb.shape}")
print(f"  ✓ Ligand shape: {ligand_emb.shape}")
print(f"  ✓ Labels shape: {labels.shape}")

# Stratification
print("\nRunning stratification...")
start = time.time()
stratifier = Stratifier(
    clustering_algorithm='kmeans',  # Use kmeans for more predictable clusters
    similarity_threshold=0.7
)

train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_embeddings=protein_emb,
    ligand_embeddings=ligand_emb,
    labels=labels,
    test_size=0.2,
    val_size=0.1
)

# Convert to int
train_idx = train_idx.astype(np.int32)
val_idx = val_idx.astype(np.int32)
test_idx = test_idx.astype(np.int32)

strat_time = time.time() - start
print(f"  ✓ Completed in {strat_time:.2f}s")
print(f"    - Train: {len(train_idx)} samples")
print(f"    - Val: {len(val_idx)} samples")
print(f"    - Test: {len(test_idx)} samples")
print(f"    - Index types: {train_idx.dtype}, {val_idx.dtype}, {test_idx.dtype}")

# Visualization
print("\nGenerating visualization...")
combined = np.concatenate([protein_emb, ligand_emb], axis=1)
print(f"  ✓ Combined shape: {combined.shape}")

start = time.time()
viz = StratificationVisualizer(
    method='pca',
    max_samples=500,
    use_incremental_pca=False
)

output_path = "results/test_quick_benchmark.png"
viz.plot_split_visualization(
    embeddings=combined,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    cluster_labels=stratifier.cluster_labels,
    title=f"Quick Benchmark ({n_samples} samples)",
    save_path=output_path,
    show=False,
    dpi=100
)
viz_time = time.time() - start
print(f"  ✓ Completed in {viz_time:.2f}s")
print(f"  ✓ Saved to: {output_path}")

print("\n" + "="*80)
print("✅ QUICK TEST SUCCESSFUL!")
print(f"Total time: {strat_time + viz_time:.2f}s")
print("="*80)
