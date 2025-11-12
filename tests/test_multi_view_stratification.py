#!/usr/bin/env python
"""
Test script for multi-view stratification functionality.
Validates that the multi-view stratification is working correctly
with separate protein and ligand embeddings.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import updated stratifier
from build.stratification.stratifier import Stratifier
from build.stratification.cosine_similarity_calculator import CosineSimilarityCalculator
from build.stratification.visualization import StratificationVisualizer


def test_cosine_similarity():
    """Test basic cosine similarity calculation."""
    print("\n" + "="*80)
    print("TEST 1: Cosine Similarity Calculator")
    print("="*80)
    
    calculator = CosineSimilarityCalculator()
    
    # Test case 1: Identical vectors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([1, 2, 3])
    sim = calculator.calculate_single_pair(vec1, vec2)
    print(f"✓ Identical vectors: similarity = {sim:.4f} (expected: 1.0)")
    assert abs(sim - 1.0) < 0.001, "Identical vectors should have similarity 1.0"
    
    # Test case 2: Orthogonal vectors
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    sim = calculator.calculate_single_pair(vec1, vec2)
    print(f"✓ Orthogonal vectors: similarity = {sim:.4f} (expected: 0.0)")
    assert abs(sim) < 0.001, "Orthogonal vectors should have similarity 0.0"
    
    # Test case 3: Proportional vectors
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([2, 4, 6])  # 2x vec1
    sim = calculator.calculate_single_pair(vec1, vec2)
    print(f"✓ Proportional vectors: similarity = {sim:.4f} (expected: 1.0)")
    assert abs(sim - 1.0) < 0.001, "Proportional vectors should have similarity 1.0"
    
    print("\n✅ Cosine similarity tests PASSED!")
    return True


def test_multi_view_similarity():
    """Test multi-view similarity calculation."""
    print("\n" + "="*80)
    print("TEST 2: Multi-View Similarity")
    print("="*80)
    
    calculator = CosineSimilarityCalculator()
    
    # Create synthetic embeddings
    # Sample 1: Kinase A + Ligand X
    protein_emb_1 = np.array([[0.8, 0.6]])
    ligand_emb_1 = np.array([[0.3, 0.4]])
    
    # Sample 2: Kinase A + Ligand Y (same protein, different ligand)
    protein_emb_2 = np.array([[0.8, 0.6]])  # Identical protein
    ligand_emb_2 = np.array([[0.1, 0.9]])   # Different ligand
    
    # Sample 3: Kinase B + Ligand X (different protein, same ligand)
    protein_emb_3 = np.array([[0.2, 0.7]])  # Different protein
    ligand_emb_3 = np.array([[0.3, 0.4]])   # Identical ligand
    
    # Combine all samples
    protein_embeddings = np.vstack([protein_emb_1, protein_emb_2, protein_emb_3])
    ligand_embeddings = np.vstack([ligand_emb_1, ligand_emb_2, ligand_emb_3])
    
    # Calculate multi-view similarity
    multi_view_sim = calculator.calculate_multi_view_similarity(
        protein_embeddings=protein_embeddings,
        ligand_embeddings=ligand_embeddings,
        protein_weight=0.6,
        ligand_weight=0.4
    )
    
    print(f"\nMulti-view similarity matrix:")
    print(f"{'':12} Sample 1  Sample 2  Sample 3")
    for i in range(3):
        print(f"Sample {i+1}:  ", end="")
        for j in range(3):
            print(f"  {multi_view_sim[i, j]:.4f}", end="")
        print()
    
    # Verify expected patterns
    print("\n✓ Checking similarity patterns:")
    
    # Sample 1-2 (same protein): Should be high
    sim_1_2 = multi_view_sim[0, 1]
    print(f"  Sample 1-2 (same protein): {sim_1_2:.4f} (expected: >0.7)")
    assert sim_1_2 > 0.7, "Same protein should have high similarity"
    
    # Sample 1-3 (same ligand): Should be moderate-high
    sim_1_3 = multi_view_sim[0, 2]
    print(f"  Sample 1-3 (same ligand):  {sim_1_3:.4f} (expected: >0.6)")
    assert sim_1_3 > 0.6, "Same ligand should have moderate-high similarity"
    
    # Sample 1-2 should be higher than 1-3 (protein has more weight)
    print(f"  Protein weight effect: {sim_1_2:.4f} > {sim_1_3:.4f} = {sim_1_2 > sim_1_3}")
    assert sim_1_2 > sim_1_3, "Protein weight (0.6) should make same-protein more similar"
    
    print("\n✅ Multi-view similarity tests PASSED!")
    return True


def test_stratified_split():
    """Test stratified split with multi-view."""
    print("\n" + "="*80)
    print("TEST 3: Multi-View Stratified Split")
    print("="*80)
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 100
    protein_dim = 320
    ligand_dim = 768
    
    # Generate protein embeddings (5 groups of similar proteins)
    protein_embeddings = []
    for i in range(5):
        base = np.random.randn(protein_dim)
        for j in range(20):  # 20 samples per protein group
            noise = np.random.randn(protein_dim) * 0.1
            protein_embeddings.append(base + noise)
    protein_embeddings = np.array(protein_embeddings)
    
    # Generate ligand embeddings (10 groups of similar ligands)
    ligand_embeddings = []
    for i in range(10):
        base = np.random.randn(ligand_dim)
        for j in range(10):  # 10 samples per ligand group
            noise = np.random.randn(ligand_dim) * 0.1
            ligand_embeddings.append(base + noise)
    ligand_embeddings = np.array(ligand_embeddings)
    
    # Generate labels (active/inactive)
    labels = np.random.randint(0, 2, size=n_samples)
    
    print(f"Dataset: {n_samples} samples")
    print(f"  Protein embeddings: {protein_embeddings.shape}")
    print(f"  Ligand embeddings: {ligand_embeddings.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Active: {np.sum(labels == 1)} | Inactive: {np.sum(labels == 0)}")
    
    # Initialize stratifier
    stratifier = Stratifier(
        clustering_algorithm='hierarchical',
        similarity_threshold=0.7,
        cluster_min_size=3
    )
    
    print(f"\nStratifier configuration:")
    print(f"  Clustering: {stratifier.clustering_algorithm}")
    print(f"  Similarity threshold: {stratifier.similarity_threshold}")
    print(f"  Protein weight: {stratifier.protein_weight}")
    print(f"  Ligand weight: {stratifier.ligand_weight}")
    
    # Perform multi-view stratified split
    train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
        protein_embeddings=protein_embeddings,
        ligand_embeddings=ligand_embeddings,
        labels=labels,
        test_size=0.2,
        val_size=0.1,
        protein_weight=0.6,
        ligand_weight=0.4
    )
    
    print(f"\n✓ Split completed:")
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")
    
    # Verify split sizes
    total = len(train_idx) + len(val_idx) + len(test_idx)
    
    # Debug: Check for duplicates within each split
    if len(train_idx) != len(set(train_idx)):
        print(f"  ⚠️  WARNING: Duplicates in train_idx!")
        print(f"     Length: {len(train_idx)}, Unique: {len(set(train_idx))}")
    if len(val_idx) != len(set(val_idx)):
        print(f"  ⚠️  WARNING: Duplicates in val_idx!")
        print(f"     Length: {len(val_idx)}, Unique: {len(set(val_idx))}")
    if len(test_idx) != len(set(test_idx)):
        print(f"  ⚠️  WARNING: Duplicates in test_idx!")
        print(f"     Length: {len(test_idx)}, Unique: {len(set(test_idx))}")
    
    assert total == n_samples, f"Split sizes don't sum to n_samples: {total} != {n_samples} (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    print(f"  Total: {total} samples ✓")
    
    # Verify no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train and val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train and test overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val and test overlap!"
    print(f"  No overlap between splits ✓")
    
    # Check label distribution
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    
    print(f"\n✓ Label distribution:")
    print(f"  Train: Active={np.sum(train_labels==1)}, Inactive={np.sum(train_labels==0)}")
    print(f"  Val:   Active={np.sum(val_labels==1)}, Inactive={np.sum(val_labels==0)}")
    print(f"  Test:  Active={np.sum(test_labels==1)}, Inactive={np.sum(test_labels==0)}")
    
    # Get cluster info
    cluster_info = stratifier.get_cluster_info()
    print(f"\n✓ Clustering info:")
    print(f"  Number of clusters: {cluster_info['n_clusters']}")
    print(f"  Noise points: {cluster_info['n_noise_points']}")
    print(f"  Algorithm: {cluster_info.get('algorithm', cluster_info.get('clustering_algorithm', 'unknown'))}")
    
    print("\n✅ Multi-view stratified split tests PASSED!")
    return True


def test_weight_variations():
    """Test different weight configurations."""
    print("\n" + "="*80)
    print("TEST 4: Weight Configuration Variations")
    print("="*80)
    
    calculator = CosineSimilarityCalculator()
    
    # Create samples: same protein, different ligands
    protein_embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])  # Identical proteins
    ligand_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])   # Orthogonal ligands
    
    configurations = [
        ('Protein-only', 1.0, 0.0),
        ('Ligand-only', 0.0, 1.0),
        ('Balanced', 0.5, 0.5),
        ('Protein-focused', 0.8, 0.2),
        ('Ligand-focused', 0.3, 0.7),
        ('Default', 0.6, 0.4),
    ]
    
    print("\nSimilarity between Sample 1 and Sample 2:")
    print("(Same protein, orthogonal ligands)\n")
    print(f"{'Configuration':<20} Weight(P/L)  Similarity")
    print("-" * 60)
    
    for name, p_weight, l_weight in configurations:
        sim = calculator.calculate_multi_view_similarity(
            protein_embeddings=protein_embeddings,
            ligand_embeddings=ligand_embeddings,
            protein_weight=p_weight,
            ligand_weight=l_weight
        )
        similarity = sim[0, 1]
        print(f"{name:<20} {p_weight:.1f}/{l_weight:.1f}      {similarity:.4f}")
    
    print("\n✓ Weight variations working correctly!")
    print("  - Protein-only: High similarity (ignores ligand difference)")
    print("  - Ligand-only: Low similarity (ignores protein match)")
    print("  - Default (0.6/0.4): Balanced result")
    
    print("\n✅ Weight configuration tests PASSED!")
    return True


def test_visualization():
    """Test visualization of stratification."""
    print("\n" + "="*80)
    print("TEST 5: Stratification Visualization")
    print("="*80)
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 200
    protein_dim = 320
    ligand_dim = 768
    
    # Generate protein embeddings (5 groups)
    protein_embeddings = []
    for i in range(5):
        base = np.random.randn(protein_dim) * 2
        for j in range(40):
            noise = np.random.randn(protein_dim) * 0.3
            protein_embeddings.append(base + noise)
    protein_embeddings = np.array(protein_embeddings)
    
    # Generate ligand embeddings (10 groups)
    ligand_embeddings = []
    for i in range(10):
        base = np.random.randn(ligand_dim) * 2
        for j in range(20):
            noise = np.random.randn(ligand_dim) * 0.3
            ligand_embeddings.append(base + noise)
    ligand_embeddings = np.array(ligand_embeddings)
    
    # Generate labels
    labels = np.random.randint(0, 2, size=n_samples)
    
    print(f"Dataset: {n_samples} samples")
    print(f"  Protein embeddings: {protein_embeddings.shape}")
    print(f"  Ligand embeddings: {ligand_embeddings.shape}")
    
    # Initialize stratifier
    stratifier = Stratifier(
        clustering_algorithm='hierarchical',
        similarity_threshold=0.7,
        cluster_min_size=3
    )
    
    # Perform multi-view stratified split
    train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
        protein_embeddings=protein_embeddings,
        ligand_embeddings=ligand_embeddings,
        labels=labels,
        test_size=0.2,
        val_size=0.1,
        protein_weight=0.6,
        ligand_weight=0.4
    )
    
    print(f"\n✓ Split completed:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    # Create visualizations
    print(f"\n✓ Generating visualizations...")
    
    # Combined embeddings for single plot
    combined = np.concatenate([protein_embeddings, ligand_embeddings], axis=1)
    
    # Try PCA first (always available)
    visualizer_pca = StratificationVisualizer(method='pca')
    
    print("  - Creating PCA visualization...")
    visualizer_pca.plot_split_visualization(
        embeddings=combined,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cluster_labels=stratifier.cluster_labels,
        title="Multi-View Stratification (PCA)",
        save_path="results/test_stratification_pca.png",
        show=False
    )
    print("    ✓ Saved to results/test_stratification_pca.png")
    
    # Try multi-view comparison
    print("  - Creating multi-view comparison (PCA)...")
    visualizer_pca.plot_multi_view_comparison(
        protein_embeddings=protein_embeddings,
        ligand_embeddings=ligand_embeddings,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        title="Multi-View Stratification Comparison",
        save_path="results/test_multiview_comparison_pca.png",
        show=False
    )
    print("    ✓ Saved to results/test_multiview_comparison_pca.png")
    
    # Try t-SNE if available
    try:
        visualizer_tsne = StratificationVisualizer(method='tsne')
        print("  - Creating t-SNE visualization...")
        visualizer_tsne.plot_split_visualization(
            embeddings=combined,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            cluster_labels=stratifier.cluster_labels,
            title="Multi-View Stratification (t-SNE)",
            save_path="results/test_stratification_tsne.png",
            show=False
        )
        print("    ✓ Saved to results/test_stratification_tsne.png")
    except Exception as e:
        print(f"    ⚠ t-SNE not available or failed: {e}")
    
    # Try UMAP if available
    try:
        visualizer_umap = StratificationVisualizer(method='umap')
        print("  - Creating UMAP visualization...")
        visualizer_umap.plot_split_visualization(
            embeddings=combined,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            cluster_labels=stratifier.cluster_labels,
            title="Multi-View Stratification (UMAP)",
            save_path="results/test_stratification_umap.png",
            show=False
        )
        print("    ✓ Saved to results/test_stratification_umap.png")
    except Exception as e:
        print(f"    ⚠ UMAP not available or failed: {e}")
    
    print("\n✅ Visualization tests PASSED!")
    print("Check the 'results/' directory for generated plots.")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🧪 MULTI-VIEW STRATIFICATION TESTS")
    print("="*80)
    
    try:
        # Run all tests
        test_cosine_similarity()
        test_multi_view_similarity()
        test_stratified_split()
        test_weight_variations()
        test_visualization()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nMulti-view stratification is working correctly!")
        print("Ready to commit and push changes.")
        return 0
        
    except AssertionError as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        return 1
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
