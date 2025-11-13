"""
Example usage of the cosine similarity-based stratification module.

This script demonstrates how to use the new stratification functionality
for creating balanced train/test/validation splits of molecular data.
"""

import numpy as np
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import sys
from pathlib import Path

# Add the src directory to Python path for this example
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.build.core import BuildConfig
from src.build.pipeline import BuildPipeline
from src.build.stratification import Stratifier, SplitValidator, CosineSimilarityCalculator


def example_basic_stratification():
    """Example: Basic stratification usage."""
    print("🧪 EXAMPLE 1: Basic Stratification Usage")
    print("-" * 50)
    
    # Create sample embeddings and labels
    np.random.seed(42)
    embeddings = np.random.rand(100, 50)  # 100 samples, 50-dimensional embeddings
    labels = np.random.randint(0, 3, 100)  # 3 classes for classification
    
    # Initialize stratifier with default parameters
    stratifier = Stratifier()
    
    # Perform stratified split
    train_idx, val_idx, test_idx = stratifier.stratified_split(
        embeddings=embeddings,
        labels=labels,
        test_size=0.2,  # 20% for test
        val_size=0.1   # 10% for validation
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    
    # Validate the splits
    validator = SplitValidator()
    validation_report = validator.validate_splits_comprehensively(
        embeddings, labels, train_idx, val_idx, test_idx
    )
    
    print(f"Overall split quality score: {validation_report['overall_quality_score']:.3f}")
    
    # Print validation report
    if validation_report['issues']:
        print(f"⚠️ Issues found: {validation_report['issues']}")
    else:
        print("✅ No significant issues found!")
    
    print()


def example_configured_stratification():
    """Example: Stratification with custom configuration."""
    print("🔧 EXAMPLE 2: Configured Stratification")
    print("-" * 50)
    
    # Create configuration with custom stratification parameters
    config = BuildConfig({
        'stratification_params': {
            'clustering_algorithm': 'dbscan',
            'similarity_threshold': 0.75,
            'cluster_min_size': 3,
            'stratify_by': 'both',
            'protein_weight': 0.6,
            'ligand_weight': 0.4
        }
    })
    
    # Initialize stratifier with configuration
    stratifier = Stratifier(config)
    
    # Create sample data (simulating protein-ligand embeddings)
    np.random.seed(123)
    protein_embeddings = np.random.rand(80, 20)  # 80 samples, 20-dim protein embeddings
    ligand_embeddings = np.random.rand(80, 15)   # 80 samples, 15-dim ligand embeddings
    labels = np.random.randint(0, 2, 80)  # Binary classification labels
    
    # Perform multi-view stratified split
    train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
        protein_embeddings=protein_embeddings,
        ligand_embeddings=ligand_embeddings,
        labels=labels,
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"Multi-view split sizes:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples") 
    print(f"  Test: {len(test_idx)} samples")
    
    # Analyze clustering
    cluster_info = stratifier.get_cluster_info()
    print(f"Number of clusters: {cluster_info['n_clusters']}")
    print(f"Number of noise points: {cluster_info['n_noise_points']}")
    
    print()


def example_pipeline_integration():
    """Example: Integrating stratification with BuildPipeline."""
    print("⚙️ EXAMPLE 3: Pipeline Integration")
    print("-" * 50)
    
    # Create configuration enabling stratification
    config = BuildConfig({
        'stratification_enabled': True,
        'stratification_params': {
            'clustering_algorithm': 'dbscan',
            'similarity_threshold': 0.8,
            'cluster_min_size': 5
        }
    })
    
    # Initialize pipeline
    pipeline = BuildPipeline(config)
    
    print(f"Pipeline initialized with {len(pipeline.components)} components")
    print(f"Stratifier available: {'stratifier' in pipeline.components}")
    print(f"Split validator available: {'split_validator' in pipeline.components}")
    
    # Show available configuration
    stratifier = pipeline.components['stratifier']
    print(f"Current clustering algorithm: {stratifier.clustering_algorithm}")
    print(f"Similarity threshold: {stratifier.similarity_threshold}")
    
    print()


def example_similarity_calculation():
    """Example: Using cosine similarity calculator directly."""
    print("📐 EXAMPLE 4: Cosine Similarity Calculation")
    print("-" * 50)
    
    # Create sample embeddings
    calc = CosineSimilarityCalculator(normalize=True)
    
    # Sample embeddings
    embedding1 = np.array([1, 0, 0, 0])
    embedding2 = np.array([0.9, 0.1, 0, 0])  # Similar to embedding1
    embedding3 = np.array([0, 0, 1, 0])      # Orthogonal to others
    
    # Calculate single pair similarity
    sim_12 = calc.calculate_single_pair(embedding1, embedding2)
    sim_13 = calc.calculate_single_pair(embedding1, embedding3)
    
    print(f"Similarity between embedding1 and embedding2: {sim_12:.3f}")
    print(f"Similarity between embedding1 and embedding3: {sim_13:.3f}")
    
    # Calculate batch similarities
    embeddings = np.array([embedding1, embedding2, embedding3])
    similarity_matrix = calc.calculate_batch(embeddings)
    print(f"Full similarity matrix shape: {similarity_matrix.shape}")
    print("Similarity matrix:")
    print(similarity_matrix)
    
    print()


def example_custom_stratification_workflow():
    """Example: Complete custom stratification workflow."""
    print("🧪 EXAMPLE 5: Complete Custom Workflow")
    print("-" * 50)
    
    # Simulate a dataset with embeddings and labels
    np.random.seed(456)
    n_samples = 200
    
    # Create embeddings with some structure (similar samples)
    embeddings = np.random.rand(n_samples, 30)
    # Introduce some similarity by adding cluster-like structure
    for i in range(0, n_samples, 10):  # Every 10th sample is similar to the first in its group
        if i + 1 < n_samples:
            embeddings[i+1:i+3] = embeddings[i] + np.random.normal(0, 0.1, (2, 30))
    
    # Create labels (binary classification)
    labels = np.random.randint(0, 2, n_samples)
    
    print(f"Dataset: {n_samples} samples, {embeddings.shape[1]}-dimensional embeddings")
    
    # Test different clustering algorithms
    algorithms = ['dbscan', 'random']
    results = {}
    
    for algo in algorithms:
        stratifier = Stratifier(clustering_algorithm=algo, similarity_threshold=0.7)
        
        try:
            train_idx, val_idx, test_idx = stratifier.stratified_split(
                embeddings=embeddings,
                labels=labels,
                test_size=0.2,
                val_size=0.1
            )
            
            # Validate this split
            validator = SplitValidator()
            validation_report = validator.validate_splits_comprehensively(
                embeddings, labels, train_idx, val_idx, test_idx
            )
            
            results[algo] = {
                'quality_score': validation_report['overall_quality_score'],
                'n_clusters': stratifier.get_cluster_info().get('n_clusters', 0),
                'size_info': {
                    'train': len(train_idx),
                    'val': len(val_idx),
                    'test': len(test_idx)
                }
            }
        except Exception as e:
            print(f"Algorithm {algo} failed: {e}")
            continue
    
    # Print comparison
    for algo, result in results.items():
        print(f"\nAlgorithm '{algo}':")
        print(f"  Quality score: {result['quality_score']:.3f}")
        print(f"  Number of clusters: {result['n_clusters']}")
        print(f"  Split sizes: {result['size_info']}")
    
    print()


def main():
    """Run all examples."""
    print("🚀 DockTKinase - Stratification Examples")
    print("=" * 60)
    
    example_basic_stratification()
    example_configured_stratification()
    example_pipeline_integration()
    example_similarity_calculation()
    example_custom_stratification_workflow()
    
    print("✅ All examples completed successfully!")


if __name__ == "__main__":
    main()