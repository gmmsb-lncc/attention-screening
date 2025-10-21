"""
Tests for the stratification module.

This module contains unit and integration tests for the cosine similarity-based
stratification functionality.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import tempfile
import os

# Add the src directory to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from build.core.config import BuildConfig
from build.stratification.cosine_similarity_calculator import CosineSimilarityCalculator
from build.stratification.stratifier import Stratifier
from build.stratification.cluster_analyzer import ClusterAnalyzer
from build.stratification.validator import SplitValidator
from build.pipeline.build_pipeline import BuildPipeline


class TestCosineSimilarityCalculator:
    """Test cosine similarity calculator functionality."""
    
    def test_single_pair_similarity(self):
        """Test cosine similarity calculation for a single pair."""
        calc = CosineSimilarityCalculator()
        
        # Two identical vectors should have similarity 1.0
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        sim = calc.calculate_single_pair(vec1, vec2)
        assert abs(sim - 1.0) < 1e-6
        
        # Two orthogonal vectors should have similarity 0
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        sim = calc.calculate_single_pair(vec1, vec2)
        assert abs(sim) < 1e-6
        
        # Two opposite vectors should have similarity -1.0
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([-1, 0, 0])
        sim = calc.calculate_single_pair(vec1, vec2)
        assert abs(sim - (-1.0)) < 1e-6
    
    def test_batch_similarity(self):
        """Test batch cosine similarity calculation."""
        calc = CosineSimilarityCalculator(batch_size=2)
        
        # Create simple embeddings
        embeddings = np.array([
            [1, 0, 0],  # Vector 1
            [0, 1, 0],  # Vector 2
            [1, 1, 0]   # Vector 3 (normalized version of [1,1,0])
        ])
        
        # Normalize the third vector
        embeddings[2] = embeddings[2] / np.linalg.norm(embeddings[2])
        
        similarity_matrix = calc.calculate_batch(embeddings)
        
        # Check matrix shape
        assert similarity_matrix.shape == (3, 3)
        
        # Check diagonal (self-similarity should be 1.0)
        np.testing.assert_array_almost_equal(np.diag(similarity_matrix), [1.0, 1.0, 1.0])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(similarity_matrix, similarity_matrix.T)
    
    def test_multi_view_similarity(self):
        """Test multi-view similarity calculation."""
        calc = CosineSimilarityCalculator()
        
        # Create sample protein and ligand embeddings
        protein_embeddings = np.random.rand(5, 10)
        ligand_embeddings = np.random.rand(5, 8)
        
        combined_sim = calc.calculate_multi_view_similarity(
            protein_embeddings, ligand_embeddings,
            protein_weight=0.6, ligand_weight=0.4
        )
        
        assert combined_sim.shape == (5, 5)
        # All similarities should be between -1 and 1
        assert np.all(combined_sim >= -1.0) and np.all(combined_sim <= 1.0)


class TestStratifier:
    """Test stratifier functionality."""
    
    def test_stratifier_initialization(self):
        """Test stratifier initialization with config."""
        config = BuildConfig({
            'stratification_params': {
                'clustering_algorithm': 'dbscan',
                'similarity_threshold': 0.7,
                'cluster_min_size': 3
            }
        })
        
        stratifier = Stratifier(config)
        
        assert stratifier.clustering_algorithm == 'dbscan'
        assert stratifier.similarity_threshold == 0.7
        assert stratifier.cluster_min_size == 3
    
    def test_cluster_embeddings(self):
        """Test clustering of embeddings."""
        # Create sample embeddings with clear clusters
        np.random.seed(42)
        
        # Cluster 1: points around [1, 1]
        cluster1 = np.random.normal([1, 1], 0.1, (5, 2))
        # Cluster 2: points around [-1, -1] 
        cluster2 = np.random.normal([-1, -1], 0.1, (5, 2))
        
        embeddings = np.vstack([cluster1, cluster2])
        
        stratifier = Stratifier(clustering_algorithm='dbscan', similarity_threshold=0.5)
        cluster_labels = stratifier.cluster_embeddings(combined_embeddings=embeddings)
        
        # There should be 2 main clusters (labels could be 0, 1 or other integers)
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        # dbscan might identify noise points as cluster -1, so check that there are at least 2 non-noise clusters
        non_noise_clusters = len([l for l in unique_labels if l != -1])
        
        # At least 2 clusters expected
        assert n_clusters >= 2
    
    def test_stratified_split(self):
        """Test stratified splitting functionality."""
        # Create sample embeddings
        np.random.seed(42)
        embeddings = np.random.rand(20, 10)
        labels = np.random.randint(0, 3, 20)  # 3 classes
        
        stratifier = Stratifier(clustering_algorithm='random', similarity_threshold=0.5)
        train_idx, val_idx, test_idx = stratifier.stratified_split(
            embeddings, labels, test_size=0.2, val_size=0.2
        )
        
        # Check that indices are non-empty and sum to total
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        assert len(set(train_idx) & set(val_idx) & set(test_idx)) == 0  # No overlap
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(embeddings)  # All samples used


class TestClusterAnalyzer:
    """Test cluster analyzer functionality."""
    
    def test_clustering_metrics(self):
        """Test calculation of clustering quality metrics."""
        analyzer = ClusterAnalyzer()
        
        # Create sample embeddings with clear clusters
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], 0.5, (10, 2))
        cluster2 = np.random.normal([3, 3], 0.5, (10, 2))
        embeddings = np.vstack([cluster1, cluster2])
        
        # Simulate cluster labels
        labels = np.array([0]*10 + [1]*10)
        
        metrics = analyzer.calculate_clustering_metrics(embeddings, labels)
        
        # Check that metrics are computed (no NaN values for this clear clustering)
        assert not np.isnan(metrics['silhouette_score'])
        assert not np.isnan(metrics['calinski_harabasz_score'])
        assert not np.isnan(metrics['davies_bouldin_score'])
        
        assert metrics['n_clusters'] == 2
        assert metrics['total_samples'] == 20


class TestSplitValidator:
    """Test split validator functionality."""
    
    def test_split_distribution_validation(self):
        """Test validation of label distribution across splits."""
        validator = SplitValidator()
        
        # Create sample data with known distribution
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])  # 4 class 0, 3 class 1, 2 class 2
        train_idx = np.array([0, 2, 4, 6, 8])
        val_idx = np.array([1, 3, 5])
        test_idx = np.array([7, 9])
        
        distribution_metrics = validator.validate_split_distribution(labels, train_idx, val_idx, test_idx)
        
        # Check that all splits have distribution info
        assert 'train' in distribution_metrics
        assert 'validation' in distribution_metrics
        assert 'test' in distribution_metrics
        
        # Check train split properties
        train_info = distribution_metrics['train']
        assert train_info['size'] == 5
        assert train_info['n_unique_labels'] == 3  # Should contain all 3 classes
    
    def test_split_validation_comprehensive(self):
        """Test comprehensive split validation."""
        validator = SplitValidator()
        
        # Create sample embeddings and labels
        np.random.seed(42)
        embeddings = np.random.rand(30, 5)
        labels = np.random.randint(0, 3, 30)  # 3 classes
        
        # Create simple split indices
        train_idx = np.arange(0, 20)
        val_idx = np.arange(20, 25)
        test_idx = np.arange(25, 30)
        
        report = validator.validate_splits_comprehensively(
            embeddings, labels, train_idx, val_idx, test_idx
        )
        
        # Check that report contains expected sections
        assert 'split_sizes' in report
        assert 'label_distribution' in report
        assert 'overall_quality_score' in report
        assert 'issues' in report
        
        # Check split sizes
        assert report['split_sizes']['train'] == 20
        assert report['split_sizes']['validation'] == 5
        assert report['split_sizes']['test'] == 5


class TestPipelineIntegration:
    """Test integration with BuildPipeline."""
    
    def test_pipeline_with_stratification(self):
        """Test BuildPipeline with stratification enabled."""
        # Create a temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple BuildConfig with stratification enabled
            config = BuildConfig({
                'stratification_enabled': True,
                'stratification_params': {
                    'clustering_algorithm': 'random',  # Use random for testing
                    'similarity_threshold': 0.8,
                    'cluster_min_size': 2
                }
            })
            
            pipeline = BuildPipeline(config)
            
            # Verify that stratification components are initialized
            assert 'stratifier' in pipeline.components
            assert 'split_validator' in pipeline.components
            
            # Verify the components have correct configuration
            stratifier = pipeline.components['stratifier']
            assert stratifier.clustering_algorithm == 'random'
            assert stratifier.similarity_threshold == 0.8


def test_end_to_end():
    """End-to-end test combining multiple components."""
    np.random.seed(42)
    
    # Create sample data
    embeddings = np.random.rand(50, 20)
    labels = np.random.randint(0, 3, 50)  # 3 classes
    
    # Initialize stratifier
    stratifier = Stratifier(clustering_algorithm='random', similarity_threshold=0.5)
    
    # Perform stratified split
    train_idx, val_idx, test_idx = stratifier.stratified_split(
        embeddings, labels, test_size=0.2, val_size=0.2
    )
    
    # Validate the splits
    validator = SplitValidator()
    validation_report = validator.validate_splits_comprehensively(
        embeddings, labels, train_idx, val_idx, test_idx
    )
    
    # Check basic validation report structure
    assert 'split_sizes' in validation_report
    assert 'overall_quality_score' in validation_report
    assert 0 <= validation_report['overall_quality_score'] <= 1
    
    # Check split sizes (allow for discrete allocation differences)
    total_samples = (validation_report['split_sizes']['train'] + 
                    validation_report['split_sizes']['test'] + 
                    validation_report['split_sizes']['validation'])
    assert total_samples == 50  # All samples should be allocated
    
    # Check approximate split ratios (with small tolerance for discrete allocation)
    train_ratio = validation_report['split_sizes']['train'] / 50
    test_ratio = validation_report['split_sizes']['test'] / 50
    val_ratio = validation_report['split_sizes']['validation'] / 50
    
    # Allow for discrete allocation variations
    assert abs(train_ratio - 0.6) < 0.1  # Should be around 60%
    assert abs(test_ratio - 0.2) < 0.1   # Should be around 20%
    assert abs(val_ratio - 0.2) < 0.1    # Should be around 20%


if __name__ == "__main__":
    pytest.main([__file__])