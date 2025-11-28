"""
Tests for the leakage_aware stratification method.

This module tests the new leakage_aware method in AdaptiveClustering that
optimizes for train/val/test split quality by:
1. Minimizing similarity (leakage) between splits
2. Maintaining representativity within each split  
3. Preserving label distribution balance

Author: DockTKinase Team
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from build.stratification.adaptive_clustering import AdaptiveClustering, ClusteringMetrics
from build.stratification.stratifier import Stratifier


class TestLeakageAwareMethod:
    """Tests for the leakage_aware threshold selection method."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        # Create 3 clusters with some overlap
        n_per_cluster = 50
        
        # Cluster 1: centered around [0, 0]
        cluster1 = np.random.randn(n_per_cluster, 64) * 0.3 + np.array([0] * 64)
        # Cluster 2: centered around [2, 2]  
        cluster2 = np.random.randn(n_per_cluster, 64) * 0.3 + np.array([2] * 64)
        # Cluster 3: centered around [-2, 2]
        cluster3 = np.random.randn(n_per_cluster, 64) * 0.3 + np.array([-2] * 64)
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        return embeddings
    
    @pytest.fixture
    def sample_labels(self, sample_embeddings):
        """Create sample labels (pIC50 values) for testing."""
        np.random.seed(42)
        n_samples = len(sample_embeddings)
        # Create labels with some correlation to clusters
        labels = np.random.rand(n_samples) * 4 + 5  # pIC50 range: 5-9
        return labels
    
    @pytest.fixture
    def homogeneous_embeddings(self):
        """Create highly homogeneous embeddings (challenging case)."""
        np.random.seed(42)
        n_samples = 100
        # All embeddings very similar
        base = np.ones(64) * 0.5
        embeddings = base + np.random.randn(n_samples, 64) * 0.05
        return embeddings
    
    @pytest.fixture
    def homogeneous_labels(self, homogeneous_embeddings):
        """Labels for homogeneous embeddings."""
        np.random.seed(42)
        return np.random.rand(len(homogeneous_embeddings)) * 4 + 5

    # =========================================================================
    # Basic functionality tests
    # =========================================================================
    
    def test_leakage_aware_initialization(self):
        """Test that AdaptiveClustering initializes correctly with leakage_aware method."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            test_size=0.2,
            val_size=0.1
        )
        
        assert ac.method == 'leakage_aware'
        assert ac.test_size == 0.2
        assert ac.val_size == 0.1
    
    def test_set_target_labels(self, sample_embeddings, sample_labels):
        """Test setting target labels for leakage_aware."""
        ac = AdaptiveClustering(method='leakage_aware')
        ac.set_target_labels(sample_labels)
        
        assert ac._target_labels is not None
        assert len(ac._target_labels) == len(sample_labels)
    
    def test_leakage_aware_requires_labels(self, sample_embeddings):
        """Test that leakage_aware falls back to target when labels not provided."""
        ac = AdaptiveClustering(method='leakage_aware')
        # Don't set labels
        
        labels = ac.cluster(sample_embeddings)
        
        # Should still work (fallback to target)
        assert labels is not None
        assert len(labels) == len(sample_embeddings)
    
    def test_leakage_aware_with_labels(self, sample_embeddings, sample_labels):
        """Test leakage_aware clustering with labels provided."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            test_size=0.2,
            val_size=0.1,
            min_clusters=3
        )
        ac.set_target_labels(sample_labels)
        
        labels = ac.cluster(sample_embeddings)
        
        # Verify output
        assert labels is not None
        assert len(labels) == len(sample_embeddings)
        
        # Verify metrics were computed
        assert ac.metrics is not None
        assert ac.optimal_threshold is not None
        
        # Verify split quality metrics exist
        assert ac.split_quality_metrics is not None

    # =========================================================================
    # Split quality evaluation tests
    # =========================================================================
    
    def test_evaluate_split_quality(self, sample_embeddings, sample_labels):
        """Test the split quality evaluation function."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        n_samples = len(sample_embeddings)
        # Create artificial split
        train_idx = np.arange(0, int(n_samples * 0.7))
        val_idx = np.arange(int(n_samples * 0.7), int(n_samples * 0.8))
        test_idx = np.arange(int(n_samples * 0.8), n_samples)
        
        metrics = ac._evaluate_split_quality(
            sample_embeddings, sample_labels,
            train_idx, val_idx, test_idx
        )
        
        # Check all expected keys
        assert 'final_score' in metrics
        assert 'separation' in metrics
        assert 'coverage' in metrics
        assert 'balance' in metrics
        assert 'sizes' in metrics
        
        # Check score ranges
        assert 0 <= metrics['final_score'] <= 1
        assert 0 <= metrics['separation']['score'] <= 1
        assert 0 <= metrics['coverage']['score'] <= 1
        assert 0 <= metrics['balance']['score'] <= 1
        
        # Check sizes
        assert metrics['sizes']['train'] == len(train_idx)
        assert metrics['sizes']['val'] == len(val_idx)
        assert metrics['sizes']['test'] == len(test_idx)
    
    def test_max_similarity_between_splits(self, sample_embeddings):
        """Test max similarity calculation between splits."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        n = len(sample_embeddings)
        idx_a = np.arange(0, n // 2)
        idx_b = np.arange(n // 2, n)
        
        max_sim = ac._max_similarity_between_splits(
            sample_embeddings, idx_a, idx_b
        )
        
        assert 0 <= max_sim <= 1
    
    def test_mean_pairwise_distance(self, sample_embeddings):
        """Test mean pairwise distance calculation."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        distance = ac._mean_pairwise_distance(sample_embeddings)
        
        assert 0 <= distance <= 2  # Cosine distance range
    
    def test_label_distribution(self, sample_labels):
        """Test label distribution calculation."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        # Discretize labels for distribution
        discrete_labels = (sample_labels > np.median(sample_labels)).astype(int)
        dist = ac._label_distribution(discrete_labels)
        
        assert dist is not None
        assert np.isclose(dist.sum(), 1.0)
    
    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        
        kl = ac._kl_divergence(p, q)
        assert kl >= 0  # KL divergence is non-negative
        assert kl < 0.1  # Should be close to 0 for identical distributions
        
        # Test with different distributions
        p2 = np.array([0.9, 0.1])
        q2 = np.array([0.5, 0.5])
        
        kl2 = ac._kl_divergence(p2, q2)
        assert kl2 > kl  # Should be larger for different distributions

    # =========================================================================
    # Threshold search tests
    # =========================================================================
    
    def test_find_threshold_leakage_aware(self, sample_embeddings, sample_labels):
        """Test the leakage-aware threshold search."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            min_clusters=3,
            test_size=0.2,
            val_size=0.1
        )
        
        sim_stats = ac.analyze_similarity_distribution(sample_embeddings)
        
        threshold, history, metrics = ac.find_threshold_leakage_aware(
            sample_embeddings, sample_labels, sim_stats, n_candidates=5
        )
        
        assert threshold is not None
        assert 0 < threshold < 1
        assert len(history) > 0
        assert metrics is not None
    
    def test_threshold_search_returns_valid_history(self, sample_embeddings, sample_labels):
        """Test that threshold search returns valid history."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            min_clusters=3
        )
        
        sim_stats = ac.analyze_similarity_distribution(sample_embeddings)
        
        _, history, _ = ac.find_threshold_leakage_aware(
            sample_embeddings, sample_labels, sim_stats, n_candidates=5
        )
        
        # Check history structure
        for entry in history:
            assert 'threshold' in entry
            # May have 'skipped' or 'error' or full metrics

    # =========================================================================
    # Edge case tests
    # =========================================================================
    
    def test_homogeneous_data(self, homogeneous_embeddings, homogeneous_labels):
        """Test with highly homogeneous data (challenging case)."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            min_clusters=3,
            test_size=0.2,
            val_size=0.1
        )
        ac.set_target_labels(homogeneous_labels)
        
        # Should not raise, may fall back to target
        labels = ac.cluster(homogeneous_embeddings)
        
        assert labels is not None
        assert len(labels) == len(homogeneous_embeddings)
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 32)
        labels = np.random.rand(20) * 4 + 5
        
        ac = AdaptiveClustering(
            method='leakage_aware',
            min_clusters=2,
            min_cluster_size=2,
            test_size=0.2,
            val_size=0.1
        )
        ac.set_target_labels(labels)
        
        cluster_labels = ac.cluster(embeddings)
        
        assert cluster_labels is not None
        assert len(cluster_labels) == len(embeddings)
    
    def test_empty_splits_handled(self):
        """Test handling of edge cases with empty splits."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        embeddings = np.random.randn(10, 32)
        labels = np.random.rand(10)
        
        # Test with empty indices
        max_sim = ac._max_similarity_between_splits(
            embeddings, np.array([0, 1]), np.array([])
        )
        assert max_sim == 0.0

    # =========================================================================
    # Integration with Stratifier tests
    # =========================================================================
    
    def test_stratifier_with_leakage_aware(self, sample_embeddings, sample_labels):
        """Test Stratifier integration with leakage_aware method."""
        stratifier = Stratifier(
            clustering_algorithm='adaptive',
            adaptive_method='leakage_aware',
            test_size=0.2,
            val_size=0.1
        )
        
        train_idx, val_idx, test_idx = stratifier.stratified_split(
            sample_embeddings, sample_labels,
            test_size=0.2,
            val_size=0.1
        )
        
        # Verify split sizes (approximately correct)
        n_total = len(sample_embeddings)
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        
        # Verify no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        
        # Verify all indices covered
        all_indices = set(train_idx) | set(val_idx) | set(test_idx)
        assert all_indices == set(range(n_total))
    
    def test_stratifier_validation_of_method(self):
        """Test that Stratifier validates adaptive_method."""
        # Valid methods should work
        for method in ['silhouette', 'elbow', 'target', 'percentile', 'leakage_aware']:
            stratifier = Stratifier(
                clustering_algorithm='adaptive',
                adaptive_method=method
            )
            assert stratifier.adaptive_method == method
        
        # Invalid method should raise
        with pytest.raises(Exception):
            stratifier = Stratifier(
                clustering_algorithm='adaptive',
                adaptive_method='invalid_method'
            )
            stratifier._validate_config()

    # =========================================================================
    # Comparison tests
    # =========================================================================
    
    def test_leakage_aware_vs_target_comparison(self, sample_embeddings, sample_labels):
        """Compare leakage_aware with target method."""
        # Target method
        ac_target = AdaptiveClustering(method='target', min_clusters=3)
        labels_target = ac_target.cluster(sample_embeddings)
        
        # Leakage-aware method
        ac_leakage = AdaptiveClustering(
            method='leakage_aware', 
            min_clusters=3,
            test_size=0.2,
            val_size=0.1
        )
        ac_leakage.set_target_labels(sample_labels)
        labels_leakage = ac_leakage.cluster(sample_embeddings)
        
        # Both should produce valid clusterings
        assert len(labels_target) == len(sample_embeddings)
        assert len(labels_leakage) == len(sample_embeddings)
        
        # Leakage-aware should have split quality metrics
        assert ac_leakage.split_quality_metrics is not None


class TestClusteringMetrics:
    """Tests for ClusteringMetrics dataclass."""
    
    def test_metrics_include_split_quality(self, sample_embeddings, sample_labels):
        """Test that metrics include split_quality_metrics field."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            min_clusters=3,
            test_size=0.2,
            val_size=0.1
        )
        ac.set_target_labels(sample_labels)
        ac.cluster(sample_embeddings)
        
        metrics = ac.metrics
        
        assert isinstance(metrics, ClusteringMetrics)
        # split_quality_metrics should be set
        assert hasattr(metrics, 'split_quality_metrics')
    
    def test_metrics_to_dict(self, sample_embeddings, sample_labels):
        """Test conversion of metrics to dictionary."""
        ac = AdaptiveClustering(
            method='leakage_aware',
            min_clusters=3,
            test_size=0.2,
            val_size=0.1
        )
        ac.set_target_labels(sample_labels)
        ac.cluster(sample_embeddings)
        
        metrics_dict = ac.metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'n_clusters' in metrics_dict
        assert 'method' in metrics_dict
        assert 'split_quality_metrics' in metrics_dict

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        n_per_cluster = 50
        
        cluster1 = np.random.randn(n_per_cluster, 64) * 0.3 + np.array([0] * 64)
        cluster2 = np.random.randn(n_per_cluster, 64) * 0.3 + np.array([2] * 64)
        cluster3 = np.random.randn(n_per_cluster, 64) * 0.3 + np.array([-2] * 64)
        
        return np.vstack([cluster1, cluster2, cluster3])
    
    @pytest.fixture
    def sample_labels(self, sample_embeddings):
        """Create sample labels for testing."""
        np.random.seed(42)
        return np.random.rand(len(sample_embeddings)) * 4 + 5


class TestAnalyzeSimilarityDistribution:
    """Tests for similarity distribution analysis."""
    
    def test_similarity_stats_computed(self):
        """Test that similarity statistics are computed correctly."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 32)
        
        ac = AdaptiveClustering(method='leakage_aware')
        stats = ac.analyze_similarity_distribution(embeddings)
        
        # Check all expected keys
        expected_keys = ['min', 'max', 'mean', 'std', 'median', 
                        'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99',
                        'homogeneity']
        
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_homogeneity_classification(self):
        """Test homogeneity is correctly classified."""
        ac = AdaptiveClustering(method='leakage_aware')
        
        # Test very_high homogeneity
        np.random.seed(42)
        base = np.ones(32) * 0.5
        very_similar = base + np.random.randn(50, 32) * 0.01
        stats = ac.analyze_similarity_distribution(very_similar)
        assert stats['homogeneity'] in ['very_high', 'high']
        
        # Test low homogeneity
        np.random.seed(42)
        diverse = np.random.randn(50, 32) * 5
        stats = ac.analyze_similarity_distribution(diverse)
        # Low homogeneity has min < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
