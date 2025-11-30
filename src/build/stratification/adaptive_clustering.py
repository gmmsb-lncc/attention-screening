"""
Adaptive Clustering Strategy - Main orchestrator module.

This module provides the main AdaptiveClustering class that orchestrates
various clustering strategies and threshold optimization methods.

For large datasets (>40k samples), automatically uses scalable clustering
based on Representative Sampling + Label Propagation.

Methods implemented:
1. Silhouette-based threshold optimization
2. Elbow method for k-means
3. Target cluster count based on dataset size
4. Percentile-based threshold
5. Manual threshold
6. Leakage-aware: Optimizes for split quality (train/val/test separation)

Author: DockTKinase Team
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

from .clustering_metrics import ClusteringMetrics
from .similarity_analysis import SimilarityAnalyzer
from .threshold_optimization import ThresholdOptimizer
from .leakage_aware_optimization import LeakageAwareOptimizer
from .scalable_clustering import ScalableClustering


class AdaptiveClustering:
    """
    Adaptive clustering strategy that automatically finds optimal parameters.
    
    Solves the problem of homogeneous embeddings where a fixed threshold like 0.7
    results in a single cluster because all embeddings are very similar.
    
    Methods:
    - 'silhouette': Optimize threshold using silhouette score
    - 'elbow': Use elbow method to find optimal k for k-means
    - 'target': Aim for a target number of clusters based on dataset size
    - 'percentile': Use similarity percentile as threshold
    - 'manual': Use user-specified threshold (no automatic optimization)
    - 'leakage_aware': Optimize for split quality (minimize leakage between train/val/test)
    """
    
    # Maximum samples for full distance matrix (~8GB RAM)
    MAX_SAMPLES_FOR_FULL_MATRIX = 40000
    
    def __init__(self,
                 method: str = 'silhouette',
                 min_clusters: int = 5,
                 max_clusters: int = 100,
                 min_cluster_size: int = 3,
                 target_cluster_ratio: float = 0.01,
                 manual_threshold: Optional[float] = None,
                 test_size: float = 0.1,
                 val_size: float = 0.1,
                 auto_threshold: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize adaptive clustering.
        
        Args:
            method: Method to use ('silhouette', 'elbow', 'target', 'percentile', 'manual', 'leakage_aware')
            min_clusters: Minimum number of clusters desired
            max_clusters: Maximum number of clusters
            min_cluster_size: Minimum points per cluster
            target_cluster_ratio: Target clusters as ratio of samples (for 'target' method)
            manual_threshold: User-specified similarity threshold (for 'manual' method or override)
            test_size: Proportion for test set (for 'leakage_aware' method)
            val_size: Proportion for validation set (for 'leakage_aware' method)
            auto_threshold: Whether to use automatic threshold detection
            logger: Logger instance
        """
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.target_cluster_ratio = target_cluster_ratio
        self.manual_threshold = manual_threshold
        self.test_size = test_size
        self.val_size = val_size
        self.auto_threshold = auto_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # If manual threshold is provided, switch to manual method
        if manual_threshold is not None and method != 'elbow':
            self.method = 'manual'
            self.logger.info(f"Using manual threshold: {manual_threshold}")
        
        # Results
        self.optimal_threshold: Optional[float] = None
        self.metrics: Optional[ClusteringMetrics] = None
        self.search_history: List[Dict[str, Any]] = []
        self.split_quality_metrics: Optional[Dict[str, Any]] = None
        self._target_labels: Optional[np.ndarray] = None
        
        # Initialize sub-components
        self._similarity_analyzer = SimilarityAnalyzer(logger=self.logger)
        self._threshold_optimizer = ThresholdOptimizer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            logger=self.logger
        )
        self._leakage_optimizer = LeakageAwareOptimizer(
            test_size=test_size,
            val_size=val_size,
            min_clusters=min_clusters,
            min_cluster_size=min_cluster_size,
            logger=self.logger
        )
        self._scalable_clustering = ScalableClustering(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            target_cluster_ratio=target_cluster_ratio,
            manual_threshold=manual_threshold,
            logger=self.logger
        )
    
    def set_target_labels(self, labels: np.ndarray) -> None:
        """Set target labels for leakage_aware method."""
        self._target_labels = labels
    
    def analyze_similarity_distribution(self, 
                                        embeddings: np.ndarray,
                                        sample_size: int = 2000) -> Dict[str, float]:
        """Analyze the distribution of pairwise cosine similarities."""
        return self._similarity_analyzer.analyze_distribution(embeddings, sample_size)
    
    def _calculate_target_clusters(self, n_samples: int) -> int:
        """Calculate target number of clusters based on dataset size."""
        target = int(n_samples * self.target_cluster_ratio)
        return max(self.min_clusters, min(target, self.max_clusters))
    
    def find_optimal_threshold_silhouette(self,
                                          embeddings: np.ndarray,
                                          sim_stats: Dict[str, float],
                                          n_candidates: int = 10) -> Tuple[float, List[Dict]]:
        """Find optimal threshold by maximizing silhouette score."""
        return self._threshold_optimizer.find_optimal_threshold_silhouette(
            embeddings, sim_stats, n_candidates
        )
    
    def find_optimal_k_elbow(self,
                            embeddings: np.ndarray,
                            k_range: Optional[Tuple[int, int]] = None) -> Tuple[int, List[Dict]]:
        """Find optimal k using elbow method."""
        return self._threshold_optimizer.find_optimal_k_elbow(embeddings, k_range)
    
    def find_threshold_by_target(self,
                                embeddings: np.ndarray,
                                sim_stats: Dict[str, float],
                                target_clusters: int) -> Tuple[float, List[Dict]]:
        """Find threshold that produces approximately target number of clusters."""
        return self._threshold_optimizer.find_threshold_by_target(
            embeddings, sim_stats, target_clusters
        )
    
    def find_threshold_leakage_aware(self,
                                     embeddings: np.ndarray,
                                     labels: np.ndarray,
                                     sim_stats: Dict[str, float],
                                     n_candidates: int = 12) -> Tuple[float, List[Dict], Dict[str, Any]]:
        """Find optimal threshold by maximizing split quality score."""
        return self._leakage_optimizer.find_threshold_leakage_aware(
            embeddings, labels, sim_stats, n_candidates
        )
    
    def cluster(self, 
                embeddings: np.ndarray,
                distance_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform adaptive clustering.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            distance_matrix: Pre-computed distance matrix (optional)
            
        Returns:
            Cluster labels
        """
        n_samples = embeddings.shape[0]
        self.logger.info(f"Adaptive clustering with method='{self.method}' on {n_samples} samples")
        
        # Check if dataset is too large for full distance matrix
        if n_samples > self.MAX_SAMPLES_FOR_FULL_MATRIX and distance_matrix is None:
            estimated_memory_gb = (n_samples * n_samples * 4) / (1024**3)
            self.logger.warning(
                f"Dataset too large for full distance matrix clustering "
                f"({n_samples} samples would require ~{estimated_memory_gb:.1f} GiB). "
                f"Using scalable representative sampling approach."
            )
            labels = self._scalable_clustering.cluster(embeddings)
            self.metrics = self._scalable_clustering.metrics
            self.optimal_threshold = self._scalable_clustering.optimal_threshold
            self.search_history = self._scalable_clustering.search_history
            return labels
        
        # Analyze similarity distribution
        sim_stats = self.analyze_similarity_distribution(embeddings)
        self.logger.info(f"Similarity distribution: min={sim_stats['min']:.4f}, "
                        f"max={sim_stats['max']:.4f}, mean={sim_stats['mean']:.4f}")
        self.logger.info(f"Data homogeneity: {sim_stats['homogeneity']}")
        
        # Calculate target clusters
        target_clusters = self._calculate_target_clusters(n_samples)
        self.logger.info(f"Target clusters: {target_clusters} (ratio={self.target_cluster_ratio})")
        
        # Compute distance matrix if not provided
        if distance_matrix is None:
            self.logger.info("Computing distance matrix...")
            sim_matrix = cosine_similarity(embeddings)
            distance_matrix = np.clip(1 - sim_matrix, 0, 2)
        
        # Find optimal parameters based on method
        labels = self._cluster_by_method(embeddings, distance_matrix, sim_stats, target_clusters)
        
        # Filter small clusters
        unique, counts = np.unique(labels, return_counts=True)
        small_clusters = unique[counts < self.min_cluster_size]
        for sc in small_clusters:
            labels[labels == sc] = -1
        
        # Renumber clusters
        labels = self._renumber_clusters(labels)
        
        # Calculate and store metrics
        self.metrics = self._calculate_metrics(embeddings, labels, distance_matrix, sim_stats)
        
        # Log results
        self.logger.info(f"Optimal threshold: {self.optimal_threshold}")
        self.logger.info(f"Final clusters: {self.metrics.n_clusters}, noise: {self.metrics.n_noise}")
        if self.metrics.silhouette_score:
            self.logger.info(f"Silhouette score: {self.metrics.silhouette_score:.4f}")
        
        return labels
    
    def _cluster_by_method(self, 
                          embeddings: np.ndarray, 
                          distance_matrix: np.ndarray,
                          sim_stats: Dict[str, float],
                          target_clusters: int) -> np.ndarray:
        """Perform clustering based on selected method."""
        
        if self.method == 'silhouette':
            self.optimal_threshold, self.search_history = self.find_optimal_threshold_silhouette(
                embeddings, sim_stats
            )
            return self._agglomerative_cluster(distance_matrix)
            
        elif self.method == 'elbow':
            optimal_k, self.search_history = self.find_optimal_k_elbow(embeddings)
            self.optimal_threshold = None
            model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            return model.fit_predict(embeddings)
            
        elif self.method == 'target':
            self.optimal_threshold, self.search_history = self.find_threshold_by_target(
                embeddings, sim_stats, target_clusters
            )
            return self._agglomerative_cluster(distance_matrix)
            
        elif self.method == 'percentile':
            if sim_stats['homogeneity'] == 'very_high':
                self.optimal_threshold = sim_stats['p90']
            elif sim_stats['homogeneity'] == 'high':
                self.optimal_threshold = sim_stats['p75']
            else:
                self.optimal_threshold = sim_stats['p50']
            self.search_history = [{'method': 'percentile', 'threshold': self.optimal_threshold}]
            return self._agglomerative_cluster(distance_matrix)
        
        elif self.method == 'manual':
            if self.manual_threshold is None:
                raise ValueError("Manual threshold must be provided for method='manual'")
            
            self.optimal_threshold = self.manual_threshold
            self.logger.info(f"Using manual threshold: {self.optimal_threshold}")
            
            if sim_stats['homogeneity'] == 'very_high' and self.optimal_threshold < sim_stats['p25']:
                self.logger.warning(
                    f"Manual threshold {self.optimal_threshold:.4f} is below p25 "
                    f"({sim_stats['p25']:.4f}) for very homogeneous data. "
                    f"This may result in too few clusters."
                )
            
            self.search_history = [{
                'method': 'manual', 
                'threshold': self.optimal_threshold,
                'user_specified': True,
                'similarity_stats': sim_stats
            }]
            return self._agglomerative_cluster(distance_matrix)
        
        elif self.method == 'leakage_aware':
            if self._target_labels is None:
                self.logger.warning(
                    "No target labels set for leakage_aware method. "
                    "Using synthetic labels based on embedding clustering."
                )
                labels = np.zeros(len(embeddings), dtype=int)
            else:
                labels = self._target_labels
            
            self.optimal_threshold, self.search_history, self.split_quality_metrics = \
                self.find_threshold_leakage_aware(embeddings, labels, sim_stats)
            return self._agglomerative_cluster(distance_matrix)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _agglomerative_cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Perform agglomerative clustering with current threshold."""
        distance_thresh = 1 - self.optimal_threshold
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_thresh,
            metric='precomputed',
            linkage='average'
        )
        return model.fit_predict(distance_matrix)
    
    def _renumber_clusters(self, labels: np.ndarray) -> np.ndarray:
        """Renumber clusters to be consecutive starting from 0, keeping -1 as noise."""
        new_labels = labels.copy()
        unique_labels = np.unique(labels[labels != -1])
        
        for new_id, old_id in enumerate(sorted(unique_labels)):
            new_labels[labels == old_id] = new_id
        
        return new_labels
    
    def _calculate_metrics(self,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          distance_matrix: np.ndarray,
                          sim_stats: Dict[str, float]) -> ClusteringMetrics:
        """Calculate clustering metrics."""
        valid_mask = labels != -1
        n_clusters = len(np.unique(labels[valid_mask]))
        n_noise = np.sum(~valid_mask)
        
        # Cluster sizes
        unique, counts = np.unique(labels[valid_mask], return_counts=True)
        cluster_sizes = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
        
        # Scoring metrics
        sil_score = None
        ch_score = None
        db_score = None
        
        if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
            try:
                sil_score = float(silhouette_score(
                    distance_matrix[valid_mask][:, valid_mask],
                    labels[valid_mask],
                    metric='precomputed'
                ))
            except:
                pass
            
            try:
                ch_score = float(calinski_harabasz_score(
                    embeddings[valid_mask],
                    labels[valid_mask]
                ))
            except:
                pass
            
            try:
                db_score = float(davies_bouldin_score(
                    embeddings[valid_mask],
                    labels[valid_mask]
                ))
            except:
                pass
        
        return ClusteringMetrics(
            n_clusters=n_clusters,
            n_samples=len(labels),
            n_noise=n_noise,
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            cluster_sizes=cluster_sizes,
            threshold_used=self.optimal_threshold or 0.0,
            method=self.method,
            similarity_stats=sim_stats,
            threshold_search_history=self.search_history,
            split_quality_metrics=self.split_quality_metrics
        )
    
    def save_metrics(self, output_dir: str, prefix: str = "clustering") -> str:
        """
        Save clustering metrics to JSON.
        
        Args:
            output_dir: Output directory
            prefix: File prefix
            
        Returns:
            Path to saved file
        """
        if self.metrics is None:
            raise ValueError("No metrics available. Run cluster() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics_file = output_path / f"{prefix}_metrics.json"
        self.metrics.save_json(str(metrics_file))
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        return str(metrics_file)


class AdaptiveClusteringStrategy:
    """
    ClusteringStrategy wrapper for AdaptiveClustering.
    
    Allows AdaptiveClustering to be used as a drop-in replacement
    in the existing stratification pipeline.
    """
    
    def __init__(self,
                 method: str = 'target',
                 min_clusters: int = 5,
                 max_clusters: int = 100,
                 min_cluster_size: int = 3,
                 target_cluster_ratio: float = 0.01,
                 output_dir: Optional[str] = None):
        """
        Initialize adaptive clustering strategy.
        
        Args:
            method: Optimization method ('silhouette', 'elbow', 'target', 'percentile')
            min_clusters: Minimum clusters
            max_clusters: Maximum clusters
            min_cluster_size: Minimum points per cluster
            target_cluster_ratio: Target clusters as ratio of samples
            output_dir: Directory to save metrics JSON
        """
        self.adaptive = AdaptiveClustering(
            method=method,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
            target_cluster_ratio=target_cluster_ratio
        )
        self.output_dir = output_dir
        self._embeddings = None
    
    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """Store embeddings for later use in cluster()."""
        self._embeddings = embeddings
    
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Apply adaptive clustering.
        
        Note: This method needs embeddings to be set via set_embeddings()
        because adaptive clustering needs the original embeddings for
        some methods (like k-means).
        """
        if self._embeddings is None:
            raise ValueError("Embeddings must be set via set_embeddings() before clustering")
        
        labels = self.adaptive.cluster(self._embeddings, distance_matrix)
        
        # Save metrics if output_dir is specified
        if self.output_dir:
            self.adaptive.save_metrics(self.output_dir, "stratification_clustering")
        
        return labels
    
    def get_metrics(self) -> Optional[ClusteringMetrics]:
        """Get clustering metrics."""
        return self.adaptive.metrics
