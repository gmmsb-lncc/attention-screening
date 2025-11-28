"""
Simplified Stratifier using SOLID principles and KISS.

Single Responsibility Principle: Each class has one job
- Stratifier: Coordinates the stratification process
- EmbeddingClusterer: Clusters embeddings
- ClusterSplitter: Splits clusters into train/val/test

Open/Closed Principle: Extensible through ClusteringStrategy
Liskov Substitution: All strategies implement same interface
Interface Segregation: Small, focused interfaces
Dependency Inversion: Depends on abstractions, not concrete classes
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from src.build.core.base_builder import BaseBuilder
from src.build.core.config import BuildConfig
from src.build.core.exceptions import BuildException
from src.build.core.constants import (
    STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM,
    STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD,
    STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE
)

from .clustering import (
    ClusteringStrategy, 
    DBSCANClustering, 
    HierarchicalClustering, 
    KMeansClustering,
    RandomClustering,
    EmbeddingClusterer
)
from .cluster_splitter import ClusterSplitter
from .adaptive_clustering import AdaptiveClustering, AdaptiveClusteringStrategy


class Stratifier(BaseBuilder):
    """
    Simplified stratifier for molecular embeddings.
    
    Coordinates clustering and splitting following SOLID principles.
    """
    
    def __init__(self, 
                 config: Optional[BuildConfig] = None,
                 clustering_algorithm: str = 'adaptive',
                 similarity_threshold: Optional[float] = None,
                 cluster_min_size: int = 3,
                 adaptive_method: str = 'target',
                 target_cluster_ratio: float = 0.01,
                 output_dir: Optional[str] = None,
                 test_size: float = 0.1,
                 val_size: float = 0.1,
                 **kwargs):
        """
        Initialize stratifier.
        
        Args:
            config: Build configuration
            clustering_algorithm: Algorithm ('adaptive', 'dbscan', 'hierarchical', 'kmeans', 'random')
            similarity_threshold: Manual threshold for clustering. If None, uses automatic detection.
                                 For non-adaptive methods, defaults to 0.7.
                                 For adaptive methods, if provided, uses 'manual' mode.
            cluster_min_size: Minimum cluster size
            adaptive_method: Method for adaptive clustering ('silhouette', 'elbow', 'target', 'percentile', 'manual', 'leakage_aware')
            target_cluster_ratio: Target clusters as ratio of samples (for adaptive 'target' method)
            output_dir: Directory to save clustering metrics JSON
            test_size: Proportion for test set (for 'leakage_aware' method)
            val_size: Proportion for validation set (for 'leakage_aware' method)
        """
        self.clustering_algorithm = clustering_algorithm or STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM
        
        # Handle manual threshold - if provided with adaptive algorithm, use manual mode
        self.manual_threshold = similarity_threshold  # Store the user-provided threshold
        
        # Auto threshold is enabled if no manual threshold is provided for adaptive clustering
        self.auto_threshold = (self.clustering_algorithm == 'adaptive' and similarity_threshold is None)
        
        if self.clustering_algorithm == 'adaptive' and similarity_threshold is not None:
            self.adaptive_method = 'manual'  # Override to manual mode
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = similarity_threshold or STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD
            self.adaptive_method = adaptive_method
        
        self.cluster_min_size = cluster_min_size or STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE
        self.target_cluster_ratio = target_cluster_ratio
        self.output_dir = output_dir
        self.test_size = test_size
        self.val_size = val_size
        
        # Multi-view weights
        self.protein_weight = 0.6
        self.ligand_weight = 0.4
        
        # Components (Dependency Injection)
        self.clustering_strategy = None  # Created later when we know output_dir
        self.adaptive_clustering = None  # For adaptive method
        self.clusterer = None
        self.splitter = None
        self.cluster_labels = None
        self.clustering_metrics = None  # Store metrics for JSON export
        
        super().__init__(config, **kwargs)
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        valid_algorithms = ['adaptive', 'dbscan', 'hierarchical', 'kmeans', 'random']
        if self.clustering_algorithm not in valid_algorithms:
            raise BuildException(f"Algorithm must be one of {valid_algorithms}")
        
        valid_adaptive_methods = ['silhouette', 'elbow', 'target', 'percentile', 'manual', 'leakage_aware']
        if self.adaptive_method not in valid_adaptive_methods:
            raise BuildException(f"Adaptive method must be one of {valid_adaptive_methods}")
        
        # For adaptive clustering with manual mode, threshold is required
        if self.clustering_algorithm == 'adaptive' and self.adaptive_method == 'manual':
            if self.manual_threshold is None or self.manual_threshold <= 0:
                raise BuildException("Manual mode requires a valid threshold (0.0-1.0)")
        
        # Validate threshold range
        if self.similarity_threshold is not None and not 0 <= self.similarity_threshold <= 1:
            raise BuildException("Similarity threshold must be between 0 and 1")
        
        if self.cluster_min_size < 1:
            raise BuildException("Cluster minimum size must be positive")
        
        if not 0 < self.target_cluster_ratio < 1:
            raise BuildException("Target cluster ratio must be between 0 and 1")
    
    def _create_clustering_strategy(self, output_dir: Optional[str] = None) -> ClusteringStrategy:
        """
        Create clustering strategy (Factory Pattern).
        
        Open/Closed Principle: Easy to add new strategies without modifying existing code.
        
        Args:
            output_dir: Directory to save metrics (for adaptive clustering)
        """
        distance_threshold = 1 - self.similarity_threshold
        
        if self.clustering_algorithm == 'adaptive':
            # Determine method based on auto_threshold setting
            method = 'manual' if not self.auto_threshold else self.adaptive_method
            manual_threshold = self.similarity_threshold if not self.auto_threshold else None
            
            # Create adaptive clustering strategy
            self.adaptive_clustering = AdaptiveClustering(
                method=method,
                min_clusters=5,
                max_clusters=100,
                min_cluster_size=self.cluster_min_size,
                target_cluster_ratio=self.target_cluster_ratio,
                auto_threshold=self.auto_threshold,
                manual_threshold=manual_threshold,
                test_size=self.test_size,
                val_size=self.val_size,
                logger=self.logger
            )
            return AdaptiveClusteringStrategy(
                method=method,
                min_clusters=5,
                max_clusters=100,
                min_cluster_size=self.cluster_min_size,
                target_cluster_ratio=self.target_cluster_ratio,
                output_dir=output_dir
            )
        
        strategies = {
            'dbscan': DBSCANClustering(eps=distance_threshold, min_samples=self.cluster_min_size),
            'hierarchical': HierarchicalClustering(distance_threshold=distance_threshold),
            'kmeans': KMeansClustering(n_clusters=20),
            'random': RandomClustering(n_clusters=20)
        }
        
        return strategies.get(self.clustering_algorithm, strategies['hierarchical'])
    
    def stratified_split(self, 
                        embeddings: np.ndarray, 
                        labels: np.ndarray,
                        test_size: float = 0.1,
                        val_size: float = 0.1,
                        output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split using clustering.
        
        Args:
            embeddings: Embedding matrix
            labels: Target labels
            test_size: Test set proportion
            val_size: Validation set proportion
            output_dir: Directory to save clustering metrics JSON
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Use output_dir from parameter or instance
        metrics_dir = output_dir or self.output_dir
        
        # Create clustering strategy with output directory
        self.clustering_strategy = self._create_clustering_strategy(metrics_dir)
        
        # For adaptive clustering, we need to use the AdaptiveClustering directly
        if self.clustering_algorithm == 'adaptive':
            self.logger.info(f"Adaptive clustering with method='{self.adaptive_method}'")
            
            # For leakage_aware method, pass target labels
            if self.adaptive_method == 'leakage_aware':
                self.adaptive_clustering.set_target_labels(labels)
            
            self.cluster_labels = self.adaptive_clustering.cluster(embeddings)
            self.clustering_metrics = self.adaptive_clustering.metrics
            
            # Save metrics if output directory specified
            if metrics_dir:
                self.adaptive_clustering.save_metrics(metrics_dir, "stratification_clustering")
        else:
            # Initialize components for standard clustering
            self.clusterer = EmbeddingClusterer(self.clustering_strategy, self.logger)
            self.logger.info(f"Clustering with {self.clustering_algorithm}")
            self.cluster_labels = self.clusterer.cluster_single(embeddings)
        
        # Initialize splitter and split
        self.splitter = ClusterSplitter(test_size=test_size, val_size=val_size)
        train_idx, val_idx, test_idx = self.splitter.split(self.cluster_labels, labels)
        
        # Save split info
        if metrics_dir:
            self._save_split_info(metrics_dir, train_idx, val_idx, test_idx, labels)
        
        # Log results
        self.logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        return train_idx, val_idx, test_idx
    
    def multi_view_stratified_split(self,
                                   protein_embeddings: np.ndarray,
                                   ligand_embeddings: np.ndarray,
                                   labels: np.ndarray,
                                   test_size: float = 0.1,
                                   val_size: float = 0.1,
                                   protein_weight: float = 0.6,
                                   ligand_weight: float = 0.4,
                                   output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform multi-view stratified split.
        
        Args:
            protein_embeddings: Protein embeddings
            ligand_embeddings: Ligand embeddings
            labels: Target labels
            test_size: Test set proportion
            val_size: Validation set proportion
            protein_weight: Protein similarity weight
            ligand_weight: Ligand similarity weight
            output_dir: Directory to save clustering metrics JSON
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Update weights
        self.protein_weight = protein_weight
        self.ligand_weight = ligand_weight
        
        # Use output_dir from parameter or instance
        metrics_dir = output_dir or self.output_dir
        
        # Create clustering strategy with output directory
        self.clustering_strategy = self._create_clustering_strategy(metrics_dir)
        
        # For adaptive clustering with multi-view, concatenate embeddings
        if self.clustering_algorithm == 'adaptive':
            self.logger.info(f"Adaptive multi-view clustering with method='{self.adaptive_method}'")
            self.logger.info(f"  Protein weight: {protein_weight}")
            self.logger.info(f"  Ligand weight: {ligand_weight}")
            
            # Combine embeddings with weights (normalize first)
            from sklearn.preprocessing import normalize
            protein_norm = normalize(protein_embeddings)
            ligand_norm = normalize(ligand_embeddings)
            combined = np.hstack([
                protein_norm * np.sqrt(protein_weight),
                ligand_norm * np.sqrt(ligand_weight)
            ])
            
            # For leakage_aware method, pass target labels
            if self.adaptive_method == 'leakage_aware':
                self.adaptive_clustering.set_target_labels(labels)
            
            self.cluster_labels = self.adaptive_clustering.cluster(combined)
            self.clustering_metrics = self.adaptive_clustering.metrics
            
            # Save metrics if output directory specified
            if metrics_dir:
                self.adaptive_clustering.save_metrics(metrics_dir, "stratification_clustering")
        else:
            # Initialize components for standard clustering
            self.clusterer = EmbeddingClusterer(self.clustering_strategy, self.logger)
            
            # Cluster using multi-view
            self.logger.info(f"Multi-view clustering with {self.clustering_algorithm}")
            self.logger.info(f"  Protein weight: {protein_weight}")
            self.logger.info(f"  Ligand weight: {ligand_weight}")
            
            self.cluster_labels = self.clusterer.cluster_multi_view(
                protein_embeddings,
                ligand_embeddings,
                protein_weight=protein_weight,
                ligand_weight=ligand_weight
            )
        
        # Split clusters
        self.splitter = ClusterSplitter(test_size=test_size, val_size=val_size)
        train_idx, val_idx, test_idx = self.splitter.split(self.cluster_labels, labels)
        
        # Save split info
        if metrics_dir:
            self._save_split_info(metrics_dir, train_idx, val_idx, test_idx, labels)
        
        # Log results
        self.logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        return train_idx, val_idx, test_idx
    
    def _save_split_info(self, 
                        output_dir: str, 
                        train_idx: np.ndarray, 
                        val_idx: np.ndarray, 
                        test_idx: np.ndarray,
                        labels: np.ndarray) -> None:
        """
        Save split information to JSON for visualization.
        
        Args:
            output_dir: Output directory
            train_idx: Training indices
            val_idx: Validation indices
            test_idx: Test indices
            labels: Original labels
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate label distribution in each split
        def get_label_stats(indices, all_labels):
            split_labels = all_labels[indices]
            unique, counts = np.unique(split_labels, return_counts=True)
            return {
                'n_samples': len(indices),
                'label_distribution': {str(k): int(v) for k, v in zip(unique, counts)},
                'mean_label': float(np.mean(split_labels)),
                'std_label': float(np.std(split_labels))
            }
        
        split_info = {
            'train': get_label_stats(train_idx, labels),
            'val': get_label_stats(val_idx, labels),
            'test': get_label_stats(test_idx, labels),
            'total_samples': len(labels),
            'clustering_algorithm': self.clustering_algorithm,
            'adaptive_method': self.adaptive_method if self.clustering_algorithm == 'adaptive' else None,
            'n_clusters': int(len(np.unique(self.cluster_labels[self.cluster_labels != -1]))),
            'n_noise': int(np.sum(self.cluster_labels == -1)) if -1 in self.cluster_labels else 0
        }
        
        # Add clustering metrics if available
        if self.clustering_metrics is not None:
            split_info['clustering_metrics'] = {
                'silhouette_score': self.clustering_metrics.silhouette_score,
                'calinski_harabasz_score': self.clustering_metrics.calinski_harabasz_score,
                'davies_bouldin_score': self.clustering_metrics.davies_bouldin_score,
                'threshold_used': self.clustering_metrics.threshold_used
            }
        
        split_info_path = output_path / "stratification_split_info.json"
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"Split info saved to {split_info_path}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get clustering information.
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.cluster_labels is None:
            return {}
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        n_noise = np.sum(self.cluster_labels == -1) if -1 in unique else 0
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_sizes': dict(zip(unique, counts)),
            'algorithm': self.clustering_algorithm,
            'similarity_threshold': self.similarity_threshold
        }
    
    def build(self) -> dict:
        """Build method for BaseBuilder compatibility."""
        return {
            'clustering_algorithm': self.clustering_algorithm,
            'similarity_threshold': self.similarity_threshold,
            'cluster_min_size': self.cluster_min_size,
            'protein_weight': self.protein_weight,
            'ligand_weight': self.ligand_weight,
            'cluster_info': self.get_cluster_info()
        }
