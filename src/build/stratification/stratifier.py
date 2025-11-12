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

import numpy as np
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


class Stratifier(BaseBuilder):
    """
    Simplified stratifier for molecular embeddings.
    
    Coordinates clustering and splitting following SOLID principles.
    """
    
    def __init__(self, 
                 config: Optional[BuildConfig] = None,
                 clustering_algorithm: str = 'hierarchical',
                 similarity_threshold: float = 0.7,
                 cluster_min_size: int = 3,
                 **kwargs):
        """
        Initialize stratifier.
        
        Args:
            config: Build configuration
            clustering_algorithm: Algorithm ('dbscan', 'hierarchical', 'kmeans', 'random')
            similarity_threshold: Threshold for clustering
            cluster_min_size: Minimum cluster size
        """
        self.clustering_algorithm = clustering_algorithm or STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM
        self.similarity_threshold = similarity_threshold or STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD
        self.cluster_min_size = cluster_min_size or STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE
        
        # Multi-view weights
        self.protein_weight = 0.6
        self.ligand_weight = 0.4
        
        # Components (Dependency Injection)
        self.clustering_strategy = self._create_clustering_strategy()
        self.clusterer = None
        self.splitter = None
        self.cluster_labels = None
        
        super().__init__(config, **kwargs)
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        valid_algorithms = ['dbscan', 'hierarchical', 'kmeans', 'random']
        if self.clustering_algorithm not in valid_algorithms:
            raise BuildException(f"Algorithm must be one of {valid_algorithms}")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise BuildException("Similarity threshold must be between 0 and 1")
        
        if self.cluster_min_size < 1:
            raise BuildException("Cluster minimum size must be positive")
    
    def _create_clustering_strategy(self) -> ClusteringStrategy:
        """
        Create clustering strategy (Factory Pattern).
        
        Open/Closed Principle: Easy to add new strategies without modifying existing code.
        """
        distance_threshold = 1 - self.similarity_threshold
        
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
                        test_size: float = 0.2,
                        val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split using clustering.
        
        Args:
            embeddings: Embedding matrix
            labels: Target labels
            test_size: Test set proportion
            val_size: Validation set proportion
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Initialize components
        self.clusterer = EmbeddingClusterer(self.clustering_strategy, self.logger)
        self.splitter = ClusterSplitter(test_size=test_size, val_size=val_size)
        
        # Cluster embeddings
        self.logger.info(f"Clustering with {self.clustering_algorithm}")
        self.cluster_labels = self.clusterer.cluster_single(embeddings)
        
        # Split clusters
        train_idx, val_idx, test_idx = self.splitter.split(self.cluster_labels, labels)
        
        # Log results
        self.logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        return train_idx, val_idx, test_idx
    
    def multi_view_stratified_split(self,
                                   protein_embeddings: np.ndarray,
                                   ligand_embeddings: np.ndarray,
                                   labels: np.ndarray,
                                   test_size: float = 0.2,
                                   val_size: float = 0.1,
                                   protein_weight: float = 0.6,
                                   ligand_weight: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Update weights
        self.protein_weight = protein_weight
        self.ligand_weight = ligand_weight
        
        # Initialize components
        self.clusterer = EmbeddingClusterer(self.clustering_strategy, self.logger)
        self.splitter = ClusterSplitter(test_size=test_size, val_size=val_size)
        
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
        train_idx, val_idx, test_idx = self.splitter.split(self.cluster_labels, labels)
        
        # Log results
        self.logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        return train_idx, val_idx, test_idx
    
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
