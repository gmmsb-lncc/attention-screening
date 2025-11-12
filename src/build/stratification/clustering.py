"""
Simple clustering coordinator following SOLID principles.
"""

import numpy as np
from typing import Optional
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import logging

from .cosine_similarity_calculator import CosineSimilarityCalculator


class ClusteringStrategy:
    """
    Base interface for clustering strategies (Open/Closed Principle).
    """
    
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Apply clustering algorithm to distance matrix."""
        raise NotImplementedError


class DBSCANClustering(ClusteringStrategy):
    """DBSCAN clustering implementation."""
    
    def __init__(self, eps: float = 0.3, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
    
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        return model.fit_predict(distance_matrix)


class HierarchicalClustering(ClusteringStrategy):
    """Hierarchical clustering implementation."""
    
    def __init__(self, distance_threshold: float = 0.3):
        self.distance_threshold = distance_threshold
    
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            linkage='average'
        )
        return model.fit_predict(distance_matrix)


class KMeansClustering(ClusteringStrategy):
    """K-Means clustering implementation."""
    
    def __init__(self, n_clusters: int = 20):
        self.n_clusters = n_clusters
    
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        # K-means needs embeddings, not distance matrix
        # We'll use a simple heuristic: use distance as negative similarity
        model = KMeans(n_clusters=self.n_clusters, random_state=42)
        # Convert distance to pseudo-embeddings using MDS-like approach
        # For simplicity, just use the distance matrix directly
        return model.fit_predict(-distance_matrix)  # Negative for similarity


class RandomClustering(ClusteringStrategy):
    """Random clustering for baseline."""
    
    def __init__(self, n_clusters: int = 20):
        self.n_clusters = n_clusters
    
    def cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        n_samples = distance_matrix.shape[0]
        return np.random.choice(self.n_clusters, size=n_samples)


class EmbeddingClusterer:
    """
    Clusters molecular embeddings using similarity.
    
    Single Responsibility: Handle embedding clustering.
    Dependency Inversion: Depends on ClusteringStrategy abstraction.
    """
    
    def __init__(self, strategy: ClusteringStrategy, logger: Optional[logging.Logger] = None):
        """
        Initialize clusterer.
        
        Args:
            strategy: Clustering algorithm to use
            logger: Logger instance
        """
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.similarity_calculator = CosineSimilarityCalculator()
    
    def cluster_single(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster single embedding matrix.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            
        Returns:
            Cluster labels (n_samples,)
        """
        # Calculate cosine similarity
        similarity = self.similarity_calculator.calculate_batch(embeddings)
        
        # Convert to distance (cosine distance = 1 - cosine similarity)
        distance = np.clip(1 - similarity, 0, 2)
        
        # Apply clustering
        return self.strategy.cluster(distance)
    
    def cluster_multi_view(self, 
                          protein_embeddings: np.ndarray, 
                          ligand_embeddings: np.ndarray,
                          protein_weight: float = 0.6,
                          ligand_weight: float = 0.4) -> np.ndarray:
        """
        Cluster using multi-view similarity.
        
        Args:
            protein_embeddings: Protein embedding matrix
            ligand_embeddings: Ligand embedding matrix
            protein_weight: Weight for protein similarity
            ligand_weight: Weight for ligand similarity
            
        Returns:
            Cluster labels (n_samples,)
        """
        # Calculate multi-view similarity
        similarity = self.similarity_calculator.calculate_multi_view_similarity(
            protein_embeddings,
            ligand_embeddings,
            protein_weight=protein_weight,
            ligand_weight=ligand_weight
        )
        
        # Convert to distance
        distance = np.clip(1 - similarity, 0, 2)
        
        # Apply clustering
        return self.strategy.cluster(distance)
