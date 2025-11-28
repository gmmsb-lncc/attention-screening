"""
Stratifier module for cosine similarity-based train/test/validation splits.

This module implements multiple clustering algorithms to group similar
molecular embeddings and ensures balanced splits across these groups.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path
import logging
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings

from src.build.core.base_builder import BaseBuilder
from src.build.core.config import BuildConfig
from src.build.core.exceptions import BuildException
from src.build.core.constants import (
    STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM,
    STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD,
    STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE,
    STRATIFICATION_DEFAULT_STRATIFY_BY
)
from .cosine_similarity_calculator import CosineSimilarityCalculator


class Stratifier(BaseBuilder):
    """
    Cosine similarity-based stratifier for molecular embeddings.
    
    Groups similar embeddings using clustering algorithms and ensures
    balanced distribution across train/test/validation splits.
    """
    
    def __init__(self, 
                 config: Optional[BuildConfig] = None,
                 clustering_algorithm: Optional[str] = None,
                 similarity_threshold: Optional[float] = None,
                 cluster_min_size: Optional[int] = None,
                 stratify_by: Optional[str] = None,
                 **kwargs):
        """
        Initialize stratifier.

        Args:
            config: Build configuration
            clustering_algorithm: Algorithm to use ('dbscan', 'hierarchical', 'kmeans', 'random')
            similarity_threshold: Threshold for similarity-based grouping
            cluster_min_size: Minimum cluster size for stratification
            stratify_by: Which embeddings to stratify by ('ligand', 'protein', 'both', 'combined')
            **kwargs: Additional configuration options
        """
        # Set attributes before calling parent constructor
        # Use config values with defaults from constants if not provided explicitly
        self.clustering_algorithm = clustering_algorithm or (
            config.get('stratification_params', {}).get('clustering_algorithm') 
            if config else None
        ) or STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM
        
        self.similarity_threshold = similarity_threshold or (
            config.get('stratification_params', {}).get('similarity_threshold') 
            if config else None
        ) or STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD
        
        self.cluster_min_size = cluster_min_size or (
            config.get('stratification_params', {}).get('cluster_min_size') 
            if config else None
        ) or STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE
        
        self.stratify_by = stratify_by or (
            config.get('stratification_params', {}).get('stratify_by') 
            if config else None
        ) or STRATIFICATION_DEFAULT_STRATIFY_BY
        
        # Get weights from config or defaults
        if config:
            strat_params = config.get('stratification_params', {})
            self.protein_weight = strat_params.get('protein_weight', 0.6)
            self.ligand_weight = strat_params.get('ligand_weight', 0.4)
        else:
            self.protein_weight = 0.6
            self.ligand_weight = 0.4
            
        self.cosine_calculator = None
        self.clustering_model = None
        self.cluster_labels = None
        
        super().__init__(config, **kwargs)
        
    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.clustering_algorithm not in ['dbscan', 'hierarchical', 'kmeans', 'random']:
            raise BuildException(
                f"Clustering algorithm must be one of ['dbscan', 'hierarchical', 'kmeans', 'random'], "
                f"got {self.clustering_algorithm}"
            )
        
        if not 0 <= self.similarity_threshold <= 1:
            raise BuildException("Similarity threshold must be between 0 and 1")
        
        if self.cluster_min_size < 1:
            raise BuildException("Cluster minimum size must be positive")
        
        if self.stratify_by not in ['ligand', 'protein', 'both', 'combined']:
            raise BuildException(
                f"Stratify by must be one of ['ligand', 'protein', 'both', 'combined'], "
                f"got {self.stratify_by}"
            )
        
        if not 0 <= self.protein_weight <= 1 or not 0 <= self.ligand_weight <= 1:
            raise BuildException("Protein and ligand weights must be between 0 and 1")
        
    def _get_clustering_model(self) -> Any:
        """Get clustering model based on configuration."""
        if self.clustering_algorithm == 'dbscan':
            # Use eps based on similarity threshold (1 - threshold for distance)
            eps = 1 - self.similarity_threshold
            return DBSCAN(eps=eps, min_samples=self.cluster_min_size, metric='precomputed')
        elif self.clustering_algorithm == 'hierarchical':
            return AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=1 - self.similarity_threshold,
                linkage='ward' if self.cluster_min_size > 2 else 'average'
            )
        elif self.clustering_algorithm == 'kmeans':
            # We'll determine n_clusters dynamically based on data
            return KMeans(n_clusters=20, random_state=42)  # Placeholder, will be set later
        else:  # random fallback
            return None
    
    def _preprocess_embeddings(self, 
                              protein_embeddings: Optional[np.ndarray] = None,
                              ligand_embeddings: Optional[np.ndarray] = None,
                              combined_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Preprocess embeddings based on stratification strategy.

        Args:
            protein_embeddings: Protein embedding matrix
            ligand_embeddings: Ligand embedding matrix
            combined_embeddings: Combined embedding matrix (protein + ligand)

        Returns:
            Processed embedding matrix for clustering
        """
        if combined_embeddings is not None:
            return combined_embeddings
        
        if protein_embeddings is not None and ligand_embeddings is not None:
            if self.stratify_by == 'both' or self.stratify_by == 'combined':
                # Use multi-view similarity
                self.cosine_calculator = CosineSimilarityCalculator(config=self.config)
                combined_sim = self.cosine_calculator.calculate_multi_view_similarity(
                    protein_embeddings, 
                    ligand_embeddings,
                    protein_weight=self.protein_weight,
                    ligand_weight=self.ligand_weight
                )
                return combined_sim
            elif self.stratify_by == 'protein':
                return protein_embeddings
            elif self.stratify_by == 'ligand':
                return ligand_embeddings
        elif protein_embeddings is not None:
            return protein_embeddings
        elif ligand_embeddings is not None:
            return ligand_embeddings
        
        raise BuildException("At least one embedding type must be provided")
    
    def cluster_embeddings(self, 
                         protein_embeddings: Optional[np.ndarray] = None,
                         ligand_embeddings: Optional[np.ndarray] = None,
                         combined_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cluster embeddings based on similarity.

        Args:
            protein_embeddings: Protein embedding matrix
            ligand_embeddings: Ligand embedding matrix
            combined_embeddings: Combined embedding matrix (protein + ligand)

        Returns:
            Cluster labels for each sample
        """
        self.logger.info(f"Clustering embeddings using {self.clustering_algorithm}")
        
        # Preprocess embeddings
        processed_embeddings = self._preprocess_embeddings(
            protein_embeddings, ligand_embeddings, combined_embeddings
        )
        
        if self.clustering_algorithm == 'random':
            # Return random clusters for fallback
            n_samples = processed_embeddings.shape[0]
            n_clusters = max(2, n_samples // 20)  # Create roughly 5% of samples per cluster
            cluster_labels = np.random.choice(n_clusters, size=n_samples)
            self.cluster_labels = cluster_labels
            return cluster_labels
        
        # Calculate similarity matrix if needed
        if self.stratify_by in ['both', 'combined'] and self.cosine_calculator is not None:
            # Use precomputed similarity matrix for clustering
            similarity_matrix = processed_embeddings
            # Ensure distance is non-negative: cosine distance = 1 - cosine similarity
            # Cosine similarity ranges from -1 to 1, so distance ranges from 0 to 2
            distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
            
            # Set clustering model with precomputed metric
            if self.clustering_algorithm == 'dbscan':
                self.clustering_model = DBSCAN(
                    eps=1 - self.similarity_threshold,
                    min_samples=self.cluster_min_size,
                    metric='precomputed',
                    # For precomputed metrics, use different algorithm to avoid sklearn issue
                    algorithm='brute'  # Use brute force for precomputed distances
                )
            elif self.clustering_algorithm == 'hierarchical':
                # Convert distance threshold to linkage
                self.clustering_model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - self.similarity_threshold,
                    linkage='average'
                )
            
            cluster_labels = self.clustering_model.fit_predict(distance_matrix)
        else:
            # Calculate similarity matrix using CosineSimilarityCalculator
            calc = CosineSimilarityCalculator(config=self.config)
            similarity_matrix = calc.calculate_batch(processed_embeddings)
            # Ensure distance is non-negative
            distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
            
            # Determine number of clusters for kmeans based on similarity
            if self.clustering_algorithm == 'kmeans':
                # For k-means, we need to determine a reasonable number of clusters
                n_samples = processed_embeddings.shape[0]
                # Estimate number of clusters based on similarity threshold
                n_clusters = max(2, min(n_samples // self.cluster_min_size, 50))
                self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                # Set clustering model with precomputed metric
                if self.clustering_algorithm == 'dbscan':
                    self.clustering_model = DBSCAN(
                        eps=1 - self.similarity_threshold,
                        min_samples=self.cluster_min_size,
                        metric='precomputed',
                        algorithm='brute'  # Use brute force for precomputed distances
                    )
                elif self.clustering_algorithm == 'hierarchical':
                    self.clustering_model = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=1 - self.similarity_threshold,
                        linkage='average'
                    )
            
            cluster_labels = self.clustering_model.fit_predict(distance_matrix)
        
        # Validate cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        valid_clusters = counts >= self.cluster_min_size
        n_valid_clusters = np.sum(valid_clusters)
        
        if n_valid_clusters == 0:
            self.logger.warning(
                f"No clusters meet minimum size requirement of {self.cluster_min_size}. "
                f"Allowing smaller clusters for stratification."
            )
        
        self.cluster_labels = cluster_labels
        return cluster_labels
    
    def _balance_clusters_for_split(self, 
                                   cluster_labels: np.ndarray, 
                                   labels: np.ndarray,
                                   test_size: float = 0.1,
                                   val_size: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
        """
        Create balanced splits ensuring each cluster is properly distributed.

        Args:
            cluster_labels: Cluster labels for each sample
            labels: Target labels for stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Group indices by cluster
        cluster_to_indices = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in cluster_to_indices:
                cluster_to_indices[cluster_id] = []
            cluster_to_indices[cluster_id].append(idx)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # For each cluster, distribute samples among splits
        for cluster_id, indices in cluster_to_indices.items():
            cluster_size = len(indices)
            
            if cluster_size < 3:
                # If cluster is too small, randomly assign to splits
                if cluster_size == 1:
                    r = np.random.random()
                    if r < test_size:
                        test_indices.extend(indices)
                    elif r < test_size + val_size:
                        val_indices.extend(indices)
                    else:
                        train_indices.extend(indices)
                elif cluster_size == 2:
                    # Distribute 2 samples between splits
                    # Always give at least one to train
                    train_indices.append(indices[0])
                    
                    # Second sample: choose between val/test based on proportions
                    r = np.random.random()
                    if r < test_size/(test_size + val_size):
                        test_indices.append(indices[1])
                    else:
                        val_indices.append(indices[1])
            else:
                # Use stratified split if we have enough samples
                cluster_labels_subset = labels[indices]
                
                # Use sklearn's stratify when possible
                try:
                    # Encode labels for stratification if they're strings or not integers
                    if cluster_labels_subset.dtype.kind in ['U', 'S', 'O']:  # string or object
                        le = LabelEncoder()
                        encoded_labels = le.fit_transform(cluster_labels_subset)
                    else:
                        encoded_labels = cluster_labels_subset
                    
                    # First split: separate test set
                    if test_size > 0:
                        indices_train_val, indices_test = train_test_split(
                            indices,
                            test_size=test_size,
                            stratify=encoded_labels,
                            random_state=42
                        )
                        test_indices.extend(indices_test)
                    else:
                        indices_train_val = indices
                    
                    # Second split: separate validation from train
                    if val_size > 0 and len(indices_train_val) > 1:
                        val_proportion = val_size / (1 - test_size)
                        # Get labels for remaining samples (train+val)
                        # Map indices_train_val to positions in the original indices list
                        train_val_positions = [indices.index(i) for i in indices_train_val]
                        train_val_labels = cluster_labels_subset[train_val_positions]
                        # Re-encode if necessary
                        if train_val_labels.dtype.kind in ['U', 'S', 'O']:
                            le_tv = LabelEncoder()
                            train_val_labels = le_tv.fit_transform(train_val_labels)
                        
                        try:
                            indices_train, indices_val = train_test_split(
                                indices_train_val,
                                test_size=val_proportion,
                                stratify=train_val_labels,
                                random_state=42
                            )
                            val_indices.extend(indices_val)
                            train_indices.extend(indices_train)
                        except ValueError:
                            # If stratification fails, do random split
                            mid = int(len(indices_train_val) * (1 - val_proportion))
                            train_indices.extend(indices_train_val[:mid])
                            val_indices.extend(indices_train_val[mid:])
                    else:
                        train_indices.extend(indices_train_val)
                
                except ValueError:
                    # If stratification fails due to class imbalance, use random split
                    n_test = max(1, int(len(indices) * test_size))
                    n_val = max(1, int(len(indices) * val_size))
                    
                    np.random.shuffle(indices)
                    
                    if len(indices) < n_test + n_val + 1:
                        # Not enough samples, assign as evenly as possible
                        if len(indices) == 1:
                            train_indices.extend(indices)
                        elif len(indices) == 2:
                            train_indices.append(indices[0])
                            test_indices.append(indices[1])
                        else:  # len(indices) == 3
                            train_indices.append(indices[0])
                            val_indices.append(indices[1])
                            test_indices.append(indices[2])
                    else:
                        test_indices.extend(indices[:n_test])
                        val_indices.extend(indices[n_test:n_test + n_val])
                        train_indices.extend(indices[n_test + n_val:])
        
        return train_indices, val_indices, test_indices
    
    def stratified_split(self, 
                        embeddings: np.ndarray, 
                        labels: np.ndarray,
                        test_size: float = 0.1,
                        val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split using clustering of embeddings.

        Args:
            embeddings: Embedding matrix to cluster
            labels: Target labels for stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        cluster_labels = self.cluster_embeddings(combined_embeddings=embeddings)
        train_idx, val_idx, test_idx = self._balance_clusters_for_split(
            cluster_labels, labels, test_size, val_size
        )
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    
    def multi_view_stratified_split(self,
                                   protein_embeddings: np.ndarray,
                                   ligand_embeddings: np.ndarray,
                                   labels: np.ndarray,
                                   test_size: float = 0.1,
                                   val_size: float = 0.1,
                                   protein_weight: float = 0.6,
                                   ligand_weight: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split using both protein and ligand embeddings.

        Args:
            protein_embeddings: Protein embedding matrix
            ligand_embeddings: Ligand embedding matrix
            labels: Target labels for stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set
            protein_weight: Weight for protein similarity
            ligand_weight: Weight for ligand similarity

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        self.protein_weight = protein_weight
        self.ligand_weight = ligand_weight
        
        cluster_labels = self.cluster_embeddings(
            protein_embeddings=protein_embeddings,
            ligand_embeddings=ligand_embeddings
        )
        
        train_idx, val_idx, test_idx = self._balance_clusters_for_split(
            cluster_labels, labels, test_size, val_size
        )
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about the clustering performed.

        Returns:
            Dictionary with clustering information
        """
        if self.cluster_labels is None:
            return {}
        
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise points
        n_noise = np.sum(self.cluster_labels == -1) if -1 in self.cluster_labels else 0
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_sizes': dict(zip(unique_labels, counts)),
            'clustering_algorithm': self.clustering_algorithm,
            'similarity_threshold': self.similarity_threshold
        }

    def build(self) -> dict:
        """
        Build method for BaseBuilder compatibility.

        Returns:
            Dictionary with stratifier information
        """
        result = {
            'clustering_algorithm': self.clustering_algorithm,
            'similarity_threshold': self.similarity_threshold,
            'cluster_min_size': self.cluster_min_size,
            'stratify_by': self.stratify_by,
            'cluster_info': self.get_cluster_info()
        }
        
        return result