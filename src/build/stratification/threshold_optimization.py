"""
Threshold Optimization - Methods for finding optimal clustering thresholds.

This module implements various strategies for finding optimal clustering
parameters including silhouette optimization, elbow method, and target-based search.

Author: DockTKinase Team
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


class ThresholdOptimizer:
    """
    Optimizes clustering thresholds using various methods.
    
    Methods:
    - silhouette: Maximize silhouette score
    - elbow: Find elbow point for k-means
    - target: Binary search for target cluster count
    """
    
    def __init__(self,
                 min_clusters: int = 5,
                 max_clusters: int = 100,
                 min_cluster_size: int = 3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize threshold optimizer.
        
        Args:
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            min_cluster_size: Minimum points per cluster
            logger: Logger instance
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.logger = logger or logging.getLogger(__name__)
    
    def find_optimal_threshold_silhouette(self,
                                          embeddings: np.ndarray,
                                          sim_stats: Dict[str, float],
                                          n_candidates: int = 10) -> Tuple[float, List[Dict]]:
        """
        Find optimal threshold by maximizing silhouette score.
        
        Args:
            embeddings: Embedding matrix
            sim_stats: Similarity statistics
            n_candidates: Number of thresholds to try
            
        Returns:
            Tuple of (optimal_threshold, search_history)
        """
        # Define search range based on similarity distribution
        if sim_stats['homogeneity'] == 'very_high':
            min_thresh = sim_stats['p50']
            max_thresh = sim_stats['p99']
        elif sim_stats['homogeneity'] == 'high':
            min_thresh = sim_stats['p25']
            max_thresh = sim_stats['p95']
        else:
            min_thresh = 0.5
            max_thresh = 0.95
        
        thresholds = np.linspace(min_thresh, max_thresh, n_candidates)
        search_history = []
        best_score = -1
        best_threshold = thresholds[len(thresholds)//2]
        
        n_samples = embeddings.shape[0]
        
        # Precompute similarity matrix (sample for large datasets)
        if n_samples > 5000:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, 5000, replace=False)
            sample_emb = embeddings[sample_idx]
        else:
            sample_emb = embeddings
        
        sim_matrix = cosine_similarity(sample_emb)
        distance_matrix = np.clip(1 - sim_matrix, 0, 2)
        
        for thresh in thresholds:
            try:
                distance_thresh = 1 - thresh
                
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_thresh,
                    metric='precomputed',
                    linkage='average'
                )
                labels = model.fit_predict(distance_matrix)
                
                # Filter small clusters
                unique, counts = np.unique(labels, return_counts=True)
                small_clusters = unique[counts < self.min_cluster_size]
                for sc in small_clusters:
                    labels[labels == sc] = -1
                
                n_clusters = len(np.unique(labels[labels != -1]))
                n_noise = np.sum(labels == -1)
                
                # Calculate silhouette score
                valid_mask = labels != -1
                if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
                    score = silhouette_score(
                        distance_matrix[valid_mask][:, valid_mask],
                        labels[valid_mask],
                        metric='precomputed'
                    )
                else:
                    score = -1
                
                result = {
                    'threshold': float(thresh),
                    'n_clusters': n_clusters,
                    'n_noise': int(n_noise),
                    'silhouette_score': float(score) if score != -1 else None,
                    'valid': n_clusters >= self.min_clusters
                }
                search_history.append(result)
                
                self.logger.debug(f"Threshold {thresh:.4f}: {n_clusters} clusters, silhouette={score:.4f}")
                
                if result['valid'] and score > best_score:
                    best_score = score
                    best_threshold = thresh
                    
            except Exception as e:
                self.logger.warning(f"Failed for threshold {thresh}: {e}")
                search_history.append({
                    'threshold': float(thresh),
                    'error': str(e)
                })
        
        if best_score == -1:
            self.logger.warning("No valid threshold found, using percentile fallback")
            best_threshold = sim_stats['p75']
        
        return best_threshold, search_history
    
    def find_optimal_k_elbow(self,
                            embeddings: np.ndarray,
                            k_range: Optional[Tuple[int, int]] = None) -> Tuple[int, List[Dict]]:
        """
        Find optimal k using elbow method.
        
        Args:
            embeddings: Embedding matrix
            k_range: Range of k values to try
            
        Returns:
            Tuple of (optimal_k, search_history)
        """
        n_samples = embeddings.shape[0]
        
        if k_range is None:
            min_k = self.min_clusters
            max_k = min(self.max_clusters, n_samples // self.min_cluster_size)
            k_range = (min_k, max_k)
        
        k_values = np.linspace(k_range[0], k_range[1], min(20, k_range[1] - k_range[0] + 1), dtype=int)
        k_values = np.unique(k_values)
        
        search_history = []
        inertias = []
        
        for k in k_values:
            try:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                model.fit(embeddings)
                inertias.append(model.inertia_)
                
                labels = model.labels_
                sil_score = silhouette_score(embeddings, labels) if len(np.unique(labels)) >= 2 else None
                
                search_history.append({
                    'k': int(k),
                    'inertia': float(model.inertia_),
                    'silhouette_score': float(sil_score) if sil_score else None
                })
            except Exception as e:
                self.logger.warning(f"Failed for k={k}: {e}")
        
        # Find elbow using second derivative
        if len(inertias) >= 3:
            inertias = np.array(inertias)
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 1
            optimal_k = int(k_values[elbow_idx])
        else:
            optimal_k = int(k_values[len(k_values)//2])
        
        return optimal_k, search_history
    
    def find_threshold_by_target(self,
                                embeddings: np.ndarray,
                                sim_stats: Dict[str, float],
                                target_clusters: int) -> Tuple[float, List[Dict]]:
        """
        Find threshold that produces approximately target number of clusters.
        
        Uses binary search to find the threshold.
        
        Args:
            embeddings: Embedding matrix
            sim_stats: Similarity statistics
            target_clusters: Target number of clusters
            
        Returns:
            Tuple of (optimal_threshold, search_history)
        """
        n_samples = embeddings.shape[0]
        
        # Sample for large datasets
        if n_samples > 5000:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, 5000, replace=False)
            sample_emb = embeddings[sample_idx]
        else:
            sample_emb = embeddings
        
        sim_matrix = cosine_similarity(sample_emb)
        distance_matrix = np.clip(1 - sim_matrix, 0, 2)
        
        # Binary search
        low = sim_stats['min']
        high = sim_stats['max']
        search_history = []
        best_threshold = (low + high) / 2
        best_diff = float('inf')
        
        for iteration in range(20):
            mid = (low + high) / 2
            distance_thresh = 1 - mid
            
            try:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_thresh,
                    metric='precomputed',
                    linkage='average'
                )
                labels = model.fit_predict(distance_matrix)
                
                unique, counts = np.unique(labels, return_counts=True)
                valid_clusters = unique[counts >= self.min_cluster_size]
                n_clusters = len(valid_clusters)
                
                search_history.append({
                    'iteration': iteration,
                    'threshold': float(mid),
                    'n_clusters': n_clusters,
                    'target': target_clusters
                })
                
                diff = abs(n_clusters - target_clusters)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = mid
                
                if n_clusters < target_clusters:
                    low = mid
                elif n_clusters > target_clusters:
                    high = mid
                else:
                    break
                
                if high - low < 0.001:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Iteration {iteration} failed: {e}")
                break
        
        return best_threshold, search_history
