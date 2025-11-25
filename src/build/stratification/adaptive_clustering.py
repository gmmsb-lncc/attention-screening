"""
Adaptive Clustering Strategy - Automatically determines optimal clustering parameters.

This module provides intelligent clustering that adapts to the data distribution,
solving the problem of homogeneous embeddings where fixed thresholds don't work.

Methods implemented:
1. Silhouette-based threshold optimization
2. Elbow method for k-means
3. Gap statistic
4. Target cluster count based on dataset size

Author: DockTKinase Team
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass, asdict


@dataclass
class ClusteringMetrics:
    """Container for clustering metrics."""
    n_clusters: int
    n_samples: int
    n_noise: int
    silhouette_score: Optional[float]
    calinski_harabasz_score: Optional[float]
    davies_bouldin_score: Optional[float]
    cluster_sizes: Dict[int, int]
    threshold_used: float
    method: str
    similarity_stats: Dict[str, float]
    threshold_search_history: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


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
    """
    
    def __init__(self,
                 method: str = 'silhouette',
                 min_clusters: int = 5,
                 max_clusters: int = 100,
                 min_cluster_size: int = 3,
                 target_cluster_ratio: float = 0.01,  # 1% of samples as clusters
                 manual_threshold: Optional[float] = None,  # User-specified threshold
                 logger: Optional[logging.Logger] = None):
        """
        Initialize adaptive clustering.
        
        Args:
            method: Method to use ('silhouette', 'elbow', 'target', 'percentile', 'manual')
            min_clusters: Minimum number of clusters desired
            max_clusters: Maximum number of clusters
            min_cluster_size: Minimum points per cluster
            target_cluster_ratio: Target clusters as ratio of samples (for 'target' method)
            manual_threshold: User-specified similarity threshold (for 'manual' method or override)
            logger: Logger instance
        """
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.target_cluster_ratio = target_cluster_ratio
        self.manual_threshold = manual_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # If manual threshold is provided, switch to manual method
        if manual_threshold is not None and method != 'elbow':
            self.method = 'manual'
            self.logger.info(f"Using manual threshold: {manual_threshold}")
        
        # Results
        self.optimal_threshold: Optional[float] = None
        self.metrics: Optional[ClusteringMetrics] = None
        self.search_history: List[Dict[str, Any]] = []
    
    def analyze_similarity_distribution(self, 
                                        embeddings: np.ndarray,
                                        sample_size: int = 2000) -> Dict[str, float]:
        """
        Analyze the distribution of pairwise cosine similarities.
        
        Args:
            embeddings: Embedding matrix
            sample_size: Sample size for large datasets
            
        Returns:
            Dictionary with similarity statistics
        """
        n_samples = embeddings.shape[0]
        
        # Sample if dataset is large
        if n_samples > sample_size:
            np.random.seed(42)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            sample = embeddings[idx]
        else:
            sample = embeddings
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(sample)
        
        # Get upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(sim_matrix.shape[0], k=1)
        similarities = sim_matrix[triu_idx]
        
        stats = {
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'median': float(np.median(similarities)),
            'p5': float(np.percentile(similarities, 5)),
            'p10': float(np.percentile(similarities, 10)),
            'p25': float(np.percentile(similarities, 25)),
            'p50': float(np.percentile(similarities, 50)),
            'p75': float(np.percentile(similarities, 75)),
            'p90': float(np.percentile(similarities, 90)),
            'p95': float(np.percentile(similarities, 95)),
            'p99': float(np.percentile(similarities, 99)),
        }
        
        # Classify homogeneity
        if stats['min'] > 0.9:
            stats['homogeneity'] = 'very_high'
        elif stats['min'] > 0.7:
            stats['homogeneity'] = 'high'
        elif stats['min'] > 0.5:
            stats['homogeneity'] = 'moderate'
        else:
            stats['homogeneity'] = 'low'
        
        return stats
    
    def _calculate_target_clusters(self, n_samples: int) -> int:
        """Calculate target number of clusters based on dataset size."""
        target = int(n_samples * self.target_cluster_ratio)
        return max(self.min_clusters, min(target, self.max_clusters))
    
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
        # For homogeneous data, search in high percentiles
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
        best_threshold = thresholds[len(thresholds)//2]  # Default to middle
        
        n_samples = embeddings.shape[0]
        
        # Precompute similarity matrix (or sample for large datasets)
        if n_samples > 5000:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, 5000, replace=False)
            sample_emb = embeddings[sample_idx]
        else:
            sample_emb = embeddings
            sample_idx = np.arange(n_samples)
        
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
                    labels[labels == sc] = -1  # Mark as noise
                
                n_clusters = len(np.unique(labels[labels != -1]))
                n_noise = np.sum(labels == -1)
                
                # Calculate silhouette score (only if we have valid clusters)
                valid_mask = labels != -1
                if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
                    score = silhouette_score(distance_matrix[valid_mask][:, valid_mask], 
                                            labels[valid_mask], 
                                            metric='precomputed')
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
                
                # Update best if valid and better score
                if result['valid'] and score > best_score:
                    best_score = score
                    best_threshold = thresh
                    
            except Exception as e:
                self.logger.warning(f"Failed for threshold {thresh}: {e}")
                search_history.append({
                    'threshold': float(thresh),
                    'error': str(e)
                })
        
        # If no valid threshold found, use percentile method as fallback
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
                if len(np.unique(labels)) >= 2:
                    sil_score = silhouette_score(embeddings, labels)
                else:
                    sil_score = None
                
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
            # Calculate rate of change
            diffs = np.diff(inertias)
            # Calculate second derivative (curvature)
            diffs2 = np.diff(diffs)
            # Elbow is where curvature is maximum
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
        
        for iteration in range(20):  # Max iterations
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
                
                # Filter small clusters
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
                    # Need more clusters -> higher threshold
                    low = mid
                elif n_clusters > target_clusters:
                    # Need fewer clusters -> lower threshold
                    high = mid
                else:
                    break  # Found exact target
                
                if high - low < 0.001:  # Convergence
                    break
                    
            except Exception as e:
                self.logger.warning(f"Iteration {iteration} failed: {e}")
                break
        
        return best_threshold, search_history
    
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
        if self.method == 'silhouette':
            self.optimal_threshold, self.search_history = self.find_optimal_threshold_silhouette(
                embeddings, sim_stats
            )
            distance_thresh = 1 - self.optimal_threshold
            
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            labels = model.fit_predict(distance_matrix)
            
        elif self.method == 'elbow':
            optimal_k, self.search_history = self.find_optimal_k_elbow(embeddings)
            self.optimal_threshold = None  # Not applicable for k-means
            
            model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = model.fit_predict(embeddings)
            
        elif self.method == 'target':
            self.optimal_threshold, self.search_history = self.find_threshold_by_target(
                embeddings, sim_stats, target_clusters
            )
            distance_thresh = 1 - self.optimal_threshold
            
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            labels = model.fit_predict(distance_matrix)
            
        elif self.method == 'percentile':
            # Use percentile based on homogeneity
            if sim_stats['homogeneity'] == 'very_high':
                self.optimal_threshold = sim_stats['p75']
            elif sim_stats['homogeneity'] == 'high':
                self.optimal_threshold = sim_stats['p50']
            else:
                self.optimal_threshold = 0.7
            
            distance_thresh = 1 - self.optimal_threshold
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            labels = model.fit_predict(distance_matrix)
            self.search_history = [{'method': 'percentile', 'threshold': self.optimal_threshold}]
        
        elif self.method == 'manual':
            # Use user-specified threshold
            if self.manual_threshold is None:
                raise ValueError("Manual method requires manual_threshold to be set")
            
            self.optimal_threshold = self.manual_threshold
            self.logger.info(f"Using manual threshold: {self.optimal_threshold}")
            
            # Warn if threshold seems inappropriate for the data
            if sim_stats['homogeneity'] == 'very_high' and self.optimal_threshold < sim_stats['p25']:
                self.logger.warning(
                    f"Warning: Manual threshold {self.optimal_threshold:.4f} is below P25 "
                    f"({sim_stats['p25']:.4f}) for highly homogeneous data. "
                    f"This may result in very few clusters."
                )
            
            distance_thresh = 1 - self.optimal_threshold
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            labels = model.fit_predict(distance_matrix)
            self.search_history = [{
                'method': 'manual', 
                'threshold': self.optimal_threshold,
                'user_specified': True,
                'similarity_stats': sim_stats
            }]
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
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
            threshold_search_history=self.search_history
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
