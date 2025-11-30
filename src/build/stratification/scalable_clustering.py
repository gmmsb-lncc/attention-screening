"""
Scalable Clustering - Memory-efficient clustering for large datasets.

This module implements Representative Sampling + Label Propagation for
clustering datasets too large for full pairwise distance matrices.

Algorithm:
1. Stratified sampling to create representative subset
2. Cluster the sample using standard agglomerative clustering
3. Compute cluster centroids from the sample
4. Assign all points to nearest centroid (O(n*k) instead of O(n²))

References:
- Sculley, D. (2010). Web-scale k-means clustering. WWW '10
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding

Author: DockTKinase Team
"""

import numpy as np
import logging
from typing import Optional, Dict, Any
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .clustering_metrics import ClusteringMetrics
from .similarity_analysis import SimilarityAnalyzer


class ScalableClustering:
    """
    Memory-efficient clustering using Representative Sampling + Label Propagation.
    
    For datasets > 40k samples where full distance matrix is infeasible.
    Memory complexity: O(sample_size²) instead of O(n²)
    Time complexity: O(n*k + sample_size²) where k << n
    """
    
    # Maximum samples for which full distance matrix is feasible (~8GB RAM)
    MAX_SAMPLES_FOR_FULL_MATRIX = 40000
    
    def __init__(self,
                 min_clusters: int = 5,
                 max_clusters: int = 100,
                 min_cluster_size: int = 3,
                 target_cluster_ratio: float = 0.01,
                 manual_threshold: Optional[float] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize scalable clustering.
        
        Args:
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            min_cluster_size: Minimum points per cluster
            target_cluster_ratio: Target clusters as ratio of samples
            manual_threshold: User-specified threshold (optional)
            logger: Logger instance
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.target_cluster_ratio = target_cluster_ratio
        self.manual_threshold = manual_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.optimal_threshold: Optional[float] = None
        self.metrics: Optional[ClusteringMetrics] = None
        self.search_history = []
        
        self._similarity_analyzer = SimilarityAnalyzer(logger=self.logger)
    
    def needs_scalable_approach(self, n_samples: int) -> bool:
        """Check if dataset needs scalable clustering approach."""
        return n_samples > self.MAX_SAMPLES_FOR_FULL_MATRIX
    
    def _calculate_target_clusters(self, n_samples: int) -> int:
        """Calculate target number of clusters."""
        target = int(n_samples * self.target_cluster_ratio)
        return max(self.min_clusters, min(target, self.max_clusters))
    
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform scalable clustering using Representative Sampling + Label Propagation.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            
        Returns:
            Cluster labels for all samples
            
        Raises:
            RuntimeError: If clustering fails after all attempts
        """
        n_samples, n_features = embeddings.shape
        
        try:
            # Normalize embeddings
            embeddings_normalized = normalize(embeddings)
            
            # === STEP 1: Determine sample size ===
            sample_size = min(50000, max(5000, int(np.sqrt(n_samples) * 50)))
            sample_size = min(sample_size, n_samples)
            
            self.logger.info(f"Scalable clustering: {n_samples} samples → {sample_size} sample size")
            
            # === STEP 2: Stratified sampling ===
            sample_idx = self._get_representative_sample(embeddings_normalized, sample_size)
            sample_embeddings = embeddings_normalized[sample_idx]
            
            self.logger.info(f"Representative sample selected: {len(sample_idx)} points")
            
            # === STEP 3: Cluster the sample ===
            target_clusters = self._calculate_target_clusters(n_samples)
            target_clusters = max(self.min_clusters, min(self.max_clusters, target_clusters))
            
            sim_stats = self._similarity_analyzer.analyze_distribution(sample_embeddings)
            
            # Find optimal threshold
            if self.manual_threshold is not None:
                optimal_threshold = self.manual_threshold
            else:
                if sim_stats['homogeneity'] == 'very_high':
                    optimal_threshold = sim_stats['p75']
                elif sim_stats['homogeneity'] == 'high':
                    optimal_threshold = sim_stats['p50']
                else:
                    optimal_threshold = sim_stats['p25']
            
            self.optimal_threshold = optimal_threshold
            distance_thresh = 1 - optimal_threshold
            
            self.logger.info(f"Clustering sample with threshold={optimal_threshold:.4f}")
            
            # Compute distance matrix for sample
            sample_distances = cosine_distances(sample_embeddings)
            
            # Agglomerative clustering
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_thresh,
                metric='precomputed',
                linkage='average'
            )
            sample_labels = model.fit_predict(sample_distances)
            
            # Adjust if too few clusters
            n_sample_clusters = len(np.unique(sample_labels))
            if n_sample_clusters < self.min_clusters:
                self.logger.warning(
                    f"Sample clustering produced only {n_sample_clusters} clusters. Adjusting threshold."
                )
                for mult in [1.05, 1.1, 1.15, 1.2]:
                    new_thresh = min(0.99, optimal_threshold * mult)
                    model = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=1 - new_thresh,
                        metric='precomputed',
                        linkage='average'
                    )
                    sample_labels = model.fit_predict(sample_distances)
                    n_sample_clusters = len(np.unique(sample_labels))
                    if n_sample_clusters >= self.min_clusters:
                        self.optimal_threshold = new_thresh
                        break
            
            self.logger.info(f"Sample clustering: {n_sample_clusters} clusters found")
            
            # === STEP 4: Compute cluster centroids ===
            unique_labels = np.unique(sample_labels)
            centroids = np.zeros((len(unique_labels), n_features))
            
            for i, label in enumerate(unique_labels):
                mask = sample_labels == label
                centroids[i] = sample_embeddings[mask].mean(axis=0)
            
            centroids = normalize(centroids)
            
            # === STEP 5: Assign all points to nearest centroid ===
            batch_size = 10000
            all_labels = np.zeros(n_samples, dtype=np.int32)
            
            self.logger.info(f"Assigning {n_samples} points to {len(centroids)} centroids...")
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = embeddings_normalized[start:end]
                distances = cosine_distances(batch, centroids)
                all_labels[start:end] = unique_labels[np.argmin(distances, axis=1)]
            
            # === STEP 6: Filter small clusters ===
            unique, counts = np.unique(all_labels, return_counts=True)
            small_clusters = unique[counts < self.min_cluster_size]
            
            if len(small_clusters) > 0:
                self.logger.info(f"Merging {len(small_clusters)} small clusters")
                
                large_clusters = unique[counts >= self.min_cluster_size]
                large_centroids = np.zeros((len(large_clusters), n_features))
                for i, lc in enumerate(large_clusters):
                    mask = all_labels == lc
                    large_centroids[i] = embeddings_normalized[mask].mean(axis=0)
                large_centroids = normalize(large_centroids)
                
                for sc in small_clusters:
                    sc_mask = all_labels == sc
                    sc_points = embeddings_normalized[sc_mask]
                    distances = cosine_distances(sc_points, large_centroids)
                    nearest = large_clusters[np.argmin(distances, axis=1)]
                    all_labels[sc_mask] = nearest
            
            # === STEP 7: Renumber clusters ===
            all_labels = self._renumber_clusters(all_labels)
            
            # === STEP 8: Calculate metrics ===
            n_clusters = len(np.unique(all_labels[all_labels != -1]))
            n_noise = np.sum(all_labels == -1)
            
            sil_score = None
            if n_clusters >= 2:
                try:
                    eval_sample_size = min(5000, n_samples)
                    eval_idx = np.random.choice(n_samples, eval_sample_size, replace=False)
                    sil_score = float(silhouette_score(
                        embeddings_normalized[eval_idx],
                        all_labels[eval_idx]
                    ))
                except Exception as e:
                    self.logger.warning(f"Could not compute silhouette score: {e}")
            
            unique_final, counts_final = np.unique(all_labels[all_labels != -1], return_counts=True)
            cluster_sizes = dict(zip([int(u) for u in unique_final], [int(c) for c in counts_final]))
            
            self.search_history = [{
                'method': 'representative_sampling',
                'sample_size': sample_size,
                'n_clusters': n_clusters,
                'threshold': self.optimal_threshold,
                'n_samples': n_samples
            }]
            
            self.metrics = ClusteringMetrics(
                n_clusters=n_clusters,
                n_samples=n_samples,
                n_noise=n_noise,
                silhouette_score=sil_score,
                calinski_harabasz_score=None,
                davies_bouldin_score=None,
                cluster_sizes=cluster_sizes,
                threshold_used=self.optimal_threshold or 0.0,
                method='representative_sampling',
                similarity_stats=sim_stats,
                threshold_search_history=self.search_history,
                split_quality_metrics=None
            )
            
            self.logger.info(
                f"Scalable clustering complete: {n_clusters} clusters, "
                f"silhouette={sil_score:.4f if sil_score else 'N/A'}"
            )
            
            return all_labels
        
        except MemoryError as e:
            self._raise_memory_error(n_samples, n_features, sample_size, e)
            
        except Exception as e:
            self._raise_clustering_error(n_samples, n_features, e)
    
    def _get_representative_sample(self, embeddings: np.ndarray, sample_size: int) -> np.ndarray:
        """
        Get representative sample using PCA-based stratified sampling.
        
        Args:
            embeddings: Normalized embedding matrix
            sample_size: Target sample size
            
        Returns:
            Indices of selected samples
        """
        n_samples = embeddings.shape[0]
        
        if n_samples <= sample_size:
            return np.arange(n_samples)
        
        # Part 1: Random sample (50%)
        random_size = sample_size // 2
        random_idx = np.random.choice(n_samples, random_size, replace=False)
        
        # Part 2: PCA-stratified sample (50%)
        pca_size = sample_size - random_size
        
        try:
            n_components = min(10, embeddings.shape[1], n_samples)
            pca = PCA(n_components=n_components, random_state=42)
            projected = pca.fit_transform(embeddings)
            
            n_bins = max(2, int(np.sqrt(pca_size / n_components)))
            
            stratified_idx = set()
            samples_per_bin = pca_size // (n_bins * n_components) + 1
            
            for pc in range(min(3, n_components)):
                pc_values = projected[:, pc]
                bins = np.percentile(pc_values, np.linspace(0, 100, n_bins + 1))
                
                for i in range(n_bins):
                    if i == n_bins - 1:
                        mask = (pc_values >= bins[i]) & (pc_values <= bins[i + 1])
                    else:
                        mask = (pc_values >= bins[i]) & (pc_values < bins[i + 1])
                    
                    bin_indices = np.where(mask)[0]
                    if len(bin_indices) > 0:
                        n_select = min(samples_per_bin, len(bin_indices))
                        selected = np.random.choice(bin_indices, n_select, replace=False)
                        stratified_idx.update(selected)
            
            stratified_idx = np.array(list(stratified_idx))
            
            if len(stratified_idx) < pca_size:
                remaining = pca_size - len(stratified_idx)
                available = np.setdiff1d(np.arange(n_samples), stratified_idx)
                additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                stratified_idx = np.concatenate([stratified_idx, additional])
                
        except Exception as e:
            self.logger.warning(f"PCA stratification failed: {e}. Using random sampling.")
            stratified_idx = np.random.choice(n_samples, pca_size, replace=False)
        
        all_idx = np.unique(np.concatenate([random_idx, stratified_idx]))
        
        if len(all_idx) > sample_size:
            all_idx = np.random.choice(all_idx, sample_size, replace=False)
        
        return all_idx
    
    def _renumber_clusters(self, labels: np.ndarray) -> np.ndarray:
        """Renumber clusters to be consecutive starting from 0."""
        new_labels = labels.copy()
        unique_labels = np.unique(labels[labels != -1])
        
        for new_id, old_id in enumerate(sorted(unique_labels)):
            new_labels[labels == old_id] = new_id
        
        return new_labels
    
    def _raise_memory_error(self, n_samples: int, n_features: int, sample_size: int, e: Exception):
        """Raise detailed memory error."""
        error_msg = (
            f"CLUSTERING FAILED - Memory Error\n"
            f"{'━' * 80}\n"
            f"Dataset size: {n_samples:,} samples × {n_features} features\n"
            f"Sample size attempted: {sample_size:,} samples\n"
            f"Estimated memory: {(sample_size**2 * 4) / (1024**3):.2f} GiB\n"
            f"\n"
            f"REASON: Representative sampling approach exceeded available memory.\n"
            f"\n"
            f"RECOMMENDATIONS:\n"
            f"  1. Increase system RAM\n"
            f"  2. Reduce sample_size parameter\n"
            f"  3. Use random stratification instead\n"
            f"{'━' * 80}\n"
        )
        self.logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    def _raise_clustering_error(self, n_samples: int, n_features: int, e: Exception):
        """Raise detailed clustering error."""
        error_msg = (
            f"CLUSTERING FAILED - Unexpected Error\n"
            f"{'━' * 80}\n"
            f"Dataset size: {n_samples:,} samples × {n_features} features\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}\n"
            f"\n"
            f"RECOMMENDATIONS:\n"
            f"  1. Check for NaN or Inf values in embeddings\n"
            f"  2. Verify embedding dimensions are consistent\n"
            f"  3. Try with smaller subset to isolate issue\n"
            f"{'━' * 80}\n"
        )
        self.logger.error(error_msg)
        raise RuntimeError(error_msg) from e
