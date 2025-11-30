"""
Leakage-Aware Split Evaluation - Optimize splits for minimal data leakage.

This module provides tools for evaluating and optimizing train/val/test
splits to minimize information leakage between sets.

Author: DockTKinase Team
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import entropy


class LeakageAwareOptimizer:
    """
    Optimizes clustering thresholds for minimal leakage between splits.
    
    Evaluates splits based on:
    - Separation: Maximum similarity between splits (lower = better)
    - Coverage: Internal diversity of each split (higher = better)
    - Balance: Label distribution similarity (lower KL = better)
    """
    
    def __init__(self,
                 test_size: float = 0.1,
                 val_size: float = 0.1,
                 min_clusters: int = 5,
                 min_cluster_size: int = 3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize leakage-aware optimizer.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            min_clusters: Minimum number of clusters
            min_cluster_size: Minimum points per cluster
            logger: Logger instance
        """
        self.test_size = test_size
        self.val_size = val_size
        self.min_clusters = min_clusters
        self.min_cluster_size = min_cluster_size
        self.logger = logger or logging.getLogger(__name__)
    
    def _max_similarity_between_splits(self, 
                                       embeddings: np.ndarray, 
                                       idx_a: np.ndarray, 
                                       idx_b: np.ndarray,
                                       n_samples: int = 1000) -> float:
        """
        Calculate maximum cosine similarity between two splits.
        
        Args:
            embeddings: Full embedding matrix
            idx_a: Indices for first split
            idx_b: Indices for second split
            n_samples: Max samples per split for computation
            
        Returns:
            Maximum similarity between any pair (a, b)
        """
        if len(idx_a) == 0 or len(idx_b) == 0:
            return 0.0
        
        # Sample if needed
        if len(idx_a) > n_samples:
            idx_a = np.random.choice(idx_a, n_samples, replace=False)
        if len(idx_b) > n_samples:
            idx_b = np.random.choice(idx_b, n_samples, replace=False)
        
        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]
        
        sim_matrix = cosine_similarity(emb_a, emb_b)
        return float(np.max(sim_matrix))
    
    def _mean_pairwise_distance(self, embeddings: np.ndarray, n_samples: int = 1000) -> float:
        """
        Calculate mean pairwise distance within embeddings.
        
        Args:
            embeddings: Embedding matrix
            n_samples: Max samples for computation
            
        Returns:
            Mean pairwise distance (1 - similarity)
        """
        if len(embeddings) < 2:
            return 0.0
        
        if len(embeddings) > n_samples:
            idx = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[idx]
        
        sim_matrix = cosine_similarity(embeddings)
        triu_idx = np.triu_indices(len(embeddings), k=1)
        similarities = sim_matrix[triu_idx]
        
        return float(1 - np.mean(similarities))
    
    def _label_distribution(self, labels: np.ndarray) -> np.ndarray:
        """Get normalized label distribution."""
        if len(labels) == 0:
            return np.array([1.0])
        
        unique, counts = np.unique(labels, return_counts=True)
        return counts / counts.sum()
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence with smoothing."""
        max_len = max(len(p), len(q))
        p_padded = np.zeros(max_len)
        q_padded = np.zeros(max_len)
        p_padded[:len(p)] = p
        q_padded[:len(q)] = q
        
        epsilon = 1e-10
        p_smooth = (p_padded + epsilon) / (p_padded + epsilon).sum()
        q_smooth = (q_padded + epsilon) / (q_padded + epsilon).sum()
        
        return float(entropy(p_smooth, q_smooth))
    
    def _split_clusters_for_evaluation(self,
                                       cluster_labels: np.ndarray,
                                       target_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split samples based on cluster assignments.
        
        Args:
            cluster_labels: Cluster assignment for each sample
            target_labels: Target labels
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Group samples by cluster
        clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(idx)
        
        train_idx, val_idx, test_idx = [], [], []
        
        for cluster_id, indices in clusters.items():
            n = len(indices)
            
            if n == 1:
                train_idx.extend(indices)
            elif n == 2:
                train_idx.append(indices[0])
                test_idx.append(indices[1])
            else:
                n_test = max(1, int(n * self.test_size))
                n_val = max(1, int(n * self.val_size)) if self.val_size > 0 else 0
                
                shuffled = list(indices)
                np.random.shuffle(shuffled)
                
                test_idx.extend(shuffled[:n_test])
                val_idx.extend(shuffled[n_test:n_test + n_val])
                train_idx.extend(shuffled[n_test + n_val:])
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    
    def evaluate_split_quality(self,
                               embeddings: np.ndarray,
                               labels: np.ndarray,
                               train_idx: np.ndarray,
                               val_idx: np.ndarray,
                               test_idx: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the quality of a train/val/test split.
        
        Args:
            embeddings: Embedding matrix
            labels: Target labels
            train_idx, val_idx, test_idx: Split indices
            
        Returns:
            Dictionary with quality metrics and composite score
        """
        # === SEPARATION ===
        max_sim_train_test = self._max_similarity_between_splits(embeddings, train_idx, test_idx)
        max_sim_train_val = self._max_similarity_between_splits(embeddings, train_idx, val_idx) if len(val_idx) > 0 else 0.0
        max_sim_val_test = self._max_similarity_between_splits(embeddings, val_idx, test_idx) if len(val_idx) > 0 else 0.0
        
        separation_score = (
            0.50 * (1 - max_sim_train_test) +
            0.30 * (1 - max_sim_train_val) +
            0.20 * (1 - max_sim_val_test)
        )
        
        # === COVERAGE ===
        total_diversity = self._mean_pairwise_distance(embeddings)
        
        if total_diversity > 0:
            diversity_train = self._mean_pairwise_distance(embeddings[train_idx])
            diversity_val = self._mean_pairwise_distance(embeddings[val_idx]) if len(val_idx) > 0 else 0.0
            diversity_test = self._mean_pairwise_distance(embeddings[test_idx])
            
            coverage_score = min(1.0, (
                0.60 * (diversity_train / total_diversity) +
                0.20 * (diversity_val / total_diversity if len(val_idx) > 0 else 1.0) +
                0.20 * (diversity_test / total_diversity)
            ))
        else:
            diversity_train = diversity_val = diversity_test = 0.0
            coverage_score = 0.5
        
        # === BALANCE ===
        total_dist = self._label_distribution(labels)
        
        kl_train = self._kl_divergence(self._label_distribution(labels[train_idx]), total_dist)
        kl_val = self._kl_divergence(self._label_distribution(labels[val_idx]), total_dist) if len(val_idx) > 0 else 0.0
        kl_test = self._kl_divergence(self._label_distribution(labels[test_idx]), total_dist)
        
        balance_score = (
            0.40 * (1 / (1 + kl_train)) +
            0.30 * (1 / (1 + kl_val)) if len(val_idx) > 0 else 0.30 +
            0.30 * (1 / (1 + kl_test))
        )
        
        # === COMPOSITE SCORE ===
        final_score = 0.45 * separation_score + 0.30 * coverage_score + 0.25 * balance_score
        
        return {
            'final_score': final_score,
            'separation': {
                'score': separation_score,
                'train_test_max_sim': max_sim_train_test,
                'train_val_max_sim': max_sim_train_val,
                'val_test_max_sim': max_sim_val_test
            },
            'coverage': {
                'score': coverage_score,
                'total_diversity': total_diversity,
                'train_diversity': diversity_train,
                'val_diversity': diversity_val,
                'test_diversity': diversity_test
            },
            'balance': {
                'score': balance_score,
                'train_kl': kl_train,
                'val_kl': kl_val,
                'test_kl': kl_test
            },
            'sizes': {
                'train': len(train_idx),
                'val': len(val_idx),
                'test': len(test_idx)
            }
        }
    
    def find_threshold_leakage_aware(self,
                                     embeddings: np.ndarray,
                                     labels: np.ndarray,
                                     sim_stats: Dict[str, float],
                                     n_candidates: int = 12) -> Tuple[float, List[Dict], Dict[str, Any]]:
        """
        Find optimal threshold by maximizing split quality score.
        
        Args:
            embeddings: Embedding matrix
            labels: Target labels for stratification
            sim_stats: Similarity statistics
            n_candidates: Number of thresholds to evaluate
            
        Returns:
            Tuple of (optimal_threshold, search_history, best_split_metrics)
        """
        n_samples = embeddings.shape[0]
        self.logger.info(f"Leakage-aware threshold search with {n_candidates} candidates")
        
        # Define search range
        if sim_stats['homogeneity'] == 'very_high':
            min_thresh, max_thresh = sim_stats['p50'], sim_stats['p99']
        elif sim_stats['homogeneity'] == 'high':
            min_thresh, max_thresh = sim_stats['p25'], sim_stats['p95']
        else:
            min_thresh, max_thresh = 0.4, 0.9
        
        thresholds = np.linspace(min_thresh, max_thresh, n_candidates)
        search_history = []
        
        best_score = -1
        best_threshold = thresholds[len(thresholds) // 2]
        best_metrics = None
        
        # Sample for efficiency
        if n_samples > 5000:
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, 5000, replace=False)
            sample_emb = embeddings[sample_idx]
            sample_labels = labels[sample_idx]
        else:
            sample_emb = embeddings
            sample_labels = labels
        
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
                cluster_labels = model.fit_predict(distance_matrix)
                
                # Filter small clusters
                unique, counts = np.unique(cluster_labels, return_counts=True)
                small_clusters = unique[counts < self.min_cluster_size]
                for sc in small_clusters:
                    cluster_labels[cluster_labels == sc] = -1
                
                n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
                
                if n_clusters < self.min_clusters:
                    search_history.append({
                        'threshold': float(thresh),
                        'n_clusters': n_clusters,
                        'valid': False,
                        'reason': f'Too few clusters ({n_clusters} < {self.min_clusters})'
                    })
                    continue
                
                # Create trial split
                train_idx, val_idx, test_idx = self._split_clusters_for_evaluation(
                    cluster_labels, sample_labels
                )
                
                if len(train_idx) == 0 or len(test_idx) == 0:
                    search_history.append({
                        'threshold': float(thresh),
                        'n_clusters': n_clusters,
                        'valid': False,
                        'reason': 'Empty train or test split'
                    })
                    continue
                
                # Evaluate split quality
                metrics = self.evaluate_split_quality(
                    sample_emb, sample_labels, train_idx, val_idx, test_idx
                )
                
                result = {
                    'threshold': float(thresh),
                    'n_clusters': n_clusters,
                    'valid': True,
                    'final_score': metrics['final_score'],
                    'separation_score': metrics['separation']['score'],
                    'coverage_score': metrics['coverage']['score'],
                    'balance_score': metrics['balance']['score']
                }
                search_history.append(result)
                
                self.logger.debug(
                    f"Threshold {thresh:.4f}: {n_clusters} clusters, "
                    f"quality={metrics['final_score']:.4f}"
                )
                
                if metrics['final_score'] > best_score:
                    best_score = metrics['final_score']
                    best_threshold = thresh
                    best_metrics = metrics
                    
            except Exception as e:
                self.logger.warning(f"Failed for threshold {thresh}: {e}")
                search_history.append({
                    'threshold': float(thresh),
                    'error': str(e)
                })
        
        if best_metrics is None:
            self.logger.warning("No valid threshold found, using fallback")
            best_threshold = sim_stats.get('p75', 0.75)
            best_metrics = {'final_score': 0.0, 'fallback': True}
        
        self.logger.info(f"Best threshold: {best_threshold:.4f} (score={best_score:.4f})")
        
        return best_threshold, search_history, best_metrics
