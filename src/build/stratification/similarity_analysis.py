"""
Similarity Analysis - Analyze embedding similarity distributions.

This module provides utilities for analyzing the distribution of
pairwise cosine similarities in embedding matrices.

Author: DockTKinase Team
"""

import numpy as np
import logging
from typing import Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAnalyzer:
    """
    Analyzes the distribution of pairwise cosine similarities.
    
    Used to understand data homogeneity and guide threshold selection
    for clustering algorithms.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize similarity analyzer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_distribution(self, 
                            embeddings: np.ndarray,
                            sample_size: int = 2000) -> Dict[str, float]:
        """
        Analyze the distribution of pairwise cosine similarities.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            sample_size: Sample size for large datasets
            
        Returns:
            Dictionary with similarity statistics including:
            - min, max, mean, std, median
            - Percentiles: p5, p10, p25, p50, p75, p90, p95, p99
            - homogeneity: 'very_high', 'high', 'moderate', or 'low'
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
        stats['homogeneity'] = self._classify_homogeneity(stats['min'])
        
        return stats
    
    def _classify_homogeneity(self, min_similarity: float) -> str:
        """
        Classify data homogeneity based on minimum similarity.
        
        Args:
            min_similarity: Minimum pairwise similarity
            
        Returns:
            Homogeneity classification string
        """
        if min_similarity > 0.9:
            return 'very_high'
        elif min_similarity > 0.7:
            return 'high'
        elif min_similarity > 0.5:
            return 'moderate'
        else:
            return 'low'
    
    def get_search_range(self, sim_stats: Dict[str, float]) -> tuple:
        """
        Get appropriate threshold search range based on similarity distribution.
        
        Args:
            sim_stats: Similarity statistics from analyze_distribution()
            
        Returns:
            Tuple of (min_threshold, max_threshold)
        """
        homogeneity = sim_stats.get('homogeneity', 'moderate')
        
        if homogeneity == 'very_high':
            return sim_stats['p50'], sim_stats['p99']
        elif homogeneity == 'high':
            return sim_stats['p25'], sim_stats['p95']
        else:
            return 0.4, 0.9
    
    def compute_distance_matrix(self, 
                                embeddings: np.ndarray,
                                sample_size: Optional[int] = None) -> np.ndarray:
        """
        Compute distance matrix from embeddings.
        
        Args:
            embeddings: Embedding matrix
            sample_size: If provided, sample this many embeddings first
            
        Returns:
            Distance matrix (1 - similarity), clipped to [0, 2]
        """
        if sample_size is not None and len(embeddings) > sample_size:
            np.random.seed(42)
            idx = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[idx]
        
        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = np.clip(1 - sim_matrix, 0, 2)
        
        return distance_matrix
