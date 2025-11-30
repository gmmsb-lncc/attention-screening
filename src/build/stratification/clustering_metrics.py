"""
Clustering Metrics - Data structures for clustering results.

This module provides dataclasses and utilities for storing and
serializing clustering metrics.

Author: DockTKinase Team
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


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
    # Leakage-aware specific metrics
    split_quality_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save_json(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringMetrics':
        """Create ClusteringMetrics from dictionary."""
        return cls(**data)
    
    @classmethod
    def load_json(cls, path: str) -> 'ClusteringMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
