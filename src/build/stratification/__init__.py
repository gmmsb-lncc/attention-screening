"""
Stratification module for DockTKinase build system.

This module provides cosine similarity-based stratification methods
for creating balanced train/test/validation splits of protein-ligand
interaction data.
"""
from .cosine_similarity_calculator import CosineSimilarityCalculator
from .stratifier import Stratifier
from .validator import SplitValidator
from .cluster_analyzer import ClusterAnalyzer
from .clustering_metrics import ClusteringMetrics
from .similarity_analysis import SimilarityAnalyzer
from .threshold_optimization import ThresholdOptimizer
from .leakage_aware_optimization import LeakageAwareOptimizer
from .scalable_clustering import ScalableClustering
from .adaptive_clustering import AdaptiveClustering, AdaptiveClusteringStrategy

__all__ = [
    'CosineSimilarityCalculator',
    'Stratifier',
    'SplitValidator',
    'ClusterAnalyzer',
    'ClusteringMetrics',
    'SimilarityAnalyzer',
    'ThresholdOptimizer',
    'LeakageAwareOptimizer',
    'ScalableClustering',
    'AdaptiveClustering',
    'AdaptiveClusteringStrategy',
]