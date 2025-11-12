"""
Cluster analyzer for assessing clustering quality of molecular embeddings.

This module provides metrics and visualizations to evaluate the quality
of clusters formed by the stratifier.
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path
import logging
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from src.build.core.base_builder import BaseBuilder
from src.build.core.config import BuildConfig
from src.build.core.exceptions import BuildException


class ClusterAnalyzer(BaseBuilder):
    """
    Analyzer for assessing clustering quality of molecular embeddings.
    
    Provides quantitative and visual metrics for evaluating cluster quality.
    """
    
    def __init__(self, 
                 config: Optional[BuildConfig] = None,
                 **kwargs):
        """
        Initialize cluster analyzer.

        Args:
            config: Build configuration
            **kwargs: Additional configuration options
        """
        super().__init__(config, **kwargs)
        
    def _validate_config(self) -> None:
        """Validate configuration."""
        # No specific validation needed for analyzer
        pass
    
    def calculate_clustering_metrics(self, 
                                   embeddings: np.ndarray, 
                                   cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard clustering quality metrics.

        Args:
            embeddings: Embedding matrix used for clustering
            cluster_labels: Cluster labels assigned to each sample

        Returns:
            Dictionary with clustering quality metrics
        """
        if len(np.unique(cluster_labels)) < 2:
            self.logger.warning("Only one cluster found, cannot calculate metrics")
            return {
                'silhouette_score': np.nan,
                'calinski_harabasz_score': np.nan,
                'davies_bouldin_score': np.nan,
                'n_clusters': len(np.unique(cluster_labels)),
                'n_samples': len(cluster_labels)
            }
        
        try:
            # Silhouette score: measures how similar an object is to its own cluster
            # compared to other clusters (ranges from -1 to 1, higher is better)
            silhouette = silhouette_score(embeddings, cluster_labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate silhouette score: {e}")
            silhouette = np.nan
        
        try:
            # Calinski-Harabasz score: ratio of between-cluster dispersion to within-cluster dispersion
            # Higher values indicate better clustering
            calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate Calinski-Harabasz score: {e}")
            calinski_harabasz = np.nan
        
        try:
            # Davies-Bouldin score: average similarity between each cluster and its most similar cluster
            # Lower values indicate better clustering
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate Davies-Bouldin score: {e}")
            davies_bouldin = np.nan
        
        # Additional metrics
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))  # Exclude noise points
        n_noise = np.sum(cluster_labels == -1)
        total_samples = len(cluster_labels)
        cluster_counts = np.bincount(cluster_labels[cluster_labels >= 0]) if -1 not in cluster_labels else np.bincount(cluster_labels[cluster_labels >= 0])
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'total_samples': total_samples,
            'cluster_size_min': int(np.min(cluster_counts)) if len(cluster_counts) > 0 else 0,
            'cluster_size_max': int(np.max(cluster_counts)) if len(cluster_counts) > 0 else 0,
            'cluster_size_mean': float(np.mean(cluster_counts)) if len(cluster_counts) > 0 else 0,
            'cluster_size_std': float(np.std(cluster_counts)) if len(cluster_counts) > 0 else 0
        }
    
    def analyze_cluster_distribution(self, 
                                   cluster_labels: np.ndarray,
                                   labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze the distribution of clusters and labels within clusters.

        Args:
            cluster_labels: Cluster labels assigned to each sample
            labels: Optional target labels for distribution analysis

        Returns:
            Dictionary with distribution analysis
        """
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        
        result = {
            'cluster_counts': dict(zip(unique_clusters, cluster_counts)),
            'n_unique_clusters': len(unique_clusters),
            'cluster_size_stats': {
                'min': int(np.min(cluster_counts)),
                'max': int(np.max(cluster_counts)),
                'mean': float(np.mean(cluster_counts)),
                'std': float(np.std(cluster_counts)),
                'median': float(np.median(cluster_counts))
            }
        }
        
        if labels is not None:
            # Analyze label distribution within clusters
            label_dist_in_clusters = {}
            unique_labels = np.unique(labels)
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_labels_subset = labels[cluster_mask]
                
                if len(cluster_labels_subset) > 0:
                    cluster_label_counts = {}
                    for label in unique_labels:
                        cluster_label_counts[label] = int(np.sum(cluster_labels_subset == label))
                    
                    label_dist_in_clusters[int(cluster_id)] = cluster_label_counts
            
            result['label_distribution_in_clusters'] = label_dist_in_clusters
            
            # Calculate label diversity per cluster
            cluster_diversity = {}
            for cluster_id, label_counts in label_dist_in_clusters.items():
                # Number of unique labels in cluster
                n_unique_labels = len([count for count in label_counts.values() if count > 0])
                cluster_diversity[cluster_id] = n_unique_labels
            
            result['cluster_label_diversity'] = cluster_diversity
        
        return result
    
    def visualize_clusters(self, 
                         embeddings: np.ndarray, 
                         cluster_labels: np.ndarray,
                         labels: Optional[np.ndarray] = None,
                         output_path: Optional[Union[str, Path]] = None,
                         figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Create visualizations of clusters in reduced dimensionality.

        Args:
            embeddings: High-dimensional embedding matrix
            cluster_labels: Cluster labels for each sample
            labels: Optional target labels for color coding
            output_path: Optional path to save the plot
            figsize: Figure size

        Returns:
            Matplotlib figure object
        """
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Cluster Analysis Visualizations', fontsize=16)
        
        # Plot 1: Clusters colored by cluster ID
        scatter1 = axes[0, 0].scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=cluster_labels, 
            cmap='tab10',
            alpha=0.7
        )
        axes[0, 0].set_title('Clusters by ID')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # Plot 2: Cluster size distribution
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        axes[0, 1].bar(range(len(cluster_counts)), cluster_counts)
        axes[0, 1].set_title('Cluster Size Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Size')
        
        # Plot 3: If labels are provided, show them
        if labels is not None:
            scatter3 = axes[1, 0].scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=labels, 
                cmap='viridis',
                alpha=0.7
            )
            axes[1, 0].set_title('Target Labels')
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter3, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'No target labels provided', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Target Labels')
        
        # Plot 4: Clusters colored by cluster ID with different style
        scatter4 = axes[1, 1].scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=cluster_labels, 
            cmap='Set1',
            alpha=0.7,
            s=30
        )
        axes[1, 1].set_title('Clusters (Alternative Color Scheme)')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Cluster visualization saved to {output_path}")
        
        return fig
    
    def compare_clustering_strategies(self, 
                                    embeddings: np.ndarray,
                                    cluster_labels_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple clustering strategies using quality metrics.

        Args:
            embeddings: Embedding matrix used for clustering
            cluster_labels_dict: Dictionary mapping strategy names to cluster labels

        Returns:
            Dictionary mapping strategy names to their quality metrics
        """
        comparison_results = {}
        
        for strategy_name, cluster_labels in cluster_labels_dict.items():
            metrics = self.calculate_clustering_metrics(embeddings, cluster_labels)
            comparison_results[strategy_name] = metrics
            
        return comparison_results
    
    def identify_outliers(self, 
                         embeddings: np.ndarray,
                         cluster_labels: np.ndarray,
                         method: str = 'distance_from_centroid',
                         threshold: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Identify potential outlier samples in clusters.

        Args:
            embeddings: Embedding matrix
            cluster_labels: Cluster labels for each sample
            method: Method for outlier detection ('distance_from_centroid', 'silhouette')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary with outlier indices by cluster
        """
        outliers_by_cluster = {}
        
        unique_clusters = np.unique(cluster_labels)
        
        if method == 'distance_from_centroid':
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]
                
                if len(cluster_embeddings) <= 1:
                    continue  # Skip clusters with 1 or 0 points
                
                # Calculate centroid of cluster
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate distances from centroid
                distances = np.sqrt(np.sum((cluster_embeddings - centroid) ** 2, axis=1))
                
                # Calculate z-scores (or use median absolute deviation for robustness)
                if len(distances) > 1:
                    # Use median absolute deviation for robust outlier detection
                    median_dist = np.median(distances)
                    mad = np.median(np.abs(distances - median_dist))
                    
                    if mad > 0:
                        modified_z_scores = 0.6745 * (distances - median_dist) / mad
                        outlier_mask = modified_z_scores > threshold
                    else:
                        # If MAD is 0, use standard deviation instead
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)
                        if std_dist > 0:
                            z_scores = (distances - mean_dist) / std_dist
                            outlier_mask = z_scores > threshold
                        else:
                            outlier_mask = np.zeros(len(distances), dtype=bool)
                else:
                    outlier_mask = np.zeros(len(distances), dtype=bool)
                
                # Map back to original indices
                original_indices = np.where(cluster_mask)[0]
                cluster_outliers = original_indices[outlier_mask]
                
                if len(cluster_outliers) > 0:
                    outliers_by_cluster[int(cluster_id)] = cluster_outliers
        
        elif method == 'silhouette':
            # Calculate silhouette scores
            silhouette_scores = silhouette_score(embeddings, cluster_labels, sample_score=True)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_silhouette_scores = silhouette_scores[cluster_mask]
                
                # Identify points with low silhouette scores (potential outliers)
                mean_silhouette = np.mean(cluster_silhouette_scores)
                outlier_mask = cluster_silhouette_scores < (mean_silhouette - threshold * np.std(cluster_silhouette_scores))
                
                # Map back to original indices
                original_indices = np.where(cluster_mask)[0]
                cluster_outliers = original_indices[outlier_mask]
                
                if len(cluster_outliers) > 0:
                    outliers_by_cluster[int(cluster_id)] = cluster_outliers
        
        return outliers_by_cluster
    
    def build(self) -> dict:
        """
        Build method for BaseBuilder compatibility.

        Returns:
            Dictionary with analyzer information
        """
        return {
            'analyzer_initialized': True
        }