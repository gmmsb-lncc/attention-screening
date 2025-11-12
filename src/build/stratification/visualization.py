"""
Visualization tools for stratification analysis.

Provides methods to visualize cluster separation in 2D using
dimensionality reduction techniques (t-SNE, UMAP, PCA).

Optimized for large datasets (millions of points) with:
- Intelligent downsampling
- Incremental PCA for large matrices
- Memory-efficient plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal
import logging
import warnings

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA, IncrementalPCA


class StratificationVisualizer:
    """
    Visualize stratification results using dimensionality reduction.
    
    Supports t-SNE, UMAP, and PCA for 2D projection.
    Optimized for large datasets with automatic downsampling.
    """
    
    def __init__(self, 
                 method: Literal['tsne', 'umap', 'pca'] = 'pca', 
                 random_state: int = 42,
                 max_samples: int = 50000,
                 use_incremental_pca: bool = True):
        """
        Initialize visualizer.
        
        Args:
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            random_state: Random seed for reproducibility
            max_samples: Maximum samples for visualization (auto-downsample if exceeded)
            use_incremental_pca: Use IncrementalPCA for large datasets (>100k samples)
        """
        self.method = method
        self.random_state = random_state
        self.max_samples = max_samples
        self.use_incremental_pca = use_incremental_pca
        self.logger = logging.getLogger(__name__)
        
        # Validate method availability
        if method == 'tsne' and not HAS_TSNE:
            self.logger.warning("t-SNE not available, falling back to PCA")
            self.method = 'pca'
        elif method == 'umap' and not HAS_UMAP:
            self.logger.warning("UMAP not available, falling back to PCA")
            self.method = 'pca'
    
    def _stratified_downsample(self, 
                               embeddings: np.ndarray,
                               indices_dict: dict,
                               target_size: int) -> Tuple[np.ndarray, dict]:
        """
        Downsample while preserving split proportions.
        
        Args:
            embeddings: Full embedding matrix
            indices_dict: Dict with 'train', 'val', 'test' indices
            target_size: Target number of samples
            
        Returns:
            Tuple of (downsampled_embeddings, downsampled_indices_dict)
        """
        n_samples = len(embeddings)
        
        if n_samples <= target_size:
            return embeddings, indices_dict
        
        self.logger.info(f"Downsampling from {n_samples:,} to {target_size:,} samples")
        
        # Calculate proportional sizes
        train_idx = indices_dict['train']
        val_idx = indices_dict['val']
        test_idx = indices_dict['test']
        
        train_ratio = len(train_idx) / n_samples
        val_ratio = len(val_idx) / n_samples
        test_ratio = len(test_idx) / n_samples
        
        n_train = int(target_size * train_ratio)
        n_val = int(target_size * val_ratio)
        n_test = target_size - n_train - n_val
        
        # Random sample from each split
        np.random.seed(self.random_state)
        sampled_train = np.random.choice(train_idx, size=min(n_train, len(train_idx)), replace=False)
        sampled_val = np.random.choice(val_idx, size=min(n_val, len(val_idx)), replace=False)
        sampled_test = np.random.choice(test_idx, size=min(n_test, len(test_idx)), replace=False)
        
        # Combine and create mapping
        all_sampled = np.concatenate([sampled_train, sampled_val, sampled_test])
        sampled_embeddings = embeddings[all_sampled]
        
        # Create new indices (0-based for sampled data)
        new_indices = {
            'train': np.arange(len(sampled_train)),
            'val': np.arange(len(sampled_train), len(sampled_train) + len(sampled_val)),
            'test': np.arange(len(sampled_train) + len(sampled_val), len(all_sampled))
        }
        
        return sampled_embeddings, new_indices
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce embeddings to 2D with automatic optimization for large datasets.
        
        Args:
            embeddings: High-dimensional embedding matrix (n_samples, n_features)
            
        Returns:
            2D projection (n_samples, 2)
        """
        n_samples, n_features = embeddings.shape
        self.logger.info(f"Reducing {n_samples:,} samples with {n_features} features using {self.method.upper()}")
        
        # For very large datasets, warn about computational cost
        if n_samples > 100000 and self.method in ['tsne', 'umap']:
            self.logger.warning(
                f"{self.method.upper()} with {n_samples:,} samples may be slow. "
                f"Consider using PCA or increasing max_samples limit."
            )
        
        if self.method == 'tsne':
            # t-SNE is O(n²), limit to reasonable size
            if n_samples > 10000:
                self.logger.warning(f"t-SNE is slow for {n_samples:,} samples. Using first 10k for speed.")
                sample_idx = np.random.choice(n_samples, 10000, replace=False)
                embeddings = embeddings[sample_idx]
            
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity)
            return reducer.fit_transform(embeddings)
        
        elif self.method == 'umap':
            n_neighbors = min(15, len(embeddings) - 1)
            reducer = umap.UMAP(
                n_components=2, 
                random_state=self.random_state, 
                n_neighbors=n_neighbors,
                low_memory=n_samples > 50000  # Use low memory mode for large datasets
            )
            return reducer.fit_transform(embeddings)
        
        else:  # pca
            # Use IncrementalPCA for very large datasets
            if self.use_incremental_pca and n_samples > 100000:
                self.logger.info(f"Using IncrementalPCA for {n_samples:,} samples")
                # Process in batches to avoid memory issues
                batch_size = min(10000, n_samples // 10)
                reducer = IncrementalPCA(n_components=2, batch_size=batch_size)
                
                # Fit in batches
                for i in range(0, n_samples, batch_size):
                    batch = embeddings[i:i + batch_size]
                    reducer.partial_fit(batch)
                
                # Transform in batches
                result = np.zeros((n_samples, 2))
                for i in range(0, n_samples, batch_size):
                    batch = embeddings[i:i + batch_size]
                    result[i:i + batch_size] = reducer.transform(batch)
                
                return result
            else:
                reducer = PCA(n_components=2, random_state=self.random_state)
                return reducer.fit_transform(embeddings)
    
    def plot_split_visualization(self,
                                embeddings: np.ndarray,
                                train_idx: np.ndarray,
                                val_idx: np.ndarray,
                                test_idx: np.ndarray,
                                cluster_labels: Optional[np.ndarray] = None,
                                title: str = "Stratification Visualization",
                                figsize: Tuple[int, int] = (15, 6),
                                save_path: Optional[str] = None,
                                show: bool = True,
                                dpi: int = 150) -> plt.Figure:
        """
        Create visualization of train/val/test splits.
        
        Automatically downsamples if dataset exceeds max_samples.
        
        Args:
            embeddings: Embedding matrix
            train_idx: Training set indices
            val_idx: Validation set indices
            test_idx: Test set indices
            cluster_labels: Optional cluster labels for each sample
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            show: Whether to display the plot
            dpi: DPI for saved figure (lower for large files)
            
        Returns:
            Matplotlib figure
        """
        n_samples = len(embeddings)
        
        # Downsample if necessary
        if n_samples > self.max_samples:
            self.logger.info(f"Dataset has {n_samples:,} samples, downsampling to {self.max_samples:,}")
            
            indices_dict = {'train': train_idx, 'val': val_idx, 'test': test_idx}
            embeddings, indices_dict = self._stratified_downsample(
                embeddings, indices_dict, self.max_samples
            )
            train_idx = indices_dict['train']
            val_idx = indices_dict['val']
            test_idx = indices_dict['test']
            
            # Downsample cluster labels if provided
            if cluster_labels is not None:
                all_idx = np.concatenate([train_idx, val_idx, test_idx])
                cluster_labels = cluster_labels[all_idx]
            
            title = f"{title}\n(Downsampled to {self.max_samples:,} samples)"
        
        # Reduce to 2D
        coords_2d = self.reduce_dimensions(embeddings)
        
        # Create figure with subplots
        if cluster_labels is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
            ax2 = None
        
        # Plot 1: Split visualization (Train/Val/Test)
        self._plot_splits(ax1, coords_2d, train_idx, val_idx, test_idx)
        ax1.set_title(f"{title}\nTrain/Val/Test Split")
        ax1.set_xlabel(f"{self.method.upper()} Component 1")
        ax1.set_ylabel(f"{self.method.upper()} Component 2")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster visualization (if available)
        if cluster_labels is not None and ax2 is not None:
            self._plot_clusters(ax2, coords_2d, cluster_labels, train_idx, val_idx, test_idx)
            ax2.set_title(f"{title}\nCluster Assignments")
            ax2.set_xlabel(f"{self.method.upper()} Component 1")
            ax2.set_ylabel(f"{self.method.upper()} Component 2")
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested (use lower DPI for large files)
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)  # Free memory if not showing
        
        return fig
    
    def _plot_splits(self, ax, coords_2d, train_idx, val_idx, test_idx, use_rasterized: bool = False):
        """
        Plot train/val/test splits.
        
        Args:
            use_rasterized: Rasterize scatter plots for faster rendering with many points
        """
        # Define colors and markers
        colors = {
            'train': '#2E86AB',  # Blue
            'val': '#A23B72',    # Purple
            'test': '#F18F01'    # Orange
        }
        
        markers = {
            'train': 'o',
            'val': 's',
            'test': '^'
        }
        
        sizes = {
            'train': 30,
            'val': 50,
            'test': 50
        }
        
        alphas = {
            'train': 0.6,
            'val': 0.8,
            'test': 0.8
        }
        
        # Plot each split (rasterized for better performance with many points)
        ax.scatter(coords_2d[train_idx, 0], coords_2d[train_idx, 1],
                  c=colors['train'], marker=markers['train'], s=sizes['train'],
                  alpha=alphas['train'], label=f'Train (n={len(train_idx):,})',
                  edgecolors='white', linewidths=0.5, rasterized=use_rasterized)
        
        ax.scatter(coords_2d[val_idx, 0], coords_2d[val_idx, 1],
                  c=colors['val'], marker=markers['val'], s=sizes['val'],
                  alpha=alphas['val'], label=f'Validation (n={len(val_idx):,})',
                  edgecolors='white', linewidths=0.5, rasterized=use_rasterized)
        
        ax.scatter(coords_2d[test_idx, 0], coords_2d[test_idx, 1],
                  c=colors['test'], marker=markers['test'], s=sizes['test'],
                  alpha=alphas['test'], label=f'Test (n={len(test_idx):,})',
                  edgecolors='white', linewidths=0.5, rasterized=use_rasterized)
    
    def _plot_clusters(self, ax, coords_2d, cluster_labels, train_idx, val_idx, test_idx, use_rasterized: bool = False):
        """
        Plot cluster assignments with split overlays.
        
        Args:
            use_rasterized: Rasterize scatter plots for faster rendering
        """
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Use a colormap for clusters
        cmap = plt.cm.get_cmap('tab20' if n_clusters <= 20 else 'hsv')
        
        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            color = cmap(i / max(n_clusters, 1))
            
            # Separate by split
            train_cluster = cluster_mask & np.isin(np.arange(len(cluster_labels)), train_idx)
            val_cluster = cluster_mask & np.isin(np.arange(len(cluster_labels)), val_idx)
            test_cluster = cluster_mask & np.isin(np.arange(len(cluster_labels)), test_idx)
            
            # Plot with different markers for each split
            label = f'Cluster {cluster_id}' if cluster_id >= 0 else 'Noise'
            
            if np.any(train_cluster):
                ax.scatter(coords_2d[train_cluster, 0], coords_2d[train_cluster, 1],
                          c=[color], marker='o', s=30, alpha=0.6,
                          edgecolors='white', linewidths=0.5, rasterized=use_rasterized)
            
            if np.any(val_cluster):
                ax.scatter(coords_2d[val_cluster, 0], coords_2d[val_cluster, 1],
                          c=[color], marker='s', s=50, alpha=0.8,
                          edgecolors='white', linewidths=0.5, rasterized=use_rasterized)
            
            if np.any(test_cluster):
                ax.scatter(coords_2d[test_cluster, 0], coords_2d[test_cluster, 1],
                          c=[color], marker='^', s=50, alpha=0.8,
                          edgecolors='white', linewidths=0.5,
                          label=label, rasterized=use_rasterized)
        
        # Add legend for markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=6, alpha=0.6, label='Train', markeredgewidth=0.5, markeredgecolor='white'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=8, alpha=0.8, label='Val', markeredgewidth=0.5, markeredgecolor='white'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=8, alpha=0.8, label='Test', markeredgewidth=0.5, markeredgecolor='white')
        ]
        
        # Add legend in two parts: markers and clusters
        first_legend = ax.legend(handles=legend_elements, loc='upper left', title='Split')
        ax.add_artist(first_legend)
    
    def plot_multi_view_comparison(self,
                                  protein_embeddings: np.ndarray,
                                  ligand_embeddings: np.ndarray,
                                  train_idx: np.ndarray,
                                  val_idx: np.ndarray,
                                  test_idx: np.ndarray,
                                  title: str = "Multi-View Stratification",
                                  figsize: Tuple[int, int] = (20, 6),
                                  save_path: Optional[str] = None,
                                  show: bool = True,
                                  dpi: int = 150) -> plt.Figure:
        """
        Create side-by-side visualization of protein and ligand spaces.
        
        Automatically downsamples if dataset exceeds max_samples.
        
        Args:
            protein_embeddings: Protein embedding matrix
            ligand_embeddings: Ligand embedding matrix
            train_idx: Training set indices
            val_idx: Validation set indices
            test_idx: Test set indices
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save figure
            show: Whether to display the plot
            dpi: DPI for saved figure
            
        Returns:
            Matplotlib figure
        """
        n_samples = len(protein_embeddings)
        
        # Downsample if necessary
        if n_samples > self.max_samples:
            self.logger.info(f"Dataset has {n_samples:,} samples, downsampling to {self.max_samples:,}")
            
            indices_dict = {'train': train_idx, 'val': val_idx, 'test': test_idx}
            
            # Downsample protein
            protein_embeddings, indices_dict_p = self._stratified_downsample(
                protein_embeddings, indices_dict, self.max_samples
            )
            
            # Downsample ligand with same indices
            all_idx = np.concatenate([
                indices_dict['train'], 
                indices_dict['val'], 
                indices_dict['test']
            ])
            ligand_embeddings = ligand_embeddings[all_idx]
            
            train_idx = indices_dict_p['train']
            val_idx = indices_dict_p['val']
            test_idx = indices_dict_p['test']
            
            title = f"{title}\n(Downsampled to {self.max_samples:,} samples)"
        
        # Reduce both to 2D
        protein_2d = self.reduce_dimensions(protein_embeddings)
        ligand_2d = self.reduce_dimensions(ligand_embeddings)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Protein space
        self._plot_splits(ax1, protein_2d, train_idx, val_idx, test_idx)
        ax1.set_title(f"{title}\nProtein Space")
        ax1.set_xlabel(f"{self.method.upper()} Component 1")
        ax1.set_ylabel(f"{self.method.upper()} Component 2")
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Ligand space
        self._plot_splits(ax2, ligand_2d, train_idx, val_idx, test_idx)
        ax2.set_title(f"{title}\nLigand Space")
        ax2.set_xlabel(f"{self.method.upper()} Component 1")
        ax2.set_ylabel(f"{self.method.upper()} Component 2")
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Combined space (concatenated)
        combined = np.concatenate([protein_embeddings, ligand_embeddings], axis=1)
        combined_2d = self.reduce_dimensions(combined)
        self._plot_splits(ax3, combined_2d, train_idx, val_idx, test_idx)
        ax3.set_title(f"{title}\nCombined Space")
        ax3.set_xlabel(f"{self.method.upper()} Component 1")
        ax3.set_ylabel(f"{self.method.upper()} Component 2")
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            self.logger.info(f"Saved multi-view visualization to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)  # Free memory if not showing
        
        return fig
