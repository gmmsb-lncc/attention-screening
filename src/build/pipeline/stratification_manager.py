"""
StratificationManager: Orchestrates stratification for train/val/test splits.

This module provides a high-level interface for performing stratified dataset
splitting using the existing stratification module. It handles caching, persistence,
and fallback mechanisms.

Design Principles:
- Single Responsibility: Manages stratification workflow only
- Open/Closed: Easy to extend with new splitting strategies
- Dependency Inversion: Depends on Stratifier abstraction
"""

# Fix OpenMP conflict between FAISS and other libraries (PyTorch, sklearn, scipy)
# This MUST be set before ANY import that might use OpenMP
# Common on macOS where multiple copies of libomp can be linked
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import FAISS FIRST to avoid OpenMP conflicts
# FAISS uses its own OpenMP runtime, import before numpy/scipy/sklearn
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import logging

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from src.build.core.config import BuildConfig
from src.build.stratification.stratifier import Stratifier
from src.build.stratification.cluster_analyzer import ClusterAnalyzer
from src.build.pipeline.split_indices import SplitIndices


logger = logging.getLogger(__name__)


class StratificationManager:
    """
    Manager for dataset stratification operations.
    
    Provides high-level interface for:
    - Performing multi-view stratified splitting
    - Caching splits for reuse
    - Saving/loading splits for reproducibility
    - Fallback to random splitting on errors
    
    Attributes:
        config: BuildConfig instance
        clustering_algorithm: Algorithm for clustering ('kmeans', 'hierarchical', etc.)
        protein_weight: Weight for protein similarity (0-1)
        ligand_weight: Weight for ligand similarity (0-1)
        random_state: Random seed for reproducibility
        enable_fallback: Whether to fallback to label-based splitting on errors
    
    Example:
        >>> config = BuildConfig()
        >>> manager = StratificationManager(config)
        >>> splits = manager.stratify(protein_emb, ligand_emb, labels)
        >>> manager.save_splits('results/splits.npz')
    """
    
    def __init__(
        self,
        config: BuildConfig,
        protein_weight: float = 0.6,
        ligand_weight: float = 0.4,
        random_state: Optional[int] = None,
        enable_fallback: bool = True
    ):
        """
        Initialize StratificationManager.
        
        Uses K-means++ clustering (Arthur & Vassilvitskii, 2007) for robust
        centroid initialization. K-means++ provides O(log k) competitive ratio
        guarantees and is the gold standard for scientific applications.
        
        Implementation uses MiniBatchKMeans for computational efficiency while
        maintaining K-means++ initialization benefits.
        
        References:
            Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages 
            of careful seeding. SODA '07: Proceedings of the eighteenth annual 
            ACM-SIAM symposium on Discrete algorithms.
        
        Args:
            config: BuildConfig instance
            protein_weight: Weight for protein similarity (0-1)
            ligand_weight: Weight for ligand similarity (0-1)
            random_state: Random seed for reproducibility
            enable_fallback: Whether to fallback to label-based splitting on errors
        """
        self.config = config
        self.clustering_algorithm = 'kmeans++'  # Scientifically robust initialization
        self.protein_weight = protein_weight
        self.ligand_weight = ligand_weight
        self.random_state = random_state if random_state is not None else config.get('random_state', 42)
        self.enable_fallback = enable_fallback
        
        # Internal state
        self._cached_splits: Optional[SplitIndices] = None
        self._stratifier: Optional[Stratifier] = None
        
        logger.info(f"StratificationManager initialized with algorithm={self.clustering_algorithm}")
    
    def _get_stratifier(self) -> Stratifier:
        """
        Get or create Stratifier instance.
        
        Returns:
            Stratifier instance configured with current settings
        """
        if self._stratifier is None:
            self._stratifier = Stratifier(
                config=self.config,
                clustering_algorithm=self.clustering_algorithm,
                protein_weight=self.protein_weight,
                ligand_weight=self.ligand_weight,
                random_state=self.random_state
            )
        return self._stratifier
    
    def stratify(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.1,
        val_size: float = 0.1
    ) -> SplitIndices:
        """
        Perform stratified splitting using appropriate strategy based on dataset size.
        
        Strategy selection:
        - Small datasets (<50K): Multi-view clustering on embeddings
        - Large datasets (>=50K): Label-based stratification (memory efficient)
        
        This ensures scientifically valid splits using FAISS K-means clustering
        which is both scalable O(n) and preserves chemical similarity relationships.
        
        Args:
            protein_embeddings: Protein embeddings (n_samples, protein_dim)
            ligand_embeddings: Ligand embeddings (n_samples, ligand_dim)
            labels: Labels for stratification (n_samples,) or (n_samples, n_cols)
            test_size: Proportion of test set (0-1)
            val_size: Proportion of validation set (0-1)
        
        Returns:
            SplitIndices with train/val/test indices
        
        Raises:
            ValueError: If embeddings and labels have different lengths
        
        Example:
            >>> splits = manager.stratify(protein_emb, ligand_emb, labels)
            >>> print(f"Train: {len(splits.train_idx)}, Test: {len(splits.test_idx)}")
        """
        # Validate inputs
        if len(protein_embeddings) != len(ligand_embeddings):
            raise ValueError(
                f"Protein and ligand embeddings must have same length: "
                f"{len(protein_embeddings)} vs {len(ligand_embeddings)}"
            )
        if len(protein_embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and labels must have same length: "
                f"{len(protein_embeddings)} vs {len(labels)}"
            )
        
        n_samples = len(labels)
        logger.info(f"Stratifying {n_samples:,} samples (test={test_size}, val={val_size})")
        
        # Unified strategy: K-means++ clustering for ALL dataset sizes
        # 
        # Scientific rationale:
        # - K-means++ (Arthur & Vassilvitskii, 2007) provides O(log k) competitive
        #   ratio guarantees for centroid initialization
        # - Preserves chemical similarity by clustering on embeddings
        # - Prevents data leakage by keeping similar compounds in same split
        # - MiniBatchKMeans provides O(n) complexity for large datasets
        # - Adaptive cluster count: sqrt(n) bounded between 10-1000
        #
        return self._stratify_by_kmeans_pp(
            protein_embeddings=protein_embeddings,
            ligand_embeddings=ligand_embeddings,
            labels=labels,
            test_size=test_size,
            val_size=val_size
        )
    
    def _stratify_by_kmeans_pp(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        val_size: float
    ) -> SplitIndices:
        """
        Stratification using K-means++ clustering (scikit-learn).
        
        K-means++ is the scientifically preferred method because:
        1. Theoretical guarantees: O(log k) competitive ratio (Arthur & Vassilvitskii, 2007)
        2. Reproducibility: deterministic given random seed
        3. Wide adoption: standard in ML literature and peer-reviewed publications
        4. Better convergence: typically requires fewer iterations than random init
        
        Uses MiniBatchKMeans for computational efficiency with large datasets
        while maintaining K-means++ initialization benefits.
        
        Args:
            protein_embeddings: Protein embeddings (n_samples, protein_dim)
            ligand_embeddings: Ligand embeddings (n_samples, ligand_dim)
            labels: Labels for stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set
        
        Returns:
            SplitIndices with cluster-aware stratified splits
        """
        n_samples = len(labels)
        
        try:
            # Combine embeddings with weighting
            combined = np.concatenate([
                protein_embeddings * self.protein_weight,
                ligand_embeddings * self.ligand_weight
            ], axis=1).astype(np.float32)
            
            # L2 normalize for cosine similarity behavior
            combined = normalize(combined, norm='l2', axis=1)
            
            # Adaptive cluster count: sqrt(n), bounded [10, 1000]
            n_clusters = min(1000, max(10, int(np.sqrt(n_samples))))
            
            logger.info(
                f"K-means++ clustering: {n_samples:,} samples -> {n_clusters} clusters"
            )
            
            # MiniBatchKMeans with K-means++ initialization
            # - init='k-means++': Arthur & Vassilvitskii (2007) initialization
            # - batch_size=1024: efficient mini-batch processing
            # - n_init='auto': sklearn chooses optimal number of initializations
            # - max_iter=100: sufficient for convergence
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                init='k-means++',
                batch_size=min(1024, n_samples),
                n_init='auto',
                max_iter=100,
                random_state=self.random_state,
                verbose=0
            )
            cluster_labels = kmeans.fit_predict(combined)
            
            logger.info(
                f"Clustering complete: {len(np.unique(cluster_labels))} clusters formed, "
                f"inertia={kmeans.inertia_:.2f}"
            )
            
            # Split by clusters
            return self._split_by_clusters(
                cluster_labels=cluster_labels,
                labels=labels,
                test_size=test_size,
                val_size=val_size,
                n_samples=n_samples,
                strategy_name='kmeans++'
            )
            
        except Exception as e:
            logger.warning(f"K-means++ clustering failed: {e}. Falling back to label-based.")
            return self._stratify_by_labels(
                n_samples=n_samples,
                labels=labels,
                test_size=test_size,
                val_size=val_size
            )
    
    def _stratify_by_labels(
        self,
        n_samples: int,
        labels: np.ndarray,
        test_size: float,
        val_size: float
    ) -> SplitIndices:
        """
        Stratified splitting based on labels only (no clustering).
        
        This is the scientifically appropriate method for large datasets where:
        1. Distance matrix computation would be infeasible (O(n²) memory)
        2. Binary activity labels provide meaningful stratification
        
        For drug discovery, stratifying by activity class ensures balanced
        representation of actives/inactives in each split.
        
        Args:
            n_samples: Total number of samples
            labels: Labels array - can be 1D binary or multi-column metadata
            test_size: Proportion of test set
            val_size: Proportion of validation set
        
        Returns:
            SplitIndices with stratified splits
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Using label-based stratification for {n_samples:,} samples")
        
        # Extract binary labels for stratification
        stratify_labels = self._extract_binary_labels(labels)
        
        strategy_used = 'label_stratified' if stratify_labels is not None else 'pure_random'
        
        try:
            if stratify_labels is not None:
                # Verify stratification is possible
                from collections import Counter
                label_counts = Counter(stratify_labels)
                min_count = min(label_counts.values())
                
                # Need enough samples per class for both splits
                min_needed = max(2, int(np.ceil(n_samples * max(test_size, val_size) / len(label_counts))))
                
                if min_count < min_needed:
                    logger.warning(
                        f"Insufficient samples for stratification: min class has {min_count}, "
                        f"need {min_needed}. Using random split."
                    )
                    stratify_labels = None
                    strategy_used = 'pure_random'
            
            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                np.arange(n_samples),
                test_size=test_size,
                stratify=stratify_labels,
                random_state=self.random_state
            )
            
            # Second split: train vs val
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)
                stratify_train = stratify_labels[train_val_idx] if stratify_labels is not None else None
                
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_adjusted,
                    stratify=stratify_train,
                    random_state=self.random_state
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([], dtype=np.int32)
                
        except Exception as e:
            # Ultimate fallback: pure random
            logger.warning(f"Label stratification failed: {e}. Using pure random split.")
            strategy_used = 'pure_random_fallback'
            
            train_val_idx, test_idx = train_test_split(
                np.arange(n_samples),
                test_size=test_size,
                stratify=None,
                random_state=self.random_state
            )
            
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_adjusted,
                    stratify=None,
                    random_state=self.random_state
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([], dtype=np.int32)
        
        # Log class distribution if stratified
        if stratify_labels is not None and strategy_used == 'label_stratified':
            self._log_split_distribution(stratify_labels, train_idx, val_idx, test_idx)
        
        # Create metadata
        metadata = {
            'clustering_algorithm': strategy_used,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': self.random_state,
            'n_samples': n_samples,
            'fallback_used': True,
            'strategy': strategy_used
        }
        
        # Create SplitIndices
        splits = SplitIndices(
            train_idx=np.asarray(train_idx, dtype=np.int32),
            val_idx=np.asarray(val_idx, dtype=np.int32),
            test_idx=np.asarray(test_idx, dtype=np.int32),
            metadata=metadata
        )
        
        # Cache the result
        self._cached_splits = splits
        
        logger.info(
            f"Label-based split complete ({strategy_used}): "
            f"train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}"
        )
        
        return splits
    
    def _rebalance_clusters(
        self,
        test_clusters: list,
        val_clusters: list,
        train_clusters: list,
        cluster_sizes: dict,
        target_test: int,
        target_val: int,
        target_train: int
    ) -> tuple:
        """
        Rebalance cluster assignments to achieve target sample proportions.
        
        Uses greedy swapping to move clusters between splits when proportions
        are significantly off target (>5% deviation).
        
        Args:
            test_clusters: Current test cluster list
            val_clusters: Current validation cluster list
            train_clusters: Current train cluster list
            cluster_sizes: Dict mapping cluster_id -> sample count
            target_test: Target number of test samples
            target_val: Target number of validation samples
            target_train: Target number of train samples
            
        Returns:
            Tuple of (test_clusters, val_clusters, train_clusters) after rebalancing
        """
        def get_counts():
            return (
                sum(cluster_sizes[c] for c in test_clusters),
                sum(cluster_sizes[c] for c in val_clusters),
                sum(cluster_sizes[c] for c in train_clusters)
            )
        
        max_iterations = 50  # Prevent infinite loops
        tolerance = 0.05  # 5% tolerance
        
        for _ in range(max_iterations):
            current_test, current_val, current_train = get_counts()
            total = current_test + current_val + current_train
            
            # Check if within tolerance
            test_ratio = current_test / total
            val_ratio = current_val / total
            train_ratio = current_train / total
            
            target_test_ratio = target_test / total
            target_val_ratio = target_val / total
            target_train_ratio = target_train / total
            
            test_ok = abs(test_ratio - target_test_ratio) <= tolerance
            val_ok = abs(val_ratio - target_val_ratio) <= tolerance
            train_ok = abs(train_ratio - target_train_ratio) <= tolerance
            
            if test_ok and val_ok and train_ok:
                break
            
            # Find the most over-represented and under-represented splits
            deviations = [
                ('test', current_test - target_test, test_clusters),
                ('val', current_val - target_val, val_clusters),
                ('train', current_train - target_train, train_clusters)
            ]
            
            # Sort by deviation (most over-represented first)
            deviations.sort(key=lambda x: x[1], reverse=True)
            over_name, over_dev, over_list = deviations[0]
            under_name, under_dev, under_list = deviations[-1]
            
            if over_dev <= 0 or under_dev >= 0:
                break  # No imbalance to fix
            
            # Find smallest cluster in over-represented split to move
            if over_list:
                smallest_cluster = min(over_list, key=lambda c: cluster_sizes[c])
                over_list.remove(smallest_cluster)
                under_list.append(smallest_cluster)
        
        return test_clusters, val_clusters, train_clusters
    
    def _split_by_clusters(
        self,
        cluster_labels: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        val_size: float,
        n_samples: int,
        strategy_name: str
    ) -> SplitIndices:
        """
        Perform stratified splitting respecting cluster boundaries.
        
        This ensures samples from the same cluster stay together in the same split,
        preventing data leakage from chemically similar compounds.
        
        Args:
            cluster_labels: Cluster assignment for each sample
            labels: Original labels for within-cluster stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set
            n_samples: Total number of samples
            strategy_name: Name of clustering strategy used
        
        Returns:
            SplitIndices with cluster-aware splits
        """
        from sklearn.model_selection import train_test_split
        
        # Get unique clusters and their sizes
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Calculate cluster sizes (number of samples per cluster)
        cluster_sizes = {c: np.sum(cluster_labels == c) for c in unique_clusters}
        total_samples = sum(cluster_sizes.values())
        
        logger.info(f"Splitting {n_clusters} clusters ({total_samples} samples) into train/val/test")
        
        # Target sample counts for exact 80/10/10 split
        target_test = int(total_samples * test_size)
        target_val = int(total_samples * val_size)
        target_train = total_samples - target_test - target_val
        
        # Sort clusters by size (largest first) for greedy assignment
        sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_sizes[c], reverse=True)
        
        # Greedy assignment to achieve target proportions
        # Assign each cluster to the split that needs more samples
        test_clusters = []
        val_clusters = []
        train_clusters = []
        
        current_test = 0
        current_val = 0
        current_train = 0
        
        for cluster in sorted_clusters:
            size = cluster_sizes[cluster]
            
            # Calculate how far each split is from its target
            test_need = target_test - current_test
            val_need = target_val - current_val
            train_need = target_train - current_train
            
            # Assign to the split that:
            # 1. Still needs samples AND
            # 2. Would benefit most from this cluster (relative to its target)
            if test_need > 0 and (current_test + size <= target_test * 1.2):
                # Assign to test if it needs samples and won't exceed by much
                test_clusters.append(cluster)
                current_test += size
            elif val_need > 0 and (current_val + size <= target_val * 1.2):
                # Assign to val
                val_clusters.append(cluster)
                current_val += size
            else:
                # Assign to train (default)
                train_clusters.append(cluster)
                current_train += size
        
        # Rebalance if proportions are significantly off
        # Move clusters from over-represented to under-represented splits
        test_clusters, val_clusters, train_clusters = self._rebalance_clusters(
            test_clusters, val_clusters, train_clusters,
            cluster_sizes, target_test, target_val, target_train
        )
        
        # Convert to arrays
        test_clusters = np.array(test_clusters)
        val_clusters = np.array(val_clusters)
        train_clusters = np.array(train_clusters)
        
        # Convert cluster assignments to sample indices
        train_idx = np.where(np.isin(cluster_labels, train_clusters))[0]
        val_idx = np.where(np.isin(cluster_labels, val_clusters))[0]
        test_idx = np.where(np.isin(cluster_labels, test_clusters))[0]
        
        # Log actual proportions
        actual_train = len(train_idx) / total_samples * 100
        actual_val = len(val_idx) / total_samples * 100
        actual_test = len(test_idx) / total_samples * 100
        logger.info(
            f"Split complete: train={actual_train:.1f}%, val={actual_val:.1f}%, test={actual_test:.1f}%"
        )
        
        # Log split statistics
        binary_labels = self._extract_binary_labels(labels)
        if binary_labels is not None:
            self._log_split_distribution(binary_labels, train_idx, val_idx, test_idx)
        
        # Create metadata
        metadata = {
            'clustering_algorithm': strategy_name,
            'n_clusters': n_clusters,
            'protein_weight': self.protein_weight,
            'ligand_weight': self.ligand_weight,
            'test_size': test_size,
            'val_size': val_size,
            'random_state': self.random_state,
            'n_samples': n_samples,
            'fallback_used': False,
            'strategy': f'cluster_aware_{strategy_name}',
            'train_clusters': len(train_clusters),
            'val_clusters': len(val_clusters) if len(val_clusters) > 0 else 0,
            'test_clusters': len(test_clusters)
        }
        
        # Create SplitIndices
        splits = SplitIndices(
            train_idx=np.asarray(train_idx, dtype=np.int32),
            val_idx=np.asarray(val_idx, dtype=np.int32),
            test_idx=np.asarray(test_idx, dtype=np.int32),
            metadata=metadata
        )
        
        # Cache the result
        self._cached_splits = splits
        
        logger.info(
            f"Cluster-aware split complete: "
            f"train={len(train_idx):,} ({len(train_clusters)} clusters), "
            f"val={len(val_idx):,} ({len(val_clusters) if hasattr(val_clusters, '__len__') else 0} clusters), "
            f"test={len(test_idx):,} ({len(test_clusters)} clusters)"
        )
        
        return splits
    
    def _extract_binary_labels(self, labels: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract binary labels for stratification from various label formats.
        
        Handles:
        - 1D binary arrays (0/1)
        - Multi-column metadata arrays (extracts binary column if present)
        - interaction_labels format: [molregno, kinase, type, standard_value, pchembl_value]
          -> creates binary from pchembl >= 6.0 (active) or standard_value <= 1000 nM
        - String labels (attempts conversion)
        
        Returns:
            1D numpy array of binary labels, or None if extraction fails
        """
        try:
            if labels is None:
                return None
            
            # Handle multi-dimensional labels
            if labels.ndim > 1:
                n_cols = labels.shape[1]
                
                # Check if this is interaction_labels format (5 columns)
                # [molregno, kinase, type, standard_value, pchembl_value]
                if n_cols == 5:
                    return self._create_binary_from_activity_values(labels)
                
                # Check if this is 4-column format without pchembl
                # [molregno, kinase, type, standard_value]
                if n_cols == 4:
                    return self._create_binary_from_standard_value(labels)
                
                # Try to find a binary column (likely activity label)
                for col_idx in range(labels.shape[1]):
                    col = labels[:, col_idx]
                    
                    # Filter out None values before checking unique values
                    try:
                        # Convert to list and filter None values for unique check
                        col_values = [v for v in col if v is not None]
                        if len(col_values) == 0:
                            continue
                        unique = np.unique(col_values)
                    except (TypeError, ValueError):
                        # If comparison fails (mixed types), skip this column
                        continue
                    
                    # Check if it's binary (0/1 or 'active'/'inactive' etc.)
                    if len(unique) == 2:
                        try:
                            # Try numeric conversion
                            binary = np.asarray(col, dtype=np.float32)
                            if set(np.unique(binary)).issubset({0.0, 1.0}):
                                logger.info(f"Using column {col_idx} as binary stratification labels")
                                return binary.astype(np.int32)
                        except (ValueError, TypeError):
                            pass
                
                # No binary column found
                logger.warning(
                    f"Multi-column labels {labels.shape} - no binary column found. "
                    f"Cannot stratify by labels."
                )
                return None
            
            # 1D labels - filter None values before unique check
            try:
                labels_filtered = [v for v in labels if v is not None]
                if len(labels_filtered) == 0:
                    return None
                unique = np.unique(labels_filtered)
            except (TypeError, ValueError):
                # If comparison fails (mixed types), use fallback
                logger.warning("Labels contain incompatible types for stratification")
                return None
            
            # Already binary
            if len(unique) == 2:
                try:
                    # Replace None with a placeholder, then convert
                    labels_clean = np.array([v if v is not None else np.nan for v in labels], dtype=np.float32)
                    # Check if non-NaN values are binary
                    valid_values = labels_clean[~np.isnan(labels_clean)]
                    if set(np.unique(valid_values)).issubset({0.0, 1.0}):
                        # For stratification, treat NaN as the majority class
                        majority = 1.0 if np.sum(valid_values == 1.0) > np.sum(valid_values == 0.0) else 0.0
                        labels_clean = np.nan_to_num(labels_clean, nan=majority)
                        return labels_clean.astype(np.int32)
                except (ValueError, TypeError):
                    pass
            
            # Too many classes for effective stratification
            if len(unique) > 100:
                logger.warning(f"Too many unique labels ({len(unique)}) for stratification")
                return None
            
            # Use labels as-is if hashable
            return labels
            
        except Exception as e:
            logger.warning(f"Failed to extract binary labels: {e}")
            return None
    
    def _create_binary_from_activity_values(self, labels: np.ndarray) -> np.ndarray:
        """
        Create binary labels from interaction_labels format with 5 columns.
        
        Uses pchembl_value (column 4) which should ALWAYS be present.
        pchembl_value is preferred because it normalizes the range of values
        and facilitates calculations (logarithmic scale, typically 3-12).
        
        If pchembl_value is missing, it should have been filled from standard_value
        during label generation. As a fallback, we calculate from standard_value here.
        
        Active: pchembl >= 6.0 (equivalent to IC50 <= 1000 nM)
        
        Args:
            labels: Array with columns [molregno, kinase, type, standard_value, pchembl_value]
            
        Returns:
            Binary labels array (1=active, 0=inactive)
        """
        n_samples = len(labels)
        binary_labels = np.zeros(n_samples, dtype=np.int32)
        
        pchembl_threshold = 6.0  # pChEMBL >= 6.0 means <= 1000 nM
        
        valid_count = 0
        converted_count = 0
        
        for i in range(n_samples):
            pchembl_val = labels[i, 4]
            
            # Use pchembl_value (should always be present)
            if self._is_valid_number(pchembl_val):
                try:
                    pchembl_float = float(pchembl_val)
                    binary_labels[i] = 1 if pchembl_float >= pchembl_threshold else 0
                    valid_count += 1
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Fallback: calculate pchembl from standard_value if missing
            # This should rarely happen as pchembl should be filled during label generation
            standard_val = labels[i, 3]
            if self._is_valid_number(standard_val):
                try:
                    standard_float = float(standard_val)
                    if standard_float > 0:
                        # Convert to pchembl: pchembl = 9 - log10(nM)
                        pchembl_calculated = 9 - np.log10(standard_float)
                        binary_labels[i] = 1 if pchembl_calculated >= pchembl_threshold else 0
                        valid_count += 1
                        converted_count += 1
                        continue
                except (ValueError, TypeError):
                    pass
            
            # No valid value - this should NOT happen according to data requirements
            # Default to inactive (0) but log warning
            binary_labels[i] = 0
            logger.debug(f"Row {i}: No valid pchembl or standard_value, defaulting to inactive")
        
        if converted_count > 0:
            logger.warning(
                f"Had to convert {converted_count} values from standard_value to pchembl. "
                f"Consider ensuring pchembl_value is filled during label generation."
            )
        
        logger.info(f"Created binary labels: {valid_count}/{n_samples} valid from pchembl_value")
        
        active_count = np.sum(binary_labels == 1)
        inactive_count = np.sum(binary_labels == 0)
        logger.info(f"Binary distribution: {active_count} active, {inactive_count} inactive")
        
        return binary_labels
    
    def _create_binary_from_standard_value(self, labels: np.ndarray) -> np.ndarray:
        """
        Create binary labels from 4-column interaction_labels format.
        
        Uses standard_value (column 3) with threshold of 1000 nM.
        
        Args:
            labels: Array with columns [molregno, kinase, type, standard_value]
            
        Returns:
            Binary labels array (1=active, 0=inactive)
        """
        n_samples = len(labels)
        binary_labels = np.zeros(n_samples, dtype=np.int32)
        
        nm_threshold = 1000.0  # 1000 nM = 1 µM
        valid_count = 0
        
        for i in range(n_samples):
            standard_val = labels[i, 3]
            
            if self._is_valid_number(standard_val):
                try:
                    standard_float = float(standard_val)
                    if standard_float > 0:
                        binary_labels[i] = 1 if standard_float <= nm_threshold else 0
                        valid_count += 1
                        continue
                except (ValueError, TypeError):
                    pass
            
            # No valid value - default to inactive (0)
            binary_labels[i] = 0
        
        logger.info(f"Created binary labels from standard_value: {valid_count}/{n_samples} valid")
        
        active_count = np.sum(binary_labels == 1)
        inactive_count = np.sum(binary_labels == 0)
        logger.info(f"Binary distribution: {active_count} active, {inactive_count} inactive")
        
        return binary_labels
    
    def _is_valid_number(self, value) -> bool:
        """Check if value is a valid number (not None, NaN, or invalid string)."""
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        if isinstance(value, str):
            val_lower = value.lower().strip()
            if val_lower in ('none', 'nan', 'null', '', 'na', 'n/a'):
                return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _log_split_distribution(
        self,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> None:
        """Log class distribution in each split for verification."""
        from collections import Counter
        
        try:
            train_dist = Counter(labels[train_idx])
            val_dist = Counter(labels[val_idx]) if len(val_idx) > 0 else {}
            test_dist = Counter(labels[test_idx])
            
            logger.info("Split class distribution:")
            logger.info(f"  Train: {dict(train_dist)}")
            if val_dist:
                logger.info(f"  Val:   {dict(val_dist)}")
            logger.info(f"  Test:  {dict(test_dist)}")
        except Exception:
            pass  # Non-critical logging
    
    def get_splits(self) -> SplitIndices:
        """
        Get cached splits.
        
        Returns:
            Cached SplitIndices
        
        Raises:
            RuntimeError: If stratify() has not been called yet
        
        Example:
            >>> splits = manager.stratify(protein_emb, ligand_emb, labels)
            >>> same_splits = manager.get_splits()  # Returns cached version
        """
        if self._cached_splits is None:
            raise RuntimeError(
                "No splits available. Call stratify() first or load_splits()."
            )
        return self._cached_splits
    
    def save_splits(self, filepath: str) -> None:
        """
        Save current splits to file.
        
        Args:
            filepath: Path where to save splits (.npz format)
        
        Raises:
            RuntimeError: If no splits available to save
        
        Example:
            >>> manager.save_splits('results/splits.npz')
        """
        if self._cached_splits is None:
            raise RuntimeError(
                "No splits available to save. Call stratify() first."
            )
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self._cached_splits.save(str(filepath))
        logger.info(f"Splits saved to: {filepath}")
    
    def load_splits(self, filepath: str) -> SplitIndices:
        """
        Load splits from file.
        
        Args:
            filepath: Path to splits file (.npz format)
        
        Returns:
            Loaded SplitIndices
        
        Raises:
            FileNotFoundError: If file doesn't exist
        
        Example:
            >>> splits = manager.load_splits('results/splits.npz')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Splits file not found: {filepath}")
        
        splits = SplitIndices.load(str(filepath))
        
        # Cache the loaded splits
        self._cached_splits = splits
        
        logger.info(f"Splits loaded from: {filepath}")
        logger.info(
            f"Loaded splits: train={len(splits.train_idx)}, "
            f"val={len(splits.val_idx)}, test={len(splits.test_idx)}"
        )
        
        return splits
    
    def clear_cache(self) -> None:
        """Clear cached splits."""
        self._cached_splits = None
        logger.debug("Splits cache cleared")
    
    def generate_cluster_visualization(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        output_dir: str,
        prefix: str = 'cluster'
    ) -> Dict[str, Any]:
        """
        Generate cluster visualizations using PCA.
        
        Args:
            protein_embeddings: Protein embeddings (n_samples, protein_dim)
            ligand_embeddings: Ligand embeddings (n_samples, ligand_dim)
            labels: Original labels for coloring
            output_dir: Directory to save visualizations
            prefix: Prefix for output files
        
        Returns:
            Dictionary with paths to generated files and metrics
        
        Example:
            >>> result = manager.generate_cluster_visualization(
            ...     protein_emb, ligand_emb, labels, 'results/stratification'
            ... )
        """
        if self._stratifier is None or self._stratifier.cluster_labels is None:
            raise RuntimeError(
                "No cluster labels available. Call stratify() first."
            )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get cluster labels from stratifier
        cluster_labels = self._stratifier.cluster_labels
        
        # Concatenate embeddings
        combined_embeddings = np.concatenate(
            [protein_embeddings, ligand_embeddings], axis=1
        )
        
        # Initialize ClusterAnalyzer
        analyzer = ClusterAnalyzer(self.config)
        
        result = {}
        
        try:
            # Calculate clustering metrics
            metrics = analyzer.calculate_clustering_metrics(
                combined_embeddings, cluster_labels
            )
            result['metrics'] = metrics
            logger.info(
                f"Cluster metrics: silhouette={metrics.get('silhouette_score', 'N/A'):.4f}, "
                f"n_clusters={metrics.get('n_clusters', 0)}"
            )
            
            # Analyze cluster distribution
            distribution = analyzer.analyze_cluster_distribution(cluster_labels, labels)
            result['distribution'] = distribution
            
            # Save cluster labels
            cluster_labels_path = output_path / f'{prefix}_labels.npy'
            np.save(cluster_labels_path, cluster_labels)
            result['cluster_labels_path'] = str(cluster_labels_path)
            
            # Generate main visualization
            pca_plot_path = output_path / f'{prefix}_pca.png'
            analyzer.visualize_clusters(
                embeddings=combined_embeddings,
                cluster_labels=cluster_labels,
                labels=labels,
                output_path=str(pca_plot_path)
            )
            result['pca_visualization_path'] = str(pca_plot_path)
            
            logger.info(f"Cluster visualizations saved to: {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate cluster visualization: {e}")
            result['error'] = str(e)
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        cached = "cached" if self._cached_splits is not None else "no cache"
        return (
            f"StratificationManager("
            f"algorithm={self.clustering_algorithm}, "
            f"protein_weight={self.protein_weight:.2f}, "
            f"ligand_weight={self.ligand_weight:.2f}, "
            f"{cached}"
            f")"
        )
