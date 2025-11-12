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

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import logging

from build.core.config import BuildConfig
from build.stratification.stratifier import Stratifier
from build.pipeline.split_indices import SplitIndices


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
        enable_fallback: Whether to fallback to random splitting on errors
    
    Example:
        >>> config = BuildConfig()
        >>> manager = StratificationManager(config)
        >>> splits = manager.stratify(protein_emb, ligand_emb, labels)
        >>> manager.save_splits('results/splits.npz')
    """
    
    def __init__(
        self,
        config: BuildConfig,
        clustering_algorithm: str = 'kmeans',
        protein_weight: float = 0.6,
        ligand_weight: float = 0.4,
        random_state: Optional[int] = None,
        enable_fallback: bool = True
    ):
        """
        Initialize StratificationManager.
        
        Args:
            config: BuildConfig instance
            clustering_algorithm: Algorithm for clustering
            protein_weight: Weight for protein similarity (0-1)
            ligand_weight: Weight for ligand similarity (0-1)
            random_state: Random seed for reproducibility
            enable_fallback: Whether to fallback to random splitting on errors
        """
        self.config = config
        self.clustering_algorithm = clustering_algorithm
        self.protein_weight = protein_weight
        self.ligand_weight = ligand_weight
        self.random_state = random_state if random_state is not None else config.get('random_state', 42)
        self.enable_fallback = enable_fallback
        
        # Internal state
        self._cached_splits: Optional[SplitIndices] = None
        self._stratifier: Optional[Stratifier] = None
        
        logger.info(f"StratificationManager initialized with algorithm={clustering_algorithm}")
    
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
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> SplitIndices:
        """
        Perform stratified splitting using multi-view approach.
        
        Args:
            protein_embeddings: Protein embeddings (n_samples, protein_dim)
            ligand_embeddings: Ligand embeddings (n_samples, ligand_dim)
            labels: Labels for stratification (n_samples,)
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
        logger.info(f"Stratifying {n_samples} samples (test={test_size}, val={val_size})")
        
        try:
            # Perform stratification
            stratifier = self._get_stratifier()
            train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
                protein_embeddings=protein_embeddings,
                ligand_embeddings=ligand_embeddings,
                labels=labels,
                test_size=test_size,
                val_size=val_size,
                protein_weight=self.protein_weight,
                ligand_weight=self.ligand_weight
            )
            
            # Create metadata
            metadata = {
                'clustering_algorithm': self.clustering_algorithm,
                'protein_weight': self.protein_weight,
                'ligand_weight': self.ligand_weight,
                'test_size': test_size,
                'val_size': val_size,
                'random_state': self.random_state,
                'n_samples': n_samples,
                'fallback_used': False
            }
            
            # Create SplitIndices
            splits = SplitIndices(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                metadata=metadata
            )
            
            # Cache the result
            self._cached_splits = splits
            
            logger.info(
                f"Stratification complete: train={len(train_idx)}, "
                f"val={len(val_idx)}, test={len(test_idx)}"
            )
            
            return splits
        
        except Exception as e:
            if not self.enable_fallback:
                raise
            
            logger.warning(f"Stratification failed: {e}. Falling back to random splitting.")
            return self._fallback_random_split(
                n_samples=n_samples,
                labels=labels,
                test_size=test_size,
                val_size=val_size
            )
    
    def _fallback_random_split(
        self,
        n_samples: int,
        labels: np.ndarray,
        test_size: float,
        val_size: float
    ) -> SplitIndices:
        """
        Fallback to random stratified splitting.
        
        Args:
            n_samples: Total number of samples
            labels: Labels for stratification
            test_size: Proportion of test set
            val_size: Proportion of validation set
        
        Returns:
            SplitIndices with random splits
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=test_size,
            stratify=labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adjusted,
                stratify=labels[train_val_idx],
                random_state=self.random_state
            )
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=np.int32)
        
        # Create metadata
        metadata = {
            'clustering_algorithm': 'random',
            'test_size': test_size,
            'val_size': val_size,
            'random_state': self.random_state,
            'n_samples': n_samples,
            'fallback_used': True
        }
        
        # Create SplitIndices
        splits = SplitIndices(
            train_idx=train_idx.astype(np.int32),
            val_idx=val_idx.astype(np.int32),
            test_idx=test_idx.astype(np.int32),
            metadata=metadata
        )
        
        # Cache the result
        self._cached_splits = splits
        
        logger.info(
            f"Random split complete: train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )
        
        return splits
    
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
