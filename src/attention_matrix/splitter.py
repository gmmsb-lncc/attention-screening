"""
Data Splitter for Attention Matrix Module.

Single Responsibility: Data splitting strategies only.
Implements leakage-aware split using protein clustering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import logging
import json


logger = logging.getLogger(__name__)


class LeakageAwareSplitter:
    """
    Data splitter that prevents protein-based data leakage.
    
    Uses hierarchical clustering on protein embeddings to ensure
    that similar proteins are not split across train/val/test sets.
    
    Args:
        n_clusters: Number of protein clusters (None = auto)
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.protein_clusters: Optional[Dict[str, int]] = None
        self.split_metadata: Dict[str, Any] = {}
    
    def _load_protein_embeddings(
        self,
        protein_ids: np.ndarray,
        protein_dir: Path
    ) -> np.ndarray:
        """Load mean-pooled protein embeddings for clustering."""
        embeddings = []
        valid_ids = []
        
        for pid in protein_ids:
            path = protein_dir / f"{pid}.npy"
            if path.exists():
                emb = np.load(path)
                embeddings.append(emb.mean(axis=0))
                valid_ids.append(pid)
        
        return np.array(embeddings), np.array(valid_ids)
    
    def _cluster_proteins(
        self,
        protein_embeddings: np.ndarray,
        protein_ids: np.ndarray
    ) -> Dict[str, int]:
        """Cluster proteins using hierarchical clustering."""
        # Normalize embeddings
        embeddings_norm = normalize(protein_embeddings)
        
        # Determine number of clusters
        n_proteins = len(protein_ids)
        if self.n_clusters is None:
            # Auto: ~20% of proteins as clusters (min 10, max 100)
            self.n_clusters = max(10, min(100, n_proteins // 5))
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings_norm)
        
        return {pid: int(label) for pid, label in zip(protein_ids, labels)}
    
    def _allocate_clusters(
        self,
        cluster_sizes: Dict[int, int],
        total_samples: int
    ) -> Tuple[list, list, list]:
        """Allocate clusters to train/val/test sets."""
        target_train = (1 - self.test_size - self.val_size) * total_samples
        target_val = self.val_size * total_samples
        target_test = self.test_size * total_samples
        
        # Sort clusters by size (largest first for better distribution)
        sorted_clusters = sorted(
            cluster_sizes.keys(),
            key=lambda x: cluster_sizes[x],
            reverse=True
        )
        
        train_clusters, val_clusters, test_clusters = [], [], []
        train_size, val_size, test_size = 0, 0, 0
        
        for cluster in sorted_clusters:
            size = cluster_sizes[cluster]
            
            # Allocate to set that needs more samples
            train_need = target_train - train_size
            val_need = target_val - val_size
            test_need = target_test - test_size
            
            if train_need >= val_need and train_need >= test_need:
                train_clusters.append(cluster)
                train_size += size
            elif val_need >= test_need:
                val_clusters.append(cluster)
                val_size += size
            else:
                test_clusters.append(cluster)
                test_size += size
        
        return train_clusters, val_clusters, test_clusters
    
    def split(
        self,
        df: pd.DataFrame,
        protein_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform leakage-aware split.
        
        Args:
            df: DataFrame with seq_id column
            protein_dir: Directory with protein embeddings
            
        Returns:
            Tuple of (train_idx, val_idx, test_idx)
        """
        logger.info("Performing leakage-aware split...")
        
        # Get unique proteins
        unique_proteins = df['seq_id'].astype(str).unique()
        logger.info(f"  Unique proteins: {len(unique_proteins)}")
        
        # Load and cluster protein embeddings
        embeddings, valid_ids = self._load_protein_embeddings(
            unique_proteins, protein_dir
        )
        logger.info(f"  Loaded embeddings: {len(valid_ids)}")
        
        # Cluster proteins
        self.protein_clusters = self._cluster_proteins(embeddings, valid_ids)
        logger.info(f"  Protein clusters: {self.n_clusters}")
        
        # Assign cluster to each sample
        df = df.copy()
        df['protein_cluster'] = df['seq_id'].astype(str).map(self.protein_clusters)
        
        # Remove samples with missing clusters
        df = df.dropna(subset=['protein_cluster'])
        df['protein_cluster'] = df['protein_cluster'].astype(int)
        
        # Calculate samples per cluster
        cluster_sizes = df.groupby('protein_cluster').size().to_dict()
        
        # Allocate clusters
        train_clusters, val_clusters, test_clusters = self._allocate_clusters(
            cluster_sizes, len(df)
        )
        
        # Get indices
        train_idx = df[df['protein_cluster'].isin(train_clusters)].index.values
        val_idx = df[df['protein_cluster'].isin(val_clusters)].index.values
        test_idx = df[df['protein_cluster'].isin(test_clusters)].index.values
        
        # Verify no protein overlap
        train_proteins = set(df.loc[train_idx, 'seq_id'].astype(str))
        val_proteins = set(df.loc[val_idx, 'seq_id'].astype(str))
        test_proteins = set(df.loc[test_idx, 'seq_id'].astype(str))
        
        train_test_overlap = train_proteins & test_proteins
        train_val_overlap = train_proteins & val_proteins
        val_test_overlap = val_proteins & test_proteins
        
        # Store metadata
        self.split_metadata = {
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'n_protein_clusters': self.n_clusters,
            'n_clusters_train': len(train_clusters),
            'n_clusters_val': len(val_clusters),
            'n_clusters_test': len(test_clusters),
            'train_test_overlap': len(train_test_overlap),
            'train_val_overlap': len(train_val_overlap),
            'val_test_overlap': len(val_test_overlap)
        }
        
        logger.info(f"  Train: {len(train_idx)} samples ({len(train_clusters)} clusters)")
        logger.info(f"  Val: {len(val_idx)} samples ({len(val_clusters)} clusters)")
        logger.info(f"  Test: {len(test_idx)} samples ({len(test_clusters)} clusters)")
        logger.info(f"  Train-Test protein overlap: {len(train_test_overlap)}")
        
        return train_idx, val_idx, test_idx
    
    def save_split(
        self,
        output_dir: Path,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray
    ):
        """Save split indices and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'train_idx.npy', train_idx)
        np.save(output_dir / 'val_idx.npy', val_idx)
        np.save(output_dir / 'test_idx.npy', test_idx)
        
        with open(output_dir / 'split_metadata.json', 'w') as f:
            json.dump(self.split_metadata, f, indent=2)
        
        logger.info(f"  Split saved to: {output_dir}")
    
    @staticmethod
    def load_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load pre-computed split indices."""
        split_dir = Path(split_dir)
        
        train_idx = np.load(split_dir / 'train_idx.npy', allow_pickle=True)
        val_idx = np.load(split_dir / 'val_idx.npy', allow_pickle=True)
        test_idx = np.load(split_dir / 'test_idx.npy', allow_pickle=True)
        
        return train_idx, val_idx, test_idx


class SimpleSplitter:
    """
    Simple stratified splitter (for comparison/baseline).
    
    Standard train_test_split with stratification by activity label.
    May have data leakage if similar proteins appear in different sets.
    """
    
    def __init__(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split(
        self,
        df: pd.DataFrame,
        stratify_col: str = 'is_active'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform simple stratified split."""
        indices = df.index.values
        labels = df[stratify_col].values
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_ratio = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=labels[train_val_idx]
        )
        
        return train_idx, val_idx, test_idx
