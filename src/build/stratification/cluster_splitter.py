"""
Simple cluster-based splitter following KISS principle.
"""

import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ClusterSplitter:
    """
    Splits samples into train/val/test ensuring cluster integrity.
    
    Single Responsibility: Handle the distribution of clusters into splits.
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Initialize splitter.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        np.random.seed(random_state)
    
    def split(self, cluster_labels: np.ndarray, target_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split samples based on cluster assignments.
        
        Args:
            cluster_labels: Cluster assignment for each sample
            target_labels: Target labels for stratification
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Group samples by cluster
        clusters = self._group_by_cluster(cluster_labels)
        
        # Split each cluster proportionally
        train_idx, val_idx, test_idx = [], [], []
        
        for cluster_id, indices in clusters.items():
            train, val, test = self._split_cluster(indices, target_labels)
            train_idx.extend(train)
            val_idx.extend(val)
            test_idx.extend(test)
        
        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    
    def _group_by_cluster(self, cluster_labels: np.ndarray) -> dict:
        """Group sample indices by their cluster."""
        clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(idx)
        return clusters
    
    def _split_cluster(self, indices: List[int], target_labels: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """
        Split a single cluster into train/val/test.
        
        Strategy:
        - 1 sample: assign to train
        - 2 samples: train + (val or test)
        - 3+ samples: stratified split
        """
        n = len(indices)
        
        if n == 1:
            return indices, [], []
        
        if n == 2:
            return [indices[0]], [], [indices[1]]
        
        # 3+ samples: use stratified split
        return self._stratified_split_cluster(indices, target_labels)
    
    def _stratified_split_cluster(self, indices: List[int], target_labels: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        """Stratified split for clusters with 3+ samples."""
        labels = target_labels[indices]
        
        # Encode non-numeric labels
        if labels.dtype.kind in ['U', 'S', 'O']:
            labels = LabelEncoder().fit_transform(labels)
        
        try:
            # Split: test first
            train_val, test = train_test_split(
                indices, 
                test_size=self.test_size, 
                stratify=labels,
                random_state=self.random_state
            )
            
            # Split: val from remaining
            if self.val_size > 0 and len(train_val) > 1:
                val_ratio = self.val_size / (1 - self.test_size)
                train_val_labels = target_labels[train_val]
                
                if train_val_labels.dtype.kind in ['U', 'S', 'O']:
                    train_val_labels = LabelEncoder().fit_transform(train_val_labels)
                
                train, val = train_test_split(
                    train_val,
                    test_size=val_ratio,
                    stratify=train_val_labels,
                    random_state=self.random_state
                )
                return list(train), list(val), list(test)
            
            return list(train_val), [], list(test)
            
        except ValueError:
            # Fallback: simple split without stratification
            return self._simple_split_cluster(indices)
    
    def _simple_split_cluster(self, indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """Simple proportional split without stratification."""
        n = len(indices)
        n_test = max(1, int(n * self.test_size))
        n_val = max(1, int(n * self.val_size))
        
        # Shuffle for randomness
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        
        test = shuffled[:n_test]
        val = shuffled[n_test:n_test + n_val]
        train = shuffled[n_test + n_val:]
        
        return list(train), list(val), list(test)
