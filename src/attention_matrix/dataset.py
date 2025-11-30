"""
Dataset classes for Attention Matrix Module.

Single Responsibility: Data loading and preprocessing only.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd


class ProteinLigandDataset(Dataset):
    """
    Dataset for protein-ligand affinity prediction using matrix embeddings.
    
    Loads pre-computed matrix embeddings from .npy files and handles
    padding/truncation to fixed sequence lengths.
    
    Args:
        data: DataFrame with columns ['seq_id', 'molecule_chembl_id', 'pchembl_value']
        protein_dir: Directory containing protein embeddings (seq_id.npy)
        ligand_dir: Directory containing ligand embeddings (molecule_chembl_id.npy)
        activity_col: Column name for activity values (default: 'pchembl_value')
        max_protein_len: Maximum protein sequence length
        max_ligand_len: Maximum ligand token length
        protein_dim: Dimension of protein embeddings
        ligand_dim: Dimension of ligand embeddings
        threshold: pChEMBL threshold for binary classification
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        protein_dir: str,
        ligand_dir: str,
        activity_col: str = 'pchembl_value',
        max_protein_len: int = 256,
        max_ligand_len: int = 64,
        protein_dim: int = 320,
        ligand_dim: int = 768,
        threshold: float = 7.0
    ):
        self.data = data.reset_index(drop=True)
        self.protein_dir = Path(protein_dir)
        self.ligand_dir = Path(ligand_dir)
        self.activity_col = activity_col
        self.max_protein_len = max_protein_len
        self.max_ligand_len = max_ligand_len
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.threshold = threshold
        
        # Pre-compute binary labels
        self.data['is_active'] = (self.data[activity_col] >= threshold).astype(int)
        
        # Cache for loaded embeddings (optional)
        self._protein_cache: Dict[str, np.ndarray] = {}
        self._ligand_cache: Dict[str, np.ndarray] = {}
        self._use_cache = False
    
    def enable_cache(self):
        """Enable embedding caching for faster repeated access."""
        self._use_cache = True
    
    def disable_cache(self):
        """Disable and clear embedding cache."""
        self._use_cache = False
        self._protein_cache.clear()
        self._ligand_cache.clear()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_protein(self, seq_id: str) -> np.ndarray:
        """Load and pad/truncate protein embedding."""
        if self._use_cache and seq_id in self._protein_cache:
            return self._protein_cache[seq_id]
        
        # Load embedding
        path = self.protein_dir / f"{seq_id}.npy"
        embedding = np.load(path)
        
        # Truncate if needed
        embedding = embedding[:self.max_protein_len]
        
        # Pad to fixed length
        padded = np.zeros((self.max_protein_len, self.protein_dim), dtype=np.float32)
        padded[:len(embedding)] = embedding
        
        if self._use_cache:
            self._protein_cache[seq_id] = padded
        
        return padded
    
    def _load_ligand(self, chembl_id: str) -> np.ndarray:
        """Load and pad/truncate ligand embedding."""
        if self._use_cache and chembl_id in self._ligand_cache:
            return self._ligand_cache[chembl_id]
        
        # Load embedding
        path = self.ligand_dir / f"{chembl_id}.npy"
        embedding = np.load(path)
        
        # Truncate if needed
        embedding = embedding[:self.max_ligand_len]
        
        # Pad to fixed length
        padded = np.zeros((self.max_ligand_len, self.ligand_dim), dtype=np.float32)
        padded[:len(embedding)] = embedding
        
        if self._use_cache:
            self._ligand_cache[chembl_id] = padded
        
        return padded
    
    def _get_chembl_id(self, row) -> str:
        """Get ChEMBL ID from row, handling different column names."""
        if 'molecule_chembl_id' in row.index:
            return str(row['molecule_chembl_id'])
        elif 'chembl_id' in row.index:
            return str(row['chembl_id'])
        else:
            raise KeyError("Neither 'molecule_chembl_id' nor 'chembl_id' found in data")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - protein_embedding: (max_protein_len, protein_dim)
                - ligand_embedding: (max_ligand_len, ligand_dim)
                - activity: pChEMBL value
                - is_active: Binary activity label
        """
        row = self.data.iloc[idx]
        
        # Load embeddings
        protein = self._load_protein(str(row['seq_id']))
        chembl_id = self._get_chembl_id(row)
        ligand = self._load_ligand(chembl_id)
        
        # Get targets
        activity = row[self.activity_col]
        is_active = row['is_active']
        
        return {
            'protein_embedding': torch.from_numpy(protein),
            'ligand_embedding': torch.from_numpy(ligand),
            'activity': torch.tensor(activity, dtype=torch.float32),
            'is_active': torch.tensor(is_active, dtype=torch.long)
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a sample (for analysis)."""
        row = self.data.iloc[idx]
        return {
            'seq_id': row['seq_id'],
            'chembl_id': self._get_chembl_id(row),
            'activity': row[self.activity_col],
            'is_active': row['is_active'],
            'standard_type': row.get('standard_type', 'unknown')
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_dir: str,
    ligand_dir: str,
    activity_col: str = 'pchembl_value',
    batch_size: int = 64,
    max_protein_len: int = 256,
    max_ligand_len: int = 64,
    protein_dim: int = 320,
    ligand_dim: int = 768,
    threshold: float = 7.0,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        protein_dir: Directory with protein embeddings
        ligand_dir: Directory with ligand embeddings
        activity_col: Column name for activity values
        batch_size: Batch size for DataLoaders
        max_protein_len: Maximum protein sequence length
        max_ligand_len: Maximum ligand token length
        protein_dim: Protein embedding dimension
        ligand_dim: Ligand embedding dimension
        threshold: Activity threshold for classification
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ProteinLigandDataset(
        data=train_df,
        protein_dir=protein_dir,
        ligand_dir=ligand_dir,
        activity_col=activity_col,
        max_protein_len=max_protein_len,
        max_ligand_len=max_ligand_len,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        threshold=threshold
    )
    
    val_dataset = ProteinLigandDataset(
        data=val_df,
        protein_dir=protein_dir,
        ligand_dir=ligand_dir,
        activity_col=activity_col,
        max_protein_len=max_protein_len,
        max_ligand_len=max_ligand_len,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        threshold=threshold
    )
    
    test_dataset = ProteinLigandDataset(
        data=test_df,
        protein_dir=protein_dir,
        ligand_dir=ligand_dir,
        activity_col=activity_col,
        max_protein_len=max_protein_len,
        max_ligand_len=max_ligand_len,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        threshold=threshold
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Aliases for backward compatibility
AffinityMatrixDataset = ProteinLigandDataset
