"""Dataset and collate function for Level 5-Lite."""

import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Level5LiteDataset(Dataset):
    """Dataset for Level 5-Lite.
    
    Loads:
    - Protein matrix: ESM-2 per-residue [L, protein_dim]
    - Ligand matrix: MoLFormer per-token [T, 768]
    - Label: binary (pChEMBL >= threshold → active)
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        protein_matrix_dir: str,
        ligand_matrix_dir: str,
        max_protein_len: int = 1024,
        max_ligand_len: int = 256,
        pchembl_threshold: float = 6.0,
        seq_id_col: str = 'seq_id',
        chembl_id_col: str = 'chembl_id',
        pchembl_col: str = 'pchembl_value',
        cache_in_memory: bool = False,
    ):
        """Initialize Level5LiteDataset.
        
        Args:
            data_df: DataFrame with columns [seq_id, chembl_id, pchembl_value, ...]
            protein_matrix_dir: Directory containing {seq_id}_matrix.npy files
            ligand_matrix_dir: Directory containing {chembl_id}_molformer_matrix.npy files
            max_protein_len: Maximum protein sequence length (truncate if longer)
            max_ligand_len: Maximum ligand token length (truncate if longer)
            pchembl_threshold: Threshold for binary classification (>= is active)
            seq_id_col: Column name for protein sequence ID
            chembl_id_col: Column name for compound ChEMBL ID
            pchembl_col: Column name for pChEMBL value
            cache_in_memory: If True, cache all matrices in RAM
        """
        self.data = data_df.reset_index(drop=True)
        self.protein_dir = protein_matrix_dir
        self.ligand_dir = ligand_matrix_dir
        self.max_protein_len = max_protein_len
        self.max_ligand_len = max_ligand_len
        self.pchembl_threshold = pchembl_threshold
        self.seq_id_col = seq_id_col
        self.chembl_id_col = chembl_id_col
        self.pchembl_col = pchembl_col
        self.cache_in_memory = cache_in_memory
        
        # Caches
        self._protein_cache: Dict[str, np.ndarray] = {}
        self._ligand_cache: Dict[str, np.ndarray] = {}
        
        # Validate that required columns exist
        required_cols = [seq_id_col, chembl_id_col, pchembl_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_protein_matrix(self, seq_id: str) -> np.ndarray:
        """Load protein matrix from file or cache."""
        if seq_id in self._protein_cache:
            return self._protein_cache[seq_id]
        
        # Try different filename patterns
        patterns = [
            f"{seq_id}_matrix.npy",
            f"{seq_id}.npy",
        ]
        
        for pattern in patterns:
            path = os.path.join(self.protein_dir, pattern)
            if os.path.exists(path):
                matrix = np.load(path)
                if self.cache_in_memory:
                    self._protein_cache[seq_id] = matrix
                return matrix
        
        raise FileNotFoundError(
            f"Protein matrix not found for seq_id={seq_id}. "
            f"Tried patterns: {patterns} in {self.protein_dir}"
        )
    
    def _load_ligand_matrix(self, chembl_id: str) -> np.ndarray:
        """Load ligand matrix from file or cache."""
        if chembl_id in self._ligand_cache:
            return self._ligand_cache[chembl_id]
        
        # Try different filename patterns
        patterns = [
            f"{chembl_id}_molformer_matrix.npy",
            f"{chembl_id}_matrix.npy",
            f"{chembl_id}.npy",
        ]
        
        for pattern in patterns:
            path = os.path.join(self.ligand_dir, pattern)
            if os.path.exists(path):
                matrix = np.load(path)
                if self.cache_in_memory:
                    self._ligand_cache[chembl_id] = matrix
                return matrix
        
        raise FileNotFoundError(
            f"Ligand matrix not found for chembl_id={chembl_id}. "
            f"Tried patterns: {patterns} in {self.ligand_dir}"
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        seq_id = str(row[self.seq_id_col])
        chembl_id = str(row[self.chembl_id_col])
        pchembl = float(row[self.pchembl_col])
        
        # Load matrices
        protein_matrix = self._load_protein_matrix(seq_id)
        ligand_matrix = self._load_ligand_matrix(chembl_id)
        
        # Truncate if needed
        protein_matrix = protein_matrix[:self.max_protein_len]
        ligand_matrix = ligand_matrix[:self.max_ligand_len]
        
        # Binary label
        label = 1.0 if pchembl >= self.pchembl_threshold else 0.0
        
        return {
            'protein_matrix': torch.tensor(protein_matrix, dtype=torch.float32),
            'ligand_matrix': torch.tensor(ligand_matrix, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'pchembl': torch.tensor(pchembl, dtype=torch.float32),
            'seq_id': seq_id,
            'chembl_id': chembl_id,
        }


def collate_level5_lite(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding.
    
    Pads protein and ligand matrices to the maximum length in the batch.
    Creates padding masks (True = pad token, False = real token).
    
    Args:
        batch: List of dicts from Level5LiteDataset.__getitem__
    
    Returns:
        Dict with batched tensors:
        - protein_matrix: [B, max_protein_len, protein_dim]
        - ligand_matrix: [B, max_ligand_len, ligand_dim]
        - protein_mask: [B, max_protein_len] (True = pad)
        - ligand_mask: [B, max_ligand_len] (True = pad)
        - label: [B]
        - pchembl: [B]
    """
    # Find max lengths in batch
    max_protein_len = max(b['protein_matrix'].size(0) for b in batch)
    max_ligand_len = max(b['ligand_matrix'].size(0) for b in batch)
    
    protein_matrices = []
    ligand_matrices = []
    protein_masks = []
    ligand_masks = []
    labels = []
    pchembls = []
    seq_ids = []
    chembl_ids = []
    
    for b in batch:
        # Pad protein
        p = b['protein_matrix']
        p_len = p.size(0)
        p_dim = p.size(1)
        
        if p_len < max_protein_len:
            p_padded = F.pad(p, (0, 0, 0, max_protein_len - p_len))
        else:
            p_padded = p
        protein_matrices.append(p_padded)
        
        # Create protein mask
        p_mask = torch.zeros(max_protein_len, dtype=torch.bool)
        p_mask[p_len:] = True
        protein_masks.append(p_mask)
        
        # Pad ligand
        l = b['ligand_matrix']
        l_len = l.size(0)
        
        if l_len < max_ligand_len:
            l_padded = F.pad(l, (0, 0, 0, max_ligand_len - l_len))
        else:
            l_padded = l
        ligand_matrices.append(l_padded)
        
        # Create ligand mask
        l_mask = torch.zeros(max_ligand_len, dtype=torch.bool)
        l_mask[l_len:] = True
        ligand_masks.append(l_mask)
        
        labels.append(b['label'])
        pchembls.append(b['pchembl'])
        seq_ids.append(b['seq_id'])
        chembl_ids.append(b['chembl_id'])
    
    return {
        'protein_matrix': torch.stack(protein_matrices),
        'ligand_matrix': torch.stack(ligand_matrices),
        'protein_mask': torch.stack(protein_masks),
        'ligand_mask': torch.stack(ligand_masks),
        'label': torch.stack(labels),
        'pchembl': torch.stack(pchembls),
        'seq_id': seq_ids,
        'chembl_id': chembl_ids,
    }


def create_level5_lite_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    protein_matrix_dir: str,
    ligand_matrix_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_protein_len: int = 1024,
    max_ligand_len: int = 256,
    pchembl_threshold: float = 6.0,
    pin_memory: bool = True,
    cache_in_memory: bool = False,
) -> tuple:
    """Create DataLoaders for Level 5-Lite.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame (optional)
        protein_matrix_dir: Directory with protein matrices
        ligand_matrix_dir: Directory with ligand matrices
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        max_protein_len: Max protein sequence length
        max_ligand_len: Max ligand token length
        pchembl_threshold: Threshold for binary classification
        pin_memory: Use pinned memory
        cache_in_memory: Cache matrices in RAM
    
    Returns:
        (train_loader, val_loader, test_loader) - test_loader is None if test_df is None
    """
    from torch.utils.data import DataLoader
    
    common_kwargs = {
        'protein_matrix_dir': protein_matrix_dir,
        'ligand_matrix_dir': ligand_matrix_dir,
        'max_protein_len': max_protein_len,
        'max_ligand_len': max_ligand_len,
        'pchembl_threshold': pchembl_threshold,
        'cache_in_memory': cache_in_memory,
    }
    
    train_dataset = Level5LiteDataset(train_df, **common_kwargs)
    val_dataset = Level5LiteDataset(val_df, **common_kwargs)
    
    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_level5_lite,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    test_loader = None
    if test_df is not None:
        test_dataset = Level5LiteDataset(test_df, **common_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader
