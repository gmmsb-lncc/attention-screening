"""
Dataset and DataLoader for matrix embeddings (protein + ligand).

This module provides data loading utilities for the CrossAttentionAffinityModel,
handling variable-length protein and ligand embedding matrices with padding.

Author: DockTKinase Team
Date: November 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MatrixEmbeddingDataset(Dataset):
    """
    Dataset for protein-ligand matrix embeddings.

    Loads pre-computed embedding matrices from disk and pairs them
    with labels for training/evaluation.

    Directory structure expected:
        protein_matrix_embeddings/
            {seq_id}_matrix.npy  -> [seq_len, protein_dim]
        ligand_matrix_embeddings/
            {chembl_id}_matrix.npy  -> [token_len, ligand_dim]

    Or vector embeddings (fallback):
        protein_embeddings/
            {seq_id}_embedding.npy  -> [protein_dim]
        ligand_embeddings/
            {chembl_id}_embedding.npy  -> [ligand_dim]

    Supports multiple directories for combined datasets (e.g., human + non_human).
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        protein_matrix_dir: Union[str, Path, List[Union[str, Path]]],
        ligand_matrix_dir: Union[str, Path, List[Union[str, Path]]],
        protein_vector_dir: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        ligand_vector_dir: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        label_column: str = 'label',
        regression_column: Optional[str] = 'pchembl_value',
        protein_id_column: str = 'seq_id',
        ligand_id_column: str = 'chembl_id',
        use_matrices: bool = True,
        use_protein_matrices: Optional[bool] = None,
        use_ligand_matrices: Optional[bool] = None,
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_df: DataFrame with protein-ligand pairs and labels
            protein_matrix_dir: Directory (or list of directories) with protein matrix embeddings
            ligand_matrix_dir: Directory (or list of directories) with ligand matrix embeddings
            protein_vector_dir: Fallback directory for protein vectors
            ligand_vector_dir: Fallback directory for ligand vectors
            label_column: Column name for classification labels
            regression_column: Column name for regression targets (optional)
            protein_id_column: Column name for protein IDs
            ligand_id_column: Column name for ligand IDs
            use_matrices: If True, load matrices; if False, load vectors
            cache_in_memory: If True, cache all embeddings in memory
        """
        self.data_df = data_df.reset_index(drop=True)

        # Support single dir or list of dirs
        if isinstance(protein_matrix_dir, (str, Path)):
            self.protein_matrix_dirs = [Path(protein_matrix_dir)]
        else:
            self.protein_matrix_dirs = [Path(d) for d in protein_matrix_dir]

        if isinstance(ligand_matrix_dir, (str, Path)):
            self.ligand_matrix_dirs = [Path(ligand_matrix_dir)]
        else:
            self.ligand_matrix_dirs = [Path(d) for d in ligand_matrix_dir]

        # Keep backward compatibility
        self.protein_matrix_dir = self.protein_matrix_dirs[0]
        self.ligand_matrix_dir = self.ligand_matrix_dirs[0]

        if isinstance(protein_vector_dir, (str, Path)) or protein_vector_dir is None:
            self.protein_vector_dirs = [Path(protein_vector_dir)] if protein_vector_dir else []
        else:
            self.protein_vector_dirs = [Path(d) for d in protein_vector_dir]

        if isinstance(ligand_vector_dir, (str, Path)) or ligand_vector_dir is None:
            self.ligand_vector_dirs = [Path(ligand_vector_dir)] if ligand_vector_dir else []
        else:
            self.ligand_vector_dirs = [Path(d) for d in ligand_vector_dir]
        
        self.label_column = label_column
        self.regression_column = regression_column
        self.protein_id_column = protein_id_column
        self.ligand_id_column = ligand_id_column
        self.use_protein_matrices = use_matrices if use_protein_matrices is None else use_protein_matrices
        self.use_ligand_matrices = use_matrices if use_ligand_matrices is None else use_ligand_matrices
        self.cache_in_memory = cache_in_memory
        
        # Cache
        self._protein_cache: Dict[str, np.ndarray] = {}
        self._ligand_cache: Dict[str, np.ndarray] = {}
        
        # Validate data
        self._validate_data()
        
        logger.info(
            f"MatrixEmbeddingDataset initialized: "
            f"{len(self.data_df)} samples, use_matrices={use_matrices}"
        )
    
    def _validate_data(self):
        """Validate that required columns exist."""
        required_cols = [self.protein_id_column, self.ligand_id_column, self.label_column]
        for col in required_cols:
            if col not in self.data_df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Check directories exist (at least one must exist for each type)
        if self.use_protein_matrices or self.use_ligand_matrices:
            protein_dirs_exist = [d.exists() for d in self.protein_matrix_dirs]
            ligand_dirs_exist = [d.exists() for d in self.ligand_matrix_dirs]

            if not any(protein_dirs_exist):
                logger.warning(f"No protein matrix dirs found: {self.protein_matrix_dirs}")
            if not any(ligand_dirs_exist):
                logger.warning(f"No ligand matrix dirs found: {self.ligand_matrix_dirs}")
        if not self.use_protein_matrices and not self.protein_vector_dirs:
            logger.warning("Protein vectors requested but no protein vector dirs provided.")
        if not self.use_ligand_matrices and not self.ligand_vector_dirs:
            logger.warning("Ligand vectors requested but no ligand vector dirs provided.")
    
    def _load_embedding(
        self,
        embed_id: str,
        matrix_dirs: List[Path],
        vector_dir: Optional[Path],
        cache: Dict[str, np.ndarray],
        is_protein: bool = True
    ) -> np.ndarray:
        """
        Load embedding from disk, with fallback to vector if matrix not found.

        Searches through multiple directories to find the embedding.
        Tries multiple file naming patterns for ligands (e.g., _matrix.npy, _molformer_matrix.npy).

        Args:
            embed_id: Embedding identifier
            matrix_dirs: List of directories for matrix embeddings
            vector_dir: Fallback directory for vector embeddings
            cache: Cache dictionary
            is_protein: Whether this is a protein embedding

        Returns:
            Embedding array (matrix or vector)
        """
        # Check cache
        if embed_id in cache:
            return cache[embed_id]

        # File patterns to try (different naming conventions)
        if is_protein:
            file_patterns = [f"{embed_id}_matrix.npy"]
        else:
            # Ligands may use different naming: SMI-TED uses _matrix.npy, MoLFormer uses _molformer_matrix.npy
            file_patterns = [
                f"{embed_id}_matrix.npy",
                f"{embed_id}_molformer_matrix.npy",
            ]

        # Try loading matrix from each directory with each pattern
        checked_paths = []
        if (self.use_protein_matrices and is_protein) or (self.use_ligand_matrices and not is_protein):
            for matrix_dir in matrix_dirs:
                for pattern in file_patterns:
                    matrix_file = matrix_dir / pattern
                    checked_paths.append(str(matrix_file))
                    if matrix_file.exists():
                        embedding = np.load(matrix_file)
                        if self.cache_in_memory:
                            cache[embed_id] = embedding
                        return embedding

        # Fallback to vector
        vector_dirs = self.protein_vector_dirs if is_protein else self.ligand_vector_dirs
        for vdir in vector_dirs:
            vector_file = vdir / f"{embed_id}_embedding.npy"
            if vector_file.exists():
                embedding = np.load(vector_file)
                # Expand to [1, dim] for consistency
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                if self.cache_in_memory:
                    cache[embed_id] = embedding
                return embedding

        # Error if not found
        embed_type = "protein" if is_protein else "ligand"
        raise FileNotFoundError(
            f"{embed_type} embedding not found for '{embed_id}'. "
            f"Checked: {checked_paths}"
        )
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dict with:
                - protein_matrix: [protein_len, protein_dim]
                - ligand_matrix: [ligand_len, ligand_dim]
                - label: classification label (0 or 1)
                - regression_target: regression value (if available)
                - protein_id: protein identifier
                - ligand_id: ligand identifier
        """
        row = self.data_df.iloc[idx]
        
        protein_id = row[self.protein_id_column]
        ligand_id = row[self.ligand_id_column]
        label = row[self.label_column]
        
        # Load embeddings (search through all directories)
        protein_matrix = self._load_embedding(
            protein_id,
            self.protein_matrix_dirs,
            self.protein_vector_dir,
            self._protein_cache,
            is_protein=True
        )

        ligand_matrix = self._load_embedding(
            ligand_id,
            self.ligand_matrix_dirs,
            self.ligand_vector_dir,
            self._ligand_cache,
            is_protein=False
        )
        
        result = {
            'protein_matrix': torch.from_numpy(protein_matrix).float(),
            'ligand_matrix': torch.from_numpy(ligand_matrix).float(),
            'label': torch.tensor(label, dtype=torch.float32),
            'protein_id': protein_id,
            'ligand_id': ligand_id
        }
        
        # Add regression target if available
        if self.regression_column and self.regression_column in self.data_df.columns:
            reg_value = row[self.regression_column]
            if pd.notna(reg_value):
                result['regression_target'] = torch.tensor(reg_value, dtype=torch.float32)
                result['has_regression'] = torch.tensor(1, dtype=torch.float32)
            else:
                result['regression_target'] = torch.tensor(0.0, dtype=torch.float32)
                result['has_regression'] = torch.tensor(0, dtype=torch.float32)
        
        return result


def collate_matrix_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length matrix embeddings.
    
    Pads protein and ligand matrices to the maximum length in the batch.
    
    Args:
        batch: List of samples from MatrixEmbeddingDataset
    
    Returns:
        Collated batch with:
            - protein_matrix: [batch, max_protein_len, protein_dim]
            - ligand_matrix: [batch, max_ligand_len, ligand_dim]
            - protein_mask: [batch, max_protein_len]
            - ligand_mask: [batch, max_ligand_len]
            - labels: [batch, 1]
            - regression_targets: [batch, 1]
            - regression_mask: [batch, 1]
    """
    batch_size = len(batch)
    
    # Get dimensions
    protein_dim = batch[0]['protein_matrix'].shape[-1]
    ligand_dim = batch[0]['ligand_matrix'].shape[-1]
    
    # Find max lengths
    max_protein_len = max(sample['protein_matrix'].shape[0] for sample in batch)
    max_ligand_len = max(sample['ligand_matrix'].shape[0] for sample in batch)
    
    # Initialize padded tensors
    protein_matrices = torch.zeros(batch_size, max_protein_len, protein_dim)
    ligand_matrices = torch.zeros(batch_size, max_ligand_len, ligand_dim)
    protein_masks = torch.zeros(batch_size, max_protein_len)
    ligand_masks = torch.zeros(batch_size, max_ligand_len)
    labels = torch.zeros(batch_size, 1)
    
    # Regression
    regression_targets = torch.zeros(batch_size, 1)
    regression_mask = torch.zeros(batch_size, 1)
    
    # IDs
    protein_ids = []
    ligand_ids = []
    
    # Fill tensors
    for i, sample in enumerate(batch):
        # Protein
        prot_len = sample['protein_matrix'].shape[0]
        protein_matrices[i, :prot_len] = sample['protein_matrix']
        protein_masks[i, :prot_len] = 1
        
        # Ligand
        lig_len = sample['ligand_matrix'].shape[0]
        ligand_matrices[i, :lig_len] = sample['ligand_matrix']
        ligand_masks[i, :lig_len] = 1
        
        # Labels
        labels[i] = sample['label']
        
        # Regression
        if 'regression_target' in sample:
            regression_targets[i] = sample['regression_target']
            regression_mask[i] = sample.get('has_regression', torch.tensor(1))
        
        # IDs
        protein_ids.append(sample['protein_id'])
        ligand_ids.append(sample['ligand_id'])
    
    return {
        'protein_matrix': protein_matrices,
        'ligand_matrix': ligand_matrices,
        'protein_mask': protein_masks,
        'ligand_mask': ligand_masks,
        'labels': labels,
        'regression_targets': regression_targets,
        'regression_mask': regression_mask,
        'protein_ids': protein_ids,
        'ligand_ids': ligand_ids
    }


def create_matrix_dataloader(
    data_df: pd.DataFrame,
    protein_matrix_dir: Union[str, Path],
    ligand_matrix_dir: Union[str, Path],
    protein_vector_dir: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    ligand_vector_dir: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    use_protein_matrices: Optional[bool] = None,
    use_ligand_matrices: Optional[bool] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for matrix embeddings.
    
    Args:
        data_df: DataFrame with protein-ligand pairs
        protein_matrix_dir: Directory with protein matrices
        ligand_matrix_dir: Directory with ligand matrices
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        **dataset_kwargs: Additional arguments for MatrixEmbeddingDataset
    
    Returns:
        Configured DataLoader
    """
    dataset = MatrixEmbeddingDataset(
        data_df=data_df,
        protein_matrix_dir=protein_matrix_dir,
        ligand_matrix_dir=ligand_matrix_dir,
        protein_vector_dir=protein_vector_dir,
        ligand_vector_dir=ligand_vector_dir,
        use_protein_matrices=use_protein_matrices,
        use_ligand_matrices=use_ligand_matrices,
        **dataset_kwargs
    )
    
    dataloader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_matrix_batch
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(dataset, **dataloader_kwargs)


class VectorEmbeddingDataset(Dataset):
    """
    Dataset for concatenated vector embeddings (original pipeline).
    
    For use with MLP classifier when matrices are not available.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        protein_dir: Union[str, Path],
        ligand_dir: Union[str, Path],
        label_column: str = 'label',
        protein_id_column: str = 'seq_id',
        ligand_id_column: str = 'chembl_id'
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.protein_dir = Path(protein_dir)
        self.ligand_dir = Path(ligand_dir)
        self.label_column = label_column
        self.protein_id_column = protein_id_column
        self.ligand_id_column = ligand_id_column
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data_df.iloc[idx]
        
        protein_id = row[self.protein_id_column]
        ligand_id = row[self.ligand_id_column]
        label = row[self.label_column]
        
        # Load vectors
        protein_vec = np.load(self.protein_dir / f"{protein_id}_embedding.npy")
        ligand_vec = np.load(self.ligand_dir / f"{ligand_id}_embedding.npy")
        
        # Concatenate
        combined = np.concatenate([protein_vec, ligand_vec])
        
        return (
            torch.from_numpy(combined).float(),
            torch.tensor(label, dtype=torch.float32)
        )


if __name__ == "__main__":
    # Quick test
    print("Testing MatrixEmbeddingDataset...")
    
    # Create dummy data
    test_df = pd.DataFrame({
        'seq_id': ['P1', 'P1', 'P2'],
        'chembl_id': ['C1', 'C2', 'C1'],
        'label': [1, 0, 1],
        'pchembl_value': [7.5, None, 6.2]
    })
    
    # Create dummy directories and files
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        prot_dir = tmpdir / 'protein_matrix_embeddings'
        lig_dir = tmpdir / 'ligand_matrix_embeddings'
        prot_dir.mkdir()
        lig_dir.mkdir()
        
        # Create dummy matrices
        np.save(prot_dir / 'P1_matrix.npy', np.random.randn(50, 320))
        np.save(prot_dir / 'P2_matrix.npy', np.random.randn(100, 320))
        np.save(lig_dir / 'C1_matrix.npy', np.random.randn(30, 768))
        np.save(lig_dir / 'C2_matrix.npy', np.random.randn(20, 768))
        
        # Create dataset
        dataset = MatrixEmbeddingDataset(
            data_df=test_df,
            protein_matrix_dir=prot_dir,
            ligand_matrix_dir=lig_dir,
            use_matrices=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test single item
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Protein matrix shape: {sample['protein_matrix'].shape}")
        print(f"Ligand matrix shape: {sample['ligand_matrix'].shape}")
        
        # Test DataLoader
        dataloader = create_matrix_dataloader(
            test_df, prot_dir, lig_dir, batch_size=2
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Protein matrices: {batch['protein_matrix'].shape}")
        print(f"Ligand matrices: {batch['ligand_matrix'].shape}")
        print(f"Protein masks: {batch['protein_mask'].shape}")
        print(f"Labels: {batch['labels'].shape}")
        
    print("\n✅ All tests passed!")
