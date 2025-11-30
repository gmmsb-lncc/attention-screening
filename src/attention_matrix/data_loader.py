"""
Data Loader for Attention Matrix Pipeline.

Loads pre-computed embeddings from the existing pipeline results.
Single Responsibility: Data loading and preparation only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class EmbeddingDataLoader:
    """
    Loads pre-computed protein and ligand embeddings for the attention matrix pipeline.
    
    Supports loading from:
    1. Individual embedding files (*_embedding.npy)
    2. Pre-built embedding matrix (embedding_matrix.npy)
    3. Existing splits (train_idx.npy, val_idx.npy, test_idx.npy)
    
    Args:
        results_dir: Directory containing pipeline results with embeddings
        data_file: Original dataset TSV file
        protein_dim: Expected protein embedding dimension (default: 320 for ESM2-8M)
        ligand_dim: Expected ligand embedding dimension (default: 768 for SMI-TED)
        activity_threshold: pChEMBL threshold for binary classification (default: 7.0)
    """
    
    def __init__(
        self,
        results_dir: str,
        data_file: str,
        protein_dim: int = 320,
        ligand_dim: int = 768,
        activity_threshold: float = 7.0
    ):
        self.results_dir = Path(results_dir)
        self.data_file = Path(data_file)
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.activity_threshold = activity_threshold
        
        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.protein_embeddings: Optional[np.ndarray] = None
        self.ligand_embeddings: Optional[np.ndarray] = None
        self.binary_labels: Optional[np.ndarray] = None
        self.regression_targets: Optional[np.ndarray] = None
        self.protein_ids: Optional[np.ndarray] = None
        self.valid_indices: Optional[np.ndarray] = None
        
        # Split indices
        self.train_idx: Optional[np.ndarray] = None
        self.val_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the original dataset."""
        logger.info(f"Loading dataset: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file, sep='\t')
        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def load_embeddings_from_files(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings from files. Supports multiple directory structures:
        1. protein_matrices/ + ligand_matrices/ (individual files)
        2. build/embedding_matrix.npy (concatenated matrix)
        
        Returns:
            Tuple of (protein_embeddings, ligand_embeddings) arrays
        """
        if self.df is None:
            self.load_dataset()
        
        # Try method 1: individual files
        protein_dir = self.results_dir / 'protein_matrices'
        ligand_dir = self.results_dir / 'ligand_matrices'
        
        if protein_dir.exists() and ligand_dir.exists():
            return self._load_from_individual_files(protein_dir, ligand_dir)
        
        # Try method 2: concatenated matrix in build/
        build_dir = self.results_dir / 'build'
        emb_matrix_file = build_dir / 'embedding_matrix.npy'
        
        if emb_matrix_file.exists():
            return self._load_from_concatenated_matrix(build_dir)
        
        raise FileNotFoundError(
            f"Could not find embeddings. Expected either:\n"
            f"  1. {protein_dir} and {ligand_dir}\n"
            f"  2. {emb_matrix_file}"
        )
    
    def _load_from_concatenated_matrix(self, build_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load from build/embedding_matrix.npy (concatenated protein+ligand)."""
        logger.info(f"Loading embeddings from concatenated matrix: {build_dir}")
        
        # Load concatenated embeddings (shape: N x (protein_dim + ligand_dim))
        emb_matrix = np.load(build_dir / 'embedding_matrix.npy')
        logger.info(f"Embedding matrix shape: {emb_matrix.shape}")
        
        # Split into protein and ligand
        total_dim = emb_matrix.shape[1]
        self.protein_embeddings = emb_matrix[:, :self.protein_dim]
        self.ligand_embeddings = emb_matrix[:, self.protein_dim:]
        
        # Adjust ligand_dim if needed
        actual_ligand_dim = total_dim - self.protein_dim
        if actual_ligand_dim != self.ligand_dim:
            logger.warning(f"Adjusting ligand_dim: expected {self.ligand_dim}, got {actual_ligand_dim}")
            self.ligand_dim = actual_ligand_dim
        
        logger.info(f"Protein embeddings: {self.protein_embeddings.shape}")
        logger.info(f"Ligand embeddings: {self.ligand_embeddings.shape}")
        
        # Load labels
        labels_file = build_dir / 'binary_labels.npy'
        if labels_file.exists():
            self.binary_labels = np.load(labels_file)
        else:
            # Compute from dataset
            pchembl = self.df['pchembl_value'].fillna(5.0).values
            self.binary_labels = (pchembl >= self.activity_threshold).astype(int)
        
        # Regression targets
        self.regression_targets = self.df['pchembl_value'].fillna(5.0).values.astype(np.float32)
        
        # Valid indices (all in this case)
        self.valid_indices = np.arange(len(self.binary_labels))
        
        # Protein IDs
        if 'seq_id' in self.df.columns:
            self.protein_ids = self.df['seq_id'].values
        elif 'target_kinase' in self.df.columns:
            self.protein_ids = self.df['target_kinase'].values
        
        logger.info(f"Valid samples: {len(self.valid_indices)}")
        logger.info(f"Binary labels: {self.binary_labels.sum()} active, {len(self.binary_labels) - self.binary_labels.sum()} inactive")
        
        return self.protein_embeddings, self.ligand_embeddings
    
    def _load_from_individual_files(self, protein_dir: Path, ligand_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load from individual protein_matrices/ and ligand_matrices/ files."""
        logger.info(f"Loading from individual files: {protein_dir}, {ligand_dir}")
        
        # Load protein embeddings
        protein_emb_dict = {}
        for f in protein_dir.glob('*_embedding.npy'):
            protein_id = f.stem.replace('_embedding', '')
            emb = np.load(f)
            protein_emb_dict[protein_id] = emb.flatten()
        logger.info(f"Loaded {len(protein_emb_dict)} protein embeddings")
        
        # Load ligand embeddings
        logger.info("Loading ligand embeddings...")
        ligand_emb_dict = {}
        for f in ligand_dir.glob('*_embedding.npy'):
            ligand_id = f.stem.replace('_embedding', '')
            ligand_emb_dict[ligand_id] = np.load(f).flatten()
        logger.info(f"Loaded {len(ligand_emb_dict)} ligand embeddings")
        
        # Determine column names
        protein_col = 'seq_id'
        ligand_col = 'chembl_id'
        
        # Build aligned arrays
        logger.info("Building embedding matrix...")
        protein_embs = []
        ligand_embs = []
        binary_labels = []
        regression_targets = []
        valid_indices = []
        protein_ids_list = []
        
        for idx, row in self.df.iterrows():
            prot_id = str(row[protein_col])
            lig_id = str(row[ligand_col])
            
            if prot_id in protein_emb_dict and lig_id in ligand_emb_dict:
                protein_embs.append(protein_emb_dict[prot_id])
                ligand_embs.append(ligand_emb_dict[lig_id])
                
                # Get pChEMBL value
                pchembl = row.get('pchembl_value', 5.0)
                if pd.isna(pchembl):
                    pchembl = 5.0
                pchembl = float(pchembl)
                
                # Binary label for classification
                binary_labels.append(1 if pchembl >= self.activity_threshold else 0)
                
                # Continuous target for regression
                regression_targets.append(pchembl)
                
                valid_indices.append(idx)
                protein_ids_list.append(prot_id)
        
        self.protein_embeddings = np.array(protein_embs)
        self.ligand_embeddings = np.array(ligand_embs)
        self.binary_labels = np.array(binary_labels)
        self.regression_targets = np.array(regression_targets)
        self.valid_indices = np.array(valid_indices)
        self.protein_ids = np.array(protein_ids_list)
        
        logger.info(f"Valid samples: {len(valid_indices)} / {len(self.df)}")
        logger.info(f"Protein embeddings: {self.protein_embeddings.shape}")
        logger.info(f"Ligand embeddings: {self.ligand_embeddings.shape}")
        logger.info(f"Binary labels: {self.binary_labels.sum()} active, {len(self.binary_labels) - self.binary_labels.sum()} inactive")
        logger.info(f"Regression targets: min={self.regression_targets.min():.2f}, max={self.regression_targets.max():.2f}, mean={self.regression_targets.mean():.2f}")
        
        return self.protein_embeddings, self.ligand_embeddings
    
    def load_splits(self, splits_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load pre-computed train/val/test splits, or create them if they don't exist.
        
        Args:
            splits_dir: Directory containing split files
            
        Returns:
            Tuple of (train_idx, val_idx, test_idx) arrays
        """
        # Try to find existing splits
        search_paths = [
            self.results_dir / 'splits_leakage_aware',
            self.results_dir / 'splits',
            self.results_dir / 'build' / 'splits',
        ]
        if splits_dir:
            search_paths.insert(0, Path(splits_dir))
        
        splits_path = None
        for path in search_paths:
            if path.exists() and (path / 'train_idx.npy').exists():
                splits_path = path
                break
        
        if splits_path is not None:
            return self._load_existing_splits(splits_path)
        else:
            logger.info("No pre-computed splits found. Creating leakage-aware splits...")
            return self._create_leakage_aware_splits()
    
    def _load_existing_splits(self, splits_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load splits from existing files."""
        logger.info(f"Loading splits from: {splits_path}")
        
        train_idx_orig = np.load(splits_path / 'train_idx.npy')
        val_idx_orig = np.load(splits_path / 'val_idx.npy')
        test_idx_orig = np.load(splits_path / 'test_idx.npy')
        
        # Remap indices to valid samples if needed
        if self.valid_indices is not None and len(self.valid_indices) < len(train_idx_orig) + len(val_idx_orig) + len(test_idx_orig):
            valid_set = set(self.valid_indices)
            idx_remap = {old: new for new, old in enumerate(self.valid_indices)}
            
            self.train_idx = np.array([idx_remap[i] for i in train_idx_orig if i in valid_set])
            self.val_idx = np.array([idx_remap[i] for i in val_idx_orig if i in valid_set])
            self.test_idx = np.array([idx_remap[i] for i in test_idx_orig if i in valid_set])
        else:
            self.train_idx = train_idx_orig
            self.val_idx = val_idx_orig
            self.test_idx = test_idx_orig
        
        self._log_split_info()
        return self.train_idx, self.val_idx, self.test_idx
    
    def _create_leakage_aware_splits(
        self, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create leakage-aware splits ensuring no protein overlap."""
        from sklearn.model_selection import train_test_split
        
        n_samples = len(self.protein_embeddings)
        
        if self.protein_ids is None:
            # Simple random split if no protein IDs
            indices = np.arange(n_samples)
            train_val_idx, test_idx = train_test_split(
                indices, test_size=(1 - train_ratio - val_ratio), random_state=42
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_ratio/(train_ratio + val_ratio), random_state=42
            )
        else:
            # Leakage-aware split by protein
            unique_proteins = np.unique(self.protein_ids)
            n_proteins = len(unique_proteins)
            
            # Split proteins
            train_val_prot, test_prot = train_test_split(
                unique_proteins, test_size=(1 - train_ratio - val_ratio), random_state=42
            )
            train_prot, val_prot = train_test_split(
                train_val_prot, test_size=val_ratio/(train_ratio + val_ratio), random_state=42
            )
            
            # Map samples to splits
            train_set = set(train_prot)
            val_set = set(val_prot)
            test_set = set(test_prot)
            
            train_idx = np.array([i for i, p in enumerate(self.protein_ids) if p in train_set])
            val_idx = np.array([i for i, p in enumerate(self.protein_ids) if p in val_set])
            test_idx = np.array([i for i, p in enumerate(self.protein_ids) if p in test_set])
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        self._log_split_info()
        return self.train_idx, self.val_idx, self.test_idx
    
    def _log_split_info(self):
        """Log split statistics."""
        total = len(self.train_idx) + len(self.val_idx) + len(self.test_idx)
        logger.info(f"Splits: train={len(self.train_idx)} ({100*len(self.train_idx)/total:.1f}%), "
                   f"val={len(self.val_idx)} ({100*len(self.val_idx)/total:.1f}%), "
                   f"test={len(self.test_idx)} ({100*len(self.test_idx)/total:.1f}%)")
        
        # Check protein overlap
        if self.protein_ids is not None:
            train_prot = set(self.protein_ids[self.train_idx])
            val_prot = set(self.protein_ids[self.val_idx])
            test_prot = set(self.protein_ids[self.test_idx])
            
            overlap_train_val = len(train_prot & val_prot)
            overlap_train_test = len(train_prot & test_prot)
            overlap_val_test = len(val_prot & test_prot)
            
            logger.info(f"Protein overlap: train-val={overlap_train_val}, "
                       f"train-test={overlap_train_test}, val-test={overlap_val_test}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data."""
        if self.protein_embeddings is None:
            raise ValueError("Data not loaded. Call load_embeddings_from_files() first.")
        
        total = len(self.train_idx) + len(self.val_idx) + len(self.test_idx)
        
        return {
            'n_samples': len(self.protein_embeddings),
            'n_valid': len(self.valid_indices) if self.valid_indices is not None else len(self.protein_embeddings),
            'protein_dim': self.protein_embeddings.shape[1],
            'ligand_dim': self.ligand_embeddings.shape[1],
            'n_active': int(self.binary_labels.sum()),
            'n_inactive': int(len(self.binary_labels) - self.binary_labels.sum()),
            'pchembl_min': float(self.regression_targets.min()),
            'pchembl_max': float(self.regression_targets.max()),
            'pchembl_mean': float(self.regression_targets.mean()),
            'pchembl_std': float(self.regression_targets.std()),
            'n_train': len(self.train_idx),
            'n_val': len(self.val_idx),
            'n_test': len(self.test_idx),
            'train_ratio': len(self.train_idx) / total,
            'val_ratio': len(self.val_idx) / total,
            'test_ratio': len(self.test_idx) / total,
            'activity_threshold': self.activity_threshold
        }


def load_data_for_attention_matrix(
    results_dir: str,
    data_file: str,
    splits_dir: Optional[str] = None,
    protein_dim: int = 320,
    ligand_dim: int = 768,
    activity_threshold: float = 7.0
) -> Dict[str, Any]:
    """
    Convenience function to load all data needed for attention matrix training.
    
    Args:
        results_dir: Directory with pre-computed embeddings
        data_file: Original TSV dataset
        splits_dir: Directory with splits (optional)
        protein_dim: Protein embedding dimension
        ligand_dim: Ligand embedding dimension
        activity_threshold: pChEMBL threshold for binary classification
        
    Returns:
        Dictionary with all loaded data
    """
    loader = EmbeddingDataLoader(
        results_dir=results_dir,
        data_file=data_file,
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        activity_threshold=activity_threshold
    )
    
    # Load data
    loader.load_dataset()
    loader.load_embeddings_from_files()
    loader.load_splits(splits_dir)
    
    return {
        'protein_embeddings': loader.protein_embeddings,
        'ligand_embeddings': loader.ligand_embeddings,
        'binary_labels': loader.binary_labels,
        'regression_targets': loader.regression_targets,
        'protein_ids': loader.protein_ids,
        'train_idx': loader.train_idx,
        'val_idx': loader.val_idx,
        'test_idx': loader.test_idx,
        'metadata': loader.metadata,
        'summary': loader.get_data_summary()
    }
