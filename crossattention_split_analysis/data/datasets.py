"""Dataset classes for attention matrices and embeddings."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class AttentionMatrixDataset(Dataset):
    """
    Dataset for ESM-2 attention matrices + ligand embeddings.

    Loads attention matrices [seq_len, seq_len] and pads them to [max_len, max_len].
    The second dimension serves as the "embedding dimension" for the model.

    Supports multiple directories for combined datasets (e.g., human + non_human).

    Attributes:
        data_df: DataFrame containing sample information
        attention_matrix_dirs: List of paths to attention matrix files
        ligand_matrix_dirs: List of paths to ligand embedding files
        max_seq_len: Maximum sequence length for padding
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        attention_matrix_dir: Union[str, List[str]],
        ligand_matrix_dir: Union[str, List[str]],
        max_seq_len: int = 1024,
        label_column: str = 'label',
        protein_id_column: str = 'seq_id',
        ligand_id_column: str = 'chembl_id',
        regression_column: Optional[str] = 'pchembl_value'
    ):
        self.data_df = data_df.reset_index(drop=True)

        # Support single dir or list of dirs
        if isinstance(attention_matrix_dir, str):
            self.attention_matrix_dirs = [Path(attention_matrix_dir)]
        else:
            self.attention_matrix_dirs = [Path(d) for d in attention_matrix_dir]

        if isinstance(ligand_matrix_dir, str):
            self.ligand_matrix_dirs = [Path(ligand_matrix_dir)]
        else:
            self.ligand_matrix_dirs = [Path(d) for d in ligand_matrix_dir]

        # Backward compatibility
        self.attention_matrix_dir = self.attention_matrix_dirs[0]
        self.ligand_matrix_dir = self.ligand_matrix_dirs[0]

        self.max_seq_len = max_seq_len
        self.label_column = label_column
        self.protein_id_column = protein_id_column
        self.ligand_id_column = ligand_id_column
        self.regression_column = regression_column

    def __len__(self) -> int:
        return len(self.data_df)

    def _find_file(self, dirs: List[Path], filename: str, alt_patterns: List[str] = None) -> Path:
        """
        Find a file in multiple directories, trying alternative patterns if needed.

        Args:
            dirs: List of directories to search
            filename: Primary filename to look for
            alt_patterns: Alternative filename patterns to try (e.g., for different naming conventions)

        Returns:
            Path to the found file
        """
        patterns = [filename]
        if alt_patterns:
            patterns.extend(alt_patterns)

        checked_paths = []
        for d in dirs:
            for pattern in patterns:
                file_path = d / pattern
                checked_paths.append(str(file_path))
                if file_path.exists():
                    return file_path

        raise FileNotFoundError(
            f"File not found. Checked: {checked_paths}"
        )

    def __getitem__(self, idx: int) -> Dict:
        row = self.data_df.iloc[idx]

        protein_id = str(row[self.protein_id_column])
        ligand_id = row[self.ligand_id_column]
        label = row[self.label_column]

        # Load attention matrix [seq_len, seq_len] - search in all directories
        attn_file = self._find_file(self.attention_matrix_dirs, f"{protein_id}_attention.npy")
        attention_matrix = np.load(attn_file)

        # Pad or truncate to [max_seq_len, max_seq_len]
        seq_len = attention_matrix.shape[0]
        if seq_len < self.max_seq_len:
            padded = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)
            padded[:seq_len, :seq_len] = attention_matrix
            attention_matrix = padded
        else:
            attention_matrix = attention_matrix[:self.max_seq_len, :self.max_seq_len]
            seq_len = self.max_seq_len

        # Load ligand matrix - search in all directories with alternative naming patterns
        # SMI-TED uses _matrix.npy, MoLFormer uses _molformer_matrix.npy, ChemBERTa uses _chemberta_matrix.npy
        ligand_file = self._find_file(
            self.ligand_matrix_dirs,
            f"{ligand_id}_matrix.npy",
            alt_patterns=[f"{ligand_id}_molformer_matrix.npy", f"{ligand_id}_chemberta_matrix.npy"]
        )
        ligand_matrix = np.load(ligand_file)

        result = {
            'protein_matrix': torch.from_numpy(attention_matrix).float(),
            'ligand_matrix': torch.from_numpy(ligand_matrix).float(),
            'label': torch.tensor(label, dtype=torch.float32),
            'protein_len': seq_len,
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

        # Domain label for adversarial domain adaptation (Level 5 DA / Level 6)
        if 'domain_label' in self.data_df.columns:
            result['domain_label'] = torch.tensor(int(row['domain_label']), dtype=torch.long)

        return result


def collate_attention_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for attention matrix batches.

    Handles variable-length ligand sequences by padding to max length in batch.
    """
    batch_size = len(batch)

    # Get dimensions from first sample
    max_protein_len = batch[0]['protein_matrix'].shape[0]
    protein_dim = batch[0]['protein_matrix'].shape[1]
    ligand_dim = batch[0]['ligand_matrix'].shape[-1]

    # Find max ligand length in batch
    max_ligand_len = max(sample['ligand_matrix'].shape[0] for sample in batch)

    # Initialize tensors
    protein_matrices = torch.zeros(batch_size, max_protein_len, protein_dim)
    ligand_matrices = torch.zeros(batch_size, max_ligand_len, ligand_dim)
    protein_masks = torch.zeros(batch_size, max_protein_len)
    ligand_masks = torch.zeros(batch_size, max_ligand_len)
    labels = torch.zeros(batch_size, 1)
    regression_targets = torch.zeros(batch_size, 1)
    regression_mask = torch.zeros(batch_size, 1)

    for i, sample in enumerate(batch):
        # Protein (already padded)
        protein_matrices[i] = sample['protein_matrix']
        prot_len = sample['protein_len']
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

    result = {
        'protein_matrix': protein_matrices,
        'ligand_matrix': ligand_matrices,
        'protein_mask': protein_masks,
        'ligand_mask': ligand_masks,
        'labels': labels,
        'regression_targets': regression_targets,
        'regression_mask': regression_mask,
    }

    # Domain labels for adversarial domain adaptation (Level 5 DA / Level 6)
    if 'domain_label' in batch[0]:
        result['domain_label'] = torch.stack([s['domain_label'] for s in batch])

    return result


def create_attention_dataloader(
    data_df: pd.DataFrame,
    attention_matrix_dir: str,
    ligand_matrix_dir: str,
    max_seq_len: int = 1024,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for attention matrices.

    Args:
        data_df: DataFrame with sample information
        attention_matrix_dir: Path to attention matrix files
        ligand_matrix_dir: Path to ligand embedding files
        max_seq_len: Maximum sequence length for padding
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for AttentionMatrixDataset

    Returns:
        DataLoader instance
    """
    dataset = AttentionMatrixDataset(
        data_df=data_df,
        attention_matrix_dir=attention_matrix_dir,
        ligand_matrix_dir=ligand_matrix_dir,
        max_seq_len=max_seq_len,
        **kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_attention_batch
    )
