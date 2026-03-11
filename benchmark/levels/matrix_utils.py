"""Shared utilities for matrix-based levels (2, 3, 4).

Provides a common ``MatrixDataset``, collation, padding, and dataloader
construction so that Levels 2 and 3 operate on the **exact same input
matrices** as Level 4 — the only difference is the pooling strategy.

Handles the ``dataset="all"`` case transparently by searching across
both ``human`` and ``non_human`` embedding directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from benchmark.config import (
    EMBEDDING_BASE_PATH,
    PCHEMBL_ACTIVITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBEDDING_BASE_PATHS_ALL = [
    "./results/protein_model_benchmark_human_v2",
    "./results/protein_model_benchmark_non_human_v2",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MatrixDataset(Dataset):
    """Lazy loader for paired protein / ligand matrices.

    Accepts **multiple** directories for each modality so that the
    ``dataset="all"`` case (matrices spread across ``human`` and
    ``non_human`` directories) works seamlessly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protein_matrix_dirs: Sequence[Path],
        ligand_matrix_dirs: Sequence[Path],
    ) -> None:
        self._df = df
        self._protein_dirs = list(protein_matrix_dirs)
        self._ligand_dirs = list(ligand_matrix_dirs)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        row = self._df.iloc[idx]
        seq_id = row["seq_id"]
        chembl_id = row["chembl_id"]
        label = row["label"]

        protein_mat = self._load_from_dirs(
            self._protein_dirs,
            f"{seq_id}_matrix.npy",
            fallback_shape=(100, 320),
        )
        ligand_mat = self._load_from_dirs(
            self._ligand_dirs,
            f"{chembl_id}_matrix.npy",
            fallback_shape=(50, 768),
        )
        return protein_mat, ligand_mat, label, seq_id, chembl_id

    @staticmethod
    def _load_from_dirs(
        dirs: list[Path],
        filename: str,
        fallback_shape: tuple[int, int],
    ) -> np.ndarray:
        """Search across multiple directories for a ``.npy`` file.

        Tries both ``{id}_matrix.npy`` and ``{id}_molformer_matrix.npy``
        naming conventions to handle cross-machine inconsistencies.
        """
        # Build alternate filename: foo_matrix.npy -> foo_molformer_matrix.npy
        alt_filename = filename.replace("_matrix.npy", "_molformer_matrix.npy")
        for d in dirs:
            path = d / filename
            if path.exists():
                return np.load(path).astype(np.float32)
            alt_path = d / alt_filename
            if alt_path.exists():
                return np.load(alt_path).astype(np.float32)
        return np.zeros(fallback_shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Collation / padding
# ---------------------------------------------------------------------------

def pad_matrices(
    matrices: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad 2-D matrices to a common sequence length.

    Returns
    -------
    padded : np.ndarray
        Shape ``(batch, max_len, dim)``.
    mask : np.ndarray
        Boolean mask — ``True`` for **valid** positions, ``False`` for padding.
    """
    max_len = max(m.shape[0] for m in matrices)
    dim = matrices[0].shape[1]
    padded = np.zeros((len(matrices), max_len, dim), dtype=np.float32)
    mask = np.ones((len(matrices), max_len), dtype=bool)
    for i, mat in enumerate(matrices):
        length = mat.shape[0]
        padded[i, :length, :] = mat
        mask[i, length:] = False
    return padded, mask


def collate_matrices(batch: list) -> dict:
    """Collate function for ``MatrixDataset`` — pads and stacks."""
    protein_mats, ligand_mats, labels, seq_ids, chembl_ids = zip(*batch)

    protein_batch, protein_mask = pad_matrices(protein_mats)
    ligand_batch, ligand_mask = pad_matrices(ligand_mats)

    return {
        "protein_matrix": torch.from_numpy(protein_batch),
        "ligand_matrix": torch.from_numpy(ligand_batch),
        "protein_mask": torch.from_numpy(protein_mask),
        "ligand_mask": torch.from_numpy(ligand_mask),
        "label": torch.tensor(labels, dtype=torch.float32),
        "seq_id": seq_ids,
        "chembl_id": chembl_ids,
    }


# ---------------------------------------------------------------------------
# Mean pooling (used by Level 2)
# ---------------------------------------------------------------------------

def mean_pool(matrices: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    """Mean-pool over valid positions for each sample in the batch.

    Parameters
    ----------
    matrices : torch.Tensor
        Shape ``(B, max_len, dim)``.
    masks : torch.Tensor
        Boolean mask ``(B, max_len)`` — ``True`` = valid.

    Returns
    -------
    np.ndarray
        Shape ``(B, dim)``.
    """
    pooled: list[np.ndarray] = []
    for mat, mask in zip(matrices, masks):
        valid = mask.cpu().numpy()
        if valid.sum() > 0:
            pooled.append(mat.cpu().numpy()[valid].mean(axis=0))
        else:
            pooled.append(np.zeros(mat.shape[-1], dtype=np.float32))
    return np.stack(pooled)


# ---------------------------------------------------------------------------
# Split-file reader
# ---------------------------------------------------------------------------

def read_split_file(path: str) -> pd.DataFrame:
    """Read a scaffold split TSV (optionally gzipped)."""
    if os.path.exists(path + ".gz"):
        return pd.read_csv(path + ".gz", sep="\t", compression="gzip")
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    raise FileNotFoundError(f"Split file not found: {path} (nor {path}.gz)")


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------

def build_matrix_dataloaders(
    dataset_type: str,
    embedding_name: str,
    scaffold_split_dir: str,
    batch_size: int = 32,
    dataset_source_filter: str | None = None,
    mode: str = "test",
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build train / val / test ``DataLoader`` instances from universal splits.

    Always reads the **universal** scaffold split files.  When
    *dataset_source_filter* is set (e.g. ``"human"``), rows are filtered
    by the ``dataset_source`` column before building tensors.

    *dataset_type* still controls which embedding directories to search
    for matrix ``.npy`` files (per-corpus storage is unchanged).

    When *mode* is ``"train"``, test data is **never loaded** — the
    returned ``test_loader`` is ``None``.

    Returns
    -------
    train_loader, val_loader, test_loader (or None when mode="train")
    """
    protein_dirs, ligand_dirs = _resolve_matrix_dirs(dataset_type, embedding_name)

    train_df = read_split_file(
        os.path.join(scaffold_split_dir, "scenarios/Sc", "universal_train.tsv")
    )
    val_df = read_split_file(
        os.path.join(scaffold_split_dir, "scenarios/Sc", "universal_val.tsv")
    )

    if mode == "test":
        test_df = read_split_file(
            os.path.join(scaffold_split_dir, "universal_test.tsv")
        )
    else:
        test_df = None

    if dataset_source_filter is not None:
        train_df = train_df[train_df["dataset_source"] == dataset_source_filter].reset_index(drop=True)
        val_df = val_df[val_df["dataset_source"] == dataset_source_filter].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df[test_df["dataset_source"] == dataset_source_filter].reset_index(drop=True)

    for df in (train_df, val_df, test_df):
        if df is not None and "label" not in df.columns:
            df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

    test_loader = (
        _make_loader(test_df, protein_dirs, ligand_dirs, batch_size, shuffle=False)
        if test_df is not None
        else None
    )

    return (
        _make_loader(train_df, protein_dirs, ligand_dirs, batch_size, shuffle=True),
        _make_loader(val_df, protein_dirs, ligand_dirs, batch_size, shuffle=False),
        test_loader,
    )


def _resolve_matrix_dirs(
    dataset_type: str,
    embedding_name: str,
) -> tuple[list[Path], list[Path]]:
    """Return lists of protein and ligand matrix directories.

    For ligands, searches both ``ligand_matrices/`` and ``molformer_matrix/``
    because naming conventions vary across machines and datasets.
    """
    if dataset_type in ("all",):
        base_paths = _EMBEDDING_BASE_PATHS_ALL
    else:
        base_paths = [EMBEDDING_BASE_PATH.format(dataset_type=dataset_type)]

    protein_dirs = [
        Path(bp) / embedding_name / "build" / "protein_matrices"
        for bp in base_paths
    ]
    ligand_dirs = []
    for bp in base_paths:
        build = Path(bp) / embedding_name / "build"
        ligand_dirs.append(build / "ligand_matrices")
        ligand_dirs.append(build / "molformer_matrix")
    return protein_dirs, ligand_dirs


def _make_loader(
    df: pd.DataFrame,
    protein_dirs: list[Path],
    ligand_dirs: list[Path],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    _validate_matrix_coverage(df, protein_dirs, ligand_dirs)
    return DataLoader(
        MatrixDataset(df, protein_dirs, ligand_dirs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_matrices,
        num_workers=0,
    )


def _validate_matrix_coverage(
    df: pd.DataFrame,
    protein_dirs: list[Path],
    ligand_dirs: list[Path],
) -> None:
    """Check that matrix files exist for all unique IDs in the DataFrame.

    Raises RuntimeError with a detailed listing of missing IDs
    when any protein or ligand matrix files are absent.
    """
    unique_seqs = df["seq_id"].unique()
    unique_chembls = df["chembl_id"].unique()

    missing_prot_ids: list[str] = [
        str(sid)
        for sid in unique_seqs
        if not any((d / f"{sid}_matrix.npy").exists() for d in protein_dirs)
    ]
    missing_lig_ids: list[str] = [
        str(cid)
        for cid in unique_chembls
        if not any(
            (d / f"{cid}_matrix.npy").exists() or (d / f"{cid}_molformer_matrix.npy").exists()
            for d in ligand_dirs
        )
    ]

    if missing_prot_ids or missing_lig_ids:
        lines = [
            "FATAL: Matrix coverage gap — cannot proceed with missing embeddings.",
            f"  Protein dirs searched: {[str(d) for d in protein_dirs]}",
            f"  Ligand dirs searched:  {[str(d) for d in ligand_dirs]}",
        ]
        if missing_prot_ids:
            lines.append(
                f"  Missing protein matrices: {len(missing_prot_ids)}/{len(unique_seqs)}"
            )
            preview = missing_prot_ids[:20]
            lines.append(f"    IDs: {preview}{'...' if len(missing_prot_ids) > 20 else ''}")
        if missing_lig_ids:
            lines.append(
                f"  Missing ligand matrices: {len(missing_lig_ids)}/{len(unique_chembls)}"
            )
            preview = missing_lig_ids[:20]
            lines.append(f"    IDs: {preview}{'...' if len(missing_lig_ids) > 20 else ''}")

        msg = "\n".join(lines)
        raise RuntimeError(msg)



# ---------------------------------------------------------------------------
# Train/feature-extract split (avoids train-set optimism in train mode)
# ---------------------------------------------------------------------------

def split_loader_for_feature_extraction(
    loader: DataLoader,
    seed: int,
    model_fraction: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """Split a DataLoader into two non-overlapping subsets.

    Used by levels with learned feature extractors (L1c, L3) in
    **train mode** to avoid train-set optimism: the attention model
    trains on *model_fraction* of the data, and KNN features are
    extracted from the remaining held-out portion.

    Uses stratified splitting to preserve class balance.

    Returns
    -------
    model_loader : DataLoader
        For training the attention pooling model.
    feature_loader : DataLoader
        For extracting features to train KNN/MLP (unseen by the model).
    """
    from torch.utils.data import DataLoader as _DL, Subset

    dataset = loader.dataset
    n = len(dataset)
    # Fast path: read labels from the DataFrame if available (avoids
    # loading .npy matrices just to extract a scalar label per row).
    if hasattr(dataset, '_df') and 'label' in dataset._df.columns:
        labels = dataset._df['label'].to_numpy(dtype=int)
    else:
        labels = np.array([dataset[i][2] for i in range(n)])

    # Stratified split using label distribution
    rng = np.random.RandomState(seed)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    split_pos = int(len(idx_pos) * model_fraction)
    split_neg = int(len(idx_neg) * model_fraction)

    model_idx = np.concatenate([idx_pos[:split_pos], idx_neg[:split_neg]])
    feat_idx = np.concatenate([idx_pos[split_pos:], idx_neg[split_neg:]])

    rng.shuffle(model_idx)
    rng.shuffle(feat_idx)

    bs = loader.batch_size or 32
    model_loader = _DL(
        Subset(dataset, model_idx.tolist()),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_matrices,
        num_workers=0,
    )
    feature_loader = _DL(
        Subset(dataset, feat_idx.tolist()),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_matrices,
        num_workers=0,
    )
    return model_loader, feature_loader
