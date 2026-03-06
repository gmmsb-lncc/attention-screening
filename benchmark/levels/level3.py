"""Level 3 — Embedding matrices + mean pooling + KNN/MLP.

Loads per-residue ESM-2 protein matrices and per-token MoLFormer ligand
matrices, applies **mean pooling** for fixed-size representations, then
trains KNN and MLP classifiers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from benchmark.config import (
    EMBEDDING_BASE_PATH,
    PCHEMBL_ACTIVITY_THRESHOLD,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner


# ---------------------------------------------------------------------------
# Matrix dataset
# ---------------------------------------------------------------------------

class _MatrixDataset(Dataset):
    """Lazy loader for paired protein / ligand matrices."""

    def __init__(
        self,
        df: "pd.DataFrame",
        protein_matrix_dir: Path,
        ligand_matrix_dir: Path,
    ) -> None:
        self._df = df
        self._protein_dir = protein_matrix_dir
        self._ligand_dir = ligand_matrix_dir

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        row = self._df.iloc[idx]
        seq_id = row["seq_id"]
        chembl_id = row["chembl_id"]
        label = row["label"]

        protein_mat = self._load_npy(
            self._protein_dir / f"{seq_id}_matrix.npy",
            fallback_shape=(100, 320),
        )
        ligand_mat = self._load_npy(
            self._ligand_dir / f"{chembl_id}_molformer_matrix.npy",
            fallback_shape=(50, 768),
        )
        return protein_mat, ligand_mat, label, seq_id, chembl_id

    @staticmethod
    def _load_npy(path: Path, fallback_shape: tuple[int, int]) -> np.ndarray:
        if path.exists():
            return np.load(path).astype(np.float32)
        return np.zeros(fallback_shape, dtype=np.float32)


def _collate_matrices(batch: list) -> dict:
    """Pad protein and ligand matrices to the maximum length in the batch."""
    protein_mats, ligand_mats, labels, seq_ids, chembl_ids = zip(*batch)

    protein_batch, protein_mask = _pad_matrices(protein_mats)
    ligand_batch, ligand_mask = _pad_matrices(ligand_mats)

    return {
        "protein_matrix": torch.from_numpy(protein_batch),
        "ligand_matrix": torch.from_numpy(ligand_batch),
        "protein_mask": torch.from_numpy(protein_mask),
        "ligand_mask": torch.from_numpy(ligand_mask),
        "label": torch.tensor(labels, dtype=torch.float32),
        "seq_id": seq_ids,
        "chembl_id": chembl_ids,
    }


def _pad_matrices(
    matrices: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad a collection of 2-D matrices to a common length."""
    max_len = max(m.shape[0] for m in matrices)
    dim = matrices[0].shape[1]
    padded = np.zeros((len(matrices), max_len, dim), dtype=np.float32)
    mask = np.ones((len(matrices), max_len), dtype=bool)
    for i, mat in enumerate(matrices):
        padded[i, : mat.shape[0], :] = mat
        mask[i, mat.shape[0] :] = False
    return padded, mask


# ---------------------------------------------------------------------------
# Mean-pooling feature extraction (no training required)
# ---------------------------------------------------------------------------

def _mean_pool(matrices: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    """Mean-pool over valid positions for each sample in the batch."""
    pooled: list[np.ndarray] = []
    for mat, mask in zip(matrices, masks):
        valid = mask.cpu().numpy()
        if valid.sum() > 0:
            pooled.append(mat.cpu().numpy()[valid].mean(axis=0))
        else:
            pooled.append(np.zeros(mat.shape[-1], dtype=np.float32))
    return np.stack(pooled)


def _extract_features(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled protein+ligand features from a dataloader."""
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        protein_pooled = _mean_pool(batch["protein_matrix"], batch["protein_mask"])
        ligand_pooled = _mean_pool(batch["ligand_matrix"], batch["ligand_mask"])
        combined = np.nan_to_num(
            np.concatenate([protein_pooled, ligand_pooled], axis=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        all_features.append(combined)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def _build_dataloaders(
    dataset_type: str,
    embedding_name: str,
    scaffold_split_dir: str,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test dataloaders from scaffold splits."""
    import pandas as pd

    build_dir = Path(
        EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
        embedding_name,
        "build",
    )
    protein_dir = build_dir / "protein_matrices"
    ligand_dir = build_dir / "molformer_matrix"

    def read_split(path: str) -> "pd.DataFrame":
        if os.path.exists(path + ".gz"):
            return pd.read_csv(path + ".gz", sep="\t", compression="gzip")
        return pd.read_csv(path, sep="\t")

    train_df = read_split(os.path.join(scaffold_split_dir, "scenarios/Sc", f"{dataset_type}_train.tsv"))
    val_df = read_split(os.path.join(scaffold_split_dir, "scenarios/Sc", f"{dataset_type}_val.tsv"))
    test_df = read_split(os.path.join(scaffold_split_dir, f"{dataset_type}_test.tsv"))

    for df in (train_df, val_df, test_df):
        if "label" not in df.columns:
            df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

    def _make_loader(df: "pd.DataFrame", shuffle: bool = False) -> DataLoader:
        return DataLoader(
            _MatrixDataset(df, protein_dir, ligand_dir),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_matrices,
        )

    return _make_loader(train_df), _make_loader(val_df), _make_loader(test_df)


# ---------------------------------------------------------------------------
# KNN / MLP training helper
# ---------------------------------------------------------------------------

def _train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Train KNN and MLP classifiers and return metric dicts for each."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_proba)),
        }

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="cosine", n_jobs=-1)
    knn.fit(x_train_sc, y_train)
    knn_metrics = _metrics(y_test, knn.predict(x_test_sc), knn.predict_proba(x_test_sc)[:, 1])

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )
    mlp.fit(x_train_sc, y_train)
    mlp_metrics = _metrics(y_test, mlp.predict(x_test_sc), mlp.predict_proba(x_test_sc)[:, 1])

    return {"KNN": knn_metrics, "MLP": mlp_metrics}


# ---------------------------------------------------------------------------
# Level 3 runner
# ---------------------------------------------------------------------------

class Level3Runner(BaseLevelRunner):
    """Embedding matrices → mean pooling → KNN/MLP."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def level_tag(self) -> str:
        return "level3_mat"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Extract features via mean pooling and train KNN/MLP for one seed."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level3_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 3 results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Extracting Level 3 features (seed {seed})...")

        train_loader, _val_loader, test_loader = _build_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
        )

        tqdm.write("  Mean-pooling protein + ligand matrices...")
        x_train, y_train = _extract_features(train_loader)
        x_test, y_test = _extract_features(test_loader)

        # Sanitize features
        for name, arr in [("train", x_train), ("test", x_test)]:
            bad = np.isnan(arr).sum() + np.isinf(arr).sum()
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values → replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        tqdm.write("  Training KNN + MLP...")
        models = _train_and_evaluate(x_train, y_train, x_test, y_test, seed)

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 3 (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result
