"""Level 2 — Embedding matrices + mean pooling + KNN/MLP.

Loads per-residue ESM-2 protein matrices and per-token MoLFormer ligand
matrices, applies **mean pooling** for fixed-size representations, then
trains KNN and MLP classifiers.

This is the simplest matrix-based level — no learned pooling, just
averaging over the sequence dimension.

Input matrices are the **same** as those used by Levels 3 and 4.
The only difference is the pooling strategy:
  - Level 2: mean pooling (no parameters)
  - Level 3: learned attention pooling
  - Level 4: full cross-attention encoder

Training protocol (consistent with all levels):
  - Features extracted from the **validation** split.
  - KNN/MLP classifiers trained on val features.
  - Evaluation on the hold-out **test** split.

Classifier note: KNN and MLP are provided by ``benchmark.classifiers``
to guarantee identical hyperparameters across all four levels.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp
from benchmark.config import BenchmarkConfig
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import (
    build_matrix_dataloaders,
    mean_pool,
)


# ---------------------------------------------------------------------------
# Feature extraction (mean pooling — no training required)
# ---------------------------------------------------------------------------

def _extract_features(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled protein+ligand features from a dataloader."""
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        protein_pooled = mean_pool(batch["protein_matrix"], batch["protein_mask"])
        ligand_pooled = mean_pool(batch["ligand_matrix"], batch["ligand_mask"])
        combined = np.nan_to_num(
            np.concatenate([protein_pooled, ligand_pooled], axis=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        all_features.append(combined)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Level 2 runner
# ---------------------------------------------------------------------------

class Level2Runner(BaseLevelRunner):
    """Embedding matrices -> mean pooling -> KNN/MLP."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def level_tag(self) -> str:
        return "level2_meanpool"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Extract features via mean pooling and train KNN/MLP for one seed."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level2_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 2 results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Extracting Level 2 features (seed {seed})...")

        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
        )

        if self.mode == "train":
            tqdm.write("  Mean-pooling protein + ligand matrices (train + val)...")
            x_fit, y_fit = _extract_features(train_loader)
            x_eval, y_eval = _extract_features(val_loader)
        else:
            tqdm.write("  Mean-pooling protein + ligand matrices (val + test)...")
            x_fit, y_fit = _extract_features(val_loader)
            x_eval, y_eval = _extract_features(test_loader)

        # Sanitize features
        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            bad = np.isnan(arr).sum() + np.isinf(arr).sum()
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        models = train_knn_mlp(x_fit, y_fit, x_eval, y_eval, seed)

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 2 (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result
