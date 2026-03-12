"""Level 1b — Ligand embedding + mean pooling + KNN/MLP.

Uses the **same** MoLFormer per-token ligand matrices as Levels 2–4,
but applies **mean pooling** over the token dimension to produce a
fixed-size 768-d vector — **ligand only, no protein**.

This level isolates the contribution of *learned molecular embeddings*
vs classical fingerprints (Level 1a) while keeping the compound-only
constraint.

Comparison axes:
  - **1a vs 1b**: Classical fingerprint vs learned embedding
    (both compound-only, simple aggregation)
  - **1b vs 2**: Compound-only vs compound+protein (both mean pooling)

Training protocol (consistent with active levels):
    - In ``train`` mode: fit on train features, evaluate on val features.
    - In ``test`` mode: fit on val features, evaluate on test features.

Classifier note: KNN and MLP are provided by ``benchmark.classifiers``
to guarantee identical hyperparameters across all levels.
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
from benchmark.levels.protocol import sanitize_features, select_fit_eval


# ---------------------------------------------------------------------------
# Feature extraction (ligand-only mean pooling)
# ---------------------------------------------------------------------------

def _extract_ligand_features(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled **ligand-only** features from a dataloader."""
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        ligand_pooled = mean_pool(batch["ligand_matrix"], batch["ligand_mask"])
        ligand_pooled = np.nan_to_num(
            ligand_pooled, nan=0.0, posinf=0.0, neginf=0.0,
        )
        all_features.append(ligand_pooled)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Level 1b runner
# ---------------------------------------------------------------------------

class Level1bRunner(BaseLevelRunner):
    """Ligand embedding -> mean pooling -> KNN/MLP (no protein)."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def level_tag(self) -> str:
        return "level1b_ligmean"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Extract ligand-only mean-pooled features and train KNN/MLP."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level1b_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 1b results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Extracting Level 1b ligand features (seed {seed})...")

        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
            ligand_model=self._config.ligand_model,
        )

        split_selection = select_fit_eval(self.mode, train_loader, val_loader, test_loader)
        tqdm.write(
            f"  Mean-pooling ligand matrices ({split_selection.fit_name} + {split_selection.eval_name})..."
        )
        x_fit, y_fit = _extract_ligand_features(split_selection.fit)
        x_eval, y_eval = _extract_ligand_features(split_selection.eval)

        # Sanitize features
        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            arr_sanitized, bad = sanitize_features(arr)
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = arr_sanitized

        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        models = train_knn_mlp(x_fit, y_fit, x_eval, y_eval, seed)

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 1b (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result
