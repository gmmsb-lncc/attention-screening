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
from benchmark.levels.se3_features import SE3FeatureLoader, build_se3_loader


# ---------------------------------------------------------------------------
# Feature extraction (ligand-only mean pooling)
# ---------------------------------------------------------------------------

def _extract_ligand_features(
    loader: DataLoader,
    se3_loader: SE3FeatureLoader | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract mean-pooled **ligand-only** features from a dataloader."""
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    se3_logged = False

    for batch in loader:
        ligand_pooled = mean_pool(batch["ligand_matrix"], batch["ligand_mask"])
        if se3_loader is not None:
            se3_batch = se3_loader.get_batch(batch["chembl_id"])
            if se3_batch.shape[1] > 0:
                if not se3_logged:
                    sem_dim = int(ligand_pooled.shape[1])
                    se3_dim = int(se3_batch.shape[1])
                    fused_dim = sem_dim + se3_dim
                    tqdm.write(
                        f"  [SE3] Active fusion (Level 1b): semantic_dim={sem_dim}, se3_dim={se3_dim}, fused_ligand_dim={fused_dim}"
                    )
                    se3_logged = True
                ligand_pooled = np.concatenate([ligand_pooled, se3_batch], axis=-1)
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

        se3_loader = build_se3_loader(
            feature_dir=self._config.se3_features_dir,
            use_se3_ligand=self._config.use_se3_ligand,
            default_feature_dirs=self._config.resolved_se3_feature_dirs,
        )
        if self._config.use_se3_ligand and se3_loader is None:
            tqdm.write("  WARNING: SE3 ligand fusion requested, but no valid .npy structural vectors were found. Proceeding with semantic ligand features only.")
        elif se3_loader is not None:
            tqdm.write(
                f"  [SE3] Structural vectors loaded: dim={se3_loader.dim}. Ligand concatenation enabled."
            )

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
        x_fit, y_fit = _extract_ligand_features(split_selection.fit, se3_loader=se3_loader)
        x_eval, y_eval = _extract_ligand_features(split_selection.eval, se3_loader=se3_loader)

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
