"""Level 1 — Fingerprint + KNN/MLP baseline.

Uses Morgan fingerprints (radius 2, 1024 bits) as compound descriptors.
**No one-hot kinase encoding** is included — features are compound-only,
ensuring the only independent variable across levels is the molecular
representation strategy.

Training protocol (consistent with Levels 2–4):
  - Classifiers trained on **validation-split** features.
  - Evaluation on the hold-out **test** split.
  - Classifiers provided by ``benchmark.classifiers`` (same as all levels).

This eliminates the threshold-optimisation advantage that the legacy
``split_comparison_analysis`` path provided and removes the one-hot
kinase information asymmetry.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp
from benchmark.config import PCHEMBL_ACTIVITY_THRESHOLD, BenchmarkConfig
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import read_split_file


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def _compute_morgan_fps(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> tuple[np.ndarray, list[int]]:
    """Compute Morgan fingerprints for a list of SMILES strings.

    Returns
    -------
    fps : np.ndarray
        Shape ``(n_valid, n_bits)``, float32.
    valid_indices : list[int]
        Positional indices into *smiles_list* that produced valid molecules.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: list[np.ndarray] = []
    valid_idx: list[int] = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            fps.append(fp)
            valid_idx.append(i)

    if fps:
        return np.stack(fps), valid_idx
    return np.empty((0, n_bits), dtype=np.float32), []


def _prepare_fp_features(
    df: pd.DataFrame,
    n_bits: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fingerprint features and labels from a DataFrame.

    Returns
    -------
    x : np.ndarray   — shape ``(n_valid, n_bits)``
    y : np.ndarray   — shape ``(n_valid,)``
    """
    smiles = df["canonical_smiles"].tolist()
    fps, valid_idx = _compute_morgan_fps(smiles, n_bits=n_bits)
    if len(fps) == 0:
        return np.empty((0, n_bits), dtype=np.float32), np.array([])
    labels = df.iloc[valid_idx]["label"].values.astype(np.float32)
    return fps, labels


# ---------------------------------------------------------------------------
# Level 1 runner
# ---------------------------------------------------------------------------

class Level1Runner(BaseLevelRunner):
    """Fingerprint-based baseline: canonical KNN and MLP classifiers."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def level_tag(self) -> str:
        return "level1a_fingerprint"

    def _uses_embedding(self) -> bool:
        """Level 1 uses fingerprints, not embeddings."""
        return False

    def output_dir_for_level(self) -> str:
        return os.path.join(
            self._config.resolved_output_dir,
            self.level_tag,
            self.dataset,
        )

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Compute fingerprints from val/test splits, train canonical KNN/MLP."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level1a_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 1a results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Computing Level 1a fingerprint features (seed {seed})...")

        val_df, test_df = self._load_val_test_splits()

        x_val, y_val = _prepare_fp_features(val_df)
        x_test, y_test = _prepare_fp_features(test_df)

        if len(x_val) == 0 or len(x_test) == 0:
            tqdm.write("  WARNING: Empty feature set after FP computation. Skipping.")
            return None

        # Sanitise
        for name, arr in [("val", x_val), ("test", x_test)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        models = train_knn_mlp(x_val, y_val, x_test, y_test, seed)

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 1a (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # Split loading
    # ------------------------------------------------------------------

    def _load_val_test_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load validation and test DataFrames from universal scaffold splits.

        Always reads ``universal_val.tsv`` / ``universal_test.tsv`` and
        filters by ``dataset_source`` when ``--dataset`` is ``human`` or
        ``non_human``.
        """
        scaffold_dir = self.scaffold_split_dir

        val_df = read_split_file(
            os.path.join(scaffold_dir, "scenarios/Sc", "universal_val.tsv")
        )
        test_df = read_split_file(
            os.path.join(scaffold_dir, "universal_test.tsv")
        )

        # Filter by corpus when a specific dataset is requested
        source_filter = self._config.dataset_source_filter
        if source_filter is not None:
            val_df = val_df[val_df["dataset_source"] == source_filter].reset_index(drop=True)
            test_df = test_df[test_df["dataset_source"] == source_filter].reset_index(drop=True)

        for df in (val_df, test_df):
            if "label" not in df.columns:
                df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

        return val_df, test_df
