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


def _load_frozen_mlp_selection_from_train(
    output_dir: str,
    cache_filename: str,
) -> dict[str, object] | None:
    """Load frozen MLP selection from corresponding train artifact for same seed."""
    test_token = f"{os.sep}test{os.sep}"
    train_token = f"{os.sep}train{os.sep}"
    if test_token not in output_dir:
        return None

    train_seed_dir = output_dir.replace(test_token, train_token, 1)
    train_cache_path = os.path.join(train_seed_dir, cache_filename)
    if not os.path.exists(train_cache_path):
        return None

    with open(train_cache_path) as fh:
        payload = json.load(fh)
    scaffold_key = next(iter(payload.keys()), None)
    if not scaffold_key:
        return None
    mlp_block = payload.get(scaffold_key, {}).get("MLP", {})
    selection = mlp_block.get("mlp_selection")
    return selection if isinstance(selection, dict) else None


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
        """Compute fingerprints from splits, train canonical KNN/MLP."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level1a_knn_mlp_results.json")
        strict_freeze = (
            self.mode == "test"
            and os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        )
        if os.path.exists(cache_path) and not self.force and not strict_freeze:
            tqdm.write(f"  Loading cached Level 1a results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)
        if os.path.exists(cache_path) and not self.force and strict_freeze:
            tqdm.write("  Strict test mode: ignoring cached Level 1a results and recomputing.")

        tqdm.write(f"  Computing Level 1a fingerprint features (seed {seed})...")

        fit_df, eval_df = self._load_splits()

        x_fit, y_fit = _prepare_fp_features(fit_df)
        x_eval, y_eval = _prepare_fp_features(eval_df)

        if len(x_fit) == 0 or len(x_eval) == 0:
            tqdm.write("  WARNING: Empty feature set after FP computation. Skipping.")
            return None

        # Sanitise
        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        frozen_selection = None
        if self.mode == "test":
            frozen_selection = _load_frozen_mlp_selection_from_train(
                output_dir=output_dir,
                cache_filename="level1a_knn_mlp_results.json",
            )
            if strict_freeze and frozen_selection is None:
                raise RuntimeError(
                    "Missing frozen train selection for Level 1a test run. "
                    "Run train phase first or set BENCHMARK_REQUIRE_TRAIN_SELECTION=0."
                )

        models = train_knn_mlp(
            x_fit,
            y_fit,
            x_eval,
            y_eval,
            seed,
            frozen_mlp_selection=frozen_selection,
        )

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

    def _load_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load fit / eval DataFrames from universal scaffold splits.

        In **train** mode (default): fit=train (80%), eval=val (10%).
        Test is never loaded.

        In **test** mode: fit=val (10%), eval=test (10%).
        """
        scaffold_dir = self.scaffold_split_dir
        source_filter = self._config.dataset_source_filter

        if self.mode == "train":
            fit_df = read_split_file(
                os.path.join(scaffold_dir, "scenarios/Sc", "universal_train.tsv")
            )
            eval_df = read_split_file(
                os.path.join(scaffold_dir, "scenarios/Sc", "universal_val.tsv")
            )
        else:  # test
            fit_df = read_split_file(
                os.path.join(scaffold_dir, "scenarios/Sc", "universal_val.tsv")
            )
            eval_df = read_split_file(
                os.path.join(scaffold_dir, "universal_test.tsv")
            )

        if source_filter is not None:
            fit_df = fit_df[fit_df["dataset_source"] == source_filter].reset_index(drop=True)
            eval_df = eval_df[eval_df["dataset_source"] == source_filter].reset_index(drop=True)

        for df in (fit_df, eval_df):
            if "label" not in df.columns:
                df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

        return fit_df, eval_df
