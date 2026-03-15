"""Level 4 — Cross-Attention encoder + KNN/MLP.

Trains the full DT-Kinase model (CNN encoders + bidirectional
cross-attention) as a *feature extractor*, then feeds the learned
representations into the **canonical** KNN and MLP classifiers
(same as Levels 1–3) for a scientifically fair comparison.

Input matrices are the **same** as those used by Levels 2 and 3.
The difference is the encoding strategy:
  - Level 2: mean pooling (zero parameters)
  - Level 3: projection + attention pooling (learned query)
  - Level 4: projection + cross-attention + attention pooling (full DT-Kinase)

Training protocol (consistent with all levels):
  - Cross-attention model trained on the **training** split
    (validation split for early stopping on MCC).
  - Features extracted from the **validation** split — the model was
    *not* directly trained on val, only used it for model selection,
    so val features are free of train-set optimism.
  - KNN/MLP classifiers trained on val features.
  - Evaluation on the hold-out **test** split.

Pipeline:
  1. Train cross-attention model via ``run_single_analysis``
     (saves checkpoint).
  2. Load the best-model checkpoint.
  3. Run inference with ``return_features=True`` on val / test data
     to obtain the pre-head representation vectors.
  4. Feed those vectors into ``benchmark.classifiers.train_knn_mlp``.

Classifier note: KNN and MLP are provided by ``benchmark.classifiers``
to guarantee identical hyperparameters across all four levels.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp
from benchmark.config import (
    EMBEDDING_BASE_PATH,
    METRICS_ORDER,
    MOLFORMER_DIM,
    PROTEIN_DIMS,
    SUPPORTED_EMBEDDINGS,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import build_matrix_dataloaders


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


class Level4Runner(BaseLevelRunner):
    """Full Cross-Attention encoder → canonical KNN/MLP."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def knn_is_deterministic(self) -> bool:
        return False  # Learned feature extractor → KNN input varies per seed

    @property
    def level_tag(self) -> str:
        return "level4_crossatt"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Train cross-attention model, extract features, run canonical KNN/MLP."""
        from crossattention_split_analysis.experiment import run_single_analysis

        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level4_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 4 results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        # --- Step 1: Train the cross-attention model ---------------------
        tqdm.write(f"  Training Level 4 cross-attention encoder (seed {seed})...")

        _training_results = run_single_analysis(
            embedding_name=self.embedding_name,
            dataset_type=self.dataset,
            output_dir=output_dir,
            seeds=[seed],
            force=self.force,
            scenarios=["scaffold"],
            num_epochs=self._config.epochs,
            patience=self._config.resolved_patience or 10,
            batch_size=32,
            learning_rate=1e-4,
            hidden_dim=384,
            num_cross_attn_layers=1,
            num_heads=12,
            dropout=0.25,
            classifier_dropout=0.25,
            classification_only=True,
            use_molformer_ligand=True,
            scaffold_split_dir=self.scaffold_split_dir,
            model_variant="level5_lite",
            optimize_threshold=False,
            fixed_threshold=0.5,
            weight_decay=0.05,
        )

        # --- Step 2: Extract features from trained model -----------------
        tqdm.write("  Extracting cross-attention representations...")

        try:
            x_fit, y_fit, x_eval, y_eval = self._extract_features(
                output_dir=output_dir,
                seed=seed,
            )
        except Exception as exc:
            tqdm.write(f"  WARNING: Feature extraction failed: {exc}")
            tqdm.write("  Falling back to training metrics only.")
            return self._fallback_from_training_results(_training_results)

        # Sanitise features
        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Step 3: Train canonical KNN/MLP on fit features -------------
        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        frozen_selection = None
        if self.mode == "test":
            frozen_selection = _load_frozen_mlp_selection_from_train(
                output_dir=output_dir,
                cache_filename="level4_knn_mlp_results.json",
            )
            strict_freeze = os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            if strict_freeze and frozen_selection is None:
                raise RuntimeError(
                    "Missing frozen train selection for Level 4 test run. "
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
            f"  Level 4 (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # Feature extraction from trained cross-attention model
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        output_dir: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load checkpoint, rebuild model/data, extract pre-head features.

        Uses ``build_matrix_dataloaders`` from ``matrix_utils`` — the
        **same** data pipeline as Levels 2 and 3, ensuring that all
        matrix-based levels receive identical inputs.

        In **train** mode, features are extracted from the training split
        (for KNN/MLP fitting) and validation split (for evaluation).
        In **test** mode, features are extracted from validation (fitting)
        and test (evaluation).

        Returns
        -------
        x_fit, y_fit, x_eval, y_eval : np.ndarray
            Feature arrays from the cross-attention encoder.
        """
        from crossattention_split_analysis.config import (
            SUPPORTED_EMBEDDINGS as CA_SUPPORTED_EMBEDDINGS,
        )
        from crossattention_split_analysis.models.level5_lite import Level5LiteModel
        from crossattention_split_analysis.utils import get_checkpoint_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Resolve embedding name / dims ---
        full_emb = CA_SUPPORTED_EMBEDDINGS.get(self.embedding_name, self.embedding_name)
        protein_dim = PROTEIN_DIMS.get(full_emb, 640)

        # --- Locate checkpoint ---
        short_emb = full_emb.replace("esm2_", "").replace("_UR50D", "")
        prefix = f"{self.dataset}_molformer_{short_emb}_seed{seed}_"
        checkpoint_path = get_checkpoint_path(output_dir, prefix, "Split by Scaffold")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # --- Load best model state ---
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        best_state = checkpoint.get("best_model_state")
        if best_state is None:
            raise RuntimeError("Checkpoint does not contain best_model_state")

        model = Level5LiteModel(
            protein_input_dim=protein_dim,
            ligand_input_dim=MOLFORMER_DIM,
            hidden_dim=384,
            num_cross_attn_layers=1,
            num_heads=12,
            dropout=0.25,
            classifier_dropout=0.25,
        )
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        # --- Build dataloaders (same as L2/L3 — shared matrix_utils) ---
        # For dataset="all" this correctly concatenates human + non_human
        # splits and searches matrices in both directories.
        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=full_emb,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=64,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self._config.mode,
        )

        # --- Forward pass to collect features based on mode ---
        if self._config.mode == "train":
            x_fit, y_fit = self._collect_features(model, train_loader, device)
            x_eval, y_eval = self._collect_features(model, val_loader, device)
        else:
            x_fit, y_fit = self._collect_features(model, val_loader, device)
            x_eval, y_eval = self._collect_features(model, test_loader, device)

        return x_fit, y_fit, x_eval, y_eval

    @staticmethod
    @torch.no_grad()
    def _collect_features(
        model: "torch.nn.Module",
        loader: "torch.utils.data.DataLoader",
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run forward passes and collect pre-head feature vectors."""
        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in loader:
            protein = batch["protein_matrix"].to(device)
            ligand = batch["ligand_matrix"].to(device)
            protein_mask = batch.get("protein_mask")
            ligand_mask = batch.get("ligand_mask")
            if protein_mask is not None:
                protein_mask = protein_mask.to(device)
            if ligand_mask is not None:
                ligand_mask = ligand_mask.to(device)

            output = model(
                protein, ligand, protein_mask, ligand_mask,
                return_features=True,
            )
            features = output["features"].cpu().numpy()
            labels = batch["label"].numpy()

            all_features.append(features)
            all_labels.append(labels)

        return np.concatenate(all_features), np.concatenate(all_labels)

    # ------------------------------------------------------------------
    # Fallback — only used when feature extraction fails
    # ------------------------------------------------------------------

    def _fallback_from_training_results(
        self,
        results: Optional[Dict],
    ) -> Optional[Dict]:
        """Extract metrics directly from training results as a last resort."""
        if results is None:
            return None

        mlp_metrics = self._extract_mlp_metrics(results)
        # When falling back, KNN gets the same metrics as MLP
        # (marked clearly so the user knows this is NOT a real KNN evaluation)
        tqdm.write("  WARNING: Using model head metrics as fallback (not canonical classifiers)")
        sc_key = self._find_scaffold_key(results) or "Split by Scaffold"
        return {sc_key: {"KNN": mlp_metrics, "MLP": mlp_metrics}}

    def _extract_mlp_metrics(self, results: Dict) -> Dict[str, float]:
        """Unwrap metrics from nested cross-attention results."""
        sc_key = self._find_scaffold_key(results) or next(iter(results), "")
        sc_data = results.get(sc_key, {})

        if not isinstance(sc_data, dict):
            return self._empty_metrics()

        for nested_key in ("Level5-Lite", "level5_lite"):
            if nested_key in sc_data:
                return self._ensure_all_metrics(sc_data[nested_key])

        if "accuracy" in sc_data or "mcc" in sc_data:
            return self._ensure_all_metrics(sc_data)

        for value in sc_data.values():
            if isinstance(value, dict) and ("mcc" in value or "accuracy" in value):
                return self._ensure_all_metrics(value)

        return self._empty_metrics()

    @staticmethod
    def _ensure_all_metrics(metrics: Dict) -> Dict[str, float]:
        return {m: float(metrics.get(m, 0.0)) for m in METRICS_ORDER}

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {m: 0.0 for m in METRICS_ORDER}


# ---------------------------------------------------------------------------
# Result loading fallback
# ---------------------------------------------------------------------------

def load_crossattention_results(
    level_dir: str,
    dataset: str,
    embedding_short: str,
) -> Optional[Dict]:
    """Load cached ``crossattention_analysis_results.json`` if available."""
    full_name = SUPPORTED_EMBEDDINGS.get(embedding_short, embedding_short)
    short_name = full_name.replace("esm2_", "").replace("_UR50D", "")
    prefix = f"{dataset}_molformer_{short_name}_"
    json_path = os.path.join(level_dir, f"{prefix}crossattention_analysis_results.json")

    if not os.path.exists(json_path):
        candidates = glob.glob(os.path.join(level_dir, "*crossattention_analysis_results.json"))
        if candidates:
            json_path = candidates[0]
        else:
            return None

    try:
        with open(json_path) as fh:
            data = json.load(fh)
        return data.get("model_results")
    except (json.JSONDecodeError, KeyError):
        return None
