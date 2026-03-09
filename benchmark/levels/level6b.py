"""Level 6b — AttnPool + BAN + GRL + KNN/MLP (no cross-attention).

Parallel to L6a, but without the cross-attention encoder stage.  Protein
and ligand sequences are projected to a shared hidden space, then
AttentionPooling produces uni-modal summaries while BAN computes the
bilinear cross-modal interaction.  Both are concatenated into a
3*hidden_dim feature vector.

Pipeline:
  1.  Cluster training scaffolds (same as L5/L5b).
  2.  Train Level6bModel via ``run_single_analysis(model_variant="level6b")``.
  3.  Load checkpoint and extract AttnPool+BAN features.
  4.  Feed features into canonical KNN / MLP classifiers.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp
from benchmark.config import (
    METRICS_ORDER,
    MOLFORMER_DIM,
    PROTEIN_DIMS,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import build_matrix_dataloaders


class Level6bRunner(BaseLevelRunner):
    """BAN + GRL (no cross-attn) → canonical KNN/MLP."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def level_tag(self) -> str:
        return "level6b_ban"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        from crossattention_split_analysis.experiment import run_single_analysis

        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level6b_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 6b results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Training Level 6b encoder (seed {seed})...")

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
            learning_rate=1e-3,
            hidden_dim=256,
            num_cross_attn_layers=0,
            num_heads=8,
            dropout=0.2,
            classifier_dropout=0.2,
            classification_only=True,
            use_molformer_ligand=True,
            scaffold_split_dir=self.scaffold_split_dir,
            model_variant="level6b",
            optimize_threshold=False,
            fixed_threshold=0.5,
            weight_decay=0.01,
        )

        tqdm.write("  Extracting BAN-fused representations...")

        try:
            x_val, y_val, x_test, y_test = self._extract_features(
                output_dir=output_dir, seed=seed,
            )
        except Exception as exc:
            tqdm.write(f"  WARNING: Feature extraction failed: {exc}")
            return self._fallback_from_training_results(_training_results)

        for name, arr in [("val", x_val), ("test", x_test)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf → replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        models = train_knn_mlp(x_val, y_val, x_test, y_test, seed)

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 6b (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self, output_dir: str, seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from crossattention_split_analysis.config import (
            SUPPORTED_EMBEDDINGS as CA_SUPPORTED_EMBEDDINGS,
        )
        from crossattention_split_analysis.models.level6b import Level6bModel
        from crossattention_split_analysis.utils import get_checkpoint_path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_emb = CA_SUPPORTED_EMBEDDINGS.get(self.embedding_name, self.embedding_name)
        protein_dim = PROTEIN_DIMS.get(full_emb, 640)

        short_emb = full_emb.replace("esm2_", "").replace("_UR50D", "")
        prefix = f"{self.dataset}_ban6b_molformer_{short_emb}_seed{seed}_"
        checkpoint_path = get_checkpoint_path(output_dir, prefix, "Split by Scaffold")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        best_state = checkpoint.get("best_model_state")
        if best_state is None:
            raise RuntimeError("Checkpoint does not contain best_model_state")

        model = Level6bModel(
            protein_input_dim=protein_dim,
            ligand_input_dim=MOLFORMER_DIM,
            hidden_dim=256,
            num_heads=8,
            dropout=0.2,
            classifier_dropout=0.2,
            num_domains=16,
        )
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        _train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=full_emb,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=64,
        )

        x_val, y_val = self._collect_features(model, val_loader, device)
        x_test, y_test = self._collect_features(model, test_loader, device)

        return x_val, y_val, x_test, y_test

    @staticmethod
    @torch.no_grad()
    def _collect_features(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_from_training_results(self, results: Optional[Dict]) -> Optional[Dict]:
        if results is None:
            return None
        mlp_metrics = self._extract_mlp_metrics(results)
        tqdm.write("  WARNING: Using model head metrics as fallback")
        sc_key = self._find_scaffold_key(results) or "Split by Scaffold"
        return {sc_key: {"KNN": mlp_metrics, "MLP": mlp_metrics}}

    def _extract_mlp_metrics(self, results: Dict) -> Dict[str, float]:
        sc_key = self._find_scaffold_key(results) or next(iter(results), "")
        sc_data = results.get(sc_key, {})
        if not isinstance(sc_data, dict):
            return self._empty_metrics()
        for nested_key in ("Level6b", "level6b"):
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
