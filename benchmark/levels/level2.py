"""Level 2 — Embedding vectors + KNN/MLP.

Uses mean-pooled ESM-2 protein embeddings and attention-pooled MoLFormer
ligand embeddings as features, then trains KNN and MLP classifiers.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.levels.base import BaseLevelRunner


class Level2Runner(BaseLevelRunner):
    """Embedding vector KNN/MLP classifiers."""

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        custom_protein_embedding_dir: Optional[str] = None,
        custom_ligand_embedding_dir: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._custom_protein_dir = custom_protein_embedding_dir
        self._custom_ligand_dir = custom_ligand_embedding_dir

    @property
    def level_tag(self) -> str:
        suffix = "_finetuned" if self._custom_protein_dir or self._custom_ligand_dir else ""
        return f"level2_embedding{suffix}"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Train KNN/MLP on embedding vectors for a single seed."""
        from split_comparison_analysis import run_single_dataset

        result = run_single_dataset(
            dataset_type=self.dataset,
            output_dir=output_dir,
            force=self.force,
            seed=seed,
            scenarios=["scaffold"],
            scaffold_split_dir=self.scaffold_split_dir,
            feature_type="embedding",
            embedding_name=self.embedding_name,
            custom_protein_embedding_dir=self._custom_protein_dir,
            custom_ligand_embedding_dir=self._custom_ligand_dir,
        )

        if result is None:
            result = self._load_cached_results(output_dir)

        return result
