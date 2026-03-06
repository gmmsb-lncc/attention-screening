"""Level 1 — Fingerprint + KNN/MLP baseline.

Uses classical molecular fingerprints (no PLM embeddings) as features
and trains KNN and MLP classifiers via ``split_comparison_analysis``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.levels.base import BaseLevelRunner


class Level1Runner(BaseLevelRunner):
    """Fingerprint-based baseline: KNN and MLP classifiers."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def level_tag(self) -> str:
        return "level1_fingerprint"

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
        """Train KNN/MLP on fingerprint features for a single seed."""
        from split_comparison_analysis import run_single_dataset

        result = run_single_dataset(
            dataset_type=self.dataset,
            output_dir=output_dir,
            force=self.force,
            seed=seed,
            scenarios=["scaffold"],
            scaffold_split_dir=self.scaffold_split_dir,
            feature_type="fingerprint",
        )

        if result is None:
            result = self._load_cached_results(output_dir)

        return result
