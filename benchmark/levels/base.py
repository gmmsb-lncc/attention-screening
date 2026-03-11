"""Abstract base class and shared logic for all level runners.

Provides the ``BaseLevelRunner`` interface (Open/Closed principle) and
a reusable multi-seed aggregation helper.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from benchmark.config import METRICS_ORDER, BenchmarkConfig


class BaseLevelRunner(ABC):
    """Abstract base for every benchmark level.

    Subclasses must implement ``run_single_seed`` and ``level_tag``.
    Multi-seed aggregation is handled here so that every level
    reports results in the same format.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def level_tag(self) -> str:
        """Short identifier used in directory names (e.g. ``'level1_fingerprint'``)."""

    @abstractmethod
    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Run training / evaluation for one seed.

        Must return a dict of the form::

            {"Split by Scaffold": {"KNN": {metric: val, ...}, "MLP": {metric: val, ...}}}

        or ``None`` on failure.
        """

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    @property
    def dataset(self) -> str:
        return self._config.dataset

    @property
    def embedding_name(self) -> str:
        return self._config.embedding_name

    @property
    def embedding_short(self) -> str:
        return self._config.embedding

    @property
    def seeds(self) -> List[int]:
        return self._config.resolved_seeds

    @property
    def force(self) -> bool:
        return self._config.force

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def scaffold_split_dir(self) -> str:
        return self._config.scaffold_split_dir

    def output_dir_for_level(self) -> str:
        """Build the output directory for this level."""
        return os.path.join(
            self._config.resolved_output_dir,
            f"{self.level_tag}_{self.embedding_short}" if self._uses_embedding() else self.level_tag,
            self.dataset,
        )

    # ------------------------------------------------------------------
    # Multi-seed orchestration (Template Method pattern)
    # ------------------------------------------------------------------

    def run(self, **kwargs: object) -> Optional[Dict]:
        """Run all seeds, aggregate mean ± std, persist results."""
        level_dir = self.output_dir_for_level()
        tqdm.write(f"  Output: {level_dir}")

        seed_results_per_model: Dict[str, Dict[str, List[float]]] = {}

        for idx, seed in enumerate(self.seeds):
            seed_dir = os.path.join(level_dir, f"seed_{seed}")
            tqdm.write(f"  Seed {idx + 1}/{len(self.seeds)}: {seed}")

            result = self.run_single_seed(seed=seed, output_dir=seed_dir, **kwargs)

            if result is None:
                result = self._load_cached_results(seed_dir)

            if result is None:
                tqdm.write(f"    WARNING: seed {seed} returned no results.")
                continue

            self._accumulate_seed(result, seed_results_per_model)

        if not seed_results_per_model:
            return None

        aggregated = self._aggregate(seed_results_per_model)
        self._save_aggregated(level_dir, aggregated)

        return {"Split by Scaffold": aggregated}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _uses_embedding(self) -> bool:
        """Whether this level's directory needs the embedding shorthand."""
        return True

    @staticmethod
    def _find_scaffold_key(results: Dict) -> Optional[str]:
        """Find the scaffold scenario key in a results dict."""
        for key in results:
            if "scaffold" in key.lower():
                return key
        return next(iter(results), None) if results else None

    def _accumulate_seed(
        self,
        result: Dict,
        accumulator: Dict[str, Dict[str, List[float]]],
    ) -> None:
        """Extract per-model metrics from a single seed and append.

        KNN is deterministic (no random component) so its results are
        identical across seeds.  Only the **first** seed is accumulated
        for KNN to avoid redundant computation and a misleading std of 0.
        MLP varies with the seed and is accumulated normally.
        """
        sc_key = self._find_scaffold_key(result)
        if sc_key is None:
            return

        sc = result[sc_key]
        for model in ("KNN", "MLP"):
            if model not in sc:
                continue
            # KNN is deterministic — only keep first seed's results.
            if model == "KNN" and model in accumulator:
                continue
            if model not in accumulator:
                accumulator[model] = {}
            for metric in METRICS_ORDER:
                val = sc[model].get(metric)
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    accumulator[model].setdefault(metric, []).append(float(val))

    @staticmethod
    def _aggregate(
        seed_results_per_model: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Dict]:
        """Compute mean ± std across seeds for each model."""
        aggregated: Dict[str, Dict] = {}
        for model, metrics_dict in seed_results_per_model.items():
            agg: Dict[str, object] = {}
            for metric, values in metrics_dict.items():
                arr = np.array(values)
                agg[metric] = float(np.mean(arr))
                agg[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            agg["n_seeds"] = len(next(iter(metrics_dict.values())))
            aggregated[model] = agg
        return aggregated

    def _save_aggregated(self, level_dir: str, aggregated: Dict) -> None:
        """Persist aggregated results to JSON."""
        os.makedirs(level_dir, exist_ok=True)
        agg_path = os.path.join(level_dir, "split_comparison_results.json")
        with open(agg_path, "w") as fh:
            json.dump(
                {
                    "dataset": self.dataset,
                    "level": self.level_tag,
                    "embedding_name": self.embedding_name,
                    "seeds": self.seeds,
                    "results": {"Split by Scaffold": aggregated},
                },
                fh,
                indent=2,
            )
        tqdm.write(f"  Aggregated results saved: {agg_path}")

    @staticmethod
    def _load_cached_results(directory: str) -> Optional[Dict]:
        """Load cached per-seed results JSON if it exists.

        Each level saves its seed results as ``levelN_knn_mlp_results.json``.
        This fallback searches for any matching file in the seed directory.
        """
        import glob as _glob

        candidates = _glob.glob(os.path.join(directory, "*_knn_mlp_results.json"))
        if not candidates:
            return None
        try:
            with open(candidates[0]) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, KeyError):
            return None
