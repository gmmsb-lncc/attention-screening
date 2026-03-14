"""Abstract base class and shared logic for all level runners.

Provides the ``BaseLevelRunner`` interface (Open/Closed principle) and
a reusable multi-seed aggregation helper.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from benchmark.config import METRICS_ORDER, BenchmarkConfig
from benchmark.runtime import get_seed_workers


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
        processed_seeds: List[int] = []
        mlp_train_mcc_history: List[float] = []
        mlp_eval_mcc_history: List[float] = []

        seed_worker_count = get_seed_workers(len(self.seeds))
        if seed_worker_count <= 1:
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
                processed_seeds.append(seed)

                train_mcc, eval_mcc = self._extract_mlp_train_eval_mcc(result)
                if train_mcc is not None:
                    mlp_train_mcc_history.append(train_mcc)
                if eval_mcc is not None:
                    mlp_eval_mcc_history.append(eval_mcc)

                partial = self._aggregate(seed_results_per_model)
                self._save_intermediate(
                    level_dir,
                    partial,
                    processed_seeds,
                    mlp_train_mcc_history,
                    mlp_eval_mcc_history,
                )
        else:
            tqdm.write(
                f"  Parallel seed execution enabled: {seed_worker_count} workers "
                f"for {len(self.seeds)} seeds"
            )
            futures = {}
            seed_start_times: Dict[int, float] = {}
            level_start = time.time()
            heartbeat_seconds = 60.0
            completed_count = 0
            with ThreadPoolExecutor(max_workers=seed_worker_count) as executor:
                for seed in self.seeds:
                    seed_dir = os.path.join(level_dir, f"seed_{seed}")
                    seed_start_times[seed] = time.time()
                    tqdm.write(f"  Dispatch seed {seed} -> {seed_dir}")
                    futures[
                        executor.submit(
                            self.run_single_seed,
                            seed=seed,
                            output_dir=seed_dir,
                            **kwargs,
                        )
                    ] = (seed, seed_dir)

                pending = set(futures.keys())
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=heartbeat_seconds,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done:
                        elapsed_level = (time.time() - level_start) / 60.0
                        running = len(pending)
                        tqdm.write(
                            f"  [Heartbeat] elapsed={elapsed_level:.1f} min, "
                            f"completed={completed_count}/{len(self.seeds)}, "
                            f"running={running}"
                        )
                        continue

                    for future in done:
                        completed_count += 1
                    seed, seed_dir = futures[future]
                    elapsed_seed = time.time() - seed_start_times.get(seed, time.time())
                    tqdm.write(
                        f"  Seed done {completed_count}/{len(self.seeds)}: {seed} "
                        f"({elapsed_seed/60:.1f} min)"
                    )
                    try:
                        result = future.result()
                    except Exception as exc:
                        tqdm.write(f"    ERROR: seed {seed} failed: {exc}")
                        continue

                    if result is None:
                        result = self._load_cached_results(seed_dir)

                    if result is None:
                        tqdm.write(f"    WARNING: seed {seed} returned no results.")
                        continue

                    self._accumulate_seed(result, seed_results_per_model)
                    processed_seeds.append(seed)

                    train_mcc, eval_mcc = self._extract_mlp_train_eval_mcc(result)
                    if train_mcc is not None:
                        mlp_train_mcc_history.append(train_mcc)
                    if eval_mcc is not None:
                        mlp_eval_mcc_history.append(eval_mcc)

                    partial = self._aggregate(seed_results_per_model)
                    self._save_intermediate(
                        level_dir,
                        partial,
                        processed_seeds,
                        mlp_train_mcc_history,
                        mlp_eval_mcc_history,
                    )

        if not seed_results_per_model:
            return None

        aggregated = self._aggregate(seed_results_per_model)
        self._save_aggregated(level_dir, aggregated)

        return {"Split by Scaffold": aggregated}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def knn_is_deterministic(self) -> bool:
        """Whether KNN results are identical across seeds for this level.

        True for levels without learned feature extractors (1a, 1b, 2):
        same input → same KNN output regardless of seed.

        False for levels with learned extractors (1c, 3, 4+): the
        upstream model is seed-dependent, so KNN input varies per seed.

        Subclasses with learned components should override to return False.
        """
        return True

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
            # KNN is deterministic for levels without learned feature
            # extractors.  Skip re-accumulation only when safe.
            if model == "KNN" and model in accumulator and self.knn_is_deterministic:
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

    def _save_intermediate(
        self,
        level_dir: str,
        aggregated_partial: Dict[str, Dict],
        processed_seeds: List[int],
        mlp_train_mcc_history: List[float],
        mlp_eval_mcc_history: List[float],
    ) -> None:
        """Persist partial aggregation after each completed seed.

        This snapshot is overwritten at every successful seed and enables
        real-time monitoring of trend evolution during long benchmark runs.
        """
        os.makedirs(level_dir, exist_ok=True)
        inter_path = os.path.join(level_dir, "split_comparison_results.intermediate.json")
        alerts = self._build_mlp_alerts(mlp_train_mcc_history, mlp_eval_mcc_history)
        with open(inter_path, "w") as fh:
            json.dump(
                {
                    "dataset": self.dataset,
                    "level": self.level_tag,
                    "embedding_name": self.embedding_name,
                    "seeds_configured": self.seeds,
                    "seeds_processed": processed_seeds,
                    "progress": {
                        "completed": len(processed_seeds),
                        "total": len(self.seeds),
                    },
                    "alerts": alerts,
                    "results": {"Split by Scaffold": aggregated_partial},
                },
                fh,
                indent=2,
            )
        tqdm.write(f"    Intermediate snapshot saved: {inter_path}")

    @staticmethod
    def _extract_mlp_train_eval_mcc(result: Dict) -> tuple[Optional[float], Optional[float]]:
        """Extract per-seed MLP train/eval MCC diagnostics from raw result dict."""
        if not isinstance(result, dict):
            return None, None

        sc_key = next((k for k in result if "scaffold" in str(k).lower()), None)
        if sc_key is None:
            sc_key = next(iter(result), None)
        if sc_key is None:
            return None, None

        sc = result.get(sc_key, {})
        mlp = sc.get("MLP", {}) if isinstance(sc, dict) else {}
        if not isinstance(mlp, dict):
            return None, None

        eval_mcc_raw = mlp.get("mcc")
        eval_mcc = float(eval_mcc_raw) if isinstance(eval_mcc_raw, (int, float)) else None

        details = mlp.get("details", {})
        if not isinstance(details, dict):
            return None, eval_mcc

        model_train = details.get("model_train", {})
        if not isinstance(model_train, dict):
            return None, eval_mcc

        train_mcc_raw = model_train.get("mcc_at_default_threshold")
        train_mcc = float(train_mcc_raw) if isinstance(train_mcc_raw, (int, float)) else None
        return train_mcc, eval_mcc

    @staticmethod
    def _build_mlp_alerts(
        mlp_train_mcc_history: List[float],
        mlp_eval_mcc_history: List[float],
    ) -> Dict[str, object]:
        """Build lightweight automated alert flags for intermediate monitoring."""
        if not mlp_train_mcc_history:
            return {
                "mlp_underfitting_suspected": False,
                "reason": "insufficient_train_diagnostics",
            }

        mean_train_mcc = float(np.mean(np.array(mlp_train_mcc_history, dtype=np.float64)))
        mean_eval_mcc = (
            float(np.mean(np.array(mlp_eval_mcc_history, dtype=np.float64)))
            if mlp_eval_mcc_history
            else None
        )

        # Underfitting heuristic: persistently weak training fit.
        underfitting = mean_train_mcc < 0.90

        return {
            "mlp_underfitting_suspected": underfitting,
            "mlp_train_mcc_mean": mean_train_mcc,
            "mlp_eval_mcc_mean": mean_eval_mcc,
            "processed_points": len(mlp_train_mcc_history),
            "rule": "underfitting_if_mean_train_mcc_below_0.90",
        }

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
