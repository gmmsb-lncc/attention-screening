"""Seed parallelism and level execution planning helpers.

Provides automatic configuration for ``BENCHMARK_SEED_WORKERS`` and
simple level ordering policies to avoid delaying GPU-heavy levels.
"""

from __future__ import annotations

import os


def auto_configure_seed_workers(n_seeds: int) -> int:
    """Auto-resolve and export ``BENCHMARK_SEED_WORKERS``.

    Resolution policy:
      1) Respect existing ``BENCHMARK_SEED_WORKERS`` when valid.
      2) Else choose a conservative CPU-based value and export it.

    The heuristic uses half of available CPU cores (at least 1), bounded
    by the number of seeds to avoid creating idle workers.
    """
    env = os.getenv("BENCHMARK_SEED_WORKERS")
    if env is not None:
        try:
            val = max(1, int(env))
            os.environ["BENCHMARK_SEED_WORKERS"] = str(val)
            return min(val, max(1, n_seeds))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    recommended = max(1, cpu_count // 2)
    recommended = min(recommended, max(1, n_seeds))
    os.environ["BENCHMARK_SEED_WORKERS"] = str(recommended)
    return recommended


def prioritize_gpu_level_three(levels: list[str]) -> list[str]:
    """Return levels reordered so GPU Level 3 runs before CPU-only levels.

    Keeps relative order for all non-Level-3 entries.
    """
    if "3" not in levels:
        return list(levels)
    rest = [lv for lv in levels if lv != "3"]
    return ["3", *rest]


def is_cpu_gpu_parallel_enabled(levels: list[str]) -> bool:
    """Return whether CPU-levels and GPU Level 3 should run concurrently.

    Enable with:
      BENCHMARK_PARALLEL_CPU_GPU=1
    """
    raw = os.getenv("BENCHMARK_PARALLEL_CPU_GPU", "0").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return False

    has_gpu_level = "3" in levels
    has_cpu_levels = any(lv in levels for lv in ("1a", "1b", "1c"))
    return has_gpu_level and has_cpu_levels