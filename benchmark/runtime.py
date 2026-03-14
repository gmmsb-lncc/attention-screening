"""Runtime helpers for hardware-aware execution.

Centralizes device selection and DataLoader worker sizing so benchmark
modules share the same execution policy.
"""

from __future__ import annotations

import os

import torch


def get_torch_device() -> torch.device:
    """Return the best available torch device in priority order.

    Priority:
      1) CUDA
      2) MPS (Apple Silicon)
      3) CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_runtime_device_name() -> str:
    """Return device name string expected by finetuner APIs."""
    return str(get_torch_device())


def get_dataloader_workers(default_cap: int = 4) -> int:
    """Resolve DataLoader worker count from env or CPU topology.

    Environment override:
      BENCHMARK_NUM_WORKERS=<int>
    """
    env = os.getenv("BENCHMARK_NUM_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    # Leave one core for the main process/UI responsiveness.
    return max(0, min(default_cap, cpu_count - 1))


def get_seed_workers(n_seeds: int) -> int:
    """Resolve how many seeds can run in parallel.

    Environment override:
      BENCHMARK_SEED_WORKERS=<int>

    Defaults to sequential execution (1) when no auto-configuration has
    set ``BENCHMARK_SEED_WORKERS``.
    """
    env = os.getenv("BENCHMARK_SEED_WORKERS")
    if env is not None:
        try:
            return max(1, min(int(env), max(1, n_seeds)))
        except ValueError:
            pass
    return 1


def resolve_effective_batch_size(requested_batch_size: int) -> int:
    """Resolve effective batch size for matrix-based levels.

    Priority:
      1) BENCHMARK_BATCH_SIZE (explicit override)
      2) requested batch size (default)
      3) optional CUDA auto-scaling when BENCHMARK_AUTO_BATCH_SIZE=1
    """
    # Explicit override always wins.
    override = os.getenv("BENCHMARK_BATCH_SIZE")
    if override is not None:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    base = max(1, int(requested_batch_size))
    auto = os.getenv("BENCHMARK_AUTO_BATCH_SIZE", "0").strip().lower()
    if auto not in {"1", "true", "yes", "on"}:
        return base

    if not torch.cuda.is_available():
        return base

    # Heuristic tuned for common GPUs; keeps risk bounded.
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / (1024 ** 3)
    if total_gb >= 22:
        multiplier = 4
    elif total_gb >= 14:
        multiplier = 3
    elif total_gb >= 10:
        multiplier = 2
    else:
        multiplier = 1

    cap_env = os.getenv("BENCHMARK_BATCH_SIZE_CAP")
    cap = None
    if cap_env is not None:
        try:
            cap = max(1, int(cap_env))
        except ValueError:
            cap = None

    scaled = base * multiplier
    if cap is not None:
        scaled = min(scaled, cap)
    return max(1, scaled)


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True when an exception corresponds to CUDA OOM."""
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def batch_size_fallbacks(start_batch_size: int) -> list[int]:
    """Return descending batch-size candidates via halving."""
    start = max(1, int(start_batch_size))
    candidates: list[int] = []
    cur = start
    while cur >= 1:
        candidates.append(cur)
        if cur == 1:
            break
        cur = max(1, cur // 2)
    return candidates