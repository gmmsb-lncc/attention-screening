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