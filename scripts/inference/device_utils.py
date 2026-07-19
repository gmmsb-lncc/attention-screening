"""Device selection helper for inference: cuda → mps → cpu.

Mac M1/M2 has no CUDA but exposes Metal Performance Shaders (MPS) via
``torch.backends.mps``. Score scripts use this helper so the same code
runs on Linux GPU, Mac M1, and CPU-only fallbacks without branching.
"""
from __future__ import annotations

import os

import torch


def pick_device() -> torch.device:
    # Explicit override. On Apple Silicon the MPS backend lacks float64 and a
    # few linalg ops (e.g. linalg_qr used by MoLFormer's linear attention), so
    # forcing INFER_DEVICE=cpu is the reliable path for committee inference on
    # mac. Accepts cuda | mps | cpu.
    forced = os.environ.get("INFER_DEVICE", "").strip().lower()
    if forced in {"cuda", "mps", "cpu"}:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
