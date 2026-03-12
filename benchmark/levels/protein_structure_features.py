"""Utilities for optional protein structural features (e.g., ESMFold vectors).

Loads pre-computed per-protein vectors by ``seq_id`` and provides
batch assembly with zero fallback for missing IDs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


class ProteinStructureFeatureLoader:
    """Lazy loader for per-protein structural vectors.

    Expected files in feature directories:
    - ``{seq_id}_esmfold.npy``
    - ``{seq_id}_structure.npy``
    - ``{seq_id}.npy``

    Vectors are flattened to 1-D and cast to float32.
    """

    def __init__(self, feature_dirs: list[str]) -> None:
        self._dirs = [Path(d) for d in feature_dirs]
        self._cache: dict[str, np.ndarray] = {}
        self._dim = self._infer_dim()

    @property
    def dim(self) -> int:
        """Feature dimensionality inferred from the directory contents."""
        return self._dim

    def get_batch(self, seq_ids: Iterable[str]) -> np.ndarray:
        """Return stacked structure vectors for a batch of protein IDs."""
        vectors: list[np.ndarray] = []
        for seq_id in seq_ids:
            vectors.append(self._get_one(str(seq_id)))

        if not vectors:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack(vectors)

    def _infer_dim(self) -> int:
        for d in self._dirs:
            if not d.exists():
                continue
            for path in d.glob("*.npy"):
                try:
                    arr = np.load(path).astype(np.float32).reshape(-1)
                except Exception:
                    continue
                if arr.size > 0:
                    return int(arr.size)
        return 0

    def _resolve_path(self, seq_id: str) -> Path | None:
        for d in self._dirs:
            candidates = [
                d / f"{seq_id}_esmfold.npy",
                d / f"{seq_id}_structure.npy",
                d / f"{seq_id}.npy",
            ]
            for c in candidates:
                if c.exists():
                    return c
        return None

    def _get_one(self, seq_id: str) -> np.ndarray:
        if seq_id in self._cache:
            return self._cache[seq_id]

        if self._dim == 0:
            out = np.zeros((0,), dtype=np.float32)
            self._cache[seq_id] = out
            return out

        path = self._resolve_path(seq_id)
        if path is None:
            out = np.zeros((self._dim,), dtype=np.float32)
            self._cache[seq_id] = out
            return out

        try:
            arr = np.load(path).astype(np.float32).reshape(-1)
            if arr.size != self._dim:
                out = np.zeros((self._dim,), dtype=np.float32)
            else:
                out = arr
        except Exception:
            out = np.zeros((self._dim,), dtype=np.float32)

        self._cache[seq_id] = out
        return out


def build_protein_structure_loader(
    feature_dir: str | None,
    use_esmfold_protein: bool,
    default_feature_dirs: list[str] | None = None,
) -> ProteinStructureFeatureLoader | None:
    """Build optional protein structural feature loader from runtime config."""
    if not use_esmfold_protein:
        return None

    dirs = [feature_dir] if feature_dir else list(default_feature_dirs or [])
    if not dirs:
        return None

    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass

    if not any(os.path.isdir(d) for d in dirs):
        return None

    loader = ProteinStructureFeatureLoader(dirs)
    if loader.dim <= 0:
        return None
    return loader
