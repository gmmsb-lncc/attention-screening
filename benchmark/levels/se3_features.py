"""Utilities for optional ligand structural features from SE(3)-Transformer.

This module loads pre-computed per-ligand structural vectors by ``chembl_id``
and provides batch assembly with zero fallback for missing IDs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


class SE3FeatureLoader:
    """Lazy loader for per-ligand structural vectors.

    Expected files in ``feature_dir``:
    - ``{chembl_id}_se3.npy``
    - ``{chembl_id}.npy``

    Vectors are flattened to 1-D and cast to float32.
    """

    def __init__(self, feature_dirs: list[str]) -> None:
        self._dirs = [Path(d) for d in feature_dirs]
        self._cache: dict[str, np.ndarray] = {}
        self._dim = self._infer_dim()

    @property
    def dim(self) -> int:
        """Feature dimensionality inferred from the directory."""
        return self._dim

    def get_batch(self, chembl_ids: Iterable[str]) -> np.ndarray:
        """Return stacked SE3 vectors for a batch of chembl IDs."""
        vectors: list[np.ndarray] = []
        for chembl_id in chembl_ids:
            vec = self._get_one(str(chembl_id))
            vectors.append(vec)

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

    def _resolve_path(self, chembl_id: str) -> Path | None:
        for d in self._dirs:
            candidates = [
                d / f"{chembl_id}_se3.npy",
                d / f"{chembl_id}.npy",
            ]
            for c in candidates:
                if c.exists():
                    return c
        return None

    def _get_one(self, chembl_id: str) -> np.ndarray:
        if chembl_id in self._cache:
            return self._cache[chembl_id]

        if self._dim == 0:
            out = np.zeros((0,), dtype=np.float32)
            self._cache[chembl_id] = out
            return out

        path = self._resolve_path(chembl_id)
        if path is None:
            out = np.zeros((self._dim,), dtype=np.float32)
            self._cache[chembl_id] = out
            return out

        try:
            arr = np.load(path).astype(np.float32).reshape(-1)
            if arr.size != self._dim:
                out = np.zeros((self._dim,), dtype=np.float32)
            else:
                out = arr
        except Exception:
            out = np.zeros((self._dim,), dtype=np.float32)

        self._cache[chembl_id] = out
        return out


def build_se3_loader(
    feature_dir: str | None,
    use_se3_ligand: bool,
    default_feature_dirs: list[str] | None = None,
) -> SE3FeatureLoader | None:
    """Build optional SE3 loader from runtime configuration."""
    if not use_se3_ligand:
        return None

    dirs = [feature_dir] if feature_dir else list(default_feature_dirs or [])
    if not dirs:
        return None

    # Keep standardized layout behavior: make default dirs proactively.
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass

    if not any(os.path.isdir(d) for d in dirs):
        return None

    loader = SE3FeatureLoader(dirs)
    if loader.dim <= 0:
        return None
    return loader
