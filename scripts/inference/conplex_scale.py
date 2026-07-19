"""Scale contract for ConPLex outputs used by committee inference."""
from __future__ import annotations

import numpy as np


def canonical_similarity(values, *, atol: float = 1e-6) -> np.ndarray:
    """Return native non-negative cosine similarities on the canonical scale.

    ``SimpleCoembedding`` uses ReLU projections before cosine similarity, so
    the expected range is [0, 1]. Values just outside the interval from
    floating-point roundoff are clipped; larger excursions signal an
    architecture or scale mismatch and fail explicitly.
    """
    scores = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("ConPLex similarities must be finite")
    if np.any(scores < -atol) or np.any(scores > 1.0 + atol):
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        raise ValueError(
            f"ConPLex similarities outside canonical [0, 1] range: [{lo}, {hi}]"
        )
    return np.clip(scores, 0.0, 1.0)
