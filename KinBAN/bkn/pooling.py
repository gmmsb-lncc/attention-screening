"""CLS-guided attention pooling strategy."""
from __future__ import annotations

import torch


def cls_guided_attention_pool(
    tokens: torch.Tensor,
    cls: torch.Tensor,
    dim: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CLS-guided attention pooling.

    Uses the CLS token representation as a query that attends over all token
    representations (keys/values), producing a single context-aware vector.
    This captures global context (CLS) conditioned on local information.

    Args:
        tokens: [L, D] — all token representations.
        cls:    [D]    — CLS token representation used as the query.
        dim:    D      — embedding dimension (scaling factor).
        mask:   [L]    — int/bool mask (1=valid, 0=padding). None = all valid.

    Returns:
        [D] — pooled representation.
    """
    scores = torch.matmul(tokens, cls) / (dim ** 0.5)  # [L]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = torch.softmax(scores, dim=0)              # [L]
    return (weights.unsqueeze(1) * tokens).sum(0)       # [D]
