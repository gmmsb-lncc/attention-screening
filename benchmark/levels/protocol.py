"""Shared protocol helpers for KNN/MLP fit/eval workflow.

Centralizes mode-based split selection and feature sanitization so that
all active levels (1a, 1b, 1c, 2, 3) follow the same execution pattern.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


class FitEvalSelection(Generic[T]):
    """Container for mode-based fit/eval selection."""

    def __init__(
        self,
        fit: T,
        eval_: T,
        fit_name: str,
        eval_name: str,
    ) -> None:
        self.fit = fit
        self.eval = eval_
        self.fit_name = fit_name
        self.eval_name = eval_name


def select_fit_eval(
    mode: str,
    train_obj: T,
    val_obj: T,
    test_obj: T,
) -> FitEvalSelection[T]:
    """Select fit/eval objects according to benchmark mode.

    Train mode:
      fit=train, eval=val
    Test mode:
      fit=val, eval=test
    """
    if mode == "train":
        return FitEvalSelection(
            fit=train_obj,
            eval_=val_obj,
            fit_name="train",
            eval_name="val",
        )

    return FitEvalSelection(
        fit=val_obj,
        eval_=test_obj,
        fit_name="val",
        eval_name="test",
    )


def sanitize_features(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN/Inf with zeros and return the number of replaced values."""
    bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
    if bad == 0:
        return arr, 0
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), bad
