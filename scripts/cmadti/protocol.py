"""Pure evaluation helpers shared by CMA-DTI training and tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def best_mcc_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    order = np.argsort(-probability, kind="mergesort")
    p = probability[order]
    labels = y[order]
    last = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.cumsum(labels)[last]
    fp = last + 1 - tp
    positives = int(labels.sum())
    fn = positives - tp
    negatives = len(labels) - positives
    tn = negatives - fp
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                       where=denominator != 0)
    best = scores.max()
    tied = np.flatnonzero(np.isclose(scores, best, rtol=1e-9, atol=1e-12))
    thresholds = p[last]
    return float(thresholds[tied[np.argmin(np.abs(thresholds[tied] - 0.5))]])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    pred = probability >= threshold
    return {
        "n": int(len(y)), "positive": int(y.sum()), "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }
