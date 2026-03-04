"""Model evaluation utilities with explicit failure handling."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, log_loss,
)


class EvaluationError(Exception):
    """Raised when model evaluation fails due to numerical issues."""


class EvaluationResult:
    """
    Structured evaluation result with explicit validity tracking.

    Attributes:
        metrics: Dictionary with computed metrics
        is_valid: Whether evaluation completed successfully
        failure_reason: Description of failure if is_valid=False
        nan_count: Number of NaN values detected
        inf_count: Number of Inf values detected
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        is_valid: bool = True,
        failure_reason: Optional[str] = None,
        nan_count: int = 0,
        inf_count: int = 0,
    ):
        self.metrics = metrics
        self.is_valid = is_valid
        self.failure_reason = failure_reason
        self.nan_count = nan_count
        self.inf_count = inf_count

    def __getitem__(self, key: str) -> float:
        """Allow dict-like access for backward compatibility."""
        return self.metrics[key]

    def get(self, key: str, default: float = None) -> float:
        return self.metrics.get(key, default)

    def to_dict(self) -> Dict:
        return {
            **self.metrics,
            "_is_valid": self.is_valid,
            "_failure_reason": self.failure_reason,
            "_nan_count": self.nan_count,
            "_inf_count": self.inf_count,
        }


def _collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Collect y_true and y_prob arrays from a dataloader."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            protein_matrix = batch["protein_matrix"].to(device)
            ligand_matrix = batch["ligand_matrix"].to(device)
            protein_mask = batch["protein_mask"].to(device)
            ligand_mask = batch["ligand_mask"].to(device)
            labels = batch["labels"]

            output = model(protein_matrix, ligand_matrix, protein_mask, ligand_mask)
            probs = torch.sigmoid(output["classification"]).cpu().numpy().reshape(-1)

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().reshape(-1).tolist())

    all_probs_arr = np.asarray(all_probs, dtype=np.float64)
    all_labels_arr = np.asarray(all_labels, dtype=np.int64)

    nan_count = int(np.isnan(all_probs_arr).sum())
    inf_count = int(np.isinf(all_probs_arr).sum())
    return all_labels_arr, all_probs_arr, nan_count, inf_count


def _invalid_result(
    failure_msg: str,
    nan_count: int,
    inf_count: int,
) -> EvaluationResult:
    return EvaluationResult(
        metrics={
            "accuracy": np.nan,
            "f1": np.nan,
            "mcc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "auc": np.nan,
            "loss": np.nan,
            "decision_threshold": np.nan,
        },
        is_valid=False,
        failure_reason=failure_msg,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def _compute_metrics_from_probs(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    decision_threshold: float,
) -> Dict[str, float]:
    """Compute binary metrics from probabilities and a decision threshold."""
    preds = (all_probs >= decision_threshold).astype(np.int64)
    probs_clipped = np.clip(all_probs, 1e-7, 1.0 - 1e-7)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, preds)),
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(all_labels, preds)),
        "precision": float(precision_score(all_labels, preds, zero_division=0)),
        "recall": float(recall_score(all_labels, preds, zero_division=0)),
        "auc": float(
            roc_auc_score(all_labels, all_probs)
            if len(np.unique(all_labels)) > 1
            else np.nan
        ),
        "loss": float(log_loss(all_labels, probs_clipped, labels=[0, 1])),
        "decision_threshold": float(decision_threshold),
    }
    return metrics


def _score_thresholds(
    tp: np.ndarray,
    fp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Vectorized threshold scoring from confusion-matrix counts."""
    if metric == "mcc":
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)

    if metric == "f1":
        denom = (2 * tp) + fp + fn
        return np.where(denom > 0, (2 * tp) / denom, 0.0)

    if metric == "balanced_accuracy":
        tpr = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        tnr = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
        return 0.5 * (tpr + tnr)

    raise ValueError(f"Unsupported threshold metric: {metric!r}")


def optimize_threshold_from_predictions(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    metric: str = "mcc",
    strategy: str = "validation",
) -> Dict[str, float]:
    """
    Optimize decision threshold on provided predictions.

    Uses all unique prediction values as candidate thresholds plus one sentinel
    threshold above max(prob), and selects the threshold that maximizes the
    chosen metric. Ties are broken by choosing the threshold closest to 0.5.
    
    Args:
        strategy: "validation" (default) or "combined" (train+val average)
    """
    if len(all_labels) == 0:
        raise EvaluationError("Cannot optimize threshold on empty predictions.")

    if len(all_labels) != len(all_probs):
        raise EvaluationError("Length mismatch between labels and probabilities.")

    supported_metrics = {"mcc", "f1", "balanced_accuracy"}
    if metric not in supported_metrics:
        raise ValueError(f"metric must be one of {sorted(supported_metrics)}, got {metric!r}")

    order = np.argsort(all_probs, kind="mergesort")[::-1]
    probs_sorted = all_probs[order]
    labels_sorted = all_labels[order]

    total_pos = float((labels_sorted == 1).sum())
    total_neg = float((labels_sorted == 0).sum())

    tp_cum = np.cumsum(labels_sorted == 1, dtype=np.float64)
    fp_cum = np.cumsum(labels_sorted == 0, dtype=np.float64)

    last_indices = np.r_[np.where(np.diff(probs_sorted) != 0)[0], len(probs_sorted) - 1]
    tp = tp_cum[last_indices]
    fp = fp_cum[last_indices]
    fn = total_pos - tp
    tn = total_neg - fp
    thresholds = probs_sorted[last_indices]

    # Candidate for predicting all negatives (threshold above max probability).
    all_neg_threshold = np.nextafter(float(np.max(probs_sorted)), np.inf)
    tp = np.concatenate(([0.0], tp))
    fp = np.concatenate(([0.0], fp))
    fn = np.concatenate(([total_pos], fn))
    tn = np.concatenate(([total_neg], tn))
    thresholds = np.concatenate(([all_neg_threshold], thresholds))

    scores = _score_thresholds(tp, fp, tn, fn, metric)
    best_score = float(np.nanmax(scores))

    tie_idx = np.where(np.isclose(scores, best_score, rtol=1e-9, atol=1e-12))[0]
    best_idx = int(tie_idx[np.argmin(np.abs(thresholds[tie_idx] - 0.5))])
    best_threshold = float(thresholds[best_idx])

    best_metrics = _compute_metrics_from_probs(all_labels, all_probs, best_threshold)
    best_metrics.update(
        {
            "threshold_optimized_metric": metric,
            "threshold_optimized_score": best_score,
            "threshold_candidates": int(len(thresholds)),
        }
    )
    return best_metrics


def optimize_decision_threshold(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    metric: str = "mcc",
    raise_on_invalid: bool = True,
) -> EvaluationResult:
    """
    Optimize decision threshold using model probabilities on a validation split.

    Args:
        model: Neural network model
        data_loader: Validation loader used for threshold optimization
        device: Device for inference
        metric: Objective metric for threshold search
        raise_on_invalid: If True, raise EvaluationError on NaN/Inf probabilities
    """
    all_labels, all_probs, nan_count, inf_count = _collect_predictions(model, data_loader, device)

    if len(all_labels) == 0:
        failure_msg = "No samples available for threshold optimization."
        if raise_on_invalid:
            raise EvaluationError(failure_msg)
        return _invalid_result(failure_msg, nan_count=0, inf_count=0)

    if nan_count > 0 or inf_count > 0:
        failure_msg = (
            f"Model produced invalid values: {nan_count} NaN, {inf_count} Inf "
            f"out of {len(all_probs)} predictions."
        )
        if raise_on_invalid:
            raise EvaluationError(failure_msg)
        return _invalid_result(failure_msg, nan_count, inf_count)

    metrics = optimize_threshold_from_predictions(all_labels, all_probs, metric=metric)
    return EvaluationResult(metrics=metrics, is_valid=True)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    raise_on_invalid: bool = True,
    decision_threshold: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate model on a dataset with explicit failure handling.

    Args:
        model: Neural network model (CrossAttentionAffinityModel)
        data_loader: Data loader
        device: Device to evaluate on
        raise_on_invalid: If True, raise EvaluationError on NaN/Inf
        decision_threshold: Classification threshold applied to probabilities

    Returns:
        EvaluationResult with metrics and validity status

    Raises:
        EvaluationError: If model produces NaN/Inf and raise_on_invalid=True
    """
    all_labels, all_probs, nan_count, inf_count = _collect_predictions(model, data_loader, device)

    if len(all_labels) == 0:
        failure_msg = "No samples available for evaluation."
        if raise_on_invalid:
            raise EvaluationError(failure_msg)
        return _invalid_result(failure_msg, nan_count=0, inf_count=0)

    if nan_count > 0 or inf_count > 0:
        failure_msg = (
            f"Model produced invalid values: {nan_count} NaN, {inf_count} Inf "
            f"out of {len(all_probs)} predictions."
        )

        if raise_on_invalid:
            raise EvaluationError(failure_msg)

        return _invalid_result(failure_msg, nan_count, inf_count)

    metrics = _compute_metrics_from_probs(
        all_labels=all_labels,
        all_probs=all_probs,
        decision_threshold=decision_threshold,
    )
    return EvaluationResult(metrics=metrics, is_valid=True)
