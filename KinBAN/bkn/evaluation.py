"""Fair evaluation protocol: prediction collection, threshold sweep, and metrics."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import DRUG_EMB_DIM, ESM2_EMB_DIM
from .loader import _ensure_src_on_path

# Ensure src/ is on sys.path so 'from models import ...' resolves at call time.
_ensure_src_on_path()


def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect (y_true, y_prob) arrays from a data loader using the given model.

    Uses DRUG_EMB_DIM=768 (MoLFormer) and ESM2_EMB_DIM=1280 for tensor reshaping.
    Requires GRAPHBAN_INDUCTIVE to be in sys.path; call setup_bkn_imports() first.
    """
    from models import binary_cross_entropy, cross_entropy_logits  # GraphBAN

    y_true: list = []
    y_prob: list = []

    with torch.no_grad():
        model.eval()
        for batch in data_loader:
            v_d, sm, v_p, esm_feat, labels = batch
            sm = torch.tensor(sm, dtype=torch.float32)
            sm = torch.reshape(sm, (sm.shape[0], 1, DRUG_EMB_DIM))
            esm_feat = torch.tensor(esm_feat, dtype=torch.float32)
            esm_feat = torch.reshape(esm_feat, (sm.shape[0], 1, ESM2_EMB_DIM))
            v_d = v_d.to(device)
            sm = sm.to(device)
            v_p = v_p.to(device)
            esm_feat = esm_feat.to(device)
            labels = labels.float().to(device)
            _, _, _, score = model(v_d, sm, v_p, esm_feat, device)

            if n_class == 1:
                n, _ = binary_cross_entropy(score, labels)
            else:
                n, _ = cross_entropy_logits(score, labels)

            y_prob.extend(n.to("cpu").tolist())
            y_true.extend(labels.to("cpu").tolist())

    return np.array(y_true), np.array(y_prob)


def optimize_threshold_on_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "mcc",
) -> tuple[float, float]:
    """Find the optimal decision threshold on validation predictions.

    Vectorised sweep over all unique thresholds — no test data is touched.
    Ties are broken by choosing the threshold closest to 0.5.

    Args:
        y_true:  Ground-truth binary labels.
        y_prob:  Predicted probabilities.
        metric:  ``"mcc"`` (default) or ``"f1"``.

    Returns:
        ``(best_threshold, best_score)`` tuple.
    """
    if len(y_true) == 0:
        return 0.5, 0.0

    order = np.argsort(y_prob, kind="mergesort")[::-1]
    probs_sorted = y_prob[order]
    labels_sorted = y_true[order]

    total_pos = float((labels_sorted == 1).sum())
    total_neg = float((labels_sorted == 0).sum())

    tp_cum = np.cumsum(labels_sorted == 1, dtype=np.float64)
    fp_cum = np.cumsum(labels_sorted == 0, dtype=np.float64)

    last_indices = np.r_[
        np.where(np.diff(probs_sorted) != 0)[0], len(probs_sorted) - 1
    ]
    tp = tp_cum[last_indices]
    fp = fp_cum[last_indices]
    fn = total_pos - tp
    tn = total_neg - fp
    thresholds = probs_sorted[last_indices]

    sentinel = np.nextafter(float(np.max(probs_sorted)), np.inf)
    tp = np.concatenate(([0.0], tp))
    fp = np.concatenate(([0.0], fp))
    fn = np.concatenate(([total_pos], fn))
    tn = np.concatenate(([total_neg], tn))
    thresholds = np.concatenate(([sentinel], thresholds))

    if metric == "mcc":
        denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        valid = denom_sq > 0
        scores = np.zeros_like(denom_sq)
        scores[valid] = (
            (tp[valid] * tn[valid] - fp[valid] * fn[valid])
            / np.sqrt(denom_sq[valid])
        )
    elif metric == "f1":
        denom = (2 * tp) + fp + fn
        scores = np.where(denom > 0, (2 * tp) / denom, 0.0)
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    best_score = float(np.nanmax(scores))
    tie_idx = np.where(np.isclose(scores, best_score, rtol=1e-9, atol=1e-12))[0]
    best_idx = int(tie_idx[np.argmin(np.abs(thresholds[tie_idx] - 0.5))])

    return float(thresholds[best_idx]), best_score


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute accuracy, F1, precision, recall, MCC, and AUROC at a threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }
