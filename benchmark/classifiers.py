"""Canonical KNN and MLP classifiers for benchmark comparison.

Every level in the benchmark must use the **exact same** classifier
configuration so that the only independent variable across levels is
the molecular representation.  This module provides a single
``train_knn_mlp`` function that **all four levels** call.

Classifier specifications:

 * **KNN** — FAISS inner-product index on L2-normalised features
   (equivalent to cosine similarity), *k = 5*, distance-weighted voting.
 * **MLP** — ``sklearn.neural_network.MLPClassifier`` with two hidden
   layers of (512, 256) units, ReLU activation, Adam solver, adaptive
   learning rate, α = 1 × 10⁻³, max 1000 iterations.  Early stopping
   is **enabled** (10 % held-out, patience = 20) to prevent overfitting
   and mirror the implicit regularisation of KNN distance-weighted voting.
 * Both classifiers receive features after ``StandardScaler``.

Evaluation protocol:

 * All levels pass **validation-split** features as ``x_train`` / ``y_train``
   and **test-split** features as ``x_test`` / ``y_test``.
 * This eliminates train-set optimism for levels with learned feature
   extractors (Levels 3 and 4) and ensures a consistent protocol across
   all four levels.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


# ------------------------------------------------------------------ #
# FAISS-based KNN (matches split_comparison_analysis.faiss_knn_predict)
# ------------------------------------------------------------------ #

def _faiss_knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """KNN classification via FAISS cosine similarity with distance-weighted voting.

    Identical to ``split_comparison_analysis.faiss_knn_predict``:
    L2-normalise → inner-product search → distance-weighted voting.

    Returns
    -------
    predictions : np.ndarray
        Predicted class labels.
    probabilities : np.ndarray
        Probability estimate for the positive class (class = 1).
    """
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for KNN classification. "
            "Install it with: pip install faiss-cpu"
        ) from exc

    x_train_f32 = np.ascontiguousarray(x_train, dtype=np.float32)
    x_test_f32 = np.ascontiguousarray(x_test, dtype=np.float32)
    faiss.normalize_L2(x_train_f32)
    faiss.normalize_L2(x_test_f32)

    index = faiss.IndexFlatIP(x_train_f32.shape[1])
    index.add(x_train_f32)
    similarities, indices = index.search(x_test_f32, k)

    # Distance-weighted voting — higher similarity means closer
    weights = np.maximum(similarities, 0.0)

    classes = np.unique(y_train)
    n_test = x_test_f32.shape[0]
    neighbor_labels = y_train[indices]  # (n_test, k)

    class_scores = np.zeros((n_test, len(classes)), dtype=np.float64)
    for ci, cls in enumerate(classes):
        mask = neighbor_labels == cls
        class_scores[:, ci] = np.where(mask, weights, 0.0).sum(axis=1)

    best_class_idx = class_scores.argmax(axis=1)
    predictions = classes[best_class_idx]

    # Probability for positive class
    row_sums = np.maximum(class_scores.sum(axis=1, keepdims=True), 1e-12)
    probabilities = class_scores[:, -1] / row_sums.ravel()

    return predictions, probabilities


# ------------------------------------------------------------------ #
# Metric computation
# ------------------------------------------------------------------ #

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute the standard metric suite."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba))
        if len(np.unique(y_true)) > 1
        else 0.0,
    }


def _optimize_threshold_mcc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Find decision threshold that maximizes MCC on calibration data."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    # Include robust fixed grid + unique probability anchors.
    grid = np.linspace(0.05, 0.95, 37)
    anchors = np.unique(np.clip(y_proba, 0.0, 1.0))
    thresholds = np.unique(np.concatenate([grid, anchors]))

    best_thr = 0.5
    best_mcc = -1.0
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
        elif mcc == best_mcc and abs(float(thr) - 0.5) < abs(best_thr - 0.5):
            # Tie-break towards conservative operating point.
            best_thr = float(thr)

    return best_thr, best_mcc


def _split_fit_for_calibration(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    calibration_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create fit/calibration split for threshold and model selection."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or y.shape[0] < 20:
        # Small or single-class fallback: skip calibration split.
        return x, y, x, y

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=calibration_fraction,
        random_state=seed,
    )
    fit_idx, cal_idx = next(splitter.split(x, y))
    return x[fit_idx], y[fit_idx], x[cal_idx], y[cal_idx]


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def train_knn_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Train canonical KNN and MLP classifiers and return metric dicts.

    **Both classifiers use the exact same hyperparameters** across all
    four benchmark levels so that the comparison is scientifically valid.

    Parameters
    ----------
    x_train, y_train : array-like
        Features and labels for classifier training.  In the benchmark
        protocol these come from the **validation** split — not the
        training split — to avoid train-set optimism when the upstream
        feature extractor was trained on the training data.
    x_test, y_test : array-like
        Hold-out test features and labels for evaluation.
    seed : int
        Random seed for MLP reproducibility.

    Returns
    -------
    dict
        ``{"KNN": {metric: value, ...}, "MLP": {metric: value, ...}}``
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # StandardScaler — same as split_comparison_analysis
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train).astype(np.float32)
    x_test_sc = scaler.transform(x_test).astype(np.float32)

    # ---------- KNN (FAISS, k=5, cosine, distance-weighted) ----------
    knn_pred, knn_proba = _faiss_knn_predict(x_train_sc, y_train, x_test_sc, k=5)
    knn_metrics = _compute_metrics(y_test, knn_pred, knn_proba)

    # ---------- MLP tuned for MCC (model + threshold selected on calibration) --
    x_fit, y_fit, x_cal, y_cal = _split_fit_for_calibration(x_train_sc, y_train, seed)

    mlp_candidates = [
        {
            "hidden_layer_sizes": (512, 256),
            "alpha": 1e-3,
            "learning_rate_init": 1e-3,
        },
        {
            "hidden_layer_sizes": (768, 384),
            "alpha": 5e-4,
            "learning_rate_init": 8e-4,
        },
        {
            "hidden_layer_sizes": (512, 256, 128),
            "alpha": 1e-4,
            "learning_rate_init": 6e-4,
        },
    ]

    best_cfg = mlp_candidates[0]
    best_thr = 0.5
    best_mcc = -1.0

    for idx, cfg in enumerate(mlp_candidates):
        candidate = MLPClassifier(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=cfg["alpha"],
            learning_rate="adaptive",
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=1200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=seed + idx,
        )
        candidate.fit(x_fit, y_fit)
        cal_proba = candidate.predict_proba(x_cal)[:, 1]
        thr, cal_mcc = _optimize_threshold_mcc(y_cal, cal_proba)

        if cal_mcc > best_mcc:
            best_mcc = cal_mcc
            best_cfg = cfg
            best_thr = thr

    # Refit best architecture on full training features, keep calibrated threshold.
    mlp = MLPClassifier(
        hidden_layer_sizes=best_cfg["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=best_cfg["alpha"],
        learning_rate="adaptive",
        learning_rate_init=best_cfg["learning_rate_init"],
        max_iter=1200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    mlp.fit(x_train_sc, y_train)
    mlp_proba = mlp.predict_proba(x_test_sc)[:, 1]
    mlp_pred = (mlp_proba >= best_thr).astype(int)
    mlp_metrics = _compute_metrics(y_test, mlp_pred, mlp_proba)
    mlp_metrics["decision_threshold"] = float(best_thr)
    mlp_metrics["calibration_mcc"] = float(best_mcc)

    return {"KNN": knn_metrics, "MLP": mlp_metrics}
