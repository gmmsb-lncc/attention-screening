"""Canonical KNN and MLP classifiers for benchmark comparison.

Every level in the benchmark must use the **exact same** classifier
configuration so that the only independent variable across levels is
the molecular representation.  This module provides a single
``train_knn_mlp`` function that **all levels** call.

Classifier specifications:

 * **KNN** — FAISS inner-product index on L2-normalised features
   (equivalent to cosine similarity), *k = 5*, distance-weighted voting.
 * **MLP** — ``sklearn.neural_network.MLPClassifier`` with two hidden
   layers of (256, 128) units, ReLU activation, Adam solver, adaptive
   learning rate, α = 1 × 10⁻³, max 2000 iterations, early stopping
   with patience = 20.
 * Both classifiers receive features after ``StandardScaler``.

Evaluation protocol:

 * All levels pass **validation-split** features as ``x_train`` / ``y_train``
   and **test-split** features as ``x_test`` / ``y_test``.
 * This eliminates train-set optimism for levels with learned feature
   extractors and ensures a consistent protocol across all levels.
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
from tqdm import tqdm


# ------------------------------------------------------------------ #
# FAISS-based KNN (cosine similarity via inner product on L2-normed vecs)
# ------------------------------------------------------------------ #

def _faiss_knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """KNN classification via FAISS cosine similarity with distance-weighted voting.

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


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def train_knn_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    frozen_mlp_selection: dict[str, object] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Train canonical KNN and MLP classifiers and return metric dicts.

    **Both classifiers use the exact same hyperparameters** across all
    benchmark levels so that the comparison is scientifically valid.

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
    frozen_mlp_selection : dict or None
        Ignored (kept for API compatibility with level runners).

    Returns
    -------
    dict
        ``{"KNN": {metric: value, ...}, "MLP": {metric: value, ...}}``
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # StandardScaler — identical preprocessing for both classifiers
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train).astype(np.float32)
    x_test_sc = scaler.transform(x_test).astype(np.float32)

    # ---------- KNN (FAISS, k=5, cosine, distance-weighted) ----------
    tqdm.write("    KNN (k=5, cosine similarity)...")
    knn_pred, knn_proba = _faiss_knn_predict(x_train_sc, y_train, x_test_sc, k=5)
    knn_metrics = _compute_metrics(y_test, knn_pred, knn_proba)

    # ---------- MLP (traditional, fixed architecture) ----------
    tqdm.write("    MLP (256-128, Adam, early stopping)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-5,
        random_state=seed,
    )
    mlp.fit(x_train_sc, y_train)

    mlp_proba = mlp.predict_proba(x_test_sc)[:, 1]
    mlp_pred = (mlp_proba >= 0.5).astype(int)
    mlp_metrics = _compute_metrics(y_test, mlp_pred, mlp_proba)

    return {"KNN": knn_metrics, "MLP": mlp_metrics}


def train_mlp_only(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    frozen_mlp_selection: dict[str, object] | None = None,
) -> Dict[str, float]:
    """Train and evaluate only MLP (skip KNN entirely)."""
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train).astype(np.float32)
    x_test_sc = scaler.transform(x_test).astype(np.float32)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-5,
        random_state=seed,
    )
    mlp.fit(x_train_sc, y_train)

    mlp_proba = mlp.predict_proba(x_test_sc)[:, 1]
    mlp_pred = (mlp_proba >= 0.5).astype(int)
    return _compute_metrics(y_test, mlp_pred, mlp_proba)
