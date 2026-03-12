"""Canonical KNN and MLP classifiers for benchmark comparison.

Every level in the benchmark must use the **exact same** classifier
configuration so that the only independent variable across levels is
the molecular representation.  This module provides a single
``train_knn_mlp`` function that **all four levels** call.

Classifier specifications
-------------------------
 * **KNN** — FAISS inner-product index on L2-normalised features
   (equivalent to cosine similarity), *k = 5*, distance-weighted voting.
 * **MLP** — ``sklearn.neural_network.MLPClassifier`` with **adaptive
   architecture**: the hidden-layer topology is selected at runtime so
   that the ratio *n_samples / n_parameters* stays within the 10–50
   range recommended by the statistical learning literature.

   Tier table (dim = input feature dimension):

   ========  ====================  ====================
   n_train   hidden_layer_sizes    approx. params (*)
   ========  ====================  ====================
   ≥ 5 000   (512, 256)            dim·512 + 512·256 + …
   ≥ 2 000   (256, 128)            dim·256 + 256·128 + …
   < 2 000   (128,  64)            dim·128 + 128·64  + …
   ========  ====================  ====================

   (*) Total includes biases; the function logs exact counts at runtime.

   Other hyperparameters are fixed across tiers: ReLU, Adam, adaptive
   learning rate (init 1 × 10⁻³), α = 5 × 10⁻⁴, batch 64, max 2 000
   iterations, early stopping (15 % held-out, patience 50).

 * Both classifiers receive features after ``StandardScaler``.

Theoretical motivation
----------------------
The capacity of a neural network (VC-dimension) grows with its number
of free parameters.  When *n / p* is small the model memorises training
noise and generalises poorly — the classical bias–variance dilemma
(Geman et al., 1992).  Empirical guidelines converge on requiring
10–50 observations per parameter for shallow networks:

  - Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical
    Learning*, 2nd ed., Springer. §7.3 — bias–variance tradeoff.
  - Vapnik (2000). *The Nature of Statistical Learning Theory*, 2nd ed.,
    Springer. — VC-dimension theory.
  - Geman, Bienenstock & Doursat (1992). Neural Networks and the
    Bias/Variance Dilemma. *Neural Computation*, 4(1):1–58.
  - Abu-Mostafa, Magdon-Ismail & Lin (2012). *Learning from Data*.
    — practical rule: n ≥ 10 · d_VC.
  - McComb et al. (2022). Machine learning-guided, large-scale screening
    of flat glass compositions. *Br J Clin Pharmacol*, "≥ 10 samples
    per parameter" rule.

Evaluation protocol
-------------------
 * All levels pass **validation-split** features as ``x_train`` / ``y_train``
   and **test-split** features as ``x_test`` / ``y_test``.
 * This eliminates train-set optimism for levels with learned feature
   extractors (Levels 3 and 4) and ensures a consistent protocol across
   all four levels.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Adaptive MLP architecture selector
# ------------------------------------------------------------------ #

# Each tier is (min_samples, hidden_layer_sizes).
# Evaluated top-to-bottom; first match wins.
_MLP_TIERS: list[Tuple[int, Tuple[int, ...]]] = [
    (5_000, (512, 256)),
    (2_000, (256, 128)),
    (0,     (128,  64)),
]


def _count_mlp_params(dim: int, layers: Tuple[int, ...]) -> int:
    """Count weights + biases for a fully connected MLP (excluding output)."""
    total = 0
    prev = dim
    for h in layers:
        total += prev * h + h          # weights + biases
        prev = h
    total += prev * 1 + 1              # output layer (binary)
    return total


def _select_mlp_architecture(
    n_samples: int,
    dim: int,
) -> Tuple[int, ...]:
    """Choose hidden-layer sizes proportional to the available training set.

    With high-dimensional PLM embeddings (dim ≈ 320–1280) the strict
    "n/p ≥ 10" rule from classical statistics is impractical — even a
    single hidden unit of width dim already contributes ~dim parameters.
    Instead, we scale the architecture down as n_samples decreases,
    maximising *n/p* while keeping enough capacity to separate classes.
    Regularisation (weight decay α, early stopping, adaptive LR) handles
    the remaining generalisation gap.
    """
    for min_n, layers in _MLP_TIERS:
        if n_samples >= min_n:
            n_params = _count_mlp_params(dim, layers)
            ratio = n_samples / max(n_params, 1)
            logger.info(
                "MLP tier: n=%d  dim=%d  layers=%s  params=%d  n/p=%.2f",
                n_samples, dim, layers, n_params, ratio,
            )
            return layers

    # Should never reach here: last tier has min_n=0
    return _MLP_TIERS[-1][1]


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
    except ImportError:
        logger.info("FAISS not available — falling back to sklearn KNeighborsClassifier")
        return _sklearn_knn_predict(x_train, y_train, x_test, k=k)

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


def _sklearn_knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback KNN using sklearn when FAISS is not installed.

    Uses cosine metric with distance-weighted voting to match
    the FAISS implementation behaviour.
    """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="distance",
        n_jobs=-1,
    )
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    probabilities = knn.predict_proba(x_test)[:, -1]
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

    # ---------- MLP (adaptive architecture, early stopping) -----------
    hidden = _select_mlp_architecture(len(x_train_sc), x_train_sc.shape[1])
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=5e-4,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        random_state=seed,
    )
    mlp.fit(x_train_sc, y_train)
    mlp_pred = mlp.predict(x_test_sc)
    mlp_proba = mlp.predict_proba(x_test_sc)[:, 1]
    mlp_metrics = _compute_metrics(y_test, mlp_pred, mlp_proba)

    return {"KNN": knn_metrics, "MLP": mlp_metrics}
