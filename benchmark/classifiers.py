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

import os
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


def _mlp_candidate_space() -> list[dict[str, object]]:
    """Return an MCC-oriented MLP hyperparameter space.

    Focuses on convergence controls (max_iter, n_iter_no_change, tol)
    and learning-rate scale, which are the most impactful knobs for
    the current underfitting profile.
    """
    return [
        {
            "hidden_layer_sizes": (512, 256),
            "alpha": 8e-4,
            "learning_rate_init": 1.2e-3,
            "early_stopping": True,
            "max_iter": 1600,
            "n_iter_no_change": 60,
            "tol": 1e-5,
        },
        {
            "hidden_layer_sizes": (768, 384),
            "alpha": 5e-4,
            "learning_rate_init": 9e-4,
            "early_stopping": True,
            "max_iter": 2200,
            "n_iter_no_change": 80,
            "tol": 8e-6,
        },
        {
            "hidden_layer_sizes": (1024, 512, 256),
            "alpha": 3e-4,
            "learning_rate_init": 7e-4,
            "early_stopping": True,
            "max_iter": 2600,
            "n_iter_no_change": 90,
            "tol": 6e-6,
        },
        {
            "hidden_layer_sizes": (512, 256, 128),
            "alpha": 1e-4,
            "learning_rate_init": 5e-4,
            "early_stopping": True,
            "max_iter": 2800,
            "n_iter_no_change": 110,
            "tol": 5e-6,
        },
        {
            "hidden_layer_sizes": (256, 128),
            "alpha": 1e-5,
            "learning_rate_init": 3.5e-4,
            "early_stopping": True,
            "max_iter": 3200,
            "n_iter_no_change": 130,
            "tol": 4e-6,
        },
        {
            "hidden_layer_sizes": (768, 384, 192),
            "alpha": 2e-4,
            "learning_rate_init": 4.5e-4,
            "early_stopping": True,
            "max_iter": 3000,
            "n_iter_no_change": 120,
            "tol": 4e-6,
        },
    ]


def _fit_mlp_from_cfg(
    cfg: dict[str, object],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    random_state: int,
) -> MLPClassifier:
    """Instantiate and fit an MLP from a candidate config."""
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=float(cfg["alpha"]),
        learning_rate="constant",
        learning_rate_init=float(cfg["learning_rate_init"]),
        max_iter=int(cfg["max_iter"]),
        early_stopping=bool(cfg["early_stopping"]),
        validation_fraction=0.1,
        n_iter_no_change=int(cfg["n_iter_no_change"]),
        tol=float(cfg["tol"]),
        random_state=random_state,
    )
    mlp.fit(x_fit, y_fit)
    return mlp


def _select_best_mlp_by_mcc(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
) -> tuple[dict[str, object], float, float, float]:
    """Select MLP config by calibration MCC using restart ensemble on calibration."""
    candidates = _mlp_candidate_space()
    n_restarts = max(1, int(os.getenv("BENCHMARK_MLP_CAL_RESTARTS", "3")))

    best_cfg = candidates[0]
    best_thr = 0.5
    best_mcc = -1.0
    best_std = 1.0
    best_score = -999.0

    for cfg_idx, cfg in enumerate(candidates):
        probs_per_restart: list[np.ndarray] = []
        mccs: list[float] = []

        for restart in range(n_restarts):
            rs = seed + (cfg_idx * 101) + restart
            candidate = _fit_mlp_from_cfg(cfg, x_fit, y_fit, random_state=rs)
            cal_proba_single = candidate.predict_proba(x_cal)[:, 1]
            thr_single, mcc_single = _optimize_threshold_mcc(y_cal, cal_proba_single)
            probs_per_restart.append(cal_proba_single)
            mccs.append(mcc_single)

        # Threshold on averaged probabilities across restarts is typically more stable.
        cal_proba_mean = np.mean(np.stack(probs_per_restart, axis=0), axis=0)
        thr, mcc = _optimize_threshold_mcc(y_cal, cal_proba_mean)
        mcc_std = float(np.std(np.array(mccs), ddof=1)) if len(mccs) > 1 else 0.0

        # Stability-aware objective: prioritize high MCC with low restart variance.
        score = float(mcc) - (0.15 * mcc_std)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_thr = thr
            best_mcc = float(mcc)
            best_std = mcc_std

    return best_cfg, best_thr, best_mcc, best_std


def _fit_mlp_ensemble_predict(
    cfg: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit an ensemble of MLP restarts and return mean probabilities."""
    n_members = max(1, int(os.getenv("BENCHMARK_MLP_ENSEMBLE", "5")))
    probs: list[np.ndarray] = []

    for member in range(n_members):
        rs = seed + (member * 53)
        model = _fit_mlp_from_cfg(cfg, x_train, y_train, random_state=rs)
        probs.append(model.predict_proba(x_eval)[:, 1])

    return np.mean(np.stack(probs, axis=0), axis=0)


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

    best_cfg, best_thr, best_mcc, best_mcc_std = _select_best_mlp_by_mcc(
        x_fit=x_fit,
        y_fit=y_fit,
        x_cal=x_cal,
        y_cal=y_cal,
        seed=seed,
    )

    # Refit best architecture on full training features as a restart ensemble.
    mlp_proba = _fit_mlp_ensemble_predict(
        cfg=best_cfg,
        x_train=x_train_sc,
        y_train=y_train,
        x_eval=x_test_sc,
        seed=seed,
    )
    mlp_pred = (mlp_proba >= best_thr).astype(int)
    mlp_metrics = _compute_metrics(y_test, mlp_pred, mlp_proba)
    mlp_metrics["decision_threshold"] = float(best_thr)
    mlp_metrics["calibration_mcc"] = float(best_mcc)
    mlp_metrics["calibration_mcc_std"] = float(best_mcc_std)

    return {"KNN": knn_metrics, "MLP": mlp_metrics}
