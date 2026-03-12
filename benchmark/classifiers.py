"""Canonical KNN and MLP classifiers for benchmark comparison.

Every level in the benchmark must use the **exact same** classifier
configuration so that the only independent variable across levels is
the molecular representation.  This module provides a single
``train_knn_mlp`` function that **all active levels** call.

Classifier specifications
-------------------------
 * **KNN** — FAISS inner-product index on L2-normalised features
   (equivalent to cosine similarity), *k = 5*, distance-weighted voting.
 * **MLP** — classic ``sklearn.neural_network.MLPClassifier`` with
     fixed architecture and hyperparameters shared by all active levels:
     ``hidden_layer_sizes=(512,)``, ReLU, Adam, ``alpha=1e-4``,
     ``max_iter=500``, ``early_stopping=False``.

 * **Decision threshold** — adaptive per seed/model.
     A calibration split is carved from the fit partition; threshold is
     chosen by maximizing MCC on that calibration subset, then frozen and
     applied to the evaluation split.

 * Both classifiers receive features after ``StandardScaler``.

Evaluation protocol
-------------------
 * All levels pass **fit** features as ``x_train`` / ``y_train`` and
     **evaluation** features as ``x_test`` / ``y_test``.
 * Split selection is mode-dependent in each level runner:
     train mode uses train→val; test mode uses val→test.
 * Threshold adaptation never peeks at evaluation labels.
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
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
CALIBRATION_FRACTION = 0.2


# ------------------------------------------------------------------ #
# FAISS-based KNN (matches split_comparison_analysis.faiss_knn_predict)
# ------------------------------------------------------------------ #

def _faiss_knn_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Return positive-class probabilities from FAISS/scikit KNN.

    Identical to ``split_comparison_analysis.faiss_knn_predict``:
    L2-normalise → inner-product search → distance-weighted voting.
    """
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        logger.info("FAISS not available — falling back to sklearn KNeighborsClassifier")
        return _sklearn_knn_proba(x_train, y_train, x_test, k=k)

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

    # Probability for positive class
    if len(classes) == 1:
        return np.ones(n_test, dtype=np.float64) if int(classes[0]) == 1 else np.zeros(n_test, dtype=np.float64)

    row_sums = np.maximum(class_scores.sum(axis=1, keepdims=True), 1e-12)
    pos_idx = np.where(classes == 1)[0]
    if len(pos_idx) == 0:
        return np.zeros(n_test, dtype=np.float64)

    return class_scores[:, int(pos_idx[0])] / row_sums.ravel()


def _sklearn_knn_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 5,
) -> np.ndarray:
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
    proba_matrix = knn.predict_proba(x_test)
    class_labels = knn.classes_
    pos_idx = np.where(class_labels == 1)[0]
    if len(pos_idx) == 0:
        return np.zeros(len(x_test), dtype=np.float64)
    return proba_matrix[:, int(pos_idx[0])]


def _optimize_threshold_mcc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """Select threshold that maximizes MCC on calibration labels."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD

    clipped = np.clip(y_proba, 0.0, 1.0)
    if len(clipped) <= 200:
        candidates = np.unique(clipped)
    else:
        candidates = np.linspace(0.01, 0.99, 99)

    candidates = np.unique(np.append(candidates, DEFAULT_THRESHOLD))

    best_thr = DEFAULT_THRESHOLD
    best_mcc = -np.inf

    for thr in candidates:
        pred = (clipped >= thr).astype(int)
        mcc = float(matthews_corrcoef(y_true, pred))
        is_better = mcc > best_mcc
        is_tie_better = np.isclose(mcc, best_mcc) and abs(thr - DEFAULT_THRESHOLD) < abs(best_thr - DEFAULT_THRESHOLD)
        if is_better or is_tie_better:
            best_mcc = mcc
            best_thr = float(thr)

    return best_thr


def _split_fit_for_calibration(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    fraction: float = CALIBRATION_FRACTION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split fit data into model-train and threshold-calibration partitions."""
    if len(x_fit) < 50 or len(np.unique(y_fit)) < 2:
        return x_fit, y_fit, x_fit, y_fit

    x_model, x_cal, y_model, y_cal = train_test_split(
        x_fit,
        y_fit,
        test_size=fraction,
        random_state=seed,
        stratify=y_fit,
    )

    if len(np.unique(y_model)) < 2 or len(np.unique(y_cal)) < 2:
        return x_fit, y_fit, x_fit, y_fit

    return x_model, y_model, x_cal, y_cal


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


def _apply_ligand_block_weight(
    x: np.ndarray,
    protein_dim: int | None,
    ligand_weight: float,
) -> np.ndarray:
    """Apply relative ligand block weight on already-scaled features.

    Parameters
    ----------
    x : ndarray [n_samples, n_features]
        Feature matrix after StandardScaler.
    protein_dim : int | None
        Number of leading protein features. Remaining columns are ligand features.
        If ``None`` or invalid, no weighting is applied.
    ligand_weight : float
        Multiplicative factor for ligand block.
    """
    if ligand_weight == 1.0 or protein_dim is None:
        return x

    if protein_dim <= 0 or protein_dim >= x.shape[1]:
        logger.warning(
            "Skipping ligand weighting: invalid protein_dim=%s for n_features=%s",
            protein_dim,
            x.shape[1],
        )
        return x

    xw = np.asarray(x, dtype=np.float32).copy()
    xw[:, protein_dim:] *= float(ligand_weight)
    return xw


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def train_knn_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    protein_dim: int | None = None,
    ligand_weight: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Train canonical KNN and MLP classifiers and return metric dicts.

    **Both classifiers use the exact same hyperparameters** across all
    active benchmark levels so that the comparison is scientifically valid.

    Parameters
    ----------
    x_train, y_train : array-like
        Features and labels for classifier fitting.
        The concrete split depends on runner mode.
    x_test, y_test : array-like
        Features and labels for evaluation.
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

    x_model, y_model, x_cal, y_cal = _split_fit_for_calibration(
        x_train, y_train, seed=seed,
    )

    # StandardScaler fitted on model-train partition only.
    scaler = StandardScaler()
    x_model_sc = scaler.fit_transform(x_model).astype(np.float32)
    x_cal_sc = scaler.transform(x_cal).astype(np.float32)
    x_test_sc = scaler.transform(x_test).astype(np.float32)

    # Apply ligand block weighting after feature standardization.
    x_model_sc = _apply_ligand_block_weight(x_model_sc, protein_dim, ligand_weight)
    x_cal_sc = _apply_ligand_block_weight(x_cal_sc, protein_dim, ligand_weight)
    x_test_sc = _apply_ligand_block_weight(x_test_sc, protein_dim, ligand_weight)

    # ---------- KNN (FAISS, k=5, cosine, distance-weighted) ----------
    knn_cal_proba = _faiss_knn_proba(x_model_sc, y_model, x_cal_sc, k=5)
    knn_threshold = _optimize_threshold_mcc(y_cal, knn_cal_proba)
    knn_proba = _faiss_knn_proba(x_model_sc, y_model, x_test_sc, k=5)
    knn_pred = (knn_proba >= knn_threshold).astype(int)
    knn_metrics = _compute_metrics(y_test, knn_pred, knn_proba)
    knn_metrics["threshold"] = float(knn_threshold)
    knn_metrics["details"] = {
        "fit": {
            "n_rows": int(len(y_train)),
            "protein_dim": int(protein_dim) if protein_dim is not None else None,
            "ligand_weight": float(ligand_weight),
        },
        "model_train": {
            "n_rows": int(len(y_model)),
            "class_balance": float(np.mean(y_model)) if len(y_model) > 0 else 0.0,
        },
        "calibration": {
            "n_rows": int(len(y_cal)),
            "y_true": y_cal.astype(int).tolist(),
            "y_proba": np.asarray(knn_cal_proba, dtype=np.float64).tolist(),
            "y_pred": (np.asarray(knn_cal_proba, dtype=np.float64) >= knn_threshold).astype(int).tolist(),
        },
        "evaluation": {
            "n_rows": int(len(y_test)),
            "y_true": y_test.astype(int).tolist(),
            "y_proba": np.asarray(knn_proba, dtype=np.float64).tolist(),
            "y_pred": knn_pred.astype(int).tolist(),
        },
    }

    # ---------- MLP (classic fixed architecture) ----------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(512,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=False,
        random_state=seed,
    )
    mlp.fit(x_model_sc, y_model)
    mlp_cal_proba = mlp.predict_proba(x_cal_sc)[:, 1]
    mlp_threshold = _optimize_threshold_mcc(y_cal, mlp_cal_proba)
    mlp_proba = mlp.predict_proba(x_test_sc)[:, 1]
    mlp_pred = (mlp_proba >= mlp_threshold).astype(int)
    mlp_metrics = _compute_metrics(y_test, mlp_pred, mlp_proba)
    mlp_metrics["threshold"] = float(mlp_threshold)
    mlp_metrics["details"] = {
        "fit": {
            "n_rows": int(len(y_train)),
            "protein_dim": int(protein_dim) if protein_dim is not None else None,
            "ligand_weight": float(ligand_weight),
        },
        "model_train": {
            "n_rows": int(len(y_model)),
            "class_balance": float(np.mean(y_model)) if len(y_model) > 0 else 0.0,
        },
        "calibration": {
            "n_rows": int(len(y_cal)),
            "y_true": y_cal.astype(int).tolist(),
            "y_proba": np.asarray(mlp_cal_proba, dtype=np.float64).tolist(),
            "y_pred": (np.asarray(mlp_cal_proba, dtype=np.float64) >= mlp_threshold).astype(int).tolist(),
        },
        "evaluation": {
            "n_rows": int(len(y_test)),
            "y_true": y_test.astype(int).tolist(),
            "y_proba": np.asarray(mlp_proba, dtype=np.float64).tolist(),
            "y_pred": mlp_pred.astype(int).tolist(),
        },
    }

    return {"KNN": knn_metrics, "MLP": mlp_metrics}
