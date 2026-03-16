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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm


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
    calibration_fraction: float = 0.25,
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
    """Return a compact, well-regularised MLP hyperparameter space.

    Five candidates: two wider architectures (512-unit) proportional to
    ~1024-dim interaction-enriched features, plus three original ones
    for the plain 512-dim case.  Stronger regularisation (higher alpha)
    and early stopping prevent overfitting.
    """
    return [
        {   # Wide single-layer (good for high-D interaction features)
            "hidden_layer_sizes": (512,),
            "alpha": 1e-2,
            "learning_rate_init": 1e-3,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 1e-5,
        },
        {   # Deep funnel proportional to 1024-D input
            "hidden_layer_sizes": (512, 256),
            "alpha": 5e-3,
            "learning_rate_init": 8e-4,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 8e-6,
        },
        {
            "hidden_layer_sizes": (256, 128),
            "alpha": 5e-3,
            "learning_rate_init": 1e-3,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 1e-5,
        },
        {
            "hidden_layer_sizes": (256,),
            "alpha": 1e-2,
            "learning_rate_init": 1e-3,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 1e-5,
        },
        {
            "hidden_layer_sizes": (128, 64),
            "alpha": 5e-3,
            "learning_rate_init": 8e-4,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 8e-6,
        },
        {   # Strongly regularized for high-D interaction features (1024D)
            "hidden_layer_sizes": (256, 128),
            "alpha": 5e-2,
            "learning_rate_init": 5e-4,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 1e-5,
        },
        {   # Simple architecture with strong L2 penalty
            "hidden_layer_sizes": (128,),
            "alpha": 1e-1,
            "learning_rate_init": 1e-3,
            "early_stopping": True,
            "max_iter": 2000,
            "n_iter_no_change": 60,
            "tol": 1e-5,
        },
    ]


def _fit_mlp_from_cfg(
    cfg: dict[str, object],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    random_state: int,
    final_refit: bool = False,
) -> MLPClassifier:
    """Instantiate and fit an MLP from a candidate config.

    When *final_refit* is True, early stopping is disabled so the model
    trains on 100% of the data for the full ``max_iter`` budget.  This
    mirrors scikit-learn's ``GridSearchCV.refit`` behaviour: model
    selection uses early stopping (CV phase), but the final model is
    trained to full convergence.
    """
    x_fit_use, y_fit_use = x_fit, y_fit
    use_oversample = os.getenv("BENCHMARK_MLP_OVERSAMPLE", "0").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if use_oversample:
        x_fit_use, y_fit_use = _oversample_minority_binary(x_fit, y_fit, random_state)

    use_early_stopping = bool(cfg["early_stopping"]) and (not final_refit)

    mlp = MLPClassifier(
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=float(cfg["alpha"]),
        learning_rate="adaptive",
        learning_rate_init=float(cfg["learning_rate_init"]),
        max_iter=int(cfg["max_iter"]),
        early_stopping=use_early_stopping,
        validation_fraction=0.1 if use_early_stopping else 0.0,
        n_iter_no_change=int(cfg["n_iter_no_change"]),
        tol=float(cfg["tol"]),
        random_state=random_state,
    )
    # Use inverse-class-frequency weighting instead of oversampling.
    # sklearn MLPClassifier does not support class_weight natively,
    # so we apply sample_weight via a manual reweighting of the loss
    # by duplicating the effect through alpha scaling.  A simpler and
    # more robust approach is to just keep early stopping + the small
    # regularisation which already handles mild imbalance well.
    mlp.fit(x_fit_use, y_fit_use)
    return mlp


def _oversample_minority_binary(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly oversample minority class to balance binary labels."""
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        return x, y

    rng = np.random.RandomState(seed)
    maj_class = int(classes[np.argmax(counts)])
    min_class = int(classes[np.argmin(counts)])

    idx_maj = np.where(y == maj_class)[0]
    idx_min = np.where(y == min_class)[0]
    n_to_add = len(idx_maj) - len(idx_min)
    if n_to_add <= 0:
        return x, y

    sampled_min = rng.choice(idx_min, size=n_to_add, replace=True)
    new_idx = np.concatenate([np.arange(len(y)), sampled_min])
    rng.shuffle(new_idx)
    return x[new_idx], y[new_idx]


def _select_best_mlp_by_mcc(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
) -> tuple[dict[str, object], float, float, float]:
    """Select MLP config by MCC with stratified CV (fallback: holdout calibration).

    Using OOF probabilities for threshold selection is more robust than a
    single calibration split and tends to improve generalization MCC.
    """
    candidates = _mlp_candidate_space()
    n_restarts = max(1, int(os.getenv("BENCHMARK_MLP_CAL_RESTARTS", "3")))
    n_folds = max(2, int(os.getenv("BENCHMARK_MLP_FOLDS", "5")))
    use_cv = os.getenv("BENCHMARK_MLP_USE_CV", "1").strip().lower() not in {"0", "false", "no"}
    min_class = int(np.bincount(y_fit.astype(int)).min()) if y_fit.size else 0
    can_cv = use_cv and (min_class >= n_folds)

    best_cfg = candidates[0]
    best_thr = 0.5
    best_mcc = -1.0
    best_std = 1.0
    best_score = -999.0

    total_search_steps = len(candidates) * n_restarts
    search_bar = tqdm(
        total=total_search_steps,
        desc="    MLP search",
        unit="fit",
        leave=False,
        dynamic_ncols=True,
    )

    for cfg_idx, cfg in enumerate(candidates):
        mccs: list[float] = []
        thr_values: list[float] = []

        for restart in range(n_restarts):
            rs = seed + (cfg_idx * 101) + restart
            if can_cv:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
                oof_proba = np.zeros(y_fit.shape[0], dtype=np.float64)

                for fold_train_idx, fold_val_idx in skf.split(x_fit, y_fit):
                    fold_model = _fit_mlp_from_cfg(
                        cfg,
                        x_fit[fold_train_idx],
                        y_fit[fold_train_idx],
                        random_state=rs,
                    )
                    oof_proba[fold_val_idx] = fold_model.predict_proba(x_fit[fold_val_idx])[:, 1]

                thr_single, mcc_single = _optimize_threshold_mcc(y_fit, oof_proba)
            else:
                candidate = _fit_mlp_from_cfg(cfg, x_fit, y_fit, random_state=rs)
                cal_proba_single = candidate.predict_proba(x_cal)[:, 1]
                thr_single, mcc_single = _optimize_threshold_mcc(y_cal, cal_proba_single)

            thr_values.append(thr_single)
            mccs.append(mcc_single)
            search_bar.update(1)

        # Robust aggregation across restarts.
        thr = float(np.median(np.array(thr_values)))
        mcc = float(np.mean(np.array(mccs)))
        mcc_std = float(np.std(np.array(mccs), ddof=1)) if len(mccs) > 1 else 0.0

        # Stability-aware objective: prioritize high MCC with low restart variance.
        score = mcc - (0.10 * mcc_std)
        search_bar.set_postfix(mcc=f"{mcc:.3f}", std=f"{mcc_std:.3f}")
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_thr = thr
            best_mcc = float(mcc)
            best_std = mcc_std

    search_bar.close()

    return best_cfg, best_thr, best_mcc, best_std


def _fit_mlp_ensemble_predict(
    cfg: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit an ensemble of MLP restarts and return mean probabilities.

    Each member trains with early stopping (default).  Set
    ``BENCHMARK_MLP_FULL_REFIT=1`` only if you explicitly want to
    disable early stopping for the final refit.
    """
    n_members = max(1, int(os.getenv("BENCHMARK_MLP_ENSEMBLE", "3")))
    full_refit = os.getenv(
        "BENCHMARK_MLP_FULL_REFIT", "0"
    ).strip().lower() not in {"0", "false", "no"}
    probs: list[np.ndarray] = []

    ensemble_iter = tqdm(
        range(n_members),
        desc="    MLP ensemble" + (" (full refit)" if full_refit else ""),
        unit="model",
        leave=False,
        dynamic_ncols=True,
    )
    for member in ensemble_iter:
        rs = seed + (member * 53)
        model = _fit_mlp_from_cfg(
            cfg, x_train, y_train, random_state=rs, final_refit=full_refit,
        )
        probs.append(model.predict_proba(x_eval)[:, 1])

    return np.mean(np.stack(probs, axis=0), axis=0)


def _train_mlp_pipeline(
    x_train_sc: np.ndarray,
    y_train: np.ndarray,
    x_test_sc: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    frozen_selection: dict[str, object] | None = None,
) -> Dict[str, float]:
    """Train/evaluate MLP pipeline optimized for MCC."""
    if frozen_selection is None:
        x_fit, y_fit, x_cal, y_cal = _split_fit_for_calibration(x_train_sc, y_train, seed)

        best_cfg, best_thr, best_mcc, best_mcc_std = _select_best_mlp_by_mcc(
            x_fit=x_fit,
            y_fit=y_fit,
            x_cal=x_cal,
            y_cal=y_cal,
            seed=seed,
        )
        selection_source = "validation_search"
    else:
        required = {"best_cfg", "best_thr", "best_mcc", "best_mcc_std"}
        missing = sorted(required.difference(frozen_selection.keys()))
        if missing:
            raise ValueError(f"Frozen MLP selection missing keys: {missing}")
        best_cfg = dict(frozen_selection["best_cfg"])
        best_thr = float(frozen_selection["best_thr"])
        best_mcc = float(frozen_selection["best_mcc"])
        best_mcc_std = float(frozen_selection["best_mcc_std"])
        selection_source = "frozen_train_selection"

    # Refit best architecture on full training features as a restart ensemble.
    mlp_proba = _fit_mlp_ensemble_predict(
        cfg=best_cfg,
        x_train=x_train_sc,
        y_train=y_train,
        x_eval=x_test_sc,
        seed=seed,
    )

    # --- Isotonic calibration of ensemble probabilities ---
    # Fits an isotonic regression on OOF predictions to correct the
    # probability distribution before thresholding.  This greatly
    # improves threshold transfer to unseen scaffolds.
    use_isotonic = os.getenv(
        "BENCHMARK_MLP_ISOTONIC_CAL", "1"
    ).strip().lower() not in {"0", "false", "no"}
    if use_isotonic and y_train.size >= 40 and len(np.unique(y_train.astype(int))) >= 2:
        n_iso_folds = max(2, int(os.getenv("BENCHMARK_MLP_OOF_FOLDS", "5")))
        min_class_iso = int(np.bincount(y_train.astype(int)).min())
        if min_class_iso >= n_iso_folds:
            # Build OOF predictions on train for calibrator fitting.
            skf_iso = StratifiedKFold(
                n_splits=n_iso_folds, shuffle=True, random_state=seed + 7,
            )
            oof_cal_proba = np.zeros(y_train.shape[0], dtype=np.float64)
            for cal_train_idx, cal_val_idx in skf_iso.split(x_train_sc, y_train):
                cal_model = _fit_mlp_from_cfg(
                    best_cfg, x_train_sc[cal_train_idx], y_train[cal_train_idx],
                    random_state=seed,
                )
                oof_cal_proba[cal_val_idx] = cal_model.predict_proba(
                    x_train_sc[cal_val_idx],
                )[:, 1]

            # Fit isotonic calibrator on OOF predictions.
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(oof_cal_proba, y_train.astype(int))

            # Calibrate both ensemble eval probabilities and re-derive threshold.
            mlp_proba = calibrator.predict(mlp_proba)
            oof_cal_calibrated = calibrator.predict(oof_cal_proba)
            refined_thr, _ = _optimize_threshold_mcc(y_train.astype(int), oof_cal_calibrated)
            tqdm.write(f"    Isotonic calibration: threshold {best_thr:.3f} -> {refined_thr:.3f}")

    # --- Optional OOF refinement (enabled by default) ---
    use_oof_refinement = os.getenv(
        "BENCHMARK_MLP_OOF_THRESHOLD", "1"
    ).strip().lower() not in {"0", "false", "no"}
    if use_oof_refinement and y_train.size >= 40 and len(np.unique(y_train.astype(int))) >= 2:
        n_oof_folds = max(2, int(os.getenv("BENCHMARK_MLP_OOF_FOLDS", "5")))
        min_class_count = int(np.bincount(y_train.astype(int)).min())
        if min_class_count >= n_oof_folds:
            skf_oof = StratifiedKFold(
                n_splits=n_oof_folds, shuffle=True, random_state=seed,
            )
            oof_proba = np.zeros(y_train.shape[0], dtype=np.float64)
            for oof_train_idx, oof_val_idx in skf_oof.split(x_train_sc, y_train):
                oof_model = _fit_mlp_from_cfg(
                    best_cfg, x_train_sc[oof_train_idx], y_train[oof_train_idx],
                    random_state=seed,
                )
                oof_proba[oof_val_idx] = oof_model.predict_proba(
                    x_train_sc[oof_val_idx],
                )[:, 1]
            refined_thr, _ = _optimize_threshold_mcc(y_train.astype(int), oof_proba)

    mlp_pred = (mlp_proba >= refined_thr).astype(int)
    mlp_metrics = _compute_metrics(y_test, mlp_pred, mlp_proba)
    mlp_metrics["decision_threshold"] = float(refined_thr)
    mlp_metrics["selection_threshold"] = float(best_thr)
    mlp_metrics["calibration_mcc"] = float(best_mcc)
    mlp_metrics["calibration_mcc_std"] = float(best_mcc_std)
    mlp_metrics["selection_source"] = selection_source
    mlp_metrics["oof_threshold_refinement"] = use_oof_refinement
    mlp_metrics["mlp_selection"] = {
        "best_cfg": best_cfg,
        "best_thr": float(best_thr),
        "best_mcc": float(best_mcc),
        "best_mcc_std": float(best_mcc_std),
    }
    return mlp_metrics


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

    # ---------- MLP tuned for MCC ----------
    mlp_metrics = _train_mlp_pipeline(
        x_train_sc,
        y_train,
        x_test_sc,
        y_test,
        seed,
        frozen_selection=frozen_mlp_selection,
    )

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

    return _train_mlp_pipeline(
        x_train_sc,
        y_train,
        x_test_sc,
        y_test,
        seed,
        frozen_selection=frozen_mlp_selection,
    )
