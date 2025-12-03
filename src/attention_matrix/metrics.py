"""
Metrics Calculator for Attention Matrix Pipeline.

Computes comprehensive classification and regression metrics.
Single Responsibility: Metrics computation only.
"""

import warnings
import numpy as np
from typing import Dict, Any, Tuple

# Suppress scipy ConstantInputWarning for correlation calculations
warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

from sklearn.metrics import (
    # Classification metrics
    accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix,
    classification_report,
    # Regression metrics
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score
)
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary with all classification metrics
    """
    metrics = {
        # Basic metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        
        # ROC and PR curves
        'roc_auc': roc_auc_score(y_true, y_prob),
        'average_precision': average_precision_score(y_true, y_prob),
        
        # F1, Precision, Recall
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        
        # Matthews Correlation Coefficient
        'mcc': matthews_corrcoef(y_true, y_pred),
        
        # Specificity (True Negative Rate)
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    }
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: True continuous values (pChEMBL)
        y_pred: Predicted continuous values
        
    Returns:
        Dictionary with all regression metrics
    """
    # R² (coefficient of determination)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    
    # Spearman rank correlation
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    # Kendall's tau (rank correlation - robust to outliers)
    kendall_tau, kendall_p = kendalltau(y_true, y_pred)
    
    # Concordance Correlation Coefficient (Lin's CCC)
    # Measures agreement between predicted and observed, not just correlation
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    
    # Error metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Explained variance
    explained_var = explained_variance_score(y_true, y_pred)
    
    # Additional metrics
    # Mean Absolute Percentage Error (MAPE) - avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Max error
    max_error = np.max(np.abs(y_true - y_pred))
    
    metrics = {
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p_value': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p,
        'ccc': ccc,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'max_error': max_error,
        'explained_variance': explained_var
    }
    
    return metrics


def compute_all_metrics(
    cls_true: np.ndarray,
    cls_pred: np.ndarray,
    cls_prob: np.ndarray,
    reg_true: np.ndarray,
    reg_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute all classification and regression metrics.
    
    Args:
        cls_true: True binary labels
        cls_pred: Predicted binary labels
        cls_prob: Predicted probabilities
        reg_true: True continuous values
        reg_pred: Predicted continuous values
        
    Returns:
        Dictionary with 'classification' and 'regression' metrics
    """
    return {
        'classification': compute_classification_metrics(cls_true, cls_pred, cls_prob),
        'regression': compute_regression_metrics(reg_true, reg_pred)
    }


def print_metrics_summary(metrics: Dict[str, Dict[str, float]], prefix: str = "") -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary with 'classification' and 'regression' metrics
        prefix: Prefix for print lines (e.g., "  " for indentation)
    """
    print(f"\n{prefix}=== CLASSIFICATION METRICS (Binary: Active/Inactive) ===")
    cls = metrics['classification']
    print(f"{prefix}  Accuracy:          {cls['accuracy']:.4f}")
    print(f"{prefix}  Balanced Accuracy: {cls['balanced_accuracy']:.4f}")
    print(f"{prefix}  ROC-AUC:           {cls['roc_auc']:.4f}")
    print(f"{prefix}  Average Precision: {cls['average_precision']:.4f}")
    print(f"{prefix}  F1 Score:          {cls['f1']:.4f}")
    print(f"{prefix}  Precision:         {cls['precision']:.4f}")
    print(f"{prefix}  Recall:            {cls['recall']:.4f}")
    print(f"{prefix}  Specificity:       {cls['specificity']:.4f}")
    print(f"{prefix}  MCC:               {cls['mcc']:.4f}")
    print(f"{prefix}  Confusion Matrix:  TP={cls['true_positives']}, TN={cls['true_negatives']}, "
          f"FP={cls['false_positives']}, FN={cls['false_negatives']}")
    
    print(f"\n{prefix}=== REGRESSION METRICS (pChEMBL Prediction) ===")
    reg = metrics['regression']
    print(f"{prefix}  R² (coef. determination): {reg['r2']:.4f}")
    print(f"{prefix}  Pearson r:                {reg['pearson_r']:.4f} (p={reg['pearson_p_value']:.2e})")
    print(f"{prefix}  Spearman ρ:               {reg['spearman_r']:.4f} (p={reg['spearman_p_value']:.2e})")
    print(f"{prefix}  Kendall τ:                {reg['kendall_tau']:.4f} (p={reg['kendall_p_value']:.2e})")
    print(f"{prefix}  CCC (Lin's):              {reg['ccc']:.4f}")
    print(f"{prefix}  RMSE:                     {reg['rmse']:.4f}")
    print(f"{prefix}  MAE:                      {reg['mae']:.4f}")
    print(f"{prefix}  Max Error:                {reg['max_error']:.4f}")
    print(f"{prefix}  Explained Variance:       {reg['explained_variance']:.4f}")


def format_metrics_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics dictionary for JSON serialization.
    
    Converts numpy types to Python native types.
    """
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    return convert(metrics)
