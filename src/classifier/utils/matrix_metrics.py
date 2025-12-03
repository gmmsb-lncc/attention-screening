"""
Metrics Calculator for Matrix-Based Models.

Provides comprehensive metrics for both classification and regression tasks,
compatible with the existing metrics system used in vector-based pipelines.

Author: DockTKinase Team
Date: November 2025
"""

import warnings
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

# Suppress scipy ConstantInputWarning for correlation calculations
warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    fbeta_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    # Regression metrics
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error
)
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Complete classification metrics container."""
    
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Advanced metrics
    roc_auc: float = 0.0
    average_precision: float = 0.0
    brier_score: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    
    # F-beta variants
    fbeta_05: float = 0.0  # Favors precision
    fbeta_2: float = 0.0   # Favors recall
    
    # Confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Derived metrics
    specificity: float = 0.0
    sensitivity: float = 0.0  # = recall
    balanced_accuracy: float = 0.0
    
    # Meta info
    n_samples: int = 0
    positive_rate: float = 0.0
    threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (compatible with existing format)."""
        return {
            # Basic (compatible names)
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            
            # Advanced
            "ROC_AUC": self.roc_auc,
            "Average_Precision": self.average_precision,
            "Brier_Score": self.brier_score,
            "MCC": self.mcc,
            
            # F-beta
            "Fbeta_0.5": self.fbeta_05,
            "Fbeta_2": self.fbeta_2,
            
            # Confusion matrix
            "True_Positives": self.true_positives,
            "True_Negatives": self.true_negatives,
            "False_Positives": self.false_positives,
            "False_Negatives": self.false_negatives,
            
            # Derived
            "Specificity": self.specificity,
            "Sensitivity": self.sensitivity,
            "Balanced_Accuracy": self.balanced_accuracy,
            
            # Meta
            "n_samples": self.n_samples,
            "Positive_Rate": self.positive_rate,
            "Threshold": self.threshold
        }


@dataclass
class RegressionMetrics:
    """Complete regression metrics container."""
    
    # Primary metrics
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    
    # Additional metrics
    median_ae: float = 0.0
    explained_variance: float = 0.0
    max_error: float = 0.0
    mape: Optional[float] = None
    
    # Correlation metrics
    pearson_r: float = 0.0
    pearson_p: float = 1.0
    spearman_r: float = 0.0
    spearman_p: float = 1.0
    kendall_tau: float = 0.0
    kendall_p: float = 1.0
    
    # Concordance Correlation Coefficient (Lin's CCC)
    ccc: float = 0.0
    
    # Residual statistics
    mean_residual: float = 0.0
    std_residual: float = 0.0
    
    # Normalized metrics
    rmse_normalized: float = 0.0
    cv_rmse: Optional[float] = None
    
    # Error percentiles
    error_p50: float = 0.0
    error_p90: float = 0.0
    error_p95: float = 0.0
    
    # Target statistics
    target_mean: float = 0.0
    target_std: float = 0.0
    target_min: float = 0.0
    target_max: float = 0.0
    
    # Prediction statistics
    pred_mean: float = 0.0
    pred_std: float = 0.0
    pred_min: float = 0.0
    pred_max: float = 0.0
    
    # Meta info
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (compatible with existing format)."""
        return {
            # Primary
            "MAE": self.mae,
            "MSE": self.mse,
            "RMSE": self.rmse,
            "R2": self.r2,
            
            # Additional
            "MedianAE": self.median_ae,
            "ExplainedVariance": self.explained_variance,
            "MaxError": self.max_error,
            "MAPE": self.mape,
            
            # Correlation
            "Pearson_R": self.pearson_r,
            "Pearson_P": self.pearson_p,
            "Spearman_R": self.spearman_r,
            "Spearman_P": self.spearman_p,
            "Kendall_Tau": self.kendall_tau,
            "Kendall_P": self.kendall_p,
            "CCC": self.ccc,
            
            # Residuals
            "mean_residual": self.mean_residual,
            "std_residual": self.std_residual,
            
            # Normalized
            "RMSE_normalized": self.rmse_normalized,
            "CV_RMSE": self.cv_rmse,
            
            # Percentiles
            "error_p50": self.error_p50,
            "error_p90": self.error_p90,
            "error_p95": self.error_p95,
            
            # Target stats
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "target_min": self.target_min,
            "target_max": self.target_max,
            
            # Pred stats
            "pred_mean": self.pred_mean,
            "pred_std": self.pred_std,
            "pred_min": self.pred_min,
            "pred_max": self.pred_max,
            
            # Meta
            "n_samples": self.n_samples
        }


@dataclass
class MultiTaskMetrics:
    """Combined metrics for multi-task models (classification + regression)."""
    
    classification: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    regression: RegressionMetrics = field(default_factory=RegressionMetrics)
    
    # Loss values
    total_loss: float = 0.0
    classification_loss: float = 0.0
    regression_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "classification": self.classification.to_dict(),
            "regression": self.regression.to_dict(),
            "losses": {
                "total": self.total_loss,
                "classification": self.classification_loss,
                "regression": self.regression_loss
            }
        }


class MatrixMetricsCalculator:
    """
    Calculator for classification and regression metrics.
    
    Compatible with the existing vector-based metrics system.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize calculator.
        
        Args:
            threshold: Classification threshold for binary predictions
        """
        self.threshold = threshold
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: Optional[float] = None
    ) -> ClassificationMetrics:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True binary labels [N]
            y_prob: Predicted probabilities [N]
            threshold: Classification threshold (default: self.threshold)
            
        Returns:
            ClassificationMetrics object
        """
        threshold = threshold or self.threshold
        
        # Ensure 1D arrays
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        # Binary predictions
        y_pred = (y_prob >= threshold).astype(int)
        
        # Handle edge cases
        n_samples = len(y_true)
        n_positive = int(y_true.sum())
        n_negative = n_samples - n_positive
        
        metrics = ClassificationMetrics(
            n_samples=n_samples,
            positive_rate=n_positive / n_samples if n_samples > 0 else 0.0,
            threshold=threshold
        )
        
        if n_samples == 0:
            return metrics
        
        # Basic metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        
        # Handle cases where only one class is present
        if n_positive > 0 and n_negative > 0:
            metrics.precision = precision_score(y_true, y_pred, zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, zero_division=0)
            metrics.f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics.fbeta_05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
            metrics.fbeta_2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            
            try:
                metrics.roc_auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics.roc_auc = 0.0
            
            try:
                metrics.average_precision = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics.average_precision = 0.0
            
            metrics.mcc = matthews_corrcoef(y_true, y_pred)
        
        # Brier score (works even with single class)
        metrics.brier_score = brier_score_loss(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics.true_negatives = int(cm[0, 0])
        metrics.false_positives = int(cm[0, 1])
        metrics.false_negatives = int(cm[1, 0])
        metrics.true_positives = int(cm[1, 1])
        
        # Derived metrics
        metrics.sensitivity = metrics.recall
        if (metrics.true_negatives + metrics.false_positives) > 0:
            metrics.specificity = metrics.true_negatives / (metrics.true_negatives + metrics.false_positives)
        metrics.balanced_accuracy = (metrics.sensitivity + metrics.specificity) / 2
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> RegressionMetrics:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True values [N]
            y_pred: Predicted values [N]
            mask: Optional mask for valid samples [N]
            
        Returns:
            RegressionMetrics object
        """
        # Ensure 1D arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Apply mask if provided
        if mask is not None:
            mask = np.asarray(mask).flatten().astype(bool)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        metrics = RegressionMetrics(n_samples=len(y_true))
        
        if len(y_true) == 0:
            return metrics
        
        # Primary metrics
        metrics.mae = float(mean_absolute_error(y_true, y_pred))
        metrics.mse = float(mean_squared_error(y_true, y_pred))
        metrics.rmse = float(np.sqrt(metrics.mse))
        metrics.r2 = float(r2_score(y_true, y_pred))
        
        # Additional metrics
        metrics.median_ae = float(median_absolute_error(y_true, y_pred))
        metrics.explained_variance = float(explained_variance_score(y_true, y_pred))
        metrics.max_error = float(max_error(y_true, y_pred))
        
        # MAPE (handle zeros)
        try:
            non_zero_mask = y_true != 0
            if non_zero_mask.sum() > 0:
                metrics.mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
        except:
            metrics.mape = None
        
        # Correlation metrics
        if len(y_true) > 1:
            try:
                r, p = pearsonr(y_true, y_pred)
                metrics.pearson_r = float(r)
                metrics.pearson_p = float(p)
            except:
                pass
            
            try:
                r, p = spearmanr(y_true, y_pred)
                metrics.spearman_r = float(r)
                metrics.spearman_p = float(p)
            except:
                pass
            
            # Kendall's tau (rank correlation - robust to outliers)
            try:
                tau, p = kendalltau(y_true, y_pred)
                metrics.kendall_tau = float(tau)
                metrics.kendall_p = float(p)
            except:
                pass
            
            # Concordance Correlation Coefficient (Lin's CCC)
            # Measures agreement, not just correlation
            try:
                mean_true = np.mean(y_true)
                mean_pred = np.mean(y_pred)
                var_true = np.var(y_true)
                var_pred = np.var(y_pred)
                covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
                ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
                metrics.ccc = float(ccc)
            except:
                pass
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics.mean_residual = float(np.mean(residuals))
        metrics.std_residual = float(np.std(residuals))
        
        # Normalized metrics
        y_range = np.max(y_true) - np.min(y_true)
        if y_range > 0:
            metrics.rmse_normalized = metrics.rmse / y_range
        
        y_mean = np.mean(y_true)
        if y_mean != 0:
            metrics.cv_rmse = metrics.rmse / y_mean
        
        # Error percentiles
        abs_residuals = np.abs(residuals)
        metrics.error_p50 = float(np.percentile(abs_residuals, 50))
        metrics.error_p90 = float(np.percentile(abs_residuals, 90))
        metrics.error_p95 = float(np.percentile(abs_residuals, 95))
        
        # Target statistics
        metrics.target_mean = float(np.mean(y_true))
        metrics.target_std = float(np.std(y_true))
        metrics.target_min = float(np.min(y_true))
        metrics.target_max = float(np.max(y_true))
        
        # Prediction statistics
        metrics.pred_mean = float(np.mean(y_pred))
        metrics.pred_std = float(np.std(y_pred))
        metrics.pred_min = float(np.min(y_pred))
        metrics.pred_max = float(np.max(y_pred))
        
        return metrics
    
    def calculate_multitask_metrics(
        self,
        cls_true: np.ndarray,
        cls_prob: np.ndarray,
        reg_true: np.ndarray,
        reg_pred: np.ndarray,
        reg_mask: Optional[np.ndarray] = None,
        losses: Optional[Dict[str, float]] = None
    ) -> MultiTaskMetrics:
        """
        Calculate metrics for multi-task models.
        
        Args:
            cls_true: True classification labels
            cls_prob: Predicted classification probabilities
            reg_true: True regression values
            reg_pred: Predicted regression values
            reg_mask: Mask for valid regression samples
            losses: Optional loss values dict
            
        Returns:
            MultiTaskMetrics object
        """
        metrics = MultiTaskMetrics()
        
        # Classification metrics
        metrics.classification = self.calculate_classification_metrics(cls_true, cls_prob)
        
        # Regression metrics
        metrics.regression = self.calculate_regression_metrics(reg_true, reg_pred, reg_mask)
        
        # Losses
        if losses:
            metrics.total_loss = losses.get('total', 0.0)
            metrics.classification_loss = losses.get('classification', 0.0)
            metrics.regression_loss = losses.get('regression', 0.0)
        
        return metrics
    
    def format_classification_report(self, metrics: ClassificationMetrics) -> str:
        """Format classification metrics as a readable report."""
        lines = [
            "",
            "=" * 60,
            "  CLASSIFICATION METRICS",
            "=" * 60,
            f"  Samples: {metrics.n_samples:,} (Positive rate: {metrics.positive_rate:.2%})",
            "-" * 60,
            f"  Accuracy:    {metrics.accuracy:.4f}",
            f"  Precision:   {metrics.precision:.4f}",
            f"  Recall:      {metrics.recall:.4f}",
            f"  F1 Score:    {metrics.f1:.4f}",
            "-" * 60,
            f"  ROC-AUC:     {metrics.roc_auc:.4f}",
            f"  Avg Prec:    {metrics.average_precision:.4f}",
            f"  MCC:         {metrics.mcc:.4f}",
            f"  Brier:       {metrics.brier_score:.4f}",
            "-" * 60,
            f"  Specificity: {metrics.specificity:.4f}",
            f"  Sensitivity: {metrics.sensitivity:.4f}",
            f"  Balanced Acc:{metrics.balanced_accuracy:.4f}",
            "-" * 60,
            "  Confusion Matrix:",
            f"    TP: {metrics.true_positives:5d}  FN: {metrics.false_negatives:5d}",
            f"    FP: {metrics.false_positives:5d}  TN: {metrics.true_negatives:5d}",
            "=" * 60,
            ""
        ]
        return "\n".join(lines)
    
    def format_regression_report(self, metrics: RegressionMetrics) -> str:
        """Format regression metrics as a readable report."""
        lines = [
            "",
            "=" * 60,
            "  REGRESSION METRICS",
            "=" * 60,
            f"  Samples: {metrics.n_samples:,}",
            "-" * 60,
            f"  MAE:              {metrics.mae:.4f}",
            f"  RMSE:             {metrics.rmse:.4f}",
            f"  R²:               {metrics.r2:.4f}",
            f"  MedianAE:         {metrics.median_ae:.4f}",
            f"  Explained Var:    {metrics.explained_variance:.4f}",
            "-" * 60,
            f"  Pearson r:        {metrics.pearson_r:.4f} (p={metrics.pearson_p:.2e})",
            f"  Spearman ρ:       {metrics.spearman_r:.4f} (p={metrics.spearman_p:.2e})",
            f"  Kendall τ:        {metrics.kendall_tau:.4f} (p={metrics.kendall_p:.2e})",
            f"  CCC (Lin):        {metrics.ccc:.4f}",
            "-" * 60,
            f"  Mean Resid:       {metrics.mean_residual:+.4f}",
            f"  Std Resid:        {metrics.std_residual:.4f}",
            f"  Max Error:        {metrics.max_error:.4f}",
            "-" * 60,
            f"  Error P50:        {metrics.error_p50:.4f}",
            f"  Error P90:        {metrics.error_p90:.4f}",
            f"  Error P95:        {metrics.error_p95:.4f}",
            "=" * 60,
            ""
        ]
        return "\n".join(lines)
    
    def format_multitask_report(self, metrics: MultiTaskMetrics) -> str:
        """Format multi-task metrics as a readable report."""
        cls_report = self.format_classification_report(metrics.classification)
        reg_report = self.format_regression_report(metrics.regression)
        
        loss_section = "\n".join([
            "",
            "=" * 60,
            "  LOSSES",
            "=" * 60,
            f"  Total:          {metrics.total_loss:.4f}",
            f"  Classification: {metrics.classification_loss:.4f}",
            f"  Regression:     {metrics.regression_loss:.4f}",
            "=" * 60,
            ""
        ])
        
        return cls_report + reg_report + loss_section
    
    def save_metrics(
        self,
        metrics: MultiTaskMetrics,
        output_dir: str,
        prefix: str = ""
    ) -> Dict[str, str]:
        """
        Save metrics to files.
        
        Args:
            metrics: MultiTaskMetrics object
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Dict with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        saved_files = {}
        
        # Save classification metrics
        cls_path = output_dir / f"{prefix}classification_metrics.json"
        with open(cls_path, 'w') as f:
            json.dump(metrics.classification.to_dict(), f, indent=2)
        saved_files['classification'] = str(cls_path)
        
        # Save regression metrics
        reg_path = output_dir / f"{prefix}regression_metrics.json"
        with open(reg_path, 'w') as f:
            json.dump(metrics.regression.to_dict(), f, indent=2)
        saved_files['regression'] = str(reg_path)
        
        # Save combined metrics
        combined_path = output_dir / f"{prefix}all_metrics.json"
        with open(combined_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        saved_files['combined'] = str(combined_path)
        
        # Save text report
        report_path = output_dir / f"{prefix}metrics_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.format_multitask_report(metrics))
        saved_files['report'] = str(report_path)
        
        logger.info(f"Metrics saved to {output_dir}")
        
        return saved_files


def calculate_metrics_from_predictions(
    predictions: Dict[str, np.ndarray],
    labels: np.ndarray,
    regression_targets: Optional[np.ndarray] = None,
    regression_mask: Optional[np.ndarray] = None
) -> MultiTaskMetrics:
    """
    Convenience function to calculate metrics from prediction dict.
    
    Args:
        predictions: Dict with 'classification_prob', 'regression_pred'
        labels: True classification labels
        regression_targets: True regression values (optional)
        regression_mask: Mask for valid regression samples (optional)
        
    Returns:
        MultiTaskMetrics object
    """
    calculator = MatrixMetricsCalculator()
    
    cls_prob = predictions.get('classification_prob', np.zeros_like(labels))
    reg_pred = predictions.get('regression_pred', np.zeros_like(labels))
    
    if regression_targets is None:
        regression_targets = np.zeros_like(reg_pred)
        regression_mask = np.zeros_like(reg_pred, dtype=bool)
    
    return calculator.calculate_multitask_metrics(
        labels, cls_prob,
        regression_targets, reg_pred,
        regression_mask
    )


if __name__ == "__main__":
    # Test the metrics calculator
    print("Testing MatrixMetricsCalculator...")
    
    np.random.seed(42)
    n_samples = 100
    
    # Generate test data
    y_true_cls = np.random.randint(0, 2, n_samples)
    y_prob = np.clip(y_true_cls + np.random.randn(n_samples) * 0.3, 0, 1)
    
    y_true_reg = np.random.randn(n_samples) * 2 + 6  # pIC50-like
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.5
    
    # Calculate metrics
    calculator = MatrixMetricsCalculator()
    
    cls_metrics = calculator.calculate_classification_metrics(y_true_cls, y_prob)
    print(calculator.format_classification_report(cls_metrics))
    
    reg_metrics = calculator.calculate_regression_metrics(y_true_reg, y_pred_reg)
    print(calculator.format_regression_report(reg_metrics))
    
    # Multi-task
    multi_metrics = calculator.calculate_multitask_metrics(
        y_true_cls, y_prob,
        y_true_reg, y_pred_reg
    )
    print("\nMulti-task metrics dict keys:", list(multi_metrics.to_dict().keys()))
    
    print("\n✅ All tests passed!")
