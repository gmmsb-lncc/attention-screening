"""
Evaluator for Attention Matrix Models.

Single Responsibility: Metrics computation only.
"""

import warnings
import numpy as np
from typing import Dict, Any
import logging

# Suppress scipy ConstantInputWarning for correlation calculations
warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from scipy.stats import pearsonr, spearmanr, kendalltau


logger = logging.getLogger(__name__)


class AttentionEvaluator:
    """
    Evaluator for Cross-Attention models.
    
    Computes comprehensive metrics for both:
    - Classification (activity prediction)
    - Regression (affinity prediction)
    """
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted logits or probabilities
            threshold: Threshold for classification
            
        Returns:
            Dictionary with classification metrics
        """
        # Apply sigmoid if values are logits (outside 0-1 range)
        if np.any(y_prob < 0) or np.any(y_prob > 1):
            y_prob = 1 / (1 + np.exp(-y_prob))
        
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        # ROC-AUC (requires both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics['roc_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: True affinity values (pChEMBL)
            y_pred: Predicted affinity values
            
        Returns:
            Dictionary with regression metrics
        """
        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred))
        }
        
        # Correlations
        if len(y_true) > 2:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
            spearman_r, spearman_p = spearmanr(y_true, y_pred)
            kendall_tau, kendall_p = kendalltau(y_true, y_pred)
            
            metrics['pearson'] = float(pearson_r)
            metrics['pearson_p'] = float(pearson_p)
            metrics['spearman'] = float(spearman_r)
            metrics['spearman_p'] = float(spearman_p)
            metrics['kendall_tau'] = float(kendall_tau)
            metrics['kendall_p'] = float(kendall_p)
            
            # Concordance Correlation Coefficient (Lin's CCC)
            mean_true = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            var_true = np.var(y_true)
            var_pred = np.var(y_pred)
            covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
            ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
            metrics['ccc'] = float(ccc)
        
        # Error distribution
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        metrics['mean_error'] = float(np.mean(errors))  # bias
        metrics['std_error'] = float(np.std(errors))
        metrics['median_abs_error'] = float(np.median(abs_errors))
        metrics['p90_abs_error'] = float(np.percentile(abs_errors, 90))
        
        # IC50 interpretation
        mae_factor = 10 ** metrics['mae']
        metrics['ic50_fold_error'] = float(mae_factor)
        
        return metrics
    
    def evaluate(
        self,
        cls_preds: np.ndarray,
        cls_labels: np.ndarray,
        reg_preds: np.ndarray,
        reg_labels: np.ndarray,
        classification_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Full evaluation with predictions and labels.
        
        Args:
            cls_preds: Classification predictions (logits or probabilities)
            cls_labels: True classification labels (0/1)
            reg_preds: Regression predictions
            reg_labels: True regression values
            classification_threshold: Threshold for classification
            
        Returns:
            Dictionary with all metrics
        """
        # Classification metrics
        classification = self.compute_classification_metrics(
            cls_labels, cls_preds, classification_threshold
        )
        
        # Regression metrics
        regression = self.compute_regression_metrics(
            reg_labels, reg_preds
        )
        
        return {
            'classification': classification,
            'regression': regression,
            'n_samples': len(cls_labels)
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print formatted summary of results."""
        print('=' * 70)
        print('EVALUATION RESULTS')
        print('=' * 70)
        
        print(f'\nSamples evaluated: {results["n_samples"]}')
        
        print(f'\nCLASSIFICATION (Activity):')
        cls = results['classification']
        print(f'  Accuracy:  {cls["accuracy"]:.4f} ({100*cls["accuracy"]:.2f}%)')
        print(f'  Precision: {cls["precision"]:.4f} ({100*cls["precision"]:.2f}%)')
        print(f'  Recall:    {cls["recall"]:.4f} ({100*cls["recall"]:.2f}%)')
        print(f'  F1-Score:  {cls["f1_score"]:.4f} ({100*cls["f1_score"]:.2f}%)')
        print(f'  ROC-AUC:   {cls["roc_auc"]:.4f} ({100*cls["roc_auc"]:.2f}%)')
        
        print(f'\nREGRESSION (pChEMBL):')
        reg = results['regression']
        print(f'  MAE:         {reg["mae"]:.4f}')
        print(f'  RMSE:        {reg["rmse"]:.4f}')
        print(f'  R²:          {reg["r2"]:.4f}')
        print(f'  Pearson r:   {reg.get("pearson", 0):.4f}')
        print(f'  Spearman ρ:  {reg.get("spearman", 0):.4f}')
        print(f'  Kendall τ:   {reg.get("kendall_tau", 0):.4f}')
        print(f'  CCC (Lin):   {reg.get("ccc", 0):.4f}')
        
        print('=' * 70)
