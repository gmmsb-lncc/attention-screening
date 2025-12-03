#!/usr/bin/env python3
"""
Sklearn Multi-Model Trainer - Classification
=============================================

Manages training of multiple sklearn classification models.
Equivalent to regression trainer, but for binary classification.
"""

import time
import warnings
import numpy as np
from typing import Dict, Any, Optional, List

# Import scipy's LinAlgWarning for proper filtering
try:
    from scipy.linalg import LinAlgWarning
except ImportError:
    LinAlgWarning = UserWarning

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss
)

# Suppress harmless sklearn/LGBM warnings
# Feature names warning (happens when training with DataFrame but predicting with numpy array)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='.*was fitted with feature names.*')
# AdaBoost deprecated 'algorithm' parameter
warnings.filterwarnings('ignore', message=".*parameter 'algorithm' is deprecated.*")
# Convergence warnings (LinearSVC may not converge with default iterations)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Liblinear failed to converge.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
# Suppress scipy ConstantInputWarning  
warnings.filterwarnings('ignore', message='An input array is constant')


class ClassificationMetricsCalculator:
    """Calculates all classification metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None,
                              model_name: str = '') -> Dict[str, Any]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predictions (0/1)
            y_pred_proba: Probabilities (for ROC-AUC, etc)
            model_name: Model name (for logging)
            
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['True_Negatives'] = int(tn)
        metrics['False_Positives'] = int(fp)
        metrics['False_Negatives'] = int(fn)
        metrics['True_Positives'] = int(tp)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['Specificity'] = specificity
        
        # F-beta scores
        metrics['Fbeta_0.5'] = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        metrics['Fbeta_2'] = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
        
        # Matthews Correlation Coefficient
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                # Check for NaN (happens when only one class present)
                metrics['ROC_AUC'] = float(roc_auc) if not np.isnan(roc_auc) else 0.0
                
                avg_prec = average_precision_score(y_true, y_pred_proba)
                metrics['Average_Precision'] = float(avg_prec) if not np.isnan(avg_prec) else 0.0
                
                brier = brier_score_loss(y_true, y_pred_proba)
                metrics['Brier_Score'] = float(brier) if not np.isnan(brier) else 0.0
            except Exception as e:
                # If there's only one class, some metrics may fail
                metrics['ROC_AUC'] = 0.0
                metrics['Average_Precision'] = 0.0
                metrics['Brier_Score'] = 0.0
        else:
            metrics['ROC_AUC'] = 0.0
            metrics['Average_Precision'] = 0.0
            metrics['Brier_Score'] = 0.0
        
        return metrics


class SklearnClassificationTrainer:
    """
    Multi-model sklearn classification trainer.
    Equivalent to RegressionTrainer but for classification.
    """
    
    def __init__(self, 
                 models_dict: Dict[str, Any],
                 verbose: bool = True,
                 random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            models_dict: Dictionary {name: model}
            verbose: Show progress
            random_state: Seed for reproducibility
        """
        self.models_dict = models_dict
        self.verbose = verbose
        self.random_state = random_state
        
        self.trained_models = {}
        self.train_results = {}
        self.val_results = {}
        
        self.metrics_calculator = ClassificationMetricsCalculator()
    
    def train_single_model(self, 
                          model_name: str,
                          model: Any,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_name: Model name
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict with validation metrics (or training if val=None)
        """
        start_time = time.time()
        
        if self.verbose:
            print(f'\n   🔧 Training {model_name}...')
        
        # Train with suppressed harmless warnings (scipy ill-conditioned matrix)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=LinAlgWarning)
            warnings.filterwarnings('ignore', message='.*Ill-conditioned matrix.*')
            model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate on training
        y_train_pred = model.predict(X_train)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
        else:
            y_train_proba = None
        
        train_metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_train_pred, y_train_proba, model_name
        )
        train_metrics['training_time'] = train_time
        
        # Evaluate on validation if available
        val_metrics = None
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_val_proba = None
            
            val_metrics = self.metrics_calculator.calculate_all_metrics(
                y_val, y_val_pred, y_val_proba, model_name
            )
        
        # Store
        self.trained_models[model_name] = model
        self.train_results[model_name] = train_metrics
        if val_metrics:
            self.val_results[model_name] = val_metrics
        
        if self.verbose:
            print(f'      ✅ Train: Acc={train_metrics["Accuracy"]:.4f} | '
                  f'F1={train_metrics["F1"]:.4f} | '
                  f'ROC-AUC={train_metrics["ROC_AUC"]:.4f} | '
                  f'Time={train_time:.2f}s')
            
            if val_metrics:
                print(f'      ✅ Valid: Acc={val_metrics["Accuracy"]:.4f} | '
                      f'F1={val_metrics["F1"]:.4f} | '
                      f'ROC-AUC={val_metrics["ROC_AUC"]:.4f}')
        
        return val_metrics if val_metrics else train_metrics
    
    def train_all(self,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict with validation metrics for all models
        """
        if self.verbose:
            print(f'\n   📊 Training {len(self.models_dict)} models...')
        
        for model_name, model in self.models_dict.items():
            try:
                self.train_single_model(
                    model_name, model,
                    X_train, y_train,
                    X_val, y_val
                )
            except Exception as e:
                if self.verbose:
                    print(f'      ❌ Error training {model_name}: {str(e)}')
                continue
        
        return self.val_results if self.val_results else self.train_results
    
    def get_best_model(self, metric: str = 'ROC_AUC') -> tuple:
        """
        Return the best model based on a metric.
        
        Args:
            metric: Metric for comparison (default: ROC_AUC)
            
        Returns:
            Tuple (model_name, model, metric_value)
        """
        results = self.val_results if self.val_results else self.train_results
        
        if not results:
            return None, None, None
        
        # Find best
        best_name = max(results.items(), key=lambda x: x[1][metric])[0]
        best_model = self.trained_models[best_name]
        best_value = results[best_name][metric]
        
        return best_name, best_model, best_value
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Return summary of results from all models.
        
        Returns:
            Dict with organized summary
        """
        results = self.val_results if self.val_results else self.train_results
        
        summary = {
            'n_models': len(results),
            'models': {},
            'rankings': {}
        }
        
        # Add results for each model
        for name, metrics in results.items():
            summary['models'][name] = metrics
        
        # Rankings por métrica
        for metric in ['ROC_AUC', 'F1', 'Accuracy', 'Precision', 'Recall']:
            if metric in list(results.values())[0]:
                sorted_models = sorted(
                    results.items(),
                    key=lambda x: x[1][metric],
                    reverse=True
                )
                summary['rankings'][metric] = [
                    {'model': name, 'value': metrics[metric]}
                    for name, metrics in sorted_models
                ]
        
        return summary
    
    def print_summary(self, top_n: int = None, use_test: bool = True):
        """
        Print summary of results (deprecated - use print_results_summary in pipeline).
        
        Args:
            top_n: Number of top models to show (None = all)
            use_test: If True, use test results (default). If False, use validation.
        """
        # By default, use test results if available
        if use_test and self.test_results:
            results = self.test_results
            result_type = "Test"
        elif self.val_results:
            results = self.val_results
            result_type = "Validation"
        else:
            results = self.train_results
            result_type = "Training"
        
        if not results:
            print('⚠️  No models trained')
            return
        
        print('\n' + '=' * 80)
        print(f'📊 RESULTS SUMMARY ({result_type} Set)')
        print('=' * 80)
        
        # Ordenar por ROC-AUC
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['ROC_AUC'],
            reverse=True
        )
        
        # Limitar se necessário
        if top_n:
            sorted_results = sorted_results[:top_n]
        
        # Header
        header = f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC-AUC':>10} {'Time':>8}"
        print(header)
        print('-' * 80)
        
        # Models
        for i, (model_name, metrics) in enumerate(sorted_results):
            train_time = self.train_results[model_name].get('training_time', 0)
            row = (
                f"{model_name:<20} "
                f"{metrics['Accuracy']:>8.4f} "
                f"{metrics['Precision']:>8.4f} "
                f"{metrics['Recall']:>8.4f} "
                f"{metrics['F1']:>8.4f} "
                f"{metrics['ROC_AUC']:>10.4f} "
                f"{train_time:>7.2f}s"
            )
            
            # Destacar o melhor
            if i == 0:
                print(f'🏆 {row}')
            else:
                print(f'   {row}')
        
        print('=' * 80)
        
        # Best model
        best_name = sorted_results[0][0]
        best_metrics = sorted_results[0][1]
        
        print(f'\n🏆 BEST MODEL: {best_name}')
        print(f'   ROC-AUC: {best_metrics["ROC_AUC"]:.4f}')
        print(f'   F1-Score: {best_metrics["F1"]:.4f}')
        print(f'   Accuracy: {best_metrics["Accuracy"]:.4f}')
        print(f'   Precision: {best_metrics["Precision"]:.4f}')
        print(f'   Recall: {best_metrics["Recall"]:.4f}')
        print()


if __name__ == '__main__':
    print('SklearnClassificationTrainer - Sklearn Multi-Model Trainer')
    print('=' * 70)
    print('\nTraining module for multiple sklearn classifiers.')
    print('Use in conjunction with ClassificationModels.')
