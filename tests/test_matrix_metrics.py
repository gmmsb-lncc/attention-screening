"""
Tests for matrix_metrics module.

This module tests the comprehensive metrics system for the matrix-based pipeline,
ensuring compatibility with the vector-based pipeline metrics.

Author: DockTKinase Team
Date: November 2025
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from src.classifier.utils.matrix_metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    MultiTaskMetrics,
    MatrixMetricsCalculator
)


class TestClassificationMetrics:
    """Tests for ClassificationMetrics dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        metrics = ClassificationMetrics()
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0
        assert metrics.roc_auc == 0.0
        assert metrics.n_samples == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ClassificationMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1=0.90,
            roc_auc=0.97,
            n_samples=100
        )
        d = metrics.to_dict()
        
        assert d['Accuracy'] == 0.95
        assert d['Precision'] == 0.92
        assert d['Recall'] == 0.88
        assert d['F1'] == 0.90
        assert d['ROC_AUC'] == 0.97
        assert d['n_samples'] == 100
    
    def test_dict_compatible_keys(self):
        """Test that dict keys are compatible with vector pipeline format."""
        metrics = ClassificationMetrics()
        d = metrics.to_dict()
        
        # These keys should match vector pipeline format
        expected_keys = {
            'Accuracy', 'Precision', 'Recall', 'F1',
            'ROC_AUC', 'Average_Precision', 'Brier_Score', 'MCC',
            'Fbeta_0.5', 'Fbeta_2', 'Specificity', 'Sensitivity'
        }
        
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"


class TestRegressionMetrics:
    """Tests for RegressionMetrics dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        metrics = RegressionMetrics()
        assert metrics.mae == 0.0
        assert metrics.mse == 0.0
        assert metrics.rmse == 0.0
        assert metrics.r2 == 0.0
        assert metrics.n_samples == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RegressionMetrics(
            mae=0.5,
            mse=0.3,
            rmse=0.55,
            r2=0.85,
            pearson_r=0.92,
            spearman_r=0.90,
            n_samples=100
        )
        d = metrics.to_dict()
        
        assert d['MAE'] == 0.5
        assert d['MSE'] == 0.3
        assert d['RMSE'] == 0.55
        assert d['R2'] == 0.85
        assert d['Pearson_R'] == 0.92
        assert d['Spearman_R'] == 0.90
    
    def test_dict_compatible_keys(self):
        """Test that dict keys are compatible with vector pipeline format."""
        metrics = RegressionMetrics()
        d = metrics.to_dict()
        
        # These keys should match vector pipeline format
        expected_keys = {
            'MAE', 'MSE', 'RMSE', 'R2',
            'MedianAE', 'ExplainedVariance', 'MaxError',
            'Pearson_R', 'Spearman_R'
        }
        
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"


class TestMultiTaskMetrics:
    """Tests for MultiTaskMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        metrics = MultiTaskMetrics()
        assert isinstance(metrics.classification, ClassificationMetrics)
        assert isinstance(metrics.regression, RegressionMetrics)
        assert metrics.total_loss == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = MultiTaskMetrics()
        metrics.classification.accuracy = 0.95
        metrics.regression.rmse = 0.5
        metrics.total_loss = 0.3
        
        d = metrics.to_dict()
        
        assert 'classification' in d
        assert 'regression' in d
        assert 'losses' in d
        assert d['classification']['Accuracy'] == 0.95
        assert d['regression']['RMSE'] == 0.5
        assert d['losses']['total'] == 0.3


class TestMatrixMetricsCalculator:
    """Tests for MatrixMetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a metrics calculator."""
        return MatrixMetricsCalculator(threshold=0.5)
    
    @pytest.fixture
    def classification_data(self):
        """Generate test classification data."""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randint(0, 2, n_samples)
        # Generate probabilities correlated with true labels
        y_prob = np.clip(y_true + np.random.randn(n_samples) * 0.3, 0, 1)
        
        return y_true, y_prob
    
    @pytest.fixture
    def regression_data(self):
        """Generate test regression data."""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randn(n_samples) * 2 + 5
        y_pred = y_true + np.random.randn(n_samples) * 0.5
        
        return y_true, y_pred
    
    def test_calculate_classification_metrics(self, calculator, classification_data):
        """Test classification metrics calculation."""
        y_true, y_prob = classification_data
        
        metrics = calculator.calculate_classification_metrics(y_true, y_prob)
        
        assert isinstance(metrics, ClassificationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.roc_auc <= 1
        assert metrics.n_samples == len(y_true)
    
    def test_calculate_classification_metrics_values(self, calculator, classification_data):
        """Test that classification metrics have reasonable values."""
        y_true, y_prob = classification_data
        
        metrics = calculator.calculate_classification_metrics(y_true, y_prob)
        
        # With correlated predictions, accuracy should be high
        assert metrics.accuracy > 0.7
        assert metrics.roc_auc > 0.8
        assert metrics.f1 > 0.7
    
    def test_calculate_regression_metrics(self, calculator, regression_data):
        """Test regression metrics calculation."""
        y_true, y_pred = regression_data
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae >= 0
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert metrics.n_samples == len(y_true)
    
    def test_calculate_regression_metrics_values(self, calculator, regression_data):
        """Test that regression metrics have reasonable values."""
        y_true, y_pred = regression_data
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        # With correlated predictions, R² should be high
        assert metrics.r2 > 0.8
        assert metrics.pearson_r > 0.9
        assert metrics.spearman_r > 0.8
    
    def test_calculate_multitask_metrics(self, calculator, classification_data, regression_data):
        """Test multi-task metrics calculation."""
        cls_true, cls_prob = classification_data
        reg_true, reg_pred = regression_data
        
        metrics = calculator.calculate_multitask_metrics(
            cls_true=cls_true,
            cls_prob=cls_prob,
            reg_true=reg_true,
            reg_pred=reg_pred
        )
        
        assert isinstance(metrics, MultiTaskMetrics)
        assert isinstance(metrics.classification, ClassificationMetrics)
        assert isinstance(metrics.regression, RegressionMetrics)
    
    def test_format_classification_report(self, calculator, classification_data):
        """Test classification report formatting."""
        y_true, y_prob = classification_data
        metrics = calculator.calculate_classification_metrics(y_true, y_prob)
        
        report = calculator.format_classification_report(metrics)
        
        assert isinstance(report, str)
        assert 'CLASSIFICATION METRICS' in report
        assert 'Accuracy' in report
        assert 'Precision' in report
        assert 'Recall' in report
        assert 'F1 Score' in report
    
    def test_format_regression_report(self, calculator, regression_data):
        """Test regression report formatting."""
        y_true, y_pred = regression_data
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        report = calculator.format_regression_report(metrics)
        
        assert isinstance(report, str)
        assert 'REGRESSION METRICS' in report
        assert 'MAE' in report
        assert 'RMSE' in report
        assert 'R²' in report or 'R2' in report
    
    def test_confusion_matrix_values(self, calculator):
        """Test confusion matrix values."""
        # Create predictable data
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.4, 0.6])
        
        metrics = calculator.calculate_classification_metrics(y_true, y_prob)
        
        # Check confusion matrix components sum to total samples
        cm_sum = (metrics.true_positives + metrics.true_negatives + 
                  metrics.false_positives + metrics.false_negatives)
        assert cm_sum == len(y_true)
    
    def test_threshold_effect(self, calculator, classification_data):
        """Test that threshold affects metrics."""
        y_true, y_prob = classification_data
        
        metrics_05 = calculator.calculate_classification_metrics(y_true, y_prob, threshold=0.5)
        metrics_03 = calculator.calculate_classification_metrics(y_true, y_prob, threshold=0.3)
        metrics_07 = calculator.calculate_classification_metrics(y_true, y_prob, threshold=0.7)
        
        # Lower threshold should increase recall (more positives predicted)
        assert metrics_03.recall >= metrics_05.recall >= metrics_07.recall
        
        # Higher threshold should increase precision (fewer false positives)
        # Note: This may not always hold depending on data distribution
    
    def test_empty_arrays(self, calculator):
        """Test handling of empty arrays."""
        y_true = np.array([])
        y_prob = np.array([])
        
        metrics = calculator.calculate_classification_metrics(y_true, y_prob)
        
        assert metrics.n_samples == 0
        assert metrics.accuracy == 0.0
    
    def test_single_class(self, calculator):
        """Test handling of single class data."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.8, 0.9, 0.7, 0.95, 0.85])
        
        metrics = calculator.calculate_classification_metrics(y_true, y_prob)
        
        # Should handle single class without errors
        assert metrics.n_samples == 5
    
    def test_regression_with_mask(self, calculator, regression_data):
        """Test regression with mask."""
        y_true, y_pred = regression_data
        mask = np.array([True, True, False, True, False] * 20)  # 100 samples
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred, mask=mask)
        
        # Should only use masked samples
        expected_samples = mask.sum()
        assert metrics.n_samples == expected_samples
    
    def test_residual_statistics(self, calculator, regression_data):
        """Test residual statistics calculation."""
        y_true, y_pred = regression_data
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        # Residual mean should be close to 0 for unbiased predictions
        # (our synthetic data is unbiased)
        assert abs(metrics.mean_residual) < 0.5
        
        # Residual std should be positive
        assert metrics.std_residual > 0
    
    def test_error_percentiles(self, calculator, regression_data):
        """Test error percentiles."""
        y_true, y_pred = regression_data
        
        metrics = calculator.calculate_regression_metrics(y_true, y_pred)
        
        # Percentiles should be ordered: p50 <= p90 <= p95
        assert metrics.error_p50 <= metrics.error_p90 <= metrics.error_p95


class TestMetricsSaving:
    """Tests for metrics saving functionality."""
    
    @pytest.fixture
    def calculator(self):
        return MatrixMetricsCalculator(threshold=0.5)
    
    @pytest.fixture
    def sample_metrics(self, calculator):
        """Generate sample multi-task metrics."""
        np.random.seed(42)
        n = 100
        
        cls_true = np.random.randint(0, 2, n)
        cls_prob = np.clip(cls_true + np.random.randn(n) * 0.3, 0, 1)
        reg_true = np.random.randn(n) * 2 + 5
        reg_pred = reg_true + np.random.randn(n) * 0.5
        
        return calculator.calculate_multitask_metrics(
            cls_true=cls_true,
            cls_prob=cls_prob,
            reg_true=reg_true,
            reg_pred=reg_pred,
            losses={'total': 0.3, 'classification': 0.2, 'regression': 0.1}
        )
    
    def test_save_metrics(self, calculator, sample_metrics):
        """Test saving metrics to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = calculator.save_metrics(sample_metrics, tmpdir, prefix='test')
            
            # Check files were created
            assert Path(saved_files['classification']).exists()
            assert Path(saved_files['regression']).exists()
            assert Path(saved_files['combined']).exists()
            
            # Check JSON content
            with open(saved_files['combined']) as f:
                data = json.load(f)
            
            assert 'classification' in data
            assert 'regression' in data
            assert 'losses' in data


class TestMetricsCompatibility:
    """Tests for compatibility with vector pipeline metrics."""
    
    def test_classification_keys_match_vector_pipeline(self):
        """Test that classification metric keys match vector pipeline format."""
        metrics = ClassificationMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1=0.90,
            roc_auc=0.97,
            average_precision=0.96,
            brier_score=0.05,
            mcc=0.89,
            fbeta_05=0.91,
            fbeta_2=0.89,
            specificity=0.93,
            sensitivity=0.88
        )
        
        d = metrics.to_dict()
        
        # Vector pipeline uses these exact key names
        vector_pipeline_keys = [
            'Accuracy', 'Precision', 'Recall', 'F1',
            'ROC_AUC', 'Average_Precision', 'Brier_Score', 'MCC',
            'Fbeta_0.5', 'Fbeta_2', 'Specificity', 'Sensitivity'
        ]
        
        for key in vector_pipeline_keys:
            assert key in d, f"Missing key: {key}"
    
    def test_regression_keys_match_vector_pipeline(self):
        """Test that regression metric keys match vector pipeline format."""
        metrics = RegressionMetrics(
            mae=0.5,
            mse=0.3,
            rmse=0.55,
            r2=0.85,
            median_ae=0.45,
            explained_variance=0.84,
            max_error=1.5,
            mape=10.5,
            pearson_r=0.92,
            spearman_r=0.90
        )
        
        d = metrics.to_dict()
        
        # Vector pipeline uses these exact key names
        vector_pipeline_keys = [
            'MAE', 'MSE', 'RMSE', 'R2',
            'MedianAE', 'ExplainedVariance', 'MaxError', 'MAPE',
            'Pearson_R', 'Spearman_R'
        ]
        
        for key in vector_pipeline_keys:
            assert key in d, f"Missing key: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
