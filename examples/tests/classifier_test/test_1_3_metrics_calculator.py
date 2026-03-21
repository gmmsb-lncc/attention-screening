"""
Level 1.3: MetricsCalculator Component Test

Tests the classification metrics calculation system.

Test Coverage:
- ClassificationMetrics dataclass
- MetricsCalculator basic calculations
- MetricsAggregator for cross-validation
- Threshold optimization
- Edge cases (all same class, perfect predictions)

Author: Test Suite
Created: 2025-11-08
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classifier.utils.metrics import (
    ClassificationMetrics,
    MetricsCalculator,
    MetricsAggregator,
    calculate_threshold_metrics,
    find_optimal_threshold
)


def test_classification_metrics_dataclass():
    """Test 1.1: ClassificationMetrics dataclass"""
    print("\n" + "="*60)
    print("Test 1.1: ClassificationMetrics Dataclass")
    print("="*60)
    
    try:
        # Create sample metrics
        metrics = ClassificationMetrics(
            loss=0.5,
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            roc_auc=0.90,
            average_precision=0.88,
            brier_score=0.15,
            matthews_corrcoef=0.70,
            true_positives=75,
            true_negatives=85,
            false_positives=10,
            false_negatives=30,
            specificity=0.89,
            sensitivity=0.75,
            fbeta_05=0.78,
            fbeta_2=0.76,
            sample_count=200,
            positive_rate=0.525
        )
        
        # Test attributes
        assert metrics.loss == 0.5, "Loss not set correctly"
        assert metrics.accuracy == 0.85, "Accuracy not set correctly"
        assert metrics.f1 == 0.77, "F1 not set correctly"
        assert metrics.sample_count == 200, "Sample count not set correctly"
        
        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict), "Should return dict"
        assert "Accuracy" in metrics_dict, "Should have Accuracy key"
        assert "F1" in metrics_dict, "Should have F1 key"
        assert metrics_dict["Accuracy"] == 0.85, "Accuracy value should match"
        
        print("✅ ClassificationMetrics dataclass working")
        print(f"   - Attributes: ✓")
        print(f"   - to_dict(): {len(metrics_dict)} metrics ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_calculator_basic():
    """Test 1.2: MetricsCalculator basic calculations"""
    print("\n" + "="*60)
    print("Test 1.2: MetricsCalculator Basic")
    print("="*60)
    
    try:
        calculator = MetricsCalculator()
        
        # Create synthetic data (balanced)
        n_samples = 100
        y_true = np.array([0]*50 + [1]*50, dtype=np.float32)
        y_prob = np.random.rand(n_samples).astype(np.float32)
        y_pred = (y_prob >= 0.5).astype(np.float32)
        
        # Calculate metrics
        metrics = calculator.calculate_metrics(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            loss=0.5
        )
        
        # Validate metrics
        assert isinstance(metrics, ClassificationMetrics), "Should return ClassificationMetrics"
        assert 0 <= metrics.accuracy <= 1, f"Accuracy out of range: {metrics.accuracy}"
        assert 0 <= metrics.precision <= 1, f"Precision out of range: {metrics.precision}"
        assert 0 <= metrics.recall <= 1, f"Recall out of range: {metrics.recall}"
        assert 0 <= metrics.f1 <= 1, f"F1 out of range: {metrics.f1}"
        assert 0 <= metrics.roc_auc <= 1, f"ROC AUC out of range: {metrics.roc_auc}"
        assert metrics.sample_count == n_samples, "Sample count should match"
        
        # Check confusion matrix sums correctly
        total = metrics.true_positives + metrics.true_negatives + \
                metrics.false_positives + metrics.false_negatives
        assert total == n_samples, f"Confusion matrix sum mismatch: {total} != {n_samples}"
        
        print("✅ MetricsCalculator basic calculations working")
        print(f"   - Accuracy: {metrics.accuracy:.3f} ✓")
        print(f"   - Precision: {metrics.precision:.3f} ✓")
        print(f"   - Recall: {metrics.recall:.3f} ✓")
        print(f"   - F1: {metrics.f1:.3f} ✓")
        print(f"   - ROC AUC: {metrics.roc_auc:.3f} ✓")
        print(f"   - Confusion matrix: ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_perfect_predictions():
    """Test 1.3: Perfect predictions edge case"""
    print("\n" + "="*60)
    print("Test 1.3: Perfect Predictions")
    print("="*60)
    
    try:
        calculator = MetricsCalculator()
        
        # Perfect predictions
        n_samples = 100
        y_true = np.array([0]*50 + [1]*50, dtype=np.float32)
        y_prob = y_true.copy()  # Perfect probabilities
        y_pred = y_true.copy()  # Perfect predictions
        
        metrics = calculator.calculate_metrics(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            loss=0.0
        )
        
        # Perfect metrics should be 1.0
        assert metrics.accuracy == 1.0, f"Perfect accuracy should be 1.0: {metrics.accuracy}"
        assert metrics.precision == 1.0, f"Perfect precision should be 1.0: {metrics.precision}"
        assert metrics.recall == 1.0, f"Perfect recall should be 1.0: {metrics.recall}"
        assert metrics.f1 == 1.0, f"Perfect F1 should be 1.0: {metrics.f1}"
        assert metrics.roc_auc == 1.0, f"Perfect ROC AUC should be 1.0: {metrics.roc_auc}"
        assert metrics.matthews_corrcoef == 1.0, f"Perfect MCC should be 1.0: {metrics.matthews_corrcoef}"
        
        # No errors
        assert metrics.false_positives == 0, "Perfect predictions should have no FP"
        assert metrics.false_negatives == 0, "Perfect predictions should have no FN"
        
        print("✅ Perfect predictions handled correctly")
        print(f"   - All metrics = 1.0 ✓")
        print(f"   - No FP/FN ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_all_same_class():
    """Test 1.4: All samples same class edge case"""
    print("\n" + "="*60)
    print("Test 1.4: All Same Class")
    print("="*60)
    
    try:
        calculator = MetricsCalculator()
        
        # All class 0
        n_samples = 100
        y_true = np.zeros(n_samples, dtype=np.float32)
        y_prob = np.random.rand(n_samples).astype(np.float32)
        y_pred = (y_prob >= 0.5).astype(np.float32)
        
        metrics = calculator.calculate_metrics(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            loss=0.5
        )
        
        # Should not crash
        assert isinstance(metrics, ClassificationMetrics), "Should return metrics"
        assert metrics.sample_count == n_samples, "Sample count should match"
        
        # ROC AUC should default to 0.5 for single class
        assert metrics.roc_auc == 0.5, f"Single class ROC AUC should be 0.5: {metrics.roc_auc}"
        
        print("✅ Single class handled correctly")
        print(f"   - No crash ✓")
        print(f"   - ROC AUC = 0.5 (default) ✓")
        print(f"   - Metrics calculated: {metrics.accuracy:.3f} accuracy ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_aggregator():
    """Test 1.5: MetricsAggregator for cross-validation"""
    print("\n" + "="*60)
    print("Test 1.5: MetricsAggregator")
    print("="*60)
    
    try:
        aggregator = MetricsAggregator()
        
        # Add 5 folds of metrics
        n_folds = 5
        for i in range(n_folds):
            metrics = ClassificationMetrics(
                loss=0.5 + i*0.01,
                accuracy=0.80 + i*0.02,
                precision=0.75 + i*0.02,
                recall=0.70 + i*0.02,
                f1=0.72 + i*0.02,
                roc_auc=0.85 + i*0.01,
                average_precision=0.83 + i*0.01,
                brier_score=0.15,
                matthews_corrcoef=0.65,
                true_positives=70,
                true_negatives=80,
                false_positives=10,
                false_negatives=15,
                specificity=0.88,
                sensitivity=0.82,
                fbeta_05=0.73,
                fbeta_2=0.71,
                sample_count=175,
                positive_rate=0.485
            )
            aggregator.add_fold_metrics(metrics)
        
        # Test summary statistics
        summary = aggregator.get_summary_statistics()
        assert isinstance(summary, dict), "Should return dict"
        assert 'accuracy' in summary, "Should have accuracy stats"
        assert 'mean' in summary['accuracy'], "Should have mean"
        assert 'std' in summary['accuracy'], "Should have std"
        
        # Check reasonable values
        accuracy_mean = summary['accuracy']['mean']
        assert 0 <= accuracy_mean <= 1, f"Accuracy mean out of range: {accuracy_mean}"
        print(f"✅ Summary stats: Accuracy mean = {accuracy_mean:.3f}")
        
        # Test best fold
        best_idx, best_metrics = aggregator.get_best_fold('roc_auc')
        assert 0 <= best_idx < n_folds, "Best idx should be in range"
        assert isinstance(best_metrics, ClassificationMetrics), "Should return metrics"
        print(f"✅ Best fold: #{best_idx} (ROC AUC = {best_metrics.roc_auc:.3f})")
        
        # Test confidence interval
        ci_lower, ci_upper = aggregator.get_confidence_interval('accuracy')
        assert ci_lower <= ci_upper, "CI lower should be <= upper"
        assert 0 <= ci_lower <= 1, "CI should be in valid range"
        assert 0 <= ci_upper <= 1, "CI should be in valid range"
        print(f"✅ Confidence interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        print("✅ MetricsAggregator working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_threshold_optimization():
    """Test 1.6: Threshold optimization"""
    print("\n" + "="*60)
    print("Test 1.6: Threshold Optimization")
    print("="*60)
    
    try:
        # Create data with clear separation
        n_samples = 200
        y_true = np.array([0]*100 + [1]*100, dtype=np.float32)
        # Class 0: low probabilities, Class 1: high probabilities
        y_prob = np.concatenate([
            np.random.uniform(0.0, 0.4, 100),
            np.random.uniform(0.6, 1.0, 100)
        ]).astype(np.float32)
        
        # Test threshold metrics calculation
        threshold_metrics = calculate_threshold_metrics(y_true, y_prob)
        assert 'thresholds' in threshold_metrics, "Should have thresholds"
        assert 'f1_scores' in threshold_metrics, "Should have F1 scores"
        assert len(threshold_metrics['thresholds']) > 0, "Should have thresholds"
        print(f"✅ Calculated metrics for {len(threshold_metrics['thresholds'])} thresholds")
        
        # Test optimal threshold finding
        opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_prob, metric='f1')
        assert 0 <= opt_threshold <= 1, f"Optimal threshold out of range: {opt_threshold}"
        assert 0 <= opt_f1 <= 1, f"Optimal F1 out of range: {opt_f1}"
        print(f"✅ Optimal threshold: {opt_threshold:.3f} (F1 = {opt_f1:.3f})")
        
        # Test different metrics
        opt_acc_threshold, opt_acc = find_optimal_threshold(y_true, y_prob, metric='accuracy')
        assert 0 <= opt_acc_threshold <= 1, "Optimal accuracy threshold out of range"
        print(f"✅ Optimal accuracy threshold: {opt_acc_threshold:.3f} (Acc = {opt_acc:.3f})")
        
        print("✅ Threshold optimization working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_imbalanced_data():
    """Test 1.7: Imbalanced data handling"""
    print("\n" + "="*60)
    print("Test 1.7: Imbalanced Data")
    print("="*60)
    
    try:
        calculator = MetricsCalculator()
        
        # Highly imbalanced (90:10)
        n_samples = 100
        y_true = np.array([0]*90 + [1]*10, dtype=np.float32)
        y_prob = np.random.rand(n_samples).astype(np.float32)
        y_pred = (y_prob >= 0.5).astype(np.float32)
        
        metrics = calculator.calculate_metrics(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            loss=0.5
        )
        
        # Should not crash
        assert isinstance(metrics, ClassificationMetrics), "Should return metrics"
        assert metrics.sample_count == n_samples, "Sample count should match"
        assert abs(metrics.positive_rate - 0.1) < 0.01, \
            f"Positive rate should be ~0.1: {metrics.positive_rate}"
        
        # All metrics should be in valid range
        assert 0 <= metrics.accuracy <= 1, "Accuracy out of range"
        assert 0 <= metrics.roc_auc <= 1, "ROC AUC out of range"
        
        print("✅ Imbalanced data handled correctly")
        print(f"   - Positive rate: {metrics.positive_rate:.3f} ✓")
        print(f"   - Accuracy: {metrics.accuracy:.3f} ✓")
        print(f"   - ROC AUC: {metrics.roc_auc:.3f} ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all MetricsCalculator tests"""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.3: METRICSCALCULATOR COMPONENT TEST")
    print("="*70)
    
    tests = [
        ("ClassificationMetrics Dataclass", test_classification_metrics_dataclass),
        ("MetricsCalculator Basic", test_metrics_calculator_basic),
        ("Perfect Predictions", test_perfect_predictions),
        ("All Same Class", test_all_same_class),
        ("MetricsAggregator", test_metrics_aggregator),
        ("Threshold Optimization", test_threshold_optimization),
        ("Imbalanced Data", test_imbalanced_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*70)
    
    if passed == total:
        print("🎉 MetricsCalculator: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  Some tests failed. Please investigate.")
        return 1


if __name__ == "__main__":
    exit(main())
