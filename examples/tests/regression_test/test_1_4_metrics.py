#!/usr/bin/env python3
"""
Test Level 1.4: MetricsCalculator - Regression Metrics
Duration: ~5s
Priority: HIGH - Ensures metrics calculation works correctly

Tests the MetricsCalculator class which computes various regression metrics
including MAE, MSE, RMSE, R2, MAPE, and others.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.utils.metrics import MetricsCalculator


def test_perfect_predictions():
    """Test 1: Perfect predictions should give R2=1, errors=0."""
    print("\n" + "="*60)
    print("TEST 1.4.1: Perfect Predictions")
    print("="*60)
    
    try:
        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        print(f"\n📊 Testing perfect predictions:")
        print(f"   y_true: {y_true}")
        print(f"   y_pred: {y_pred}")
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(y_true, y_pred, model_name='Perfect')
        
        print(f"\n✅ Metrics calculated:")
        print(f"   MAE: {metrics['MAE']:.6f}")
        print(f"   RMSE: {metrics['RMSE']:.6f}")
        print(f"   R2: {metrics['R2']:.6f}")
        
        # Verify perfect metrics
        assert metrics['MAE'] < 1e-10, f"MAE should be 0, got {metrics['MAE']}"
        assert metrics['RMSE'] < 1e-10, f"RMSE should be 0, got {metrics['RMSE']}"
        assert abs(metrics['R2'] - 1.0) < 1e-10, f"R2 should be 1.0, got {metrics['R2']}"
        assert metrics['MSE'] < 1e-10, f"MSE should be 0, got {metrics['MSE']}"
        
        print("\n✅ Perfect predictions: MAE=0, RMSE=0, R2=1.0")
        print("✅ TEST 1.4.1 PASSED: Perfect prediction metrics correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.4.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_metrics():
    """Test 2: Basic metrics calculation."""
    print("\n" + "="*60)
    print("TEST 1.4.2: Basic Metrics Calculation")
    print("="*60)
    
    try:
        # Simple known case
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5, 5.5])
        
        print(f"\n📊 Testing with known values:")
        print(f"   y_true: {y_true}")
        print(f"   y_pred: {y_pred}")
        print(f"   Errors: {y_pred - y_true}")
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(y_true, y_pred, model_name='TestModel')
        
        # Manual calculation for verification
        errors = y_pred - y_true
        mae_expected = np.mean(np.abs(errors))  # |0.5| + |0.5| + |-0.5| + |-0.5| + |0.5| = 2.5 / 5 = 0.5
        mse_expected = np.mean(errors ** 2)     # 0.25 + 0.25 + 0.25 + 0.25 + 0.25 = 1.25 / 5 = 0.25
        rmse_expected = np.sqrt(mse_expected)   # sqrt(0.25) = 0.5
        
        print(f"\n✅ Calculated metrics:")
        print(f"   MAE: {metrics['MAE']:.4f} (expected: {mae_expected:.4f})")
        print(f"   MSE: {metrics['MSE']:.4f} (expected: {mse_expected:.4f})")
        print(f"   RMSE: {metrics['RMSE']:.4f} (expected: {rmse_expected:.4f})")
        print(f"   R2: {metrics['R2']:.4f}")
        
        # Verify calculations
        assert abs(metrics['MAE'] - mae_expected) < 1e-6, f"MAE mismatch: {metrics['MAE']} vs {mae_expected}"
        assert abs(metrics['MSE'] - mse_expected) < 1e-6, f"MSE mismatch: {metrics['MSE']} vs {mse_expected}"
        assert abs(metrics['RMSE'] - rmse_expected) < 1e-6, f"RMSE mismatch: {metrics['RMSE']} vs {rmse_expected}"
        
        # R2 should be reasonable (between -inf and 1)
        assert metrics['R2'] <= 1.0, f"R2 should be <= 1.0, got {metrics['R2']}"
        
        print("\n✅ All basic metrics calculated correctly")
        print("✅ TEST 1.4.2 PASSED: Basic metrics working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.4.2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_dict_structure():
    """Test 3: Verify metrics dictionary structure."""
    print("\n" + "="*60)
    print("TEST 1.4.3: Metrics Dictionary Structure")
    print("="*60)
    
    try:
        # Generate random data
        np.random.seed(42)
        y_true = np.random.uniform(0, 10, 100)
        y_pred = y_true + np.random.normal(0, 1, 100)
        
        print(f"\n📊 Testing metrics structure:")
        print(f"   Samples: {len(y_true)}")
        
        # Calculate metrics
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(y_true, y_pred, model_name='StructureTest')
        
        # Check required keys
        required_keys = ['MAE', 'MSE', 'RMSE', 'R2', 'model_name', 'n_samples']
        
        print(f"\n✅ Checking required keys:")
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
            print(f"   ✓ {key}: {metrics[key]}")
        
        # Check additional useful keys (may not all be present)
        useful_keys = ['MedianAE', 'MAPE', 'ExplainedVariance', 'MaxError']
        print(f"\n📋 Additional metrics available:")
        for key in useful_keys:
            if key in metrics:
                print(f"   ✓ {key}: {metrics[key]}")
        
        # Verify data types
        assert isinstance(metrics['MAE'], (int, float)), f"MAE should be numeric, got {type(metrics['MAE'])}"
        assert isinstance(metrics['RMSE'], (int, float)), f"RMSE should be numeric"
        assert isinstance(metrics['R2'], (int, float)), f"R2 should be numeric"
        assert isinstance(metrics['model_name'], str), f"model_name should be string"
        assert isinstance(metrics['n_samples'], int), f"n_samples should be int"
        
        print(f"\n✅ Dictionary structure correct")
        print(f"   Total keys: {len(metrics)}")
        print("✅ TEST 1.4.3 PASSED: Metrics structure correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.4.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test 4: Edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 1.4.4: Edge Cases")
    print("="*60)
    
    try:
        calculator = MetricsCalculator()
        
        print("\n🔍 Testing edge cases...")
        
        # Test 1: Constant predictions (R2 should be 0 or negative)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Always predict mean
        
        metrics = calculator.calculate_all_metrics(y_true, y_pred, model_name='Constant')
        print(f"   ✓ Constant predictions: R2={metrics['R2']:.4f} (should be ~0)")
        assert metrics['R2'] <= 0.1, f"Constant predictions should have R2 near 0"
        
        # Test 2: Mismatched sizes should raise error
        try:
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.0, 2.0])
            metrics = calculator.calculate_all_metrics(y_true, y_pred)
            print("   ❌ ERROR: Mismatched sizes should have been rejected")
            return False
        except ValueError as e:
            print(f"   ✓ Mismatched sizes rejected: {str(e)[:50]}...")
        
        # Test 3: Single sample
        y_true = np.array([5.0])
        y_pred = np.array([4.5])
        metrics = calculator.calculate_all_metrics(y_true, y_pred, model_name='SingleSample')
        assert metrics['MAE'] == 0.5, f"Wrong MAE for single sample: {metrics['MAE']}"
        print(f"   ✓ Single sample: MAE={metrics['MAE']}")
        
        print("\n✅ All edge cases handled correctly")
        print("✅ TEST 1.4.4 PASSED: Edge cases working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.4.4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Level 1.4 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.4: METRICS CALCULATOR TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_perfect_predictions": test_perfect_predictions(),
        "test_basic_metrics": test_basic_metrics(),
        "test_metrics_dict_structure": test_metrics_dict_structure(),
        "test_edge_cases": test_edge_cases()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 1.4 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
