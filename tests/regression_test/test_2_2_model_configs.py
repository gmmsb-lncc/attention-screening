#!/usr/bin/env python3
"""
Test Level 2.2: Model Configurations - Hyperparameters
Duration: ~10s
Priority: MEDIUM - Ensures model hyperparameters are properly configured

Tests that each model has appropriate hyperparameters configured,
including random_state, n_jobs, max_iter, and algorithm-specific settings.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.models.models import RegressionModels


def test_random_forest_config():
    """Test 1: RandomForest hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2.2.1: RandomForest Configuration")
    print("="*60)
    
    try:
        model = RegressionModels.get_model('RandomForest', random_state=42)
        params = model.get_params()
        
        print("\n📋 RandomForest hyperparameters:")
        print(f"   n_estimators: {params['n_estimators']}")
        print(f"   max_depth: {params['max_depth']}")
        print(f"   min_samples_split: {params['min_samples_split']}")
        print(f"   min_samples_leaf: {params['min_samples_leaf']}")
        print(f"   random_state: {params['random_state']}")
        print(f"   n_jobs: {params['n_jobs']}")
        
        # Verify key parameters
        assert params['n_estimators'] > 0, "n_estimators should be > 0"
        assert params['max_depth'] is None or params['max_depth'] > 0, "max_depth invalid"
        assert params['random_state'] == 42, f"random_state should be 42, got {params['random_state']}"
        assert params['n_jobs'] == -1, f"n_jobs should be -1 (all cores)"
        
        print("\n✅ RandomForest properly configured")
        print("✅ TEST 2.2.1 PASSED: RF config correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.2.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_boosting_config():
    """Test 2: GradientBoosting hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2.2.2: GradientBoosting Configuration")
    print("="*60)
    
    try:
        model = RegressionModels.get_model('GradientBoosting', random_state=42)
        params = model.get_params()
        
        print("\n📋 GradientBoosting hyperparameters:")
        print(f"   n_estimators: {params['n_estimators']}")
        print(f"   max_depth: {params['max_depth']}")
        print(f"   learning_rate: {params['learning_rate']}")
        print(f"   subsample: {params['subsample']}")
        print(f"   random_state: {params['random_state']}")
        
        # Verify key parameters
        assert params['n_estimators'] > 0, "n_estimators should be > 0"
        assert 0 < params['learning_rate'] <= 1, "learning_rate should be in (0, 1]"
        assert 0 < params['subsample'] <= 1, "subsample should be in (0, 1]"
        assert params['random_state'] == 42, f"random_state should be 42"
        
        print("\n✅ GradientBoosting properly configured")
        print("✅ TEST 2.2.2 PASSED: GB config correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.2.2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_linear_models_config():
    """Test 3: Linear models (Ridge, Lasso, ElasticNet) hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2.2.3: Linear Models Configuration")
    print("="*60)
    
    try:
        print("\n📋 Testing linear models configurations...")
        
        # Ridge
        ridge = RegressionModels.get_model('Ridge', random_state=42)
        ridge_params = ridge.get_params()
        print(f"\n   Ridge:")
        print(f"      alpha: {ridge_params['alpha']}")
        print(f"      random_state: {ridge_params['random_state']}")
        assert ridge_params['alpha'] > 0, "Ridge alpha should be > 0"
        
        # Lasso
        lasso = RegressionModels.get_model('Lasso', random_state=42)
        lasso_params = lasso.get_params()
        print(f"\n   Lasso:")
        print(f"      alpha: {lasso_params['alpha']}")
        print(f"      max_iter: {lasso_params['max_iter']}")
        print(f"      random_state: {lasso_params['random_state']}")
        assert lasso_params['alpha'] > 0, "Lasso alpha should be > 0"
        assert lasso_params['max_iter'] >= 1000, "Lasso max_iter should be >= 1000 for convergence"
        
        # ElasticNet
        elastic = RegressionModels.get_model('ElasticNet', random_state=42)
        elastic_params = elastic.get_params()
        print(f"\n   ElasticNet:")
        print(f"      alpha: {elastic_params['alpha']}")
        print(f"      l1_ratio: {elastic_params['l1_ratio']}")
        print(f"      max_iter: {elastic_params['max_iter']}")
        print(f"      random_state: {elastic_params['random_state']}")
        assert elastic_params['alpha'] > 0, "ElasticNet alpha should be > 0"
        assert 0 <= elastic_params['l1_ratio'] <= 1, "l1_ratio should be in [0, 1]"
        
        print("\n✅ All linear models properly configured")
        print("✅ TEST 2.2.3 PASSED: Linear models config correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.2.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_svr_config():
    """Test 4: SVR hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2.2.4: SVR Configuration")
    print("="*60)
    
    try:
        model = RegressionModels.get_model('SVR', random_state=42)
        params = model.get_params()
        
        print("\n📋 SVR hyperparameters:")
        print(f"   kernel: {params['kernel']}")
        print(f"   C: {params['C']}")
        print(f"   epsilon: {params['epsilon']}")
        print(f"   cache_size: {params['cache_size']}")
        
        # Verify key parameters
        assert params['kernel'] in ['linear', 'poly', 'rbf', 'sigmoid'], f"Invalid kernel: {params['kernel']}"
        assert params['C'] > 0, "C should be > 0"
        assert params['epsilon'] >= 0, "epsilon should be >= 0"
        
        print("\n✅ SVR properly configured")
        print("✅ TEST 2.2.4 PASSED: SVR config correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.2.4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_knn_config():
    """Test 5: KNN hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2.2.5: KNN Configuration")
    print("="*60)
    
    try:
        model = RegressionModels.get_model('KNN', random_state=42)
        params = model.get_params()
        
        print("\n📋 KNN hyperparameters:")
        print(f"   n_neighbors: {params['n_neighbors']}")
        print(f"   weights: {params['weights']}")
        print(f"   n_jobs: {params['n_jobs']}")
        
        # Verify key parameters
        assert params['n_neighbors'] > 0, "n_neighbors should be > 0"
        assert params['weights'] in ['uniform', 'distance'], f"Invalid weights: {params['weights']}"
        assert params['n_jobs'] == -1, "n_jobs should be -1 for parallel processing"
        
        print("\n✅ KNN properly configured")
        print("✅ TEST 2.2.5 PASSED: KNN config correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.2.5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mlp_config():
    """Test 6: MLP (Neural Network) hyperparameters."""
    print("\n" + "="*60)
    print("TEST 2.2.6: MLP Configuration")
    print("="*60)
    
    try:
        model = RegressionModels.get_model('MLP', random_state=42)
        params = model.get_params()
        
        print("\n📋 MLP hyperparameters:")
        print(f"   hidden_layer_sizes: {params['hidden_layer_sizes']}")
        print(f"   activation: {params['activation']}")
        print(f"   solver: {params['solver']}")
        print(f"   max_iter: {params['max_iter']}")
        print(f"   early_stopping: {params['early_stopping']}")
        print(f"   random_state: {params['random_state']}")
        
        # Verify key parameters
        assert isinstance(params['hidden_layer_sizes'], tuple), "hidden_layer_sizes should be tuple"
        assert len(params['hidden_layer_sizes']) > 0, "Should have at least 1 hidden layer"
        assert params['activation'] in ['identity', 'logistic', 'tanh', 'relu'], f"Invalid activation"
        assert params['solver'] in ['lbfgs', 'sgd', 'adam'], f"Invalid solver"
        assert params['max_iter'] > 0, "max_iter should be > 0"
        assert params['random_state'] == 42, "random_state should be 42"
        
        print("\n✅ MLP properly configured")
        print("✅ TEST 2.2.6 PASSED: MLP config correct")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.2.6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Level 2.2 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 2.2: MODEL CONFIGURATIONS TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_random_forest_config": test_random_forest_config(),
        "test_gradient_boosting_config": test_gradient_boosting_config(),
        "test_linear_models_config": test_linear_models_config(),
        "test_svr_config": test_svr_config(),
        "test_knn_config": test_knn_config(),
        "test_mlp_config": test_mlp_config()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 2.2 TEST SUMMARY")
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
