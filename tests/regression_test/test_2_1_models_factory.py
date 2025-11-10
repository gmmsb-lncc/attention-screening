#!/usr/bin/env python3
"""
Test Level 2.1: RegressionModels Factory - Model Creation
Duration: ~10s
Priority: HIGH - Ensures all models can be instantiated

Tests the RegressionModels factory class which creates and configures
all available regression algorithms (RandomForest, GradientBoosting, Ridge,
Lasso, ElasticNet, SVR, KNN, MLP, XGBoost, LightGBM, CatBoost).
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.models.models import RegressionModels


def test_get_all_models():
    """Test 1: Get all available models."""
    print("\n" + "="*60)
    print("TEST 2.1.1: Get All Models")
    print("="*60)
    
    try:
        # Get all models
        print("\n📦 Getting all available models...")
        models = RegressionModels.get_all_models(random_state=42)
        
        print(f"\n✅ Models loaded: {len(models)} models")
        for name, model in models.items():
            print(f"   ✓ {name}: {type(model).__name__}")
        
        # Verify we have at least the core models
        core_models = ['RandomForest', 'GradientBoosting', 'Ridge', 'Lasso', 
                      'ElasticNet', 'SVR', 'KNN', 'MLP']
        
        for model_name in core_models:
            assert model_name in models, f"Missing core model: {model_name}"
        
        print(f"\n✅ All {len(core_models)} core models available")
        
        # Check optional models
        optional_models = ['XGBoost', 'LightGBM', 'CatBoost']
        available_optional = [m for m in optional_models if m in models]
        
        if available_optional:
            print(f"✅ Optional models available: {', '.join(available_optional)}")
        else:
            print("ℹ️  No optional models installed (XGBoost, LightGBM, CatBoost)")
        
        print(f"\n✅ TEST 2.1.1 PASSED: {len(models)} models loaded successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.1.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_instantiation():
    """Test 2: All models can be instantiated."""
    print("\n" + "="*60)
    print("TEST 2.1.2: Model Instantiation")
    print("="*60)
    
    try:
        models = RegressionModels.get_all_models(random_state=42)
        
        print(f"\n🔍 Testing instantiation of {len(models)} models...")
        
        for name, model in models.items():
            # Verify model has required methods
            assert hasattr(model, 'fit'), f"{name} missing 'fit' method"
            assert hasattr(model, 'predict'), f"{name} missing 'predict' method"
            assert hasattr(model, 'get_params'), f"{name} missing 'get_params' method"
            
            # Get params to verify configuration
            params = model.get_params()
            assert isinstance(params, dict), f"{name} get_params() should return dict"
            
            print(f"   ✓ {name}: fit, predict, get_params available")
        
        print(f"\n✅ All {len(models)} models properly instantiated")
        print("✅ TEST 2.1.2 PASSED: Model instantiation working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.1.2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_training_basic():
    """Test 3: Basic training on synthetic data."""
    print("\n" + "="*60)
    print("TEST 2.1.3: Basic Model Training")
    print("="*60)
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        print(f"\n📊 Synthetic data: {X.shape} features, {y.shape} targets")
        
        # Get models
        models = RegressionModels.get_all_models(random_state=42)
        
        print(f"\n🏋️  Training {len(models)} models...")
        
        trained_count = 0
        for name, model in models.items():
            try:
                # Train
                model.fit(X, y)
                
                # Predict
                y_pred = model.predict(X)
                
                # Verify predictions
                assert y_pred.shape == y.shape, f"{name} wrong prediction shape"
                assert not np.isnan(y_pred).any(), f"{name} produced NaN predictions"
                assert not np.isinf(y_pred).any(), f"{name} produced inf predictions"
                
                trained_count += 1
                print(f"   ✓ {name}: trained and predicted successfully")
                
            except Exception as e:
                print(f"   ⚠️  {name} training failed: {str(e)[:50]}...")
        
        # At least core models should train
        assert trained_count >= 8, f"Too few models trained: {trained_count}/8"
        
        print(f"\n✅ {trained_count}/{len(models)} models trained successfully")
        print("✅ TEST 2.1.3 PASSED: Basic training working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.1.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_specific_model():
    """Test 4: Get specific model by name."""
    print("\n" + "="*60)
    print("TEST 2.1.4: Get Specific Model")
    print("="*60)
    
    try:
        print("\n🎯 Testing specific model retrieval...")
        
        # Test getting specific models
        model_names = ['RandomForest', 'Ridge', 'GradientBoosting']
        
        for name in model_names:
            model = RegressionModels.get_model(name, random_state=123)
            
            assert model is not None, f"Failed to get {name}"
            assert hasattr(model, 'fit'), f"{name} missing fit method"
            
            # Verify random_state was set
            params = model.get_params()
            if 'random_state' in params:
                assert params['random_state'] == 123, f"{name} wrong random_state"
            
            print(f"   ✓ {name}: retrieved successfully")
        
        print("\n✅ Specific model retrieval working")
        print("✅ TEST 2.1.4 PASSED: Get specific model working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.1.4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_names():
    """Test 5: Verify model names and types."""
    print("\n" + "="*60)
    print("TEST 2.1.5: Model Names and Types")
    print("="*60)
    
    try:
        models = RegressionModels.get_all_models(random_state=42)
        
        print(f"\n📋 Checking {len(models)} model names and types...")
        
        # Expected model types
        expected_types = {
            'RandomForest': 'RandomForestRegressor',
            'GradientBoosting': 'GradientBoostingRegressor',
            'Ridge': 'Ridge',
            'Lasso': 'Lasso',
            'ElasticNet': 'ElasticNet',
            'SVR': 'SVR',
            'KNN': 'KNeighborsRegressor',
            'MLP': 'MLPRegressor'
        }
        
        for name, expected_type in expected_types.items():
            assert name in models, f"Missing model: {name}"
            actual_type = type(models[name]).__name__
            assert actual_type == expected_type, f"{name}: wrong type {actual_type} vs {expected_type}"
            print(f"   ✓ {name}: {actual_type}")
        
        print(f"\n✅ All model names and types correct")
        print("✅ TEST 2.1.5 PASSED: Model names/types verified")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.1.5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_reproducibility():
    """Test 6: Random state ensures reproducibility."""
    print("\n" + "="*60)
    print("TEST 2.1.6: Reproducibility")
    print("="*60)
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        print("\n🔁 Testing reproducibility with same random_state...")
        
        # Train same model twice with same random_state
        model1 = RegressionModels.get_model('RandomForest', random_state=999)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = RegressionModels.get_model('RandomForest', random_state=999)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Should give identical predictions
        assert np.allclose(pred1, pred2), "Predictions differ with same random_state"
        
        print("   ✓ Same random_state → identical predictions")
        
        # Train with different random_state
        model3 = RegressionModels.get_model('RandomForest', random_state=111)
        model3.fit(X, y)
        pred3 = model3.predict(X)
        
        # Should give different predictions
        assert not np.allclose(pred1, pred3), "Predictions identical with different random_state"
        
        print("   ✓ Different random_state → different predictions")
        
        print("\n✅ Reproducibility working correctly")
        print("✅ TEST 2.1.6 PASSED: Random state working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2.1.6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Level 2.1 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 2.1: MODELS FACTORY TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_get_all_models": test_get_all_models(),
        "test_model_instantiation": test_model_instantiation(),
        "test_model_training_basic": test_model_training_basic(),
        "test_get_specific_model": test_get_specific_model(),
        "test_model_names": test_model_names(),
        "test_reproducibility": test_reproducibility()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 2.1 TEST SUMMARY")
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
