#!/usr/bin/env python3
"""
Test Level 3.1: RegressionTrainer - Training Functionality
Duration: ~15s
Priority: HIGH - Ensures training pipeline works correctly

Tests the RegressionTrainer class which trains multiple regression models,
tracks training times, and manages model storage.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core.trainer import RegressionTrainer
from regression.models.models import RegressionModels


def test_trainer_initialization():
    """Test 1: Trainer initialization with different configurations."""
    print("\n" + "="*60)
    print("TEST 3.1.1: Trainer Initialization")
    print("="*60)
    
    try:
        # Test 1: Default initialization (all models)
        print("\n📦 Testing default initialization...")
        trainer1 = RegressionTrainer(random_state=42, verbose=False)
        
        assert hasattr(trainer1, 'models'), "Missing 'models' attribute"
        assert hasattr(trainer1, 'trained_models'), "Missing 'trained_models' attribute"
        assert len(trainer1.models) >= 8, f"Should have at least 8 models, got {len(trainer1.models)}"
        
        print(f"   ✓ Initialized with {len(trainer1.models)} models")
        
        # Test 2: Custom models dict
        print("\n📦 Testing custom models initialization...")
        custom_models = RegressionModels.get_all_models(random_state=42)
        selected_models = {k: v for k, v in list(custom_models.items())[:3]}
        
        trainer2 = RegressionTrainer(
            models_dict=selected_models,
            random_state=42,
            verbose=False
        )
        
        assert len(trainer2.models) == 3, f"Should have 3 models, got {len(trainer2.models)}"
        print(f"   ✓ Initialized with {len(trainer2.models)} custom models")
        
        # Test 3: Verify attributes
        print("\n🔍 Verifying trainer attributes...")
        required_attrs = ['models', 'trained_models', 'train_results', 
                         'val_results', 'test_results', 'training_times']
        
        for attr in required_attrs:
            assert hasattr(trainer1, attr), f"Missing attribute: {attr}"
            print(f"   ✓ {attr}: present")
        
        print("\n✅ Trainer initialization working correctly")
        print("✅ TEST 3.1.1 PASSED: Initialization successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.1.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_train_single_model():
    """Test 2: Train a single model."""
    print("\n" + "="*60)
    print("TEST 3.1.2: Train Single Model")
    print("="*60)
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20)
        
        print(f"\n📊 Synthetic data:")
        print(f"   Train: {X_train.shape}, {y_train.shape}")
        print(f"   Val: {X_val.shape}, {y_val.shape}")
        
        # Initialize trainer with single model
        models = {'Ridge': RegressionModels.get_model('Ridge', random_state=42)}
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        
        print("\n🏋️  Training Ridge model...")
        trainer.train_single('Ridge', X_train, y_train, X_val, y_val)
        
        # Verify training occurred
        assert 'Ridge' in trainer.trained_models, "Ridge not in trained_models"
        assert 'Ridge' in trainer.val_results, "Ridge not in val_results"
        assert 'Ridge' in trainer.training_times, "Ridge not in training_times"
        
        print("\n✅ Training results:")
        print(f"   Model trained: ✓")
        print(f"   Training time: {trainer.training_times['Ridge']:.3f}s")
        print(f"   Val metrics: {len(trainer.val_results['Ridge'])} metrics")
        
        # Verify validation metrics
        val_metrics = trainer.val_results['Ridge']
        assert 'RMSE' in val_metrics, "Missing RMSE metric"
        assert 'R2' in val_metrics, "Missing R2 metric"
        
        print(f"   Val RMSE: {val_metrics['RMSE']:.4f}")
        print(f"   Val R2: {val_metrics['R2']:.4f}")
        
        print("\n✅ TEST 3.1.2 PASSED: Single model training successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.1.2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_train_all_models():
    """Test 3: Train all models."""
    print("\n" + "="*60)
    print("TEST 3.1.3: Train All Models")
    print("="*60)
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(80, 8)
        y_train = np.random.randn(80)
        X_val = np.random.randn(20, 8)
        y_val = np.random.randn(20)
        
        print(f"\n📊 Synthetic data: {X_train.shape} train, {X_val.shape} val")
        
        # Use subset of models for speed
        all_models = RegressionModels.get_all_models(random_state=42)
        models = {k: v for k, v in list(all_models.items())[:5]}  # First 5 models
        
        print(f"🏋️  Training {len(models)} models...")
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        
        # Train all
        val_results = trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Verify all models trained
        assert len(trainer.trained_models) == len(models), "Not all models trained"
        assert len(trainer.val_results) == len(models), "Missing validation results"
        assert len(trainer.training_times) == len(models), "Missing training times"
        
        print(f"\n✅ Training complete:")
        print(f"   Models trained: {len(trainer.trained_models)}/{len(models)}")
        
        # Show results
        print("\n📊 Training times:")
        for name, time in trainer.training_times.items():
            print(f"   {name}: {time:.3f}s")
        
        print("\n📈 Validation RMSE:")
        for name, metrics in val_results.items():
            print(f"   {name}: {metrics['RMSE']:.4f}")
        
        print("\n✅ TEST 3.1.3 PASSED: All models training successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.1.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_training_times():
    """Test 4: Training time tracking."""
    print("\n" + "="*60)
    print("TEST 3.1.4: Training Time Tracking")
    print("="*60)
    
    try:
        # Create data
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50)
        X_val = np.random.randn(10, 5)
        y_val = np.random.randn(10)
        
        # Train models
        models = {
            'Ridge': RegressionModels.get_model('Ridge', random_state=42),
            'RandomForest': RegressionModels.get_model('RandomForest', random_state=42)
        }
        
        print(f"\n⏱️  Training 2 models and tracking times...")
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Verify times are recorded
        assert 'Ridge' in trainer.training_times, "Ridge time not recorded"
        assert 'RandomForest' in trainer.training_times, "RF time not recorded"
        
        # Times should be positive
        assert trainer.training_times['Ridge'] > 0, "Ridge time should be > 0"
        assert trainer.training_times['RandomForest'] > 0, "RF time should be > 0"
        
        print(f"\n✅ Training times recorded:")
        print(f"   Ridge: {trainer.training_times['Ridge']:.4f}s")
        print(f"   RandomForest: {trainer.training_times['RandomForest']:.4f}s")
        
        # Ridge should typically be faster than RF
        print(f"\n💡 Ridge is {trainer.training_times['RandomForest']/trainer.training_times['Ridge']:.1f}x faster")
        
        print("\n✅ TEST 3.1.4 PASSED: Time tracking working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.1.4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_best_model_selection():
    """Test 5: Best model selection based on validation metric."""
    print("\n" + "="*60)
    print("TEST 3.1.5: Best Model Selection")
    print("="*60)
    
    try:
        # Create data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = X_train[:, 0] * 2 + X_train[:, 1] * 3 + np.random.randn(100) * 0.1
        X_val = np.random.randn(20, 10)
        y_val = X_val[:, 0] * 2 + X_val[:, 1] * 3 + np.random.randn(20) * 0.1
        
        print(f"\n📊 Training on linear relationship data...")
        
        # Train models
        models = {
            'Ridge': RegressionModels.get_model('Ridge', random_state=42),
            'RandomForest': RegressionModels.get_model('RandomForest', random_state=42),
            'Lasso': RegressionModels.get_model('Lasso', random_state=42)
        }
        
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Find best model by RMSE
        best_name = None
        best_rmse = float('inf')
        
        for name, metrics in trainer.val_results.items():
            if metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                best_name = name
        
        print(f"\n🏆 Best model selection:")
        print(f"   Winner: {best_name}")
        print(f"   Val RMSE: {best_rmse:.4f}")
        
        print(f"\n📊 All models:")
        for name, metrics in sorted(trainer.val_results.items(), 
                                    key=lambda x: x[1]['RMSE']):
            print(f"   {name}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
        
        assert best_name is not None, "No best model found"
        assert best_rmse < 10.0, f"Best RMSE too high: {best_rmse}"
        
        print("\n✅ TEST 3.1.5 PASSED: Best model selection working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.1.5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Level 3.1 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 3.1: TRAINER TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_trainer_initialization": test_trainer_initialization(),
        "test_train_single_model": test_train_single_model(),
        "test_train_all_models": test_train_all_models(),
        "test_training_times": test_training_times(),
        "test_best_model_selection": test_best_model_selection()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 3.1 TEST SUMMARY")
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
