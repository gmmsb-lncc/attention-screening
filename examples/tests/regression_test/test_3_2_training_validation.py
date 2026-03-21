#!/usr/bin/env python3
"""
Test Level 3.2: Training Validation - Model Performance Checks
Duration: ~10s
Priority: MEDIUM - Ensures training produces valid predictions

Tests that trained models produce valid predictions, handle edge cases,
and maintain consistency across training runs.
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


def test_predictions_validity():
    """Test 1: Predictions are valid (no NaN, no Inf)."""
    print("\n" + "="*60)
    print("TEST 3.2.1: Predictions Validity")
    print("="*60)
    
    try:
        # Create data
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50)
        X_val = np.random.randn(10, 5)
        y_val = np.random.randn(10)
        X_test = np.random.randn(10, 5)
        
        print(f"\n📊 Testing prediction validity...")
        
        # Train models
        models = {
            'Ridge': RegressionModels.get_model('Ridge', random_state=42),
            'RandomForest': RegressionModels.get_model('RandomForest', random_state=42)
        }
        
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Test predictions on new data
        for name, model in trainer.trained_models.items():
            pred = model.predict(X_test)
            
            # Check for invalid values
            assert not np.isnan(pred).any(), f"{name} produced NaN predictions"
            assert not np.isinf(pred).any(), f"{name} produced Inf predictions"
            assert pred.shape == (len(X_test),), f"{name} wrong prediction shape"
            
            print(f"   ✓ {name}: {len(pred)} valid predictions")
            print(f"      Range: [{pred.min():.2f}, {pred.max():.2f}]")
        
        print("\n✅ All predictions valid")
        print("✅ TEST 3.2.1 PASSED: Predictions validity check successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.2.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_overfitting_detection():
    """Test 2: Detect potential overfitting."""
    print("\n" + "="*60)
    print("TEST 3.2.2: Overfitting Detection")
    print("="*60)
    
    try:
        # Create data with noise
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = X_train[:, 0] + np.random.randn(100) * 0.5
        X_val = np.random.randn(30, 5)
        y_val = X_val[:, 0] + np.random.randn(30) * 0.5
        
        print(f"\n📊 Training models and checking for overfitting...")
        
        # Train models
        models = {
            'Ridge': RegressionModels.get_model('Ridge', random_state=42),
            'RandomForest': RegressionModels.get_model('RandomForest', random_state=42)
        }
        
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Calculate training predictions
        print(f"\n📈 Performance comparison:")
        for name, model in trainer.trained_models.items():
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
            val_rmse = trainer.val_results[name]['RMSE']
            
            overfitting_ratio = val_rmse / train_rmse
            
            print(f"\n   {name}:")
            print(f"      Train RMSE: {train_rmse:.4f}")
            print(f"      Val RMSE: {val_rmse:.4f}")
            print(f"      Ratio: {overfitting_ratio:.2f}x")
            
            if overfitting_ratio > 3.0:
                print(f"      ⚠️  Possible overfitting")
            else:
                print(f"      ✓ Reasonable generalization")
        
        print("\n✅ TEST 3.2.2 PASSED: Overfitting detection working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.2.2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_small_dataset_handling():
    """Test 3: Handle small datasets gracefully."""
    print("\n" + "="*60)
    print("TEST 3.2.3: Small Dataset Handling")
    print("="*60)
    
    try:
        # Very small dataset
        np.random.seed(42)
        X_train = np.random.randn(15, 3)
        y_train = np.random.randn(15)
        X_val = np.random.randn(5, 3)
        y_val = np.random.randn(5)
        
        print(f"\n📊 Testing with tiny dataset:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        
        # Train models
        models = {
            'Ridge': RegressionModels.get_model('Ridge', random_state=42),
            'KNN': RegressionModels.get_model('KNN', random_state=42)
        }
        
        trainer = RegressionTrainer(models_dict=models, verbose=False)
        trainer.train_all(X_train, y_train, X_val, y_val)
        
        # Verify models trained despite small size
        assert len(trainer.trained_models) == 2, "Not all models trained"
        
        print(f"\n✅ Both models trained successfully:")
        for name in trainer.trained_models.keys():
            print(f"   ✓ {name}: RMSE={trainer.val_results[name]['RMSE']:.4f}")
        
        print("\n✅ TEST 3.2.3 PASSED: Small dataset handling successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.2.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_reproducibility():
    """Test 4: Training is reproducible with same random_state."""
    print("\n" + "="*60)
    print("TEST 3.2.4: Training Reproducibility")
    print("="*60)
    
    try:
        # Create data
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50)
        X_val = np.random.randn(10, 5)
        y_val = np.random.randn(10)
        
        print(f"\n🔁 Training same model twice with same random_state...")
        
        # First training
        models1 = {'RandomForest': RegressionModels.get_model('RandomForest', random_state=999)}
        trainer1 = RegressionTrainer(models_dict=models1, verbose=False)
        trainer1.train_all(X_train, y_train, X_val, y_val)
        pred1 = trainer1.trained_models['RandomForest'].predict(X_val)
        
        # Second training
        models2 = {'RandomForest': RegressionModels.get_model('RandomForest', random_state=999)}
        trainer2 = RegressionTrainer(models_dict=models2, verbose=False)
        trainer2.train_all(X_train, y_train, X_val, y_val)
        pred2 = trainer2.trained_models['RandomForest'].predict(X_val)
        
        # Should be identical
        assert np.allclose(pred1, pred2), "Predictions differ with same random_state"
        
        print(f"   ✓ Predictions identical: max diff = {np.max(np.abs(pred1 - pred2)):.10f}")
        
        # Train with different random_state
        models3 = {'RandomForest': RegressionModels.get_model('RandomForest', random_state=111)}
        trainer3 = RegressionTrainer(models_dict=models3, verbose=False)
        trainer3.train_all(X_train, y_train, X_val, y_val)
        pred3 = trainer3.trained_models['RandomForest'].predict(X_val)
        
        # Should be different
        assert not np.allclose(pred1, pred3), "Predictions identical with different random_state"
        
        print(f"   ✓ Predictions differ with different random_state")
        
        print("\n✅ TEST 3.2.4 PASSED: Reproducibility working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3.2.4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Level 3.2 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 3.2: TRAINING VALIDATION TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_predictions_validity": test_predictions_validity(),
        "test_overfitting_detection": test_overfitting_detection(),
        "test_small_dataset_handling": test_small_dataset_handling(),
        "test_reproducibility": test_reproducibility()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 3.2 TEST SUMMARY")
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
