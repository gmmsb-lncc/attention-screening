#!/usr/bin/env python3
"""
Test Level 1.1: DataManager - Data Loading and Preparation (Regression)
Duration: ~10s
Priority: CRITICAL - Foundation for all other tests

Tests the DataManager class which loads .npy embeddings and target values,
creates stratified train/val/test splits, and manages data caching.

This is the first and most critical test as all other components depend on
proper data loading for regression tasks.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.core.data_loader import DataManager


def test_data_loading():
    """Test 1: Basic data loading from .npy files for regression."""
    print("\n" + "="*60)
    print("TEST 1.1: DataManager - Data Loading (Regression)")
    print("="*60)
    
    # Create synthetic data
    n_samples = 100
    n_features = 64
    
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    targets = np.random.uniform(0, 10, size=n_samples).astype(np.float32)
    
    print(f"\n📊 Created synthetic data:")
    print(f"   Embeddings: {embeddings.shape} {embeddings.dtype}")
    print(f"   Targets: {targets.shape} {targets.dtype}")
    print(f"   Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"   Target mean±std: {targets.mean():.2f}±{targets.std():.2f}")
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_tar:
        np.save(f_tar.name, targets)
        tar_path = f_tar.name
    
    try:
        # Load data using DataManager
        print("\n📂 Loading data with DataManager...")
        data_manager = DataManager(
            embeddings_path=emb_path,
            targets_path=tar_path
        )
        
        # Load and split data
        data_manager.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Train: X={X_train.shape}, y={y_train.shape}")
        print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
        
        # Verify splits
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == n_samples, f"Sample count mismatch: {total_samples} != {n_samples}"
        
        # Verify data types
        assert X_train.dtype == np.float32, f"Wrong dtype: {X_train.dtype}"
        assert y_train.dtype == np.float32, f"Wrong dtype: {y_train.dtype}"
        
        print("\n✅ TEST 1.1 PASSED: Data loading successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        os.unlink(emb_path)
        os.unlink(tar_path)


def test_stratified_split():
    """Test 2: Stratified split for regression (bin-based stratification)."""
    print("\n" + "="*60)
    print("TEST 1.2: Stratified Split for Regression")
    print("="*60)
    
    # Create synthetic data with clear bins
    n_samples = 200
    n_features = 64
    
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create targets in 3 ranges: low (0-3), medium (3-7), high (7-10)
    targets = np.concatenate([
        np.random.uniform(0, 3, 70),   # Low
        np.random.uniform(3, 7, 80),   # Medium
        np.random.uniform(7, 10, 50)   # High
    ]).astype(np.float32)
    np.random.shuffle(targets)
    
    print(f"\n📊 Created synthetic data with 3 target ranges:")
    print(f"   Low (0-3): {np.sum((targets >= 0) & (targets < 3))} samples")
    print(f"   Medium (3-7): {np.sum((targets >= 3) & (targets < 7))} samples")
    print(f"   High (7-10): {np.sum(targets >= 7)} samples")
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_tar:
        np.save(f_tar.name, targets)
        tar_path = f_tar.name
    
    try:
        # Load with stratification
        print("\n📂 Loading with stratified split...")
        data_manager = DataManager(
            embeddings_path=emb_path,
            targets_path=tar_path
        )
        
        data_manager.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            stratify_bins=3  # 3 bins for low/medium/high
        )
        
        print(f"\n✅ Stratified split created:")
        print(f"   Train: {len(y_train)} samples")
        print(f"   Val:   {len(y_val)} samples")
        print(f"   Test:  {len(y_test)} samples")
        
        # Check distribution in each split
        def print_distribution(y, name):
            low = np.sum((y >= 0) & (y < 3))
            medium = np.sum((y >= 3) & (y < 7))
            high = np.sum(y >= 7)
            total = len(y)
            print(f"   {name}: Low={low/total:.1%}, Medium={medium/total:.1%}, High={high/total:.1%}")
        
        print("\n📊 Target distribution in splits:")
        print_distribution(y_train, "Train")
        print_distribution(y_val, "Val")
        print_distribution(y_test, "Test")
        
        print("\n✅ TEST 1.2 PASSED: Stratified split successful")
        return True
        
    except Exception as e:
        print(f"\n⚠️  TEST 1.2 FAILED/SKIPPED: {str(e)}")
        print("   (Stratification may not be implemented yet)")
        return False
        
    finally:
        # Cleanup
        os.unlink(emb_path)
        os.unlink(tar_path)


def test_data_shapes():
    """Test 3: Verify correct data shapes and dimensions."""
    print("\n" + "="*60)
    print("TEST 1.3: Data Shapes and Dimensions")
    print("="*60)
    
    n_samples = 150
    n_features = 128
    
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    targets = np.random.uniform(0, 10, size=n_samples).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_tar:
        np.save(f_tar.name, targets)
        tar_path = f_tar.name
    
    try:
        data_manager = DataManager(
            embeddings_path=emb_path,
            targets_path=tar_path
        )
        
        data_manager.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        # Verify feature dimensions
        assert X_train.shape[1] == n_features, f"Wrong features: {X_train.shape[1]} != {n_features}"
        assert X_val.shape[1] == n_features, f"Wrong features: {X_val.shape[1]} != {n_features}"
        assert X_test.shape[1] == n_features, f"Wrong features: {X_test.shape[1]} != {n_features}"
        
        # Verify target dimensions (should be 1D)
        assert y_train.ndim == 1, f"Targets should be 1D, got {y_train.ndim}D"
        assert y_val.ndim == 1, f"Targets should be 1D, got {y_val.ndim}D"
        assert y_test.ndim == 1, f"Targets should be 1D, got {y_test.ndim}D"
        
        # Verify matching samples
        assert len(X_train) == len(y_train), "Train X/y size mismatch"
        assert len(X_val) == len(y_val), "Val X/y size mismatch"
        assert len(X_test) == len(y_test), "Test X/y size mismatch"
        
        print(f"\n✅ All shape checks passed:")
        print(f"   Features: {n_features}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        
        print("\n✅ TEST 1.3 PASSED: Shape validation successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.unlink(emb_path)
        os.unlink(tar_path)


def test_data_caching():
    """Test 4: Data caching functionality."""
    print("\n" + "="*60)
    print("TEST 1.4: Data Caching")
    print("="*60)
    
    n_samples = 80
    n_features = 32
    
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    targets = np.random.uniform(0, 10, size=n_samples).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_tar:
        np.save(f_tar.name, targets)
        tar_path = f_tar.name
    
    try:
        # First load
        print("\n📂 First load (should read from disk)...")
        data_manager = DataManager(
            embeddings_path=emb_path,
            targets_path=tar_path
        )
        
        # Load data
        X_1, y_1 = data_manager.load_data()
        
        # Second access (should use cache)
        print("📂 Second access (should use cache)...")
        X_2, y_2 = data_manager.load_data()
        
        # Verify same data
        assert np.array_equal(X_1, X_2), "Cached data differs!"
        assert np.array_equal(y_1, y_2), "Cached targets differ!"
        
        # Verify same object (true caching)
        assert X_1 is X_2, "Not using cached object!"
        assert y_1 is y_2, "Not using cached object!"
        
        print("\n✅ Caching verified:")
        print(f"   Same data: ✓")
        print(f"   Same object reference: ✓")
        
        print("\n✅ TEST 1.4 PASSED: Caching works correctly")
        return True
        
    except Exception as e:
        print(f"\n⚠️  TEST 1.4 FAILED/SKIPPED: {str(e)}")
        print("   (Caching may not be implemented yet)")
        return False
        
    finally:
        os.unlink(emb_path)
        os.unlink(tar_path)


def run_all_tests():
    """Run all Level 1.1 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.1: DATA LOADER TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_data_loading": test_data_loading(),
        "test_stratified_split": test_stratified_split(),
        "test_data_shapes": test_data_shapes(),
        "test_data_caching": test_data_caching()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 1.1 TEST SUMMARY")
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
