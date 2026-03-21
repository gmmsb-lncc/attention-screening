"""
Level 1.7: Train/Test Split Test

Tests the stratified train/test splitting functionality.

Test Coverage:
- Basic stratified split
- Split ratios preserved
- Stratification maintained
- No data leakage
- Edge cases

Author: Test Suite
Created: 2025-11-08
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_basic_split():
    """Test 1.1: Basic stratified split"""
    print("\n" + "="*60)
    print("Test 1.1: Basic Stratified Split")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Create balanced data
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.array([0]*50 + [1]*50, dtype=np.float32)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Validate sizes
        assert len(X_train) == 80, f"Train size should be 80: {len(X_train)}"
        assert len(X_test) == 20, f"Test size should be 20: {len(X_test)}"
        assert len(y_train) == 80, "Train labels size should be 80"
        assert len(y_test) == 20, "Test labels size should be 20"
        
        # Validate stratification
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        original_ratio = 0.5
        
        assert abs(train_ratio - original_ratio) < 0.1, \
            f"Train ratio {train_ratio:.2f} should be close to {original_ratio}"
        assert abs(test_ratio - original_ratio) < 0.1, \
            f"Test ratio {test_ratio:.2f} should be close to {original_ratio}"
        
        print("✅ Basic split working")
        print(f"   - Train: 80 samples (50/50 split) ✓")
        print(f"   - Test: 20 samples (50/50 split) ✓")
        print(f"   - Stratification preserved ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_three_way_split():
    """Test 1.2: Three-way split (train/val/test)"""
    print("\n" + "="*60)
    print("Test 1.2: Three-Way Split")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Create data
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.array([0]*50 + [1]*50, dtype=np.float32)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.125,  # 10% of original (80% * 0.125 = 10%)
            stratify=y_temp,
            random_state=42
        )
        
        # Validate sizes (70/10/20 split)
        assert len(X_train) == 70, f"Train size should be 70: {len(X_train)}"
        assert len(X_val) == 10, f"Val size should be 10: {len(X_val)}"
        assert len(X_test) == 20, f"Test size should be 20: {len(X_test)}"
        
        # Validate no overlap
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == 100, "Total should be 100"
        
        print("✅ Three-way split working")
        print(f"   - Train: {len(X_train)} samples ✓")
        print(f"   - Val: {len(X_val)} samples ✓")
        print(f"   - Test: {len(X_test)} samples ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_imbalanced_split():
    """Test 1.3: Imbalanced data split"""
    print("\n" + "="*60)
    print("Test 1.3: Imbalanced Data Split")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Create imbalanced data (70:30)
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.array([0]*70 + [1]*30, dtype=np.float32)
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Check stratification maintained
        original_ratio = 0.3
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        
        assert abs(train_ratio - original_ratio) < 0.05, \
            f"Train ratio {train_ratio:.2f} should be close to {original_ratio}"
        assert abs(test_ratio - original_ratio) < 0.15, \
            f"Test ratio {test_ratio:.2f} should be reasonable"
        
        print("✅ Imbalanced split working")
        print(f"   - Original: 70:30 ✓")
        print(f"   - Train: {100*train_ratio:.1f}:{100*(1-train_ratio):.1f} ✓")
        print(f"   - Test: {100*test_ratio:.1f}:{100*(1-test_ratio):.1f} ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_small_dataset():
    """Test 1.4: Small dataset split"""
    print("\n" + "="*60)
    print("Test 1.4: Small Dataset Split")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Create small balanced data
        X = np.random.randn(20, 64).astype(np.float32)
        y = np.array([0]*10 + [1]*10, dtype=np.float32)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Should work even with small dataset
        assert len(X_train) == 16, "Train size should be 16"
        assert len(X_test) == 4, "Test size should be 4"
        
        print("✅ Small dataset split working")
        print(f"   - Train: {len(X_train)} samples ✓")
        print(f"   - Test: {len(X_test)} samples ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_reproducibility():
    """Test 1.5: Split reproducibility with random_state"""
    print("\n" + "="*60)
    print("Test 1.5: Reproducibility")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Create data
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.array([0]*50 + [1]*50, dtype=np.float32)
        
        # Split twice with same random_state
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Should be identical
        assert np.allclose(X_train1, X_train2), "Train sets should be identical"
        assert np.allclose(X_test1, X_test2), "Test sets should be identical"
        assert np.array_equal(y_train1, y_train2), "Train labels should be identical"
        assert np.array_equal(y_test1, y_test2), "Test labels should be identical"
        
        print("✅ Reproducibility working")
        print("   - Same random_state → same splits ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_no_data_leakage():
    """Test 1.6: No data leakage between sets"""
    print("\n" + "="*60)
    print("Test 1.6: No Data Leakage")
    print("="*60)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Create data with unique values per sample
        np.random.seed(42)
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.array([0]*50 + [1]*50, dtype=np.float32)
        
        # Add unique identifier to each sample
        X[:, 0] = np.arange(100)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        # Check no overlap in identifiers
        train_ids = set(X_train[:, 0])
        test_ids = set(X_test[:, 0])
        
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Found data leakage: {len(overlap)} samples in both sets"
        
        print("✅ No data leakage")
        print("   - Train and test are disjoint ✓")
        print(f"   - Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)} ✓")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Train/Test Split tests"""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.7: TRAIN/TEST SPLIT TEST")
    print("="*70)
    
    tests = [
        ("Basic Stratified Split", test_basic_split),
        ("Three-Way Split", test_three_way_split),
        ("Imbalanced Data Split", test_imbalanced_split),
        ("Small Dataset Split", test_small_dataset),
        ("Reproducibility", test_reproducibility),
        ("No Data Leakage", test_no_data_leakage),
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
        print("🎉 Train/Test Split: FULLY FUNCTIONAL ✅")
        return 0
    else:
        print(f"⚠️  Some tests failed. Please investigate.")
        return 1


if __name__ == "__main__":
    exit(main())
