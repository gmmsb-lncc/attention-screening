#!/usr/bin/env python3
"""
Test Level 1.3: Data Validation - Input Validation Functions  
Duration: ~5s
Priority: HIGH - Ensures data quality checks work correctly

Tests the validation module which validates input data shapes, types,
missing values, and other data quality checks.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from regression.validation import validate_regression_data


def test_valid_data():
    """Test 1: Valid data should pass validation."""
    print("\n" + "="*60)
    print("TEST 1.3.1: Valid Data Validation")
    print("="*60)
    
    try:
        # Create valid data
        X = np.random.randn(100, 50).astype(np.float64)
        y = np.random.randn(100).astype(np.float64)
        
        print(f"\n📊 Testing valid data:")
        print(f"   X shape: {X.shape}, dtype: {X.dtype}")
        print(f"   y shape: {y.shape}, dtype: {y.dtype}")
        
        # Validate
        X_val, y_val = validate_regression_data(X, y)
        
        # Verify returned data
        assert X_val.shape == X.shape, f"X shape changed: {X.shape} -> {X_val.shape}"
        assert y_val.shape == y.shape, f"y shape changed: {y.shape} -> {y_val.shape}"
        assert X_val.dtype == np.float64, f"Wrong X dtype: {X_val.dtype}"
        assert y_val.dtype == np.float64, f"Wrong y dtype: {y_val.dtype}"
        
        print("\n✅ Valid data passed validation")
        print(f"   Returned X: {X_val.shape}, {X_val.dtype}")
        print(f"   Returned y: {y_val.shape}, {y_val.dtype}")
        
        print("\n✅ TEST 1.3.1 PASSED: Valid data validation working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.3.1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_shapes():
    """Test 2: Invalid shapes should raise errors."""
    print("\n" + "="*60)
    print("TEST 1.3.2: Invalid Shape Handling")
    print("="*60)
    
    try:
        print("\n🔍 Testing invalid shapes...")
        
        # Test 1: X with wrong dimensions (1D)
        X_1d = np.random.randn(100)
        y = np.random.randn(100)
        
        try:
            validate_regression_data(X_1d, y)
            print("   ❌ ERROR: 1D X should have been rejected")
            return False
        except ValueError as e:
            print(f"   ✓ 1D X rejected: {str(e)[:50]}...")
        
        # Test 2: X with too many dimensions (3D)
        X_3d = np.random.randn(10, 5, 3)
        y = np.random.randn(10)
        
        try:
            validate_regression_data(X_3d, y)
            print("   ❌ ERROR: 3D X should have been rejected")
            return False
        except ValueError as e:
            print(f"   ✓ 3D X rejected: {str(e)[:50]}...")
        
        # Test 3: y with wrong dimensions (2D with multiple columns)
        X = np.random.randn(10, 5)
        y_2d = np.random.randn(10, 3)
        
        try:
            validate_regression_data(X, y_2d)
            print("   ❌ ERROR: Multi-column y should have been rejected")
            return False
        except ValueError as e:
            print(f"   ✓ Multi-column y rejected: {str(e)[:50]}...")
        
        # Test 4: Mismatched sample counts
        X = np.random.randn(100, 5)
        y = np.random.randn(80)
        
        try:
            validate_regression_data(X, y)
            print("   ❌ ERROR: Mismatched samples should have been rejected")
            return False
        except ValueError as e:
            print(f"   ✓ Mismatched samples rejected: {str(e)[:50]}...")
        
        print("\n✅ All invalid shapes properly rejected")
        print("✅ TEST 1.3.2 PASSED: Shape validation working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.3.2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_type_conversion():
    """Test 3: Type conversion and coercion."""
    print("\n" + "="*60)
    print("TEST 1.3.3: Type Conversion")
    print("="*60)
    
    try:
        print("\n🔄 Testing type conversion...")
        
        # Test 1: int to float conversion
        X_int = np.random.randint(0, 100, size=(50, 10))
        y_int = np.random.randint(0, 100, size=50)
        
        print(f"   Input: X dtype={X_int.dtype}, y dtype={y_int.dtype}")
        
        X_val, y_val = validate_regression_data(X_int, y_int)
        
        assert X_val.dtype == np.float64, f"X not converted to float64: {X_val.dtype}"
        assert y_val.dtype == np.float64, f"y not converted to float64: {y_val.dtype}"
        
        print(f"   Output: X dtype={X_val.dtype}, y dtype={y_val.dtype}")
        print("   ✓ Integer to float64 conversion successful")
        
        # Test 2: List to array conversion
        X_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y_list = [10, 20, 30]
        
        X_val, y_val = validate_regression_data(X_list, y_list)
        
        assert isinstance(X_val, np.ndarray), "X not converted to ndarray"
        assert isinstance(y_val, np.ndarray), "y not converted to ndarray"
        assert X_val.dtype == np.float64, f"X not float64: {X_val.dtype}"
        assert y_val.dtype == np.float64, f"y not float64: {y_val.dtype}"
        
        print("   ✓ List to array conversion successful")
        
        # Test 3: 2D y with single column should flatten
        X = np.random.randn(20, 5)
        y_2d_single = np.random.randn(20, 1)
        
        X_val, y_val = validate_regression_data(X, y_2d_single)
        
        assert y_val.ndim == 1, f"y not flattened: {y_val.ndim}D"
        assert len(y_val) == 20, f"Wrong y length: {len(y_val)}"
        
        print("   ✓ 2D single-column y flattened to 1D")
        
        print("\n✅ All type conversions working correctly")
        print("✅ TEST 1.3.3 PASSED: Type conversion working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1.3.3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Level 1.3 tests."""
    print("\n" + "="*70)
    print("🧪 LEVEL 1.3: VALIDATION TESTS (REGRESSION)")
    print("="*70)
    
    results = {
        "test_valid_data": test_valid_data(),
        "test_invalid_shapes": test_invalid_shapes(),
        "test_type_conversion": test_type_conversion()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 LEVEL 1.3 TEST SUMMARY")
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
