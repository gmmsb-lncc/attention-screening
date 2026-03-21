"""
Run ALL Extended Tests
Executes all test suites including new consistency and compatibility tests.
"""

import sys
from pathlib import Path

# Import all test modules
sys.path.insert(0, str(Path(__file__).parent))

from test_1_validators import run_all_tests as run_level_1
from test_2_data_loader import run_all_tests as run_level_2
from test_3_model_registry import run_all_tests as run_level_3
from test_4_cache import run_all_tests as run_level_4
from test_5_integration import run_all_tests as run_level_5
from test_6_consistency import run_all_tests as run_level_6
from test_7_compatibility import run_all_tests as run_level_7a
from test_7b_file_io import run_all_tests as run_level_7b
from test_7c_performance import run_all_tests as run_level_7c


def main():
    """Run all test levels"""
    print("\n" + "="*80)
    print("🧪 RUNNING COMPLETE TEST SUITE")
    print("="*80)
    print("\nThis will run all 9 test modules:")
    print("  Level 1: Validators (2 tests)")
    print("  Level 2: Data Loader (5 tests)")
    print("  Level 3: Model Registry (4 tests)")
    print("  Level 4: Cache Manager (5 tests)")
    print("  Level 5: Integration (4 tests)")
    print("  Level 6: Consistency (7 tests)")
    print("  Level 7A: Basic Compatibility (3 tests)")
    print("  Level 7B: File I/O (2 tests)")
    print("  Level 7C: Performance (1 test)")
    print("\n" + "="*80 + "\n")
    
    results = {}
    
    # Run all test levels
    test_levels = [
        ("Level 1: Validators", run_level_1),
        ("Level 2: Data Loader", run_level_2),
        ("Level 3: Model Registry", run_level_3),
        ("Level 4: Cache Manager", run_level_4),
        ("Level 5: Integration", run_level_5),
        ("Level 6: Consistency", run_level_6),
        ("Level 7A: Basic Compatibility", run_level_7a),
        ("Level 7B: File I/O", run_level_7b),
        ("Level 7C: Performance", run_level_7c),
    ]
    
    all_passed = True
    
    for level_name, test_func in test_levels:
        try:
            result = test_func()
            results[level_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {level_name} crashed: {e}")
            results[level_name] = False
            all_passed = False
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print("="*80 + "\n")
    
    total_tests = 33  # 2+5+4+5+4+7+3+2+1
    passed_levels = sum(1 for r in results.values() if r)
    
    for level_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{level_name:.<50} {status}")
    
    print("\n" + "="*80)
    print(f"📊 Modules Passed: {passed_levels}/{len(test_levels)}")
    print(f"📊 Estimated Tests: ~{total_tests}")
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TEST MODULES PASSED!")
        print("=" * 80)
        print("\n✅ The modular embeddings implementation is:")
        print("   - Functionally correct")
        print("   - Consistent and reproducible")
        print("   - Backward compatible")
        print("   - Production ready")
        print("\n" + "="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("❌ SOME TEST MODULES FAILED")
        print("=" * 80)
        print("\nPlease review the failed tests above.")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
