"""
Master Test Runner for Level 8 - Robustness Tests
Executes all robustness tests to validate edge cases, stress scenarios, and resilience.
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_file, test_name):
    """Run a single test file"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {test_name}")
    print(f"{'='*80}")
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0


def main():
    tests_dir = Path(__file__).parent
    
    # Define test groups
    test_groups = {
        "Level 8A - Edge Cases & Stress (Moderate ~2-3min)": [
            ("test_8a1_edge_cases.py", "8A.1: Edge Cases"),
            ("test_8a2_stress.py", "8A.2: Stress Testing"),
        ],
        "Level 8B - Data Resilience (Fast ~1min)": [
            ("test_8b1_resilience.py", "8B.1: Malformed Inputs"),
        ]
    }
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    print("\n" + "="*80)
    print("LEVEL 8 - ROBUSTNESS TESTS")
    print("="*80)
    print("\n⚠️  These tests intentionally check edge cases and failure scenarios")
    print("   Some 'failures' are expected behavior (graceful error handling)")
    print("="*80)
    
    for group_name, tests in test_groups.items():
        print(f"\n{'='*80}")
        print(f"GROUP: {group_name}")
        print(f"{'='*80}")
        
        for test_file, test_name in tests:
            total_tests += 1
            test_path = tests_dir / test_file
            
            if not test_path.exists():
                print(f"❌ {test_name}: File not found")
                results[test_name] = False
                continue
            
            success = run_test(test_path, test_name)
            results[test_name] = success
            
            if success:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
    
    # Final summary
    print("\n" + "="*80)
    print("LEVEL 8 ROBUSTNESS TEST SUMMARY")
    print("="*80)
    
    for group_name, tests in test_groups.items():
        print(f"\n{group_name}:")
        for test_file, test_name in tests:
            status = "✅ PASSED" if results.get(test_name, False) else "❌ FAILED"
            print(f"  {status} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print(f"Success Rate: {100*passed_tests/total_tests:.1f}%")
    print(f"{'='*80}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL ROBUSTNESS TESTS PASSED!")
        print("   The modular embeddings are robust and production-ready!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        print("   Review failures above for details")
    
    print("="*80)
    
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
