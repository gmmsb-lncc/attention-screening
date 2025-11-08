"""
Master Test Runner for Level 7 - Modular Tests
Executes all Level 7 tests in organized groups.
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
        "Level 7A - Basic Compatibility (Fast ~30s)": [
            ("test_7a1_output_format.py", "7A.1: Output Format"),
            ("test_7a2_api_interface.py", "7A.2: API Interface"),
            ("test_7a3_error_messages.py", "7A.3: Error Messages"),
        ],
        "Level 7B - File I/O (Moderate ~1min)": [
            ("test_7b1_file_input.py", "7B.1: File Input"),
            ("test_7b2_dataframe.py", "7B.2: DataFrame"),
        ],
        "Level 7C - Performance (Slower ~2min)": [
            ("test_7c1_performance.py", "7C.1: Performance Baseline"),
        ]
    }
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    print("\n" + "="*80)
    print("LEVEL 7 - EXTENDED TESTS (MODULAR)")
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
    print("LEVEL 7 TEST SUMMARY")
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
    
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
