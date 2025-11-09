#!/usr/bin/env python3
"""
Master Test Runner for Level 1 - Foundation/Unit Tests

This script runs all 7 Level 1 tests in sequence and provides
aggregate statistics and timing information.

Test Structure (7 tests, ~10s execution):
    1.1: DataManager (9 tests)
    1.2: DeviceManager (10 tests)
    1.3: MetricsCalculator (7 tests)
    1.4: ConfigManager (7 tests)
    1.5: DataValidator (7 tests)
    1.6: Import Utils (7 tests)
    1.7: Train/Test Split (6 tests)

Total: 53 individual tests
"""

import subprocess
import time
import sys
from pathlib import Path


def run_test_file(test_file: str, test_name: str) -> dict:
    """
    Run a single test file and capture results.
    
    Args:
        test_file: Path to test file
        test_name: Human-readable test name
        
    Returns:
        dict with status, time, output
    """
    print(f"\n{'='*70}")
    print(f"🧪 Running: {test_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30  # 30s timeout per test
        )
        
        elapsed = time.time() - start_time
        
        # Check if test passed (return code 0)
        passed = result.returncode == 0
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return {
            'name': test_name,
            'file': test_file,
            'passed': passed,
            'time': elapsed,
            'output': result.stdout,
            'error': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"❌ TIMEOUT after {elapsed:.2f}s")
        return {
            'name': test_name,
            'file': test_file,
            'passed': False,
            'time': elapsed,
            'output': '',
            'error': 'Test timeout'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR: {str(e)}")
        return {
            'name': test_name,
            'file': test_file,
            'passed': False,
            'time': elapsed,
            'output': '',
            'error': str(e)
        }


def main():
    """Run all Level 1 tests and generate summary report."""
    
    print("\n" + "="*70)
    print("🚀 LEVEL 1: FOUNDATION/UNIT TESTS - MASTER TEST RUNNER")
    print("="*70)
    print("Running 7 test files with 53 individual tests...")
    print("Estimated time: ~10 seconds")
    print("="*70)
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Define test files in execution order
    tests = [
        (test_dir / "test_1_1_data_loader.py", "Level 1.1: DataManager (9 tests)"),
        (test_dir / "test_1_2_device_manager.py", "Level 1.2: DeviceManager (10 tests)"),
        (test_dir / "test_1_3_metrics_calculator.py", "Level 1.3: MetricsCalculator (7 tests)"),
        (test_dir / "test_1_4_config_manager.py", "Level 1.4: ConfigManager (7 tests)"),
        (test_dir / "test_1_5_data_validation.py", "Level 1.5: DataValidator (7 tests)"),
        (test_dir / "test_1_6_import_utils.py", "Level 1.6: Import Utils (7 tests)"),
        (test_dir / "test_1_7_train_test_split.py", "Level 1.7: Train/Test Split (6 tests)"),
    ]
    
    # Run all tests
    start_time = time.time()
    results = []
    
    for test_file, test_name in tests:
        if not test_file.exists():
            print(f"\n⚠️  WARNING: Test file not found: {test_file}")
            results.append({
                'name': test_name,
                'file': str(test_file),
                'passed': False,
                'time': 0,
                'output': '',
                'error': 'File not found'
            })
            continue
        
        result = run_test_file(str(test_file), test_name)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate summary report
    print("\n" + "="*70)
    print("📊 LEVEL 1 TEST SUMMARY REPORT")
    print("="*70)
    
    passed_count = sum(1 for r in results if r['passed'])
    failed_count = len(results) - passed_count
    
    print(f"\nTest Results:")
    print("-" * 70)
    
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} | {result['time']:6.2f}s | {result['name']}")
        if not result['passed'] and result['error']:
            print(f"       Error: {result['error']}")
    
    print("-" * 70)
    print(f"\nOverall Statistics:")
    print(f"  Total Tests:     {len(results)}/7 files")
    print(f"  Passed:          {passed_count}/7 ({passed_count/len(results)*100:.1f}%)")
    print(f"  Failed:          {failed_count}/7")
    print(f"  Total Time:      {total_time:.2f}s")
    print(f"  Average Time:    {total_time/len(results):.2f}s per test file")
    
    print("\n" + "="*70)
    
    if passed_count == len(results):
        print("🎉 SUCCESS: All Level 1 tests passed!")
        print("✅ Foundation components are fully validated")
        print("📌 Ready to proceed to Level 2 (Model Tests)")
        print("="*70)
        return 0
    else:
        print("⚠️  WARNING: Some tests failed!")
        print(f"   {failed_count} test file(s) need attention")
        print("   Please fix failures before proceeding to Level 2")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
