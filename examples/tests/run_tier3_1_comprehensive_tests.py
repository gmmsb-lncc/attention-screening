#!/usr/bin/env python3
"""
Master Test Suite Runner for Tier 3.1 Embedding Optimization

Runs all Tier 3.1 tests in a structured manner and generates a report.

Usage:
    python run_tier3_1_comprehensive_tests.py
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def run_test_file(test_file: Path, verbose: bool = True) -> Dict[str, Any]:
    """Run a single test file and capture results."""
    print(f"\n{'='*80}")
    print(f"Running: {test_file.name}")
    print('='*80)
    
    cmd = [sys.executable, "-m", "unittest", str(test_file)]
    if verbose:
        cmd.append("-v")
    
    result = subprocess.run(
        cmd,
        cwd=str(test_file.parent.parent),
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    print(output)
    
    return {
        "file": test_file.name,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def parse_unittest_output(output: str) -> Dict[str, int]:
    """Parse unittest output to extract test statistics."""
    stats = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    
    # Look for the summary line (e.g., "Ran 29 tests in 1.234s")
    import re
    
    match = re.search(r'Ran (\d+) test', output)
    if match:
        stats["tests"] = int(match.group(1))
    
    # Count failures
    match = re.search(r'FAILED.*failures=(\d+)', output)
    if match:
        stats["failures"] = int(match.group(1))
    
    # Count errors
    match = re.search(r'FAILED.*errors=(\d+)', output)
    if match:
        stats["errors"] = int(match.group(1))
    
    # Look for skipped
    if "skipped" in output.lower():
        match = re.search(r'skipped=(\d+)', output)
        if match:
            stats["skipped"] = int(match.group(1))
    
    return stats


def main():
    """Main entry point."""
    test_dir = Path(__file__).parent
    
    # Define test files to run (in order)
    test_files = [
        test_dir / "test_tier3_embedding_optimization.py",
        test_dir / "test_tier_3_1_integration.py",
    ]
    
    # Filter to existing files
    existing_tests = [f for f in test_files if f.exists()]
    
    if not existing_tests:
        print("❌ No test files found!")
        return 1
    
    print("\n" + "="*80)
    print("TIER 3.1 EMBEDDING OPTIMIZATION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"\n📊 Found {len(existing_tests)} test file(s) to run")
    
    all_results = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    # Run each test file
    for test_file in existing_tests:
        result = run_test_file(test_file)
        all_results.append(result)
        
        # Parse output
        combined_output = result["stdout"] + result["stderr"]
        stats = parse_unittest_output(combined_output)
        
        total_tests += stats["tests"]
        total_failures += stats["failures"]
        total_errors += stats["errors"]
        
        if result["success"]:
            print(f"\n✅ {test_file.name}: PASSED ({stats['tests']} tests)")
        else:
            print(f"\n❌ {test_file.name}: FAILED")
    
    # Generate summary report
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    total_passed = total_tests - total_failures - total_errors
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 Overall Statistics:")
    print(f"   • Total tests:  {total_tests}")
    print(f"   • Passed:       {total_passed}")
    print(f"   • Failures:     {total_failures}")
    print(f"   • Errors:       {total_errors}")
    print(f"   • Success rate: {success_rate:.1f}%")
    
    # Results
    if total_failures == 0 and total_errors == 0:
        print(f"\n✅ ALL TESTS PASSED!")
        exit_code = 0
    else:
        print(f"\n❌ SOME TESTS FAILED")
        exit_code = 1
    
    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": total_passed,
            "failures": total_failures,
            "errors": total_errors,
            "success_rate_percent": success_rate,
        },
        "files": all_results,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = test_dir / f"tier3_1_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved: {report_file}")
    print("\n" + "="*80)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
