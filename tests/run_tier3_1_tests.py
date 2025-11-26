#!/usr/bin/env python3
"""
Master Test Suite Runner for Tier 3.1 Embedding Optimization

This script runs all Tier 3.1 tests in a structured manner and generates
a comprehensive report with statistics and recommendations.

Usage:
    python run_tier3_1_tests.py [--verbose] [--failfast] [--junit-xml]
"""

import sys
import os
import time
import unittest
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from io import StringIO
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSuiteRunner:
    """Orchestrates running and reporting on all Tier 3.1 tests."""
    
    def __init__(self, verbose: bool = False, failfast: bool = False):
        """Initialize test runner."""
        self.verbose = verbose
        self.failfast = failfast
        self.results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None
        self.test_dir = Path(__file__).parent.parent
        
    def discover_tests(self) -> unittest.TestSuite:
        """Discover all Tier 3.1 tests."""
        loader = unittest.TestLoader()
        
        # Use unittest discovery to find tests
        # Look in the tests directory for test files matching pattern
        suite = loader.discover(
            start_dir=str(self.test_dir),
            pattern='test_tier*.py',
            top_level_dir=str(self.test_dir.parent)
        )
        
        return suite
    
    def run_tests(self) -> Tuple[unittest.TestResult, float]:
        """Run all discovered tests."""
        suite = self.discover_tests()
        total_tests = suite.countTestCases()
        
        print("\n" + "="*80)
        print("TIER 3.1 EMBEDDING OPTIMIZATION TEST SUITE")
        print("="*80)
        print(f"\n📊 Running {total_tests} tests...")
        print("-"*80)
        
        # Create runner
        stream: Any = StringIO() if not self.verbose else sys.stdout
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if self.verbose else 1,
            failfast=self.failfast
        )
        
        # Run tests
        self.start_time = time.time()
        result = runner.run(suite)
        self.end_time = time.time()
        
        elapsed = self.end_time - self.start_time
        
        # Store output if not verbose
        if not self.verbose and hasattr(stream, 'getvalue'):
            self.test_output = stream.getvalue()  # type: ignore
        
        return result, elapsed
    
    def generate_report(self, result: unittest.TestResult, elapsed: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": elapsed,
            "summary": {
                "total_tests": result.testsRun,
                "successes": result.testsRun - len(result.failures) - len(result.errors),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
                "success_rate_percent": (
                    ((result.testsRun - len(result.failures) - len(result.errors)) 
                     / result.testsRun * 100) 
                    if result.testsRun > 0 else 0
                ),
            },
            "performance": {
                "avg_time_per_test_ms": (elapsed / result.testsRun * 1000) if result.testsRun > 0 else 0,
                "total_time_seconds": elapsed,
            },
            "failures": [],
            "errors": [],
            "skipped": [],
        }
        
        # Process failures
        for test, traceback in result.failures:
            report["failures"].append({  # type: ignore
                "test": str(test),
                "traceback": traceback,
            })
        
        # Process errors
        for test, traceback in result.errors:
            report["errors"].append({  # type: ignore
                "test": str(test),
                "traceback": traceback,
            })
        
        # Process skipped
        for test, reason in result.skipped:
            report["skipped"].append({  # type: ignore
                "test": str(test),
                "reason": reason,
            })
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted report."""
        summary = report["summary"]
        perf = report["performance"]
        
        # Success indicator
        success_icon = "✅" if summary["failures"] == 0 and summary["errors"] == 0 else "❌"
        success_rate = summary["success_rate_percent"]
        rate_icon = "🟢" if success_rate == 100 else "🟡" if success_rate >= 90 else "🔴"
        
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n{success_icon} Overall Status: {'PASSED' if summary['failures'] == 0 and summary['errors'] == 0 else 'FAILED'}")
        print(f"\n📊 Test Statistics:")
        print(f"   • Total tests: {summary['total_tests']}")
        print(f"   • Successes:   {summary['successes']}")
        print(f"   • Failures:    {summary['failures']}")
        print(f"   • Errors:      {summary['errors']}")
        print(f"   • Skipped:     {summary['skipped']}")
        print(f"\n{rate_icon} Success Rate: {success_rate:.1f}%")
        
        print(f"\n⏱️  Performance:")
        print(f"   • Total time: {perf['total_time_seconds']:.2f}s")
        print(f"   • Per test:   {perf['avg_time_per_test_ms']:.2f}ms")
        
        # Print failures if any
        if report["failures"]:
            print(f"\n❌ FAILURES ({len(report['failures'])}):")
            print("-"*80)
            for failure in report["failures"]:
                print(f"\n{failure['test']}")
                print(failure["traceback"])
        
        # Print errors if any
        if report["errors"]:
            print(f"\n🔴 ERRORS ({len(report['errors'])}):")
            print("-"*80)
            for error in report["errors"]:
                print(f"\n{error['test']}")
                print(error["traceback"])
        
        # Print skipped if any
        if report["skipped"]:
            print(f"\n⏭️  SKIPPED ({len(report['skipped'])}):")
            print("-"*80)
            for skip in report["skipped"]:
                print(f"   • {skip['test']}")
                print(f"     Reason: {skip['reason']}")
        
        print("\n" + "="*80)
    
    def save_report(self, report: Dict[str, Any], filename: str | None = None) -> Path:
        """Save report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tier3_1_test_report_{timestamp}.json"
        
        output_path = self.test_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {output_path}")
        return output_path
    
    def generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        summary = report["summary"]
        perf = report["performance"]
        
        # Code quality recommendations
        if summary["success_rate_percent"] == 100:
            recommendations.append("✅ All tests passing - code quality excellent")
        elif summary["success_rate_percent"] >= 90:
            recommendations.append("🟡 High pass rate (90%+) - address remaining failures")
        else:
            recommendations.append("🔴 Low pass rate - significant issues need attention")
        
        # Performance recommendations
        if perf["avg_time_per_test_ms"] < 100:
            recommendations.append("✅ Tests run quickly (<100ms avg) - good performance")
        elif perf["avg_time_per_test_ms"] < 500:
            recommendations.append("🟡 Tests are acceptable speed (100-500ms) - consider optimization")
        else:
            recommendations.append("🔴 Tests are slow (>500ms) - optimization needed")
        
        # Coverage recommendations
        if summary["total_tests"] >= 20:
            recommendations.append("✅ Comprehensive test coverage (20+ tests)")
        elif summary["total_tests"] >= 10:
            recommendations.append("🟡 Good test coverage (10-20 tests)")
        else:
            recommendations.append("🔴 Limited test coverage (<10 tests)")
        
        # Error handling
        if summary["errors"] == 0:
            recommendations.append("✅ No runtime errors detected")
        else:
            recommendations.append(f"🔴 {summary['errors']} runtime errors found")
        
        return recommendations
    
    def run(self, save_json: bool = False) -> int:
        """Run complete test suite."""
        try:
            result, elapsed = self.run_tests()
            report = self.generate_report(result, elapsed)
            
            self.print_report(report)
            
            # Generate and print recommendations
            recommendations = self.generate_recommendations(report)
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   {rec}")
            
            # Save JSON report if requested
            if save_json:
                self.save_report(report)
            
            # Return exit code
            if result.failures or result.errors:
                return 1
            return 0
            
        except Exception as e:
            print(f"\n❌ Error running tests: {e}")
            import traceback
            traceback.print_exc()
            return 2


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tier 3.1 Embedding Optimization tests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast", "-f",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save JSON report"
    )
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner(
        verbose=args.verbose,
        failfast=args.failfast
    )
    
    exit_code = runner.run(save_json=args.save_json)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
