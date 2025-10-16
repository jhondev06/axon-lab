#!/usr/bin/env python3
"""
AXON Test Runner Script

Comprehensive test execution with coverage reporting and performance monitoring.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(".2f"        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(".2f"        print(f"❌ {description} failed")
        print(f"Error output:\n{e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AXON Test Runner")
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "performance", "robustness", "validation", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )

    args = parser.parse_args()

    print("🚀 AXON Test Suite Runner")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("src").exists() or not Path("tests").exists():
        print("❌ Error: Must be run from AXON project root directory")
        sys.exit(1)

    # Build pytest command
    pytest_cmd = ["python", "-m", "pytest"]

    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")

    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])

    if args.fail_fast:
        pytest_cmd.append("--tb=short")
    else:
        pytest_cmd.append("--tb=long")

    if args.coverage:
        pytest_cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=90"
        ])

    # Select test files based on type
    if args.test_type == "unit":
        test_files = [
            "tests/test_models.py",
            "tests/test_xgboost_unit.py",
            "tests/test_catboost_unit.py",
            "tests/test_lstm_unit.py",
            "tests/test_ensemble_unit.py",
            "tests/test_sklearn_interface.py"
        ]
    elif args.test_type == "integration":
        test_files = ["tests/test_integration_pipeline.py"]
    elif args.test_type == "performance":
        test_files = ["tests/test_performance.py"]
    elif args.test_type == "robustness":
        test_files = ["tests/test_robustness.py"]
    elif args.test_type == "validation":
        test_files = ["tests/test_validation.py"]
    else:  # all
        test_files = [
            "tests/test_models.py",
            "tests/test_xgboost_unit.py",
            "tests/test_catboost_unit.py",
            "tests/test_lstm_unit.py",
            "tests/test_ensemble_unit.py",
            "tests/test_sklearn_interface.py",
            "tests/test_integration_pipeline.py",
            "tests/test_performance.py",
            "tests/test_robustness.py",
            "tests/test_validation.py"
        ]

    pytest_cmd.extend(test_files)

    # Run tests
    success = run_command(" ".join(pytest_cmd), f"Running {args.test_type} tests")

    if not success:
        sys.exit(1)

    # Run benchmarks if requested
    if args.benchmark:
        benchmark_cmd = [
            "python", "-m", "pytest",
            "tests/test_performance.py",
            "-k", "benchmark",
            "--benchmark-only",
            "--benchmark-json=benchmark_results.json"
        ]

        if not run_command(" ".join(benchmark_cmd), "Running performance benchmarks"):
            sys.exit(1)

    # Generate coverage report if requested
    if args.coverage:
        print(f"\n{'='*60}")
        print("📊 Coverage Report Summary")
        print(f"{'='*60}")

        try:
            result = subprocess.run(
                ["python", "-m", "coverage", "report"],
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate coverage report: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print("🎉 Test Suite Completed")
    print(f"{'='*60}")

    if success:
        print("✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report available in htmlcov/index.html")
        if args.benchmark:
            print("📈 Benchmark results saved to benchmark_results.json")
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()