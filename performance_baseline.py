#!/usr/bin/env python3
"""
AXON Performance Baseline and Regression Detection

This script establishes performance baselines and detects performance regressions
in the AXON codebase.
"""

import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime
import numpy as np


class PerformanceBaseline:
    """Manages performance baselines and regression detection."""

    def __init__(self, baseline_file="performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline_data = self._load_baseline()

    def _load_baseline(self):
        """Load existing baseline data."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_baseline(self):
        """Save baseline data."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baseline_data, f, indent=2)

    def update_baseline(self, benchmark_results, threshold=0.05):
        """
        Update baseline with new benchmark results.

        Args:
            benchmark_results: Dictionary with benchmark results
            threshold: Threshold for considering a result stable (5% by default)
        """
        print("📊 Updating performance baseline...")

        for test_name, metrics in benchmark_results.items():
            if test_name not in self.baseline_data:
                self.baseline_data[test_name] = {
                    'runs': [],
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'last_updated': None
                }

            baseline = self.baseline_data[test_name]
            baseline['runs'].append({
                'timestamp': datetime.now().isoformat(),
                'value': metrics['value'],
                'unit': metrics.get('unit', 'seconds')
            })

            # Keep only last 10 runs for rolling baseline
            if len(baseline['runs']) > 10:
                baseline['runs'] = baseline['runs'][-10:]

            # Update statistics
            values = [run['value'] for run in baseline['runs']]
            baseline['mean'] = statistics.mean(values)
            baseline['std'] = statistics.stdev(values) if len(values) > 1 else 0
            baseline['min'] = min(values)
            baseline['max'] = max(values)
            baseline['last_updated'] = datetime.now().isoformat()

        self._save_baseline()
        print(f"✅ Baseline updated with {len(benchmark_results)} benchmarks")

    def check_regression(self, benchmark_results, threshold=0.1):
        """
        Check for performance regressions.

        Args:
            benchmark_results: Current benchmark results
            threshold: Regression threshold (10% by default)

        Returns:
            Dictionary with regression analysis
        """
        print("🔍 Checking for performance regressions...")

        regressions = {}
        improvements = {}

        for test_name, metrics in benchmark_results.items():
            if test_name not in self.baseline_data:
                print(f"⚠️  No baseline found for {test_name}, skipping regression check")
                continue

            baseline = self.baseline_data[test_name]
            current_value = metrics['value']
            baseline_mean = baseline['mean']
            baseline_std = baseline['std']

            # Calculate regression
            if baseline_mean > 0:
                change_pct = (current_value - baseline_mean) / baseline_mean

                # Check if change is significant (beyond 2 standard deviations)
                significant_threshold = 2 * baseline_std / baseline_mean if baseline_mean > 0 else 0

                if change_pct > max(threshold, significant_threshold):
                    regressions[test_name] = {
                        'current': current_value,
                        'baseline': baseline_mean,
                        'change_pct': change_pct,
                        'threshold': max(threshold, significant_threshold),
                        'unit': metrics.get('unit', 'seconds')
                    }
                elif change_pct < -max(threshold, significant_threshold):
                    improvements[test_name] = {
                        'current': current_value,
                        'baseline': baseline_mean,
                        'change_pct': change_pct,
                        'threshold': max(threshold, significant_threshold),
                        'unit': metrics.get('unit', 'seconds')
                    }

        return {
            'regressions': regressions,
            'improvements': improvements,
            'total_tests': len(benchmark_results),
            'regressed_tests': len(regressions),
            'improved_tests': len(improvements)
        }

    def get_baseline_summary(self):
        """Get summary of current baseline."""
        summary = {
            'total_benchmarks': len(self.baseline_data),
            'benchmarks': {}
        }

        for test_name, data in self.baseline_data.items():
            summary['benchmarks'][test_name] = {
                'mean': data['mean'],
                'std': data['std'],
                'runs': len(data['runs']),
                'last_updated': data['last_updated']
            }

        return summary

    def export_baseline_report(self, output_file="baseline_report.md"):
        """Export baseline report in Markdown format."""
        summary = self.get_baseline_summary()

        report = f"""# AXON Performance Baseline Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Benchmarks**: {summary['total_benchmarks']}
- **Last Updated**: {max([b['last_updated'] for b in summary['benchmarks'].values()] or ['Never'])}

## Benchmark Details

| Benchmark | Mean | Std Dev | Runs | Last Updated |
|-----------|------|---------|------|--------------|
"""

        for name, data in summary['benchmarks'].items():
            report += f"| {name} | {data['mean']:.4f} | {data['std']:.4f} | {data['runs']} | {data['last_updated'][:10]} |\n"

        report += "\n## Notes\n\n"
        report += "- Values are in seconds unless otherwise specified\n"
        report += "- Baseline uses rolling window of last 10 runs\n"
        report += "- Regression detection uses 10% threshold or 2σ, whichever is larger\n"

        with open(output_file, 'w') as f:
            f.write(report)

        print(f"📄 Baseline report exported to {output_file}")


def load_benchmark_results(filename):
    """Load benchmark results from pytest-benchmark JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    results = {}
    for benchmark in data.get('benchmarks', []):
        name = benchmark['name']
        # Use mean time as the primary metric
        results[name] = {
            'value': benchmark['stats']['mean'],
            'unit': 'seconds',
            'stats': benchmark['stats']
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="AXON Performance Baseline Management")
    parser.add_argument(
        'action',
        choices=['update', 'check', 'report', 'summary'],
        help='Action to perform'
    )
    parser.add_argument(
        '--benchmark-file',
        default='benchmark_results.json',
        help='Benchmark results file'
    )
    parser.add_argument(
        '--baseline-file',
        default='performance_baseline.json',
        help='Baseline file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Regression threshold (default: 10%)'
    )

    args = parser.parse_args()

    baseline = PerformanceBaseline(args.baseline_file)

    if args.action == 'update':
        if not Path(args.benchmark_file).exists():
            print(f"❌ Benchmark file {args.benchmark_file} not found")
            return 1

        benchmark_results = load_benchmark_results(args.benchmark_file)
        baseline.update_baseline(benchmark_results)

    elif args.action == 'check':
        if not Path(args.benchmark_file).exists():
            print(f"❌ Benchmark file {args.benchmark_file} not found")
            return 1

        benchmark_results = load_benchmark_results(args.benchmark_file)
        regression_analysis = baseline.check_regression(benchmark_results, args.threshold)

        print(f"\n📈 Regression Analysis Results:")
        print(f"   Total tests: {regression_analysis['total_tests']}")
        print(f"   Regressions: {regression_analysis['regressed_tests']}")
        print(f"   Improvements: {regression_analysis['improved_tests']}")

        if regression_analysis['regressions']:
            print(f"\n❌ Performance Regressions Detected:")
            for test, data in regression_analysis['regressions'].items():
                print(f"   {test}: {data['change_pct']:.1%} slower "
                      f"({data['current']:.4f} vs {data['baseline']:.4f} {data['unit']})")

        if regression_analysis['improvements']:
            print(f"\n✅ Performance Improvements Detected:")
            for test, data in regression_analysis['improvements'].items():
                print(f"   {test}: {abs(data['change_pct']):.1%} faster "
                      f"({data['current']:.4f} vs {data['baseline']:.4f} {data['unit']})")

        # Exit with error code if regressions found
        if regression_analysis['regressed_tests'] > 0:
            print(f"\n❌ Performance regression detected! Failing CI.")
            return 1

    elif args.action == 'report':
        baseline.export_baseline_report()

    elif args.action == 'summary':
        summary = baseline.get_baseline_summary()
        print(f"📊 Baseline Summary:")
        print(f"   Total benchmarks: {summary['total_benchmarks']}")

        if summary['total_benchmarks'] > 0:
            print("
   Benchmarks:"
            for name, data in summary['benchmarks'].items():
                print(f"     {name}: {data['mean']:.4f} ± {data['std']:.4f} "
                      f"({data['runs']} runs)")

    return 0


if __name__ == "__main__":
    exit(main())