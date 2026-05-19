#!/usr/bin/env python3
"""
Analyze timing results from profiling CSV files

Usage:
    python analyze_timings.py profiling_results/timing_profile_*.csv
    python analyze_timings.py --compare file1.csv file2.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_timing_csv(csv_path: str) -> List[Dict]:
    """Load timing data from CSV file"""
    measurements = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['duration_ms'] = float(row['duration_ms'])
            measurements.append(row)
    return measurements


def analyze_by_component(measurements: List[Dict]) -> Dict:
    """Aggregate measurements by component"""
    by_component = defaultdict(lambda: {'operations': [], 'total': 0.0})

    for m in measurements:
        comp = m['component']
        by_component[comp]['operations'].append({
            'operation': m['operation'],
            'duration_ms': m['duration_ms']
        })
        by_component[comp]['total'] += m['duration_ms']

    return dict(by_component)


def print_summary(measurements: List[Dict], csv_path: str):
    """Print detailed summary of timing measurements"""
    print("=" * 80)
    print(f"Analysis of: {csv_path}")
    print("=" * 80)

    if not measurements:
        print("No measurements found!")
        return

    # Get trace ID and total time
    trace_id = measurements[0]['trace_id']
    e2e_measurement = [m for m in measurements if m['operation'] == 'complete_pipeline']

    print(f"\nTrace ID: {trace_id}")

    if e2e_measurement:
        total_time = e2e_measurement[0]['duration_ms']
        print(f"End-to-End Time: {total_time:.2f}ms ({total_time/1000:.2f}s)")

    # Analyze by component
    by_component = analyze_by_component(measurements)

    print("\n" + "=" * 80)
    print("BREAKDOWN BY COMPONENT")
    print("=" * 80)

    for component in sorted(by_component.keys()):
        data = by_component[component]
        total = data['total']

        if e2e_measurement:
            percentage = (total / e2e_measurement[0]['duration_ms'] * 100)
            print(f"\n{component}: {total:.2f}ms ({percentage:.1f}%)")
        else:
            print(f"\n{component}: {total:.2f}ms")

        print("-" * 60)

        # Sort operations by duration (descending)
        operations = sorted(data['operations'], key=lambda x: x['duration_ms'], reverse=True)

        for op in operations:
            print(f"  • {op['operation']:<45} {op['duration_ms']:>10.2f}ms")

    # Find bottlenecks
    print("\n" + "=" * 80)
    print("TOP 10 SLOWEST OPERATIONS")
    print("=" * 80)

    all_ops = [(m['component'], m['operation'], m['duration_ms']) for m in measurements]
    all_ops.sort(key=lambda x: x[2], reverse=True)

    for i, (comp, op, duration) in enumerate(all_ops[:10], 1):
        print(f"{i:2d}. [{comp}] {op:<40} {duration:>10.2f}ms")

    print("\n" + "=" * 80)


def compare_profiles(csv_paths: List[str]):
    """Compare multiple profiling runs"""
    print("=" * 80)
    print("COMPARING MULTIPLE RUNS")
    print("=" * 80)

    all_measurements = []
    for path in csv_paths:
        measurements = load_timing_csv(path)
        all_measurements.append({
            'path': Path(path).name,
            'measurements': measurements,
            'by_component': analyze_by_component(measurements)
        })

    # Compare end-to-end times
    print("\nEnd-to-End Times:")
    print("-" * 60)

    for data in all_measurements:
        e2e = [m for m in data['measurements'] if m['operation'] == 'complete_pipeline']
        if e2e:
            total_time = e2e[0]['duration_ms']
            print(f"  {data['path']:<40} {total_time:>10.2f}ms ({total_time/1000:.2f}s)")

    # Compare by component
    print("\nComponent Breakdown (average across runs):")
    print("-" * 60)

    # Get all unique components
    all_components = set()
    for data in all_measurements:
        all_components.update(data['by_component'].keys())

    component_stats = {}
    for component in all_components:
        totals = []
        for data in all_measurements:
            if component in data['by_component']:
                totals.append(data['by_component'][component]['total'])

        if totals:
            component_stats[component] = {
                'avg': sum(totals) / len(totals),
                'min': min(totals),
                'max': max(totals),
                'runs': len(totals)
            }

    for component in sorted(component_stats.keys()):
        stats = component_stats[component]
        print(f"\n{component}:")
        print(f"  Average: {stats['avg']:>10.2f}ms")
        print(f"  Min:     {stats['min']:>10.2f}ms")
        print(f"  Max:     {stats['max']:>10.2f}ms")
        print(f"  Runs:    {stats['runs']}")

    print("\n" + "=" * 80)


def export_to_json(measurements: List[Dict], output_path: str):
    """Export measurements to JSON format"""
    by_component = analyze_by_component(measurements)

    output = {
        'trace_id': measurements[0]['trace_id'] if measurements else None,
        'total_measurements': len(measurements),
        'by_component': by_component,
        'all_measurements': measurements
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze timing profiling results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single run
  python analyze_timings.py profiling_results/timing_profile_20241112_143022_abc123.csv

  # Compare multiple runs
  python analyze_timings.py --compare run1.csv run2.csv run3.csv

  # Export to JSON
  python analyze_timings.py --export results.json timing_profile.csv
        """
    )

    parser.add_argument(
        'csv_files',
        nargs='+',
        help='One or more CSV timing files to analyze'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple profiling runs'
    )

    parser.add_argument(
        '--export',
        metavar='JSON_FILE',
        help='Export results to JSON file'
    )

    args = parser.parse_args()

    # Validate files exist
    for csv_file in args.csv_files:
        if not Path(csv_file).exists():
            print(f"Error: File not found: {csv_file}", file=sys.stderr)
            return 1

    try:
        if args.compare and len(args.csv_files) > 1:
            compare_profiles(args.csv_files)
        else:
            # Analyze single file
            measurements = load_timing_csv(args.csv_files[0])
            print_summary(measurements, args.csv_files[0])

            if args.export:
                export_to_json(measurements, args.export)

        return 0

    except Exception as e:
        print(f"Error analyzing timing data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
