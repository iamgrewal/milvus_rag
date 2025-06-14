#!/usr/bin/env python3
"""
Performance regression detection script for GitHub Actions
"""

import json
import argparse
import sys
from typing import Dict, List, Tuple

def load_benchmark_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Benchmark file not found: {filepath}")
        return {"benchmarks": []}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {filepath}")
        return {"benchmarks": []}

def compare_benchmarks(current: Dict, baseline: Dict, threshold: float) -> Tuple[bool, List[str]]:
    """Compare benchmark results and detect regressions"""
    
    current_benchmarks = {b["name"]: b for b in current.get("benchmarks", [])}
    baseline_benchmarks = {b["name"]: b for b in baseline.get("benchmarks", [])}
    
    regressions = []
    has_regression = False
    
    for name, current_bench in current_benchmarks.items():
        if name not in baseline_benchmarks:
            continue
            
        baseline_bench = baseline_benchmarks[name]
        
        current_time = current_bench["stats"]["mean"]
        baseline_time = baseline_bench["stats"]["mean"]
        
        # Calculate percentage change
        change_percent = ((current_time - baseline_time) / baseline_time) * 100
        
        if change_percent > threshold:
            has_regression = True
            regressions.append(
                f"🔴 {name}: {change_percent:.1f}% slower "
                f"({current_time:.3f}s vs {baseline_time:.3f}s)"
            )
        elif change_percent < -5:  # Improvement
            regressions.append(
                f"🟢 {name}: {abs(change_percent):.1f}% faster "
                f"({current_time:.3f}s vs {baseline_time:.3f}s)"
            )
    
    return has_regression, regressions

def main():
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument("--current", required=True, help="Current benchmark results JSON")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark results JSON")
    parser.add_argument("--threshold", type=float, default=20.0, 
                       help="Regression threshold percentage (default: 20%)")
    
    args = parser.parse_args()
    
    print("🔍 Checking for performance regressions...")
    
    current = load_benchmark_results(args.current)
    baseline = load_benchmark_results(args.baseline)
    
    if not baseline.get("benchmarks"):
        print("⚠️  No baseline benchmarks found. Creating baseline...")
        sys.exit(0)
    
    has_regression, results = compare_benchmarks(current, baseline, args.threshold)
    
    print("\n📊 Performance Comparison Results:")
    print("=" * 50)
    
    for result in results:
        print(result)
    
    if has_regression:
        print(f"\n❌ Performance regression detected (>{args.threshold}% threshold)")
        print("Please investigate and optimize before merging.")
        sys.exit(1)
    else:
        print(f"\n✅ No significant performance regressions detected")
        sys.exit(0)

if __name__ == "__main__":
    main()
