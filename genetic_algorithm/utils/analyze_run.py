#!/usr/bin/env python3
"""
Post-Run Analysis CLI

Standalone tool to analyze completed GA run outputs without re-running.
Reads the generation CSV, run metadata, detailed JSON, and HoF to produce
a diagnostic report on convergence, timing, overfitting, and stability.

Usage:
    python -m genetic_algorithm.utils.analyze_run genetic_algorithm/output/overnight_run/
    python -m genetic_algorithm.utils.analyze_run --csv generation_stats.csv
    python -m genetic_algorithm.utils.analyze_run --compare run_A/ run_B/
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    """Load generation_stats.csv into list of dicts."""
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def find_file(directory: Path, pattern: str) -> Optional[Path]:
    """Find the newest file matching a glob pattern in directory."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _safe_float(val: str, default: float = 0.0) -> float:
    """Convert CSV string to float, returning default on empty/error."""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_convergence(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze fitness convergence from generation CSV rows.

    Returns dict with convergence metrics:
    - generations_total, best_fitness_final, improvement_curve
    - stagnation windows, convergence generation (when 95% of final best reached)
    """
    if not rows:
        return {'error': 'no data'}

    best_vals = [_safe_float(r.get('best_fitness', '')) for r in rows]
    avg_vals = [_safe_float(r.get('avg_fitness', '')) for r in rows]

    n = len(best_vals)
    final_best = best_vals[-1] if best_vals else 0
    first_best = best_vals[0] if best_vals else 0
    total_improvement = final_best - first_best

    # When did we reach 95% of final improvement?
    threshold_95 = first_best + total_improvement * 0.95 if total_improvement > 0 else final_best
    convergence_gen = n - 1
    for i, v in enumerate(best_vals):
        if v >= threshold_95:
            convergence_gen = i
            break

    # Stagnation: consecutive gens with <0.1% improvement
    stagnation_runs = []
    run_start = None
    for i in range(1, n):
        if best_vals[i] <= best_vals[i - 1] * 1.001:
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None and (i - run_start) >= 3:
                stagnation_runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None and (n - run_start) >= 3:
        stagnation_runs.append((run_start, n - 1))

    # Best-per-generation improvement rate (moving average of deltas)
    deltas = [best_vals[i] - best_vals[i - 1] for i in range(1, n)]

    return {
        'generations_total': n,
        'first_best': round(first_best, 4),
        'final_best': round(final_best, 4),
        'total_improvement': round(total_improvement, 4),
        'convergence_gen_95pct': convergence_gen,
        'stagnation_windows': stagnation_runs,
        'num_stagnation_windows': len(stagnation_runs),
        'avg_improvement_per_gen': round(total_improvement / max(n - 1, 1), 6),
        'final_avg_fitness': round(avg_vals[-1], 4) if avg_vals else None,
        'best_avg_gap': round(final_best - (avg_vals[-1] if avg_vals else 0), 4),
    }


def analyze_timing(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze timing from generation CSV rows.

    Shows wall time distribution, eval bottleneck %, and outliers.
    """
    wall_times = [_safe_float(r.get('wall_seconds', '')) for r in rows if r.get('wall_seconds')]
    eval_times = [_safe_float(r.get('eval_seconds', '')) for r in rows if r.get('eval_seconds')]

    if not wall_times:
        return {'error': 'no timing data'}

    total_wall = sum(wall_times)
    total_eval = sum(eval_times) if eval_times else 0

    sorted_wall = sorted(wall_times)
    n = len(sorted_wall)
    p50 = sorted_wall[n // 2]
    p90 = sorted_wall[int(n * 0.9)]
    p99 = sorted_wall[min(int(n * 0.99), n - 1)]

    return {
        'total_wall_seconds': round(total_wall, 1),
        'total_wall_human': _format_duration(total_wall),
        'total_eval_seconds': round(total_eval, 1),
        'eval_pct': round(total_eval / max(total_wall, 0.001) * 100, 1),
        'avg_per_gen': round(total_wall / n, 2),
        'p50': round(p50, 2),
        'p90': round(p90, 2),
        'p99': round(p99, 2),
        'min': round(sorted_wall[0], 2),
        'max': round(sorted_wall[-1], 2),
        'generations_timed': n,
    }


def analyze_diversity(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze diversity trajectory from CSV rows."""
    div_vals = [_safe_float(r.get('diversity_score', '')) for r in rows]
    gen_div = [_safe_float(r.get('genetic_diversity', '')) for r in rows]

    if not any(v != 0 for v in div_vals):
        return {'error': 'no diversity data'}

    n = len(div_vals)
    return {
        'initial_diversity': round(div_vals[0], 4) if div_vals else None,
        'final_diversity': round(div_vals[-1], 4) if div_vals else None,
        'diversity_change': round(div_vals[-1] - div_vals[0], 4) if div_vals else None,
        'min_diversity': round(min(div_vals), 4),
        'min_diversity_gen': div_vals.index(min(div_vals)),
        'initial_genetic_div': round(gen_div[0], 4) if gen_div and gen_div[0] else None,
        'final_genetic_div': round(gen_div[-1], 4) if gen_div and gen_div[-1] else None,
    }


def analyze_holdout_trend(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Analyze holdout degradation trend from CSV rows."""
    holdout_rows = [
        (int(r.get('generation', 0)), _safe_float(r.get('holdout_avg_degradation', '')))
        for r in rows if r.get('holdout_avg_degradation')
    ]

    if not holdout_rows:
        return {'holdout_monitored': False}

    gens = [h[0] for h in holdout_rows]
    degs = [h[1] for h in holdout_rows]

    # Is degradation trending up? (sign of progressive overfitting)
    if len(degs) >= 3:
        first_half = degs[:len(degs) // 2]
        second_half = degs[len(degs) // 2:]
        trend = 'worsening' if (sum(second_half) / len(second_half)) > (sum(first_half) / len(first_half)) * 1.1 else 'stable_or_improving'
    else:
        trend = 'insufficient_data'

    return {
        'holdout_monitored': True,
        'checks_performed': len(holdout_rows),
        'first_degradation': round(degs[0], 2),
        'last_degradation': round(degs[-1], 2),
        'min_degradation': round(min(degs), 2),
        'max_degradation': round(max(degs), 2),
        'trend': trend,
    }


def compare_runs(dir_a: Path, dir_b: Path) -> Dict[str, Any]:
    """
    Compare two runs side by side.

    Requires generation_stats.csv and run_metadata.json in each dir.
    """
    result = {'run_a': str(dir_a), 'run_b': str(dir_b)}

    for label, d in [('a', dir_a), ('b', dir_b)]:
        csv_path = find_file(d, 'generation_stats.csv')
        meta_path = find_file(d, 'run_metadata.json')

        if csv_path:
            rows = load_csv(csv_path)
            result[f'{label}_convergence'] = analyze_convergence(rows)
            result[f'{label}_timing'] = analyze_timing(rows)
            result[f'{label}_diversity'] = analyze_diversity(rows)
            result[f'{label}_holdout'] = analyze_holdout_trend(rows)
        else:
            result[f'{label}_error'] = 'generation_stats.csv not found'

        if meta_path:
            result[f'{label}_metadata'] = load_json(meta_path)

    # Deltas
    if 'a_convergence' in result and 'b_convergence' in result:
        a_best = result['a_convergence'].get('final_best', 0)
        b_best = result['b_convergence'].get('final_best', 0)
        result['delta'] = {
            'best_fitness_change': round(b_best - a_best, 4),
            'convergence_gen_change': (
                result['b_convergence'].get('convergence_gen_95pct', 0) -
                result['a_convergence'].get('convergence_gen_95pct', 0)
            ),
        }
        if 'a_timing' in result and 'b_timing' in result:
            a_avg = result['a_timing'].get('avg_per_gen', 0)
            b_avg = result['b_timing'].get('avg_per_gen', 0)
            result['delta']['avg_gen_time_change'] = round(b_avg - a_avg, 2)
            if a_avg > 0:
                result['delta']['speedup_pct'] = round((a_avg - b_avg) / a_avg * 100, 1)

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def print_analysis(run_dir: Path) -> None:
    """Run full analysis on a run directory and print results."""
    print(f"\n{'='*80}")
    print(f"  GA RUN ANALYSIS: {run_dir}")
    print(f"{'='*80}\n")

    # Load metadata
    meta_path = find_file(run_dir, 'run_metadata.json')
    if meta_path:
        meta = load_json(meta_path)
        print("RUN METADATA")
        print("-" * 40)
        print(f"  Run ID:        {meta.get('run_id', 'N/A')}")
        print(f"  Start:         {meta.get('start_time', 'N/A')}")
        print(f"  End:           {meta.get('end_time', 'N/A')}")
        dur = meta.get('duration_seconds')
        print(f"  Duration:      {_format_duration(dur) if dur else 'N/A'}")
        print(f"  Config hash:   {meta.get('config_hash', 'N/A')}")
        print(f"  Git SHA:       {meta.get('git_sha', 'N/A')}")
        print(f"  Python:        {meta.get('python_version', 'N/A')}")
        cfg = meta.get('config', {})
        print(f"  Population:    {cfg.get('population_size')}")
        print(f"  Generations:   {cfg.get('generations')}")
        print(f"  Pairs:         {cfg.get('pairs')}")
        print(f"  Timerange:     {cfg.get('timerange')}")
        print(f"  Walk-Forward:  {cfg.get('walk_forward_enabled')}")
        print(f"  Holdout:       {cfg.get('holdout_enabled')} ({cfg.get('holdout_pct')})")
        print(f"  Parallel:      {cfg.get('parallel_enabled')} ({cfg.get('num_workers')} workers)")
        print()
    else:
        print("  (run_metadata.json not found)\n")

    # Load CSV
    csv_path = find_file(run_dir, 'generation_stats.csv')
    if not csv_path:
        print("  generation_stats.csv not found — cannot analyze.\n")
        # Try to analyze JSON report instead
        json_path = find_file(run_dir, 'results_detailed_*.json')
        if json_path:
            print(f"  Found detailed results: {json_path.name}")
            report = load_json(json_path)
            _print_json_summary(report)
        return

    rows = load_csv(csv_path)
    print(f"  CSV loaded: {len(rows)} generations from {csv_path.name}\n")

    # Convergence
    conv = analyze_convergence(rows)
    print("CONVERGENCE")
    print("-" * 40)
    print(f"  Generations:       {conv['generations_total']}")
    print(f"  First best:        {conv['first_best']}")
    print(f"  Final best:        {conv['final_best']}")
    print(f"  Total improvement: {conv['total_improvement']}")
    print(f"  95% converge gen:  {conv['convergence_gen_95pct']}")
    print(f"  Avg improve/gen:   {conv['avg_improvement_per_gen']}")
    print(f"  Final avg fitness: {conv['final_avg_fitness']}")
    print(f"  Best-avg gap:      {conv['best_avg_gap']}")
    if conv['stagnation_windows']:
        print(f"  Stagnation windows ({conv['num_stagnation_windows']}):")
        for start, end in conv['stagnation_windows']:
            print(f"    Gen {start}-{end} ({end - start + 1} gens)")
    else:
        print(f"  Stagnation:        None detected")
    print()

    # Timing
    timing = analyze_timing(rows)
    if 'error' not in timing:
        print("TIMING")
        print("-" * 40)
        print(f"  Total wall time:  {timing['total_wall_human']}")
        print(f"  Eval time:        {timing['total_eval_seconds']}s ({timing['eval_pct']}%)")
        print(f"  Avg/gen:          {timing['avg_per_gen']}s")
        print(f"  P50/P90/P99:      {timing['p50']}s / {timing['p90']}s / {timing['p99']}s")
        print(f"  Min/Max:          {timing['min']}s / {timing['max']}s")
        print()

    # Diversity
    div = analyze_diversity(rows)
    if 'error' not in div:
        print("DIVERSITY")
        print("-" * 40)
        print(f"  Initial → Final:   {div['initial_diversity']} → {div['final_diversity']} "
              f"({div['diversity_change']:+.4f})")
        print(f"  Min diversity:     {div['min_diversity']} (gen {div['min_diversity_gen']})")
        if div.get('initial_genetic_div') is not None:
            print(f"  Genetic div:       {div['initial_genetic_div']} → {div['final_genetic_div']}")
        print()

    # Holdout
    holdout = analyze_holdout_trend(rows)
    if holdout.get('holdout_monitored'):
        print("HOLDOUT MONITORING")
        print("-" * 40)
        print(f"  Checks:       {holdout['checks_performed']}")
        print(f"  First degrad:  {holdout['first_degradation']}%")
        print(f"  Last degrad:   {holdout['last_degradation']}%")
        print(f"  Min/Max:       {holdout['min_degradation']}% / {holdout['max_degradation']}%")
        print(f"  Trend:         {holdout['trend']}")
        print()

    # Detailed results JSON
    json_path = find_file(run_dir, 'results_detailed_*.json')
    if json_path:
        report = load_json(json_path)
        _print_json_summary(report)

    print(f"{'='*80}")
    print("  Analysis complete.")
    print(f"{'='*80}\n")


def _print_json_summary(report: Dict[str, Any]) -> None:
    """Print summary from a detailed results JSON."""
    summary = report.get('summary', {})
    if summary:
        print("OVERFITTING SUMMARY")
        print("-" * 40)
        print(f"  Total strategies: {summary.get('total', 0)}")
        print(f"  SAFE: {summary.get('safe', 0)}  "
              f"WARNING: {summary.get('warning', 0)}  "
              f"OVERFIT: {summary.get('overfit', 0)}  "
              f"UNKNOWN: {summary.get('unknown', 0)}")
        avg_cs = summary.get('avg_composite_score')
        if avg_cs is not None:
            print(f"  Avg composite:    {avg_cs:.3f}")
        print()

    strategies = report.get('strategies', [])
    if strategies:
        print("TOP STRATEGIES")
        print("-" * 40)
        print(f"  {'Rank':<5} {'Fitness':>8} {'Holdout':>8} {'Degrad':>7} {'MC-Rob':>7} {'Label':<8}")
        for s in strategies[:10]:
            a = s.get('assessment', {})
            hf = a.get('holdout_fitness')
            hd = a.get('holdout_degradation')
            mc = a.get('mc_robustness')
            print(f"  {s['rank']:<5} "
                  f"{a.get('fitness', 0):>8.4f} "
                  f"{(f'{hf:.4f}' if hf is not None else 'N/A'):>8} "
                  f"{(f'{hd:.1%}' if hd is not None else 'N/A'):>7} "
                  f"{(f'{mc:.1%}' if mc is not None else 'N/A'):>7} "
                  f"{a.get('overall_label', '?'):<8}")
        print()


def print_comparison(dir_a: Path, dir_b: Path) -> None:
    """Print side-by-side comparison of two runs."""
    result = compare_runs(dir_a, dir_b)

    print(f"\n{'='*80}")
    print(f"  RUN COMPARISON")
    print(f"  A: {dir_a}")
    print(f"  B: {dir_b}")
    print(f"{'='*80}\n")

    for label, title in [('a', 'RUN A'), ('b', 'RUN B')]:
        conv = result.get(f'{label}_convergence')
        if conv and 'error' not in conv:
            print(f"  {title}: best={conv['final_best']:.4f}  "
                  f"converge@{conv['convergence_gen_95pct']}  "
                  f"stagnations={conv['num_stagnation_windows']}")
        timing = result.get(f'{label}_timing')
        if timing and 'error' not in timing:
            print(f"         avg/gen={timing['avg_per_gen']}s  "
                  f"total={timing['total_wall_human']}  "
                  f"eval%={timing['eval_pct']}%")
        print()

    delta = result.get('delta', {})
    if delta:
        print("DELTA (B - A)")
        print("-" * 40)
        for k, v in delta.items():
            print(f"  {k}: {v}")
        print()

    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze completed GA run outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Analyze a single run directory
  python -m genetic_algorithm.utils.analyze_run genetic_algorithm/output/overnight_run/

  # Compare two runs
  python -m genetic_algorithm.utils.analyze_run --compare run_before/ run_after/

  # Export analysis as JSON
  python -m genetic_algorithm.utils.analyze_run --json genetic_algorithm/output/overnight_run/
""",
    )
    parser.add_argument('run_dir', nargs='?', type=Path, help='Path to run output directory')
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('DIR_A', 'DIR_B'),
                        help='Compare two run directories')
    parser.add_argument('--json', action='store_true', help='Output analysis as JSON')
    parser.add_argument('--csv', type=Path, help='Path to a specific generation_stats.csv')

    args = parser.parse_args()

    if args.compare:
        if args.json:
            result = compare_runs(args.compare[0], args.compare[1])
            print(json.dumps(result, indent=2, default=str))
        else:
            print_comparison(args.compare[0], args.compare[1])
        return 0

    run_dir = args.run_dir
    if args.csv:
        run_dir = args.csv.parent

    if not run_dir:
        parser.print_help()
        return 1

    if not run_dir.exists():
        print(f"Error: directory not found: {run_dir}")
        return 1

    if args.json:
        csv_path = args.csv or find_file(run_dir, 'generation_stats.csv')
        if csv_path:
            rows = load_csv(csv_path)
            analysis = {
                'convergence': analyze_convergence(rows),
                'timing': analyze_timing(rows),
                'diversity': analyze_diversity(rows),
                'holdout': analyze_holdout_trend(rows),
            }
            meta_path = find_file(run_dir, 'run_metadata.json')
            if meta_path:
                analysis['metadata'] = load_json(meta_path)
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print(json.dumps({'error': 'no generation_stats.csv found'}, indent=2))
    else:
        print_analysis(run_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
