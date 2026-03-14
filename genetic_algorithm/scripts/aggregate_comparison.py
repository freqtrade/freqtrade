#!/usr/bin/env python3
"""
Aggregate and compare results from the server comparison suite.

Reads log files and output directories from multiple GA runs and
produces a structured comparison report.

Usage:
    python genetic_algorithm/scripts/aggregate_comparison.py <output_dir>
    
Example:
    python genetic_algorithm/scripts/aggregate_comparison.py \
        genetic_algorithm/output/server_comparison_20260314_200000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def extract_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    """Extract key metrics from a GA run log file."""
    metrics = {
        'best_fitness': None,
        'best_profit': None,
        'safe_score': None,
        'total_generations': None,
        'total_runtime_min': None,
        'converged': False,
        'best_strategy_id': None,
        'avg_fitness_final': None,
        'diversity_final': None,
        'memory_peak_mb': None,
        'total_evaluations': 0,
    }
    
    if not log_path.exists():
        return metrics
    
    text = log_path.read_text(errors='replace')
    
    # Best fitness (last occurrence)
    matches = re.findall(r'best_fitness[=: ]+([0-9.]+)', text, re.IGNORECASE)
    if matches:
        metrics['best_fitness'] = float(matches[-1])
    
    # Also try "[NEW BEST]" pattern
    matches = re.findall(r'\[NEW BEST\].*fitness[= ]+([0-9.]+)', text)
    if matches:
        best = max(float(m) for m in matches)
        if metrics['best_fitness'] is None or best > metrics['best_fitness']:
            metrics['best_fitness'] = best
    
    # Best profit
    matches = re.findall(r'profit[=: ]+([-0-9.]+)%', text)
    if matches:
        metrics['best_profit'] = float(matches[-1])
    
    # SAFE score
    match = re.search(r'(\d+)/(\d+)\s*SAFE', text)
    if match:
        metrics['safe_score'] = f"{match.group(1)}/{match.group(2)}"
    
    # Total generations completed
    matches = re.findall(r'GENERATION\s+(\d+)/(\d+)', text)
    if matches:
        last_gen, total_gen = matches[-1]
        metrics['total_generations'] = int(last_gen)
    
    # Convergence
    if 'Converged' in text or 'convergence' in text.lower():
        metrics['converged'] = True
    
    # Runtime
    match = re.search(r'[Tt]otal.*?([0-9.]+)\s*(min|minutes|hour|hours|sec|seconds)', text)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if 'hour' in unit:
            metrics['total_runtime_min'] = value * 60
        elif 'sec' in unit:
            metrics['total_runtime_min'] = value / 60
        else:
            metrics['total_runtime_min'] = value
    
    # Final stats
    stats_matches = re.findall(r'\[STATS\].*Best: ([0-9.]+).*Avg: ([0-9.]+)', text)
    if stats_matches:
        _, avg = stats_matches[-1]
        metrics['avg_fitness_final'] = float(avg)
    
    # Diversity
    div_matches = re.findall(r'Diversity: ([0-9.]+)', text)
    if div_matches:
        metrics['diversity_final'] = float(div_matches[-1])
    
    # Memory peak
    mem_matches = re.findall(r'RSS: (\d+)MB', text)
    if mem_matches:
        metrics['memory_peak_mb'] = max(int(m) for m in mem_matches)
    
    # Total evaluations count
    eval_matches = re.findall(r'\[PARALLEL\] Evaluating (\d+) strategies', text)
    if eval_matches:
        metrics['total_evaluations'] = sum(int(m) for m in eval_matches)
    
    return metrics


def find_best_strategy(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Find the best strategy file in a run output directory."""
    strategy_files = list(output_dir.rglob('*.json'))
    best = None
    
    for sf in strategy_files:
        try:
            data = json.loads(sf.read_text())
            fitness = data.get('fitness', data.get('raw_fitness', 0))
            if best is None or fitness > best.get('fitness', 0):
                best = {
                    'file': str(sf.relative_to(output_dir)),
                    'fitness': fitness,
                    'profit': data.get('metrics', {}).get('profit', 'N/A'),
                    'trades': data.get('metrics', {}).get('num_trades', 'N/A'),
                    'sharpe': data.get('metrics', {}).get('sharpe_ratio', 'N/A'),
                    'drawdown': data.get('metrics', {}).get('max_drawdown', 'N/A'),
                }
        except (json.JSONDecodeError, KeyError):
            continue
    
    return best


def format_table(headers: List[str], rows: List[List[str]], col_widths: List[int] = None) -> str:
    """Format a simple text table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) 
                      for i, h in enumerate(headers)]
    
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|"
    
    lines = [sep, header_row, sep]
    for row in rows:
        lines.append("|" + "|".join(f" {str(v):<{w}} " for v, w in zip(row, col_widths)) + "|")
    lines.append(sep)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate GA comparison results")
    parser.add_argument("output_dir", type=str, help="Path to comparison output directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON (for programmatic use)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Discover runs
    run_names = {
        "run_A_standard": "Standard (single-pop)",
        "run_B_island": "Island Model (parallel)",
        "run_C_nsga2": "NSGA-II (multi-obj)",
    }
    
    results = {}
    for run_name, description in run_names.items():
        log_path = output_dir / f"{run_name}.log"
        run_dir = output_dir / run_name
        
        metrics = extract_metrics_from_log(log_path)
        best_strategy = find_best_strategy(run_dir) if run_dir.exists() else None
        
        results[run_name] = {
            'description': description,
            'metrics': metrics,
            'best_strategy': best_strategy,
            'log_exists': log_path.exists(),
            'output_exists': run_dir.exists(),
        }
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return
    
    # ── Print comparison report ──
    print()
    print("=" * 78)
    print("  SERVER COMPARISON — AGGREGATED RESULTS")
    print(f"  Output: {output_dir}")
    print("=" * 78)
    print()
    
    # Summary table
    headers = ["Run", "Best Fit", "Profit", "SAFE", "Gens", "Time(min)", "Peak MB"]
    rows = []
    for run_name, data in results.items():
        m = data['metrics']
        rows.append([
            data['description'][:25],
            f"{m['best_fitness']:.4f}" if m['best_fitness'] else "N/A",
            f"{m['best_profit']:.2f}%" if m['best_profit'] else "N/A",
            m['safe_score'] or "N/A",
            str(m['total_generations'] or "N/A"),
            f"{m['total_runtime_min']:.0f}" if m['total_runtime_min'] else "N/A",
            str(m['memory_peak_mb'] or "N/A"),
        ])
    
    print(format_table(headers, rows))
    print()
    
    # Best strategies
    print("── Best Strategies ──")
    print()
    for run_name, data in results.items():
        print(f"  {data['description']}:")
        if data['best_strategy']:
            bs = data['best_strategy']
            print(f"    Fitness:  {bs['fitness']}")
            print(f"    Profit:   {bs['profit']}%")
            print(f"    Trades:   {bs['trades']}")
            print(f"    Sharpe:   {bs['sharpe']}")
            print(f"    Drawdown: {bs['drawdown']}")
            print(f"    File:     {bs['file']}")
        else:
            print("    (no strategy output found)")
        print()
    
    # Recommendations
    print("── Recommendations ──")
    print()
    
    valid_runs = {name: data for name, data in results.items() 
                  if data['metrics']['best_fitness'] is not None}
    
    if valid_runs:
        best_run = max(valid_runs.items(), key=lambda x: x[1]['metrics']['best_fitness'])
        print(f"  Highest fitness:  {best_run[1]['description']} "
              f"({best_run[1]['metrics']['best_fitness']:.4f})")
        
        safe_runs = {name: data for name, data in valid_runs.items()
                     if data['metrics']['safe_score'] and data['metrics']['safe_score'].startswith(('4/', '5/'))}
        if safe_runs:
            print(f"  Best SAFE score:  {', '.join(d['description'] for d in safe_runs.values())}")
        
        if any(d['metrics']['total_runtime_min'] for d in valid_runs.values()):
            fastest = min(
                ((name, data) for name, data in valid_runs.items() 
                 if data['metrics']['total_runtime_min']),
                key=lambda x: x[1]['metrics']['total_runtime_min'],
            )
            print(f"  Fastest run:      {fastest[1]['description']} "
                  f"({fastest[1]['metrics']['total_runtime_min']:.0f} min)")
    else:
        print("  No valid results found. Check log files for errors.")
    
    print()
    
    # Save structured report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")
    print()


if __name__ == '__main__':
    main()
