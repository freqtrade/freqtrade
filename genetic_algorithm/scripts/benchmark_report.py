#!/usr/bin/env python3
"""
Benchmark Report Generator

Reads output from a benchmark suite run and generates a structured
comparison report (text + optional charts).

Usage:
    python genetic_algorithm/scripts/benchmark_report.py <benchmark_output_dir>

Example:
    python genetic_algorithm/scripts/benchmark_report.py \
        genetic_algorithm/output/benchmark_20260312_140000
"""

import sys
import os
import csv
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# ─── Run metadata ─────────────────────────────────────────────────────────

RUN_LABELS = {
    # v1 labels
    'run1_baseline_raw':        'R1: Baseline (raw)',
    'run2_walkforward_only':    'R2: Walk-Forward only',
    'run3_full_antioverfit':    'R3: Full anti-overfit',
    'run4_island_regime':       'R4: Island + regime',
    'run5_multi_pair':          'R5: Multi-pair (4)',
    'run6_nsga2_multiobjective':'R6: NSGA-II',
    'run7_short_selling':       'R7: Short selling',
    'run8_fee_noise_robust':    'R8: Fee noise + MC',
    # v2 labels
    'run1_baseline':            'R1: Baseline (3y)',
    'run2_walkforward':         'R2: Walk-Forward 90d/30d',
    'run4_island_sma_slope':    'R4: Island + sma_slope',
    'run5_multi_pair':          'R5: Multi-pair (4)',
    'run6_nsga2':               'R6: NSGA-II',
    'run7_short_selling':       'R7: Short selling',
    'run8_island_ensemble':     'R8: Island + ensemble',
}

FEATURE_MATRIX = {
    # v1 features
    'run1_baseline_raw':        {'WF': False, 'Holdout': False, 'MC': False, 'DSR': False, 'Parsimony': False, 'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False},
    'run2_walkforward_only':    {'WF': True,  'Holdout': False, 'MC': False, 'DSR': False, 'Parsimony': False, 'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False},
    'run3_full_antioverfit':    {'WF': True,  'Holdout': True,  'MC': True,  'DSR': True,  'Parsimony': True,  'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False},
    'run4_island_regime':       {'WF': False, 'Holdout': False, 'MC': False, 'DSR': True,  'Parsimony': True,  'Island': True,  'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': True,  'MultiPair': False},
    'run5_multi_pair':          {'WF': True,  'Holdout': True,  'MC': False, 'DSR': True,  'Parsimony': True,  'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': True},
    'run6_nsga2_multiobjective':{'WF': False, 'Holdout': True,  'MC': False, 'DSR': True,  'Parsimony': True,  'Island': False, 'NSGA2': True,  'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False},
    'run7_short_selling':       {'WF': True,  'Holdout': True,  'MC': False, 'DSR': True,  'Parsimony': True,  'Island': False, 'NSGA2': False, 'Short': True,  'FeeNoise': False, 'Regime': False, 'MultiPair': False},
    'run8_fee_noise_robust':    {'WF': True,  'Holdout': True,  'MC': True,  'DSR': True,  'Parsimony': True,  'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': True,  'Regime': False, 'MultiPair': False},
    # v2 features
    'run1_baseline':            {'WF': False, 'Holdout': False, 'MC': False, 'DSR': False, 'Parsimony': False, 'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False, 'SmaSlope': False},
    'run2_walkforward':         {'WF': True,  'Holdout': False, 'MC': False, 'DSR': False, 'Parsimony': False, 'Island': False, 'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False, 'SmaSlope': False},
    'run4_island_sma_slope':    {'WF': False, 'Holdout': False, 'MC': False, 'DSR': True,  'Parsimony': True,  'Island': True,  'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': True,  'MultiPair': False, 'SmaSlope': True},
    'run6_nsga2':               {'WF': False, 'Holdout': True,  'MC': False, 'DSR': True,  'Parsimony': True,  'Island': False, 'NSGA2': True,  'Short': False, 'FeeNoise': False, 'Regime': False, 'MultiPair': False, 'SmaSlope': False},
    'run8_island_ensemble':     {'WF': False, 'Holdout': False, 'MC': False, 'DSR': True,  'Parsimony': True,  'Island': True,  'NSGA2': False, 'Short': False, 'FeeNoise': False, 'Regime': True,  'MultiPair': False, 'SmaSlope': False},
}


def read_file_safe(path: Path) -> str:
    """Read file, return empty string on failure."""
    try:
        return path.read_text()
    except Exception:
        return ''


def parse_duration(benchmark_dir: Path, run_name: str) -> int:
    """Read duration from .duration file."""
    content = read_file_safe(benchmark_dir / f"{run_name}.duration")
    try:
        return int(content.strip())
    except (ValueError, AttributeError):
        return 0


def parse_status(benchmark_dir: Path, run_name: str) -> str:
    """Read status from .status file."""
    content = read_file_safe(benchmark_dir / f"{run_name}.status")
    return content.strip() if content.strip() else 'UNKNOWN'


def parse_log_metrics(benchmark_dir: Path, run_name: str) -> Dict[str, Any]:
    """Extract key metrics from a run's log file."""
    log_file = benchmark_dir / f"{run_name}.log"
    content = read_file_safe(log_file)
    if not content:
        return {}

    metrics = {}

    # Best fitness (multiple formats)
    for pattern in [
        r'Best fitness.*?:\s*([\d.]+)',
        r'best_fitness["\']:\s*([\d.]+)',
        r'Best ever:\s*([\d.]+)',
    ]:
        m = re.findall(pattern, content)
        if m:
            metrics['best_fitness'] = float(m[-1])
            break

    # Average fitness
    m = re.findall(r'(?:avg|average|mean).*?fitness.*?:\s*([\d.]+)', content, re.I)
    if m:
        metrics['avg_fitness'] = float(m[-1])

    # Generations completed
    m = re.findall(r'GENERATION\s+(\d+)/(\d+)', content)
    if m:
        metrics['gen_completed'] = int(m[-1][0])
        metrics['gen_total'] = int(m[-1][1])

    # Convergence type
    if 'Converged:' in content or 'converged early' in content.lower():
        metrics['convergence'] = 'early'
    elif 'TIME LIMIT' in content:
        metrics['convergence'] = 'time_limit'
    else:
        metrics['convergence'] = 'complete'

    # Diversity
    m = re.findall(r'[Dd]iversity.*?:\s*([\d.]+)', content)
    if m:
        metrics['final_diversity'] = float(m[-1])

    # Total trades in best strategy
    m = re.findall(r'[Tt]otal.*?trades.*?:\s*(\d+)', content)
    if m:
        metrics['best_trades'] = int(m[-1])

    # Holdout degradation
    m = re.findall(r'[Hh]oldout.*?degradation.*?:\s*([\d.]+)', content)
    if m:
        metrics['holdout_degradation'] = float(m[-1])

    # Profit percent from top strategy
    m = re.findall(r'[Pp]rofit.*?:\s*([-\d.]+)%', content)
    if m:
        metrics['best_profit_pct'] = float(m[-1])

    # Sharpe ratio
    m = re.findall(r'[Ss]harpe.*?:\s*([-\d.]+)', content)
    if m:
        metrics['best_sharpe'] = float(m[-1])

    # Overfit classification counts
    safe = len(re.findall(r'\bSAFE\b', content))
    warning = len(re.findall(r'\b(?:WARNING|CAUTION)\b', content))
    overfit = len(re.findall(r'\b(?:OVERFIT|DANGER)\b', content))
    if safe + warning + overfit > 0:
        metrics['overfit_safe'] = safe
        metrics['overfit_warning'] = warning
        metrics['overfit_overfit'] = overfit

    return metrics


def parse_generation_csv(benchmark_dir: Path, run_name: str) -> List[Dict[str, float]]:
    """Read generation_stats.csv for convergence curve data."""
    # Try multiple possible locations
    candidates = [
        benchmark_dir / run_name / 'generation_stats.csv',
        benchmark_dir / run_name / 'evolution_stats.csv',
    ]

    for csv_path in candidates:
        if csv_path.exists():
            rows = []
            try:
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        parsed = {}
                        for k, v in row.items():
                            try:
                                parsed[k] = float(v)
                            except (ValueError, TypeError):
                                parsed[k] = v
                        rows.append(parsed)
                return rows
            except Exception:
                continue
    return []


def parse_detailed_json(benchmark_dir: Path, run_name: str) -> Dict:
    """Read detailed results JSON if available."""
    run_dir = benchmark_dir / run_name
    if not run_dir.exists():
        return {}

    for f in run_dir.iterdir():
        if f.name.startswith('results_detailed') and f.suffix == '.json':
            try:
                return json.loads(f.read_text())
            except Exception:
                pass
    return {}


def format_duration(seconds: int) -> str:
    """Format seconds into Mm Ss."""
    return f"{seconds // 60}m {seconds % 60}s"


def generate_report(benchmark_dir: Path) -> str:
    """Generate the full comparison report."""
    lines = []

    def add(text=''):
        lines.append(text)

    # ─── Header ──────────────────────────────────────────────────────
    add('=' * 72)
    add('  GA BENCHMARK COMPARISON REPORT')
    add(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    add(f'  Source: {benchmark_dir}')
    add('=' * 72)
    add()

    # ─── Collect all data ────────────────────────────────────────────
    # Auto-detect which runs are present in this benchmark directory
    active_runs = []
    for run_name in RUN_LABELS:
        status_file = benchmark_dir / f"{run_name}.status"
        log_file = benchmark_dir / f"{run_name}.log"
        if status_file.exists() or log_file.exists() or (benchmark_dir / run_name).is_dir():
            active_runs.append(run_name)

    # Fallback: if no known runs found, discover from .status files
    if not active_runs:
        for f in sorted(benchmark_dir.glob('*.status')):
            run_name = f.stem
            if run_name not in active_runs:
                active_runs.append(run_name)
                if run_name not in RUN_LABELS:
                    RUN_LABELS[run_name] = run_name

    all_data = {}
    for run_name in active_runs:
        data = {
            'label': RUN_LABELS[run_name],
            'status': parse_status(benchmark_dir, run_name),
            'duration': parse_duration(benchmark_dir, run_name),
            'metrics': parse_log_metrics(benchmark_dir, run_name),
            'gen_stats': parse_generation_csv(benchmark_dir, run_name),
            'features': FEATURE_MATRIX.get(run_name, {}),
        }
        all_data[run_name] = data

    # ─── Feature Matrix ──────────────────────────────────────────────
    add('FEATURE MATRIX')
    add('-' * 72)
    feature_keys = ['WF', 'Holdout', 'MC', 'DSR', 'Parsimony', 'Island',
                    'NSGA2', 'Short', 'FeeNoise', 'Regime', 'MultiPair', 'SmaSlope']
    header = f"{'Run':<30}" + ''.join(f'{k:>9}' for k in feature_keys)
    add(header)
    add('-' * len(header))
    for run_name in active_runs:
        feats = FEATURE_MATRIX.get(run_name, {})
        row = f"{RUN_LABELS.get(run_name, run_name):<30}"
        for k in feature_keys:
            row += f"{'  ✓':>9}" if feats.get(k) else f"{'  ·':>9}"
        add(row)
    add()

    # ─── Performance Comparison Table ────────────────────────────────
    add('PERFORMANCE COMPARISON')
    add('-' * 72)
    perf_header = (f"{'Run':<30} {'Status':>8} {'Time':>8} {'BestFit':>8} "
                   f"{'AvgFit':>8} {'Gens':>6} {'Conv':>10}")
    add(perf_header)
    add('-' * len(perf_header))

    for run_name in active_runs:
        d = all_data[run_name]
        m = d['metrics']
        status = d['status'][:8]
        dur = format_duration(d['duration']) if d['duration'] > 0 else 'N/A'
        best_fit = f"{m.get('best_fitness', 0):.4f}" if 'best_fitness' in m else 'N/A'
        avg_fit = f"{m.get('avg_fitness', 0):.4f}" if 'avg_fitness' in m else 'N/A'
        gens = f"{m.get('gen_completed', '?')}/{m.get('gen_total', '?')}" if 'gen_completed' in m else 'N/A'
        conv = m.get('convergence', 'N/A')

        row = f"{RUN_LABELS.get(run_name, run_name):<30} {status:>8} {dur:>8} {best_fit:>8} {avg_fit:>8} {gens:>6} {conv:>10}"
        add(row)
    add()

    # ─── Robustness Comparison ───────────────────────────────────────
    add('ROBUSTNESS & OVERFITTING')
    add('-' * 72)
    rob_header = (f"{'Run':<30} {'HoldDeg':>8} {'Sharpe':>8} "
                  f"{'SAFE':>5} {'WARN':>5} {'OFIT':>5}")
    add(rob_header)
    add('-' * len(rob_header))

    for run_name in active_runs:
        d = all_data[run_name]
        m = d['metrics']
        hold_deg = f"{m['holdout_degradation']:.2f}" if 'holdout_degradation' in m else 'N/A'
        sharpe = f"{m['best_sharpe']:.2f}" if 'best_sharpe' in m else 'N/A'
        safe = str(m.get('overfit_safe', '-'))
        warn = str(m.get('overfit_warning', '-'))
        ofit = str(m.get('overfit_overfit', '-'))

        row = f"{RUN_LABELS.get(run_name, run_name):<30} {hold_deg:>8} {sharpe:>8} {safe:>5} {warn:>5} {ofit:>5}"
        add(row)
    add()

    # ─── Convergence Analysis ────────────────────────────────────────
    add('CONVERGENCE CURVES (best fitness per generation)')
    add('-' * 72)

    for run_name in active_runs:
        d = all_data[run_name]
        gen_stats = d['gen_stats']
        if not gen_stats:
            add(f"  {RUN_LABELS.get(run_name, run_name)}: No generation data available")
            continue

        # Extract best fitness per gen
        best_per_gen = []
        for row in gen_stats:
            for key in ['best_fitness', 'best_raw_fitness', 'best']:
                if key in row:
                    try:
                        best_per_gen.append(float(row[key]))
                    except (ValueError, TypeError):
                        pass
                    break

        if best_per_gen:
            # ASCII sparkline
            if len(best_per_gen) > 1:
                min_v = min(best_per_gen)
                max_v = max(best_per_gen)
                rng = max_v - min_v if max_v > min_v else 1.0
                bars = '▁▂▃▄▅▆▇█'
                spark = ''.join(bars[min(int((v - min_v) / rng * 7), 7)] for v in best_per_gen)
            else:
                spark = '█'
            add(f"  {RUN_LABELS.get(run_name, run_name)}: {spark}")
            vals = ' → '.join(f"{v:.3f}" for v in [best_per_gen[0], best_per_gen[-1]])
            add(f"    Start → End: {vals}  (Δ = {best_per_gen[-1] - best_per_gen[0]:+.3f})")
        else:
            add(f"  {RUN_LABELS.get(run_name, run_name)}: No fitness data in CSV")

    add()

    # ─── Rankings ────────────────────────────────────────────────────
    add('RANKINGS')
    add('-' * 72)

    # Rank by best fitness
    fitness_ranks = []
    for run_name in active_runs:
        m = all_data[run_name]['metrics']
        if 'best_fitness' in m:
            fitness_ranks.append((run_name, m['best_fitness']))
    fitness_ranks.sort(key=lambda x: x[1], reverse=True)

    add('  By Best Fitness (highest = best):')
    for i, (rn, fit) in enumerate(fitness_ranks, 1):
        add(f"    {i}. {RUN_LABELS.get(rn, rn):<30} {fit:.4f}")
    add()

    # Rank by efficiency (fitness / minutes)
    efficiency_ranks = []
    for run_name in active_runs:
        m = all_data[run_name]['metrics']
        dur = all_data[run_name]['duration']
        if 'best_fitness' in m and dur > 60:
            eff = m['best_fitness'] / (dur / 60.0)
            efficiency_ranks.append((run_name, eff))
    efficiency_ranks.sort(key=lambda x: x[1], reverse=True)

    add('  By Efficiency (fitness / minute):')
    for i, (rn, eff) in enumerate(efficiency_ranks, 1):
        add(f"    {i}. {RUN_LABELS.get(rn, rn):<30} {eff:.4f}/min")
    add()

    # ─── Key Insights ────────────────────────────────────────────────
    add('KEY COMPARISONS (automated)')
    add('-' * 72)

    # Auto-detect baseline run (v1: run1_baseline_raw, v2: run1_baseline)
    r1_name = 'run1_baseline' if 'run1_baseline' in all_data else 'run1_baseline_raw'
    r2_name = 'run2_walkforward' if 'run2_walkforward' in all_data else 'run2_walkforward_only'
    r1 = all_data.get(r1_name, {}).get('metrics', {})
    r2 = all_data.get(r2_name, {}).get('metrics', {})

    # R1 vs R2: Cost of walk-forward
    if 'best_fitness' in r1 and 'best_fitness' in r2:
        delta = r2['best_fitness'] - r1['best_fitness']
        pct = (delta / r1['best_fitness'] * 100) if r1['best_fitness'] != 0 else 0
        add(f"  Walk-Forward cost:     R1 → R2 fitness change = {delta:+.4f} ({pct:+.1f}%)")
        if delta < 0:
            add(f"    → WF reduces in-sample fitness by {abs(pct):.1f}% (expected — it penalizes overfitting)")
        else:
            add(f"    → WF increased fitness (unusual — may indicate WF bonus for stable strategies)")

    # R4 vs R8: sma_slope vs ensemble (v2)
    r4_sma = all_data.get('run4_island_sma_slope', {}).get('metrics', {})
    r8_ens = all_data.get('run8_island_ensemble', {}).get('metrics', {})
    if 'best_fitness' in r4_sma and 'best_fitness' in r8_ens:
        delta = r4_sma['best_fitness'] - r8_ens['best_fitness']
        pct = (delta / r8_ens['best_fitness'] * 100) if r8_ens['best_fitness'] != 0 else 0
        add(f"  sma_slope vs ensemble: R4 → R8 fitness change = {delta:+.4f} ({pct:+.1f}%)")
        if delta > 0:
            add(f"    → sma_slope outperforms ensemble by {abs(pct):.1f}%")
        else:
            add(f"    → ensemble outperforms sma_slope by {abs(pct):.1f}%")

    # R1 vs R5: Multi-pair generalization
    r5 = all_data.get('run5_multi_pair', {}).get('metrics', {})
    if 'best_fitness' in r1 and 'best_fitness' in r5:
        delta = r5['best_fitness'] - r1['best_fitness']
        pct = (delta / r1['best_fitness'] * 100) if r1['best_fitness'] != 0 else 0
        add(f"  Multi-pair impact:     R1 → R5 fitness change = {delta:+.4f} ({pct:+.1f}%)")

    # R1 vs R6: NSGA-II
    r6_name = 'run6_nsga2' if 'run6_nsga2' in all_data else 'run6_nsga2_multiobjective'
    r6 = all_data.get(r6_name, {}).get('metrics', {})
    if 'best_fitness' in r1 and 'best_fitness' in r6:
        delta = r6['best_fitness'] - r1['best_fitness']
        pct = (delta / r1['best_fitness'] * 100) if r1['best_fitness'] != 0 else 0
        add(f"  NSGA-II impact:        R1 → R6 fitness change = {delta:+.4f} ({pct:+.1f}%)")

    # R1 vs R7: Short selling
    r7 = all_data.get('run7_short_selling', {}).get('metrics', {})
    if 'best_fitness' in r1 and 'best_fitness' in r7:
        delta = r7['best_fitness'] - r1['best_fitness']
        pct = (delta / r1['best_fitness'] * 100) if r1['best_fitness'] != 0 else 0
        add(f"  Short selling impact:  R1 → R7 fitness change = {delta:+.4f} ({pct:+.1f}%)")

    # v1 comparisons (backward compat)
    r3 = all_data.get('run3_full_antioverfit', {}).get('metrics', {})
    if 'best_fitness' in r1 and 'best_fitness' in r3:
        delta = r3['best_fitness'] - r1['best_fitness']
        pct = (delta / r1['best_fitness'] * 100) if r1['best_fitness'] != 0 else 0
        add(f"  Full anti-overfit cost: R1 → R3 fitness change = {delta:+.4f} ({pct:+.1f}%)")

    r8_old = all_data.get('run8_fee_noise_robust', {}).get('metrics', {})
    if 'best_fitness' in r1 and 'best_fitness' in r8_old:
        delta = r8_old['best_fitness'] - r1['best_fitness']  
        pct = (delta / r1['best_fitness'] * 100) if r1['best_fitness'] != 0 else 0
        add(f"  Fee noise impact:      R1 → R8 fitness change = {delta:+.4f} ({pct:+.1f}%)")

    add()
    add('=' * 72)
    add('  END OF REPORT')
    add('=' * 72)

    return '\n'.join(lines)


def try_generate_charts(benchmark_dir: Path, all_data: Dict = None):
    """Try to generate matplotlib comparison charts. Non-fatal on failure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping charts")
        return

    # Re-read data if not passed
    if all_data is None:
        all_data = {}
        # Auto-detect runs
        active_runs = []
        for f in sorted(benchmark_dir.glob('*.status')):
            run_name = f.stem
            active_runs.append(run_name)
        for run_name in active_runs:
            all_data[run_name] = {
                'label': RUN_LABELS.get(run_name, run_name),
                'metrics': parse_log_metrics(benchmark_dir, run_name),
                'gen_stats': parse_generation_csv(benchmark_dir, run_name),
                'duration': parse_duration(benchmark_dir, run_name),
            }
    else:
        active_runs = list(all_data.keys())

    # ─── Chart 1: Best Fitness Bar Chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    run_names = []
    fitnesses = []
    for rn in active_runs:
        m = all_data[rn]['metrics']
        if 'best_fitness' in m:
            label = RUN_LABELS.get(rn, rn)
            run_names.append(label.split(':')[0] if ':' in label else rn[:6])
            fitnesses.append(m['best_fitness'])

    if fitnesses:
        colors = plt.cm.Set2(np.linspace(0, 1, len(fitnesses)))
        axes[0].barh(run_names, fitnesses, color=colors)
        axes[0].set_xlabel('Best Fitness')
        axes[0].set_title('Best Fitness by Run')
        axes[0].invert_yaxis()

    # ─── Chart 2: Convergence Curves ─────────────────────────────────
    for rn in active_runs:
        gen_stats = all_data[rn]['gen_stats']
        if not gen_stats:
            continue
        best_per_gen = []
        for row in gen_stats:
            for key in ['best_fitness', 'best_raw_fitness', 'best']:
                if key in row:
                    try:
                        best_per_gen.append(float(row[key]))
                    except (ValueError, TypeError):
                        pass
                    break
        if best_per_gen:
            label = RUN_LABELS.get(rn, rn)
            axes[1].plot(range(1, len(best_per_gen) + 1), best_per_gen,
                        label=label.split(':')[0] if ':' in label else rn[:6],
                        marker='o', markersize=3)

    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Best Fitness')
    axes[1].set_title('Convergence Curves')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = benchmark_dir / 'benchmark_comparison.png'
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {chart_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_report.py <benchmark_output_dir>")
        print()
        print("Example:")
        print("  python genetic_algorithm/scripts/benchmark_report.py \\")
        print("      genetic_algorithm/output/benchmark_20260312_140000")
        sys.exit(1)

    benchmark_dir = Path(sys.argv[1])
    if not benchmark_dir.exists():
        print(f"Error: Directory not found: {benchmark_dir}")
        sys.exit(1)

    # Generate text report
    report = generate_report(benchmark_dir)

    # Print to console
    print(report)

    # Save to file
    report_path = benchmark_dir / 'benchmark_comparison_report.txt'
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Try to generate charts
    try_generate_charts(benchmark_dir)


if __name__ == '__main__':
    main()
