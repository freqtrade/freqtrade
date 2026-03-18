#!/usr/bin/env python3
"""
Wave Comparison — Aggregate and compare results from parallel wave experiments.

Reads log files, output directories, and Hall of Fame data from all experiments
in a wave to produce a comprehensive comparison report.

Usage:
    python genetic_algorithm/scripts/wave_comparison.py wave1
    python genetic_algorithm/scripts/wave_comparison.py wave1 --json
    python genetic_algorithm/scripts/wave_comparison.py wave1 --csv wave1_results.csv
    python genetic_algorithm/scripts/wave_comparison.py /absolute/path/to/output_dir

Features:
    - Extracts metrics from log files (fitness, profit, diversity, errors)
    - Classifies experiment outcomes (SAFE/WARNING/OVERFIT)
    - Computes cross-experiment rankings
    - Flags anomalies (0-trade runs, crashed processes, collapsed diversity)
    - Outputs text table, JSON, or CSV
"""

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentMetrics:
    """Metrics extracted from a single experiment run."""
    name: str
    config_file: str = ""

    # Completion
    status: str = "UNKNOWN"          # OK, FAILED, CRASHED, RUNNING, NO_LOG
    exit_generation: Optional[int] = None
    target_generations: Optional[int] = None
    converged: bool = False
    early_stopped: bool = False
    runtime_minutes: Optional[float] = None

    # Fitness
    best_fitness: Optional[float] = None
    avg_fitness_final: Optional[float] = None
    best_profit_pct: Optional[float] = None
    best_sharpe: Optional[float] = None
    best_drawdown: Optional[float] = None
    best_win_rate: Optional[float] = None
    best_trade_count: Optional[int] = None

    # Robustness
    safe_count: Optional[int] = None
    safe_total: Optional[int] = None
    safe_score: Optional[str] = None
    overfit_classifications: Dict[str, int] = field(default_factory=dict)

    # Diversity / Evolution dynamics
    diversity_final: Optional[float] = None
    diversity_min: Optional[float] = None
    mutation_rate_final: Optional[float] = None
    catastrophic_restarts: int = 0
    stagnation_count: int = 0

    # Errors
    error_count: int = 0
    warning_count: int = 0
    zero_trade_backtests: int = 0
    backtest_timeouts: int = 0

    # LLM (if applicable)
    llm_calls: int = 0
    llm_strategies_generated: int = 0
    llm_failures: int = 0

    # Memory
    memory_peak_mb: Optional[int] = None

    # Anomaly flags
    anomalies: List[str] = field(default_factory=list)


def extract_metrics(log_path: Path, name: str) -> ExperimentMetrics:
    """Extract comprehensive metrics from a GA run log file."""
    m = ExperimentMetrics(name=name)

    if not log_path.exists():
        m.status = "NO_LOG"
        m.anomalies.append("No log file found")
        return m

    text = log_path.read_text(errors='replace')
    if not text.strip():
        m.status = "NO_LOG"
        m.anomalies.append("Empty log file")
        return m

    # ── Completion status ──
    if re.search(r'Evolution complete|All generations complete|EVOLUTION COMPLETE', text):
        m.status = "OK"
    elif re.search(r'Converge[d]', text, re.IGNORECASE):
        m.status = "OK"
        m.converged = True
    elif re.search(r'(Traceback|FATAL|Segmentation fault)', text):
        m.status = "CRASHED"
    elif re.search(r'(KeyboardInterrupt|SIGINT|shutdown)', text, re.IGNORECASE):
        m.status = "STOPPED"
    else:
        m.status = "UNKNOWN"

    # ── Generation progress ──
    gen_matches = re.findall(r'GENERATION\s+(\d+)/(\d+)', text)
    if gen_matches:
        m.exit_generation = int(gen_matches[-1][0])
        m.target_generations = int(gen_matches[-1][1])

    # ── Runtime ──
    runtime_match = re.search(r'[Tt]otal.*?(\d+\.?\d*)\s*(min|minutes|hour|hours|sec)', text)
    if runtime_match:
        val = float(runtime_match.group(1))
        unit = runtime_match.group(2).lower()
        if 'hour' in unit:
            m.runtime_minutes = val * 60
        elif 'sec' in unit:
            m.runtime_minutes = val / 60
        else:
            m.runtime_minutes = val

    # ── Best fitness ──
    best_matches = re.findall(r'\[NEW BEST\].*?fitness[= ]+([0-9.]+)', text)
    if best_matches:
        m.best_fitness = max(float(x) for x in best_matches)
    else:
        # Fallback
        best_match2 = re.findall(r'[Bb]est.*?fitness[=: ]+([0-9.]+)', text)
        if best_match2:
            m.best_fitness = max(float(x) for x in best_match2)

    # ── Avg fitness ──
    avg_matches = re.findall(r'Avg: ([0-9.]+)', text)
    if avg_matches:
        m.avg_fitness_final = float(avg_matches[-1])

    # ── Best profit ──
    profit_matches = re.findall(r'profit[=: ]+([-0-9.]+)%', text)
    if profit_matches:
        m.best_profit_pct = float(profit_matches[-1])

    # ── Best Sharpe ──
    sharpe_matches = re.findall(r'[Ss]harpe[=: ]+([-0-9.]+)', text)
    if sharpe_matches:
        m.best_sharpe = float(sharpe_matches[-1])

    # ── Best drawdown ──
    dd_matches = re.findall(r'[Dd]rawdown[=: ]+([-0-9.]+)', text)
    if dd_matches:
        m.best_drawdown = float(dd_matches[-1])

    # ── Best trade count ──
    trade_matches = re.findall(r'trades?[=: ]+(\d+)', text, re.IGNORECASE)
    if trade_matches:
        m.best_trade_count = int(trade_matches[-1])

    # ── SAFE score ──
    safe_match = re.search(r'(\d+)/(\d+)\s*SAFE', text)
    if safe_match:
        m.safe_count = int(safe_match.group(1))
        m.safe_total = int(safe_match.group(2))
        m.safe_score = f"{m.safe_count}/{m.safe_total}"

    # ── Overfit classifications ──
    for label in ['SAFE', 'WARNING', 'OVERFIT', 'UNKNOWN']:
        count = len(re.findall(rf'OverfitAssessment.*?{label}|assessment.*?{label}', text, re.IGNORECASE))
        if count > 0:
            m.overfit_classifications[label] = count

    # ── Diversity ──
    div_matches = re.findall(r'[Dd]iversity[: ]+([0-9.]+)', text)
    if div_matches:
        m.diversity_final = float(div_matches[-1])
        m.diversity_min = min(float(x) for x in div_matches)

    # ── Mutation rate ──
    mut_matches = re.findall(r'mutation_rate[=: ]+([0-9.]+)', text)
    if mut_matches:
        m.mutation_rate_final = float(mut_matches[-1])

    # ── Catastrophic restarts ──
    m.catastrophic_restarts = len(re.findall(r'[Cc]atastrophic', text))

    # ── Stagnation ──
    m.stagnation_count = len(re.findall(r'[Ss]tagnant|[Ss]tagnation', text))

    # ── Early stop ──
    if re.search(r'[Ee]arly.?stop', text):
        m.early_stopped = True

    # ── Errors ──
    m.error_count = len(re.findall(r'^.*(?:ERROR|Exception|Traceback)', text, re.MULTILINE))
    m.warning_count = len(re.findall(r'^.*WARNING', text, re.MULTILINE))

    # ── Zero-trade backtests ──
    m.zero_trade_backtests = len(re.findall(r'0 trades?|num_trades.*?[:= ]+0\b', text))

    # ── Backtest timeouts ──
    m.backtest_timeouts = len(re.findall(r'[Tt]imeout|backtest.*?timed out', text))

    # ── LLM stats ──
    m.llm_calls = len(re.findall(r'LLM.*?call|API.*?request', text, re.IGNORECASE))
    llm_gen_matches = re.findall(r'(?:Generated|created)\s+(\d+)\s+LLM', text, re.IGNORECASE)
    m.llm_strategies_generated = sum(int(x) for x in llm_gen_matches)
    m.llm_failures = len(re.findall(r'LLM.*?fail|LLM.*?error', text, re.IGNORECASE))

    # ── Memory ──
    mem_matches = re.findall(r'RSS: (\d+)MB', text)
    if mem_matches:
        m.memory_peak_mb = max(int(x) for x in mem_matches)

    # ── Anomaly detection ──
    if m.best_fitness is not None and m.best_fitness < 0.05:
        m.anomalies.append("VERY_LOW_FITNESS (< 0.05)")
    if m.diversity_min is not None and m.diversity_min < 0.05:
        m.anomalies.append("COLLAPSED_DIVERSITY (min < 0.05)")
    if m.zero_trade_backtests > 5:
        m.anomalies.append(f"EXCESSIVE_ZERO_TRADES ({m.zero_trade_backtests})")
    if m.error_count > 20:
        m.anomalies.append(f"HIGH_ERROR_COUNT ({m.error_count})")
    if m.backtest_timeouts > 10:
        m.anomalies.append(f"MANY_TIMEOUTS ({m.backtest_timeouts})")
    if m.memory_peak_mb and m.memory_peak_mb > 12000:
        m.anomalies.append(f"HIGH_MEMORY ({m.memory_peak_mb}MB)")
    if m.status == "CRASHED":
        m.anomalies.append("PROCESS_CRASHED")

    return m


def format_table(experiments: List[ExperimentMetrics]) -> str:
    """Format experiments as a text comparison table."""
    lines = []

    # Header
    lines.append("=" * 100)
    lines.append(f"  WAVE COMPARISON REPORT — {len(experiments)} experiments")
    lines.append("=" * 100)
    lines.append("")

    # ── Summary table ──
    headers = ["Experiment", "Status", "Gen", "Best Fit", "Avg Fit", "Profit%",
               "SAFE", "Diversity", "Errors", "Anomalies"]
    col_widths = [30, 8, 7, 9, 8, 8, 6, 9, 6, 20]

    sep = "  " + "  ".join("-" * w for w in col_widths)
    header_row = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))

    lines.append(header_row)
    lines.append(sep)

    for e in experiments:
        gen_str = f"{e.exit_generation or '?'}/{e.target_generations or '?'}"
        row = [
            e.name[:30],
            e.status[:8],
            gen_str[:7],
            f"{e.best_fitness:.4f}" if e.best_fitness else "—",
            f"{e.avg_fitness_final:.4f}" if e.avg_fitness_final else "—",
            f"{e.best_profit_pct:.2f}" if e.best_profit_pct is not None else "—",
            e.safe_score or "—",
            f"{e.diversity_final:.3f}" if e.diversity_final else "—",
            str(e.error_count),
            "; ".join(e.anomalies[:2]) if e.anomalies else "—",
        ]
        lines.append("  " + "  ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)))

    lines.append(sep)
    lines.append("")

    # ── Rankings ──
    lines.append("── Rankings ──")

    # By fitness
    ranked = sorted([e for e in experiments if e.best_fitness], key=lambda e: e.best_fitness, reverse=True)
    if ranked:
        lines.append(f"  By Best Fitness:   {'  >  '.join(f'{e.name}({e.best_fitness:.4f})' for e in ranked)}")

    # By SAFE score
    ranked_safe = sorted([e for e in experiments if e.safe_count is not None],
                         key=lambda e: e.safe_count, reverse=True)
    if ranked_safe:
        lines.append(f"  By SAFE Score:     {'  >  '.join(f'{e.name}({e.safe_score})' for e in ranked_safe)}")

    # By diversity
    ranked_div = sorted([e for e in experiments if e.diversity_final],
                        key=lambda e: e.diversity_final, reverse=True)
    if ranked_div:
        lines.append(f"  By Diversity:      {'  >  '.join(f'{e.name}({e.diversity_final:.3f})' for e in ranked_div)}")

    lines.append("")

    # ── Detailed per-experiment notes ──
    lines.append("── Detailed Notes ──")
    for e in experiments:
        lines.append(f"\n  {e.name}:")
        lines.append(f"    Status: {e.status}, Gen: {e.exit_generation}/{e.target_generations}")
        if e.runtime_minutes:
            lines.append(f"    Runtime: {e.runtime_minutes:.1f} min")
        if e.converged:
            lines.append(f"    Converged: yes")
        if e.early_stopped:
            lines.append(f"    Early-stopped: yes")
        if e.catastrophic_restarts > 0:
            lines.append(f"    Catastrophic restarts: {e.catastrophic_restarts}")
        if e.stagnation_count > 0:
            lines.append(f"    Stagnation events: {e.stagnation_count}")
        if e.mutation_rate_final:
            lines.append(f"    Final mutation rate: {e.mutation_rate_final:.3f}")
        if e.overfit_classifications:
            lines.append(f"    Overfit classifications: {e.overfit_classifications}")
        if e.llm_calls > 0:
            lines.append(f"    LLM: {e.llm_calls} calls, {e.llm_strategies_generated} generated, {e.llm_failures} failures")
        if e.memory_peak_mb:
            lines.append(f"    Memory peak: {e.memory_peak_mb} MB")
        if e.anomalies:
            lines.append(f"    ⚠ Anomalies: {'; '.join(e.anomalies)}")

    lines.append("")

    # ── Anomaly summary ──
    all_anomalies = [(e.name, a) for e in experiments for a in e.anomalies]
    if all_anomalies:
        lines.append("── Anomaly Summary ──")
        for name, anomaly in all_anomalies:
            lines.append(f"  ⚠ {name}: {anomaly}")
        lines.append("")

    # ── Recommendations ──
    lines.append("── Recommendations ──")

    crashed = [e for e in experiments if e.status in ("CRASHED", "FAILED")]
    if crashed:
        lines.append(f"  • INVESTIGATE: {', '.join(e.name for e in crashed)} crashed — check logs for tracebacks")

    low_fitness = [e for e in experiments if e.best_fitness and e.best_fitness < 0.1]
    if low_fitness:
        lines.append(f"  • LOW FITNESS: {', '.join(e.name for e in low_fitness)} — may indicate config or data issues")

    collapsed = [e for e in experiments if e.diversity_min and e.diversity_min < 0.05]
    if collapsed:
        lines.append(f"  • DIVERSITY COLLAPSED: {', '.join(e.name for e in collapsed)} — consider higher mutation/sharing")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Wave Comparison — Aggregate experiment results")
    parser.add_argument("wave", type=str, help="Wave name (e.g., wave1) or absolute path to output dir")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    parser.add_argument("--log-dir", type=str, help="Override log directory path")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent.parent
    wave = args.wave

    # Determine log dir
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = repo_dir / "genetic_algorithm" / "logs"

    # Determine config dir to list experiments
    if os.path.isabs(wave):
        output_dir = Path(wave)
        config_dir = None
    else:
        output_dir = repo_dir / "genetic_algorithm" / "output" / "exploration" / wave
        config_dir = repo_dir / "genetic_algorithm" / "config" / "exploration" / wave

    # Discover experiments
    experiment_names = []

    # Method 1: from config dir (preferred — lists all expected experiments)
    if config_dir and config_dir.exists():
        for f in sorted(config_dir.glob("*.yaml")):
            experiment_names.append(f.stem)

    # Method 2: from output directories
    if not experiment_names and output_dir.exists():
        for d in sorted(output_dir.iterdir()):
            if d.is_dir() and not d.name.startswith('.'):
                experiment_names.append(d.name)

    # Method 3: from log files
    if not experiment_names:
        for f in sorted(log_dir.glob(f"{wave}_*.log")):
            name = f.stem.replace(f"{wave}_", "")
            if name not in experiment_names:
                experiment_names.append(name)

    if not experiment_names:
        print(f"ERROR: No experiments found for wave '{wave}'")
        print(f"  Checked: {config_dir}, {output_dir}, {log_dir}")
        sys.exit(1)

    # Extract metrics for each experiment
    experiments = []
    for name in experiment_names:
        log_path = log_dir / f"{wave}_{name}.log"
        m = extract_metrics(log_path, name)

        if config_dir:
            m.config_file = str(config_dir / f"{name}.yaml")

        experiments.append(m)

    # Output
    if args.json:
        print(json.dumps([asdict(e) for e in experiments], indent=2, default=str))
    elif args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["name", "status", "gen", "best_fitness", "avg_fitness",
                             "profit_pct", "safe_score", "diversity", "errors", "anomalies",
                             "runtime_min", "memory_mb"])
            for e in experiments:
                writer.writerow([
                    e.name, e.status,
                    f"{e.exit_generation}/{e.target_generations}" if e.exit_generation else "—",
                    e.best_fitness, e.avg_fitness_final, e.best_profit_pct,
                    e.safe_score, e.diversity_final, e.error_count,
                    "; ".join(e.anomalies), e.runtime_minutes, e.memory_peak_mb
                ])
        print(f"CSV saved to: {csv_path}")
    else:
        report = format_table(experiments)
        print(report)

        # Save to file
        report_path = output_dir / f"wave_comparison_{wave}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
