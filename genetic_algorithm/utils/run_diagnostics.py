"""
Run Diagnostics — Generation CSV, Timing, and Run Metadata

Provides structured diagnostic output for every GA run:
- Per-generation CSV with fitness, diversity, holdout, and timing data
- Wall-clock timing instrumentation (per-generation, per-phase)
- Run metadata manifest (config hash, start/end, environment info)

Usage in evolution.py:
    from genetic_algorithm.utils.run_diagnostics import RunDiagnostics

    diag = RunDiagnostics(output_dir)
    diag.start_run(config)

    for gen in range(generations):
        diag.start_generation(gen)
        # ... evaluate, select, crossover, mutate ...
        diag.end_generation(gen, stats, population, extras={})

    diag.end_run(top_strategies)
"""

import csv
import json
import hashlib
import logging
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing tracker
# ---------------------------------------------------------------------------

@dataclass
class GenerationTiming:
    """Wall-clock timing for one generation."""
    generation: int
    wall_seconds: float = 0.0
    eval_seconds: float = 0.0  # time inside evaluate_population
    selection_seconds: float = 0.0
    crossover_seconds: float = 0.0
    mutation_seconds: float = 0.0
    holdout_seconds: float = 0.0
    overhead_seconds: float = 0.0  # everything else

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TimingTracker:
    """
    Lightweight stopwatch for measuring phases within a generation.

    Usage:
        tt = TimingTracker()
        tt.start_generation(gen)
        tt.start_phase('eval')
        # ... evaluate ...
        tt.end_phase('eval')
        timing = tt.end_generation(gen)
    """

    def __init__(self):
        self._gen_start: float = 0.0
        self._phase_start: float = 0.0
        self._current_gen: int = -1
        self._phases: Dict[str, float] = {}
        self.history: List[GenerationTiming] = []

    def start_generation(self, gen: int) -> None:
        self._gen_start = time.monotonic()
        self._current_gen = gen
        self._phases = {}

    def start_phase(self, name: str) -> None:
        self._phase_start = time.monotonic()

    def end_phase(self, name: str) -> None:
        elapsed = time.monotonic() - self._phase_start
        self._phases[name] = self._phases.get(name, 0.0) + elapsed

    def end_generation(self, gen: int) -> GenerationTiming:
        wall = time.monotonic() - self._gen_start
        known = sum(self._phases.values())
        timing = GenerationTiming(
            generation=gen,
            wall_seconds=round(wall, 3),
            eval_seconds=round(self._phases.get('eval', 0.0), 3),
            selection_seconds=round(self._phases.get('selection', 0.0), 3),
            crossover_seconds=round(self._phases.get('crossover', 0.0), 3),
            mutation_seconds=round(self._phases.get('mutation', 0.0), 3),
            holdout_seconds=round(self._phases.get('holdout', 0.0), 3),
            overhead_seconds=round(max(wall - known, 0.0), 3),
        )
        self.history.append(timing)
        return timing

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate timing summary across all generations."""
        if not self.history:
            return {}
        n = len(self.history)
        total_wall = sum(t.wall_seconds for t in self.history)
        total_eval = sum(t.eval_seconds for t in self.history)
        return {
            'generations_timed': n,
            'total_wall_seconds': round(total_wall, 1),
            'avg_wall_per_gen': round(total_wall / n, 2),
            'total_eval_seconds': round(total_eval, 1),
            'eval_pct': round(total_eval / max(total_wall, 0.001) * 100, 1),
            'fastest_gen': round(min(t.wall_seconds for t in self.history), 2),
            'slowest_gen': round(max(t.wall_seconds for t in self.history), 2),
        }


# ---------------------------------------------------------------------------
# Generation CSV writer
# ---------------------------------------------------------------------------

# Column order for the CSV — first columns are always present,
# remaining are optional (written as empty string if None).
_CSV_COLUMNS = [
    'generation',
    'best_fitness',
    'avg_fitness',
    'worst_fitness',
    'median_fitness',
    'best_raw_fitness',
    'avg_raw_fitness',
    'diversity_score',
    'genetic_diversity',
    'population_size',
    # Holdout monitoring
    'holdout_avg_degradation',
    'holdout_best_degradation',
    'holdout_num_evaluated',
    'holdout_num_profitable',
    # Best individual metrics
    'best_profit',
    'best_sharpe',
    'best_win_rate',
    'best_drawdown',
    'best_num_trades',
    # Timing
    'wall_seconds',
    'eval_seconds',
    # Mutation rate (if adaptive)
    'mutation_rate',
    # New feature tracking (Plan 1 + Plan 2)
    'holdout_penalties_applied',
    'avg_holdout_penalty',
    'avg_unused_indicators',
    'llm_seeds_count',
    'llm_immigrants_count',
]


class GenerationCSVWriter:
    """
    Writes one row per generation to a CSV file.

    Lazily creates the file on first write so the header is guaranteed.
    """

    def __init__(self, output_dir: Path, filename: str = 'generation_stats.csv'):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._output_dir / filename
        self._writer: Optional[csv.DictWriter] = None
        self._file = None

    @property
    def filepath(self) -> Path:
        return self._filepath

    def _ensure_open(self) -> None:
        if self._writer is None:
            self._file = open(self._filepath, 'w', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=_CSV_COLUMNS,
                                          extrasaction='ignore')
            self._writer.writeheader()

    def write_row(self, row: Dict[str, Any]) -> None:
        """Write a single generation row. Missing keys become empty strings."""
        self._ensure_open()
        # Replace None with empty string for clean CSV
        clean = {k: ('' if v is None else v) for k, v in row.items()}
        self._writer.writerow(clean)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None


# ---------------------------------------------------------------------------
# Run metadata manifest
# ---------------------------------------------------------------------------

def _config_hash(config: Dict[str, Any]) -> str:
    """Deterministic SHA-256 of the config dict (first 12 hex chars)."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _git_sha() -> str:
    """Best-effort git HEAD SHA (returns 'unknown' on failure)."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def build_run_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build machine-readable run metadata.

    Called once at start of run; end_time and duration filled in later.
    """
    ga = config.get('genetic_algorithm', {})
    bt = config.get('backtesting', {})
    wf = config.get('walk_forward', {})
    par = config.get('parallel_evaluation', {})
    ho = config.get('holdout_validation', {})
    mc = config.get('monte_carlo', {})

    return {
        'run_id': f"run_{int(time.time())}",
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration_seconds': None,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'git_sha': _git_sha(),
        'config_hash': _config_hash(config),
        'config': {
            'population_size': ga.get('population_size'),
            'generations': ga.get('generations'),
            'mutation_rate': ga.get('mutation_rate'),
            'crossover_rate': ga.get('crossover_rate'),
            'elite_size': ga.get('elite_size'),
            'selection_method': ga.get('selection_method'),
            'crossover_method': ga.get('crossover_method', 'uniform'),
            'pairs': bt.get('pairs', []),
            'timerange': bt.get('timerange', ''),
            'stake_currency': bt.get('stake_currency', ''),
            'walk_forward_enabled': wf.get('enabled', False),
            'walk_forward_windows': wf.get('num_windows'),
            'holdout_enabled': ho.get('enabled', False),
            'holdout_pct': ho.get('holdout_pct'),
            'monte_carlo_enabled': mc.get('enabled', False),
            'parallel_enabled': par.get('enabled', False),
            'num_workers': par.get('num_workers'),
        },
    }


def save_run_metadata(metadata: Dict[str, Any], output_dir: Path) -> Path:
    """Save metadata JSON to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / 'run_metadata.json'
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Run metadata saved to: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Orchestrator: RunDiagnostics (wires everything together)
# ---------------------------------------------------------------------------

class RunDiagnostics:
    """
    High-level facade that manages CSV, timing, and metadata for a single run.

    Drop-in integration point for evolution.py — call start_run / record_generation / end_run.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timing = TimingTracker()
        self.csv_writer = GenerationCSVWriter(self.output_dir)
        self.metadata: Dict[str, Any] = {}
        self._run_start: float = 0.0

    # -- lifecycle ----------------------------------------------------------

    def start_run(self, config: Dict[str, Any]) -> None:
        """Call once before the evolution loop starts."""
        self._run_start = time.monotonic()
        self.metadata = build_run_metadata(config)
        save_run_metadata(self.metadata, self.output_dir)
        logger.info(f"[DIAGNOSTICS] Run {self.metadata['run_id']} started, "
                     f"CSV → {self.csv_writer.filepath}")

    def start_generation(self, gen: int) -> None:
        """Call at the top of each generation."""
        self.timing.start_generation(gen)

    def start_phase(self, name: str) -> None:
        """Mark the beginning of a named phase (eval, selection, crossover, mutation, holdout)."""
        self.timing.start_phase(name)

    def end_phase(self, name: str) -> None:
        """Mark the end of a named phase."""
        self.timing.end_phase(name)

    def end_generation(
        self,
        gen: int,
        stats,  # PopulationStats
        population=None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Finish a generation: record timing + write CSV row.

        Args:
            gen: Current generation number.
            stats: PopulationStats for this generation.
            population: Current population (used to extract best individual metrics).
            extras: Optional dict of extra columns (e.g. mutation_rate).
        """
        gtiming = self.timing.end_generation(gen)

        row: Dict[str, Any] = {
            'generation': gen,
            'best_fitness': getattr(stats, 'best_fitness', None),
            'avg_fitness': getattr(stats, 'avg_fitness', None),
            'worst_fitness': getattr(stats, 'worst_fitness', None),
            'median_fitness': getattr(stats, 'median_fitness', None),
            'best_raw_fitness': getattr(stats, 'best_raw_fitness', None),
            'avg_raw_fitness': getattr(stats, 'avg_raw_fitness', None),
            'diversity_score': getattr(stats, 'diversity_score', None),
            'genetic_diversity': getattr(stats, 'genetic_diversity', None),
            'population_size': getattr(stats, 'size', None),
            'holdout_avg_degradation': getattr(stats, 'holdout_avg_degradation', None),
            'holdout_best_degradation': getattr(stats, 'holdout_best_degradation', None),
            'holdout_num_evaluated': getattr(stats, 'holdout_num_evaluated', None),
            'holdout_num_profitable': getattr(stats, 'holdout_num_profitable', None),
            'wall_seconds': gtiming.wall_seconds,
            'eval_seconds': gtiming.eval_seconds,
        }

        # Extract best individual metrics
        if population is not None:
            try:
                best = population.get_best(1)
                if best:
                    m = best[0].metrics or {}
                    row['best_profit'] = m.get('profit')
                    row['best_sharpe'] = m.get('sharpe_ratio')
                    row['best_win_rate'] = m.get('win_rate')
                    row['best_drawdown'] = m.get('max_drawdown')
                    row['best_num_trades'] = m.get('num_trades')
            except Exception:
                pass

        if extras:
            row.update(extras)

        self.csv_writer.write_row(row)

    def end_run(self, top_strategies: Optional[list] = None) -> Dict[str, Any]:
        """
        Finalize the run: close CSV, update metadata with end time + timing summary.

        Returns the timing summary dict.
        """
        self.csv_writer.close()

        # Update metadata
        duration = time.monotonic() - self._run_start
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = round(duration, 1)
        self.metadata['timing_summary'] = self.timing.get_summary()

        if top_strategies:
            self.metadata['result_summary'] = {
                'top_strategies_returned': len(top_strategies),
                'best_fitness': max(
                    (getattr(s, 'fitness', 0) or 0 for s in top_strategies), default=0
                ),
            }

        # Re-save with final data
        save_run_metadata(self.metadata, self.output_dir)

        # Log summary
        ts = self.metadata.get('timing_summary', {})
        logger.info(f"[DIAGNOSTICS] Run complete in {self.metadata['duration_seconds']}s")
        if ts:
            logger.info(f"  Avg gen: {ts.get('avg_wall_per_gen', '?')}s | "
                        f"Eval%: {ts.get('eval_pct', '?')}% | "
                        f"Fastest: {ts.get('fastest_gen', '?')}s | "
                        f"Slowest: {ts.get('slowest_gen', '?')}s")

        return ts
