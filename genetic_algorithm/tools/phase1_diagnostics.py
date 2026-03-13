#!/usr/bin/env python3
"""
Phase 1 Diagnostics Collector & Comparison Tool

Collects, parses, and compares diagnostic data from Phase 1 test runs.
Produces a consolidated comparison report (text + JSON) for evaluating
how different Phase 1 feature combinations perform.

Usage:
    # After runs complete, compare all results:
    python -m genetic_algorithm.tools.phase1_diagnostics \\
        --results-dir genetic_algorithm/output/phase1_tests \\
        --report-out genetic_algorithm/output/phase1_comparison.json

    # Compare specific runs:
    python -m genetic_algorithm.tools.phase1_diagnostics \\
        --runs P1A P1B P1C \\
        --results-dir genetic_algorithm/output/phase1_tests
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Data collection helpers
# ──────────────────────────────────────────────────────────────────


def find_run_dirs(base_dir: Path, run_ids: Optional[List[str]] = None) -> List[Path]:
    """
    Discover run directories under base_dir.

    Each run dir is expected to contain at least one of:
    - run_metadata.json  (from RunDiagnostics)
    - phase1_quality_report.json (from _phase1_quality_report)
    - generation_stats.csv (from GenerationCSVWriter)
    """
    base = Path(base_dir)
    if not base.exists():
        logger.warning("Results directory does not exist: %s", base)
        return []

    dirs = []
    # Try subdirectories
    for child in sorted(base.iterdir()):
        if child.is_dir():
            if run_ids and child.name not in run_ids:
                continue
            # Check if it has any expected output files
            has_meta = (child / 'run_metadata.json').exists()
            has_qr = (child / 'phase1_quality_report.json').exists()
            has_csv = (child / 'generation_stats.csv').exists()
            if has_meta or has_qr or has_csv:
                dirs.append(child)

    # If no subdirs found, check base itself
    if not dirs:
        has_meta = (base / 'run_metadata.json').exists()
        has_qr = (base / 'phase1_quality_report.json').exists()
        if has_meta or has_qr:
            dirs.append(base)

    return dirs


def load_json_safe(path: Path) -> Optional[Dict]:
    """Load JSON file, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("Could not load %s: %s", path, e)
        return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Load CSV as list of row dicts."""
    try:
        with open(path, newline='') as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logger.debug("Could not load CSV %s: %s", path, e)
        return []


# ──────────────────────────────────────────────────────────────────
# Per-run data aggregation
# ──────────────────────────────────────────────────────────────────


class RunResult:
    """Parsed results from a single Phase 1 test run."""

    def __init__(self, run_dir: Path, run_label: str = ''):
        self.run_dir = Path(run_dir)
        self.label = run_label or run_dir.name
        self.metadata: Optional[Dict] = None
        self.quality_report: Optional[Dict] = None
        self.generation_rows: List[Dict] = []
        self.config_yaml_path: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.log_tail: str = ''
        self._load()

    def _load(self):
        self.metadata = load_json_safe(self.run_dir / 'run_metadata.json')
        self.quality_report = load_json_safe(self.run_dir / 'phase1_quality_report.json')
        self.generation_rows = load_csv_rows(self.run_dir / 'generation_stats.csv')

        # Check for runner metadata (written by our runner script)
        runner_meta = load_json_safe(self.run_dir / 'runner_metadata.json')
        if runner_meta:
            self.exit_code = runner_meta.get('exit_code')
            self.config_yaml_path = runner_meta.get('config_path')

    # ── Derived metrics ──

    @property
    def success(self) -> bool:
        return self.exit_code == 0 if self.exit_code is not None else bool(self.metadata)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.metadata:
            return self.metadata.get('duration_seconds')
        return None

    @property
    def n_generations(self) -> int:
        return len(self.generation_rows)

    @property
    def best_fitness(self) -> Optional[float]:
        if not self.generation_rows:
            return None
        vals = []
        for r in self.generation_rows:
            v = r.get('best_fitness', '')
            if v != '':
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        return max(vals) if vals else None

    @property
    def avg_fitness_final(self) -> Optional[float]:
        if not self.generation_rows:
            return None
        last = self.generation_rows[-1]
        v = last.get('avg_fitness', '')
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @property
    def best_profit(self) -> Optional[float]:
        if not self.generation_rows:
            return None
        vals = []
        for r in self.generation_rows:
            v = r.get('best_profit', '')
            if v != '':
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        return max(vals) if vals else None

    @property
    def best_sharpe(self) -> Optional[float]:
        if not self.generation_rows:
            return None
        vals = []
        for r in self.generation_rows:
            v = r.get('best_sharpe', '')
            if v != '':
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        return max(vals) if vals else None

    @property
    def calibration_bands(self) -> Optional[Dict]:
        if self.quality_report:
            return self.quality_report.get('bands')
        return None

    @property
    def band_coverage(self) -> Optional[Dict]:
        if self.quality_report:
            return self.quality_report.get('band_coverage')
        return None

    @property
    def score_distribution(self) -> Optional[Dict]:
        if self.quality_report:
            return self.quality_report.get('score_distribution')
        return None

    @property
    def regime_statistics(self) -> Optional[Dict]:
        if self.quality_report:
            return self.quality_report.get('regime_statistics')
        return None

    @property
    def cross_pair_consistency(self) -> Optional[Dict]:
        if self.quality_report:
            return self.quality_report.get('cross_pair_consistency')
        return None

    @property
    def method(self) -> str:
        if self.quality_report:
            return self.quality_report.get('method', '?')
        return '?'

    def summary_dict(self) -> Dict[str, Any]:
        """Compact summary for comparison table."""
        return {
            'label': self.label,
            'success': self.success,
            'method': self.method,
            'duration_s': self.duration_seconds,
            'n_generations': self.n_generations,
            'best_fitness': self.best_fitness,
            'avg_fitness_final': self.avg_fitness_final,
            'best_profit': self.best_profit,
            'best_sharpe': self.best_sharpe,
            'bands': self.calibration_bands,
            'band_coverage': self.band_coverage,
            'score_mean': (self.score_distribution or {}).get('mean'),
            'score_std': (self.score_distribution or {}).get('std'),
            'score_skew': (self.score_distribution or {}).get('skew'),
            'regime_stats': self.regime_statistics,
            'cross_pair': self.cross_pair_consistency,
        }


# ──────────────────────────────────────────────────────────────────
# Comparison report generator
# ──────────────────────────────────────────────────────────────────


class Phase1Comparator:
    """
    Compares multiple Phase 1 test runs and produces a consolidated report.
    """

    def __init__(self, results: List[RunResult]):
        self.results = results

    def comparison_table(self) -> str:
        """
        Format a text comparison table across all runs.

        Columns: Label | Status | Method | Duration | BestFit | AvgFit | Profit | Sharpe | Bands | Coverage
        """
        lines = []
        header = (
            f"{'Label':<20} {'OK':>3} {'Method':<18} {'Dur(s)':>7} "
            f"{'BestFit':>8} {'AvgFit':>8} {'Profit':>8} {'Sharpe':>7} "
            f"{'BullBand':>9} {'BearBand':>9} "
            f"{'Bull%':>6} {'Side%':>6} {'Bear%':>6}"
        )
        lines.append(header)
        lines.append('─' * len(header))

        for r in self.results:
            bands = r.calibration_bands or {}
            cov = r.band_coverage or {}

            bmin = bands.get('bullish_min', '?')
            bmax = bands.get('bearish_max', '?')
            bull_pct = cov.get('bullish_pct', '?')
            side_pct = cov.get('sideways_pct', '?')
            bear_pct = cov.get('bearish_pct', '?')

            def _fmt(v, fmt='%.3f'):
                if v is None:
                    return '   —'
                try:
                    return fmt % v
                except (TypeError, ValueError):
                    return str(v)[:8]

            line = (
                f"{r.label:<20} "
                f"{'✓' if r.success else '✗':>3} "
                f"{r.method:<18} "
                f"{_fmt(r.duration_seconds, '%.1f'):>7} "
                f"{_fmt(r.best_fitness):>8} "
                f"{_fmt(r.avg_fitness_final):>8} "
                f"{_fmt(r.best_profit, '%.2f'):>8} "
                f"{_fmt(r.best_sharpe, '%.2f'):>7} "
                f"{_fmt(bmin, '%.3f') if isinstance(bmin, (int, float)) else str(bmin):>9} "
                f"{_fmt(bmax, '%.3f') if isinstance(bmax, (int, float)) else str(bmax):>9} "
                f"{_fmt(bull_pct, '%.1f') if isinstance(bull_pct, (int, float)) else str(bull_pct):>6} "
                f"{_fmt(side_pct, '%.1f') if isinstance(side_pct, (int, float)) else str(side_pct):>6} "
                f"{_fmt(bear_pct, '%.1f') if isinstance(bear_pct, (int, float)) else str(bear_pct):>6}"
            )
            lines.append(line)

        return '\n'.join(lines)

    def regime_comparison(self) -> str:
        """Compare regime statistics across runs."""
        lines = ['', '═══ Regime Statistics Comparison ═══', '']
        for regime in ('bullish', 'sideways', 'bearish'):
            lines.append(f"── {regime.upper()} ──")
            header = (
                f"  {'Label':<20} {'Seg':>4} {'Bars':>6} {'Return':>8} "
                f"{'Volatil':>8} {'Sharpe':>8} {'Confid':>7} {'Days':>6}"
            )
            lines.append(header)

            for r in self.results:
                rs = (r.regime_statistics or {}).get(regime, {})
                if not rs or rs.get('n_segments', 0) == 0:
                    lines.append(f"  {r.label:<20}  — no data —")
                    continue

                lines.append(
                    f"  {r.label:<20} "
                    f"{rs.get('n_segments', 0):>4} "
                    f"{rs.get('total_bars', 0):>6} "
                    f"{rs.get('mean_return', 0):>8.4f} "
                    f"{rs.get('mean_volatility', 0):>8.4f} "
                    f"{rs.get('sharpe_like', 0):>8.4f} "
                    f"{rs.get('avg_confidence', 0):>7.3f} "
                    f"{rs.get('total_days', 0):>6}"
                )
            lines.append('')
        return '\n'.join(lines)

    def cross_pair_comparison(self) -> str:
        """Compare cross-pair consistency across runs."""
        lines = ['', '═══ Cross-Pair Consistency ═══', '']
        any_data = False
        for r in self.results:
            cp = r.cross_pair_consistency
            if not cp:
                continue
            any_data = True
            lines.append(f"  {r.label}:")
            for pair, stats in cp.items():
                lines.append(
                    f"    {pair}: consistency={stats.get('consistency', 0):.1%} "
                    f"({stats.get('matched_segments', 0)}/{stats.get('total_segments', 0)} segments)"
                )
        if not any_data:
            lines.append('  (no multi-pair runs)')
        return '\n'.join(lines)

    def fitness_progression_comparison(self) -> str:
        """Compare fitness progression across generations for each run."""
        lines = ['', '═══ Fitness Progression ═══', '']

        for r in self.results:
            if not r.generation_rows:
                lines.append(f"  {r.label}: (no generation data)")
                continue

            fitnesses = []
            for row in r.generation_rows:
                v = row.get('best_fitness', '')
                try:
                    fitnesses.append(float(v))
                except (ValueError, TypeError):
                    fitnesses.append(None)

            if fitnesses:
                fit_str = ' → '.join(
                    f"{f:.3f}" if f is not None else "?" for f in fitnesses
                )
                lines.append(f"  {r.label}: {fit_str}")
        return '\n'.join(lines)

    def score_distribution_comparison(self) -> str:
        """Compare score distributions across runs."""
        lines = ['', '═══ Score Distribution Comparison ═══', '']
        header = (
            f"  {'Label':<20} {'Mean':>7} {'Std':>7} "
            f"{'Skew':>7} {'Kurt':>7}"
        )
        lines.append(header)

        for r in self.results:
            sd = r.score_distribution
            if not sd:
                lines.append(f"  {r.label:<20}  — no score data —")
                continue
            lines.append(
                f"  {r.label:<20} "
                f"{sd.get('mean', 0):>7.4f} "
                f"{sd.get('std', 0):>7.4f} "
                f"{sd.get('skew', 0):>7.4f} "
                f"{sd.get('kurtosis', 0):>7.4f}"
            )
        return '\n'.join(lines)

    def full_report(self) -> str:
        """Generate the complete comparison report."""
        parts = [
            '╔══════════════════════════════════════════════════════════════╗',
            '║   Phase 1 Feature Test — Comparison Report                  ║',
            f'║   Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S"):<44} ║',
            f'║   Runs compared: {len(self.results):<41} ║',
            '╚══════════════════════════════════════════════════════════════╝',
            '',
            '═══ Summary Comparison Table ═══',
            '',
            self.comparison_table(),
            self.score_distribution_comparison(),
            self.regime_comparison(),
            self.cross_pair_comparison(),
            self.fitness_progression_comparison(),
            '',
            '═══ End of Report ═══',
        ]
        return '\n'.join(parts)

    def to_json(self) -> Dict[str, Any]:
        """Structured JSON report for programmatic analysis."""
        return {
            'generated': datetime.now().isoformat(),
            'num_runs': len(self.results),
            'runs': [r.summary_dict() for r in self.results],
            'comparisons': {
                'fitness': {
                    r.label: {
                        'best': r.best_fitness,
                        'avg_final': r.avg_fitness_final,
                        'profit': r.best_profit,
                        'sharpe': r.best_sharpe,
                    }
                    for r in self.results
                },
                'calibration': {
                    r.label: {
                        'bands': r.calibration_bands,
                        'coverage': r.band_coverage,
                        'score_dist': r.score_distribution,
                    }
                    for r in self.results
                },
                'regimes': {
                    r.label: r.regime_statistics
                    for r in self.results
                },
            },
        }


# ──────────────────────────────────────────────────────────────────
# Smoke test validation
# ──────────────────────────────────────────────────────────────────


def smoke_test_config(config_path: str) -> Tuple[bool, str]:
    """
    Validate a GA config file can be loaded and Phase 1 data step
    initializes without crashing.

    Returns (success: bool, message: str)
    """
    import yaml as _yaml
    try:
        path = Path(config_path)
        if not path.exists():
            return False, f"Config not found: {path}"

        with open(path) as f:
            config = _yaml.safe_load(f)

        if not config:
            return False, "Config is empty or invalid YAML"

        # Validate required sections
        required = ['genetic_algorithm', 'backtesting', 'island_model']
        missing = [s for s in required if s not in config]
        if missing:
            return False, f"Missing required sections: {missing}"

        # Validate island_model structure
        im = config.get('island_model', {})
        if not im.get('enabled'):
            return False, "island_model.enabled is False"

        islands = im.get('islands', [])
        if len(islands) < 2:
            return False, f"Need >= 2 islands, got {len(islands)}"

        rd = im.get('regime_detection', {})
        if not rd.get('pair'):
            return False, "No regime_detection.pair specified"

        # Validate Phase 1 config
        p1 = rd.get('phase1', {})
        if p1.get('auto_calibrate'):
            cal_pairs = p1.get('calibration_pairs', [])
            if not cal_pairs:
                return False, "auto_calibrate=true but no calibration_pairs"

        # Validate storage paths are writeable intent
        storage = config.get('storage', {})
        if not storage.get('database'):
            return False, "No storage.database path"

        # Validate backtesting pairs exist
        bt = config.get('backtesting', {})
        pairs = bt.get('pairs', [])
        if not pairs:
            return False, "No backtesting pairs specified"

        # Check data directory
        datadir = Path(bt.get('datadir', 'user_data/data/binance'))
        if not datadir.exists():
            return False, f"Data directory does not exist: {datadir}"

        # Check at least one pair has data files
        pair_found = False
        for pair in pairs:
            pair_file = pair.replace('/', '_')
            possible = list(datadir.glob(f"*{pair_file}*"))
            if possible:
                pair_found = True
                break
        if not pair_found:
            return False, f"No data files found for pairs: {pairs}"

        return True, "Config validation passed"

    except Exception as e:
        return False, f"Validation error: {e}"


def smoke_test_import() -> Tuple[bool, str]:
    """
    Verify that the core GA modules can be imported without error.
    Returns (success, message).
    """
    errors = []
    modules = [
        ('genetic_algorithm.core.island_model', 'IslandModelEvolution'),
        ('genetic_algorithm.utils.regime_detector', 'RegimeDetector'),
        ('genetic_algorithm.tools.calibrate_bands', 'BandCalibrator'),
        ('genetic_algorithm.utils.run_diagnostics', 'RunDiagnostics'),
    ]
    for mod_name, cls_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            if not hasattr(mod, cls_name):
                errors.append(f"{mod_name}: missing {cls_name}")
        except Exception as e:
            errors.append(f"{mod_name}: import failed — {e}")

    if errors:
        return False, '; '.join(errors)
    return True, "All imports OK"


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 1 diagnostics: collect, compare, and smoke-test GA runs.",
    )
    sub = parser.add_subparsers(dest='command', help='sub-command')

    # compare
    cmp = sub.add_parser('compare', help='Compare Phase 1 test run results')
    cmp.add_argument('--results-dir', type=str,
                     default='genetic_algorithm/output/phase1_tests',
                     help='Directory containing per-run output folders')
    cmp.add_argument('--runs', nargs='*', default=None,
                     help='Specific run folder names to compare (all if omitted)')
    cmp.add_argument('--report-out', type=str, default=None,
                     help='Save JSON report to this path')
    cmp.add_argument('--quiet', action='store_true',
                     help='Only output the comparison table (no headers)')

    # smoke
    smk = sub.add_parser('smoke', help='Smoke-test GA config files')
    smk.add_argument('configs', nargs='+',
                     help='Paths to YAML config files to validate')
    smk.add_argument('--import-check', action='store_true',
                     help='Also verify core module imports')

    return parser.parse_args(argv)


def cmd_compare(args):
    base = Path(args.results_dir)
    dirs = find_run_dirs(base, args.runs)

    if not dirs:
        print(f"No run results found in: {base}")
        return 1

    results = [RunResult(d) for d in dirs]
    comparator = Phase1Comparator(results)

    report_text = comparator.full_report()
    print(report_text)

    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(comparator.to_json(), f, indent=2, default=str)
        print(f"\nJSON report saved: {out_path}")

    return 0


def cmd_smoke(args):
    # Import check first
    if args.import_check:
        ok, msg = smoke_test_import()
        status = '✓' if ok else '✗'
        print(f"{status} Import check: {msg}")
        if not ok:
            return 1

    all_ok = True
    for cfg_path in args.configs:
        ok, msg = smoke_test_config(cfg_path)
        status = '✓' if ok else '✗'
        print(f"{status} {Path(cfg_path).name}: {msg}")
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    if args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'smoke':
        return cmd_smoke(args)
    else:
        print("Usage: python -m genetic_algorithm.tools.phase1_diagnostics {compare|smoke} ...")
        return 1


if __name__ == '__main__':
    sys.exit(main())
