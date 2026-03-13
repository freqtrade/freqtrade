#!/usr/bin/env python3
"""
Phase 1 Test Runner — Parallel Execution (7 cores)

Runs all Phase 1 test configs (P1A-P1H) in parallel using up to 7 worker
processes. Each run is independent and writes its own logs/output.

For ML runs (P1G, P1H): retrains the ML model with the appropriate label
mode before launching the GA run.

Usage:
    # Run all 8 configs with 7 cores:
    python genetic_algorithm/scripts/run_phase1_tests.py

    # Run specific configs:
    python genetic_algorithm/scripts/run_phase1_tests.py --runs A B C

    # Smoke test only (no actual GA runs):
    python genetic_algorithm/scripts/run_phase1_tests.py --smoke-only

    # Change parallelism:
    python genetic_algorithm/scripts/run_phase1_tests.py --workers 4

    # Skip ML pre-training (if models already exist):
    python genetic_algorithm/scripts/run_phase1_tests.py --skip-ml-pretrain
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

CONFIG_DIR = _PROJECT_ROOT / 'genetic_algorithm' / 'config'
LOG_DIR = _PROJECT_ROOT / 'genetic_algorithm' / 'logs'
OUTPUT_DIR = _PROJECT_ROOT / 'genetic_algorithm' / 'output' / 'phase1_tests'

# Test matrix: label → (config filename, description, ml_pretrain_label_mode)
TEST_MATRIX: Dict[str, Dict[str, Any]] = {
    'P1A': {
        'config': 'ga_config_phase1_test_A_control.yaml',
        'description': 'Control — ensemble, no Phase 1',
        'ml_pretrain': None,
    },
    'P1B': {
        'config': 'ga_config_phase1_test_B_autocal_single.yaml',
        'description': 'Auto-calibrate single pair (BTC)',
        'ml_pretrain': None,
    },
    'P1C': {
        'config': 'ga_config_phase1_test_C_autocal_multi.yaml',
        'description': 'Auto-calibrate multi-pair (BTC+ETH)',
        'ml_pretrain': None,
    },
    'P1D': {
        'config': 'ga_config_phase1_test_D_ensemble_score.yaml',
        'description': 'Ensemble continuous score, no auto-cal',
        'ml_pretrain': None,
    },
    'P1E': {
        'config': 'ga_config_phase1_test_E_tf_sweep.yaml',
        'description': 'Timeframe sweep + auto-cal + quality report',
        'ml_pretrain': None,
    },
    'P1F': {
        'config': 'ga_config_phase1_test_F_full_stack.yaml',
        'description': 'Full stack — all features combined',
        'ml_pretrain': None,
    },
    'P1G': {
        'config': 'ga_config_phase1_test_G_ml_scoreband.yaml',
        'description': 'ML trainer with score_band labels',
        'ml_pretrain': 'score_band',
    },
    'P1H': {
        'config': 'ga_config_phase1_test_H_ml_advensemble.yaml',
        'description': 'ML trainer with advanced_ensemble labels',
        'ml_pretrain': 'advanced_ensemble',
    },
}

# ──────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────


def smoke_test_all(run_ids: List[str]) -> List[Tuple[str, bool, str]]:
    """
    Validate all specified configs without running GA.
    Returns list of (run_id, passed, message).
    """
    from genetic_algorithm.tools.phase1_diagnostics import smoke_test_config, smoke_test_import

    results = []

    # Import check
    ok, msg = smoke_test_import()
    results.append(('IMPORTS', ok, msg))
    if not ok:
        logger.error("Import check failed: %s", msg)

    for run_id in run_ids:
        entry = TEST_MATRIX.get(run_id)
        if not entry:
            results.append((run_id, False, f"Unknown run ID: {run_id}"))
            continue

        config_path = str(CONFIG_DIR / entry['config'])
        ok, msg = smoke_test_config(config_path)
        results.append((run_id, ok, msg))

    return results


# ──────────────────────────────────────────────────────────────────
# ML pre-training
# ──────────────────────────────────────────────────────────────────


def pretrain_ml_model(
    label_mode: str,
    config_path: str,
    run_id: str,
) -> Tuple[bool, str]:
    """
    Retrain the ML regime model with the specified label mode.
    Runs as a subprocess.
    """
    log_file = LOG_DIR / f'ml_pretrain_{run_id}.log'

    cmd = [
        sys.executable, '-m', 'genetic_algorithm.ml.train_regime',
        '--config', config_path,
        '--pairs', 'BTC/USDT',
        '--timeframe', '4h',
        '--timerange', '20230101-20260228',
        '--label-mode', label_mode,
    ]

    logger.info("[%s] Pre-training ML model with label_mode=%s", run_id, label_mode)
    try:
        with open(log_file, 'w') as lf:
            result = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
                timeout=600,  # 10 min timeout for training
            )
        if result.returncode == 0:
            return True, f"ML pretrain OK (label_mode={label_mode})"
        else:
            return False, f"ML pretrain FAILED (exit={result.returncode}), see {log_file}"
    except subprocess.TimeoutExpired:
        return False, f"ML pretrain TIMEOUT after 600s"
    except Exception as e:
        return False, f"ML pretrain ERROR: {e}"


# ──────────────────────────────────────────────────────────────────
# Single run executor (runs in subprocess)
# ──────────────────────────────────────────────────────────────────


def run_single_config(
    run_id: str,
    config_filename: str,
    description: str,
    output_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a single GA config as a subprocess.
    Returns a result dict with timing, exit code, etc.

    This function is designed to be called from ProcessPoolExecutor.
    """
    config_path = str(CONFIG_DIR / config_filename)
    log_file = str(LOG_DIR / f'ga_p1test_{run_id}.log')
    run_output_dir = str(OUTPUT_DIR / (output_subdir or run_id))

    # Set output_dir in environment so the GA can find it
    env = os.environ.copy()
    env['GA_OUTPUT_DIR'] = run_output_dir

    result = {
        'run_id': run_id,
        'config': config_filename,
        'description': description,
        'config_path': config_path,
        'log_file': log_file,
        'output_dir': run_output_dir,
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration_seconds': None,
        'exit_code': None,
        'status': 'STARTED',
        'error': None,
    }

    cmd = [
        sys.executable, '-m', 'genetic_algorithm.run_ga',
        '--config', config_path,
        '--no-interactive',
        '--yes',
    ]

    start = time.monotonic()

    try:
        with open(log_file, 'w') as lf:
            # Write header
            lf.write(f"# Phase 1 Test: {run_id} — {description}\n")
            lf.write(f"# Config: {config_path}\n")
            lf.write(f"# Started: {result['start_time']}\n")
            lf.write(f"# Command: {' '.join(cmd)}\n")
            lf.write("=" * 70 + "\n\n")
            lf.flush()

            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
                env=env,
                timeout=3600,  # 1 hour timeout per run
            )
            result['exit_code'] = proc.returncode

    except subprocess.TimeoutExpired:
        result['exit_code'] = -1
        result['error'] = 'TIMEOUT after 3600s'
        result['status'] = 'TIMEOUT'
    except Exception as e:
        result['exit_code'] = -2
        result['error'] = str(e)
        result['status'] = 'ERROR'

    elapsed = time.monotonic() - start
    result['end_time'] = datetime.now().isoformat()
    result['duration_seconds'] = round(elapsed, 1)

    if result['exit_code'] == 0:
        result['status'] = 'PASS'
    elif result['status'] == 'STARTED':
        result['status'] = f"FAIL (exit={result['exit_code']})"

    # Save runner metadata in output dir
    try:
        out_dir = Path(run_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / 'runner_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    except Exception:
        pass

    return result


# ──────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────


def print_summary(results: List[Dict[str, Any]]):
    """Print a tabular summary of all run results."""
    print("\n" + "=" * 78)
    print("  PHASE 1 TEST SUITE — RESULTS SUMMARY")
    print("=" * 78)

    total_time = sum(r.get('duration_seconds', 0) or 0 for r in results)
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    failed = len(results) - passed

    print(f"\n  Total runs: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print()

    # Table
    header = f"  {'Run':<6} {'Status':<16} {'Time':>8} {'Description':<45}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in results:
        dur = r.get('duration_seconds')
        dur_str = f"{dur:.0f}s" if dur else "—"
        status = r.get('status', '?')
        if status == 'PASS':
            status_str = '✓ PASS'
        else:
            status_str = f'✗ {status}'

        print(
            f"  {r['run_id']:<6} {status_str:<16} {dur_str:>8} "
            f"{r.get('description', ''):<45}"
        )

    print()

    # Log file locations
    print("  Log files:")
    for r in results:
        print(f"    {r['run_id']}: {r.get('log_file', '?')}")

    print("\n" + "=" * 78)


def run_comparison_report(results: List[Dict[str, Any]]):
    """Run the Phase 1 diagnostics comparison after all tests complete."""
    try:
        from genetic_algorithm.tools.phase1_diagnostics import (
            Phase1Comparator, RunResult, find_run_dirs,
        )

        dirs = find_run_dirs(OUTPUT_DIR)
        if not dirs:
            print("\n  No output directories found for comparison report.")
            return

        run_results = [RunResult(d) for d in dirs]
        comparator = Phase1Comparator(run_results)
        report = comparator.full_report()
        print(report)

        # Save JSON report
        report_path = OUTPUT_DIR / 'phase1_comparison.json'
        with open(report_path, 'w') as f:
            json.dump(comparator.to_json(), f, indent=2, default=str)
        print(f"\n  JSON comparison report: {report_path}")

    except Exception as e:
        logger.warning("Could not generate comparison report: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Test Runner — parallel execution of P1A-P1H configs",
    )
    parser.add_argument(
        '--runs', nargs='*', default=None,
        help='Specific runs to execute (e.g. P1A P1B). Default: all.',
    )
    parser.add_argument(
        '--workers', type=int, default=7,
        help='Number of parallel worker processes (default: 7)',
    )
    parser.add_argument(
        '--smoke-only', action='store_true',
        help='Only run smoke tests, do not execute GA runs.',
    )
    parser.add_argument(
        '--skip-smoke', action='store_true',
        help='Skip smoke tests before running.',
    )
    parser.add_argument(
        '--skip-ml-pretrain', action='store_true',
        help='Skip ML model pre-training for P1G/P1H.',
    )
    parser.add_argument(
        '--skip-comparison', action='store_true',
        help='Skip comparison report after runs complete.',
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help='Run sequentially instead of in parallel (for debugging).',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable DEBUG logging.',
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure dirs exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which runs
    if args.runs:
        # Normalize: accept both "A" and "P1A"
        run_ids = []
        for r in args.runs:
            if r.startswith('P1'):
                run_ids.append(r)
            else:
                run_ids.append(f'P1{r}')
    else:
        run_ids = list(TEST_MATRIX.keys())

    # Validate run IDs
    invalid = [r for r in run_ids if r not in TEST_MATRIX]
    if invalid:
        print(f"Unknown run IDs: {invalid}")
        print(f"Available: {list(TEST_MATRIX.keys())}")
        return 1

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Phase 1 Feature Test Suite                               ║")
    print(f"║   Runs: {', '.join(run_ids):<52}║")
    print(f"║   Workers: {args.workers:<50}║")
    print(f"║   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<50}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # === SMOKE TEST ===
    if not args.skip_smoke:
        print("── Phase 0: Smoke Tests ──")
        smoke_results = smoke_test_all(run_ids)

        all_pass = True
        for label, ok, msg in smoke_results:
            status = '✓' if ok else '✗'
            print(f"  {status} {label}: {msg}")
            if not ok:
                all_pass = False

        if not all_pass:
            print("\n  ⚠ Some smoke tests FAILED. Review above and fix before running.")
            if args.smoke_only:
                return 1
            # Continue anyway — let the actual runs fail with better error messages
            print("  Continuing with actual runs despite smoke test failures...")
        else:
            print("\n  All smoke tests passed ✓")

        if args.smoke_only:
            return 0

    # === ML PRE-TRAINING ===
    if not args.skip_ml_pretrain:
        ml_runs = [r for r in run_ids if TEST_MATRIX[r].get('ml_pretrain')]
        if ml_runs:
            print("\n── Phase 0.5: ML Model Pre-Training ──")
            for run_id in ml_runs:
                entry = TEST_MATRIX[run_id]
                config_path = str(CONFIG_DIR / entry['config'])
                ok, msg = pretrain_ml_model(
                    entry['ml_pretrain'], config_path, run_id,
                )
                status = '✓' if ok else '✗'
                print(f"  {status} {run_id}: {msg}")
                if not ok:
                    print(f"    ⚠ ML pre-training failed for {run_id}, "
                          f"GA run may use stale model")

    # === MAIN RUNS ===
    print(f"\n── Phase 1: Running {len(run_ids)} GA Tests ──")

    suite_start = time.monotonic()

    if args.sequential:
        # Sequential execution
        results = []
        for run_id in run_ids:
            entry = TEST_MATRIX[run_id]
            print(f"\n  [{run_id}] Starting: {entry['description']}")
            result = run_single_config(
                run_id=run_id,
                config_filename=entry['config'],
                description=entry['description'],
                output_subdir=run_id,
            )
            results.append(result)
            print(f"  [{run_id}] {result['status']} ({result.get('duration_seconds', 0):.0f}s)")
    else:
        # Parallel execution with ProcessPoolExecutor
        results = []
        futures = {}

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for run_id in run_ids:
                entry = TEST_MATRIX[run_id]
                future = executor.submit(
                    run_single_config,
                    run_id=run_id,
                    config_filename=entry['config'],
                    description=entry['description'],
                    output_subdir=run_id,
                )
                futures[future] = run_id
                print(f"  → Submitted {run_id}: {entry['description']}")

            print(f"\n  Waiting for {len(futures)} runs to complete...")
            print()

            for future in as_completed(futures):
                run_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    dur = result.get('duration_seconds', 0)
                    status = result.get('status', '?')
                    print(f"  ◆ {run_id} completed: {status} ({dur:.0f}s)")
                except Exception as e:
                    results.append({
                        'run_id': run_id,
                        'status': f'EXCEPTION: {e}',
                        'exit_code': -3,
                        'duration_seconds': 0,
                        'description': TEST_MATRIX[run_id]['description'],
                        'log_file': str(LOG_DIR / f'ga_p1test_{run_id}.log'),
                    })
                    print(f"  ◆ {run_id} EXCEPTION: {e}")

    suite_elapsed = time.monotonic() - suite_start

    # Sort results by run_id for consistent display
    results.sort(key=lambda r: r.get('run_id', ''))

    # === SUMMARY ===
    print_summary(results)
    print(f"  Suite wall-clock: {suite_elapsed:.0f}s ({suite_elapsed/60:.1f}m)")

    # Save suite summary
    suite_meta = {
        'started': datetime.now().isoformat(),
        'total_runs': len(results),
        'workers': args.workers,
        'sequential': args.sequential,
        'suite_duration_seconds': round(suite_elapsed, 1),
        'results': results,
    }
    summary_path = OUTPUT_DIR / 'suite_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(suite_meta, f, indent=2, default=str)
    print(f"  Suite summary: {summary_path}")

    # === COMPARISON REPORT ===
    if not args.skip_comparison:
        print("\n── Phase 2: Comparison Report ──")
        run_comparison_report(results)

    return 0


if __name__ == '__main__':
    sys.exit(main())
