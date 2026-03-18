"""
Parallel Evaluation Module

Provides parallel strategy evaluation using multiprocessing for
significant speedup on multi-core systems.

Features:
    - Per-backtest timeout to prevent pathological strategies from blocking workers
    - Automatic orphaned worker cleanup via atexit handlers
    - Walk-forward disabled in workers (use post-hoc WF validation on elites instead)

Usage:
    Enable in ga_config.yaml:
        parallel_evaluation:
            enabled: true
            num_workers: 4  # null for auto-detect
            backtest_timeout: 120  # seconds per individual
            
Benchmark results (8-core system, 50 strategies):
    - Sequential: ~174 seconds
    - Parallel (4 workers): ~49 seconds (3.5x speedup)
    - Parallel (8 workers): ~37 seconds (4.7x speedup)
"""

import atexit
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Global evaluator for worker processes (initialized once per worker)
_worker_evaluator = None
_worker_config = None

# Track active executors for cleanup
_active_executors: List['ProcessPoolExecutor'] = []


def _cleanup_executors():
    """Shutdown all active process pool executors on exit."""
    for executor in _active_executors:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _active_executors.clear()


atexit.register(_cleanup_executors)


def _kill_pool_processes(executor: ProcessPoolExecutor):
    """
    Force-kill any lingering worker processes owned by the given executor.
    
    After ``executor.shutdown()``, some worker processes may remain as
    zombies (especially when FreqTrade Backtesting holds resources).  This
    function reads the executor's internal ``_processes`` dict and sends
    SIGKILL to any that are still alive.
    
    Safe to call even if the executor was already fully shut down.
    """
    processes = getattr(executor, '_processes', None)
    if not processes:
        return
    
    killed = 0
    for pid, proc in list(processes.items()):
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
                killed += 1
        except Exception:
            # Process may have already exited
            pass
    
    if killed:
        logger.debug(f"[PARALLEL] Force-killed {killed} lingering worker process(es)")


def _pool_health_check_fn():
    """Module-level function for pool health checks (lambdas can't be pickled)."""
    return True


def _init_worker(config: Dict[str, Any]):
    """
    Initialize worker process with its own FitnessEvaluator.
    
    Called once when each worker starts. Creates a separate evaluator
    instance to avoid sharing state between processes.
    
    Walk-forward is always disabled in workers to prevent the N×W
    backtest explosion that causes deadlocks. Walk-forward validation
    should be applied post-hoc on elite candidates only.
    
    Args:
        config: Configuration dictionary
    """
    global _worker_evaluator, _worker_config
    
    # ── Silence worker logging FIRST (before any imports trigger log calls) ──
    # Strip ALL console StreamHandlers inherited from the parent process (fork)
    # and set levels so init messages from DirectBacktester etc.
    # never reach the terminal and corrupt the rich Live display.
    worker_log_level = getattr(
        logging, config.get('worker_log_level', 'WARNING').upper(), logging.WARNING
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(worker_log_level)
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    logging.getLogger('GeneticAlgorithm').setLevel(worker_log_level)
    logging.getLogger('freqtrade').setLevel(worker_log_level)
    
    # Import here to avoid circular imports and ensure each process has its own imports
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    
    # Disable walk-forward in worker processes to prevent N×W backtest explosion
    # Walk-forward validation is applied post-hoc on elites instead
    worker_config = dict(config)
    if 'walk_forward' in worker_config:
        worker_config['walk_forward'] = dict(worker_config['walk_forward'])
        worker_config['walk_forward']['enabled'] = False
    
    _worker_config = worker_config
    _worker_evaluator = FitnessEvaluator(worker_config)


def _evaluate_strategy_in_worker(
    strategy_gene_dict: Dict[str, Any],
    strategy_index: int,
    nsga2_mode: bool = False,
    objectives_config: List[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate a single strategy in a worker process.
    
    This function runs in a subprocess and uses the pre-initialized
    evaluator to backtest the strategy.
    
    Args:
        strategy_gene_dict: Serialized StrategyGene as dictionary
        strategy_index: Index of strategy in population (for tracking)
        nsga2_mode: Whether to extract objectives for NSGA-II
        objectives_config: NSGA-II objectives configuration
        
    Returns:
        Dictionary with evaluation results:
        {
            'index': strategy_index,
            'fitness': float,
            'metrics': dict,
            'objectives': list (if nsga2_mode),
            'success': bool,
            'error': str (if failed)
        }
    """
    global _worker_evaluator
    
    if _worker_evaluator is None:
        return {
            'index': strategy_index,
            'fitness': 0.0,
            'metrics': {},
            'success': False,
            'error': 'Worker not initialized'
        }
    
    try:
        # Import here to avoid issues with multiprocessing
        from genetic_algorithm.core.strategy_gene import StrategyGene
        from genetic_algorithm.core.nsga2 import extract_objectives_from_metrics
        
        # Reconstruct StrategyGene from dict
        strategy_gene = StrategyGene.from_dict(strategy_gene_dict)
        
        # Evaluate
        fitness, metrics = _worker_evaluator.evaluate(strategy_gene)
        
        result = {
            'index': strategy_index,
            'fitness': fitness,
            'metrics': metrics,
            'success': True
        }
        
        # Extract objectives for NSGA-II if needed
        if nsga2_mode and objectives_config:
            nsga2_min_trades = kwargs.get('nsga2_min_trades', 0)
            objectives = extract_objectives_from_metrics(
                metrics, objectives_config, min_trades=nsga2_min_trades
            )
            result['objectives'] = objectives
        
        return result
        
    except Exception as e:
        return {
            'index': strategy_index,
            'fitness': 0.0,
            'metrics': {'error': str(e)},
            'success': False,
            'error': str(e)
        }


# ── Walk-Forward Post-hoc Parallel Workers ──

# Worker-local WF evaluator (separate from the non-WF worker evaluator)
_wf_worker_evaluator = None


def _init_wf_worker(config: Dict[str, Any]):
    """
    Initialize a worker process with walk-forward ENABLED.
    
    Used for parallel WF post-hoc validation of elite candidates.
    Unlike _init_worker(), this keeps walk_forward.enabled = True.
    """
    global _wf_worker_evaluator
    
    # Silence worker logging FIRST (before imports trigger log calls)
    worker_log_level = getattr(
        logging, config.get('worker_log_level', 'WARNING').upper(), logging.WARNING
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(worker_log_level)
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    logging.getLogger('GeneticAlgorithm').setLevel(worker_log_level)
    logging.getLogger('freqtrade').setLevel(worker_log_level)
    
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    
    _wf_worker_evaluator = FitnessEvaluator(config)


def _evaluate_wf_in_worker(
    strategy_gene_dict: Dict[str, Any],
    candidate_index: int,
    individual_id: str,
) -> Dict[str, Any]:
    """
    Evaluate a single strategy with walk-forward validation in a worker process.
    
    Args:
        strategy_gene_dict: Serialized StrategyGene
        candidate_index: Index for result tracking
        individual_id: ID of the individual (for logging)
        
    Returns:
        Dict with WF evaluation results
    """
    global _wf_worker_evaluator
    
    if _wf_worker_evaluator is None:
        return {
            'index': candidate_index,
            'id': individual_id,
            'fitness': 0.0,
            'metrics': {},
            'success': False,
            'error': 'WF worker not initialized',
        }
    
    try:
        from genetic_algorithm.core.strategy_gene import StrategyGene
        
        strategy_gene = StrategyGene.from_dict(strategy_gene_dict)
        fitness, metrics = _wf_worker_evaluator.evaluate(strategy_gene)
        
        return {
            'index': candidate_index,
            'id': individual_id,
            'fitness': fitness,
            'metrics': metrics,
            'success': True,
        }
    except Exception as e:
        return {
            'index': candidate_index,
            'id': individual_id,
            'fitness': 0.0,
            'metrics': {'error': str(e)},
            'success': False,
            'error': str(e),
        }


def _evaluate_single_window_in_worker(
    strategy_gene_dict: Dict[str, Any],
    candidate_index: int,
    individual_id: str,
    window_timerange: str,
    window_index: int,
    window_role: str,
) -> Dict[str, Any]:
    """
    Evaluate a single strategy on a single timerange window.

    This is the FLAT parallelization approach: instead of running full
    walk-forward (W sequential backtests) inside each worker, the caller
    creates all (candidate × window) combinations and submits each as an
    independent single-backtest task.  Workers reuse the persistent non-WF
    pool initialised by ``_init_worker()``.

    Args:
        strategy_gene_dict: Serialized StrategyGene
        candidate_index: Which candidate this belongs to
        individual_id: Human-readable ID of the individual
        window_timerange: Timerange string for this window (e.g. '20240301-20240601')
        window_index: Sequential index of this window
        window_role: 'train' or 'val'

    Returns:
        Dict with single-window backtest results
    """
    global _worker_evaluator

    if _worker_evaluator is None:
        return {
            'candidate_index': candidate_index,
            'id': individual_id,
            'window_index': window_index,
            'window_role': window_role,
            'fitness': 0.0,
            'metrics': {},
            'success': False,
            'error': 'Worker not initialized',
        }

    try:
        from genetic_algorithm.core.strategy_gene import StrategyGene

        strategy_gene = StrategyGene.from_dict(strategy_gene_dict)

        # Generate code & run a single-window backtest via the worker evaluator
        strategy_code = _worker_evaluator.strategy_generator.generate_strategy_code(strategy_gene)
        name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"

        result = _worker_evaluator._backtest_with_timerange(
            strategy_code, name, window_timerange,
            strategy_max_open_trades=strategy_gene.max_open_trades
        )

        if result.success:
            metrics = _worker_evaluator._backtest_result_to_metrics(result)
            metrics['complexity'] = strategy_gene.calculate_complexity()
            fitness = _worker_evaluator.calculate_fitness(metrics, strategy_gene)
        else:
            fitness = 0.0
            metrics = {
                'profit': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
            }

        return {
            'candidate_index': candidate_index,
            'id': individual_id,
            'window_index': window_index,
            'window_role': window_role,
            'fitness': fitness,
            'metrics': metrics,
            'total_trades': result.total_trades if result.success else 0,
            'success': True,
        }
    except Exception as e:
        return {
            'candidate_index': candidate_index,
            'id': individual_id,
            'window_index': window_index,
            'window_role': window_role,
            'fitness': 0.0,
            'metrics': {'error': str(e)},
            'total_trades': 0,
            'success': False,
            'error': str(e),
        }


def parallel_walk_forward_validation(
    candidates: list,
    config: Dict[str, Any],
    num_workers: int,
    backtest_timeout: int = 300,
) -> int:
    """
    Run walk-forward validation on candidate individuals in parallel.
    
    Spawns WF-enabled workers to evaluate multiple candidates concurrently,
    replacing the old sequential loop. Updates individuals in-place.
    
    Args:
        candidates: List of Individual objects to validate
        config: Full GA configuration dictionary (with walk_forward.enabled=True)
        num_workers: Number of parallel workers
        backtest_timeout: Per-candidate timeout in seconds
        
    Returns:
        Number of successfully validated candidates
    """
    if not candidates:
        return 0
    
    start_time = time.time()
    validated = 0
    failed = 0
    
    # Limit workers to number of candidates (no point having idle workers)
    actual_workers = min(num_workers, len(candidates))
    
    logger.info(f"[WF-POSTHOC-PARALLEL] Validating {len(candidates)} candidates "
                f"with {actual_workers} workers...")
    
    # Prepare tasks
    tasks = []
    for i, ind in enumerate(candidates):
        tasks.append({
            'gene_dict': ind.strategy_gene.to_dict(),
            'index': i,
            'id': ind.id,
        })
    
    executor = ProcessPoolExecutor(
        max_workers=actual_workers,
        initializer=_init_wf_worker,
        initargs=(config,),
    )
    _active_executors.append(executor)
    
    try:
        futures = {
            executor.submit(
                _evaluate_wf_in_worker,
                task['gene_dict'],
                task['index'],
                task['id'],
            ): task['index']
            for task in tasks
        }
        
        # Total timeout = per-candidate timeout × number of "rounds" (batched by workers)
        total_timeout = backtest_timeout * (len(candidates) // actual_workers + 1)
        
        for future in as_completed(futures, timeout=total_timeout):
            try:
                result = future.result(timeout=backtest_timeout)
                idx = result['index']
                ind = candidates[idx]
                
                if result['success']:
                    original_fitness = ind.fitness
                    ind.set_fitness(result['fitness'], result['metrics'])
                    validated += 1
                    logger.debug(
                        f"[WF-POSTHOC] {result['id']}: {original_fitness:.4f} -> "
                        f"{result['fitness']:.4f} "
                        f"(gap={result['metrics'].get('train_val_gap', 0):.4f})"
                    )
                else:
                    failed += 1
                    logger.warning(f"[WF-POSTHOC] Failed {result['id']}: {result.get('error')}")
                    
            except FuturesTimeoutError:
                failed += 1
                logger.warning(f"[WF-POSTHOC] Candidate timed out after {backtest_timeout}s")
            except Exception as e:
                failed += 1
                logger.error(f"[WF-POSTHOC] Worker error: {e}")
                
    except FuturesTimeoutError:
        for future in futures:
            if not future.done():
                future.cancel()
        logger.warning("[WF-POSTHOC] Batch timeout reached")
    except BrokenProcessPool:
        logger.error("[WF-POSTHOC] Worker pool crashed during WF validation")
    finally:
        # Aggressive cleanup: WF pools are ephemeral (different initializer
        # than the main evaluation pool) so we want them fully torn down.
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        if executor in _active_executors:
            _active_executors.remove(executor)
        # Kill any lingering worker processes
        _kill_pool_processes(executor)
        # Reap any zombie child processes
        _reap_zombies()
    
    elapsed = time.time() - start_time
    logger.info(
        f"[WF-POSTHOC-PARALLEL] Done: {validated} validated, {failed} failed "
        f"in {elapsed:.1f}s"
    )
    
    return validated


def parallel_walk_forward_flat(
    candidates: list,
    config: Dict[str, Any],
    evaluator: 'ParallelEvaluator',
    backtest_timeout: int = 180,
) -> int:
    """
    Flat walk-forward parallelization: submit each (candidate × window) as
    an independent single-backtest task to the PERSISTENT worker pool.

    Instead of running full WF evaluations inside separate WF workers
    (which caused deadlocks due to N×W backtest explosion / memory
    exhaustion), this function:

    1. Creates WF windows once (they're the same for all candidates).
    2. For each candidate × window, submits a single-backtest task to
       the existing persistent pool (same ``_init_worker`` initializer).
    3. Collects all results and aggregates per-candidate WF fitness scores
       using the configured aggregation method.
    4. Updates candidate ``Individual.fitness`` and ``metrics`` in-place.

    This eliminates:
    - Ephemeral WF pool creation (expensive, leaked resources)
    - Per-worker W sequential backtests (deadlock root cause)
    - Memory exhaustion from N×W concurrent Backtesting instances

    Args:
        candidates: List of Individual objects to validate
        config: Full GA configuration dictionary
        evaluator: The existing persistent ParallelEvaluator
        backtest_timeout: Timeout per individual backtest (seconds)

    Returns:
        Number of successfully validated candidates
    """
    if not candidates:
        return 0

    start_time = time.time()
    wf_config = config.get('walk_forward', {})
    if not wf_config.get('enabled', False):
        return 0

    # ── Step 1: Create WF windows (shared across all candidates) ──
    from genetic_algorithm.evaluation.fitness import FitnessEvaluator
    from genetic_algorithm.utils.timerange import (
        create_walk_forward_windows, aggregate_validation_scores
    )

    # Build a temporary evaluator just to get the window creation logic
    temp_evaluator = FitnessEvaluator(config)
    original_timerange = config.get('backtesting', {}).get('timerange', '')
    effective_timerange = temp_evaluator.backtester.get_available_data_range()
    timerange_for_windows = effective_timerange or original_timerange

    adjusted = temp_evaluator._auto_adjust_walk_forward_params(timerange_for_windows)
    if adjusted is None:
        logger.warning("[WF-FLAT] Cannot create WF windows — data too short")
        return 0

    try:
        windows = create_walk_forward_windows(
            timerange=timerange_for_windows,
            train_days=adjusted['train_days'],
            validation_days=adjusted['validation_days'],
            step_days=adjusted['step_days'],
            mode=wf_config.get('mode', 'rolling'),
            embargo_days=wf_config.get('embargo_days', 0),
            max_windows=wf_config.get('max_windows', None),
        )
    except ValueError as e:
        logger.warning(f"[WF-FLAT] Window creation failed: {e}")
        return 0

    n_candidates = len(candidates)
    n_windows = len(windows)
    logger.info(
        f"[WF-FLAT] Submitting {n_candidates} candidates × {n_windows} windows "
        f"= {n_candidates * n_windows * 2} tasks (train+val) to persistent pool"
    )

    # ── Step 2: Submit all (candidate × window × role) tasks ──
    evaluator._check_pool_health()
    executor = evaluator._get_executor()

    futures = {}
    for c_idx, ind in enumerate(candidates):
        gene_dict = ind.strategy_gene.to_dict()
        for window in windows:
            # Submit TRAIN backtest
            ft = executor.submit(
                _evaluate_single_window_in_worker,
                gene_dict, c_idx, ind.id,
                window.train_timerange, window.window_index, 'train',
            )
            futures[ft] = (c_idx, window.window_index, 'train')

            # Submit VAL backtest
            fv = executor.submit(
                _evaluate_single_window_in_worker,
                gene_dict, c_idx, ind.id,
                window.val_timerange, window.window_index, 'val',
            )
            futures[fv] = (c_idx, window.window_index, 'val')

    # ── Step 3: Collect results ──
    # results[candidate_idx] = { window_idx: {'train': result, 'val': result} }
    from collections import defaultdict
    results: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

    total_timeout = backtest_timeout * (n_candidates * n_windows * 2 // max(evaluator.num_workers, 1) + 1)
    completed = 0
    errors = 0

    try:
        for future in as_completed(futures, timeout=total_timeout):
            try:
                result = future.result(timeout=backtest_timeout)
                c_idx = result['candidate_index']
                w_idx = result['window_index']
                role = result['window_role']
                results[c_idx][w_idx][role] = result
                completed += 1
            except FuturesTimeoutError:
                errors += 1
                logger.debug("[WF-FLAT] Single window task timed out")
            except Exception as e:
                errors += 1
                logger.debug(f"[WF-FLAT] Task error: {e}")
    except FuturesTimeoutError:
        logger.warning("[WF-FLAT] Batch timeout reached, cancelling remaining")
        for f in futures:
            if not f.done():
                f.cancel()
    except BrokenProcessPool:
        logger.error("[WF-FLAT] Worker pool crashed during flat WF — recycling")
        evaluator._shutdown_executor()
        return 0

    # ── Step 4: Aggregate per-candidate WF fitness ──
    aggregation_method = wf_config.get('aggregation', 'mean')
    gap_penalty_config = wf_config.get('gap_penalty', {})
    gap_penalty_enabled = gap_penalty_config.get('enabled', True)
    gap_threshold = gap_penalty_config.get('threshold', 0.1)
    gap_max_penalty = gap_penalty_config.get('max_penalty', 0.5)

    validated = 0
    for c_idx, ind in enumerate(candidates):
        window_results = results.get(c_idx, {})
        val_scores = []
        train_scores = []

        for w_idx in sorted(window_results.keys()):
            wr = window_results[w_idx]
            train_r = wr.get('train', {})
            val_r = wr.get('val', {})

            if not train_r.get('success') or train_r.get('total_trades', 0) == 0:
                continue  # Skip window if training failed

            train_scores.append(train_r.get('fitness', 0.0))

            if val_r.get('success') and val_r.get('total_trades', 0) > 0:
                val_scores.append(val_r.get('fitness', 0.0))
            else:
                val_scores.append(0.0)

        if not val_scores:
            continue  # All windows failed for this candidate

        # Aggregate validation scores
        if aggregation_method == 'weighted' and val_scores:
            n = len(val_scores)
            weights = [i / sum(range(1, n + 1)) for i in range(1, n + 1)]
            wf_fitness = aggregate_validation_scores(val_scores, method=aggregation_method, weights=weights)
        else:
            wf_fitness = aggregate_validation_scores(val_scores, method=aggregation_method)

        # Apply window failure penalty
        success_ratio = len([s for s in val_scores if s > 0]) / n_windows
        if success_ratio < 1.0:
            wf_fitness *= max(success_ratio, 0.3)

        # Apply train-val gap penalty
        avg_train = sum(train_scores) / len(train_scores) if train_scores else 0.0
        avg_val = sum(val_scores) / len(val_scores) if val_scores else 0.0
        gap = avg_train - avg_val

        if gap_penalty_enabled and gap > gap_threshold:
            excess = gap - gap_threshold
            penalty_factor = max(1.0 - gap_max_penalty, 1.0 - excess * 2.0)
            wf_fitness *= penalty_factor

        # Build aggregated metrics
        all_val_metrics = [
            window_results.get(w, {}).get('val', {}).get('metrics', {})
            for w in sorted(window_results.keys())
            if window_results.get(w, {}).get('val', {}).get('success')
        ]
        agg_metrics = _aggregate_flat_metrics(all_val_metrics)
        agg_metrics['walk_forward'] = True
        agg_metrics['num_windows'] = n_windows
        agg_metrics['successful_windows'] = len(val_scores)
        agg_metrics['avg_train_fitness'] = avg_train
        agg_metrics['avg_val_fitness'] = avg_val
        agg_metrics['train_val_gap'] = gap

        original_fitness = ind.fitness
        ind.set_fitness(wf_fitness, agg_metrics)
        validated += 1

        logger.debug(
            f"[WF-FLAT] {ind.id}: {original_fitness:.4f} -> {wf_fitness:.4f} "
            f"(gap={gap:.4f}, {len(val_scores)}/{n_windows} windows)"
        )

    elapsed = time.time() - start_time
    logger.info(
        f"[WF-FLAT] Done: {validated}/{n_candidates} validated, "
        f"{completed} tasks completed, {errors} errors in {elapsed:.1f}s"
    )
    return validated


def _aggregate_flat_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metric dicts from multiple windows into averages."""
    if not metrics_list:
        return {'profit': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'num_trades': 0}

    keys = ['profit', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'sortino_ratio',
            'profit_factor', 'num_trades']
    result = {}
    for key in keys:
        values = [m.get(key, 0) or 0 for m in metrics_list if key in m]
        result[key] = sum(values) / len(values) if values else 0
    result['num_trades'] = int(result.get('num_trades', 0))
    return result


def _reap_zombies():
    """Reap any zombie child processes to prevent accumulation."""
    try:
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            logger.debug(f"[PARALLEL] Reaped zombie process {pid}")
    except ChildProcessError:
        pass  # No child processes


# ── Parallel Parsimony Workers ──


def _evaluate_parsimony_candidate_in_worker(
    trial_gene_dict: Dict[str, Any],
    elite_index: int,
    candidate_index: int,
    candidate_kind: str,
    candidate_component_index: int,
) -> Dict[str, Any]:
    """
    Evaluate a single parsimony removal candidate in a worker process.

    Reuses the existing ``_worker_evaluator`` (non-WF) initialised by
    ``_init_worker``.

    Args:
        trial_gene_dict: Serialized StrategyGene with one component removed.
        elite_index: Which elite this candidate belongs to.
        candidate_index: Flat index across *all* candidates (for futures tracking).
        candidate_kind: 'indicator' | 'entry_condition' | 'exit_condition'.
        candidate_component_index: Index of the removed component.

    Returns:
        Dict with evaluation results keyed for the parsimony orchestrator.
    """
    global _worker_evaluator

    if _worker_evaluator is None:
        return {
            'elite_index': elite_index,
            'candidate_index': candidate_index,
            'kind': candidate_kind,
            'component_index': candidate_component_index,
            'fitness': 0.0,
            'metrics': {},
            'success': False,
            'error': 'Worker not initialized',
        }

    try:
        from genetic_algorithm.core.strategy_gene import StrategyGene

        strategy_gene = StrategyGene.from_dict(trial_gene_dict)
        fitness, metrics = _worker_evaluator.evaluate(strategy_gene)

        return {
            'elite_index': elite_index,
            'candidate_index': candidate_index,
            'kind': candidate_kind,
            'component_index': candidate_component_index,
            'fitness': fitness,
            'metrics': metrics,
            'success': True,
        }
    except Exception as e:
        return {
            'elite_index': elite_index,
            'candidate_index': candidate_index,
            'kind': candidate_kind,
            'component_index': candidate_component_index,
            'fitness': 0.0,
            'metrics': {'error': str(e)},
            'success': False,
            'error': str(e),
        }


def parallel_parsimony(
    elites: list,
    config: Dict[str, Any],
    ga_config: Dict[str, Any],
    num_workers: int,
    backtest_timeout: int = 120,
    evaluator: Optional['ParallelEvaluator'] = None,
) -> int:
    """
    Apply parsimony pressure to elites with parallel candidate evaluation.

    Instead of evaluating removal candidates one-by-one, this function
    builds *all* trial genes for *all* elites in the main process (cheap),
    submits them to a ``ProcessPoolExecutor`` in one batch, collects
    results, and for each elite picks the *best* acceptable removal
    (lowest relative fitness drop ≤ epsilon).

    Supports ``max_removals > 1`` by looping: after each round of accepted
    removals the candidate set is regenerated from the simplified genes and
    a new batch is submitted.

    Args:
        elites: List of ``Individual`` objects (mutated in-place on success).
        config: Parsimony config section (``epsilon``, ``max_removals``).
        ga_config: Full GA configuration dictionary (passed to workers).
        num_workers: Number of parallel workers.
        backtest_timeout: Per-candidate timeout in seconds.

    Returns:
        Total number of components removed across all elites.
    """
    from genetic_algorithm.core.parsimony import _build_removal_candidates, _apply_removal

    epsilon = config.get('epsilon', 0.02)
    max_removals = config.get('max_removals', 1)

    # Filter elites that are eligible for simplification
    eligible: List[Tuple[int, Any]] = []
    for i, ind in enumerate(elites):
        base_fitness = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
        if base_fitness is not None and base_fitness > 0:
            eligible.append((i, ind))

    if not eligible:
        return 0

    total_removed = 0
    start_time = time.time()

    # Track current state per elite: (current_gene, current_fitness)
    elite_state: Dict[int, Tuple[Any, float]] = {}
    for idx, ind in eligible:
        base = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
        elite_state[idx] = (ind.strategy_gene.copy(), base)

    for removal_round in range(max_removals):
        # Phase 1 (main process): build trial genes for every eligible elite
        tasks: List[Dict[str, Any]] = []
        # Map: flat_index -> (elite_idx, kind, comp_idx, trial_gene)
        task_meta: Dict[int, Tuple[int, str, int, Any]] = {}
        flat_idx = 0

        for elite_idx, (current_gene, current_fitness) in elite_state.items():
            candidates = _build_removal_candidates(current_gene)
            if not candidates:
                continue

            for kind, comp_idx in candidates:
                trial = _apply_removal(current_gene, kind, comp_idx)
                if trial is None:
                    continue

                # Assign unique identity to avoid filename collisions
                # when multiple trials for the same elite run in parallel.
                # Use large offset to guarantee no overlap with normal IDs.
                trial.generation = current_gene.generation
                trial.individual_id = 9000 + flat_idx  # parsimony namespace: 9000+

                task_meta[flat_idx] = (elite_idx, kind, comp_idx, trial)
                tasks.append({
                    'trial_dict': trial.to_dict(),
                    'elite_index': elite_idx,
                    'candidate_index': flat_idx,
                    'kind': kind,
                    'component_index': comp_idx,
                })
                flat_idx += 1

        if not tasks:
            break  # nothing left to simplify

        actual_workers = min(num_workers, len(tasks))
        logger.info(
            f"[PARSIMONY-PARALLEL] Round {removal_round + 1}: "
            f"{len(tasks)} candidates across {len(elite_state)} elites, "
            f"{actual_workers} workers"
        )

        # Phase 2: submit all candidates in parallel
        # Reuse the persistent pool from ParallelEvaluator when available
        # to avoid expensive per-call pool creation and data reloading.
        owns_executor = evaluator is None
        if owns_executor:
            executor = ProcessPoolExecutor(
                max_workers=actual_workers,
                initializer=_init_worker,
                initargs=(ga_config,),
            )
            _active_executors.append(executor)
        else:
            executor = evaluator._get_executor()

        results_by_elite: Dict[int, List[Dict[str, Any]]] = {
            idx: [] for idx in elite_state
        }

        try:
            futures = {
                executor.submit(
                    _evaluate_parsimony_candidate_in_worker,
                    task['trial_dict'],
                    task['elite_index'],
                    task['candidate_index'],
                    task['kind'],
                    task['component_index'],
                ): task['candidate_index']
                for task in tasks
            }

            total_timeout = backtest_timeout * (len(tasks) // actual_workers + 1)

            for future in as_completed(futures, timeout=total_timeout):
                try:
                    result = future.result(timeout=backtest_timeout)
                    if result['success']:
                        results_by_elite[result['elite_index']].append(result)
                except FuturesTimeoutError:
                    logger.warning("[PARSIMONY-PARALLEL] Candidate timed out")
                except Exception as e:
                    logger.debug(f"[PARSIMONY-PARALLEL] Worker error: {e}")

        except FuturesTimeoutError:
            for future in futures:
                if not future.done():
                    future.cancel()
            logger.warning("[PARSIMONY-PARALLEL] Batch timeout reached")
        except BrokenProcessPool:
            logger.error("[PARSIMONY-PARALLEL] Worker pool crashed")
        finally:
            # Only shut down the pool if we created it (ephemeral mode).
            # When reusing ParallelEvaluator's pool, it manages its own lifecycle.
            if owns_executor:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                if executor in _active_executors:
                    _active_executors.remove(executor)
                _kill_pool_processes(executor)

        # Phase 3: for each elite, pick the best acceptable removal
        removed_this_round = 0
        elites_to_remove: List[int] = []

        for elite_idx in list(elite_state.keys()):
            current_gene, current_fitness = elite_state[elite_idx]
            candidate_results = results_by_elite.get(elite_idx, [])
            if not candidate_results:
                elites_to_remove.append(elite_idx)
                continue

            # Find the best acceptable removal (smallest fitness drop)
            best_candidate = None
            best_trial_fitness = -1.0
            best_drop = float('inf')

            for result in candidate_results:
                trial_fitness = result['fitness']
                if trial_fitness <= 0:
                    continue
                fitness_drop = (current_fitness - trial_fitness) / max(abs(current_fitness), 1e-9)
                if fitness_drop <= epsilon and trial_fitness > best_trial_fitness:
                    flat = result['candidate_index']
                    best_candidate = task_meta[flat]
                    best_trial_fitness = trial_fitness
                    best_drop = fitness_drop

            if best_candidate is not None:
                _, kind, comp_idx, trial_gene = best_candidate
                elite_state[elite_idx] = (trial_gene, best_trial_fitness)
                removed_this_round += 1
                logger.debug(
                    f"[PARSIMONY-PARALLEL] Elite {elite_idx}: removed {kind}[{comp_idx}] "
                    f"fitness {current_fitness:.4f} -> {best_trial_fitness:.4f} "
                    f"(drop {best_drop:.2%} <= eps={epsilon:.2%})"
                )
            else:
                elites_to_remove.append(elite_idx)

        # Remove elites that had no acceptable removal from further rounds
        for idx in elites_to_remove:
            elite_state.pop(idx, None)

        total_removed += removed_this_round

        if removed_this_round == 0 or not elite_state:
            break  # no progress → stop

    # Phase 4: apply accepted simplifications back to Individual objects
    for elite_idx, (simplified_gene, new_fitness) in elite_state.items():
        ind = elites[elite_idx]
        original_complexity = ind.strategy_gene.calculate_complexity()
        new_complexity = simplified_gene.calculate_complexity()
        if new_complexity < original_complexity:
            # Restore original identity (parsimony trials used 9000+ namespace)
            simplified_gene.generation = ind.strategy_gene.generation
            simplified_gene.individual_id = ind.strategy_gene.individual_id
            ind.strategy_gene = simplified_gene
            ind.raw_fitness = new_fitness
            ind.fitness = new_fitness
            ind.metrics = ind.metrics or {}
            ind.metrics['parsimony_removed'] = original_complexity - new_complexity

    elapsed = time.time() - start_time
    if total_removed > 0:
        logger.info(
            f"[PARSIMONY-PARALLEL] Removed {total_removed} component(s) "
            f"from elites in {elapsed:.1f}s"
        )

    return total_removed


@dataclass
class ParallelEvaluationResult:
    """Results from parallel evaluation."""
    successful: int
    failed: int
    total_time: float
    speedup_estimate: float  # Estimated speedup vs sequential


class ParallelEvaluator:
    """
    Parallelizes strategy evaluation using ProcessPoolExecutor.
    
    Each worker process has its own FitnessEvaluator instance to avoid
    sharing state. Strategies are serialized as dictionaries for pickling.
    
    Features:
    - **Persistent pool**: Workers are created once and reused across
      generations, avoiding the expensive data-reload on each pool
      creation and eliminating zombie-process accumulation.
    - Per-backtest timeout prevents pathological strategies from blocking
    - Walk-forward disabled in workers (applied post-hoc on elites instead)
    - Automatic cleanup of worker processes on shutdown/exit
    - Context manager support (``with ParallelEvaluator(...) as pe:``)
    
    Example:
        evaluator = ParallelEvaluator(config, num_workers=4)
        results = evaluator.evaluate_batch(individuals, nsga2_mode=True)
        evaluator.shutdown()
    """
    
    def __init__(self, config: Dict[str, Any], num_workers: Optional[int] = None):
        """
        Initialize parallel evaluator.
        
        The worker pool is created lazily on the first ``evaluate_batch``
        call, so construction is cheap and safe even if evaluation never
        happens.
        
        Args:
            config: Configuration dictionary
            num_workers: Number of worker processes (default: CPU count)
        """
        self.config = config
        
        # Get parallel config
        parallel_config = config.get('parallel_evaluation', {})
        
        # Determine number of workers
        if num_workers is not None:
            self.num_workers = num_workers
        elif parallel_config.get('num_workers'):
            self.num_workers = parallel_config['num_workers']
        else:
            # Auto-detect: use CPU count but leave 1 core free
            self.num_workers = max(1, os.cpu_count() - 1)
        
        # Memory-aware scaling: cap workers based on available RAM
        if parallel_config.get('memory_aware', False):
            self.num_workers = self._memory_aware_worker_cap(
                self.num_workers, parallel_config
            )
        
        # Per-backtest timeout (seconds). 0 = no timeout.
        self.backtest_timeout = parallel_config.get('backtest_timeout', 120)
        
        # NSGA-II config
        self.nsga2_mode = config.get('genetic_algorithm', {}).get('mode') == 'nsga2'
        self.objectives_config = config.get('nsga2', {}).get('objectives', [])
        
        # Track whether WF is configured (for post-hoc validation)
        wf_config = config.get('walk_forward', {})
        self.walk_forward_configured = wf_config.get('enabled', False)
        if self.walk_forward_configured:
            logger.info("[PARALLEL] Walk-forward detected — will be run post-hoc on elites, not inside workers")
        
        # Persistent pool — created lazily on first evaluate_batch()
        self._executor: Optional[ProcessPoolExecutor] = None
        self._pool_generation_count = 0  # How many batches this pool has served
        
        logger.info(f"[PARALLEL] Initialized with {self.num_workers} workers, "
                    f"timeout={self.backtest_timeout}s per backtest")
    
    @staticmethod
    def _memory_aware_worker_cap(
        desired_workers: int,
        parallel_config: Dict[str, Any],
    ) -> int:
        """
        Cap the number of workers based on available system RAM.
        
        Each worker process loads a full FreqTrade Backtesting engine + OHLCV
        data.  On a 16 GB system with 5 workers at ~800 MB each, that's ~4 GB
        for workers alone plus ~2 GB for OS + main process.
        
        This method prevents OOM by checking available RAM at startup and 
        capping workers accordingly.
        
        Args:
            desired_workers: Number of workers requested (from CPU count or config)
            parallel_config: parallel_evaluation config section
            
        Returns:
            Capped number of workers (>= 1)
        """
        estimated_worker_mb = parallel_config.get('estimated_worker_memory_mb', 800)
        min_free_ram_mb = parallel_config.get('min_free_ram_mb', 2048)
        
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            usable_mb = available_mb - min_free_ram_mb
            
            if usable_mb <= 0:
                logger.warning(
                    "[PARALLEL] Very low available RAM (%.0f MB free, need %.0f MB reserve). "
                    "Using 1 worker.",
                    available_mb, min_free_ram_mb,
                )
                return 1
            
            max_by_memory = max(1, int(usable_mb / estimated_worker_mb))
            
            if max_by_memory < desired_workers:
                logger.info(
                    "[PARALLEL] Memory-aware scaling: capping workers from %d → %d "
                    "(%.0f MB available, %.0f MB reserved, ~%d MB/worker)",
                    desired_workers, max_by_memory, available_mb,
                    min_free_ram_mb, estimated_worker_mb,
                )
                return max_by_memory
            else:
                logger.info(
                    "[PARALLEL] Memory check passed: %d workers OK "
                    "(%.0f MB available, ~%d MB/worker, %.0f MB headroom)",
                    desired_workers, available_mb, estimated_worker_mb,
                    usable_mb - desired_workers * estimated_worker_mb,
                )
                return desired_workers
                
        except ImportError:
            logger.warning(
                "[PARALLEL] psutil not installed — memory-aware scaling disabled. "
                "Install with: pip install psutil"
            )
            return desired_workers
        except Exception as e:
            logger.warning(
                "[PARALLEL] Memory check failed (%s) — using %d workers",
                e, desired_workers,
            )
            return desired_workers
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """
        Get current process memory usage in MB (RSS).
        
        Returns peak RSS for the main process. Useful for tracking
        memory drift over long evolution runs.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            try:
                import resource
                # ru_maxrss is in KB on Linux
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except Exception:
                return 0.0
        except Exception:
            return 0.0

    def _get_executor(self) -> ProcessPoolExecutor:
        """
        Return the persistent ProcessPoolExecutor, creating it on first call.
        
        The pool is reused across generations to avoid:
        - Expensive per-worker data reloading (FreqTrade Backtesting init)
        - Zombie process accumulation from repeated create/destroy cycles
        - Deadlocks from pool creation during active evaluation
        """
        if self._executor is None:
            logger.info(f"[PARALLEL] Creating persistent worker pool ({self.num_workers} workers)...")
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_worker,
                initargs=(self.config,),
            )
            _active_executors.append(self._executor)
            logger.info("[PARALLEL] Worker pool created — workers will persist across generations")
        return self._executor
    
    def _check_pool_health(self) -> bool:
        """
        Verify the persistent pool is still responsive.
        
        Submits a trivial task and checks it completes within 10s.
        If the pool is broken, shuts it down so _get_executor() creates a fresh one.
        
        Returns:
            True if pool is healthy, False if it was recycled.
        """
        if self._executor is None:
            return True  # Will be freshly created
        
        try:
            future = self._executor.submit(_pool_health_check_fn)
            result = future.result(timeout=10)
            return result is True
        except Exception as e:
            logger.warning(f"[PARALLEL] Pool health check failed ({e}), recycling pool...")
            self._shutdown_executor()
            return False
    
    def _shutdown_executor(self):
        """Shut down the persistent executor if it exists."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.debug(f"[PARALLEL] Error during pool shutdown: {e}")
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            if self._executor in _active_executors:
                _active_executors.remove(self._executor)
            _kill_pool_processes(self._executor)
            self._executor = None
            _reap_zombies()
            logger.info("[PARALLEL] Worker pool shut down")
    
    def shutdown(self):
        """Explicitly shutdown any active executors."""
        self._shutdown_executor()
        _cleanup_executors()
        logger.info("[PARALLEL] Shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
    
    def __del__(self):
        """Best-effort cleanup if shutdown() was never called."""
        try:
            self._shutdown_executor()
        except Exception:
            pass
    
    def evaluate_batch(
        self, 
        individuals: List['Individual'],
        progress_callback: Optional[callable] = None
    ) -> ParallelEvaluationResult:
        """
        Evaluate multiple individuals in parallel.
        
        Args:
            individuals: List of unevaluated Individual objects
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            ParallelEvaluationResult with statistics
        """
        from genetic_algorithm.core.individual import Individual
        from genetic_algorithm.core.nsga2 import extract_objectives_from_metrics
        
        if not individuals:
            return ParallelEvaluationResult(
                successful=0, failed=0, total_time=0.0, speedup_estimate=1.0
            )
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        # Prepare tasks: serialize strategies to dicts for pickling
        tasks = []
        nsga2_min_trades = self.config.get('nsga2', {}).get('min_trades', 0)
        for i, ind in enumerate(individuals):
            tasks.append({
                'strategy_gene_dict': ind.strategy_gene.to_dict(),
                'strategy_index': i,
                'nsga2_mode': self.nsga2_mode,
                'objectives_config': self.objectives_config,
                'nsga2_min_trades': nsga2_min_trades,
            })
        
        self._pool_generation_count += 1
        logger.info(f"[PARALLEL] Evaluating {len(tasks)} strategies with {self.num_workers} workers "
                    f"(pool batch #{self._pool_generation_count})...")
        
        # Health-check the persistent pool (recycles if broken)
        self._check_pool_health()
        executor = self._get_executor()
        
        try:
            # Submit all tasks
            futures = {
                executor.submit(
                    _evaluate_strategy_in_worker,
                    task['strategy_gene_dict'],
                    task['strategy_index'],
                    task['nsga2_mode'],
                    task['objectives_config']
                ): task['strategy_index']
                for task in tasks
            }
            
            # Collect results as they complete (with per-future timeout)
            completed = 0
            timed_out = 0
            for future in as_completed(futures, timeout=self.backtest_timeout * len(tasks) if self.backtest_timeout else None):
                try:
                    result = future.result(timeout=self.backtest_timeout if self.backtest_timeout else None)
                    idx = result['index']
                    ind = individuals[idx]
                    
                    if result['success']:
                        ind.set_fitness(result['fitness'], result['metrics'])
                        
                        # Set objectives for NSGA-II
                        if self.nsga2_mode and 'objectives' in result:
                            ind.set_objectives(result['objectives'], result['metrics'])
                        
                        successful += 1
                        logger.debug(
                            f"[PARALLEL] {ind.id}: fitness={result['fitness']:.4f}, "
                            f"profit={result['metrics'].get('profit', 0):.2f}%"
                        )
                    else:
                        ind.set_fitness(0.0, {
                            'error': result.get('error', 'Unknown error'),
                            'profit': 0.0,
                            'sharpe_ratio': 0.0,
                            'max_drawdown': 1.0,
                            'win_rate': 0.0,
                            'num_trades': 0
                        })
                        failed += 1
                        logger.warning(f"[PARALLEL] {ind.id} failed: {result.get('error')}")
                    
                except FuturesTimeoutError:
                    timed_out += 1
                    failed += 1
                    logger.warning(f"[PARALLEL] Strategy timed out after {self.backtest_timeout}s")
                except Exception as e:
                    failed += 1
                    logger.error(f"[PARALLEL] Worker error: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(tasks))
            
            # Cancel any remaining futures that haven't completed
            for future in futures:
                if not future.done():
                    future.cancel()
                    timed_out += 1
                    
        except FuturesTimeoutError:
            # Batch-level timeout — cancel remaining
            for future in futures:
                if not future.done():
                    future.cancel()
                    timed_out += 1
            logger.warning(f"[PARALLEL] Batch timeout reached, {timed_out} strategies cancelled")
        except BrokenProcessPool:
            # Pool died — recycle it for next generation
            logger.error("[PARALLEL] Worker pool crashed! Will recycle for next batch.")
            self._shutdown_executor()
            failed += len(tasks) - completed
        # NOTE: No executor.shutdown() here — pool is persistent!
        
        # Set fitness for any individuals that never got results (timed out)
        for ind in individuals:
            if not ind.evaluated:
                ind.set_fitness(0.0, {
                    'error': 'Timed out or cancelled',
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0
                })
                failed += 1
        
        total_time = time.time() - start_time
        
        # Estimate speedup (assume linear scaling with workers)
        # A more accurate estimate would require benchmarking
        speedup_estimate = min(self.num_workers, len(tasks) / max(1, self.num_workers / 2))
        
        logger.info(
            f"[PARALLEL] Complete: {successful} succeeded, {failed} failed "
            f"({timed_out} timed out) in {total_time:.1f}s "
            f"(~{speedup_estimate:.1f}x speedup vs sequential)"
        )
        
        return ParallelEvaluationResult(
            successful=successful,
            failed=failed,
            total_time=total_time,
            speedup_estimate=speedup_estimate
        )


def _test_parallel_worker() -> bool:
    """Simple test function for parallel availability check."""
    return True


def is_parallel_available() -> bool:
    """
    Check if parallel evaluation is available on this system.
    
    Returns:
        True if multiprocessing is available and working
    """
    try:
        # Test if we can create a process pool
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_test_parallel_worker)
            return future.result(timeout=5)
    except Exception:
        return False


def get_recommended_workers() -> int:
    """
    Get recommended number of workers for this system.
    
    Returns:
        Recommended worker count based on CPU cores
    """
    cpu_count = os.cpu_count() or 1
    # Leave 1 core free for main process and OS
    return max(1, cpu_count - 1)
