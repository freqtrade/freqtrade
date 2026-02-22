# Changelog

All notable changes to the Freqtrade Genetic Algorithm fork.

## [Unreleased] - 2025-02-22

### Added

#### NSGA-II Multi-Objective Optimization
- Complete implementation of NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Multi-objective fitness evaluation with configurable objectives:
  - Profit maximization
  - Sharpe ratio optimization
  - Win rate improvement
  - Drawdown minimization
- Pareto front tracking across generations
- Non-dominated sorting with crowding distance for diversity preservation
- Full integration with existing evolution framework

#### Parallel Strategy Evaluation
- Multi-process parallel backtesting using `ProcessPoolExecutor`
- Configurable worker count (auto-detects optimal based on CPU cores)
- **Benchmark Results:**
  - 6 strategies, 4 workers: 22.73s → 7.41s (**3.07x speedup**)
  - 12 strategies, 6 workers: 33.02s → 9.11s (**3.62x speedup**)
- Automatic fallback to sequential evaluation on errors
- Worker process isolation for stability
- Graceful shutdown and resource cleanup

#### Configuration Options
New `parallel_evaluation` section in `ga_config.yaml`:
```yaml
parallel_evaluation:
  enabled: true
  num_workers: null  # Auto-detect optimal workers
  worker_log_level: "WARNING"
```

### Documentation
- `PARALLEL_EVALUATION_GUIDE.md` - Complete parallel evaluation documentation
- Updated `TODO_ga_improvements.md` with completed features
- Benchmark script at `genetic_algorithm/benchmark_parallel.py`

### Technical Details
- New module: `genetic_algorithm/evaluation/parallel.py`
- Dependencies: Uses standard library `concurrent.futures` (no new deps)
- Test suite: `tests/test_parallel_evaluation.py`

---

## [Previous] - 2025-02-19

### Features (Pre-existing)
- Walk-Forward Analysis integration
- Dynamic max_open_trades optimization
- Genetic algorithm-based strategy parameter optimization
- Multi-criteria fitness evaluation
- Configuration-driven evolution parameters
