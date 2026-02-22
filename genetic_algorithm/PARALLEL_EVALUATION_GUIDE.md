# Parallel Evaluation Guide

This guide explains how to enable and configure parallel evaluation in the Genetic Algorithm for significant speedup on multi-core systems.

## Overview

Parallel evaluation allows backtesting multiple strategies simultaneously using Python's `multiprocessing` module. Each worker process runs independent backtests, achieving near-linear speedup with the number of CPU cores.

## Quick Start

Enable parallel evaluation in `ga_config.yaml`:

```yaml
parallel_evaluation:
  enabled: true
  num_workers: null  # Auto-detect (CPU cores - 1)
```

That's it! The GA will now evaluate strategies in parallel.

## Configuration Options

### `parallel_evaluation.enabled`
- **Type:** boolean
- **Default:** `false`
- **Description:** Enable/disable parallel evaluation

### `parallel_evaluation.num_workers`
- **Type:** integer or null
- **Default:** `null` (auto-detect)
- **Description:** Number of worker processes
- **Recommendations:**
  - `null`: Auto-detect (uses CPU cores - 1)
  - For 8-core CPU: 6-7 workers
  - For 4-core CPU: 3 workers
  - Leave 1 core free for main process

## Expected Performance

| CPU Cores | Workers | Speedup | Notes |
|-----------|---------|---------|-------|
| 4         | 3       | 2.5-3x  | Good for development |
| 8         | 6-7     | 4-5x    | Recommended for production |
| 12        | 10-11   | 6-8x    | High-end workstation |
| 16+       | 12-14   | 8-10x   | Server-grade |

**Note:** Actual speedup depends on:
- Population size (more strategies = better parallelization)
- Walk-forward windows (more windows = more work per strategy)
- Strategy complexity
- Available memory

## When to Use Parallel Evaluation

### Recommended For:
- Population sizes > 20 strategies
- Walk-forward optimization enabled
- Multi-timeframe strategies
- Production runs with many generations

### Not Recommended For:
- Very small populations (< 10)
- Quick testing/debugging
- Memory-constrained systems

## How It Works

1. **Worker Pool Initialization**
   - Creates N worker processes at start
   - Each worker initializes its own `FitnessEvaluator` and `DirectBacktester`
   - Workers remain alive for the entire evolution

2. **Parallel Evaluation**
   - Main process serializes strategy genes to dictionaries
   - Dictionary tasks distributed to workers via queue
   - Workers evaluate and return (fitness, metrics) results
   - Main process collects results and updates individuals

3. **Process Isolation**
   - Each worker runs in separate memory space
   - No shared state between workers
   - Thread-safe result collection

## Memory Considerations

Each worker process uses approximately:
- 200-500 MB base memory
- Additional memory for data loading
- Additional memory per backtest

**Estimate total memory:**
```
Total RAM = Base + (Workers × Per-Worker)
         = 2 GB + (Workers × 0.5 GB)
```

For 8 workers: ~6 GB RAM recommended

## Troubleshooting

### "Parallel evaluation not available"
- Check if multiprocessing works: `python -c "from concurrent.futures import ProcessPoolExecutor; print('OK')"`
- Ensure no conflicting process spawning

### Slow startup
- First parallel evaluation spawns workers (1-2 seconds)
- Subsequent evaluations reuse workers

### Memory errors
- Reduce `num_workers`
- Increase swap space
- Use 64-bit Python

### Workers failing silently
- Check logs for worker errors
- Strategies that fail are marked with fitness=0

## Benchmarking

Run the benchmark script to measure speedup on your system:

```bash
python genetic_algorithm/benchmark_parallel.py --strategies 20 --workers 4
```

Options:
- `--strategies N`: Number of strategies to benchmark (default: 20)
- `--workers N`: Number of workers (default: auto)
- `--config PATH`: Config file path
- `--skip-sequential`: Skip sequential benchmark (faster)

## Example Benchmark Results

System: AMD Ryzen 12-core, 32 GB RAM

```
======================================================================
BENCHMARK RESULTS
======================================================================
Sequential time:    174.32s
Parallel time:      48.76s
Speedup:            3.58x
Efficiency:         89.4%
Time saved:         125.56s

RECOMMENDATION:
✅ Parallel evaluation provides 3.6x speedup - RECOMMENDED
   Enable in ga_config.yaml:
   parallel_evaluation:
     enabled: true
     num_workers: 4
======================================================================
```

## Integration with Other Features

### Walk-Forward Optimization
Parallel evaluation works with walk-forward. Each strategy's windows are evaluated sequentially within the worker, but strategies are parallelized across workers.

### NSGA-II Multi-Objective
Fully compatible. Workers extract objectives and return them along with fitness.

### Multi-Timeframe
Fully compatible. Higher timeframe data is loaded independently in each worker.

## API Reference

### ParallelEvaluator

```python
from genetic_algorithm.evaluation.parallel import ParallelEvaluator

evaluator = ParallelEvaluator(config, num_workers=4)
result = evaluator.evaluate_batch(individuals)

print(f"Successful: {result.successful}")
print(f"Failed: {result.failed}")
print(f"Time: {result.total_time:.2f}s")
print(f"Speedup: {result.speedup_estimate:.2f}x")
```

### Utility Functions

```python
from genetic_algorithm.evaluation.parallel import (
    is_parallel_available,
    get_recommended_workers
)

# Check if parallel evaluation is available
if is_parallel_available():
    workers = get_recommended_workers()
    print(f"Recommended workers: {workers}")
```

## Files

- `genetic_algorithm/evaluation/parallel.py` - Main parallel evaluation module
- `genetic_algorithm/benchmark_parallel.py` - Benchmark script
- `tests/test_parallel_evaluation.py` - Test suite
