# GA Parameter Experiments

This folder contains experiments to test which GA parameters lead to actual fitness improvements over generations.

## Quick Start

```bash
cd /home/kali/trading/freqtradeForkGA

# List available experiments
python genetic_algorithm/experiments/run_experiments.py --list

# Run a single experiment
python genetic_algorithm/experiments/run_experiments.py exp01_baseline

# Run all experiments
python genetic_algorithm/experiments/run_experiments.py --all

# Run specific experiments and compare
python genetic_algorithm/experiments/run_experiments.py exp01_baseline exp02_high_mutation --compare
```

## Experiment Configurations

| Experiment | Key Changes | Hypothesis |
|------------|-------------|------------|
| **exp01_baseline** | Default params (mut=0.15, tourn=3) | Establish baseline performance |
| **exp02_high_mutation** | mutation_rate=0.30, no adaptive | Higher exploration, may find better optima |
| **exp03_low_mutation_adaptive** | mutation_rate=0.08, adaptive=true, adaptation_factor=3.0 | Stable exploitation with escape mechanism |
| **exp04_large_population** | population=30, generations=6 | Larger search space, better initial coverage |
| **exp05_high_selection_pressure** | tournament_size=5, more immigrants | Faster convergence, risk of premature conv. |
| **exp06_rank_selection** | selection_method='rank' | Gradual selection, better diversity |

## Common Settings (All Experiments)

- **Parallel evaluation**: Enabled (for speed)
- **Walk-forward**: Disabled (too slow for experiments)
- **Regime detection**: Enabled (sma_adx method)
- **Population**: 20 (except exp04)
- **Generations**: 8 (except exp04: 6)
- **Timeframe**: 5m/15m/1h
- **Pairs**: ETH/BTC, LTC/BTC

## What to Look For

1. **Fitness Improvement**: Does best fitness increase over generations?
2. **Improvement Consistency**: How many generations show improvement?
3. **Convergence Speed**: How quickly does fitness plateau?
4. **Final Fitness**: Which config achieves highest final fitness?
5. **Diversity**: Does the population maintain diversity?

## Results Format

Results are saved as JSON in `results/` folder:

```json
{
  "experiment": "exp01_baseline",
  "config_summary": {...},
  "generation_stats": [
    {"generation": 1, "best_fitness": 0.45, "avg_fitness": 0.32, ...},
    {"generation": 2, "best_fitness": 0.48, "avg_fitness": 0.35, ...},
    ...
  ],
  "improvement": {
    "best_fitness_change": 0.15,
    "best_fitness_percent": 33.3,
    "generations_with_improvement": 5
  },
  "best_individual": {...}
}
```

## Folder Structure

```
experiments/
├── README.md                    # This file
├── run_experiments.py           # Runner script
├── configs/
│   ├── exp01_baseline.yaml
│   ├── exp02_high_mutation.yaml
│   ├── exp03_low_mutation_adaptive.yaml
│   ├── exp04_large_population.yaml
│   ├── exp05_high_selection_pressure.yaml
│   └── exp06_rank_selection.yaml
├── results/                     # JSON results from runs
└── logs/                        # Detailed logs (if enabled)
```

## Expected Runtime

With parallel evaluation and 20 population / 8 generations:
- Estimated: 5-10 minutes per experiment
- Total for all 6 experiments: ~30-60 minutes

## After Running

Compare results to identify:
1. Which parameters help vs hurt fitness improvement
2. Trade-offs between exploration and exploitation
3. Best configuration for future runs

Use the `--compare` flag to get a side-by-side comparison table.
