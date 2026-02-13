# Genetic Algorithm for FreqTrade Strategy Evolution

## Overview

This module implements a Genetic Algorithm (GA) system for autonomously developing and optimizing trading strategies for FreqTrade. The system evolves strategies over multiple generations through selection, mutation, and crossover operations.

## Project Goals

The main goal is to create a system that:
- Generates and tests trading strategies automatically
- Evaluates strategies through backtesting and dry-run trading
- Selects the best performing strategies based on multiple metrics
- Evolves strategies over time through genetic operations
- Minimizes risk and drawdown while maximizing profits
- Provides the top N best strategies at any given time

## Architecture

### Directory Structure

```
genetic_algorithm/
├── core/                    # Core GA components
│   ├── population.py       # Population management
│   ├── selection.py        # Selection algorithms
│   ├── crossover.py        # Crossover operators
│   ├── mutation.py         # Mutation operators
│   └── evolution.py        # Main evolution loop
├── strategies/             # Strategy generation and management
│   ├── generator.py        # Strategy generator
│   ├── template.py         # Strategy templates
│   ├── components.py       # Modular strategy components
│   └── validator.py        # Strategy validation
├── evaluation/             # Fitness and evaluation
│   ├── fitness.py          # Fitness function
│   ├── backtest.py         # Backtesting integration
│   ├── metrics.py          # Performance metrics
│   └── live_test.py        # Dry-run testing
├── utils/                  # Utility functions
│   ├── logging.py          # Logging utilities
│   ├── storage.py          # Strategy storage
│   └── visualization.py    # Results visualization
├── config/                 # Configuration files
│   └── ga_config.yaml      # GA parameters
└── tests/                  # Unit tests
```

## Key Components

### 1. Strategy Representation

Strategies are represented as modular components that can be easily mutated and combined:
- Entry conditions (indicators, thresholds, combinations)
- Exit conditions (take profit, stop loss, trailing stops)
- Risk management parameters
- Timeframe and pair selection

### 2. Fitness Function

Multi-objective fitness function considering:
- Total profit/return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades
- Stability metrics
- Risk-adjusted returns

### 3. Genetic Operations

- **Selection**: Tournament selection, roulette wheel, rank-based
- **Crossover**: Single-point, multi-point, uniform crossover
- **Mutation**: Parameter mutation, component replacement, rule modification

### 4. Evolution Process

```
1. Initialize population (N strategies)
2. Evaluate fitness (backtest each strategy)
3. Select top performers
4. Apply crossover and mutation
5. Create new generation
6. Repeat from step 2
```

## Usage

### Basic Usage

```python
from genetic_algorithm.core.evolution import GeneticAlgorithm

# Initialize GA
ga = GeneticAlgorithm(
    population_size=100,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.7
)

# Run evolution
best_strategies = ga.evolve()

# Get top 5 strategies
top_5 = ga.get_top_strategies(n=5)
```

### Configuration

Edit `config/ga_config.yaml` to customize:
- Population size
- Number of generations
- Mutation/crossover rates
- Fitness function weights
- Backtesting parameters

## Integration with FreqTrade

The GA system integrates with FreqTrade by:
1. Generating valid FreqTrade strategy files
2. Using FreqTrade's backtesting engine for evaluation
3. Supporting dry-run mode for live testing
4. Following FreqTrade's strategy interface (IStrategy)

## Future Enhancements

1. **ML Integration**: Use FreqAI for parameter optimization
2. **LLM Integration**: Generate strategies using LLMs (Grok API, OpenAI)
3. **Island Model**: Multiple parallel populations for better diversity
4. **Real-time Adaptation**: Continuous learning from live trading results
5. **Multi-exchange Support**: Optimize for different exchanges

## Development Status

See [TODO.md](TODO.md) for current development status and roadmap.

## References

- FreqTrade Documentation: https://www.freqtrade.io/
- Genetic Algorithms: https://en.wikipedia.org/wiki/Genetic_algorithm
- Trading Strategy Optimization: Research papers and articles
