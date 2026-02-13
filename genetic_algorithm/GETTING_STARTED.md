# Getting Started with Genetic Algorithm for FreqTrade

## Overview

This guide will help you get started with the Genetic Algorithm (GA) system for autonomously developing trading strategies for FreqTrade.

## Prerequisites

- Python 3.11 or higher
- FreqTrade installed and configured
- Basic understanding of trading concepts
- Historical market data for backtesting

## Installation

### 1. Ensure FreqTrade is Installed

This GA system is designed to work with FreqTrade. Make sure FreqTrade is properly installed:

```bash
# Verify FreqTrade installation
freqtrade --version
```

### 2. Install GA-specific Dependencies

```bash
pip install -r genetic_algorithm/requirements.txt
```

### 3. Set Up Directory Structure

The necessary directories are already created, but ensure you have:
- `genetic_algorithm/data/` - for database and checkpoints
- `genetic_algorithm/logs/` - for log files
- `genetic_algorithm/plots/` - for visualization outputs
- `user_data/strategies/ga_generated/` - for generated strategies

```bash
mkdir -p genetic_algorithm/data genetic_algorithm/logs genetic_algorithm/plots
mkdir -p user_data/strategies/ga_generated
```

## Quick Start

### 1. Try the Example Scripts

Before running the full evolution, try the provided examples:

**Test the components:**
```bash
python genetic_algorithm/test_generation.py
```

This will verify that strategy generation, population management, and code generation work correctly.

**Generate example strategies:**
```bash
python genetic_algorithm/example_usage.py
```

This will:
- Initialize the GA with your configuration
- Create a population of 100 random strategies
- Show details of the first 3 strategies
- Generate a complete FreqTrade strategy file

The generated example will be saved to `genetic_algorithm/examples/example_strategy.py`.

### 2. Download Historical Data

Before running the GA, you need historical data for backtesting:

```bash
freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 5m 15m 1h \
  --timerange 20230101-20231231
```

### 2. Download Historical Data

Edit `genetic_algorithm/config/ga_config.yaml` to customize:

```yaml
genetic_algorithm:
  population_size: 100      # Start with 50-100 for testing
  generations: 50           # 20-50 generations for initial runs
  mutation_rate: 0.15
  crossover_rate: 0.7

backtesting:
  timerange: "20230101-20231231"  # Your downloaded data range
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
    - "BNB/USDT"
```

### 3. Configure the GA

```python
from genetic_algorithm.core.evolution import GeneticAlgorithm

# Initialize the GA
ga = GeneticAlgorithm(config_path="genetic_algorithm/config/ga_config.yaml")

# Run evolution
print("Starting evolution...")
best_strategies = ga.evolve()

# Get top 5 strategies
top_5 = ga.get_top_strategies(n=5)

print(f"Evolution complete! Best fitness: {best_strategies[0].fitness}")
print(f"Top 5 strategies saved to user_data/strategies/ga_generated/")
```

Or use the command-line interface:

```bash
python -m genetic_algorithm.main --config genetic_algorithm/config/ga_config.yaml
```

### 4. Run Your First Evolution (When Backtesting is Integrated)

After evolution completes, you'll find:
- Generated strategies in `user_data/strategies/ga_generated/`
- Performance plots in `genetic_algorithm/plots/`
- Detailed logs in `genetic_algorithm/logs/ga.log`
- Strategy database in `genetic_algorithm/data/strategies.db`

**Note**: Full evolution with backtesting integration is not yet complete. Currently you can:
- Generate random strategies
- Test the genetic operators
- Inspect generated strategy code

To complete the integration, see Phase 4.2 in TODO.md.

### 5. View Results

Test one of the top strategies with FreqTrade:

```bash
freqtrade backtesting \
  --config config.json \
  --strategy GAStrategy_Gen50_Best1 \
  --timerange 20230101-20231231
```

### 6. Test a Generated Strategy

Once you're satisfied with a strategy, test it in dry-run (paper trading):

```bash
freqtrade trade \
  --config config.json \
  --strategy GAStrategy_Gen50_Best1 \
  --dry-run
```

### 7. Run in Dry-Run Mode

### Strategy Files

Generated strategies follow this naming pattern:
```
GAStrategy_Gen{generation}_Ind{individual_id}.py
```

For example: `GAStrategy_Gen50_Ind001.py`

### Fitness Metrics

Each strategy is evaluated on:
- **Profit**: Total return percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Trade Frequency**: Number of trades (balanced scoring)

### Plots

The system generates several plots:
1. **Fitness Evolution**: Shows how fitness improves over generations
2. **Population Diversity**: Shows genetic diversity over time
3. **Top Strategies**: Compares metrics of best strategies

## Configuration Guide

### Key Parameters to Tune

#### Population Size
- **Small (20-50)**: Faster iterations, less diversity
- **Medium (50-100)**: Good balance
- **Large (100-200)**: More diversity, slower convergence

#### Mutation Rate
- **Low (0.05-0.1)**: More stable, slower exploration
- **Medium (0.1-0.2)**: Balanced
- **High (0.2-0.3)**: More exploration, less stable

#### Generations
- **Short (10-20)**: Quick tests
- **Medium (30-50)**: Standard runs
- **Long (50-100+)**: Deep optimization

### Fitness Weights

Adjust based on your trading goals:

**Conservative (low risk)**:
```yaml
fitness_weights:
  profit: 0.20
  sharpe_ratio: 0.30
  drawdown: 0.30
  win_rate: 0.15
  trade_frequency: 0.05
```

**Aggressive (high profit)**:
```yaml
fitness_weights:
  profit: 0.50
  sharpe_ratio: 0.20
  drawdown: 0.10
  win_rate: 0.10
  trade_frequency: 0.10
```

**Balanced**:
```yaml
fitness_weights:
  profit: 0.30
  sharpe_ratio: 0.25
  drawdown: 0.20
  win_rate: 0.15
  trade_frequency: 0.10
```

## Best Practices

### 1. Start Small
- Begin with a small population (20-30)
- Run for few generations (10-20)
- Test on a short time period (1-3 months)

### 2. Validate Results
- Always backtest generated strategies manually
- Test on out-of-sample data
- Run dry-run before live trading
- Never trust a strategy based solely on backtest

### 3. Monitor Evolution
- Watch for convergence (fitness stops improving)
- Check population diversity
- Review top strategies manually

### 4. Iterate
- Adjust fitness weights based on results
- Modify indicator ranges
- Add constraints if needed

### 5. Avoid Overfitting
- Use walk-forward analysis
- Test on multiple time periods
- Validate with dry-run
- Be skeptical of "too good" results

## Troubleshooting

### Issue: Strategies have no trades
**Solution**: Loosen entry conditions in config, adjust indicator thresholds

### Issue: Evolution is too slow
**Solution**: 
- Reduce population size
- Reduce number of pairs
- Shorten backtest timerange
- Enable caching

### Issue: All strategies perform poorly
**Solution**:
- Check if downloaded data is correct
- Verify indicator ranges make sense
- Adjust fitness function weights
- Try different time periods

### Issue: Strategies overfit to backtest
**Solution**:
- Use walk-forward validation
- Test on different time periods
- Add stricter constraints
- Increase diversity (higher mutation rate)

## Next Steps

1. **Read the Documentation**: Check `README.md` and `DEVELOPMENT_PLAN.md`
2. **Understand the Code**: Review the core modules
3. **Experiment**: Try different configurations
4. **Analyze**: Study what makes good strategies work
5. **Iterate**: Continuously improve based on results

## Support

For questions and issues:
- Check the TODO.md for development status
- Review FreqTrade documentation: https://www.freqtrade.io/
- Read about Genetic Algorithms for strategy optimization

## Warning

⚠️ **Important**: 
- This is experimental software
- Always test thoroughly before live trading
- Never risk money you can't afford to lose
- Past performance does not guarantee future results
- Backtesting results may not reflect live trading performance

Happy evolving! 🧬📈
