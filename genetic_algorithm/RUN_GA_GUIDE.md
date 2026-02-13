# Genetic Algorithm Runner - Quick Start Guide

## Overview

`run_ga.py` is your pre-configured "run button" to start the Genetic Algorithm for evolving trading strategies. It automates the entire evolution process and outputs the top 5 most successful strategies.

## Quick Start

### 1. Simple Run (with defaults)

```bash
python genetic_algorithm/run_ga.py
```

This will:
- Use default configuration from `genetic_algorithm/config/ga_config.yaml`
- Run 20 generations with population size of 50
- Display the top 5 strategies at the end
- Save strategies to `genetic_algorithm/output/`

### 2. Custom Configuration

Edit the **USER CONFIGURATION** section at the top of `run_ga.py`:

```python
# Basic GA Parameters
POPULATION_SIZE = 50          # Number of strategies per generation
GENERATIONS = 20              # Number of generations to evolve
MUTATION_RATE = 0.15          # Probability of mutation (0.0-1.0)
CROSSOVER_RATE = 0.7          # Probability of crossover (0.0-1.0)
ELITE_SIZE = 5                # Number of top strategies to preserve

# Number of top strategies to display and save
TOP_STRATEGIES_COUNT = 5

# Output configuration
SAVE_STRATEGIES = True        # Save top strategies to files
OUTPUT_DIR = Path("genetic_algorithm/output")
```

## Configuration Options

### GA Parameters

- **POPULATION_SIZE**: Number of strategies in each generation
  - Smaller (20-50): Faster, may converge prematurely
  - Larger (100-200): Slower, more diverse exploration
  - Recommended: 50-100

- **GENERATIONS**: Number of evolution cycles
  - Fewer (10-20): Quick tests, may not find optimal strategies
  - More (50-100): Better results, takes longer
  - Recommended: 20-50

- **MUTATION_RATE**: Probability of random changes
  - Lower (0.05-0.10): Slower exploration, more stable
  - Higher (0.20-0.30): Faster exploration, less stable
  - Recommended: 0.10-0.20

- **CROSSOVER_RATE**: Probability of combining parent strategies
  - Lower (0.5): More mutation-driven evolution
  - Higher (0.8-0.9): More recombination of successful traits
  - Recommended: 0.6-0.8

- **ELITE_SIZE**: Number of best strategies preserved unchanged
  - Should be 5-10% of population size
  - Recommended: 5-10

### Output Options

- **TOP_STRATEGIES_COUNT**: Number of top strategies to display (default: 5)
- **SAVE_STRATEGIES**: Whether to save strategies to files (default: True)
- **OUTPUT_DIR**: Where to save strategy files and reports

## What Happens During a Run

1. **Initialization**
   - Loads configuration
   - Displays current settings
   - Waits for confirmation

2. **Evolution Loop** (for each generation)
   - Evaluates all strategies via backtesting
   - Selects best performers
   - Creates offspring via crossover and mutation
   - Preserves elite strategies

3. **Results**
   - Displays top 5 strategies with metrics
   - Saves strategy Python files
   - Creates summary report

## Output Files

After a successful run, you'll find:

### Strategy Files
`genetic_algorithm/output/strategy_rank1_genX_indY_TIMESTAMP.py`
- Ready-to-use FreqTrade strategy files
- Can be copied to `user_data/strategies/`

### Summary Report
`genetic_algorithm/output/ga_summary_TIMESTAMP.txt`
- Overview of the run
- Top strategies with metrics

### Log File
`genetic_algorithm/logs/ga_run_TIMESTAMP.log`
- Detailed execution log
- Useful for debugging

## Understanding Strategy Metrics

Each strategy is evaluated on multiple metrics:

- **Fitness Score**: Overall quality (0-1, higher is better)
  - Weighted combination of all metrics
  
- **Profit**: Total return percentage
  - Target: 10-50%+

- **Sharpe Ratio**: Risk-adjusted returns
  - Target: 1.0+, excellent: 2.0+

- **Max Drawdown**: Largest peak-to-trough decline
  - Target: < 20%, excellent: < 10%

- **Win Rate**: Percentage of profitable trades
  - Target: > 50%, excellent: > 60%

- **Total Trades**: Number of trades executed
  - Target: 20-50 (not too few, not too many)

- **Profit Factor**: Gross profit / Gross loss
  - Target: > 1.5, excellent: > 2.0

## Next Steps After GA Run

1. **Review Results**
   - Check the displayed top strategies
   - Review metrics and parameters

2. **Backtest with More Data**
   ```bash
   # Copy strategy to user_data/strategies/
   cp genetic_algorithm/output/strategy_rank1_*.py user_data/strategies/
   
   # Run FreqTrade backtest
   freqtrade backtesting --strategy <StrategyName>
   ```

3. **Test in Dry-Run**
   ```bash
   freqtrade trade --dry-run --strategy <StrategyName>
   ```

4. **Validate Performance**
   - Run for several days in dry-run
   - Check if performance matches backtests
   - Monitor risk metrics

5. **Deploy to Live Trading** (only when confident)
   ```bash
   freqtrade trade --strategy <StrategyName>
   ```

## Troubleshooting

### "Configuration file not found"
- Ensure you're running from the repository root
- Check that `genetic_algorithm/config/ga_config.yaml` exists

### "No module named 'genetic_algorithm'"
- Make sure you're in the correct directory
- The script adds the parent directory to Python path

### Evolution is very slow
- Reduce POPULATION_SIZE (try 20-30)
- Reduce GENERATIONS (try 10-15)
- Enable caching in ga_config.yaml

### All strategies have low fitness
- Check backtesting configuration
- Verify data files exist in tests/testdata
- Review fitness weights in ga_config.yaml

### Out of memory
- Reduce POPULATION_SIZE
- Disable parallel evaluation
- Close other applications

## Advanced Configuration

For more advanced configuration, edit `genetic_algorithm/config/ga_config.yaml`:

- **Fitness weights**: Adjust importance of different metrics
- **Indicator constraints**: Modify available indicators and parameters
- **Backtesting settings**: Change trading pairs, timerange, fees
- **Strategy constraints**: Set limits on stop loss, ROI, etc.

## Tips for Best Results

1. **Start Small**: Use smaller population and fewer generations for initial tests
2. **Iterate**: Run multiple times with different configurations
3. **Diversify**: Try different fitness weight combinations
4. **Validate**: Always backtest top strategies with more data
5. **Be Patient**: Good strategies take time to evolve
6. **Monitor**: Watch for overfitting (too good on backtest data)

## Example Workflow

```bash
# Quick test run (5-10 minutes)
# Edit run_ga.py: POPULATION_SIZE=20, GENERATIONS=10
python genetic_algorithm/run_ga.py

# Full run (30-60 minutes)
# Edit run_ga.py: POPULATION_SIZE=50, GENERATIONS=20
python genetic_algorithm/run_ga.py

# Intensive search (several hours)
# Edit run_ga.py: POPULATION_SIZE=100, GENERATIONS=50
python genetic_algorithm/run_ga.py
```

## Support

For issues or questions:
1. Check the log files in `genetic_algorithm/logs/`
2. Review the configuration in `genetic_algorithm/config/ga_config.yaml`
3. Consult the main documentation in `genetic_algorithm/README.md`
