# Genetic Algorithm for FreqTrade - Usage Tutorial

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Understanding the Genetic Algorithm](#understanding-the-genetic-algorithm)
6. [Running Evolution](#running-evolution)
7. [Working with Data](#working-with-data)
8. [Using Generated Strategies](#using-generated-strategies)
9. [Configuration](#configuration)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This Genetic Algorithm (GA) system automatically generates and evolves trading strategies for FreqTrade. It uses evolutionary principles to create, test, and improve strategies over multiple generations.

**Key Features:**
- ✅ Generates valid FreqTrade strategies automatically
- ✅ Uses FreqTrade's real backtesting engine (not mocked)
- ✅ Works with actual OHLCV data from files
- ✅ Strategies can be used directly with FreqTrade trading bot
- ✅ Supports multiple indicators (RSI, MACD, Bollinger Bands, etc.)
- ✅ Configurable fitness functions and genetic parameters

---

## Prerequisites

### Software Requirements
- Python 3.11 or higher
- FreqTrade (already installed in this repository)
- TA-Lib (for technical indicators)
- All dependencies from requirements.txt

### Knowledge Requirements
- Basic understanding of trading concepts
- Familiarity with FreqTrade (recommended)
- Basic Python knowledge (for customization)

---

## Installation

1. **Install Dependencies**
   ```bash
   # Install FreqTrade requirements
   pip install -r requirements.txt
   
   # Install genetic algorithm specific requirements
   pip install -r genetic_algorithm/requirements.txt
   ```

2. **Verify Installation**
   ```bash
   # Run verification test
   python genetic_algorithm/test_real_backtest.py
   ```
   
   You should see output indicating that strategies can be generated and backtested successfully.

---

## Quick Start

### 1. Generate a Single Strategy

```python
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml

# Load configuration
with open('genetic_algorithm/config/ga_config.yaml') as f:
    config = yaml.safe_load(f)

# Create generator
generator = StrategyGenerator(config)

# Generate a random strategy
strategy_gene = generator.generate_random_strategy(generation=0, individual_id=1)

# Generate Python code
strategy_code = generator.generate_strategy_code(strategy_gene)

# Save to file
with open('user_data/strategies/MyStrategy.py', 'w') as f:
    f.write(strategy_code)

print("Strategy generated: user_data/strategies/MyStrategy.py")
```

### 2. Backtest a Generated Strategy

```python
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
import yaml

# Load configuration
with open('genetic_algorithm/config/ga_config.yaml') as f:
    config = yaml.safe_load(f)

# Create backtester
backtester = DirectBacktester(config)

# Backtest the strategy
result = backtester.backtest_strategy(strategy_code, "MyStrategy")

# Print results
if result.success:
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Profit: {result.total_profit:.4f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
else:
    print(f"Backtest failed: {result.error_message}")
```

### 3. Run Complete Evolution (Coming Soon)

```python
from genetic_algorithm.core.evolution import GeneticAlgorithm

# Initialize GA with config
ga = GeneticAlgorithm('genetic_algorithm/config/ga_config.yaml')

# Run evolution for N generations
best_strategies = ga.evolve()

# Get top 5 strategies
top_5 = ga.get_top_strategies(n=5)

# Save best strategies
for i, strategy in enumerate(top_5):
    filename = f"user_data/strategies/GA_Top{i+1}.py"
    ga.save_strategy(strategy, filename)
```

---

## Understanding the Genetic Algorithm

### How It Works

1. **Initialization**: Creates a population of random strategies
2. **Evaluation**: Backtests each strategy using FreqTrade
3. **Selection**: Selects the best-performing strategies
4. **Crossover**: Combines strategies to create offspring
5. **Mutation**: Randomly modifies strategies to explore new possibilities
6. **Repeat**: Goes back to step 2 for the next generation

### Strategy Representation

Each strategy is represented as a "gene" with the following components:

- **Indicators**: Technical indicators (RSI, MACD, etc.)
  - Type (RSI, MACD, Bollinger Bands, etc.)
  - Parameters (periods, thresholds)
  - Weight (importance)

- **Entry Conditions**: When to enter a trade
  - Indicator comparisons
  - Threshold values
  - Logic operators (AND/OR)

- **Exit Conditions**: When to exit a trade
  - Similar to entry conditions
  - Can use ROI and stop-loss instead

- **Risk Management**:
  - Stop loss percentage
  - Take profit (ROI) levels
  - Trailing stop settings

- **Timeframe**: Trading timeframe (5m, 15m, 1h, etc.)

### Fitness Function

Strategies are evaluated based on multiple metrics:

- **Profit** (30%): Total profit/return
- **Sharpe Ratio** (25%): Risk-adjusted returns
- **Drawdown** (20%): Maximum drawdown (penalty)
- **Win Rate** (15%): Percentage of winning trades
- **Trade Frequency** (10%): Number of trades (balanced)

These weights can be customized in `config/ga_config.yaml`.

---

## Running Evolution

### Basic Evolution Run

```bash
# Run evolution with default settings
python genetic_algorithm/example_usage.py
```

### Custom Evolution Run

```python
from genetic_algorithm.core.evolution import GeneticAlgorithm

# Load and customize config
config_path = 'genetic_algorithm/config/ga_config.yaml'
ga = GeneticAlgorithm(config_path)

# Override some settings
ga.population_size = 50
ga.generations = 20
ga.mutation_rate = 0.2

# Run evolution
best_strategies = ga.evolve()

# Analyze results
print(f"Best fitness: {best_strategies[0].fitness}")
print(f"Strategy metrics:")
print(f"  - Win Rate: {best_strategies[0].win_rate:.2%}")
print(f"  - Profit: {best_strategies[0].total_profit:.4f}")
print(f"  - Sharpe: {best_strategies[0].sharpe_ratio:.2f}")
```

### Monitoring Progress

The evolution process logs progress to:
- Console output (INFO level)
- Log file: `genetic_algorithm/logs/ga.log`
- Checkpoints: `genetic_algorithm/data/checkpoints/`

You can monitor:
- Current generation
- Best fitness in generation
- Average fitness
- Number of valid strategies
- Execution time per generation

---

## Working with Data

### Using Test Data (Default)

The system includes test data in `tests/testdata/` for immediate use:
- Pairs: UNITTEST/BTC, ETH/BTC, LTC/BTC
- Timeframes: 5m
- Date range: January 2018 (19 days)

This is perfect for:
- Testing the system
- Quick iterations during development
- Verifying strategy logic

### Downloading Real Market Data

**Note**: Due to network restrictions in this environment, downloading data from exchanges may not work directly. However, when you run this on your local machine or server with internet access, you can download real data as follows:

#### Method 1: Using FreqTrade CLI

```bash
# Download data for specific pairs
freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 5m 15m 1h \
  --days 90

# Data will be saved to: user_data/data/binance/
```

#### Method 2: Using the Download Script

```bash
# Edit download_data.py to configure pairs and timeframes
python genetic_algorithm/download_data.py
```

Edit the script to customize:
```python
pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Your pairs
timeframes = ['5m', '1h']  # Your timeframes
exchange = 'binance'  # Your exchange
days = 90  # Days of history
```

#### Method 3: Use Downloaded Data from Another Source

If you already have data:
1. Place CSV/JSON files in `user_data/data/<exchange>/`
2. Update `config/ga_config.yaml` to point to your data directory
3. Update the pairs list in the config

### Configuring Data for Evolution

Edit `genetic_algorithm/config/ga_config.yaml`:

```yaml
backtesting:
  # Data directory (relative to project root)
  datadir: "user_data/data/binance"  # Change as needed
  
  # Trading pairs to use
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
    - "BNB/USDT"
  
  # Time range for backtesting
  timerange: "20230101-20231231"  # Optional: YYYYMMDD-YYYYMMDD
  
  # Stake amount per trade
  stake_amount: 0.05
  
  # Fee for trades
  fee: 0.001
```

---

## Using Generated Strategies

### Locating Generated Strategies

Strategies are saved to:
```
user_data/strategies/ga_generated/GAStrategy_Gen{X}_Ind{Y}.py
```

Where:
- `X` is the generation number
- `Y` is the individual ID

### Copying to FreqTrade

```bash
# Copy a strategy to main strategies directory
cp user_data/strategies/ga_generated/GAStrategy_Gen5_Ind3.py \
   user_data/strategies/MyBestStrategy.py
```

### Backtesting with FreqTrade CLI

```bash
# Backtest the strategy
freqtrade backtesting \
  --strategy MyBestStrategy \
  --timeframe 5m \
  --timerange 20230101-20231231

# View detailed results
freqtrade backtesting-show
```

### Running in Dry-Run Mode

```bash
# Create a config file for dry-run
cp config_examples/config.json user_data/config.json

# Edit config.json to set:
# - "dry_run": true
# - "strategy": "MyBestStrategy"

# Start bot in dry-run mode
freqtrade trade --config user_data/config.json
```

### Deploying to Live Trading

⚠️ **IMPORTANT**: Always test thoroughly before live trading!

1. **Extensive Backtesting**
   - Test with multiple time periods
   - Verify on different market conditions
   - Check for overfitting

2. **Dry-Run Testing**
   - Run in dry-run mode for at least 1-2 weeks
   - Monitor performance and behavior
   - Verify trade logic is sound

3. **Start Small**
   - Begin with minimal stake amounts
   - Use only a small portion of capital
   - Monitor closely for the first few days

4. **Live Trading**
   ```bash
   # Edit config to set dry_run: false
   # Configure exchange API keys
   freqtrade trade --config user_data/config_live.json
   ```

---

## Configuration

### Main Configuration File

`genetic_algorithm/config/ga_config.yaml`

#### Genetic Algorithm Parameters

```yaml
genetic_algorithm:
  population_size: 100      # Number of strategies per generation
  generations: 50           # Number of generations to evolve
  mutation_rate: 0.15       # Probability of mutation (0-1)
  crossover_rate: 0.7       # Probability of crossover (0-1)
  elite_size: 10            # Top N strategies preserved unchanged
  tournament_size: 3        # Tournament selection size
  selection_method: 'tournament'  # or 'roulette', 'rank'
  convergence_patience: 10  # Stop if no improvement after N gens
```

#### Fitness Weights

```yaml
fitness_weights:
  profit: 0.30              # Total profit importance
  sharpe_ratio: 0.25        # Risk-adjusted returns
  drawdown: 0.20            # Max drawdown penalty
  win_rate: 0.15            # Win rate importance
  trade_frequency: 0.10     # Trade count balance
```

#### Strategy Constraints

```yaml
strategy_constraints:
  min_trades: 10                    # Minimum trades for valid strategy
  max_drawdown: 0.25                # Maximum allowed drawdown
  min_win_rate: 0.35                # Minimum win rate
  timeframes: ["5m", "15m", "1h"]   # Allowed timeframes
  stoploss_range: [-0.20, -0.05]    # Stop loss range
  roi_range: [0.01, 0.10]           # ROI range
```

#### Indicator Configuration

```yaml
indicators:
  available:
    - "RSI"
    - "MACD"
    - "BBANDS"
    - "EMA"
    - "SMA"
    - "STOCH"
    - "ATR"
    - "ADX"
    - "CCI"
  
  max_per_strategy: 5
  min_per_strategy: 2
  
  RSI:
    period: [7, 21]
    buy_threshold: [20, 40]
    sell_threshold: [60, 80]
  
  # ... (see config file for all indicators)
```

---

## Advanced Usage

### Custom Fitness Functions

You can create custom fitness evaluations:

```python
from genetic_algorithm.evaluation.fitness import FitnessEvaluator

class CustomFitness(FitnessEvaluator):
    def calculate_fitness(self, backtest_result):
        # Your custom logic
        profit_score = backtest_result.profit_percent / 100
        risk_score = 1 - backtest_result.max_drawdown
        trades_score = min(backtest_result.total_trades / 50, 1)
        
        # Custom weighting
        fitness = (profit_score * 0.5 + 
                  risk_score * 0.3 + 
                  trades_score * 0.2)
        
        return fitness
```

### Parallel Evolution

For faster evolution with multiple cores:

```yaml
advanced:
  parallel_evaluation: true
  num_workers: 4
```

### Island Model

Run multiple populations in parallel with periodic migration:

```yaml
advanced:
  island_model:
    enabled: true
    num_islands: 4
    migration_interval: 5
    migration_size: 5
```

### Checkpointing

Evolution automatically saves checkpoints:

```yaml
storage:
  checkpoint_dir: "genetic_algorithm/data/checkpoints"
  checkpoint_interval: 5  # Save every 5 generations
```

To resume from a checkpoint:

```python
ga = GeneticAlgorithm(config_path)
ga.load_checkpoint('genetic_algorithm/data/checkpoints/gen_25.pkl')
ga.evolve(start_generation=26)
```

---

## Troubleshooting

### Issue: "No module named 'talib'"

**Solution**: Install TA-Lib

```bash
# On Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# On macOS
brew install ta-lib
pip install TA-Lib

# On Windows
# Download from: https://github.com/cgohlke/talib-build/releases
pip install <downloaded-file>.whl
```

### Issue: "No data available for backtesting"

**Solution**: Download or copy data files

```bash
# Option 1: Use test data
# Already available in tests/testdata/

# Option 2: Download new data
freqtrade download-data --exchange binance --pairs BTC/USDT

# Option 3: Copy your own data
cp /path/to/data/*.feather user_data/data/binance/
```

### Issue: "Strategy generates no trades"

**Possible causes**:
1. Entry conditions too strict
2. Data timeframe mismatch
3. Insufficient data

**Solution**: 
- Check strategy entry conditions
- Verify data exists for the timeframe
- Test with known working strategy first

### Issue: "Backtest takes too long"

**Solution**: Optimize settings

```yaml
backtesting:
  enable_cache: true  # Cache results
  
genetic_algorithm:
  population_size: 50  # Reduce population
  
advanced:
  parallel_evaluation: true  # Enable parallel processing
  num_workers: 4
```

### Issue: "All strategies perform poorly"

**Possible causes**:
1. Data quality issues
2. Overly restrictive constraints
3. Poor fitness function weights

**Solution**:
- Verify data integrity
- Relax constraints in config
- Adjust fitness weights
- Run for more generations
- Try different market periods

---

## Next Steps

### Immediate Actions

1. **Verify Installation**
   ```bash
   python genetic_algorithm/test_real_backtest.py
   ```

2. **Generate Your First Strategy**
   ```bash
   python genetic_algorithm/example_usage.py
   ```

3. **Customize Configuration**
   - Edit `config/ga_config.yaml`
   - Adjust population size, generations
   - Modify fitness weights

### Recommended Workflow

1. **Development Phase**
   - Use test data for quick iterations
   - Small population (20-50)
   - Few generations (10-20)
   - Test and refine fitness function

2. **Evolution Phase**
   - Download real market data (90+ days)
   - Larger population (100-200)
   - More generations (50-100)
   - Run evolution overnight

3. **Validation Phase**
   - Backtest top strategies on different periods
   - Test in dry-run mode (1-2 weeks)
   - Compare against baseline strategies

4. **Deployment Phase**
   - Start with small stake amounts
   - Monitor closely
   - Gradually increase if successful

### Resources

- **FreqTrade Documentation**: https://www.freqtrade.io/
- **FreqTrade Strategies**: https://github.com/freqtrade/freqtrade-strategies
- **TA-Lib Indicators**: https://ta-lib.github.io/ta-lib-python/
- **Genetic Algorithms**: https://en.wikipedia.org/wiki/Genetic_algorithm

### Getting Help

- Check the logs: `genetic_algorithm/logs/ga.log`
- Review examples in `genetic_algorithm/examples/`
- Read the code documentation
- FreqTrade Discord: https://discord.gg/p7nuUNVfP7

---

## Disclaimer

**IMPORTANT**: This software is for educational purposes only.

- Do not risk money you cannot afford to lose
- Always test thoroughly before live trading
- Past performance does not guarantee future results
- Genetic algorithms can overfit to historical data
- Market conditions change; strategies need monitoring
- The authors assume no responsibility for trading results

Always start with paper trading and small amounts!

---

## License

This project follows the FreqTrade license. See LICENSE file for details.

---

**Happy Trading! 🚀📈**
