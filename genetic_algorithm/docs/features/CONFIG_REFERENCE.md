# GA Configuration Reference Guide

This document provides a comprehensive reference for all configuration options available in the Genetic Algorithm strategy optimizer.

---

## Part 1: Quick Reference

### Genetic Algorithm Parameters

| Option | Description |
|--------|-------------|
| `random_seed` | Set an integer for reproducible results, or `null` for random behavior. |
| `population_size` | Number of trading strategies maintained in each generation. |
| `generations` | Total number of evolutionary cycles to run. |
| `mutation_rate` | Probability (0-1) that a strategy will be randomly altered. |
| `crossover_rate` | Probability (0-1) that two parent strategies will combine genes. |
| `elite_size` | Number of top-performing strategies preserved unchanged between generations. |
| `tournament_size` | Number of strategies competing in tournament selection. |
| `selection_method` | Algorithm for choosing parents: `tournament`, `roulette`, or `rank`. |
| `convergence_patience` | Stop early if no improvement after this many generations. |
| `adaptive_mutation` | Automatically increase mutation rate when evolution stagnates. |
| `max_adaptation_factor` | Maximum multiplier for adaptive mutation rate. |
| `adaptation_step` | How much to increase mutation rate per stagnant generation. |
| `fitness_sharing` | Reduce fitness of similar strategies to maintain diversity. |
| `sharing_radius` | Similarity threshold (0-1) for fitness sharing. |
| `diversity_threshold` | Minimum genetic diversity level to maintain. |
| `allow_self_crossover` | Whether the same parent can be selected twice for crossover. |
| `random_immigrants` | Fresh random strategies injected each generation to prevent stagnation. |
| `mode` | Evolution mode: `single_objective` (weighted fitness) or `nsga2` (Pareto front). |

### NSGA-II Multi-Objective Configuration

| Option | Description |
|--------|-------------|
| `objectives` | List of metrics to optimize simultaneously with type and scale. |
| `pareto_front_size` | Number of Pareto-optimal strategies to return. |
| `crowding_distance_percentile` | Reserved for future diversity calculations. |

### Fitness Weights

| Option | Description |
|--------|-------------|
| `profit` | Weight for total return percentage in fitness calculation. |
| `sharpe_ratio` | Weight for risk-adjusted returns (volatility-normalized). |
| `sortino_ratio` | Weight for downside risk-adjusted returns. |
| `profit_factor` | Weight for gross profit / gross loss ratio. |
| `drawdown` | Weight for maximum drawdown penalty. |
| `win_rate` | Weight for percentage of profitable trades. |
| `trade_frequency` | Weight for number of trades (penalizes extremes). |

### Fitness Penalties

| Option | Description |
|--------|-------------|
| `min_trades` | Penalize strategies with fewer trades than this threshold. |
| `max_drawdown` | Penalize strategies exceeding this drawdown percentage. |
| `min_win_rate` | Penalize strategies below this win rate. |
| `complexity_weight` | Penalty multiplier for complex strategies (many indicators/conditions). |

### Backtesting Configuration

| Option | Description |
|--------|-------------|
| `timerange` | Date range for backtesting in YYYYMMDD-YYYYMMDD format. |
| `stake_amount` | Amount of base currency to use per trade. |
| `pairs` | List of trading pairs to test strategies on. |
| `max_open_trades` | Global limit on concurrent open trades. |
| `fee` | Trading fee percentage (e.g., 0.001 = 0.1%). |
| `exchange` | Exchange name for data and configuration. |
| `auto_download_data` | Automatically download missing historical data. |
| `enable_cache` | Cache backtest results to speed up re-evaluation. |
| `timeout` | Maximum seconds allowed per individual backtest. |

### Walk-Forward Optimization

| Option | Description |
|--------|-------------|
| `enabled` | Enable out-of-sample validation via walk-forward analysis. |
| `train_days` | Number of days in each training window. |
| `validation_days` | Number of days in each validation window. |
| `step_days` | Days to shift the window forward between iterations. |
| `mode` | Window type: `rolling` (fixed size) or `anchored` (expanding). |
| `aggregation` | How to combine validation scores: `mean`, `min`, `harmonic_mean`, `weighted`. |
| `min_train_trades` | Skip windows with fewer trades than this. |
| `weights` | Custom weights for `weighted` aggregation (optional). |

### Strategy Constraints

| Option | Description |
|--------|-------------|
| `min_trades` | Minimum trades required for a valid strategy. |
| `max_drawdown` | Maximum allowable drawdown percentage. |
| `min_win_rate` | Minimum required win rate. |
| `timeframes` | Allowed trading timeframes strategies can use. |
| `stoploss_range` | Min/max range for stop-loss percentage. |
| `roi_range` | Min/max range for return-on-investment targets. |
| `max_open_trades_range` | Range for per-strategy concurrent trade limits. |

### Multi-Timeframe Configuration

| Option | Description |
|--------|-------------|
| `enabled` | Allow strategies to use indicators from multiple timeframes. |
| `available` | List of higher timeframes available for analysis. |
| `max_timeframes` | Maximum number of informative timeframes per strategy. |
| `higher_timeframe_preference` | Indicators preferred on higher timeframes. |

### Indicator Configuration

| Option | Description |
|--------|-------------|
| `available` | List of technical indicators strategies can use. |
| `max_per_strategy` | Maximum indicators allowed per strategy. |
| `min_per_strategy` | Minimum indicators required per strategy. |
| `[INDICATOR_NAME]` | Parameter ranges for each specific indicator. |

### Storage Configuration

| Option | Description |
|--------|-------------|
| `database` | SQLite database file path for storing strategies. |
| `strategy_dir` | Directory where generated strategy files are saved. |
| `checkpoint_dir` | Directory for evolution checkpoints. |
| `checkpoint_interval` | Save checkpoint every N generations. |
| `keep_history` | Retain full history of all generations. |

### Logging Configuration

| Option | Description |
|--------|-------------|
| `level` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `file` | Path to log file. |
| `console` | Enable console output. |
| `format` | Python logging format string. |

### Visualization Configuration

| Option | Description |
|--------|-------------|
| `enabled` | Enable real-time plotting during evolution. |
| `update_interval` | Regenerate plots every N generations. |
| `plots` | List of plot types to generate. |
| `output_dir` | Directory to save plot images. |

### Advanced Features

| Option | Description |
|--------|-------------|
| `parallel_evaluation` | Evaluate multiple strategies simultaneously. |
| `num_workers` | Number of parallel worker processes. |
| `enable_dry_run` | Test top strategies in dry-run mode. |
| `dry_run_duration` | Days to run dry-run testing. |
| `enable_ml_fitness` | Use machine learning for fitness prediction. |
| `enable_llm` | Use LLM for strategy generation assistance. |
| `llm` | LLM provider configuration (provider, api_key, model). |
| `island_model` | Island model settings for parallel evolution. |

---

## Part 2: Detailed Configuration Guide

### Genetic Algorithm Parameters

#### `random_seed`

**What it does:** Controls the random number generator seed for the entire evolution process.

**What it's good for:** Setting a fixed seed (e.g., `42`) ensures that running the same configuration produces identical results, which is essential for:
- Debugging and troubleshooting
- Comparing different configurations fairly
- Reproducing research results
- Sharing experiments with others

**Combinations:** Works with all other settings. Set to `null` for production runs where you want true randomization.

**Reference values:**
- `null` - Random behavior (recommended for production)
- `42` - Common convention for reproducible experiments
- Any integer - Your choice for reproducibility

**Example:**
```yaml
genetic_algorithm:
  random_seed: 42  # Reproducible experiment
```

---

#### `population_size`

**What it does:** Determines how many individual trading strategies exist in each generation. Each strategy is a unique combination of indicators, entry/exit conditions, and trade parameters.

**What it's good for:** 
- Larger populations explore more of the search space but take longer to evaluate
- Smaller populations evolve faster but may miss optimal solutions
- Population size affects genetic diversity - too small leads to premature convergence

**Combinations:** 
- Combine with higher `elite_size` (10-15% of population) to preserve winners
- Use larger populations with `fitness_sharing` to maintain diversity
- Balance with `generations` - more population OR more generations for thorough search

**Reference values:**
| Use Case | Population Size |
|----------|----------------|
| Quick test | 10-20 |
| Standard optimization | 30-50 |
| Thorough search | 100-200 |
| Research/exhaustive | 500+ |

**Example:**
```yaml
genetic_algorithm:
  population_size: 50
  elite_size: 6       # ~12% are elite
  generations: 20     # More generations with moderate population
```

---

#### `generations`

**What it does:** Sets the maximum number of evolutionary cycles. Each generation involves selection, crossover, mutation, and fitness evaluation of the entire population.

**What it's good for:**
- More generations allow incremental improvements to compound
- Enables strategies to evolve from random combinations to optimized solutions
- Should be enough for fitness to plateau (convergence)

**Combinations:**
- Use with `convergence_patience` to stop early when improvement stalls
- Higher mutation rate needs more generations to explore
- Larger populations may need fewer generations (already exploring more)

**Reference values:**
| Data Complexity | Generations |
|-----------------|-------------|
| Simple (1 pair, short timerange) | 5-10 |
| Medium (few pairs, months) | 15-30 |
| Complex (many pairs, years) | 50-100 |

**Example:**
```yaml
genetic_algorithm:
  generations: 25
  convergence_patience: 8  # Stop early if no improvement for 8 gens
```

---

#### `mutation_rate`

**What it does:** Probability (0.0 to 1.0) that each strategy will undergo random modification after crossover. Mutations alter indicator parameters, add/remove conditions, or change trade settings.

**What it's good for:**
- Introduces new genetic material to prevent stagnation
- Helps escape local optima (suboptimal solutions)
- Maintains exploration throughout evolution

**Combinations:**
- Enable `adaptive_mutation` to automatically increase when stuck
- Higher mutation with smaller populations to compensate for less diversity
- Balance with `crossover_rate` - both are exploration mechanisms

**Reference values:**
| Scenario | Mutation Rate |
|----------|---------------|
| Conservative (stable optimization) | 0.05-0.10 |
| Standard | 0.10-0.20 |
| Aggressive (high diversity needed) | 0.20-0.35 |
| Very aggressive | 0.40+ (can disrupt good solutions) |

**Example:**
```yaml
genetic_algorithm:
  mutation_rate: 0.18
  adaptive_mutation: true  # Will increase if stuck
  max_adaptation_factor: 2.5  # Can reach 0.45 (0.18 * 2.5)
```

---

#### `crossover_rate`

**What it does:** Probability that two parent strategies will exchange genetic material to create offspring. Crossover combines indicators and conditions from both parents.

**What it's good for:**
- Primary mechanism for combining good traits from different strategies
- Higher rates accelerate convergence toward good solutions
- Lower rates preserve existing good combinations

**Combinations:**
- Balance with `mutation_rate` - together they control exploration vs exploitation
- Works with `selection_method` to determine which parents are chosen
- Higher crossover + lower mutation = faster convergence
- Lower crossover + higher mutation = more exploration

**Reference values:**
| Scenario | Crossover Rate |
|----------|----------------|
| Low (exploitation focus) | 0.4-0.6 |
| Standard | 0.6-0.8 |
| High (rapid combination) | 0.8-0.95 |

**Example:**
```yaml
genetic_algorithm:
  crossover_rate: 0.75
  mutation_rate: 0.15
  # 75% of offspring come from crossover, 15% get mutated
```

---

#### `elite_size`

**What it does:** Number of top-performing strategies copied unchanged to the next generation. Elites bypass selection, crossover, and mutation.

**What it's good for:**
- Guarantees best solutions are never lost
- Provides stability between generations
- Prevents regression (losing good progress)

**Combinations:**
- Should be ~10-15% of population_size
- Higher elite_size = more stability, less exploration
- Works with `random_immigrants` to balance preservation with diversity

**Reference values:**
| Population Size | Recommended Elite |
|-----------------|-------------------|
| 15-20 | 2-3 |
| 30-50 | 4-6 |
| 100 | 10-15 |
| 200+ | 20-30 |

**Example:**
```yaml
genetic_algorithm:
  population_size: 50
  elite_size: 6  # Top 6 (12%) preserved each generation
```

---

#### `tournament_size`

**What it does:** In tournament selection, this many random strategies compete, and the best one becomes a parent. Larger tournaments create stronger selection pressure.

**What it's good for:**
- Controls selection pressure (how strongly fitness matters)
- Larger = faster convergence to good solutions but less diversity
- Smaller = more randomness, more diversity, slower convergence

**Combinations:**
- Only used when `selection_method: 'tournament'`
- Balance with `fitness_sharing` - higher tournament needs sharing for diversity
- Combine with larger populations for high tournament sizes

**Reference values:**
| Tournament Size | Effect |
|-----------------|--------|
| 2-3 | Low pressure, high diversity |
| 4-5 | Moderate pressure (recommended) |
| 7-10 | High pressure, fast convergence |
| 15+ | Very aggressive, may lose diversity |

**Example:**
```yaml
genetic_algorithm:
  selection_method: 'tournament'
  tournament_size: 4  # Moderate selection pressure
  fitness_sharing: true  # Maintain diversity despite pressure
```

---

#### `selection_method`

**What it does:** Algorithm used to choose parent strategies for reproduction.

**Options:**
- **`tournament`**: Random groups compete, winner reproduces. Most common choice.
- **`roulette`**: Probability proportional to fitness. Can be dominated by single excellent strategy.
- **`rank`**: Selection based on fitness rank, not absolute value. Reduces dominance.

**What it's good for:**
- `tournament` is the go-to choice - predictable, tunable via tournament_size
- `roulette` for when fitness differences are meaningful
- `rank` when fitness scale varies widely

**Combinations:**
- `tournament` + `tournament_size` for fine control
- `rank` + high `population_size` for steady improvement
- All methods work with `fitness_sharing`

**Reference values:**
```yaml
# Recommended for most cases
selection_method: 'tournament'
tournament_size: 4

# For fitness with meaningful differences
selection_method: 'roulette'

# When fitness scale varies widely
selection_method: 'rank'
```

---

#### `adaptive_mutation`

**What it does:** Automatically increases mutation rate when evolution stagnates (no improvement in best fitness).

**What it's good for:**
- Self-correcting exploration mechanism
- Breaks out of local optima automatically
- Reduces need to tune mutation rate manually

**Combinations:**
- Use with `max_adaptation_factor` to cap maximum mutation
- Use with `adaptation_step` to control ramp-up speed
- Complementary to `random_immigrants`

**Reference values:**
```yaml
adaptive_mutation: true
max_adaptation_factor: 2.0  # Mutation can double
adaptation_step: 0.1         # +10% increase per stagnant generation
```

---

#### `fitness_sharing`

**What it does:** Reduces fitness scores for strategies that are genetically similar to others. Forces the population to spread across different solution niches.

**What it's good for:**
- Prevents the entire population from converging to one solution
- Discovers multiple viable strategies in different regions of search space
- Critical for multi-modal optimization (many local optima)

**Combinations:**
- Adjust `sharing_radius` to control similarity threshold
- Higher `tournament_size` + fitness_sharing = balanced pressure
- Works well with NSGA-II mode for diverse Pareto fronts

**Reference values:**
```yaml
fitness_sharing: true
sharing_radius: 0.3  # 30% similarity threshold
diversity_threshold: 0.15  # Alert/boost immigrants if diversity drops below 15%
```

---

#### `random_immigrants`

**What it does:** Injects completely random new strategies each generation, bringing fresh genetic material.

**What it's good for:**
- Prevents premature convergence
- Maintains diversity even after many generations
- Enables discovering new indicator combinations late in evolution

**Combinations:**
- Works with `diversity_threshold` - doubles when diversity is low
- Lower `elite_size` to make room for immigrants
- Essential for long evolutions (many generations)

**Reference values:**
| Population Size | Recommended Immigrants |
|-----------------|------------------------|
| 15-20 | 2-3 |
| 30-50 | 4-6 |
| 100+ | 8-12 |

---

#### `mode`

**What it does:** Selects between single-objective optimization (weighted fitness) and multi-objective optimization (NSGA-II Pareto front).

**Options:**
- **`single_objective`**: Combines all metrics into one fitness score using weights. Returns best single strategy.
- **`nsga2`**: Optimizes multiple objectives simultaneously. Returns set of Pareto-optimal strategies.

**What it's good for:**
- `single_objective` when you have clear preferences (e.g., profit > risk)
- `nsga2` when you want to explore trade-offs (profit vs drawdown vs consistency)

**Combinations:**
- `single_objective` uses `fitness_weights` section
- `nsga2` uses `nsga2.objectives` section

**Example:**
```yaml
# Single objective - weighted combination
genetic_algorithm:
  mode: 'single_objective'
fitness_weights:
  profit: 0.30
  sharpe_ratio: 0.25
  drawdown: 0.20
  win_rate: 0.15
  trade_frequency: 0.10

# Multi-objective - Pareto optimization
genetic_algorithm:
  mode: 'nsga2'
nsga2:
  objectives:
    - name: 'profit'
      type: 'maximize'
      scale: 100.0
    - name: 'max_drawdown'
      type: 'minimize'
      scale: 1.0
```

---

### NSGA-II Configuration

#### `objectives`

**What it does:** Defines which metrics to optimize and how. Each objective has a name, type, and scaling factor.

**Types:**
- `maximize`: Higher is better (profit, sharpe, win_rate)
- `minimize`: Lower is better (drawdown, losing_trades)
- `goldilocks`: Target a specific range (trade_frequency)

**What it's good for:**
- Discovering the trade-off between profit and risk
- Finding strategies that excel in different scenarios
- Avoiding over-optimization on single metric

**Example:**
```yaml
nsga2:
  objectives:
    - name: 'profit'
      type: 'maximize'
      scale: 100.0  # Profit of 50% → 0.5 normalized
    - name: 'max_drawdown'
      type: 'minimize'
      scale: 1.0    # Drawdown already 0-1
    - name: 'sortino_ratio'
      type: 'maximize'
      scale: 5.0    # Sortino of 3.0 → 0.6 normalized
  pareto_front_size: 30
```

---

### Fitness Weights

#### Understanding the Fitness Function

The single-objective fitness is calculated as:

```
fitness = (profit_weight × normalized_profit) +
          (sharpe_weight × normalized_sharpe) +
          (sortino_weight × normalized_sortino) +
          ...
          - (complexity_penalty)
          - (drawdown_penalty)
          - (low_trade_penalty)
```

**Reference weight distributions:**

| Strategy Style | profit | sharpe | sortino | profit_factor | drawdown | win_rate | trade_freq |
|----------------|--------|--------|---------|---------------|----------|----------|------------|
| Aggressive | 0.40 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| Balanced | 0.29 | 0.13 | 0.13 | 0.10 | 0.15 | 0.10 | 0.10 |
| Conservative | 0.15 | 0.20 | 0.20 | 0.10 | 0.20 | 0.10 | 0.05 |
| Consistency | 0.20 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.05 |

**Example:**
```yaml
# Balanced approach
fitness_weights:
  profit: 0.29
  sharpe_ratio: 0.13
  sortino_ratio: 0.13
  profit_factor: 0.10
  drawdown: 0.15
  win_rate: 0.10
  trade_frequency: 0.10
```

---

### Backtesting Configuration

#### `timerange`

**What it does:** Defines the historical period for backtesting in `YYYYMMDD-YYYYMMDD` format.

**What it's good for:**
- Longer periods capture more market conditions (bull, bear, sideways)
- Should include various market regimes for robust strategies
- Must have downloaded data covering this range

**Combinations:**
- Works with `walk_forward.enabled` for out-of-sample testing
- Ensure data exists for all `pairs` in this range
- Match with downloaded data: `freqtrade list-data --show-timerange`

**Reference values:**
| Use Case | Timerange Length |
|----------|-----------------|
| Quick testing | 1-3 months |
| Standard optimization | 6-12 months |
| Robust strategy | 1-2 years |
| Full validation | 3+ years |

**Example:**
```yaml
backtesting:
  timerange: "20250101-20260101"  # Full year
  # Verify data first: freqtrade list-data --show-timerange
```

---

#### `stake_amount`

**What it does:** Amount of base currency used per trade.

**What it's good for:**
- Affects position sizing and realistic trade execution
- Should match your intended real trading stake
- Impacts fee calculations and profit/loss

**Reference values:**
- For BTC pairs: `0.01` to `0.1` BTC per trade
- For USDT pairs: `10` to `100` USDT per trade
- Use `"unlimited"` for full capital per trade (not recommended)

---

#### `max_open_trades`

**What it does:** Maximum number of concurrent open positions.

**What it's good for:**
- Risk management - limits capital at risk
- Affects strategy fitness (more concurrent trades = more opportunity)
- Should match your risk tolerance and capital

**Combinations:**
- Use with `strategy_constraints.max_open_trades_range` for per-strategy evolution
- Balance with `stake_amount` for total capital deployment

**Reference values:**
| Trading Style | max_open_trades |
|---------------|-----------------|
| Conservative | 1-2 |
| Moderate | 3-5 |
| Aggressive | 6-10 |
| Scalping | 10+ |

---

#### `auto_download_data`

**What it does:** Automatically downloads missing historical data files before backtesting.

**What it's good for:**
- Convenience - no manual data preparation needed
- Ensures data exists for all configured pairs and timeframes
- Reduces setup errors

**Combinations:**
- Requires valid `exchange` configuration
- Internet connection required
- May slow initial run while downloading

**Example:**
```yaml
backtesting:
  auto_download_data: true
  exchange: "binance"
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
```

---

### Walk-Forward Optimization

Walk-forward optimization is a crucial technique for avoiding overfitting. It trains on one period and validates on a following unseen period.

#### `mode`

**Options:**
- **`rolling`**: Fixed-size training window slides forward. Each segment sees same amount of history.
- **`anchored`**: Training window expands from start date. Later segments have more history.

**Example configuration:**
```yaml
walk_forward:
  enabled: true
  train_days: 60
  validation_days: 15
  step_days: 15
  mode: 'rolling'
```

This creates windows like:
```
Window 1: Train Jan-Feb → Validate Mar 1-15
Window 2: Train Feb-Mar → Validate Mar 15-30
Window 3: Train Mar-Apr → Validate Apr 1-15
...
```

#### `aggregation`

**Methods:**
- **`mean`**: Average validation performance. Balanced approach.
- **`min`**: Worst validation performance. Conservative - ensures no bad periods.
- **`harmonic_mean`**: Penalizes inconsistency more than arithmetic mean.
- **`weighted`**: Custom weights for each window (e.g., prioritize recent).

**Example:**
```yaml
walk_forward:
  aggregation: 'weighted'
  # Weights auto-generated if not provided (linear recency)
  # Or specify manually:
  # weights: [0.1, 0.2, 0.3, 0.4]  # Must sum to 1.0
```

---

### Strategy Constraints

#### `stoploss_range` and `roi_range`

**What they do:**
- `stoploss_range`: Min/max stop-loss as negative percentages (e.g., `[-0.20, -0.05]` = 5-20%)
- `roi_range`: Min/max return-on-investment targets (e.g., `[0.01, 0.10]` = 1-10%)

**What they're good for:**
- Bounds the search space for trade parameters
- Prevents unrealistic strategies (e.g., 50% stop-loss)
- Encourages reasonable risk/reward ratios

**Reference values:**
```yaml
# Conservative
stoploss_range: [-0.10, -0.03]
roi_range: [0.01, 0.05]

# Balanced
stoploss_range: [-0.15, -0.05]
roi_range: [0.02, 0.08]

# Aggressive
stoploss_range: [-0.25, -0.08]
roi_range: [0.05, 0.15]
```

---

### Multi-Timeframe Configuration

#### How It Works

Multi-timeframe strategies use indicators from higher timeframes to confirm signals on the trading timeframe.

**Example flow:**
1. Strategy trades on 5m timeframe
2. Uses 1h EMA for trend direction
3. Only takes 5m buy signals when 1h trend is bullish

```yaml
multi_timeframe:
  enabled: true
  available: ['15m', '1h', '4h']
  max_timeframes: 2
  higher_timeframe_preference:
    - 'EMA'    # Good for trend
    - 'SMA'    # Trend baseline
    - 'BBANDS' # Volatility context
    - 'ADX'    # Trend strength
```

---

### Indicator Configuration

#### Available Indicators

Each indicator has parameter ranges that strategies can evolve:

```yaml
indicators:
  available:
    - "RSI"      # Relative Strength Index
    - "MACD"     # Moving Average Convergence Divergence
    - "BBANDS"   # Bollinger Bands
    - "EMA"      # Exponential Moving Average
    - "SMA"      # Simple Moving Average
    - "STOCH"    # Stochastic Oscillator
    - "ATR"      # Average True Range
    - "ADX"      # Average Directional Index
    - "CCI"      # Commodity Channel Index
  
  max_per_strategy: 5
  min_per_strategy: 2
  
  # Example parameter ranges
  RSI:
    period: [7, 21]           # RSI calculation period
    buy_threshold: [20, 40]    # Oversold level for buying
    sell_threshold: [60, 80]   # Overbought level for selling
  
  MACD:
    fast_period: [8, 21]
    slow_period: [21, 50]
    signal_period: [5, 14]
```

**Combinations:**
- Limit `max_per_strategy` to reduce complexity
- Match indicator types with trading style:
  - Trend: EMA, SMA, ADX
  - Momentum: RSI, MACD, STOCH
  - Volatility: BBANDS, ATR

---

### Advanced Features

#### `parallel_evaluation`

**What it does:** Evaluates multiple strategies simultaneously using multiple CPU cores.

**Caution:** FreqTrade backtesting may have its own parallelization. Test thoroughly before enabling.

```yaml
advanced:
  parallel_evaluation: true
  num_workers: 4  # Match your CPU cores
```

#### `island_model`

**What it does:** Runs multiple isolated populations that periodically exchange individuals.

**What it's good for:**
- Maintains diverse solution niches
- Allows different regions of search space to develop independently
- Prevents premature convergence on large problems

```yaml
advanced:
  island_model:
    enabled: true
    num_islands: 4             # 4 separate populations
    migration_interval: 5       # Exchange individuals every 5 generations
    migration_size: 3           # Send 3 individuals between islands
```

---

## Configuration Templates

### Quick Test Configuration
```yaml
genetic_algorithm:
  population_size: 10
  generations: 5
  mutation_rate: 0.20
  crossover_rate: 0.70
  elite_size: 2
  tournament_size: 3
  selection_method: 'tournament'
  random_immigrants: 2

backtesting:
  timerange: "20250101-20250201"
  pairs: ["BTC/USDT"]
  max_open_trades: 1

walk_forward:
  enabled: false
```

### Production Configuration
```yaml
genetic_algorithm:
  population_size: 100
  generations: 50
  mutation_rate: 0.15
  crossover_rate: 0.75
  elite_size: 12
  tournament_size: 5
  selection_method: 'tournament'
  adaptive_mutation: true
  fitness_sharing: true
  random_immigrants: 8
  convergence_patience: 15

backtesting:
  timerange: "20240101-20260101"
  pairs:
    - "BTC/USDT"
    - "ETH/USDT"
    - "SOL/USDT"
  max_open_trades: 5
  auto_download_data: true

walk_forward:
  enabled: true
  train_days: 90
  validation_days: 30
  step_days: 30
  mode: 'rolling'
  aggregation: 'harmonic_mean'
```

### Multi-Objective (NSGA-II) Configuration
```yaml
genetic_algorithm:
  mode: 'nsga2'
  population_size: 100
  generations: 40
  fitness_sharing: true
  sharing_radius: 0.25

nsga2:
  objectives:
    - name: 'profit'
      type: 'maximize'
      scale: 100.0
    - name: 'max_drawdown'
      type: 'minimize'
      scale: 1.0
    - name: 'sharpe_ratio'
      type: 'maximize'
      scale: 5.0
  pareto_front_size: 30
```
---

## Part 6: Regime Detection

### Overview

Regime detection classifies market conditions into distinct states (bullish, bearish, sideways) to enable regime-aware strategy optimization. This helps the GA evaluate strategies across different market conditions and avoid overfitting to a single regime.

### Why Use Regime Detection?

1. **Balanced Evaluation**: Ensures strategies are tested proportionally across bull, bear, and sideways markets
2. **Robustness**: Penalizes strategies that only perform well in one market condition
3. **Adaptability**: Enables regime-specific parameter tuning
4. **Future-Proofing**: Strategies that work across regimes are more likely to perform well out-of-sample

### Available Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `adx_di_hysteresis` | ADX + Directional Index with hysteresis smoothing | **Default - Best balance of accuracy and stability** |
| `rolling_returns` | Cumulative returns over rolling windows | Simple trend detection |
| `hmm` | Hidden Markov Model with multi-feature approach | Probabilistic regime inference |
| `ensemble` | Weighted voting across multiple methods | Maximum robustness |

### Configuration

```yaml
regime_detection:
  enabled: true
  method: 'adx_di_hysteresis'  # Best performer
  
  # Method-specific parameters
  adx_period: 14
  adx_threshold: 25
  di_threshold: 5
  smoothing_window: 3
  
  # For ensemble method
  ensemble_methods:
    - 'adx_di_hysteresis'
    - 'rolling_returns'
    - 'hmm'
  ensemble_weights: [0.5, 0.25, 0.25]
```

### Detection Method Details

#### ADX + DI with Hysteresis (Recommended)

Uses the Average Directional Index (ADX) combined with Plus/Minus Directional Indicators (+DI/-DI) for trend direction. Hysteresis smoothing prevents rapid regime flipping at threshold boundaries.

```python
from genetic_algorithm.utils.regime_detector import RegimeDetector

detector = RegimeDetector(
    method='adx_di_hysteresis',
    adx_period=14,
    adx_threshold=25,
    smoothing_window=3
)
labels = detector.detect(df)  # Returns Series of RegimeType
```

**Validation Results** (BTC/USDT 4h):
- Bullish periods: +0.188% avg bar return
- Bearish periods: -0.128% avg bar return
- Flip rate: 5.9% (stable classifications)

#### Rolling Returns

Simple threshold-based detection using cumulative returns over a lookback window.

```python
detector = RegimeDetector(
    method='rolling_returns',
    lookback_period=20,
    threshold=0.0
)
```

#### Hidden Markov Model (HMM)

Probabilistic model inferring hidden regime states from multiple features (returns, volatility, volume).

```python
detector = RegimeDetector(
    method='hmm',
    n_states=3,
    n_iter=100
)
```

**Note**: HMM requires `hmmlearn` package.

#### Ensemble Method

Combines multiple detection methods via weighted voting for maximum robustness.

```python
detector = RegimeDetector(
    method='ensemble',
    ensemble_methods=['adx_di_hysteresis', 'rolling_returns', 'hmm'],
    ensemble_weights=[0.5, 0.25, 0.25]
)
```

### Regime Types

| Regime | Description | Characteristics |
|--------|-------------|-----------------|
| `bullish` | Uptrend | Positive returns, +DI > -DI, ADX > threshold |
| `bearish` | Downtrend | Negative returns, -DI > +DI, ADX > threshold |
| `sideways` | Ranging | Low ADX, oscillating price |
| `volatile` | High volatility | Rapid price swings, high ATR |
| `uncertain` | Insufficient data | First few bars until indicators warm up |

### Persistence and Analysis

Save and load regime labels for offline analysis:

```python
from genetic_algorithm.utils.regime_detector import (
    RegimeDetector, save_labels_to_parquet, load_labels_from_parquet
)
from pathlib import Path

# Detect and save
detector = RegimeDetector(method='adx_di_hysteresis')
labels = detector.detect(df)
save_labels_to_parquet(
    df, labels, 
    Path('regime_labels.parquet'),
    method='adx_di_hysteresis',
    metadata={'pair': 'BTC/USDT', 'timeframe': '4h'}
)

# Load for analysis
loaded_df, metadata = load_labels_from_parquet(Path('regime_labels.parquet'))
print(metadata['method'])  # 'adx_di_hysteresis'
```

### Integration with GA Fitness

Enable regime-aware fitness evaluation:

```yaml
fitness:
  regime_aware: true
  regime_weights:
    bullish: 0.33
    bearish: 0.33
    sideways: 0.34
```

This ensures strategies are evaluated proportionally across all market conditions rather than being dominated by the most frequent regime in the training data.

### Validation and Plots

Visualization plots are available in `genetic_algorithm/docs/plots/`:
- `regime_detection_validation.png` - Overview with metrics
- `regime_methods_comparison.png` - Comparison of all methods

### Best Practices

1. **Use `adx_di_hysteresis`** as the default method - it provides the best balance of accuracy and stability
2. **Check flip rate** - Should be <10% for stable classifications
3. **Validate conditional returns** - Bullish regime should have higher returns than bearish
4. **Test on multiple pairs** - Good detection methods should work across different assets
5. **Consider ensemble** for maximum robustness in production

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Too many regime flips | Increase `smoothing_window` or decrease `adx_threshold` |
| No bullish/bearish detection | Decrease `adx_threshold` (default 25) |
| Bearish returns > bullish | Data may have issues, try different timeframe |
| HMM convergence warnings | Increase `n_iter` or use more data |