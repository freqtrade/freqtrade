# Max Open Trades as Evolvable Strategy Parameter

## Overview

Added `max_open_trades` as an evolvable parameter in the genetic algorithm. Each strategy can now have its own maximum number of open trades, allowing the GA to discover optimal position sizing strategies.

## Why This Matters

**Previous Limitation**: FreqTrade defaults to allowing only 1 concurrent trade per pair. With 2 pairs (ETH/BTC and LTC/BTC), the maximum concurrent trades was limited to 2, regardless of the `max_open_trades` config setting.

**New Feature**: Each evolved strategy can now have its own `max_open_trades` value (range: 1-10), which is passed directly to FreqTrade during backtesting. This allows:
- Multiple positions per pair
- Different strategies with different position sizing approaches
- GA evolution to discover optimal trade capacity for each strategy

## Implementation Details

### 1. StrategyGene Dataclass
**File**: `genetic_algorithm/core/strategy_gene.py`
```python
@dataclass
class StrategyGene:
    # ... existing fields ...
    max_open_trades: int = 3  # NEW: Evolvable parameter
```

### 2. Configuration
**File**: `genetic_algorithm/config/ga_config.yaml`
```yaml
strategy_constraints:
  max_open_trades_range: [1, 10]  # NEW: Range for max_open_trades
```

### 3. Strategy Generation
**File**: `genetic_algorithm/strategies/generator.py`
- Random value assignment during strategy creation
- Included in generated strategy code

### 4. Mutation
**File**: `genetic_algorithm/core/mutation.py`
- Added mutation logic for `max_open_trades`
- Random integer selection within configured range

### 5. Backtesting Integration
**File**: `genetic_algorithm/evaluation/direct_backtester.py`
- Added `strategy_max_open_trades` parameter to:
  - `backtest_strategy()`
  - `_run_backtest_direct()`
  - `_create_backtest_config()`
- Configuration priority: strategy-specific value > global config value

### 6. Fitness Evaluation
**File**: `genetic_algorithm/evaluation/fitness.py`
- Pass `strategy_gene.max_open_trades` to backtester
- Applied to both standard and walk-forward evaluations

### 7. Generated Strategy Code
**File**: `genetic_algorithm/strategies/generator.py`
- Strategies now include: `max_open_trades = <value>`
- FreqTrade respects this parameter during backtesting

## Usage

No changes needed to run the GA - the feature is automatically enabled:

```bash
python genetic_algorithm/run_ga.py
```

Each strategy will be assigned a random `max_open_trades` value (1-10) during creation. The value can mutate during evolution, allowing the GA to discover optimal position sizing.

## Example Output

Generated strategies now include:
```python
class GAStrategy_Gen1_Ind5(IStrategy):
    timeframe = '15m'
    stoploss = -0.05
    minimal_roi = {"0": 0.10, "30": 0.05, "60": 0.02}
    trailing_stop = False
    max_open_trades = 7  # <-- NEW: Strategy-specific value
```

## Configuration Customization

To change the range of evolvable `max_open_trades` values:

1. Edit `genetic_algorithm/config/ga_config.yaml`
2. Modify `max_open_trades_range: [min, max]`

Example: Allow 1-20 concurrent trades:
```yaml
strategy_constraints:
  max_open_trades_range: [1, 20]
```

## Technical Notes

- Default value: 3 (if not specified)
- Mutation rate: Same as other integer parameters
- Passed through entire evaluation pipeline
- Overrides global `max_open_trades` from GA config during strategy backtest
- Applies to both training and validation windows in walk-forward optimization

## Files Modified

1. `genetic_algorithm/core/strategy_gene.py` - Added field
2. `genetic_algorithm/config/ga_config.yaml` - Added range config
3. `genetic_algorithm/strategies/generator.py` - Random assignment + code generation
4. `genetic_algorithm/core/mutation.py` - Mutation logic
5. `genetic_algorithm/evaluation/direct_backtester.py` - Parameter threading
6. `genetic_algorithm/evaluation/fitness.py` - Pass to backtester

## Verification

Run a test with minimal configuration to verify:
```bash
python genetic_algorithm/run_ga.py
```

Check the generated strategy files in `user_data/strategies/` to confirm `max_open_trades` is present and varies between strategies.
