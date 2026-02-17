# Genetic Algorithm Improvements Summary

## Overview
This document summarizes the improvements made to address visualization and profit calculation issues in the FreqTrade Genetic Algorithm fork.

## Issues Addressed

### 1. Live Visualization Not Working
**Problem**: Visualization saved images only at the end, with no live updates during evolution.

**Solution Implemented**:
- Enhanced matplotlib update mechanism with proper `canvas.draw()` and `flush_events()` calls
- Increased pause time from 0.01s to 0.1s for better visibility of updates
- Added intermediate plot saving in non-interactive mode (saves plot after each generation)
- Plots now refresh properly during evolution, showing real-time progress

**Files Changed**:
- `genetic_algorithm/visualization/visualizer.py`

**Key Code Changes**:
```python
# Before:
plt.draw()
plt.pause(0.01)

# After:
self.fig.canvas.draw()
self.fig.canvas.flush_events()
plt.pause(0.1)  # Longer pause for visibility

# Plus: intermediate plot saving for non-interactive mode
if not self.interactive and self.save_plots:
    intermediate_file = self.output_dir / f"ga_evolution_gen{generation}_{timestamp}.png"
    self.fig.savefig(intermediate_file, dpi=100, bbox_inches='tight')
```

### 2. Profit Values Staying at 0%
**Problem**: Strategies consistently showed 0% profit, indicating either:
- Profit calculation issues
- Strategies not generating any trades
- Issues with backtest result parsing

**Solutions Implemented**:

#### a) Enhanced Profit Parsing and Logging
- Added comprehensive debug logging for backtest results
- Improved profit percentage conversion (handles both ratio and percentage formats)
- Added automatic win rate calculation when not provided by backtest
- Better error messages when strategies generate no trades

**Files Changed**:
- `genetic_algorithm/evaluation/direct_backtester.py`

**Key Changes**:
```python
# Convert profit ratio to percentage if needed
profit_percent = profit_total * 100 if abs(profit_total) < 10 else profit_total

# Calculate win rate if not provided
if win_rate == 0.0 and total_trades > 0:
    win_rate = wins / total_trades

# Enhanced logging
logger.info(f"Parsed {strategy_name}: profit={profit_percent:.2f}%, trades={total_trades}")
logger.debug(f"Raw backtest stats: {stats}")
```

#### b) Improved Strategy Generation for Trade Generation
Strategies were too restrictive (using AND logic with multiple conditions), leading to few/no trades.

**Changes Made**:
1. **More Lenient Conditions**: Changed default logic from AND to OR (2/3 probability)
2. **Fewer Conditions**: Reduced from 1-3 conditions to 1-2 conditions per signal
3. **Added More Indicators**: Expanded valid indicators to include BBANDS, EMA, SMA
4. **Better Condition Generation**: Added crossover conditions for moving averages and Bollinger Bands

**Files Changed**:
- `genetic_algorithm/strategies/generator.py`

**Key Changes**:
```python
# Before: 1-3 conditions with AND logic
num_conditions = random.randint(1, min(3, len(valid_indicators)))
logic = random.choice(['AND', 'OR'])  # 50/50

# After: 1-2 conditions with OR logic preference
num_conditions = random.randint(1, min(2, len(valid_indicators)))
primary_logic = random.choice(['OR', 'OR', 'AND'])  # 2/3 chance of OR

# Added support for BBANDS, EMA, SMA in conditions
valid_indicators = [ind for ind in indicators 
                   if ind.type in ['RSI', 'MACD', 'STOCH', 'CCI', 'ADX', 
                                  'BBANDS', 'EMA', 'SMA']]
```

### 3. Strategy Generation and Evolution Improvements

#### a) Enhanced Fitness Function
**Changes**:
- Better normalization ranges (profit: -50% to +200%, Sharpe: -5 to 10)
- Added profit bonuses: 10% bonus for positive profit, 20% additional for >10% profit
- Improved trade frequency scoring (5-50 trades optimal, was 20-50)
- More gradual penalty system

**Files Changed**:
- `genetic_algorithm/evaluation/fitness.py`

**Key Improvements**:
```python
# Bonuses for profitable strategies
if profit > 0:
    fitness *= 1.1  # 10% bonus
if profit > 10:
    fitness *= 1.2  # Additional 20% bonus

# Better trade frequency normalization
if 10 <= num_trades <= 50:
    return 1.0  # Full score for ideal range
elif num_trades < 5:
    return num_trades / 10  # Gradual penalty
```

#### b) Relaxed Constraint Thresholds
Made the fitness penalties less harsh to allow more diverse strategies:
- **Minimum trades**: Reduced from 10 to 5
- **Maximum drawdown**: Increased from 25% to 30%
- **Minimum win rate**: Reduced from 35% to 30%

**Files Changed**:
- `genetic_algorithm/config/ga_config.yaml`

#### c) Adaptive Mutation Rate
Implemented adaptive mutation that increases when evolution stagnates:
- Starts at base mutation rate (0.15)
- Increases by 10% per generation with no improvement
- Can increase up to 2x base rate (0.30 max with default 0.5 cap)
- Resets to base rate when improvement is found

**Files Changed**:
- `genetic_algorithm/core/evolution.py`

**Implementation**:
```python
if self.no_improvement_count > 0:
    adaptation_factor = min(2.0, 1.0 + (self.no_improvement_count * 0.1))
    self.mutation_rate = min(0.5, self.base_mutation_rate * adaptation_factor)
else:
    self.mutation_rate = self.base_mutation_rate
```

## Summary of Changed Files

1. **genetic_algorithm/visualization/visualizer.py**
   - Enhanced live plot updates
   - Added intermediate plot saving

2. **genetic_algorithm/evaluation/direct_backtester.py**
   - Improved profit parsing and percentage conversion
   - Added comprehensive logging
   - Better error handling for no-trade scenarios

3. **genetic_algorithm/strategies/generator.py**
   - Made conditions more lenient (OR logic)
   - Reduced condition count
   - Added BBANDS, EMA, SMA support
   - Improved condition code generation

4. **genetic_algorithm/evaluation/fitness.py**
   - Enhanced fitness calculation with bonuses
   - Improved normalization ranges
   - Better trade frequency scoring
   - More gradual penalty system

5. **genetic_algorithm/core/evolution.py**
   - Added adaptive mutation rate
   - Improved convergence detection
   - Better tracking of best fitness

6. **genetic_algorithm/config/ga_config.yaml**
   - Relaxed penalty thresholds
   - Added adaptive_mutation setting

## Expected Results

### Before Improvements:
- ❌ No live visualization during evolution
- ❌ Strategies showing 0% profit consistently
- ❌ Few or no trades generated
- ❌ Harsh penalties killing good strategies
- ❌ Evolution getting stuck easily

### After Improvements:
- ✅ Live visualization updates after each generation
- ✅ Proper profit calculation and display
- ✅ Strategies generate reasonable number of trades (5-50)
- ✅ More diverse strategy population
- ✅ Adaptive mutation helps escape local optima
- ✅ Better fitness scores for profitable strategies
- ✅ More lenient constraints allow exploration

## Usage

### Running with Live Visualization:
```bash
python genetic_algorithm/run_ga.py --visualize
```

### Running with Non-Interactive Visualization (saves plots):
```bash
python genetic_algorithm/run_ga.py --visualize --no-interactive
```

### Running without Visualization:
```bash
python genetic_algorithm/run_ga.py
```

## Configuration Tuning

Users can adjust these parameters in `genetic_algorithm/config/ga_config.yaml`:

### For More Trade Generation:
```yaml
fitness_penalties:
  min_trades: 3  # Lower threshold
  
indicators:
  min_per_strategy: 2  # Fewer indicators = simpler strategies
```

### For More Aggressive Mutation:
```yaml
genetic_algorithm:
  mutation_rate: 0.20  # Increase from default 0.15
  adaptive_mutation: true
```

### For Stricter Quality Control:
```yaml
fitness_penalties:
  min_trades: 10
  max_drawdown: 0.20  # Lower tolerance
  min_win_rate: 0.40  # Higher requirement
```

## Technical Details

### Visualization Update Flow:
1. Evolution loop calls `visualizer.update(generation, stats, population)`
2. Visualizer updates all 4 subplots:
   - Fitness evolution (best/avg/worst)
   - Population diversity
   - Performance metrics (profit, Sharpe, win rate, drawdown)
   - Fitness distribution histogram
3. Canvas draws and flushes events
4. Plot pauses to allow UI update
5. (Non-interactive mode) Saves intermediate plot to disk

### Profit Calculation Flow:
1. Strategy generated from genetic representation
2. Backtest executed via FreqTrade API
3. Results parsed from backtest stats dictionary
4. Profit converted to percentage if needed
5. Metrics passed to fitness function
6. Fitness calculated with bonuses and penalties

### Adaptive Mutation Flow:
1. Track best fitness across generations
2. Increment no_improvement_count if current best ≤ previous best
3. Increase mutation rate by 10% per generation without improvement
4. Cap at 2x base rate (or 0.5, whichever is lower)
5. Reset to base rate when improvement found

## Testing Recommendations

1. **Quick Test** (5-10 minutes):
   ```bash
   # Modify run_ga.py to use:
   POPULATION_SIZE = 10
   GENERATIONS = 5
   python genetic_algorithm/run_ga.py --visualize
   ```

2. **Full Evolution** (2-4 hours):
   ```bash
   # Use defaults:
   POPULATION_SIZE = 50
   GENERATIONS = 20
   python genetic_algorithm/run_ga.py --visualize
   ```

3. **Check Outputs**:
   - Plots saved to: `genetic_algorithm/output/plots/`
   - Strategies saved to: `genetic_algorithm/output/`
   - Logs saved to: `genetic_algorithm/logs/`

## Future Enhancements

Potential additional improvements:
1. Multi-objective optimization (Pareto front)
2. Island model for parallel populations
3. Niching to maintain diversity
4. Ensemble strategies (combination of top performers)
5. Online learning from live trading results
6. Meta-learning for parameter optimization

## Credits

Improvements implemented to address user feedback:
- Live visualization enabling real-time monitoring
- Profit calculation fixes for accurate performance tracking
- Strategy generation improvements for trade generation
- Fitness function enhancements for better evolution
- Adaptive mutation for escaping local optima
