# Walk-Forward Optimization Guide

## What is Walk-Forward Optimization?

Walk-forward optimization is a technique to **prevent overfitting** in trading strategy development. Without it, evolved strategies often look great in backtests but fail in live trading.

### The Problem: Data Snooping Bias

Traditional backtesting trains and tests on the same data period:
```
[=================== Full Data Period ===================]
         Train & Test on entire period
         ❌ Strategy "sees" all data during evolution
         ❌ High backtest performance, poor live performance
```

### The Solution: Walk-Forward Validation

Walk-forward splits the data into multiple rolling windows:
```
Window 1:  [====== Train ======][=== Val ===]
Window 2:         [====== Train ======][=== Val ===]
Window 3:                [====== Train ======][=== Val ===]
Window 4:                       [====== Train ======][=== Val ===]

✅ Train on past data, validate on future (unseen) data
✅ Fitness = validation score, NOT training score
✅ Multiple windows ensure robustness
```

## Quick Start

### 1. Enable Walk-Forward in Configuration

Edit `genetic_algorithm/config/ga_config.yaml`:

```yaml
walk_forward:
  enabled: true              # Enable walk-forward optimization
  train_days: 60            # 60 days for training
  validation_days: 15       # 15 days for validation
  step_days: 15             # Slide forward by 15 days
  mode: 'rolling'           # 'rolling' or 'anchored'
  aggregation: 'mean'       # 'mean', 'min', 'harmonic_mean'
  min_train_trades: 10      # Skip windows with < 10 training trades
```

### 2. Ensure Sufficient Data

Walk-forward requires more data than standard backtesting:

- **Minimum**: `train_days + validation_days` (e.g., 75 days for one window)
- **Recommended**: `train_days + (num_desired_windows * step_days) + validation_days`
  - Example: 60 + (5 × 15) + 15 = **150 days** for 5 windows

Update your timerange accordingly:
```yaml
backtesting:
  timerange: "20230101-20230531"  # 5 months of data
```

### 3. Run Evolution

```bash
# Standard usage - walk-forward uses config setting
python genetic_algorithm/run_ga.py

# With visualization
python genetic_algorithm/run_ga.py --visualize
```

The GA will automatically use walk-forward if enabled in config.

## Configuration Options

### Window Modes

#### Rolling Window (Recommended)
Fixed-size training window that slides forward:
```
Window 1: [======60 days======][==15==]
Window 2:         [======60 days======][==15==]
Window 3:                [======60 days======][==15==]
```

**Pros:** Adapts to recent market conditions  
**Use when:** Market regime changes over time

```yaml
walk_forward:
  mode: 'rolling'
  train_days: 60
```

#### Anchored Window
Expanding training window from start:
```
Window 1: [==30==][=10=]
Window 2: [======40======][=10=]
Window 3: [==========50==========][=10=]
```

**Pros:** Uses all available history  
**Use when:** Long-term patterns are important

```yaml
walk_forward:
  mode: 'anchored'
  train_days: 30  # Initial size, grows each window
```

### Aggregation Methods

How validation scores are combined across windows:

#### Mean (Default)
```python
fitness = (val₁ + val₂ + val₃) / 3
```
**Balanced approach.** Use for most cases.

#### Min (Conservative)
```python
fitness = min(val₁, val₂, val₃)
```
**Worst-case performance.** Use for risk-averse strategies.

#### Harmonic Mean
```python
fitness = 3 / (1/val₁ + 1/val₂ + 1/val₃)
```
**Penalizes inconsistency.** Use to prefer consistent performers.

#### Weighted
```python
fitness = w₁×val₁ + w₂×val₂ + w₃×val₃
```
**Custom weighting.** Use to emphasize recent windows.

```yaml
walk_forward:
  aggregation: 'weighted'
  weights: [0.2, 0.3, 0.5]  # More weight to recent windows
```

## Expected Results

### Before Walk-Forward
```
Training fitness: 15.0%
Live performance: 3.0%
❌ Massive overfitting (12% gap)
```

### After Walk-Forward
```
Training fitness: 10.0%
Validation fitness: 8.5%
Live performance: 7.0%
✅ Much closer (1.5% gap)
```

**Trade-off:** Lower training fitness, but **much better real-world performance**.

## Performance Considerations

### Evaluation Time

Walk-forward runs multiple backtests per strategy:
- **Standard**: 1 backtest per strategy
- **Walk-forward**: (2 × num_windows) backtests per strategy

**Example with 4 windows:**
- Standard: 20 strategies × 1 backtest = **20 backtests** per generation
- Walk-forward: 20 strategies × 8 backtests = **160 backtests** per generation

**Mitigation:**
1. **Caching**: Training windows are cached (50-70% hit rate for elite individuals)
2. **Reduce population**: Use smaller population size (10-15 instead of 20)
3. **Fewer generations**: Reduce generations (3-5 instead of 10)
4. **Larger step size**: Increase `step_days` to reduce number of windows

### Recommended Settings for Fast Iteration

```yaml
genetic_algorithm:
  population_size: 10     # Reduced from 20
  generations: 3          # Reduced from 10
  
walk_forward:
  enabled: true
  train_days: 45          # Reduced from 60
  validation_days: 10     # Reduced from 15
  step_days: 20           # Increased from 15 (fewer windows)
```

## Interpreting Results

### Walk-Forward Metrics

The GA logs additional metrics when walk-forward is enabled:

```
Walk-forward complete for GAStrategy_Gen2_Ind5:
  Final fitness=0.6234 (train avg=0.7123, val avg=0.6234, gap=0.0889)
```

**Key metrics:**
- **Final fitness**: Aggregated validation score (what GA optimizes)
- **Train avg**: Average training fitness across windows
- **Val avg**: Average validation fitness across windows
- **Gap**: Train - Val (positive = potential overfit, negative = validation exceeds training)

### Interpreting the Gap

- **< 0** (negative): Validation better than training! Rare but excellent
- **0-10%**: Excellent! Strategy generalizes well
- **10-20%**: Good, acceptable level
- **20-30%**: Moderate concern, use cautiously
- **> 30%**: High overfitting risk, strategy likely won't work live

### Cache Performance

```
Walk-forward cache stats: 142 hits, 78 misses
```

- **Hit rate**: 142/(142+78) = 64.5%
- **Good hit rate**: > 50% (elite individuals reused)
- **Low hit rate**: < 30% (population not converging)

## Troubleshooting

### "No valid windows could be created"

**Cause:** Insufficient data for given parameters  
**Solution:** Either:
1. Increase your timerange
2. Reduce `train_days` or `validation_days`
3. Increase `step_days` (fewer windows needed)

### "Insufficient training trades"

**Cause:** Training windows don't generate enough trades  
**Solution:**
1. Reduce `min_train_trades` (default: 10)
2. Use more liquid pairs
3. Relax strategy constraints

### Very slow evolution

**Cause:** Too many backtests per generation  
**Solution:**
1. Reduce population size
2. Increase `step_days` (fewer windows)
3. Reduce `train_days` and `validation_days`
4. Use shorter overall timerange

## Advanced Usage

### Custom Aggregation Weights

Give more importance to recent windows:

```yaml
walk_forward:
  aggregation: 'weighted'
  # 4 windows: oldest → newest
  weights: [0.1, 0.2, 0.3, 0.4]  # Sum must equal 1.0
```

### Combining with Other Features

Walk-forward works with all other GA features:

```yaml
genetic_algorithm:
  fitness_sharing: true        # ✅ Diversity preservation
  adaptive_mutation: true      # ✅ Adaptive mutation
  random_immigrants: 3         # ✅ Fresh strategies
  
walk_forward:
  enabled: true                # ✅ Anti-overfitting
  mode: 'rolling'
```

### Disabling Walk-Forward for Quick Tests

```yaml
walk_forward:
  enabled: false  # Temporarily disable for fast iteration
```

No code changes needed - just toggle the config.

## Best Practices

### 1. Start Simple
Begin with conservative settings:
```yaml
walk_forward:
  enabled: true
  train_days: 45
  validation_days: 10
  step_days: 15
  mode: 'rolling'
  aggregation: 'mean'
```

### 2. Monitor Degradation
Watch the train/val gap. If consistently > 20%, your strategies are overfitting.

### 3. Use Sufficient Data
- Minimum: 3 months for short-term strategies
- Recommended: 6-12 months for robust validation

### 4. Validate on Different Timeranges
After evolution, backtest the best strategy on a completely separate timerange.

### 5. Consider Market Regimes
Use rolling windows (not anchored) if market conditions change significantly.

## References

- [Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*](https://www.amazon.com/Evaluation-Optimization-Trading-Strategies/dp/0470128011) - Chapter on Walk-Forward Analysis
- [Walk-Forward Analysis Explained (QuantConnect)](https://www.quantconnect.com/docs/v2/writing-algorithms/optimization/key-concepts#07-Walk-Forward-Optimization)
- [Overfitting in Trading Strategies](https://www.investopedia.com/articles/trading/08/backtest-overfitting.asp)

## Example Configuration Files

See `genetic_algorithm/config/`:
- `ga_config.yaml` - Default configuration with walk-forward disabled
- `ga_config_walkforward.yaml` - Example with walk-forward enabled (create this yourself)

---

**Next Steps:**
1. Enable walk-forward in your config
2. Run a test evolution with 10 strategies, 2 generations
3. Check degradation metrics
4. Adjust parameters based on results
5. Run full evolution with optimized settings
