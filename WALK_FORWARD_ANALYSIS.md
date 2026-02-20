# Walk-Forward Backtest Analysis

**Date:** 2026-02-20  
**Test Run:** 1 strategy with walk-forward, multi-timeframe, 2 pairs, 3 timeframes

---

## ✅ VERIFICATION: Everything is Working Correctly!

### Summary
You saw **24 backtests** for 1 strategy, which is **EXPECTED behavior** for walk-forward optimization with 12 windows.

---

## How Walk-Forward Works

### Configuration Used
```yaml
backtesting:
  pairs: ["ETH/BTC", "LTC/BTC"]
  timerange: "20250620-20260218"

walk_forward:
  enabled: true
  train_days: 60
  validation_days: 15
  step_days: 15
  mode: rolling
  aggregation: mean

strategy_constraints:
  timeframes: ["5m", "15m", "1h"]

multi_timeframe:
  enabled: true
  available: ['15m', '1h', '4h']
```

### What Happens

#### 1. Window Creation
The timerange (20250620-20260218 = 244 days) is split into:
- **12 rolling windows** (confirmed in log: "Window 0/11" through "Window 11/11")
- Each window has:
  - **Training period:** 60 days
  - **Validation period:** 15 days
  - **Step forward:** 15 days

#### 2. Backtesting Per Window
For **each of the 12 windows:**
- ✅ **1 training backtest** on training data (60 days)
- ✅ **1 validation backtest** on validation data (15 days)
- **Total:** 2 backtests per window

#### 3. Total Backtests
```
12 windows × 2 backtests per window = 24 total backtests
```

---

## What Gets Tested in Each Backtest

### Pairs (2 currencies)
Each backtest runs on **BOTH pairs simultaneously**:
- ✅ ETH/BTC
- ✅ LTC/BTC

**NOT 2 separate backtests per pair** - they run together in one backtest!

### Timeframes
The 3 timeframes `["5m", "15m", "1h"]` are **allowed options** for strategy generation:
- The strategy **picks ONE** as its base timeframe (this run: `1h`)
- **NOT 3 separate backtests per timeframe**
- The strategy uses indicators on that timeframe

### Multi-Timeframe
With `multi_timeframe: enabled: true`:
- The strategy **CAN use indicators from higher timeframes** (e.g., 4h indicators on a 1h strategy)
- This provides trend confirmation but **doesn't create extra backtests**
- Data for all timeframes is loaded once, indicators calculated, then one backtest runs

---

## Result Aggregation Analysis

### Per-Window Results (from log)

| Window | Train Fitness | Train Trades | Val Fitness | Val Trades |
|--------|---------------|--------------|-------------|------------|
| 0      | 0.0821        | 1332         | 0.0842      | 352        |
| 1      | 0.0827        | 1340         | 0.0811      | 349        |
| 2      | 0.0806        | 1324         | 0.0825      | 302        |
| 3      | 0.0800        | 1295         | 0.0887      | 309        |
| 4      | 0.0791        | 1325         | 0.0813      | 352        |
| 5      | 0.0785        | 1321         | 0.0905      | 287        |
| 6      | 0.0811        | 1261         | 0.0797      | 361        |
| 7      | 0.0799        | 1323         | 0.0806      | 394        |
| 8      | 0.0773        | 1412         | 0.0813      | 297        |
| 9      | 0.0777        | 1360         | 0.0802      | 320        |
| 10     | 0.0750        | 1390         | 0.0818      | 319        |
| 11     | 0.0759        | 1342         | 0.0820      | 362        |

### Aggregation Calculation (Mean Method)

✅ **Validation Fitness (what GA uses):**
```
Mean = (0.0842 + 0.0811 + 0.0825 + 0.0887 + 0.0813 + 0.0905 + 
        0.0797 + 0.0806 + 0.0813 + 0.0802 + 0.0818 + 0.0820) / 12
     = 0.0828 ✓ CORRECT
```

✅ **Training Fitness (for comparison):**
```
Mean = (0.0821 + 0.0827 + 0.0806 + 0.0800 + 0.0791 + 0.0785 + 
        0.0811 + 0.0799 + 0.0773 + 0.0777 + 0.0750 + 0.0759) / 12
     = 0.0792 ✓ CORRECT
```

✅ **Train-Val Gap:**
```
Gap = 0.0792 - 0.0828 = -0.0037 ✓ CORRECT
```

**From log:**
```
Walk-forward complete for GAStrategy_Gen0_Ind0: 
Final fitness=0.0828 (train avg=0.0792, val avg=0.0828, gap=-0.0037)
```

**✅ All calculations match perfectly!**

---

## Why This Is Correct

### 1. Walk-Forward Prevents Overfitting
- **Without walk-forward:** 1 backtest on all data → Strategy sees all data → Overfits
- **With walk-forward:** 12 train/validate cycles → Strategy only sees training data → Better generalization

### 2. Fitness Uses Validation Data
The **final fitness (0.0828)** comes from **validation windows**, not training windows!
- Training avg: 0.0792 (what strategy saw during evolution)
- Validation avg: 0.0828 (performance on unseen data)
- Gap: -0.0037 (NEGATIVE = validation better than training! Rare and good)

### 3. Pairs Are Tested Together
Both ETH/BTC and LTC/BTC are tested in the same backtest:
- FreqTrade's backtest engine handles multiple pairs natively
- Simulates real trading with multiple pairs open simultaneously
- More realistic than separate per-pair backtests

### 4. Timeframes Work Correctly
- Strategy base timeframe: `1h` (one of the 3 options)
- Multi-timeframe indicators: Can use 4h data for trend confirmation
- One backtest runs with all necessary data loaded

---

## Performance Notes

### Cache Efficiency
From log: `Walk-forward cache stats: 0 hits, 12 misses`
- **0 cache hits:** First generation, no cached training results yet
- In generation 2+, elite individuals will have ~50-70% cache hits
- Cache stores training backtests only (validation always fresh)

### Execution Time
- 24 backtests completed in ~50 seconds
- ~2 seconds per backtest
- Reasonable for this data size

---

## Common Misconceptions

### ❌ "Each pair is backtested separately"
**Incorrect.** Both pairs run together in one backtest (per window).

### ❌ "Each timeframe is backtested separately"  
**Incorrect.** Strategy picks one base timeframe. Multi-timeframe uses data from other TFs but doesn't run separate backtests.

### ❌ "24 backtests means something is wrong"
**Incorrect.** 12 windows × 2 (train + validate) = 24 is expected and correct.

### ❌ "Results are just averaged naively"
**Incorrect.** Fitness uses **only validation scores** (out-of-sample data), not training scores. Training scores are logged for comparison but don't affect fitness.

---

## What Would Change With Different Settings

### More Pairs
```yaml
pairs: ["ETH/BTC", "LTC/BTC", "XRP/BTC"]  # 3 pairs
```
**Effect:** Still 24 backtests (3 pairs run together in each backtest)

### More Population
```yaml
population_size: 10
```
**Effect:** 24 backtests × 10 strategies = 240 backtests per generation

### Different Aggregation
```yaml
aggregation: 'min'  # Use worst-case performance
```
**Effect:** Final fitness = min(all validation scores) = 0.0797 (most conservative)

### Disable Walk-Forward
```yaml
walk_forward:
  enabled: false
```
**Effect:** Only 1 backtest per strategy (entire timerange)

---

## Conclusion

✅ **Walk-forward is working perfectly**  
✅ **Aggregation is mathematically correct**  
✅ **Pairs and timeframes are handled properly**  
✅ **24 backtests for 12 windows is expected behavior**  

**No bugs found. Everything is functioning as designed!**

---

## Technical Details

### Code Verification
- Window creation: `genetic_algorithm/utils/timerange.py`
- Fitness evaluation: `genetic_algorithm/evaluation/fitness.py`
- Aggregation: `aggregate_validation_scores()` using mean method
- Backtesting: `genetic_algorithm/evaluation/direct_backtester.py`

### Verification Method
1. ✅ Counted windows in log: 12 (Window 0-11)
2. ✅ Counted backtests per window: 2 (train + validate)
3. ✅ Manually calculated mean: 0.0828 (matches final fitness)
4. ✅ Verified train-val gap: -0.0037 (matches log)
5. ✅ Checked pair handling: Both pairs in one backtest

**All checks passed!**
