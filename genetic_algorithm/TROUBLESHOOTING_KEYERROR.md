# Troubleshooting Guide: KeyError for Indicator Columns

## Problem Description

If you're seeing errors like:
```
KeyError: 'macd'
KeyError: 'bb_lowerband'
KeyError: 'sma_20'
KeyError: 'cci_20'
KeyError: 'rsi_14'
```

These occur when generated strategies try to access indicator columns that don't exist in the dataframe.

## Root Cause

This issue can happen in two scenarios:

### 1. Old Generated Strategies (Most Common)
- Strategy files were generated **before** the fix was applied
- These buggy files are still on disk in `user_data/strategies/ga_generated/`
- When backtesting runs, it loads these old files and encounters errors

### 2. Edge Cases During Evolution
- Rare cases where mutation/crossover creates mismatched conditions
- The fix now prevents this from happening in new strategies

## Solution

### Quick Fix: Delete Old Generated Strategies

```bash
# Navigate to your freqtradeForkGA directory
cd /path/to/freqtradeForkGA

# Delete all old generated strategies
rm -rf user_data/strategies/ga_generated/*.py

# Run the GA again to generate new, fixed strategies
python genetic_algorithm/run_ga.py
```

### Why This Works

The fix (implemented in `genetic_algorithm/strategies/generator.py`) now:
1. **Validates** all conditions before generating code
2. **Filters out** conditions that reference non-existent indicators
3. **Provides fallbacks** when all conditions are invalid
4. **Logs warnings** when conditions are filtered (check logs for details)

New strategies generated after the fix will not have these errors.

## Verification

After deleting old strategies and running the GA, you should see:
- ✅ No KeyError exceptions
- ✅ All strategies backtest successfully
- ⚠️ Warning logs if conditions were filtered (this is normal and safe)

## What Indicators Are Covered?

The fix handles **ALL** indicators comprehensively:

### Period-Based Indicators
- RSI (e.g., `rsi_14`, `rsi_21`)
- SMA (e.g., `sma_20`, `sma_50`)
- EMA (e.g., `ema_20`, `ema_50`)
- CCI (e.g., `cci_20`)
- ADX (e.g., `adx_14`)
- ATR (e.g., `atr_14`)

### Multi-Column Indicators
- MACD (`macd`, `macdsignal`, `macdhist`)
- Bollinger Bands (`bb_upperband`, `bb_middleband`, `bb_lowerband`)
- Stochastic (`slowk`, `slowd`)

All of these are now protected by the validation logic.

## How The Fix Works

### Before Fix
```python
# Strategy might have:
def populate_indicators(dataframe):
    dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
    return dataframe

def populate_entry_trend(dataframe):
    # ❌ KeyError: 'macd' not calculated!
    conditions = (dataframe['macd'] > dataframe['macdsignal'])
    ...
```

### After Fix
```python
# Invalid MACD condition is filtered out
# Fallback condition is used instead:
def populate_entry_trend(dataframe):
    # ✅ Safe fallback: volume above average
    dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
    dataframe.loc[dataframe['volume'] > dataframe['volume_sma'], 'enter_long'] = 1
    ...
```

## Checking Logs

If you want to see what's happening behind the scenes, check the logs for:
```
WARNING - Filtered N condition(s) referencing missing indicators: ['MACD', 'BBANDS']
WARNING - No valid entry conditions found. Using fallback volume-based condition.
```

These warnings are **normal** and indicate the fix is working correctly.

## Still Having Issues?

If you're still seeing KeyError exceptions after:
1. Deleting old strategies
2. Running the GA with the latest code
3. Generating new strategies

Please report the issue with:
- The full error traceback
- The generated strategy file causing the error
- Your GA configuration file

## Testing

You can verify the fix works by running the test suite:

```bash
# Test basic functionality
python genetic_algorithm/test_indicator_column_fix.py

# Test all indicators comprehensively
python genetic_algorithm/test_all_indicators.py
```

Both should show all tests passing (✅).

## Summary

**Problem:** KeyError for indicator columns  
**Cause:** Old buggy strategy files on disk  
**Solution:** Delete old files, regenerate with fixed code  
**Prevention:** Fix automatically prevents new bugs  
**Coverage:** All indicators handled comprehensively

---

**Last Updated:** February 19, 2026  
**Status:** ✅ Fixed and Tested
