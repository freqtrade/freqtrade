# Deployment Runs - Quick Start Guide

## Overview

This directory contains everything you need to run 3 comprehensive evolution runs designed to generate **deployable trading strategies** with varying levels of anti-overfitting validation.

**Total Runtime:** 8-10 hours  
**Goal:** Generate robust strategies validated for real-world deployment

---

## Three Run Configurations

### Run 1: Basic Exploration (2-3 hours)
**Config:** [ga_config_deploy_run1_basic.yaml](config/ga_config_deploy_run1_basic.yaml)  
**Population:** 40 | **Generations:** 20

**Philosophy:** "Find what works first, validate later"

- ❌ No walk-forward validation
- ❌ No regime-aware evaluation
- ❌ No holdout monitoring
- ✅ Fast fitness evaluation
- ✅ High mutation for exploration
- ✅ Fitness sharing for diversity

**Purpose:** Rapid exploration to discover profitable patterns without computational overhead of validation. Use this to identify promising strategy archetypes.

---

### Run 2: Balanced Validation (2-3 hours)
**Config:** [ga_config_deploy_run2_balanced.yaml](config/ga_config_deploy_run2_balanced.yaml)  
**Population:** 40 | **Generations:** 20

**Philosophy:** "Find what works AND validates on unseen data"

- ✅ Walk-forward optimization (90/30 day windows)
- ✅ Holdout validation (15% out-of-sample)
- ✅ Holdout monitoring during evolution
- ✅ Parsimony pressure
- ❌ No regime-aware (saved for Run 3)

**Purpose:** Balance between exploration and validation. Strategies must perform well on both training and validation windows.

---

### Run 3: Full Robustness (3-4 hours)
**Config:** [ga_config_deploy_run3_full.yaml](config/ga_config_deploy_run3_full.yaml)  
**Population:** 50 | **Generations:** 25

**Philosophy:** "Survive and thrive in all market conditions"

- ✅ Walk-forward optimization (120/45 day windows)
- ✅ Holdout validation (20% out-of-sample)
- ✅ Holdout monitoring with early stop
- ✅ **Regime-aware evaluation** (bull/bear/sideways)
- ✅ Parsimony pressure
- ✅ Strict fitness penalties

**Purpose:** Production-grade strategies validated across different market regimes. These strategies should withstand bull markets, bear markets, and sideways consolidation.

---

## Common Configuration

All runs use:

- **Pairs:** BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT
- **Timeframes:** 15m, 1h
- **Timerange:** 20230101-20260228 (3+ years)
- **Fee:** 0.1% (realistic for Binance)
- **Slippage:** 0.05%
- **Stake:** 15% per trade
- **Max Open Trades:** 3

---

## Quick Start

### Step 1: Verify Data

Ensure you have all required data for the 3-year timerange:

```bash
bash genetic_algorithm/scripts/verify_data.sh
```

This will:
- Check for existing data files
- Show timerange coverage
- Offer to download/update missing data
- Run this first to avoid data issues during evolution

### Step 2: Run Evolution Runs

**Option A: Run all 3 sequentially (recommended)**

```bash
bash genetic_algorithm/scripts/run_deploy_sequential.sh
```

This executes all three runs one after another with comprehensive error handling and progress tracking.

**Option B: Run in background with nohup**

```bash
nohup bash genetic_algorithm/scripts/run_deploy_sequential.sh > deploy_runs.log 2>&1 &
echo $! > deploy_runs.pid
```

**Option C: Run individual configs manually**

```bash
# Run 1 only
python genetic_algorithm/run_ga.py \
  --config genetic_algorithm/config/ga_config_deploy_run1_basic.yaml \
  --visualize --yes

# Run 2 only
python genetic_algorithm/run_ga.py \
  --config genetic_algorithm/config/ga_config_deploy_run2_balanced.yaml \
  --visualize --yes

# Run 3 only
python genetic_algorithm/run_ga.py \
  --config genetic_algorithm/config/ga_config_deploy_run3_full.yaml \
  --visualize --yes
```

---

## Monitoring Progress

### Watch Summary File

```bash
tail -f genetic_algorithm/logs/deploy_runs_summary.txt
```

### Monitor Individual Runs

```bash
# Run 1
tail -f genetic_algorithm/logs/deploy_run1_basic.log

# Run 2
tail -f genetic_algorithm/logs/deploy_run2_balanced.log

# Run 3
tail -f genetic_algorithm/logs/deploy_run3_full.log
```

### Check System Resources

```bash
# CPU usage
top -p $(pgrep -f run_ga.py)

# Memory usage
ps aux | grep run_ga.py
```

---

## Resume Interrupted Runs

If a run is interrupted, you can resume from a specific run number:

```bash
# Resume from Run 2
bash genetic_algorithm/scripts/run_deploy_sequential.sh 2

# Resume from Run 3
bash genetic_algorithm/scripts/run_deploy_sequential.sh 3
```

Each run can also be resumed individually using the `--resume` flag if checkpoints exist.

---

## Results Location

After completion, results will be in:

```
genetic_algorithm/output/
├── deploy_run1_basic/
│   ├── hall_of_fame.json          # Top 10 strategies
│   ├── fitness_progression.png    # Fitness over generations
│   ├── diversity_metrics.png      # Population diversity
│   ├── checkpoints/               # Generation checkpoints
│   └── generation_*.json          # Per-generation data
│
├── deploy_run2_balanced/
│   ├── hall_of_fame.json
│   ├── fitness_progression.png
│   ├── holdout_tracking.png       # OOS performance
│   └── ...
│
└── deploy_run3_full/
    ├── hall_of_fame.json
    ├── fitness_progression.png
    ├── regime_performance.png     # Bull/bear/sideways breakdown
    └── ...
```

---

## Analyzing Results

### View Hall of Fame

```bash
# Pretty-print JSON
cat genetic_algorithm/output/deploy_run1_basic/hall_of_fame.json | jq

# Extract top strategy fitness
cat genetic_algorithm/output/deploy_run1_basic/hall_of_fame.json | jq '.[0].fitness'

# Compare Run 2 with Run 1
diff <(cat genetic_algorithm/output/deploy_run1_basic/hall_of_fame.json | jq) \
     <(cat genetic_algorithm/output/deploy_run2_balanced/hall_of_fame.json | jq)
```

### View Fitness Plots

```bash
# Open plots (Linux with GUI)
xdg-open genetic_algorithm/output/deploy_run1_basic/fitness_progression.png
xdg-open genetic_algorithm/output/deploy_run2_balanced/fitness_progression.png
xdg-open genetic_algorithm/output/deploy_run3_full/fitness_progression.png

# Or view remotely
scp user@host:path/to/fitness_progression.png .
```

### Success Criteria

**Good Run Indicators:**
- Best fitness > 0.50 (excellent if > 0.70)
- Fitness steadily increases over generations
- Diversity remains > 0.10 throughout
- Hall of fame has multiple distinct strategies
- Sharpe ratio > 2.0 for top strategies
- Win rate 45-65% (sweet spot)
- 10-50 trades per strategy (not too few, not excessive)

**Red Flags:**
- Best fitness < 0.30
- Fitness plateaus early (< generation 10)
- Diversity collapses to near 0
- All hall of fame strategies identical
- Negative Sharpe ratio
- < 5 trades or > 200 trades

---

## Comparing the Three Runs

### Expected Outcomes

| Metric | Run 1 (Basic) | Run 2 (Balanced) | Run 3 (Full) |
|--------|---------------|------------------|--------------|
| **Fitness** | Highest | Medium | Lower |
| **Robustness** | Lowest | Medium | Highest |
| **Overfitting Risk** | High | Medium | Low |
| **Deployment Ready** | ⚠️ Test first | ✓ With caution | ✓ Production-grade |
| **Runtime** | Fastest | Medium | Slowest |

### Key Comparisons

1. **Fitness Progression:**
   - Run 1 should show highest raw fitness (no penalties)
   - Run 2 should be lower (validation penalty)
   - Run 3 should be lowest (regime + validation penalties)

2. **Strategy Complexity:**
   - Run 1: May produce complex strategies
   - Run 2: Moderate complexity (parsimony pressure)
   - Run 3: Simpler strategies (stronger parsimony)

3. **Holdout Degradation:**
   - Run 1: Expect 30-50% degradation (not validated)
   - Run 2: Expect 10-25% degradation (validated)
   - Run 3: Expect < 15% degradation (comprehensive validation)

---

## Next Steps After Runs Complete

### 1. Review Hall of Fame from Each Run

```bash
cat genetic_algorithm/output/deploy_run1_basic/hall_of_fame.json | jq '.[0]'
cat genetic_algorithm/output/deploy_run2_balanced/hall_of_fame.json | jq '.[0]'
cat genetic_algorithm/output/deploy_run3_full/hall_of_fame.json | jq '.[0]'
```

### 2. Export Top Strategies

Top strategies are automatically exported as standalone strategy files in:
```
genetic_algorithm/output/deploy_run*/strategies/
```

### 3. Backtest on Fresh Data

```bash
# Backtest Run 3's best strategy on a holdout period
freqtrade backtesting \
  --strategy genetic_algorithm/output/deploy_run3_full/strategies/strategy_rank_1.py \
  --timerange 20260301-20260309 \
  --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT
```

### 4. Paper Trading

Before live deployment, test in paper trading mode:

```bash
freqtrade trade \
  --strategy genetic_algorithm/output/deploy_run3_full/strategies/strategy_rank_1.py \
  --config user_data/config_paper.json \
  --dry-run
```

### 5. Compare Performance Across Runs

Create a comparison table:

| Strategy | Fitness | Sharpe | Sortino | Drawdown | Trades | Holdout % |
|----------|---------|--------|---------|----------|--------|-----------|
| Run 1 #1 | 0.72 | 2.3 | 2.8 | -18% | 35 | 68% |
| Run 2 #1 | 0.68 | 2.5 | 3.1 | -15% | 28 | 89% |
| Run 3 #1 | 0.61 | 2.8 | 3.4 | -12% | 24 | 94% |

---

## Troubleshooting

### "Data not found" Error

```bash
# Re-run data verification
bash genetic_algorithm/scripts/verify_data.sh

# Manually download for specific pair/timeframe
freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT \
  --timeframes 1h \
  --timerange 20230101-20260228
```

### Run Crashes or Hangs

```bash
# Check logs for errors
tail -n 100 genetic_algorithm/logs/deploy_run1_basic.log

# Check disk space
df -h

# Check memory
free -h

# Kill hung process
pkill -f run_ga.py
```

### Low Fitness Scores

If all runs produce fitness < 0.30:
- Check data quality: `freqtrade list-data --show-timerange`
- Reduce constraints (min_trades, min_win_rate) in configs
- Increase population size or generations
- Verify timerange covers diverse market conditions

### Premature Convergence

If diversity drops to near-zero early:
- Increase mutation_rate (0.25 → 0.30)
- Increase random_immigrants (10 → 20)
- Enable adaptive_mutation
- Reduce sharing_radius (0.25 → 0.15)

---

## Configuration Customization

To modify runs for your needs:

### Adjust Runtime

**Faster runs (1-2h each):**
```yaml
population_size: 30
generations: 15
```

**Longer runs (4-6h each):**
```yaml
population_size: 60
generations: 30
```

### Change Pairs

Edit the `pairs` section in each config:
```yaml
pairs:
  - "BTC/USDT"
  - "ETH/USDT"
  - "ADA/USDT"  # Add more pairs
```

### Adjust Timerange

```yaml
timerange: "20220101-20260228"  # Extend to 4 years
```

### Modify Fitness Weights

Emphasize different metrics:
```yaml
fitness_weights:
  profit: 0.15          # Reduce profit weight
  sharpe_ratio: 0.30    # Increase risk-adjusted returns
  drawdown: 0.25        # Increase drawdown control
```

---

## Support & Documentation

- **Main GA README:** [genetic_algorithm/README.md](README.md)
- **Config Documentation:** [genetic_algorithm/config/README.md](config/README.md)
- **FreqTrade Docs:** https://www.freqtrade.io/
- **Issue Tracker:** Report issues on GitHub

---

## Summary Commands

```bash
# Full workflow
bash genetic_algorithm/scripts/verify_data.sh
bash genetic_algorithm/scripts/run_deploy_sequential.sh

# Monitor progress
tail -f genetic_algorithm/logs/deploy_runs_summary.txt

# View results
cat genetic_algorithm/output/deploy_run3_full/hall_of_fame.json | jq '.[0]'
```

---

**Good luck with your evolution runs! May your strategies be profitable and robust! 🚀📈**
