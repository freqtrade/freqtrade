# GA Configuration Guide

## Quick Start

### Option 1: Use Example Config (Recommended for Real Data)

1. **Download market data:**
   ```bash
   freqtrade download-data --exchange binance --pairs BTC/USDT --timeframes 1h --days 90
   ```

2. **Check your data range:**
   ```bash
   freqtrade list-data --show-timerange
   ```

3. **Copy example config:**
   ```bash
   cp genetic_algorithm/config/ga_config_example.yaml my_config.yaml
   ```

4. **Edit `my_config.yaml`:**
   - Update `timerange` to match your data
   - Update `pairs` to match downloaded pairs

5. **Run GA:**
   ```bash
   python genetic_algorithm/run_ga.py --config my_config.yaml
   ```

### Option 2: Quick Test (Uses 2018 Test Data)

```bash
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config_test.yaml
```

## Available Configs

| Config File | Purpose | Population | Generations | Data |
|-------------|---------|------------|-------------|------|
| `ga_config.yaml` | Default (test) | 100 | 50 | UNITTEST/BTC (2018) |
| `ga_config_example.yaml` | Template for real data | 20 | 10 | BTC/USDT (configurable) |
| `ga_config_test.yaml` | Quick test | 3 | 2 | UNITTEST/BTC (2018) |

## Configuration Sections

### Backtesting
```yaml
backtesting:
  timerange: "20250120-20250219"  # Date range YYYYMMDD-YYYYMMDD
  pairs: ["BTC/USDT"]              # Must have data for these
  stake_amount: 0.1                # Per-trade stake (in quote currency)
  max_open_trades: 1               # Concurrent trades
```

### Genetic Algorithm
```yaml
genetic_algorithm:
  population_size: 20   # Number of strategies per generation
  generations: 10       # Evolution iterations
  mutation_rate: 0.15   # Probability of mutation
  crossover_rate: 0.7   # Probability of crossover
```

## Troubleshooting

### "No data found" Error
```bash
# Check if you have data
freqtrade list-data

# If not, download it
freqtrade download-data --pairs BTC/USDT --timeframes 1h --days 90
```

### "Using 2018 test data" Warning
Your config has `UNITTEST/BTC` pairs. Update to real pairs:
```yaml
pairs:
  - "BTC/USDT"  # Instead of UNITTEST/BTC
```

### Empty Timerange Warning
Set a specific date range:
```yaml
timerange: "20250120-20250219"  # Instead of ""
```

## Command-Line Usage

```bash
# Use default config
python genetic_algorithm/run_ga.py

# Use custom config
python genetic_algorithm/run_ga.py --config my_config.yaml

# Validate config without running
python genetic_algorithm/run_ga.py --config my_config.yaml --validate-only
```
