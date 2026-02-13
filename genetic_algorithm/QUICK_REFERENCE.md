# Quick Reference - Genetic Algorithm for FreqTrade

**Last Updated**: February 13, 2026  
**Status**: ✅ System verified working  
**Quick Test**: `python genetic_algorithm/example_usage.py` ✅

---

## 🚀 5-Minute Quick Start

```bash
# 1. Run the example
python genetic_algorithm/example_usage.py

# 2. Check the output
cat genetic_algorithm/examples/example_strategy.py

# 3. Backtest it (if you have data)
freqtrade backtesting --strategy GAStrategy_Gen0_Ind0
```

**Result**: ✅ Valid FreqTrade strategy ready to use!

---

## 📋 Common Commands

### Generate a Single Strategy
```python
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml

with open('genetic_algorithm/config/ga_config.yaml') as f:
    config = yaml.safe_load(f)

generator = StrategyGenerator(config)
strategy = generator.generate_random_strategy(0, 1)
code = generator.generate_strategy_code(strategy)

with open('user_data/strategies/MyStrategy.py', 'w') as f:
    f.write(code)
```

### Generate a Population
```python
from genetic_algorithm.core.evolution import GeneticAlgorithm

ga = GeneticAlgorithm('genetic_algorithm/config/ga_config.yaml')
population = ga.initialize_population()  # 100 strategies
```

### Run Evolution
```python
ga = GeneticAlgorithm('genetic_algorithm/config/ga_config.yaml')
best = ga.evolve()  # Evolve over generations
```

---

## 🔧 Quick Config Changes

Edit `genetic_algorithm/config/ga_config.yaml`:

### Conservative Strategy
```yaml
fitness_weights:
  profit: 0.20
  sharpe_ratio: 0.35
  drawdown: 0.30
  win_rate: 0.15
```

### Aggressive Strategy
```yaml
fitness_weights:
  profit: 0.50
  sharpe_ratio: 0.20
  drawdown: 0.10
  win_rate: 0.20
```

---

## 📖 Essential Documentation

1. **STATUS_REPORT.md** - ⭐ **START HERE** - Full system status
2. **TUTORIAL.md** - Complete usage guide with examples
3. **README.md** - Project overview and architecture
4. **ACCOMPLISHMENTS.md** - What's been implemented
5. **TODO.md** - Task list with completion status
6. **NEXT_STEPS.md** - Future features and roadmap

---

## 🎯 What's Working (Verified Feb 13, 2026)

✅ **Strategy Generation** - Creates valid FreqTrade IStrategy classes  
✅ **Real Backtesting** - Uses FreqTrade's actual backtesting engine  
✅ **Multi-Generation Evolution** - Full GA loop implemented  
✅ **Genetic Operators** - Selection, crossover, mutation working  
✅ **Configuration System** - YAML-based with all parameters  
✅ **Example Script** - `example_usage.py` verified working  

---

## 📁 Key Files & Directories

```
genetic_algorithm/
├── STATUS_REPORT.md           # ⭐ Current status summary
├── README.md                  # Project overview
├── TUTORIAL.md                # Usage guide
├── example_usage.py           # ✅ Working example
├── config/
│   └── ga_config.yaml         # All configuration
├── examples/
│   └── example_strategy.py    # Generated example
└── user_data/strategies/      # Save your strategies here
```

---

## 🧬 Quick Concepts

### What is a Strategy?
A trading strategy defines:
- **Indicators** (RSI, MACD, etc.)
- **Entry conditions** (when to buy)
- **Exit conditions** (when to sell)
- **Risk management** (stop-loss, take-profit)

### How Evolution Works
1. Create 100 random strategies
2. Backtest each → get fitness score
3. Select best performers
4. Combine & mutate → new generation
5. Repeat → strategies improve!

### Fitness Score
```
Fitness = 30% Profit + 25% Sharpe + 20% (1-Drawdown) + 15% WinRate + 10% Trades
```

---

## 💡 Quick Tips

### For Best Results
1. Use 6+ months of historical data
2. Test on multiple market conditions
3. Start with dry-run before live trading
4. Monitor diversity to avoid convergence
5. Save checkpoints during long runs

### Common Mistakes to Avoid
1. ⚠️ Don't overtrust single backtest
2. ⚠️ Don't skip out-of-sample validation
3. ⚠️ Don't use strategies without understanding them
4. ⚠️ Don't risk more than you can afford to lose

---

## 🔍 Troubleshooting

### "Config file not found"
```bash
# Check you're in the right directory
cd /path/to/freqtradeForkGA
ls genetic_algorithm/config/ga_config.yaml
```

### "Strategy generates no trades"
- Check indicator parameters (might be too restrictive)
- Review entry/exit conditions
- Try different timeframe
- Verify data quality

### "Evolution is slow"
- Reduce population size (100 → 20)
- Reduce generations (50 → 10)
- Use shorter timerange for testing

---

## 📞 Getting Help

1. Check **STATUS_REPORT.md** for capabilities
2. Read **TUTORIAL.md** for detailed guide
3. Review code comments (extensive documentation)
4. Look at test files for usage examples
5. See FreqTrade docs: https://www.freqtrade.io/

---

## 🎉 Ready to Start!

```bash
# Verify system works
python genetic_algorithm/example_usage.py

# Success? You're ready to:
# - Generate strategies
# - Run backtests
# - Evolve populations
# - Deploy to trading!
```

**Status**: 🚀 **SYSTEM READY FOR USE**

For full details, see **STATUS_REPORT.md**

Happy trading! 📈
