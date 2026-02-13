# Quick Reference Guide - Genetic Algorithm for FreqTrade

## 📁 What You Have

### Documentation (Start Here!)
1. **README.md** - Architecture overview, what the system does
2. **GETTING_STARTED.md** - How to use the system (when complete)
3. **TODO.md** - Detailed checklist of all tasks (200+ items)
4. **DEVELOPMENT_PLAN.md** - Week-by-week implementation plan
5. **INTEGRATION_PLAN.md** - How to adapt GAFreqTrade components
6. **PROJECT_SUMMARY.md** - What's done, what's next
7. **QUICK_REFERENCE.md** - This file!

### Configuration
- **config/ga_config.yaml** - All settings (200+ parameters)
  - Population size, generations, mutation rates
  - Fitness weights
  - Indicator parameters
  - Backtesting settings

### Core Components (Skeleton)
All in `genetic_algorithm/core/`:

1. **strategy_gene.py** - How strategies are represented
   - `IndicatorGene` - Single indicator with parameters
   - `ConditionGene` - Entry/exit condition
   - `StrategyGene` - Complete strategy representation

2. **individual.py** - Strategy wrapper
   - Tracks fitness score
   - Stores performance metrics
   - Records parent/mutation history

3. **population.py** - Manages collection of strategies
   - Add/remove individuals
   - Sort by fitness
   - Calculate statistics

4. **selection.py** - Choose parents for breeding
   - Tournament selection
   - Roulette wheel
   - Rank-based

5. **crossover.py** - Combine two strategies
   - Single-point
   - Uniform
   - Component-based

6. **mutation.py** - Introduce variations
   - Parameter mutation
   - Indicator mutation
   - Condition mutation
   - Structural mutation

7. **evolution.py** - Main GA engine
   - Initialize population
   - Run evolution loop
   - Track best strategies

### Strategy Components (Ready!)
All in `genetic_algorithm/strategies/`:

1. **components.py** - Indicator library
   - 10 technical indicators (RSI, MACD, BB, etc.)
   - Parameter ranges for each
   - 30+ condition templates

2. **template.py** - Strategy code template
   - FreqTrade-compatible format
   - Proper structure and imports

3. **generator.py** - Creates strategies
   - Needs completion for full code generation

### Evaluation (Skeleton)
- **evaluation/fitness.py** - Calculates strategy fitness
  - Multi-objective function
  - Configurable weights
  - Penalty system

## 🎯 Current Status

### ✅ Complete
- Project structure
- Documentation (6 files, 50+ pages)
- Configuration system
- Indicator library (10 indicators)
- Condition templates (30+ patterns)
- Core framework (skeleton)

### 🚧 In Progress / Next Steps
- Strategy code generation
- Backtesting integration
- Full genetic operation logic
- Storage system

### ❌ Not Started
- Testing
- Visualization
- Advanced features (ML, LLM)

## 📚 Key Concepts

### What is a Strategy Gene?
A strategy represented as data that can be:
- Mutated (changed randomly)
- Crossed over (combined with another)
- Evaluated (tested for profitability)

Example:
```python
strategy_gene = {
    'indicators': [RSI(period=14), MACD(fast=12, slow=26)],
    'entry_conditions': ['RSI < 30', 'MACD crosses up'],
    'exit_conditions': ['RSI > 70', 'MACD crosses down'],
    'timeframe': '5m',
    'stoploss': -0.10
}
```

### How Evolution Works
1. **Generation 0**: Create 100 random strategies
2. **Evaluate**: Backtest each strategy
3. **Select**: Keep top 10 (highest profit, lowest risk)
4. **Breed**: Combine top strategies to create 90 new ones
5. **Mutate**: Randomly modify some parameters
6. **Repeat**: Go to step 2 with new generation

After many generations, strategies improve!

### Fitness Function
How strategies are scored:
```
Fitness = 
  30% * Profit +
  25% * Sharpe Ratio +
  20% * (1 - Drawdown) +
  15% * Win Rate +
  10% * Trade Frequency
```

All configurable in `ga_config.yaml`!

## 🔧 Configuration Quick Reference

### Essential Settings (ga_config.yaml)

```yaml
genetic_algorithm:
  population_size: 100     # Number of strategies per generation
  generations: 50          # How many generations to evolve
  mutation_rate: 0.15      # How often to mutate (15%)
  crossover_rate: 0.7      # How often to breed (70%)
  elite_size: 10           # Top N to keep unchanged

fitness_weights:
  profit: 0.30             # Most important
  sharpe_ratio: 0.25       # Risk-adjusted returns
  drawdown: 0.20           # Lower is better
  win_rate: 0.15           # Percent profitable
  trade_frequency: 0.10    # Not too many or too few

backtesting:
  timerange: "20230101-20231231"  # Test period
  pairs: ["BTC/USDT", "ETH/USDT"]  # Trading pairs
  stake_amount: 100        # How much per trade
```

### Tuning for Different Goals

**Conservative (Low Risk)**:
```yaml
fitness_weights:
  profit: 0.20
  sharpe_ratio: 0.30
  drawdown: 0.30
  win_rate: 0.15
  trade_frequency: 0.05
```

**Aggressive (High Profit)**:
```yaml
fitness_weights:
  profit: 0.50
  sharpe_ratio: 0.20
  drawdown: 0.10
  win_rate: 0.10
  trade_frequency: 0.10
```

## 📂 File Organization

```
freqtradeForkGA/                    # Main FreqTrade repo
├── freqtrade/                      # FreqTrade code
├── user_data/                      # FreqTrade data
│   └── strategies/
│       └── ga_generated/           # GA strategies go here!
└── genetic_algorithm/              # Our GA system
    ├── README.md                   # Start here
    ├── GETTING_STARTED.md          # Usage guide
    ├── TODO.md                     # Task list
    ├── PROJECT_SUMMARY.md          # Status report
    ├── config/
    │   └── ga_config.yaml          # All settings
    ├── core/                       # GA engine
    ├── strategies/                 # Strategy generation
    ├── evaluation/                 # Fitness calculation
    └── utils/                      # Helpers
```

## 🚀 Next Steps (For Implementation)

### 1. Complete Strategy Generator
File: `genetic_algorithm/strategies/generator.py`
- Take StrategyGene
- Convert to Python code
- Save as .py file in user_data/strategies/ga_generated/

### 2. Implement Backtester
File: `genetic_algorithm/evaluation/backtester.py`
- Run FreqTrade backtest command
- Parse JSON results
- Return metrics

### 3. Connect Everything
File: `genetic_algorithm/core/evolution.py`
- Generate strategies
- Backtest them
- Calculate fitness
- Evolve population

### 4. Test End-to-End
- Run evolution loop
- Verify strategies are valid
- Check fitness improves over generations

## 📖 Useful Commands (When Complete)

### Run Evolution
```bash
python -m genetic_algorithm.main
```

### View Top Strategies
```bash
python -m genetic_algorithm.show_leaderboard
```

### Test a Generated Strategy
```bash
freqtrade backtesting \
  --strategy GAStrategy_Gen50_Ind001 \
  --timerange 20230101-20231231
```

### Monitor Progress
```bash
python -m genetic_algorithm.monitor
```

## 🔗 Integration with GAFreqTrade

We found an existing implementation! Use it to:
1. Copy working backtester
2. Copy strategy generator
3. Copy genetic operations
4. Adapt to freqtradeForkGA structure

See **INTEGRATION_PLAN.md** for details.

## ⚠️ Important Notes

1. **This is experimental** - Don't risk real money without extensive testing
2. **Backtest != Live** - Strategies may perform differently in live trading
3. **Overfitting risk** - Strategies optimized for past data may fail on new data
4. **Start small** - Test with small amounts, short time periods
5. **Validate manually** - Review generated strategies before using

## 🎓 Learning Resources

- **FreqTrade Docs**: https://www.freqtrade.io/
- **Genetic Algorithms**: https://en.wikipedia.org/wiki/Genetic_algorithm
- **Technical Indicators**: TradingView, Investopedia
- **GAFreqTrade Repo**: https://github.com/Edogor/GAFreqTrade.git

## 📞 Help & Support

1. Read **GETTING_STARTED.md** for usage
2. Check **TODO.md** for what's implemented
3. See **DEVELOPMENT_PLAN.md** for architecture
4. Review **INTEGRATION_PLAN.md** for next steps
5. Consult **PROJECT_SUMMARY.md** for current status

## 🎉 Summary

You have a **complete framework** ready for implementation!

**What works**: Structure, docs, config, indicator library
**What's next**: Code generation, backtesting, testing
**Estimated time**: 2-3 weeks to MVP

Start with **README.md** to understand the system, then **TODO.md** to see what to build next!
