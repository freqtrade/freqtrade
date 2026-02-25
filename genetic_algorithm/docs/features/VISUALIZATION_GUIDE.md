# Live Visualization Guide

## Overview

The Genetic Algorithm now supports **live plotting of generation scores** with real-time visualization of the evolution process. This guide explains how to set up and use the visualization feature.

## Quick Start

### 1. Install Dependencies

First, make sure you have all required dependencies installed:

**Linux/macOS:**
```bash
./genetic_algorithm/setup_ga.sh
```

**Windows (PowerShell):**
```powershell
.\genetic_algorithm\setup_ga.ps1
```

**Or manually:**
```bash
pip install -r genetic_algorithm/requirements.txt
```

### 2. Run with Visualization

**Simple way (Linux/macOS):**
```bash
./genetic_algorithm/run_with_visualization.sh
```

**Windows:**
```powershell
.\genetic_algorithm\run_with_visualization.ps1
```

**Or directly:**
```bash
python genetic_algorithm/run_ga.py --visualize
```

## What You'll See

The live visualization displays **4 real-time plots** that update after each generation:

### 1. **Fitness Evolution** (Top-Left)
- 📈 **Best Fitness**: Green line showing the best fitness score in each generation
- 📊 **Average Fitness**: Blue line showing the average fitness across the population
- 📉 **Worst Fitness**: Red line showing the worst fitness in each generation
- **Shaded Area**: Visual representation of fitness range
- **Info Box**: Shows current generation, best fitness, and average fitness

**What to look for:**
- Upward trend indicates strategies are improving
- Convergence (flattening) suggests optimal strategies found
- Wide spread indicates diverse population

### 2. **Population Diversity** (Top-Right)
- 🎨 **Diversity Score**: Purple line showing standard deviation of fitness values
- **Filled Area**: Visual emphasis of diversity changes

**What to look for:**
- High diversity: Population has variety (good early on)
- Low diversity: Population converging (expected later)
- Sudden drops: May indicate premature convergence

### 3. **Performance Metrics** (Bottom-Left)
Shows the **best strategy's** key trading metrics:
- 💰 **Profit %**: Green line (left Y-axis)
- 📊 **Sharpe Ratio**: Blue line (left Y-axis)
- ✅ **Win Rate %**: Orange dashed line (right Y-axis)
- ⚠️ **Max Drawdown %**: Red dotted line (right Y-axis)
- **Info Box**: Current values for all metrics

**What to look for:**
- Rising profit: Strategies becoming more profitable
- Increasing Sharpe: Better risk-adjusted returns
- Stable win rate: Consistent performance
- Decreasing drawdown: Lower risk

### 4. **Fitness Distribution** (Bottom-Right)
- 📊 **Histogram**: Distribution of current population's fitness scores
- 🌈 **Color Gradient**: Red (low fitness) → Green (high fitness)
- **Statistics Box**: Mean, median, standard deviation, and count

**What to look for:**
- Right shift over time: Overall improvement
- Narrow distribution: Population converging
- Multiple peaks: Different strategy clusters

## Visualization Modes

### Interactive Mode (Default)
```bash
python genetic_algorithm/run_ga.py --visualize
```

**Features:**
- ✅ Live updating window showing evolution in real-time
- ✅ Can interact with plot (zoom, pan, save)
- ✅ Final plot saved to `genetic_algorithm/output/plots/`
- ✅ Window stays open at end for inspection

**Best for:** Desktop environments, local development

### Non-Interactive Mode
```bash
python genetic_algorithm/run_ga.py --visualize --no-interactive
```

**Features:**
- ✅ Plot saved after **every generation** (intermediate snapshots)
- ✅ Final plot saved at end
- ✅ No live window (works in headless environments)
- ✅ All plots saved to `genetic_algorithm/output/plots/`

**Best for:** Servers, remote execution, automated runs

## Testing the Visualization

Before running a full GA evolution, test the visualization works:

### Quick Test (Mock Data)
```bash
# Interactive test
python genetic_algorithm/test_visualization.py

# Non-interactive test
python genetic_algorithm/test_visualization.py --non-interactive
```

This simulates 10 generations with mock data and verifies:
- ✅ Matplotlib is installed correctly
- ✅ Plots render properly
- ✅ Updates work in real-time
- ✅ Files save correctly

## Troubleshooting

### "No module named 'matplotlib'"

**Problem:** Visualization dependencies not installed

**Solution:**
```bash
# Install GA requirements
pip install -r genetic_algorithm/requirements.txt

# Or just matplotlib
pip install matplotlib numpy
```

### "No display found" or TkAgg errors

**Problem:** Running in headless environment without display

**Solution:** Use non-interactive mode
```bash
python genetic_algorithm/run_ga.py --visualize --no-interactive
```

### Plots not updating in real-time

**Problem:** Wrong matplotlib backend or interactive mode disabled

**Solutions:**
1. Make sure you're using `--visualize` flag
2. Don't use `--no-interactive` if you want live updates
3. Try setting backend explicitly:
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```

### Plots saved but can't see them during run

**Problem:** Running in non-interactive mode

**Solution:** Remove `--no-interactive` flag for live viewing
```bash
python genetic_algorithm/run_ga.py --visualize
```

### Window appears but doesn't update

**Problem:** matplotlib event loop not processing

**Solution:** This is already handled in the code, but ensure:
- You have latest version of matplotlib: `pip install --upgrade matplotlib`
- Close other plotting windows before starting

## Output Files

### Saved Plots Location
```
genetic_algorithm/output/plots/
├── ga_evolution_20260216_143052.png          # Final plot
├── ga_evolution_gen0_20260216_143002.png     # Generation 0 (non-interactive)
├── ga_evolution_gen1_20260216_143015.png     # Generation 1 (non-interactive)
└── ...
```

### File Naming Convention
- **Final plot**: `ga_evolution_{timestamp}.png`
- **Intermediate**: `ga_evolution_gen{N}_{timestamp}.png`

## Configuration

### In Code
Visualization settings in `genetic_algorithm/core/evolution.py`:

```python
ga = GeneticAlgorithm(
    config_path="genetic_algorithm/config/ga_config.yaml",
    visualize=True,      # Enable visualization
    interactive=True     # Interactive mode (live window)
)
```

### Via YAML Config
Edit `genetic_algorithm/config/ga_config.yaml`:

```yaml
visualization:
  enabled: true
  update_interval: 1  # Update every N generations
  plot_types:
    - fitness
    - diversity
    - metrics
    - distribution
  output_dir: "genetic_algorithm/output/plots"
```

## Performance Considerations

### Memory Usage
- Each generation adds a few data points (~100 bytes)
- Plots are stored in memory until saved
- **Recommendation**: For very long runs (>500 generations), use non-interactive mode to save periodically

### Speed Impact
- Plotting adds ~0.1-0.5 seconds per generation
- Negligible compared to backtesting time (10-60 seconds per strategy)
- **Impact**: <1% slowdown on typical runs

## Advanced Usage

### Custom Plot Styling
Edit `genetic_algorithm/visualization/visualizer.py` to customize:
- Colors, line styles, markers
- Plot titles, labels, legends
- Figure size, DPI, layout
- Additional metrics or plots

### Saving Specific Generations
```python
# In your code
if generation % 10 == 0:  # Every 10 generations
    visualizer.save_final_plot(f"checkpoint_gen{generation}.png")
```

### Multiple Runs Comparison
Save plots with descriptive names:
```bash
python genetic_algorithm/run_ga.py --visualize
# Plots saved with timestamp automatically

# Later, compare plots manually or write script to overlay them
```

## Examples

### Example 1: Quick Test Run
```bash
# 1. Set up environment
./genetic_algorithm/setup_ga.sh

# 2. Test visualization
python genetic_algorithm/test_visualization.py

# 3. Run small GA with visualization
# Edit run_ga.py to set GENERATIONS=5, POPULATION_SIZE=10
python genetic_algorithm/run_ga.py --visualize
```

### Example 2: Long Run on Server
```bash
# 1. Install dependencies
pip install -r genetic_algorithm/requirements.txt

# 2. Run in non-interactive mode (saves plots, no window)
nohup python genetic_algorithm/run_ga.py --visualize --no-interactive > ga.log 2>&1 &

# 3. Monitor progress
tail -f ga.log

# 4. View intermediate plots
ls -lh genetic_algorithm/output/plots/
```

### Example 3: Custom Configuration
```python
from genetic_algorithm.core.evolution import GeneticAlgorithm

# Create GA with custom settings
ga = GeneticAlgorithm(
    "genetic_algorithm/config/ga_config.yaml",
    visualize=True,
    interactive=True
)

# Run evolution
ga.population_size = 100
ga.generations = 50
best_strategies = ga.evolve()

# Plots automatically update and save
```

## Summary

| Feature | Command | Output |
|---------|---------|--------|
| **Setup** | `./genetic_algorithm/setup_ga.sh` | Installs dependencies |
| **Live Interactive** | `python genetic_algorithm/run_ga.py --visualize` | Real-time window + final plot |
| **Non-Interactive** | `python genetic_algorithm/run_ga.py --visualize --no-interactive` | Plots saved per generation |
| **Test Visualization** | `python genetic_algorithm/test_visualization.py` | Mock data demo |
| **Quick Start** | `./genetic_algorithm/run_with_visualization.sh` | Auto-setup + run |

## Need Help?

1. **Check logs**: Look at console output for errors
2. **Test first**: Run `test_visualization.py` before full GA
3. **Dependencies**: Ensure matplotlib is installed
4. **Environment**: Use non-interactive mode on servers
5. **Documentation**: See `README.md` and `RUN_GA_GUIDE.md`

## What's Visualized

✅ **Generation Scores** (fitness values per generation)  
✅ **Best/Average/Worst fitness** (evolution progress)  
✅ **Population Diversity** (genetic variation)  
✅ **Performance Metrics** (profit, Sharpe, win rate, drawdown)  
✅ **Fitness Distribution** (population spread)

The visualization provides complete insight into how your trading strategies evolve!
