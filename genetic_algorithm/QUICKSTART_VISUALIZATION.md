# Quick Start: Live Plotting of Generation Scores

## Problem Solved ✅

This branch now has **full support for live plotting of generation scores**. You can see real-time visualizations of:
- 📈 Fitness evolution (best/average/worst per generation)
- 🎨 Population diversity over time
- 💰 Performance metrics (profit, Sharpe ratio, win rate, drawdown)
- 📊 Fitness distribution histogram

## 3-Step Setup

### Step 1: Install Dependencies

Choose your operating system:

**Linux/macOS:**
```bash
cd /path/to/freqtradeForkGA
./genetic_algorithm/setup_ga.sh
```

**Windows (PowerShell):**
```powershell
cd C:\path\to\freqtradeForkGA
.\genetic_algorithm\setup_ga.ps1
```

**Or manually:**
```bash
pip install -r genetic_algorithm/requirements.txt
```

This installs matplotlib and other required packages.

### Step 2: Test the Visualization

Before running a full GA, verify visualization works:

```bash
python genetic_algorithm/test_visualization.py
```

This runs a quick test with mock data (takes ~5 seconds). You should see:
- A window with 4 plots updating in real-time
- Final plot saved to `genetic_algorithm/output/plots/`

### Step 3: Run GA with Live Visualization

Now run the actual genetic algorithm with live plotting:

**Option A: Simple script (checks dependencies automatically)**
```bash
# Linux/macOS
./genetic_algorithm/run_with_visualization.sh

# Windows
.\genetic_algorithm\run_with_visualization.ps1
```

**Option B: Direct command**
```bash
python genetic_algorithm/run_ga.py --visualize
```

## What You'll See

As the GA runs, you'll see a live window with 4 plots that update after each generation:

```
┌─────────────────────────────────────────────────────────────┐
│     Genetic Algorithm Evolution Progress                   │
├──────────────────────────┬──────────────────────────────────┤
│ Fitness Evolution        │ Population Diversity             │
│ ▲                       │ ▲                                │
│ │  ╱───────             │ │  ╲╱╲                          │
│ │ ╱                     │ │      ╲                         │
│ └─────────────>         │ └─────────────>                  │
├──────────────────────────┼──────────────────────────────────┤
│ Performance Metrics      │ Fitness Distribution             │
│ ▲  Profit ━━━━          │ ▲                                │
│ │  Sharpe ━━━━          │ │  ██                            │
│ │  Win Rate ━━━━        │ │  ████                          │
│ └─────────────>         │ └─────────────>                  │
└──────────────────────────┴──────────────────────────────────┘
```

## Examples of Different Modes

### 1. Interactive Mode (Recommended)
```bash
python genetic_algorithm/run_ga.py --visualize
```
- ✅ Live window updates in real-time
- ✅ Can zoom, pan, and interact with plots
- ✅ Window stays open at end for review
- ✅ Final plot saved automatically

**Best for:** Running on your local machine

### 2. Non-Interactive Mode (for Servers)
```bash
python genetic_algorithm/run_ga.py --visualize --no-interactive
```
- ✅ Plots saved after **each generation**
- ✅ No window displayed (works on headless servers)
- ✅ Can review progress remotely
- ✅ All plots saved to `genetic_algorithm/output/plots/`

**Best for:** Remote servers, cloud instances, automated runs

### 3. No Visualization (Fastest)
```bash
python genetic_algorithm/run_ga.py
```
- ✅ No plotting overhead
- ✅ Fastest execution
- ✅ Still shows text progress

**Best for:** Production runs where you only need final results

## Troubleshooting

### Issue: "No module named 'matplotlib'"

**Solution:** Run the setup script
```bash
./genetic_algorithm/setup_ga.sh  # Linux/macOS
```

### Issue: "No display found" or "TkAgg backend error"

**Solution:** Use non-interactive mode
```bash
python genetic_algorithm/run_ga.py --visualize --no-interactive
```

### Issue: Window appears but plots don't update

**Solution:** Make sure you're using `--visualize` flag and NOT `--no-interactive`
```bash
python genetic_algorithm/run_ga.py --visualize
```

### Issue: Want to see plots from non-interactive run

**Solution:** Check the output directory
```bash
ls -lh genetic_algorithm/output/plots/
# Copy plots to your local machine to view
scp user@server:freqtradeForkGA/genetic_algorithm/output/plots/*.png .
```

## Output Files

All plots are automatically saved to:
```
genetic_algorithm/output/plots/
├── ga_evolution_20260216_143052.png          # Final plot
├── ga_evolution_gen0_20260216_143002.png     # Generation 0 (non-interactive only)
├── ga_evolution_gen1_20260216_143015.png     # Generation 1 (non-interactive only)
├── ga_evolution_gen2_20260216_143025.png     # Generation 2 (non-interactive only)
└── ...
```

## Full Documentation

For complete details, see:
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - Complete visualization documentation
- **[README.md](README.md)** - Main GA documentation
- **[RUN_GA_GUIDE.md](RUN_GA_GUIDE.md)** - Configuration options

## Summary

✅ **Live plotting works** - Real-time visualization of generation scores  
✅ **Easy setup** - One command to install dependencies  
✅ **Multiple modes** - Interactive, non-interactive, or no visualization  
✅ **Comprehensive docs** - Complete guides and troubleshooting  
✅ **Ready to use** - Just run the scripts!

---

**Need help?** Check [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for complete troubleshooting.
