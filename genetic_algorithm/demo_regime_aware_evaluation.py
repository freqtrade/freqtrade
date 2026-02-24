#!/usr/bin/env python3
"""
Regime-Aware Evaluation Visualization

Generates visual reports showing:
1. Price chart with detected regime overlays
2. Segment breakdown by regime type
3. Fitness aggregation demonstration

Run: python genetic_algorithm/demo_regime_aware_evaluation.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType
from genetic_algorithm.evaluation.regime_aware import RegimeAwareEvaluator


def create_synthetic_market_data(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic market data with clear regime patterns.
    
    Returns DataFrame with OHLCV data showing:
    - Days 0-100: Bullish trend
    - Days 100-180: Bearish trend
    - Days 180-280: Sideways consolidation
    - Days 280-365: Bullish recovery
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    prices = []
    price = 100.0
    
    for i in range(days):
        if i < 100:  # Bullish
            drift, volatility = 0.003, 0.01
        elif i < 180:  # Bearish
            drift, volatility = -0.002, 0.012
        elif i < 280:  # Sideways
            drift, volatility = 0.0001, 0.006
        else:  # Bullish again
            drift, volatility = 0.0025, 0.011
        
        change = np.random.normal(drift, volatility)
        price *= (1 + change)
        prices.append(price)
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.003, 0.003, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.008, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.008, days))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, days),
    }, index=dates)
    
    return df


def print_ascii_price_chart(df: pd.DataFrame, regime_series: pd.Series, width: int = 80, height: int = 20):
    """Print an ASCII price chart with regime coloring."""
    
    # Resample to fit width
    step = max(1, len(df) // width)
    prices = df['close'].values[::step]
    regimes = regime_series.values[::step]
    
    # Normalize prices to chart height
    price_min, price_max = prices.min(), prices.max()
    price_range = price_max - price_min
    if price_range == 0:
        price_range = 1
    
    # Create chart grid
    chart = [[' ' for _ in range(len(prices))] for _ in range(height)]
    
    # Map regime types to symbols
    regime_symbols = {
        RegimeType.BULLISH: '▲',
        RegimeType.BEARISH: '▼',
        RegimeType.SIDEWAYS: '─',
        RegimeType.VOLATILE: '~',
        RegimeType.UNCERTAIN: '?',
    }
    
    # Plot prices
    for x, (price, regime) in enumerate(zip(prices, regimes)):
        y = int((price - price_min) / price_range * (height - 1))
        y = height - 1 - y  # Invert for display
        y = max(0, min(height - 1, y))
        symbol = regime_symbols.get(regime, '*')
        chart[y][x] = symbol
    
    # Print chart with borders
    print()
    print(f"Price Chart (365 days) - ▲=Bullish, ▼=Bearish, ─=Sideways")
    print("─" * (len(prices) + 4))
    print(f"${price_max:6.1f} │", end="")
    for row in chart[0]:
        print(row, end="")
    print("│")
    
    for i, row in enumerate(chart[1:-1], 1):
        if i == height // 2:
            mid_price = (price_max + price_min) / 2
            print(f"${mid_price:6.1f} │", end="")
        else:
            print(f"       │", end="")
        for cell in row:
            print(cell, end="")
        print("│")
    
    print(f"${price_min:6.1f} │", end="")
    for row in chart[-1]:
        print(row, end="")
    print("│")
    print("─" * (len(prices) + 4))
    print("        Jan       Feb       Mar       Apr       May       Jun       Jul       Aug       Sep       Oct       Nov       Dec")
    print()


def print_regime_bar(regime_series: pd.Series, width: int = 80):
    """Print a horizontal bar showing regime distribution over time."""
    step = max(1, len(regime_series) // width)
    regimes = regime_series.values[::step]
    
    # Color codes (ANSI)
    colors = {
        RegimeType.BULLISH: '\033[92m█\033[0m',   # Green
        RegimeType.BEARISH: '\033[91m█\033[0m',   # Red
        RegimeType.SIDEWAYS: '\033[93m█\033[0m',  # Yellow
        RegimeType.VOLATILE: '\033[95m█\033[0m',  # Magenta
        RegimeType.UNCERTAIN: '\033[90m░\033[0m', # Gray
    }
    
    # Plain text version (for non-ANSI terminals)
    plain = {
        RegimeType.BULLISH: '▓',
        RegimeType.BEARISH: '░',
        RegimeType.SIDEWAYS: '▒',
        RegimeType.VOLATILE: '~',
        RegimeType.UNCERTAIN: '?',
    }
    
    print("Regime Timeline:")
    print("├" + "─" * width + "┤")
    print("│", end="")
    for regime in regimes:
        print(colors.get(regime, '?'), end="")
    print("│")
    print("├" + "─" * width + "┤")
    print("Legend: \033[92m█\033[0m Bullish  \033[91m█\033[0m Bearish  \033[93m█\033[0m Sideways")
    print()


def print_segment_table(segments):
    """Print a table of regime segments."""
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                         REGIME SEGMENTS BREAKDOWN                            │")
    print("├───────────────────────┬────────────┬────────────┬────────────┬───────────────┤")
    print("│ Segment ID            │ Regime     │ Confidence │ Days       │ Role          │")
    print("├───────────────────────┼────────────┼────────────┼────────────┼───────────────┤")
    
    for seg in segments:
        print(f"│ {seg.segment_id:21s} │ {seg.regime.value:10s} │ {seg.confidence:10.1%} │ {seg.duration_days:10d} │ {seg.role:13s} │")
    
    print("└───────────────────────┴────────────┴────────────┴────────────┴───────────────┘")


def print_aggregation_demo():
    """Demonstrate different fitness aggregation methods."""
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                    FITNESS AGGREGATION METHODS DEMO                          │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    
    # Simulated fitness scores from different regime segments
    demo_scores = {
        'Bullish Seg 1': 0.75,
        'Bullish Seg 2': 0.68,
        'Bearish Seg 1': 0.42,
        'Bearish Seg 2': 0.38,
        'Sideways Seg 1': 0.55,
        'Sideways Seg 2': 0.51,
    }
    
    print("│                                                                             │")
    print("│ Strategy fitness scores across different regime segments:                   │")
    print("│                                                                             │")
    
    for name, score in demo_scores.items():
        bar_len = int(score * 40)
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"│   {name:16s} │ {bar} │ {score:.2f}  │")
    
    print("│                                                                             │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    
    scores = list(demo_scores.values())
    
    # Calculate different aggregations
    mean_score = sum(scores) / len(scores)
    min_score = min(scores)
    
    from statistics import harmonic_mean
    hm_score = harmonic_mean(scores)
    
    # CVaR (bottom 20%)
    sorted_scores = sorted(scores)
    n_worst = max(1, int(len(sorted_scores) * 0.2))
    cvar_score = sum(sorted_scores[:n_worst]) / n_worst
    
    print("│ Aggregated Fitness Results:                                                │")
    print("│                                                                             │")
    print(f"│   Mean (average)         : {mean_score:.4f}  - Balanced across all regimes      │")
    print(f"│   Min (worst-case)       : {min_score:.4f}  - Conservative, risk-averse         │")
    print(f"│   Harmonic Mean          : {hm_score:.4f}  - Penalizes inconsistency (⭐)       │")
    print(f"│   CVaR (bottom 20%)      : {cvar_score:.4f}  - Focus on worst scenarios          │")
    print("│                                                                             │")
    print("│   ⭐ Harmonic mean is recommended: penalizes strategies that only perform   │")
    print("│      well in one regime while failing in others.                            │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")


def print_comparison_table():
    """Print comparison: single-period vs regime-aware evaluation."""
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│           SINGLE-PERIOD vs REGIME-AWARE EVALUATION COMPARISON               │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  Issue with Single-Period:                                                  │")
    print("│  ───────────────────────                                                    │")
    print("│  • Strategy trained on Bull market → fitness = 0.85 (great!)                │")
    print("│  • Same strategy in Bear market → actual fitness = 0.25 (terrible!)         │")
    print("│  • Strategy is OVERFIT to one market condition                              │")
    print("│                                                                             │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│                                                                             │")
    print("│  Regime-Aware Evaluation:                                                   │")
    print("│  ───────────────────────                                                    │")
    print("│  • Tests strategy across BULL, BEAR, and SIDEWAYS segments                  │")
    print("│  • Aggregates fitness using harmonic mean (penalizes inconsistency)         │")
    print("│  • Result: More robust strategies that work in ALL market conditions        │")
    print("│                                                                             │")
    print("│          Bull   Bear   Side   │ Harmonic Mean  │ Robustness               │")
    print("│  ───────────────────────────────────────────────────────────────────────   │")
    print("│  Strat A: 0.85   0.25   0.30  │     0.39       │ ❌ Poor (regime-specific) │")
    print("│  Strat B: 0.65   0.58   0.60  │     0.61       │ ✅ Good (balanced)        │")
    print("│  Strat C: 0.55   0.52   0.58  │     0.55       │ ✅ Best (consistent)      │")
    print("│                                                                             │")
    print("│  → Strategy C wins under regime-aware evaluation!                           │")
    print("│                                                                             │")
    print("└─────────────────────────────────────────────────────────────────────────────┘")


def main():
    """Run the complete visualization demo."""
    print()
    print("=" * 77)
    print("          REGIME-AWARE FITNESS EVALUATION - VISUAL DEMONSTRATION")
    print("=" * 77)
    print()
    print("This demo shows how regime-aware evaluation helps produce robust strategies")
    print("that perform well across different market conditions (bull/bear/sideways).")
    print()
    
    # Step 1: Create synthetic data
    print("📊 STEP 1: Creating synthetic market data with regime patterns...")
    print("─" * 77)
    df = create_synthetic_market_data(days=365)
    
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"   Generated 365 days of synthetic OHLCV data")
    print(f"   Total return: {total_return:.1f}%")
    print(f"   Designed regimes: BULL(Jan-Apr), BEAR(Apr-Jul), SIDE(Jul-Oct), BULL(Oct-Dec)")
    
    # Step 2: Detect regimes
    print("\n🔍 STEP 2: Detecting market regimes using SMA/ADX method...")
    print("─" * 77)
    
    detector = RegimeDetector(method='sma_adx')
    regime_series = detector.detect(df)
    
    # Print regime distribution
    regime_counts = regime_series.value_counts()
    print("\n   Detected regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_series) * 100
        bar_len = int(pct // 2)
        bar = '█' * bar_len
        print(f"   {regime.value:10s}: {bar:25s} {count:3d} days ({pct:.1f}%)")
    
    # Print ASCII chart
    print_ascii_price_chart(df, regime_series)
    
    # Print regime timeline bar
    print_regime_bar(regime_series)
    
    # Step 3: Create segments
    print("📁 STEP 3: Classifying into regime segments...")
    print("─" * 77)
    
    segments = detector.classify_periods(
        df, 
        period_days=60, 
        min_period_days=40,
        embargo_days=5,
        warmup_bars=50
    )
    
    print_segment_table(segments)
    
    # Step 4: Balance and split
    print("\n⚖️ STEP 4: Balancing segments and splitting train/holdout...")
    print("─" * 77)
    
    balanced = detector.get_balanced_segments(segments, segments_per_regime=2)
    splits = detector.split_segments_by_role(
        balanced,
        optimization_ratio=0.70,
        model_selection_ratio=0.0,
        holdout_ratio=0.30,
    )
    
    print(f"\n   Optimization segments: {len(splits['optimization'])}")
    for seg in splits['optimization']:
        print(f"      • {seg.segment_id} ({seg.regime.value}, {seg.confidence:.0%} confidence)")
    
    print(f"\n   Holdout segments: {len(splits['holdout'])}")
    for seg in splits['holdout']:
        print(f"      • {seg.segment_id} ({seg.regime.value}, {seg.confidence:.0%} confidence)")
    
    # Step 5: Demonstrate aggregation
    print_aggregation_demo()
    
    # Step 6: Show comparison
    print_comparison_table()
    
    # Summary
    print("\n" + "=" * 77)
    print("                              ✅ SUMMARY")
    print("=" * 77)
    print("""
    The Regime-Aware Evaluation system:
    
    1. DETECTS market regimes (bullish, bearish, sideways) in historical data
    2. SEGMENTS the data into regime-labeled periods  
    3. EVALUATES strategies across ALL segments (not just one period)
    4. AGGREGATES fitness using harmonic mean (penalizes inconsistency)
    5. RESERVES holdout segments for final out-of-sample validation
    
    Result: Strategies that are robust across different market conditions,
            not just strategies that overfit to a single regime.
    
    Enable in ga_config.yaml:
    
        regime_aware:
          enabled: true
          method: 'sma_adx'
          aggregation: 'harmonic_mean'
    """)
    print("=" * 77)


if __name__ == '__main__':
    main()
