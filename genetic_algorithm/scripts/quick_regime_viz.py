#!/usr/bin/env python3
"""
Quick Regime Detection Visualization

Generates a quick visual overview of regime detection on your data.
Useful for validating that regime detection is working correctly
before running a full GA evolution.

Usage:
    python genetic_algorithm/scripts/quick_regime_viz.py
    python genetic_algorithm/scripts/quick_regime_viz.py --pair BTC/USDT --timeframe 4h
    python genetic_algorithm/scripts/quick_regime_viz.py --method ensemble

Author: GA System
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.utils.regime_detector import RegimeDetector


def load_data(pair: str, timeframe: str, data_dir: Path) -> pd.DataFrame:
    """Load price data for a pair/timeframe."""
    # Format pair for filename
    pair_formatted = pair.replace('/', '_')
    filename = f"{pair_formatted}-{timeframe}.feather"
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_feather(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_regime_visualization(
    df: pd.DataFrame,
    method: str,
    pair: str,
    timeframe: str,
    output_dir: Path,
    months: int = 6
):
    """Create comprehensive regime visualization."""
    
    # Detect regimes
    print(f"Detecting regimes using method: {method}")
    detector = RegimeDetector(method=method)
    labels = detector.detect(df)
    
    # Add regime to dataframe
    df = df.copy()
    df['regime'] = labels.apply(lambda x: x.value if hasattr(x, 'value') else str(x)).values
    df['returns'] = df['close'].pct_change()
    df = df.set_index('date')
    
    # Get recent data
    df_recent = df.last(f'{months}M')
    
    # Create figure
    fig = plt.figure(figsize=(16, 14))
    
    # Colors for regimes
    colors = {
        'bullish': 'green',
        'bearish': 'red',
        'sideways': 'gray',
        'volatile': 'orange',
        'uncertain': 'blue'
    }
    
    # === Plot 1: Price with regime overlay ===
    ax1 = fig.add_subplot(3, 2, (1, 2))
    
    for regime, color in colors.items():
        mask = df_recent['regime'] == regime
        if mask.any():
            ax1.fill_between(
                df_recent.index,
                df_recent['close'].min(),
                df_recent['close'].max(),
                where=mask,
                alpha=0.3,
                color=color,
                label=regime.title()
            )
    
    ax1.plot(df_recent.index, df_recent['close'], 'k-', linewidth=0.8)
    ax1.set_title(f'{pair} {timeframe} - Regime Detection ({method})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Regime Distribution (Pie) ===
    ax2 = fig.add_subplot(3, 2, 3)
    
    regime_counts = df['regime'].value_counts()
    colors_pie = [colors.get(r, 'gray') for r in regime_counts.index]
    
    wedges, texts, autotexts = ax2.pie(
        regime_counts.values,
        labels=regime_counts.index,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('Regime Distribution (Full Data)', fontsize=12, fontweight='bold')
    
    # === Plot 3: Conditional Returns ===
    ax3 = fig.add_subplot(3, 2, 4)
    
    regime_returns = df.groupby('regime')['returns'].mean() * 100 * 6  # Daily returns
    colors_bar = [colors.get(r, 'gray') for r in regime_returns.index]
    
    bars = ax3.bar(regime_returns.index, regime_returns.values, color=colors_bar)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Average Daily Return by Regime', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Daily Return (%)')
    
    for bar, val in zip(bars, regime_returns.values):
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.02 if val >= 0 else -0.05),
            f'{val:.3f}%',
            ha='center',
            va='bottom' if val >= 0 else 'top',
            fontsize=10
        )
    
    # === Plot 4: Regime Transitions ===
    ax4 = fig.add_subplot(3, 2, 5)
    
    df_trans = df_recent.copy()
    regime_map = {'bullish': 2, 'sideways': 1, 'bearish': 0, 'volatile': 1.5, 'uncertain': 1}
    df_trans['regime_num'] = df_trans['regime'].map(regime_map)
    
    ax4.step(df_trans.index, df_trans['regime_num'], where='post', color='blue', linewidth=1)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Bearish', 'Sideways', 'Bullish'])
    ax4.set_title('Regime Transitions', fontsize=12, fontweight='bold')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # === Plot 5: Validation Metrics ===
    ax5 = fig.add_subplot(3, 2, 6)
    ax5.axis('off')
    
    # Calculate metrics
    flip_rate = (df['regime'] != df['regime'].shift(1)).sum() / len(df) * 100
    bull_ret = df[df['regime'] == 'bullish']['returns'].mean() * 100
    bear_ret = df[df['regime'] == 'bearish']['returns'].mean() * 100
    side_ret = df[df['regime'] == 'sideways']['returns'].mean() * 100
    
    validation_pass = bull_ret > bear_ret
    
    metrics_text = f"""
REGIME DETECTION VALIDATION
{'━' * 36}

Method: {method}
Pair: {pair} ({timeframe})
Data: {len(df)} candles

DISTRIBUTION:
  • Bullish:  {regime_counts.get('bullish', 0)/len(df)*100:.1f}%
  • Bearish:  {regime_counts.get('bearish', 0)/len(df)*100:.1f}%
  • Sideways: {regime_counts.get('sideways', 0)/len(df)*100:.1f}%

STABILITY:
  • Flip Rate: {flip_rate:.2f}% (lower = better)
  
CONDITIONAL RETURNS (per bar):
  • Bullish:  {bull_ret:+.4f}%
  • Bearish:  {bear_ret:+.4f}%
  • Sideways: {side_ret:+.4f}%

VALIDATION: {'✓ PASS' if validation_pass else '✗ FAIL'}
(Expected: Bullish > Bearish returns)
"""
    
    ax5.text(
        0.1, 0.5, metrics_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"regime_viz_{pair.replace('/', '_')}_{timeframe}_{method}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    
    plt.close()
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Quick regime detection visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default visualization
    python quick_regime_viz.py
    
    # Specific pair and timeframe
    python quick_regime_viz.py --pair ETH/BTC --timeframe 4h
    
    # Test different detection methods
    python quick_regime_viz.py --method ensemble
    python quick_regime_viz.py --method rolling_returns
    python quick_regime_viz.py --method hmm
        """
    )
    
    parser.add_argument(
        '--pair', '-p',
        default='BTC/USDT',
        help='Trading pair (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--timeframe', '-t',
        default='4h',
        help='Timeframe (default: 4h)'
    )
    
    parser.add_argument(
        '--method', '-m',
        default='adx_di_hysteresis',
        choices=['adx_di_hysteresis', 'rolling_returns', 'hmm', 'ensemble', 'sma_adx'],
        help='Detection method (default: adx_di_hysteresis)'
    )
    
    parser.add_argument(
        '--months',
        type=int,
        default=6,
        help='Months of data to visualize (default: 6)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='genetic_algorithm/output/regime_viz',
        help='Output directory'
    )
    
    parser.add_argument(
        '--data-dir', '-d',
        default='user_data/data/binance',
        help='Data directory'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading data for {args.pair} {args.timeframe}...")
    
    try:
        df = load_data(args.pair, args.timeframe, data_dir)
        print(f"Loaded {len(df)} candles")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable data files:")
        for f in data_dir.glob("*.feather"):
            print(f"  {f.name}")
        return 1
    
    filepath = create_regime_visualization(
        df=df,
        method=args.method,
        pair=args.pair,
        timeframe=args.timeframe,
        output_dir=output_dir,
        months=args.months
    )
    
    print(f"\nVisualization complete!")
    print(f"Open with: xdg-open {filepath}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
