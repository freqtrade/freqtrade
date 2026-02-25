#!/usr/bin/env python3
"""
Regime Detection Visualization

Creates visual plots of detected market regimes overlaid on price charts.
Generates plots for BTC/USDT showing how different detection methods classify
market regimes (bullish, bearish, sideways).
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the freqtradeForkGA directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType


# Regime color mapping
REGIME_COLORS = {
    RegimeType.BULLISH: '#2ecc71',    # Green
    RegimeType.BEARISH: '#e74c3c',    # Red
    RegimeType.SIDEWAYS: '#f39c12',   # Orange/Yellow
    RegimeType.VOLATILE: '#9b59b6',   # Purple
    RegimeType.UNCERTAIN: '#95a5a6',  # Gray
}

REGIME_ALPHA = 0.3  # Transparency for background fills


def load_feather_data(filepath: Path) -> pd.DataFrame:
    """Load OHLCV data from feather file."""
    df = pd.read_feather(filepath)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    return df


def plot_regime_background(ax, df: pd.DataFrame, regimes: pd.Series):
    """
    Add colored background regions for each regime.
    
    Args:
        ax: Matplotlib axis
        df: OHLCV DataFrame
        regimes: Series of RegimeType values
    """
    # Find regime change points
    regime_changes = regimes != regimes.shift(1)
    change_indices = df.index[regime_changes]
    
    # Add regime background fills
    prev_idx = df.index[0]
    prev_regime = regimes.iloc[0]
    
    for idx in change_indices[1:]:
        color = REGIME_COLORS.get(prev_regime, '#95a5a6')
        ax.axvspan(prev_idx, idx, alpha=REGIME_ALPHA, color=color, linewidth=0)
        prev_idx = idx
        prev_regime = regimes.loc[idx]
    
    # Fill the last segment
    color = REGIME_COLORS.get(prev_regime, '#95a5a6')
    ax.axvspan(prev_idx, df.index[-1], alpha=REGIME_ALPHA, color=color, linewidth=0)


def plot_price_with_regimes(
    df: pd.DataFrame,
    regimes: pd.Series,
    title: str,
    method: str,
    ax=None,
    show_indicators: bool = False,
    detector: RegimeDetector = None,
):
    """
    Plot price chart with regime-colored background.
    
    Args:
        df: OHLCV DataFrame
        regimes: Series of RegimeType for each bar
        title: Plot title
        method: Detection method name
        ax: Optional existing axis
        show_indicators: Whether to show the detection indicators
        detector: RegimeDetector instance (for showing indicators)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    
    # Add regime backgrounds
    plot_regime_background(ax, df, regimes)
    
    # Plot candlestick-style price (simplified as line with fills)
    ax.plot(df.index, df['close'], color='black', linewidth=0.8, alpha=0.9, label='Close')
    
    # Add high-low range as a subtle fill
    ax.fill_between(df.index, df['low'], df['high'], alpha=0.1, color='blue')
    
    # Formatting
    ax.set_title(f'{title}\nMethod: {method}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (USDT)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Legend for regimes
    legend_elements = [
        Patch(facecolor=REGIME_COLORS[RegimeType.BULLISH], alpha=0.5, label='Bullish'),
        Patch(facecolor=REGIME_COLORS[RegimeType.BEARISH], alpha=0.5, label='Bearish'),
        Patch(facecolor=REGIME_COLORS[RegimeType.SIDEWAYS], alpha=0.5, label='Sideways'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    return ax


def plot_regime_distribution(regimes: pd.Series, title: str, ax=None):
    """
    Create a pie chart showing regime distribution.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    counts = regimes.value_counts()
    labels = [r.value.capitalize() for r in counts.index]
    colors = [REGIME_COLORS[r] for r in counts.index]
    sizes = counts.values
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(sizes)
    )
    
    ax.set_title(f'{title}\nRegime Distribution', fontsize=12, fontweight='bold')
    
    return ax


def plot_regime_timeline(df: pd.DataFrame, regimes: pd.Series, title: str, ax=None):
    """
    Create a timeline bar showing regime changes over time.
    Uses optimized contiguous block drawing instead of per-bar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 1.5))
    
    # Find regime change points and draw contiguous blocks
    regime_changes = (regimes != regimes.shift(1)).fillna(True)
    change_indices = df.index[regime_changes].tolist()
    change_indices.append(df.index[-1])  # Add end point
    
    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]
        regime = regimes.loc[start_idx]
        color = REGIME_COLORS.get(regime, '#95a5a6')
        ax.axvspan(start_idx, end_idx, color=color, linewidth=0)
    
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_yticks([])
    ax.set_title(f'{title} - Regime Timeline', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return ax


def create_comprehensive_plot(
    df: pd.DataFrame,
    pair: str,
    timeframe: str,
    output_dir: Path,
    methods: list = None,
):
    """
    Create a comprehensive visualization comparing all detection methods.
    """
    if methods is None:
        methods = ['sma_adx', 'adx_di', 'returns', 'bollinger', 'ensemble']
    
    # Create figure with subplots
    n_methods = len(methods)
    fig = plt.figure(figsize=(18, 4 * n_methods + 4))
    
    # Create grid spec for layout
    gs = fig.add_gridspec(n_methods + 1, 4, height_ratios=[1] * n_methods + [0.5],
                          width_ratios=[3, 1, 3, 1], hspace=0.4, wspace=0.3)
    
    print(f"\n📊 Creating regime visualization for {pair} {timeframe}...")
    print(f"   Data range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"   Total candles: {len(df):,}")
    
    for i, method in enumerate(methods):
        print(f"\n   🔍 Processing method: {method}...")
        
        # Detect regimes
        detector = RegimeDetector(method=method)
        regimes = detector.detect(df)
        
        # Count regimes
        counts = regimes.value_counts()
        total = len(regimes)
        bull_pct = counts.get(RegimeType.BULLISH, 0) / total * 100
        bear_pct = counts.get(RegimeType.BEARISH, 0) / total * 100
        side_pct = counts.get(RegimeType.SIDEWAYS, 0) / total * 100
        print(f"      Bullish: {bull_pct:.1f}%, Bearish: {bear_pct:.1f}%, Sideways: {side_pct:.1f}%")
        
        # Price chart with regimes (left side, spans 3 columns)
        ax_price = fig.add_subplot(gs[i, 0:3])
        plot_price_with_regimes(
            df, regimes,
            title=f'{pair} {timeframe}',
            method=method.upper(),
            ax=ax_price,
        )
        
        # Regime distribution pie (right side)
        ax_pie = fig.add_subplot(gs[i, 3])
        plot_regime_distribution(regimes, '', ax=ax_pie)
        ax_pie.set_title(f'{method.upper()}\nDistribution', fontsize=10)
    
    # Combined timeline at the bottom
    ax_timeline = fig.add_subplot(gs[n_methods, :])
    
    # Use ensemble method for the summary timeline
    detector = RegimeDetector(method='ensemble')
    regimes = detector.detect(df)
    plot_regime_timeline(df, regimes, f'{pair} {timeframe} - Ensemble', ax=ax_timeline)
    
    # Main title
    fig.suptitle(
        f'Market Regime Detection: {pair} ({timeframe})\n'
        f'Data: {df.index.min().strftime("%Y-%m-%d")} to {df.index.max().strftime("%Y-%m-%d")}',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'regime_detection_{pair.replace("/", "_")}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    filepath = output_dir / filename
    
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"\n   ✅ Saved: {filepath}")
    
    return filepath


def create_method_comparison_plot(
    df: pd.DataFrame,
    pair: str,
    timeframe: str,
    output_dir: Path,
):
    """
    Create a single plot comparing all methods side by side with a shared x-axis.
    """
    methods = ['sma_adx', 'adx_di', 'returns', 'bollinger', 'ensemble']
    
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(18, 12), sharex=True)
    
    # Top plot: Price chart
    ax_price = axes[0]
    ax_price.plot(df.index, df['close'], color='black', linewidth=0.8)
    ax_price.fill_between(df.index, df['low'], df['high'], alpha=0.15, color='blue')
    ax_price.set_ylabel('Price', fontsize=10)
    ax_price.set_title(f'{pair} {timeframe} - Regime Detection Comparison', fontsize=14, fontweight='bold')
    ax_price.grid(True, alpha=0.3)
    
    # One row per method showing regime timeline
    for i, method in enumerate(methods):
        ax = axes[i + 1]
        
        detector = RegimeDetector(method=method)
        regimes = detector.detect(df)
        
        # Create colored segments - optimized drawing of contiguous blocks
        regime_changes = (regimes != regimes.shift(1)).fillna(True)
        change_indices = df.index[regime_changes].tolist()
        change_indices.append(df.index[-1])
        
        for j in range(len(change_indices) - 1):
            start_idx = change_indices[j]
            end_idx = change_indices[j + 1]
            regime = regimes.loc[start_idx]
            color = REGIME_COLORS.get(regime, '#95a5a6')
            ax.axvspan(start_idx, end_idx, color=color, linewidth=0)
        
        ax.set_ylabel(method.upper(), fontsize=10, rotation=0, labelpad=50, va='center')
        ax.set_yticks([])
        ax.set_xlim(df.index[0], df.index[-1])
    
    # Legend
    legend_elements = [
        Patch(facecolor=REGIME_COLORS[RegimeType.BULLISH], alpha=0.7, label='Bullish'),
        Patch(facecolor=REGIME_COLORS[RegimeType.BEARISH], alpha=0.7, label='Bearish'),
        Patch(facecolor=REGIME_COLORS[RegimeType.SIDEWAYS], alpha=0.7, label='Sideways'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=3)
    
    # X-axis formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'regime_comparison_{pair.replace("/", "_")}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    filepath = output_dir / filename
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"   ✅ Saved comparison: {filepath}")
    
    return filepath


def main():
    print("\n" + "=" * 80)
    print("🎨 REGIME DETECTION VISUALIZATION")
    print("=" * 80)
    
    # Setup paths
    data_dir = Path(__file__).parent.parent.parent / "user_data" / "data" / "binance"
    output_dir = Path(__file__).parent.parent / "output" / "regime_visualizations"
    
    # Data files to process
    pairs_timeframes = [
        ('BTC/USDT', '30m', 'BTC_USDT-30m.feather'),
        ('BTC/USDT', '1d', 'BTC_USDT-1d.feather'),
    ]
    
    saved_files = []
    
    for pair, timeframe, filename in pairs_timeframes:
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"\n⚠️  Data file not found: {filepath}")
            print(f"   Run: freqtrade download-data --pairs {pair} --timeframes {timeframe} --days 365")
            continue
        
        print(f"\n📂 Loading {pair} {timeframe} from {filename}...")
        df = load_feather_data(filepath)
        
        print(f"   Loaded {len(df):,} candles")
        print(f"   Range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Create comprehensive plot
        plot_path = create_comprehensive_plot(df, pair, timeframe, output_dir)
        saved_files.append(plot_path)
        
        # Create comparison plot
        comparison_path = create_method_comparison_plot(df, pair, timeframe, output_dir)
        saved_files.append(comparison_path)
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print("\n📊 Generated files:")
    for f in saved_files:
        print(f"   - {f.name}")
    
    print("\n" + "=" * 80 + "\n")
    
    return saved_files


if __name__ == '__main__':
    main()
