#!/usr/bin/env python3
"""
Plot BTC/USDT 4h chart with regime segments color-coded.
Shows bullish (green), bearish (red), sideways (yellow) classification.
"""
import sys
sys.path.insert(0, '/home/kali/trading/freqtradeForkGA')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from genetic_algorithm.utils.regime_detector import (
    RegimeDetector, RegimeType, load_ohlcv_data,
)


def main():
    # ── Load data ──
    datadir = Path('user_data/data/binance')
    timerange = '20240301-20260228'
    pair = 'BTC/USDT'
    timeframe = '4h'

    print(f"Loading {pair} {timeframe} data ({timerange})...")
    df = load_ohlcv_data(pair=pair, timeframe=timeframe, datadir=datadir, timerange=timerange)

    if df.empty:
        print("ERROR: No data loaded.")
        return

    print(f"  Loaded {len(df)} candles: {df.index[0]} → {df.index[-1]}")

    # ── Detect regimes (ensemble method, same as island model) ──
    print("Detecting regimes (ensemble method)...")
    detector = RegimeDetector(method='ensemble')
    regimes = detector.detect(df)

    # ── Classify into segments (60-day periods, same as config) ──
    print("Classifying into 60-day segments...")
    segments = detector.classify_periods(
        df=df,
        period_days=60,
        min_period_days=30,
        embargo_days=3,
        warmup_bars=200,
    )

    print(f"  Found {len(segments)} segments:")
    for seg in segments:
        print(f"    {seg.regime.value:>10s}: {seg.start_date.strftime('%Y-%m-%d')} → "
              f"{seg.end_date.strftime('%Y-%m-%d')} ({seg.duration_days}d, "
              f"conf={seg.confidence:.2f})")

    # Count by regime
    regime_counts = {}
    for seg in segments:
        key = seg.regime.value
        regime_counts[key] = regime_counts.get(key, 0) + 1
    print(f"\n  Regime distribution: {regime_counts}")

    # ── Plot 1: Price chart with per-candle regime coloring ──
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), 
                              gridspec_kw={'height_ratios': [3, 1, 1]},
                              sharex=True)
    fig.suptitle(f'{pair} {timeframe} — Regime Classification (Ensemble Method)\n'
                 f'{timerange}', fontsize=16, fontweight='bold')

    ax_price = axes[0]
    ax_regime = axes[1]
    ax_segments = axes[2]

    # Colors
    regime_colors = {
        RegimeType.BULLISH: '#2ecc71',    # Green
        RegimeType.BEARISH: '#e74c3c',    # Red
        RegimeType.SIDEWAYS: '#f39c12',   # Orange/Yellow
        RegimeType.VOLATILE: '#9b59b6',   # Purple
        RegimeType.UNCERTAIN: '#95a5a6',  # Gray
    }

    # --- Subplot 1: Price with regime background ---
    dates = df.index
    prices = df['close'].values

    # Plot price line
    ax_price.plot(dates, prices, color='#2c3e50', linewidth=0.8, alpha=0.9, zorder=3)

    # Color background by per-candle regime
    regimes_aligned = regimes.reindex(df.index)
    prev_regime = None
    start_idx = 0

    for i in range(len(dates)):
        current = regimes_aligned.iloc[i] if i < len(regimes_aligned) else RegimeType.UNCERTAIN
        
        if current != prev_regime and prev_regime is not None:
            # Fill background for previous regime block
            color = regime_colors.get(prev_regime, '#95a5a6')
            ax_price.axvspan(dates[start_idx], dates[i], alpha=0.15, color=color, zorder=1)
            start_idx = i
        prev_regime = current

    # Final block
    if prev_regime is not None and start_idx < len(dates):
        color = regime_colors.get(prev_regime, '#95a5a6')
        ax_price.axvspan(dates[start_idx], dates[-1], alpha=0.15, color=color, zorder=1)

    ax_price.set_ylabel('Price (USDT)', fontsize=12)
    ax_price.set_title('Price with Per-Candle Regime Background', fontsize=13)
    ax_price.grid(True, alpha=0.3)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Legend
    legend_patches = [
        mpatches.Patch(color=regime_colors[RegimeType.BULLISH], alpha=0.4, label='Bullish'),
        mpatches.Patch(color=regime_colors[RegimeType.BEARISH], alpha=0.4, label='Bearish'),
        mpatches.Patch(color=regime_colors[RegimeType.SIDEWAYS], alpha=0.4, label='Sideways'),
    ]
    ax_price.legend(handles=legend_patches, loc='upper left', fontsize=11)

    # --- Subplot 2: Regime timeline (bar chart) ---
    regime_numeric = regimes_aligned.map({
        RegimeType.BULLISH: 1,
        RegimeType.BEARISH: -1,
        RegimeType.SIDEWAYS: 0,
        RegimeType.VOLATILE: 0.5,
        RegimeType.UNCERTAIN: 0,
    }).fillna(0)

    colors_bar = [regime_colors.get(r, '#95a5a6') for r in regimes_aligned]
    ax_regime.bar(dates, regime_numeric.values, color=colors_bar, width=0.2, alpha=0.7)
    ax_regime.set_ylabel('Regime', fontsize=12)
    ax_regime.set_title('Per-Candle Regime Signal (1=Bull, 0=Sideways, -1=Bear)', fontsize=13)
    ax_regime.set_yticks([-1, 0, 1])
    ax_regime.set_yticklabels(['Bearish', 'Sideways', 'Bullish'])
    ax_regime.grid(True, alpha=0.3)
    ax_regime.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # --- Subplot 3: 60-day Segment Classification ---
    ax_segments.set_ylim(-0.5, 0.5)
    ax_segments.set_title('60-Day Segment Classification (used by Island Model)', fontsize=13)

    for seg in segments:
        color = regime_colors.get(seg.regime, '#95a5a6')
        ax_segments.axvspan(seg.start_date, seg.end_date, alpha=0.5, color=color, zorder=2)
        # Label
        mid = seg.start_date + (seg.end_date - seg.start_date) / 2
        ax_segments.text(mid, 0, f"{seg.regime.value}\n{seg.confidence:.0%}",
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='black', zorder=3)

    ax_segments.set_ylabel('Segments', fontsize=12)
    ax_segments.set_yticks([])
    ax_segments.grid(True, alpha=0.3, axis='x')

    # Format x-axis
    ax_segments.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_segments.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save
    output_path = Path('genetic_algorithm/output/regime_chart.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path.absolute()}")
    plt.close()

    # ── Also print a segment summary table ──
    print(f"\n{'='*70}")
    print("REGIME SEGMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'#':>3} {'Regime':>10} {'Start':>12} {'End':>12} {'Days':>5} {'Conf':>5}")
    print(f"{'-'*50}")
    for i, seg in enumerate(segments, 1):
        print(f"{i:>3} {seg.regime.value:>10} {seg.start_date.strftime('%Y-%m-%d'):>12} "
              f"{seg.end_date.strftime('%Y-%m-%d'):>12} {seg.duration_days:>5} "
              f"{seg.confidence:>5.0%}")

    # ── Island assignment summary ──
    print(f"\n{'='*70}")
    print("ISLAND ASSIGNMENT (what each island trains on)")
    print(f"{'='*70}")
    
    # Simulate the same logic as island_model.py
    balanced = detector.get_balanced_segments(
        segments,
        segments_per_regime=max(3, len(segments) // 4),
    )
    
    splits = detector.split_segments_by_role(
        balanced,
        optimization_ratio=0.80,
        model_selection_ratio=0.0,
        holdout_ratio=0.20,
    )
    
    for role in ['optimization', 'holdout']:
        segs = splits.get(role, [])
        print(f"\n  {role.upper()} segments ({len(segs)}):")
        for seg in segs:
            print(f"    {seg.regime.value:>10}: {seg.start_date.strftime('%Y-%m-%d')} → "
                  f"{seg.end_date.strftime('%Y-%m-%d')} ({seg.duration_days}d)")
    
    print(f"\n  Bullish island  → trains on bullish optimization segments only")
    print(f"  Bearish island  → trains on bearish optimization segments only")
    print(f"  Sideways island → trains on sideways optimization segments only")
    print(f"  Master island   → trains on ALL optimization segments")


if __name__ == '__main__':
    main()
