#!/usr/bin/env python3
"""
Demo: Regime Detection with Real Market Data

This script demonstrates the RegimeDetector working with actual ETH/BTC data
from the user_data/data/binance directory.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the freqtradeForkGA directory to path (grandparent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType, save_segments_to_yaml


def load_feather_data(filepath: Path) -> pd.DataFrame:
    """Load OHLCV data from feather file."""
    df = pd.read_feather(filepath)
    
    # Set datetime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    return df


def main():
    print("\n" + "=" * 80)
    print("REGIME DETECTION DEMO - Real Market Data")
    print("=" * 80)
    
    # Load ETH/BTC 4h data
    data_dir = Path(__file__).parent.parent.parent / "user_data" / "data" / "binance"
    feather_file = data_dir / "ETH_BTC-4h.feather"
    
    if not feather_file.exists():
        print(f"Data file not found: {feather_file}")
        return
    
    print(f"\n📊 Loading data from: {feather_file.name}")
    df = load_feather_data(feather_file)
    
    print(f"   Loaded {len(df):,} candles")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"   Columns: {list(df.columns)}")
    
    # Initialize regime detector
    print("\n🔍 Initializing RegimeDetector (method='sma_adx')...")
    detector = RegimeDetector(method='sma_adx')
    
    # Detect regimes for all bars
    print("\n📈 Detecting regimes...")
    regimes = detector.detect(df)
    
    # Show regime distribution
    print("\n📊 Per-bar Regime Distribution:")
    regime_counts = regimes.value_counts()
    total = len(regimes)
    for regime, count in regime_counts.items():
        bar = "█" * int(count / total * 40)
        print(f"   {regime.value:12s}: {count:6,} ({count/total*100:5.1f}%) {bar}")
    
    # Classify into periods
    print("\n📅 Classifying into regime periods...")
    segments = detector.classify_periods(
        df,
        period_days=90,
        min_period_days=60,
        embargo_days=7,
        warmup_bars=200
    )
    
    print(f"\n📋 Created {len(segments)} segments:")
    for seg in segments:
        confidence_bar = "●" * int(seg.confidence * 10)
        print(f"   {seg.timerange}: {seg.regime.value:10s} ({seg.confidence:.0%}) {confidence_bar}")
        if seg.metadata:
            ret = seg.metadata.get('total_return', 0)
            vol = seg.metadata.get('volatility', 0)
            print(f"      └── Return: {ret*100:+.1f}%, Volatility: {vol*100:.2f}%")
    
    # Get balanced segments
    print("\n⚖️  Selecting balanced segments (2 per regime)...")
    balanced = detector.get_balanced_segments(
        segments,
        segments_per_regime=2,
        target_regimes=[RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS]
    )
    
    print(f"\n📦 Balanced selection ({len(balanced)} segments):")
    regime_counts = {}
    for seg in balanced:
        regime_counts[seg.regime.value] = regime_counts.get(seg.regime.value, 0) + 1
        print(f"   {seg.segment_id}: {seg.regime.value} ({seg.confidence:.0%})")
    
    print(f"\n   Regime counts: {regime_counts}")
    
    # Split into train/holdout
    if len(segments) >= 3:
        print("\n🔀 Splitting segments into train/model_selection/holdout...")
        splits = detector.split_segments_by_role(
            segments,
            optimization_ratio=0.60,
            model_selection_ratio=0.20,
            holdout_ratio=0.20
        )
        
        print(f"\n📂 Split results:")
        for role, segs in splits.items():
            print(f"   {role}: {len(segs)} segments")
            for seg in segs:
                print(f"      - {seg.timerange}: {seg.regime.value}")
        
        # Save to YAML
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        yaml_path = output_dir / f"segments_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        print(f"\n💾 Saving segments to: {yaml_path.name}")
        save_segments_to_yaml(
            splits,
            yaml_path,
            metadata={
                'pair': 'ETH/BTC',
                'timeframe': '4h',
                'detector_method': 'sma_adx',
                'data_source': str(feather_file.name)
            }
        )
    
    print("\n" + "=" * 80)
    print("✅ Demo completed successfully")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
