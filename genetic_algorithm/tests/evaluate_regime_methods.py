#!/usr/bin/env python3
"""
Phase 6: Advanced Regime Detection - Evaluation Framework

This script evaluates and compares different regime detection methods:
1. Current SMA(50/200) + ADX method (baseline)
2. Rolling Returns Classification
3. HMM (Hidden Markov Model) using hmmlearn
4. Markov Switching using statsmodels
5. Change Point Detection using ruptures
6. ADX + DI with hysteresis

Metrics:
- Regime distribution (should be ~30-40% each)
- Flip rate (regime changes per time unit)
- Dwell time distribution (how long regimes persist)
- Conditional returns per regime (regimes should have different mean returns)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add the freqtradeForkGA directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import existing detector
from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType

# New libraries for advanced detection
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available")

try:
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels Markov models not available")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures not available")

try:
    from ta.trend import ADXIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not available")


@dataclass
class RegimeMetrics:
    """Metrics for evaluating a regime detection method."""
    method_name: str
    distribution: Dict[str, float]  # Regime -> percentage
    flip_rate: float  # Regime changes per 100 bars
    mean_dwell_time: float  # Average bars per regime
    conditional_returns: Dict[str, float]  # Regime -> mean daily return
    coverage: float  # Percentage of bars with valid regime
    

def load_feather_data(filepath: Path) -> pd.DataFrame:
    """Load OHLCV data from feather file."""
    df = pd.read_feather(filepath)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    # Ensure lowercase columns
    df.columns = df.columns.str.lower()
    return df


def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate log returns from close prices."""
    return np.log(df['close'] / df['close'].shift(1))


def calculate_flip_rate(regimes: pd.Series) -> float:
    """Calculate regime changes per 100 bars."""
    changes = (regimes != regimes.shift(1)).sum()
    return (changes / len(regimes)) * 100


def calculate_dwell_times(regimes: pd.Series) -> List[int]:
    """Calculate list of consecutive regime durations."""
    dwell_times = []
    current_regime = None
    current_count = 0
    
    for regime in regimes:
        if pd.isna(regime):
            continue
        if regime == current_regime:
            current_count += 1
        else:
            if current_count > 0:
                dwell_times.append(current_count)
            current_regime = regime
            current_count = 1
    
    if current_count > 0:
        dwell_times.append(current_count)
    
    return dwell_times


def evaluate_method(regimes: pd.Series, returns: pd.Series, method_name: str) -> RegimeMetrics:
    """Evaluate a regime detection method."""
    # Filter out NaN regimes
    valid_mask = ~pd.isna(regimes)
    valid_regimes = regimes[valid_mask]
    valid_returns = returns[valid_mask]
    
    # Coverage
    coverage = valid_mask.sum() / len(regimes)
    
    # Distribution
    dist_counts = valid_regimes.value_counts(normalize=True)
    distribution = {str(k): v for k, v in dist_counts.items()}
    
    # Flip rate
    flip_rate = calculate_flip_rate(valid_regimes)
    
    # Dwell times
    dwell_times = calculate_dwell_times(valid_regimes)
    mean_dwell = np.mean(dwell_times) if dwell_times else 0
    
    # Conditional returns
    conditional_returns = {}
    for regime in valid_regimes.unique():
        mask = valid_regimes == regime
        regime_returns = valid_returns[mask]
        conditional_returns[str(regime)] = regime_returns.mean() if len(regime_returns) > 0 else 0
    
    return RegimeMetrics(
        method_name=method_name,
        distribution=distribution,
        flip_rate=flip_rate,
        mean_dwell_time=mean_dwell,
        conditional_returns=conditional_returns,
        coverage=coverage
    )


def print_metrics(metrics: RegimeMetrics):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"📊 {metrics.method_name}")
    print(f"{'='*60}")
    
    print(f"\n📈 Regime Distribution (target: ~33% each):")
    for regime, pct in sorted(metrics.distribution.items()):
        bar = "█" * int(pct * 40)
        regime_str = str(regime).replace("RegimeType.", "")
        print(f"   {regime_str:15s}: {pct*100:5.1f}% {bar}")
    
    print(f"\n📉 Stability Metrics:")
    print(f"   Flip Rate:    {metrics.flip_rate:.2f} changes per 100 bars")
    print(f"   Mean Dwell:   {metrics.mean_dwell_time:.1f} bars per regime")
    print(f"   Coverage:     {metrics.coverage*100:.1f}%")
    
    print(f"\n💰 Conditional Returns (should differ by regime):")
    for regime, ret in sorted(metrics.conditional_returns.items()):
        regime_str = str(regime).replace("RegimeType.", "")
        print(f"   {regime_str:15s}: {ret*100:+.4f}% per bar")


# ============================================================================
# REGIME DETECTION METHODS
# ============================================================================

def detect_baseline_sma_adx(df: pd.DataFrame) -> pd.Series:
    """Baseline: Current SMA(50/200) + ADX method."""
    detector = RegimeDetector(method='sma_adx')
    regimes = detector.detect(df)
    # Convert to string labels for consistency
    return regimes.apply(lambda x: x.value if isinstance(x, RegimeType) else str(x))


def detect_rolling_returns(df: pd.DataFrame, window: int = 20, 
                          threshold: float = 0.001) -> pd.Series:
    """
    Rolling Returns Classification.
    
    Rules:
    - Bullish: mean daily return > threshold
    - Bearish: mean daily return < -threshold
    - Sideways: |mean return| <= threshold
    
    Args:
        window: Rolling window size in bars
        threshold: Return threshold for trend classification (per bar)
    """
    returns = df['close'].pct_change()
    roll_mean = returns.rolling(window=window, min_periods=window).mean()
    
    regime = pd.Series(index=df.index, dtype=str)
    regime[:] = np.nan
    
    # Classify
    regime[roll_mean > threshold] = 'bullish'
    regime[roll_mean < -threshold] = 'bearish'
    regime[(roll_mean >= -threshold) & (roll_mean <= threshold)] = 'sideways'
    
    return regime


def detect_adx_di_hysteresis(df: pd.DataFrame, 
                             adx_enter: int = 25, 
                             adx_exit: int = 20,
                             period: int = 14) -> pd.Series:
    """
    ADX + DI with Hysteresis.
    
    Rules:
    - Enter trend mode when ADX > adx_enter
    - Exit trend mode when ADX < adx_exit  
    - In trend mode: +DI > -DI = bullish, else bearish
    - In range mode: sideways
    """
    if not TA_AVAILABLE:
        return pd.Series(index=df.index, dtype=str)
    
    # Calculate ADX and DI
    adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=period)
    adx = adx_indicator.adx()
    plus_di = adx_indicator.adx_pos()
    minus_di = adx_indicator.adx_neg()
    
    regime = pd.Series(index=df.index, dtype=str)
    regime[:] = np.nan
    
    # Track trend mode with hysteresis
    in_trend_mode = False
    
    for i in range(len(df)):
        if pd.isna(adx.iloc[i]):
            continue
            
        current_adx = adx.iloc[i]
        
        # Hysteresis logic
        if not in_trend_mode and current_adx > adx_enter:
            in_trend_mode = True
        elif in_trend_mode and current_adx < adx_exit:
            in_trend_mode = False
        
        # Classify
        if in_trend_mode:
            if plus_di.iloc[i] > minus_di.iloc[i]:
                regime.iloc[i] = 'bullish'
            else:
                regime.iloc[i] = 'bearish'
        else:
            regime.iloc[i] = 'sideways'
    
    return regime


def detect_hmm_gaussian(df: pd.DataFrame, n_states: int = 3, 
                        n_iter: int = 100) -> pd.Series:
    """
    Hidden Markov Model with Gaussian emissions.
    
    Uses returns as observations and fits a 3-state HMM.
    States are mapped to bullish/bearish/sideways by mean returns.
    """
    if not HMM_AVAILABLE:
        return pd.Series(index=df.index, dtype=str)
    
    # Prepare returns
    returns = df['close'].pct_change().dropna()
    rets = returns.values.reshape(-1, 1)
    
    # Handle NaN/Inf
    valid_mask = np.isfinite(rets).flatten()
    if not valid_mask.all():
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.1, neginf=-0.1)
    
    # Fit HMM
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42
        )
        model.fit(rets)
        hidden_states = model.predict(rets)
        
        # Map states by mean returns
        state_means = []
        for state in range(n_states):
            mask = hidden_states == state
            if mask.sum() > 0:
                state_means.append((state, rets[mask].mean()))
            else:
                state_means.append((state, 0))
        
        # Sort by mean return
        state_means.sort(key=lambda x: x[1])
        
        # Map: lowest -> bearish, highest -> bullish, middle -> sideways
        state_map = {
            state_means[0][0]: 'bearish',
            state_means[-1][0]: 'bullish',
        }
        # Middle states are sideways
        for i in range(1, len(state_means) - 1):
            state_map[state_means[i][0]] = 'sideways'
        
        # If only 2 states, map both extremes
        if n_states == 2:
            state_map = {
                state_means[0][0]: 'bearish',
                state_means[1][0]: 'bullish',
            }
        
        # Create regime series (aligned with returns index)
        regime_values = [state_map[s] for s in hidden_states]
        regime = pd.Series(regime_values, index=returns.index)
        
        # Reindex to full dataframe
        regime = regime.reindex(df.index)
        
        return regime
        
    except Exception as e:
        print(f"   HMM fitting failed: {e}")
        return pd.Series(index=df.index, dtype=str)


def detect_markov_switching(df: pd.DataFrame, k_regimes: int = 3) -> pd.Series:
    """
    Markov Switching Autoregression Model.
    
    Uses statsmodels MarkovAutoregression to detect regime switches.
    """
    if not STATSMODELS_AVAILABLE:
        return pd.Series(index=df.index, dtype=str)
    
    # Prepare returns
    returns = df['close'].pct_change().dropna() * 100  # Scale for numerical stability
    
    # Drop any remaining NaN/Inf
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) < 200:
        print("   Insufficient data for Markov model")
        return pd.Series(index=df.index, dtype=str)
    
    try:
        # Fit Markov Switching model
        model = MarkovRegression(
            returns, 
            k_regimes=k_regimes, 
            switching_variance=True
        )
        result = model.fit(disp=False)
        
        # Get smoothed probabilities
        smoothed_probs = result.smoothed_marginal_probabilities
        
        # Get most likely state at each time
        states = smoothed_probs.idxmax(axis=1)
        
        # Map states by mean returns in each state
        state_returns = {}
        for state in range(k_regimes):
            mask = states == state
            if mask.sum() > 0:
                state_returns[state] = returns[mask].mean()
            else:
                state_returns[state] = 0
        
        # Sort by mean return
        sorted_states = sorted(state_returns.items(), key=lambda x: x[1])
        
        # Map to labels
        state_map = {}
        if k_regimes == 3:
            state_map[sorted_states[0][0]] = 'bearish'
            state_map[sorted_states[1][0]] = 'sideways'
            state_map[sorted_states[2][0]] = 'bullish'
        else:
            # For 2 states
            state_map[sorted_states[0][0]] = 'bearish'
            state_map[sorted_states[-1][0]] = 'bullish'
        
        # Create regime series
        regime_values = [state_map.get(s, 'sideways') for s in states]
        regime = pd.Series(regime_values, index=returns.index)
        
        # Reindex to full dataframe
        regime = regime.reindex(df.index)
        
        return regime
        
    except Exception as e:
        print(f"   Markov Switching failed: {e}")
        return pd.Series(index=df.index, dtype=str)


def detect_ruptures_changepoint(df: pd.DataFrame, 
                                min_segment: int = 100,
                                n_bkps: int = 10) -> pd.Series:
    """
    Change Point Detection with Ruptures.
    
    Detects structural breaks in returns, then classifies each segment.
    """
    if not RUPTURES_AVAILABLE:
        return pd.Series(index=df.index, dtype=str)
    
    # Prepare signal: use volatility or returns
    returns = df['close'].pct_change().dropna()
    signal = returns.values.reshape(-1, 1)
    
    # Handle NaN/Inf
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.1, neginf=-0.1)
    
    try:
        # Use BinSeg algorithm (faster than PELT for large data)
        algo = rpt.Binseg(model="l2", min_size=min_segment).fit(signal)
        breakpoints = algo.predict(n_bkps=n_bkps)  # Fixed number of breakpoints
        
        # Classify each segment
        regime = pd.Series(index=df.index, dtype=str)
        
        prev_bp = 0
        segment_labels = []
        
        for bp in breakpoints:
            # Get returns in this segment
            if bp > len(returns):
                bp = len(returns)
            
            segment_returns = returns.iloc[prev_bp:bp]
            
            if len(segment_returns) > 0:
                mean_ret = segment_returns.mean()
                
                # Classify by mean return
                if mean_ret > 0.0005:  # 0.05% per bar
                    label = 'bullish'
                elif mean_ret < -0.0005:
                    label = 'bearish'
                else:
                    label = 'sideways'
            else:
                label = 'sideways'
            
            segment_labels.append((prev_bp, bp, label))
            prev_bp = bp
        
        # Fill regime series
        # Align with returns index (which is offset by 1 from df.index)
        returns_idx = returns.index
        for start, end, label in segment_labels:
            if start < len(returns_idx) and end <= len(returns_idx):
                regime.loc[returns_idx[start:end]] = label
        
        return regime
        
    except Exception as e:
        print(f"   Ruptures failed: {e}")
        return pd.Series(index=df.index, dtype=str)


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("PHASE 6: ADVANCED REGIME DETECTION - EVALUATION FRAMEWORK")
    print("=" * 80)
    
    # Load BTC/USDT 4h data  
    data_dir = Path(__file__).parent.parent.parent / "user_data" / "data" / "binance"
    feather_file = data_dir / "BTC_USDT-4h.feather"
    
    if not feather_file.exists():
        print(f"❌ Data file not found: {feather_file}")
        return
    
    print(f"\n📊 Loading data from: {feather_file.name}")
    df = load_feather_data(feather_file)
    
    print(f"   Loaded {len(df):,} candles")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Calculate returns once
    returns = calculate_returns(df)
    
    # ========================================================================
    # Run all detection methods
    # ========================================================================
    
    methods = {}
    
    # 1. Baseline: SMA + ADX (current default)
    print("\n🔍 Testing: Baseline SMA(50/200) + ADX...")
    methods['baseline_sma_adx'] = detect_baseline_sma_adx(df)
    
    # 2. NEW: ADX + DI with Hysteresis (integrated into RegimeDetector)
    print("\n🔍 Testing: ADX + DI with Hysteresis (NEW)...")
    detector = RegimeDetector(method='adx_di_hysteresis')
    regimes = detector.detect(df)
    methods['adx_di_hysteresis_NEW'] = regimes.apply(lambda x: x.value if isinstance(x, RegimeType) else str(x))
    
    # 3. NEW: Rolling Returns (integrated into RegimeDetector)
    print("\n🔍 Testing: Rolling Returns (NEW)...")
    detector = RegimeDetector(method='rolling_returns')
    regimes = detector.detect(df)
    methods['rolling_returns_NEW'] = regimes.apply(lambda x: x.value if isinstance(x, RegimeType) else str(x))
    
    # 4. NEW: HMM with hysteresis (integrated into RegimeDetector)
    if HMM_AVAILABLE:
        print("\n🔍 Testing: HMM with Hysteresis (NEW)...")
        detector = RegimeDetector(method='hmm')
        regimes = detector.detect(df)
        methods['hmm_with_hysteresis_NEW'] = regimes.apply(lambda x: x.value if isinstance(x, RegimeType) else str(x))
    
    # 5. Original test implementations for comparison
    print("\n🔍 Testing: Rolling Returns (window=50, original test)...")
    methods['rolling_returns_50_test'] = detect_rolling_returns(df, window=50, threshold=0.0005)
    
    print("\n🔍 Testing: ADX + DI Hysteresis (original test)...")
    methods['adx_di_hysteresis_test'] = detect_adx_di_hysteresis(df)
    
    # 6. Ruptures Change Point (if available)
    if RUPTURES_AVAILABLE:
        print("\n🔍 Testing: Ruptures Change Point...")
        methods['ruptures_changepoint'] = detect_ruptures_changepoint(df)
    
    # ========================================================================
    # Evaluate all methods
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    all_metrics = []
    
    for name, regimes in methods.items():
        metrics = evaluate_method(regimes, returns, name)
        all_metrics.append(metrics)
        print_metrics(metrics)
    
    # ========================================================================
    # Summary comparison table
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<25} {'Bull%':>8} {'Bear%':>8} {'Side%':>8} {'Flip':>8} {'Dwell':>8}")
    print("-" * 80)
    
    for m in all_metrics:
        bull = m.distribution.get('bullish', m.distribution.get('RegimeType.BULLISH', 0)) * 100
        bear = m.distribution.get('bearish', m.distribution.get('RegimeType.BEARISH', 0)) * 100
        side = m.distribution.get('sideways', m.distribution.get('RegimeType.SIDEWAYS', 0)) * 100
        
        print(f"{m.method_name:<25} {bull:>7.1f}% {bear:>7.1f}% {side:>7.1f}% {m.flip_rate:>7.2f} {m.mean_dwell_time:>7.1f}")
    
    print("\n✅ Evaluation complete!")
    print("\n📝 Notes:")
    print("   - Target distribution: ~33% each regime")
    print("   - Lower flip rate = more stable")
    print("   - Higher dwell time = regimes persist longer")
    print("   - Conditional returns should differ meaningfully between regimes")


if __name__ == "__main__":
    main()
