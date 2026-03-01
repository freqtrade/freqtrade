"""
Indicator Factory

Shared utility for creating random indicators used by both
strategy generation and mutation operations.
"""

import random
from typing import Dict, Any

from genetic_algorithm.core.strategy_gene import IndicatorGene


def create_random_indicator(indicator_type: str, indicator_config: Dict[str, Any]) -> IndicatorGene:
    """
    Create a random indicator of the given type.
    
    This function consolidates the indicator generation logic that was
    previously duplicated in both generator.py and mutation.py.
    
    Args:
        indicator_type: Type of indicator to create (e.g., 'RSI', 'MACD')
        indicator_config: Configuration dict with parameter ranges for indicators
    
    Returns:
        IndicatorGene: A new indicator with random parameters
    """
    # Sanitize corrupted CDL type names (e.g., CDL_MORNINGSTAR_0_0 -> CDL_MORNINGSTAR)
    if indicator_type.startswith('CDL_'):
        from genetic_algorithm.core.strategy_gene import StrategyGene
        indicator_type = StrategyGene._strip_cdl_suffixes(indicator_type)
    
    ind_config = indicator_config.get(indicator_type, {})
    parameters = {}
    
    if indicator_type == 'RSI':
        period_range = ind_config.get('period', [7, 21])
        parameters['period'] = random.randint(*period_range)
    
    elif indicator_type == 'MACD':
        parameters['fast_period'] = random.randint(*ind_config.get('fast_period', [8, 21]))
        parameters['slow_period'] = random.randint(*ind_config.get('slow_period', [21, 50]))
        parameters['signal_period'] = random.randint(*ind_config.get('signal_period', [5, 14]))
    
    elif indicator_type == 'BBANDS':
        parameters['period'] = random.randint(*ind_config.get('period', [15, 30]))
        parameters['std_dev'] = random.uniform(*ind_config.get('std_dev', [1.5, 3.0]))
    
    elif indicator_type in ['EMA', 'SMA']:
        parameters['period'] = random.randint(*ind_config.get('period', [10, 50]))
    
    elif indicator_type == 'STOCH':
        parameters['k_period'] = random.randint(*ind_config.get('k_period', [5, 21]))
        parameters['d_period'] = random.randint(*ind_config.get('d_period', [3, 14]))
    
    elif indicator_type in ['ATR', 'ADX', 'CCI']:
        parameters['period'] = random.randint(*ind_config.get('period', [10, 20]))
    
    # New indicators for richer grammar
    elif indicator_type == 'MFI':
        # Money Flow Index (volume-weighted RSI)
        parameters['period'] = random.randint(*ind_config.get('period', [10, 20]))
    
    elif indicator_type == 'OBV':
        # On-Balance Volume (no parameters, uses volume)
        parameters = {}
    
    elif indicator_type == 'WILLR':
        # Williams %R
        parameters['period'] = random.randint(*ind_config.get('period', [10, 20]))
    
    elif indicator_type == 'ROC':
        # Rate of Change
        parameters['period'] = random.randint(*ind_config.get('period', [5, 20]))
    
    elif indicator_type == 'TEMA':
        # Triple Exponential Moving Average
        parameters['period'] = random.randint(*ind_config.get('period', [10, 30]))
    
    elif indicator_type == 'KAMA':
        # Kaufman Adaptive Moving Average
        parameters['period'] = random.randint(*ind_config.get('period', [10, 30]))
    
    elif indicator_type == 'SAR':
        # Parabolic SAR
        parameters['acceleration'] = random.uniform(*ind_config.get('acceleration', [0.01, 0.05]))
        parameters['maximum'] = random.uniform(*ind_config.get('maximum', [0.1, 0.3]))
    
    elif indicator_type == 'AROON':
        # Aroon indicator (trend strength)
        parameters['period'] = random.randint(*ind_config.get('period', [10, 25]))
    
    # === NEW INDICATORS ===
    
    elif indicator_type == 'SUPERTREND':
        # SuperTrend - trend following with dynamic stop
        parameters['period'] = random.randint(*ind_config.get('period', [7, 14]))
        parameters['multiplier'] = random.uniform(*ind_config.get('multiplier', [2.0, 4.0]))
    
    elif indicator_type == 'ICHIMOKU':
        # Ichimoku Cloud - comprehensive trend/support/resistance
        parameters['tenkan_period'] = random.randint(*ind_config.get('tenkan_period', [7, 12]))
        parameters['kijun_period'] = random.randint(*ind_config.get('kijun_period', [20, 30]))
        parameters['senkou_b_period'] = random.randint(*ind_config.get('senkou_b_period', [40, 60]))
    
    elif indicator_type == 'DONCHIAN':
        # Donchian Channels - breakout detection
        parameters['period'] = random.randint(*ind_config.get('period', [10, 30]))
    
    elif indicator_type == 'VWAP':
        # Volume Weighted Average Price - intraday mean reversion anchor
        parameters = {}  # No parameters, uses volume and typical price
    
    elif indicator_type == 'CMF':
        # Chaikin Money Flow - volume-based momentum
        parameters['period'] = random.randint(*ind_config.get('period', [10, 25]))
    
    elif indicator_type == 'VROC':
        # Volume Rate of Change
        parameters['period'] = random.randint(*ind_config.get('period', [5, 20]))
    
    elif indicator_type == 'PSAR':
        # Parabolic SAR (alias for SAR)
        parameters['acceleration'] = random.uniform(*ind_config.get('acceleration', [0.01, 0.05]))
        parameters['maximum'] = random.uniform(*ind_config.get('maximum', [0.1, 0.3]))
    
    # === CANDLESTICK PATTERNS ===
    # TALib CDL* functions detect candlestick patterns
    # All patterns have no parameters - they analyze OHLC data directly
    
    elif indicator_type == 'CDL_ENGULFING':
        # Bullish/Bearish Engulfing pattern
        parameters = {}
    
    elif indicator_type == 'CDL_HAMMER':
        # Hammer / Hanging Man pattern
        parameters = {}
    
    elif indicator_type == 'CDL_DOJI':
        # Doji pattern (indecision)
        parameters = {}
    
    elif indicator_type == 'CDL_MORNINGSTAR':
        # Morning Star (bullish reversal)
        parameters['penetration'] = random.uniform(*ind_config.get('penetration', [0.0, 0.3]))
    
    elif indicator_type == 'CDL_EVENINGSTAR':
        # Evening Star (bearish reversal)
        parameters['penetration'] = random.uniform(*ind_config.get('penetration', [0.0, 0.3]))
    
    elif indicator_type == 'CDL_SHOOTINGSTAR':
        # Shooting Star (bearish reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_HARAMI':
        # Harami pattern (reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_PIERCING':
        # Piercing Line (bullish reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_DARKCLOUD':
        # Dark Cloud Cover (bearish reversal)
        parameters = {}
    
    elif indicator_type == 'CDL_3WHITESOLDIERS':
        # Three White Soldiers (strong bullish)
        parameters = {}
    
    elif indicator_type == 'CDL_3BLACKCROWS':
        # Three Black Crows (strong bearish)
        parameters = {}
    
    # Validate MACD parameters (slow must be > fast)
    if indicator_type == 'MACD':
        if parameters.get('fast_period', 12) >= parameters.get('slow_period', 26):
            parameters['fast_period'] = max(8, parameters['slow_period'] - 5)
    
    return IndicatorGene(
        type=indicator_type,
        parameters=parameters,
        weight=random.uniform(0.3, 1.0)
    )
