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
    
    return IndicatorGene(
        type=indicator_type,
        parameters=parameters,
        weight=random.uniform(0.3, 1.0)
    )
