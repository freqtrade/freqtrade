"""
Seed Strategy Archetypes

Provides known-good strategy archetypes to seed the initial GA population.
Instead of starting from 100% random strategies, seeding 10-20% of the population
with proven patterns gives the GA high-quality building blocks for crossover
and accelerates convergence.

Each archetype encodes a well-known trading strategy pattern as a StrategyGene.
"""

import random
import logging
from typing import List, Dict, Any

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

logger = logging.getLogger(__name__)


def create_rsi_mean_reversion(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    RSI overbought/oversold mean reversion strategy.
    
    Entry: RSI < 30 (oversold)
    Exit: RSI > 70 (overbought)
    Trend filter: Price above SMA (only buy in uptrend)
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
            IndicatorGene(type='SMA', parameters={'period': 50}, weight=0.8),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='SMA', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70, logic='AND'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.08,
        minimal_roi={"0": 0.06, "30": 0.03, "60": 0.015},
        max_open_trades=3,
        trailing_stop=True,
        trailing_stop_positive=0.015,
        trailing_stop_positive_offset=0.03,
    )


def create_macd_crossover_adx(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    MACD crossover with ADX trend strength filter.
    
    Entry: MACD crosses above signal AND ADX > 25 (strong trend)
    Exit: MACD crosses below signal
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=1.0),
            IndicatorGene(type='ADX', parameters={'period': 14}, weight=0.9),
        ],
        entry_conditions=[
            ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND'),
            ConditionGene(indicator='ADX', operator='>', threshold=25, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='MACD', operator='cross_below', threshold=0, logic='AND'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.07,
        minimal_roi={"0": 0.05, "30": 0.025, "60": 0.01},
        max_open_trades=3,
        trailing_stop=False,
    )


def create_bollinger_bounce_volume(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    Bollinger Band bounce with volume confirmation.
    
    Entry: Price crosses below lower BB AND CMF positive (buying pressure)
    Exit: Price crosses above upper BB
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}, weight=1.0),
            IndicatorGene(type='CMF', parameters={'period': 20}, weight=0.8),
        ],
        entry_conditions=[
            ConditionGene(indicator='BBANDS', operator='cross_below', threshold=0, logic='AND'),
            ConditionGene(indicator='CMF', operator='>', threshold=0.05, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='BBANDS', operator='cross_above', threshold=0, logic='AND'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.06,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
        trailing_stop=True,
        trailing_stop_positive=0.01,
        trailing_stop_positive_offset=0.025,
    )


def create_ema_crossover_rsi(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    EMA crossover with RSI filter.
    
    Entry: Price above EMA_20 AND RSI increasing from oversold
    Exit: Price below EMA_20 AND RSI > 65
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='EMA', parameters={'period': 20}, weight=1.0),
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=0.9),
        ],
        entry_conditions=[
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0, logic='AND'),
            ConditionGene(indicator='RSI', operator='increasing', threshold=0, logic='AND', lookback=3),
        ],
        exit_conditions=[
            ConditionGene(indicator='EMA', operator='cross_below', threshold=0, logic='AND'),
            ConditionGene(indicator='RSI', operator='>', threshold=65, logic='OR'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.08,
        minimal_roi={"0": 0.05, "30": 0.025, "60": 0.012},
        max_open_trades=4,
        trailing_stop=False,
    )


def create_supertrend_follow(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    SuperTrend trend-following strategy.
    
    Entry: SuperTrend turns bullish AND ADX > 20 (trend exists)
    Exit: SuperTrend turns bearish
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='SUPERTREND', parameters={'period': 10, 'multiplier': 3.0}, weight=1.0),
            IndicatorGene(type='ADX', parameters={'period': 14}, weight=0.8),
        ],
        entry_conditions=[
            ConditionGene(indicator='SUPERTREND', operator='cross_above', threshold=0, logic='AND'),
            ConditionGene(indicator='ADX', operator='>', threshold=20, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='SUPERTREND', operator='cross_below', threshold=0, logic='AND'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.10,
        minimal_roi={"0": 0.07, "30": 0.035, "60": 0.015},
        max_open_trades=3,
        trailing_stop=True,
        trailing_stop_positive=0.02,
        trailing_stop_positive_offset=0.04,
    )


def create_stoch_rsi_reversal(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    Stochastic + RSI reversal strategy.
    
    Entry: Stochastic K < 20 AND RSI < 35 (double oversold confirmation)
    Exit: Stochastic K > 80 OR RSI > 70
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='STOCH', parameters={'k_period': 14, 'd_period': 3}, weight=1.0),
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=0.9),
        ],
        entry_conditions=[
            ConditionGene(indicator='STOCH', operator='<', threshold=20, logic='AND'),
            ConditionGene(indicator='RSI', operator='<', threshold=35, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='STOCH', operator='>', threshold=80, logic='OR'),
            ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.06,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
        trailing_stop=False,
    )


def create_donchian_breakout(generation: int, individual_id: int, config: Dict[str, Any]) -> StrategyGene:
    """
    Donchian Channel breakout (turtle trading inspired).
    
    Entry: Price breaks above upper Donchian AND VROC positive (volume confirms)
    Exit: Price breaks below lower Donchian
    """
    constraints = config.get('strategy_constraints', {})
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[
            IndicatorGene(type='DONCHIAN', parameters={'period': 20}, weight=1.0),
            IndicatorGene(type='VROC', parameters={'period': 12}, weight=0.7),
        ],
        entry_conditions=[
            ConditionGene(indicator='DONCHIAN', operator='cross_above', threshold=0, logic='AND'),
            ConditionGene(indicator='VROC', operator='>', threshold=50, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='DONCHIAN', operator='cross_below', threshold=0, logic='AND'),
        ],
        timeframe=random.choice(constraints.get('timeframes', ['1h'])),
        stoploss=-0.09,
        minimal_roi={"0": 0.06, "30": 0.03, "60": 0.01},
        max_open_trades=3,
        trailing_stop=True,
        trailing_stop_positive=0.02,
        trailing_stop_positive_offset=0.035,
    )


# Registry of all seed strategy archetypes
SEED_ARCHETYPES = [
    ('RSI Mean Reversion + SMA Filter', create_rsi_mean_reversion),
    ('MACD Crossover + ADX Filter', create_macd_crossover_adx),
    ('Bollinger Bounce + Volume', create_bollinger_bounce_volume),
    ('EMA Crossover + RSI Momentum', create_ema_crossover_rsi),
    ('SuperTrend Follow + ADX', create_supertrend_follow),
    ('Stochastic + RSI Double Reversal', create_stoch_rsi_reversal),
    ('Donchian Breakout + Volume', create_donchian_breakout),
]


def create_seed_population(
    generation: int,
    count: int,
    config: Dict[str, Any],
    start_id: int = 0
) -> List[StrategyGene]:
    """
    Create seed strategies from known-good archetypes.
    
    Each archetype is slightly randomized (timeframe varies) to provide
    diverse starting points while maintaining the core trading logic.
    
    Args:
        generation: Generation number (usually 0)
        count: Number of seed strategies to create
        config: GA configuration dict
        start_id: Starting individual_id for seeded strategies
        
    Returns:
        List of StrategyGene objects seeded from archetypes
    """
    seeds = []
    
    for i in range(count):
        # Cycle through archetypes, repeating if count > len(archetypes)
        archetype_idx = i % len(SEED_ARCHETYPES)
        name, factory = SEED_ARCHETYPES[archetype_idx]
        
        strategy = factory(generation, start_id + i, config)
        strategy.assign_instance_ids()
        
        seeds.append(strategy)
        logger.info(f"Seeded strategy {start_id + i}: {name}")
    
    logger.info(f"Created {len(seeds)} seed strategies from {min(count, len(SEED_ARCHETYPES))} archetypes")
    return seeds
