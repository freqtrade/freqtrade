"""
Debug test to understand why validation isn't working during GA execution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def test_validation_case_sensitivity():
    """Test if case sensitivity is causing issues."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'CCI', 'STOCH'],
            'min_per_strategy': 2,
            'max_per_strategy': 3,
        },
        'strategy_constraints': {
            'timeframes': ['5m'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10]
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Test 1: CCI indicator with CCI condition
    indicators = [
        IndicatorGene(type='CCI', parameters={'period': 20}, weight=1.0, instance_id='CCI_0'),
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0, instance_id='RSI_0')
    ]
    
    conditions = [
        ConditionGene(indicator='CCI', operator='<', threshold=-161, logic='AND'),
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND')
    ]
    
    print("\n" + "="*80)
    print("TEST: Validation Logic")
    print("="*80)
    
    print("\nIndicators in strategy:")
    for ind in indicators:
        print(f"  - type='{ind.type}', instance_id='{ind.instance_id}'")
    
    print("\nConditions to validate:")
    for cond in conditions:
        print(f"  - indicator='{cond.indicator}', operator='{cond.operator}', threshold={cond.threshold}")
        result = generator._condition_has_valid_indicator(cond, indicators)
        print(f"    -> Valid: {result}")
    
    # Test 2: MACD condition without MACD indicator
    print("\n" + "-"*80)
    macd_condition = ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND')
    print(f"\nMACD condition without MACD indicator:")
    print(f"  Condition: indicator='{macd_condition.indicator}'")
    result = generator._condition_has_valid_indicator(macd_condition, indicators)
    print(f"  -> Valid: {result} (should be False)")
    
    # Test 3: STOCH condition without STOCH indicator
    print("\n" + "-"*80)
    stoch_condition = ConditionGene(indicator='STOCH', operator='<', threshold=31, logic='AND')
    print(f"\nSTOCH condition without STOCH indicator:")
    print(f"  Condition: indicator='{stoch_condition.indicator}'")
    result = generator._condition_has_valid_indicator(stoch_condition, indicators)
    print(f"  -> Valid: {result} (should be False)")
    
    print("\n" + "="*80)
    print("Validation logic appears to be working correctly.")
    print("="*80)


def test_actual_generation():
    """Test actual strategy generation to see what's produced."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'CCI', 'STOCH'],
            'min_per_strategy': 2,
            'max_per_strategy': 3,
            'RSI': {'period': [7, 21]},
            'CCI': {'period': [14, 25]},
            'MACD': {'fast_period': [8, 16], 'slow_period': [20, 30], 'signal_period': [7, 12]},
            'STOCH': {'k_period': [5, 21], 'd_period': [3, 14]},
        },
        'strategy_constraints': {
            'timeframes': ['5m'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10]
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Generate multiple random strategies and check them
    print("\n" + "="*80)
    print("TEST: Generating Random Strategies")
    print("="*80)
    
    for i in range(5):
        strategy = generator.generate_random_strategy(generation=1, individual_id=i)
        
        print(f"\n--- Strategy {i} ---")
        print(f"Indicators ({len(strategy.indicators)}):")
        for ind in strategy.indicators:
            print(f"  - {ind.type} (id: {ind.instance_id})")
        
        print(f"Entry conditions ({len(strategy.entry_conditions)}):")
        for cond in strategy.entry_conditions:
            valid = generator._condition_has_valid_indicator(cond, strategy.indicators)
            status = "✅" if valid else "❌"
            print(f"  {status} {cond.indicator} {cond.operator} {cond.threshold}")
        
        print(f"Exit conditions ({len(strategy.exit_conditions)}):")
        for cond in strategy.exit_conditions:
            valid = generator._condition_has_valid_indicator(cond, strategy.indicators)
            status = "✅" if valid else "❌"
            print(f"  {status} {cond.indicator} {cond.operator} {cond.threshold}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    test_validation_case_sensitivity()
    test_actual_generation()
