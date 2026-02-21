"""
Additional tests for SMA, CCI, and RSI indicator column errors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def test_sma_with_period():
    """Test SMA indicator with specific period in condition."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'CCI'],
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
    
    # Create strategy with SMA that has period=20
    strategy_gene = StrategyGene(
        generation=1,
        individual_id=7,
        indicators=[
            IndicatorGene(
                type='SMA',
                parameters={'period': 20},
                weight=1.0,
                instance_id='SMA_0'
            )
        ],
        entry_conditions=[
            ConditionGene(
                indicator='SMA',
                operator='cross_above',
                threshold=0,
                logic='AND'
            )
        ],
        exit_conditions=[
            ConditionGene(
                indicator='SMA',
                operator='cross_below',
                threshold=0,
                logic='AND'
            )
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
    )
    
    code = generator.generate_strategy_code(strategy_gene)
    
    print("\n" + "="*80)
    print("TEST: SMA with period=20")
    print("="*80)
    print(code)
    
    # Verify SMA is calculated with the correct period
    assert "dataframe['sma_20']" in code, "SMA with period 20 should be calculated"
    
    # Verify condition references the correct column
    assert "sma_20" in code, "Condition should reference sma_20"
    
    print("✅ SMA test passed!\n")


def test_cci_with_period():
    """Test CCI indicator with specific period."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'CCI'],
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
    
    # Create strategy with CCI that has period=20
    strategy_gene = StrategyGene(
        generation=2,
        individual_id=11,
        indicators=[
            IndicatorGene(
                type='CCI',
                parameters={'period': 20},
                weight=1.0,
                instance_id='CCI_0'
            )
        ],
        entry_conditions=[
            ConditionGene(
                indicator='CCI',
                operator='<',
                threshold=-155,
                logic='AND'
            )
        ],
        exit_conditions=[ConditionGene(indicator="CCI", operator=">", threshold=155, logic="AND")],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
    )
    
    code = generator.generate_strategy_code(strategy_gene)
    
    print("\n" + "="*80)
    print("TEST: CCI with period=20")
    print("="*80)
    print(code)
    
    # Verify CCI is calculated with the correct period
    assert "dataframe['cci_20']" in code, "CCI with period 20 should be calculated"
    
    # Verify condition references the correct column
    assert "cci_20" in code and "< -155" in code, "Condition should reference cci_20 with threshold"
    
    print("✅ CCI test passed!\n")


def test_rsi_with_period():
    """Test RSI indicator with specific period."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'CCI'],
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
    
    # Create strategy with RSI that has period=14
    strategy_gene = StrategyGene(
        generation=2,
        individual_id=13,
        indicators=[
            IndicatorGene(
                type='RSI',
                parameters={'period': 14},
                weight=1.0,
                instance_id='RSI_0'
            )
        ],
        entry_conditions=[
            ConditionGene(
                indicator='RSI',
                operator='<',
                threshold=36,
                logic='AND'
            )
        ],
        exit_conditions=[],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
    )
    
    code = generator.generate_strategy_code(strategy_gene)
    
    print("\n" + "="*80)
    print("TEST: RSI with period=14")
    print("="*80)
    print(code)
    
    # Verify RSI is calculated with the correct period
    assert "dataframe['rsi_14']" in code, "RSI with period 14 should be calculated"
    
    # Verify condition references the correct column
    assert "rsi_14" in code and "< 36" in code, "Condition should reference rsi_14 with threshold"
    
    print("✅ RSI test passed!\n")


def test_all_indicators_comprehensive():
    """Test that all available indicators work correctly."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'CCI', 'ADX', 'ATR', 'STOCH'],
            'min_per_strategy': 2,
            'max_per_strategy': 9,
        },
        'strategy_constraints': {
            'timeframes': ['5m'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10]
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Test each indicator type individually
    indicator_tests = [
        ('RSI', {'period': 14}, 'rsi_14'),
        ('SMA', {'period': 20}, 'sma_20'),
        ('EMA', {'period': 20}, 'ema_20'),
        ('CCI', {'period': 20}, 'cci_20'),
        ('ADX', {'period': 14}, 'adx_14'),
        ('ATR', {'period': 14}, 'atr_14'),
    ]
    
    print("\n" + "="*80)
    print("TEST: All Indicators Comprehensive")
    print("="*80)
    
    for ind_type, params, expected_col in indicator_tests:
        strategy_gene = StrategyGene(
            generation=0,
            individual_id=0,
            indicators=[
                IndicatorGene(
                    type=ind_type,
                    parameters=params,
                    weight=1.0,
                    instance_id=f'{ind_type}_0'
                )
            ],
            entry_conditions=[
                ConditionGene(
                    indicator=ind_type,
                    operator='>' if ind_type in ['ADX', 'ATR'] else '<',
                    threshold=50,
                    logic='AND'
                )
            ],
            exit_conditions=[
                ConditionGene(
                    indicator=ind_type,
                    operator='<' if ind_type in ['ADX', 'ATR'] else '>',
                    threshold=30,
                    logic='AND'
                )
            ],
            timeframe='5m',
            stoploss=-0.10,
            minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
        )
        
        code = generator.generate_strategy_code(strategy_gene)
        
        # Check that the indicator is calculated
        assert f"dataframe['{expected_col}']" in code or expected_col in code, \
            f"{ind_type} indicator should create column {expected_col}"
        
        print(f"  ✅ {ind_type} -> {expected_col}")
    
    print("\n✅ All indicators test passed!\n")


if __name__ == '__main__':
    try:
        test_sma_with_period()
        test_cci_with_period()
        test_rsi_with_period()
        test_all_indicators_comprehensive()
        
        print("="*80)
        print("ALL ADDITIONAL TESTS PASSED! ✅")
        print("="*80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
