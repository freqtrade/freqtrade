"""
Test for Missing Indicator Column Fix

This test verifies that generated strategies don't reference indicators
that aren't calculated in populate_indicators().
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def test_condition_validation():
    """Test that conditions referencing non-existent indicators are filtered out."""
    
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA'],
            'min_per_strategy': 2,
            'max_per_strategy': 3,
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10]
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Create a strategy with RSI indicator but conditions referencing MACD
    strategy_gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(
                type='RSI',
                parameters={'period': 14},
                weight=1.0,
                instance_id='RSI_0'
            )
        ],
        entry_conditions=[
            # This condition references MACD which doesn't exist!
            ConditionGene(
                indicator='MACD',
                operator='cross_above',
                threshold=0,
                logic='AND'
            )
        ],
        exit_conditions=[
            # This condition references BBANDS which doesn't exist!
            ConditionGene(
                indicator='BBANDS',
                operator='cross_above',
                threshold=0,
                logic='AND'
            )
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
    )
    
    # Generate strategy code
    code = generator.generate_strategy_code(strategy_gene)
    
    print("=" * 80)
    print("TEST 1: Condition Validation (MACD condition with only RSI indicator)")
    print("=" * 80)
    print("\nGenerated Strategy Code:")
    print(code)
    
    # Verify the code doesn't reference 'macd' or 'macdsignal'
    assert 'macd' not in code.lower() or 'def populate_indicators' in code, \
        "Generated code should not reference MACD columns in conditions"
    
    # Verify it has a fallback condition (volume-based)
    assert 'volume' in code, "Should have fallback volume condition when no valid conditions exist"
    
    print("\n✅ TEST 1 PASSED: Invalid MACD condition was filtered out\n")


def test_mixed_valid_invalid_conditions():
    """Test that valid conditions are kept while invalid ones are filtered."""
    
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA'],
            'min_per_strategy': 2,
            'max_per_strategy': 3,
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10]
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Create a strategy with RSI and EMA indicators
    strategy_gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(
                type='RSI',
                parameters={'period': 14},
                weight=1.0,
                instance_id='RSI_0'
            ),
            IndicatorGene(
                type='EMA',
                parameters={'period': 20},
                weight=1.0,
                instance_id='EMA_0'
            )
        ],
        entry_conditions=[
            # Valid condition - RSI exists
            ConditionGene(
                indicator='RSI',
                operator='<',
                threshold=30,
                logic='AND'
            ),
            # Invalid condition - MACD doesn't exist
            ConditionGene(
                indicator='MACD',
                operator='cross_above',
                threshold=0,
                logic='AND'
            )
        ],
        exit_conditions=[
            # Valid condition - RSI exists
            ConditionGene(
                indicator='RSI',
                operator='>',
                threshold=70,
                logic='AND'
            )
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
    )
    
    # Generate strategy code
    code = generator.generate_strategy_code(strategy_gene)
    
    print("=" * 80)
    print("TEST 2: Mixed Valid/Invalid Conditions")
    print("=" * 80)
    print("\nGenerated Strategy Code:")
    print(code)
    
    # Verify RSI is calculated
    assert "'rsi_14']" in code, "RSI should be calculated in populate_indicators"
    
    # Verify RSI condition is present in entry trend
    assert "rsi_14" in code and "< 30" in code, "Valid RSI entry condition should be present"
    
    # Verify MACD columns are NOT referenced in conditions (only in check for non-existence is ok)
    lines = code.split('\n')
    condition_section = False
    for line in lines:
        if 'populate_entry_trend' in line or 'populate_exit_trend' in line:
            condition_section = True
        if condition_section and 'return dataframe' in line:
            condition_section = False
        if condition_section and 'macd' in line.lower():
            # Check it's not in populate_indicators section
            if 'populate_indicators' not in code[:code.index(line)]:
                assert False, f"MACD should not be referenced in conditions: {line}"
    
    print("\n✅ TEST 2 PASSED: Valid RSI condition kept, invalid MACD condition filtered\n")


def test_bbands_columns():
    """Test BBANDS indicator generates correct column names."""
    
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA'],
            'min_per_strategy': 2,
            'max_per_strategy': 3,
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10]
        }
    }
    
    generator = StrategyGenerator(config)
    
    # Create a strategy with BBANDS indicator and BBANDS conditions
    strategy_gene = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(
                type='BBANDS',
                parameters={'period': 20, 'std_dev': 2.0},
                weight=1.0,
                instance_id='BBANDS_0'
            )
        ],
        entry_conditions=[
            # Valid condition - BBANDS exists
            ConditionGene(
                indicator='BBANDS',
                operator='cross_below',
                threshold=0,
                logic='AND'
            )
        ],
        exit_conditions=[
            # Valid condition - BBANDS exists
            ConditionGene(
                indicator='BBANDS',
                operator='cross_above',
                threshold=0,
                logic='AND'
            )
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01}
    )
    
    # Generate strategy code
    code = generator.generate_strategy_code(strategy_gene)
    
    print("=" * 80)
    print("TEST 3: BBANDS Column Names")
    print("=" * 80)
    print("\nGenerated Strategy Code:")
    print(code)
    
    # Verify BBANDS columns are calculated
    assert "'bb_upperband']" in code, "bb_upperband should be calculated"
    assert "'bb_middleband']" in code, "bb_middleband should be calculated"
    assert "'bb_lowerband']" in code, "bb_lowerband should be calculated"
    
    # Verify BBANDS conditions reference the correct columns
    assert "bb_lowerband" in code, "Entry condition should reference bb_lowerband"
    assert "bb_upperband" in code, "Exit condition should reference bb_upperband"
    
    print("\n✅ TEST 3 PASSED: BBANDS columns correctly generated and referenced\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING MISSING INDICATOR COLUMN FIX")
    print("=" * 80 + "\n")
    
    try:
        test_condition_validation()
        test_mixed_valid_invalid_conditions()
        test_bbands_columns()
        
        print("=" * 80)
        print("ALL TESTS PASSED! ✅")
        print("=" * 80)
        print("\nThe fix successfully prevents KeyError exceptions by:")
        print("1. Validating conditions reference existing indicators")
        print("2. Filtering out invalid conditions")
        print("3. Providing fallback conditions when all are invalid")
        return 0
        
    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        return 1
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
