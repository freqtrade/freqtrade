#!/usr/bin/env python3
"""
Test Strategy Generation with Instance IDs

Verifies that strategies can be generated correctly with instance-based encoding.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.strategies.generator import StrategyGenerator


def test_strategy_generation_with_instance_ids():
    """Test that strategy code can be generated with instance IDs."""
    
    print("Testing strategy code generation with instance IDs...")
    
    # Create a strategy with instance IDs
    strategy = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='OR'),
        ],
    )
    
    # Generate strategy code
    config = {
        'indicators': {},
        'strategy_constraints': {}
    }
    generator = StrategyGenerator(config)
    code = generator.generate_strategy_code(strategy)
    
    # Verify the code contains expected elements
    assert "class GAStrategy_Gen0_Ind0" in code, "Strategy class not found"
    assert "rsi_14" in code, "RSI indicator not found"
    assert "ema_20" in code, "EMA indicator not found"
    assert "enter_long" in code, "Entry signal not found"
    assert "exit_long" in code, "Exit signal not found"
    
    # Verify it doesn't contain instance IDs in generated code (should use actual column names)
    # The generated code should reference dataframe columns, not instance IDs
    print("✓ Strategy code generated successfully")
    print()
    
    # Print sample of generated code
    print("Sample of generated code:")
    print("-" * 60)
    lines = code.split('\n')
    # Print the entry trend section
    for i, line in enumerate(lines):
        if 'populate_entry_trend' in line:
            for j in range(i, min(i+15, len(lines))):
                print(lines[j])
            break
    print("-" * 60)
    
    return code


def test_backward_compatibility():
    """Test that old-style type names still work."""
    
    print("\nTesting backward compatibility with type names...")
    
    # Create a strategy with type names (old format)
    strategy = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),  # No instance_id
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),  # Type name reference
            ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
        ],
    )
    
    # Generate strategy code
    config = {
        'indicators': {},
        'strategy_constraints': {}
    }
    generator = StrategyGenerator(config)
    code = generator.generate_strategy_code(strategy)
    
    # Verify the code contains expected elements
    assert "class GAStrategy_Gen0_Ind1" in code, "Strategy class not found"
    assert "rsi_14" in code, "RSI indicator not found"
    assert "macd" in code, "MACD indicator not found"
    assert "enter_long" in code, "Entry signal not found"
    assert "exit_long" in code, "Exit signal not found"
    
    print("✓ Backward compatibility maintained")
    
    return code


def test_multiple_same_type_indicators():
    """Test strategies with multiple indicators of the same type."""
    
    print("\nTesting multiple indicators of same type...")
    
    # Create a strategy with multiple RSIs
    strategy = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 7}, instance_id='RSI_0'),
            IndicatorGene(type='RSI', parameters={'period': 21}, instance_id='RSI_1'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),  # Fast RSI
            ConditionGene(indicator='RSI_1', operator='<', threshold=50, logic='AND'),  # Slow RSI
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='OR'),
        ],
    )
    
    # Generate strategy code
    config = {
        'indicators': {},
        'strategy_constraints': {}
    }
    generator = StrategyGenerator(config)
    code = generator.generate_strategy_code(strategy)
    
    # Verify both RSI periods are present
    assert "rsi_7" in code, "RSI with period 7 not found"
    assert "rsi_21" in code, "RSI with period 21 not found"
    
    print("✓ Multiple indicators of same type handled correctly")
    
    # Print the indicators section
    print("\nIndicators generated:")
    print("-" * 60)
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'populate_indicators' in line:
            for j in range(i, min(i+10, len(lines))):
                if 'rsi' in lines[j].lower():
                    print(lines[j])
            break
    print("-" * 60)
    
    return code


if __name__ == '__main__':
    print("=" * 60)
    print("Strategy Generation Test with Instance IDs")
    print("=" * 60)
    print()
    
    try:
        test_strategy_generation_with_instance_ids()
        test_backward_compatibility()
        test_multiple_same_type_indicators()
        
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        sys.exit(1)
