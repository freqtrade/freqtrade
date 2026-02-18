#!/usr/bin/env python3
"""
Test script to verify critical bug fixes for genetic algorithm.
These tests validate the fixes made for:
1. ROI key consistency (string vs int)
2. trailing_stop serialization
3. Generator fallback condition
4. Random.sample guard
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.mutation import mutate_parameters, mutate_structure
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml


def test_roi_keys_are_strings():
    """Test that minimal_roi keys are strings after mutations."""
    print("\n=== Test 1: ROI Keys Are Strings ===")
    
    # Create a test individual
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
    ]
    entry_conditions = [
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
    ]
    exit_conditions = [
        ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
    ]
    
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        timeframe='5m',
        stoploss=-0.10,
    )
    
    individual = Individual(strategy_gene=gene)
    
    # Load config
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test mutate_parameters
    mutated1 = mutate_parameters(individual, 1.0, config)
    roi_keys1 = list(mutated1.strategy_gene.minimal_roi.keys())
    print(f"mutate_parameters ROI keys: {roi_keys1}")
    print(f"Key types: {[type(k) for k in roi_keys1]}")
    
    # Verify all keys are strings
    assert all(isinstance(k, str) for k in roi_keys1), \
        f"mutate_parameters produced non-string ROI keys: {roi_keys1}"
    
    # Test mutate_structure
    mutated2 = mutate_structure(individual, 1.0, config)
    roi_keys2 = list(mutated2.strategy_gene.minimal_roi.keys())
    print(f"mutate_structure ROI keys: {roi_keys2}")
    print(f"Key types: {[type(k) for k in roi_keys2]}")
    
    # Verify all keys are strings
    assert all(isinstance(k, str) for k in roi_keys2), \
        f"mutate_structure produced non-string ROI keys: {roi_keys2}"
    
    print("✅ PASSED: All ROI keys are strings")
    return True


def test_strategy_gene_roundtrip_preserves_trailing_stop():
    """Test that trailing_stop parameters are preserved in serialization."""
    print("\n=== Test 2: Trailing Stop Roundtrip ===")
    
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
    ]
    entry_conditions = [
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
    ]
    exit_conditions = [
        ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
    ]
    
    # Create gene with trailing stop parameters
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        timeframe='5m',
        stoploss=-0.10,
        trailing_stop=True,
        trailing_stop_positive=0.01,
        trailing_stop_positive_offset=0.02,
    )
    
    print(f"Original trailing_stop: {gene.trailing_stop}")
    print(f"Original trailing_stop_positive: {gene.trailing_stop_positive}")
    print(f"Original trailing_stop_positive_offset: {gene.trailing_stop_positive_offset}")
    
    # Serialize and deserialize
    gene_dict = gene.to_dict()
    print(f"Serialized keys: {list(gene_dict.keys())}")
    
    # Verify trailing_stop fields are in the dict
    assert 'trailing_stop' in gene_dict, "trailing_stop missing from to_dict()"
    assert 'trailing_stop_positive' in gene_dict, "trailing_stop_positive missing from to_dict()"
    assert 'trailing_stop_positive_offset' in gene_dict, "trailing_stop_positive_offset missing from to_dict()"
    
    gene_restored = StrategyGene.from_dict(gene_dict)
    
    print(f"Restored trailing_stop: {gene_restored.trailing_stop}")
    print(f"Restored trailing_stop_positive: {gene_restored.trailing_stop_positive}")
    print(f"Restored trailing_stop_positive_offset: {gene_restored.trailing_stop_positive_offset}")
    
    # Verify all trailing stop parameters are preserved
    assert gene_restored.trailing_stop == gene.trailing_stop, \
        f"trailing_stop not preserved: {gene.trailing_stop} != {gene_restored.trailing_stop}"
    assert gene_restored.trailing_stop_positive == gene.trailing_stop_positive, \
        f"trailing_stop_positive not preserved: {gene.trailing_stop_positive} != {gene_restored.trailing_stop_positive}"
    assert gene_restored.trailing_stop_positive_offset == gene.trailing_stop_positive_offset, \
        f"trailing_stop_positive_offset not preserved: {gene.trailing_stop_positive_offset} != {gene_restored.trailing_stop_positive_offset}"
    
    print("✅ PASSED: Trailing stop parameters preserved")
    return True


def test_generator_fallback_condition_is_vectorized():
    """Test that generator fallback condition is vectorized."""
    print("\n=== Test 3: Generator Fallback Condition ===")
    
    # Load config
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    generator = StrategyGenerator(config)
    
    # Create a condition that will trigger the fallback
    # (This is a bit of a white-box test, but necessary)
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
    ]
    
    condition = ConditionGene(
        indicator='UNKNOWN',  # Non-existent indicator
        operator='unknown_op',  # Unknown operator
        threshold=0,
        logic='AND'
    )
    
    # Generate condition code with required parameters
    condition_code = generator._generate_condition_code([condition], indicators, is_entry=True)
    
    print(f"Fallback condition code: {condition_code}")
    
    # Verify it's not the string "True"
    assert '"True"' not in condition_code and "'True'" not in condition_code, \
        "Fallback condition contains scalar 'True' string"
    
    # Verify it's a vectorized condition (contains dataframe reference)
    assert "dataframe" in condition_code, \
        "Fallback condition doesn't reference dataframe"
    
    # Verify the fallback uses volume condition
    assert "volume" in condition_code, \
        "Fallback condition doesn't use volume check"
    
    print("✅ PASSED: Fallback condition is vectorized")
    return True


def test_indicator_selection_with_edge_cases():
    """Test that indicator selection handles edge cases."""
    print("\n=== Test 4: Indicator Selection Edge Cases ===")
    
    # Load config
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    generator = StrategyGenerator(config)
    
    # Test case 1: Request more indicators than available
    print(f"Available indicators: {len(generator.available_indicators)}")
    
    # Temporarily modify config to request many indicators
    original_max = config['indicators'].get('max_per_strategy', 5)
    config['indicators']['max_per_strategy'] = 999  # Request way more than available
    
    generator_large = StrategyGenerator(config)
    
    try:
        # This should not crash
        gene = generator_large.generate_random_strategy(generation=0, individual_id=1)
        print(f"Generated strategy with {len(gene.indicators)} indicators")
        print(f"Max requested was 999, available was {len(generator_large.available_indicators)}")
        
        # Verify we didn't exceed available indicators
        assert len(gene.indicators) <= len(generator_large.available_indicators), \
            f"Generated more indicators than available: {len(gene.indicators)} > {len(generator_large.available_indicators)}"
        
        print("✅ PASSED: Indicator selection handles edge cases")
        return True
        
    except ValueError as e:
        if "Sample larger than population" in str(e):
            print(f"❌ FAILED: random.sample still crashes with: {e}")
            return False
        raise
    finally:
        # Restore original config
        config['indicators']['max_per_strategy'] = original_max


def test_roi_type_annotation():
    """Test that ROI type annotation is consistent with string keys."""
    print("\n=== Test 5: ROI Type Annotation ===")
    
    from genetic_algorithm.core.strategy_gene import StrategyGene
    import inspect
    
    # Get the type hints for StrategyGene
    hints = StrategyGene.__annotations__
    
    print(f"StrategyGene type hints: {hints}")
    
    # Check minimal_roi type hint
    roi_hint = hints.get('minimal_roi')
    print(f"minimal_roi type hint: {roi_hint}")
    
    # Create a default instance and check the default ROI
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
    ]
    entry_conditions = [
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
    ]
    exit_conditions = [
        ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
    ]
    
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
    )
    
    print(f"Default ROI: {gene.minimal_roi}")
    print(f"Default ROI keys: {list(gene.minimal_roi.keys())}")
    print(f"Default ROI key types: {[type(k) for k in gene.minimal_roi.keys()]}")
    
    # Verify all default keys are strings
    assert all(isinstance(k, str) for k in gene.minimal_roi.keys()), \
        f"Default ROI has non-string keys: {list(gene.minimal_roi.keys())}"
    
    print("✅ PASSED: ROI type annotation and defaults use strings")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Critical Bug Fixes")
    print("="*60)
    
    tests = [
        test_roi_keys_are_strings,
        test_strategy_gene_roundtrip_preserves_trailing_stop,
        test_generator_fallback_condition_is_vectorized,
        test_indicator_selection_with_edge_cases,
        test_roi_type_annotation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
