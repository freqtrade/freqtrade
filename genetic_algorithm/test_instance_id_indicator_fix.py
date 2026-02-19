#!/usr/bin/env python3
"""
Test script to verify that ensure_indicators_for_conditions correctly
handles instance_id-style references (e.g., 'RSI_0' -> 'RSI') and
that mutation cleanup handles instance_id references properly.

This tests the fix for the KeyError bug where generated strategies
referenced indicator columns (cci_20, sma_20, rsi_14, bb_upperband)
that weren't computed in populate_indicators().
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.mutation import mutate_indicators
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml


def test_ensure_indicators_extracts_base_type():
    """
    Test that ensure_indicators_for_conditions correctly extracts base type
    from instance_id-style references like 'RSI_0' -> 'RSI'.
    """
    print("\n=== Test: ensure_indicators extracts base type from instance_ids ===")

    # Create strategy with SMA indicator but conditions referencing instance_ids
    # of indicators that are NOT present (simulating post-crossover mismatch)
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='SMA', parameters={'period': 20}, weight=1.0, instance_id='SMA_0'),
        ],
        entry_conditions=[
            # This references RSI_0 (instance_id) but no RSI indicator exists
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            # This references CCI_0 (instance_id) but no CCI indicator exists
            ConditionGene(indicator='CCI_0', operator='>', threshold=100, logic='OR'),
        ],
    )

    # Call ensure_indicators_for_conditions
    gene.ensure_indicators_for_conditions({})

    # Check that RSI and CCI indicators were added with correct base types
    indicator_types = {ind.type for ind in gene.indicators}
    assert 'RSI' in indicator_types, f"RSI indicator not added. Types: {indicator_types}"
    assert 'CCI' in indicator_types, f"CCI indicator not added. Types: {indicator_types}"

    # Verify the newly added indicators have valid parameters
    for ind in gene.indicators:
        if ind.type == 'RSI':
            assert 'period' in ind.parameters, "RSI indicator missing 'period' parameter"
        elif ind.type == 'CCI':
            assert 'period' in ind.parameters, "CCI indicator missing 'period' parameter"

    print("  ✓ ensure_indicators correctly extracts base types from instance_ids")
    return True


def test_ensure_indicators_handles_bbands_instance_id():
    """Test that BBANDS_0 is correctly resolved to BBANDS."""
    print("\n=== Test: BBANDS instance_id resolution ===")

    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='BBANDS_0', operator='cross_above', threshold=0, logic='OR'),
        ],
    )

    gene.ensure_indicators_for_conditions({})

    indicator_types = {ind.type for ind in gene.indicators}
    assert 'BBANDS' in indicator_types, f"BBANDS not added. Types: {indicator_types}"

    # Verify BBANDS has correct parameters
    for ind in gene.indicators:
        if ind.type == 'BBANDS':
            assert 'period' in ind.parameters, "BBANDS missing 'period'"
            assert 'std_dev' in ind.parameters, "BBANDS missing 'std_dev'"

    print("  ✓ BBANDS instance_id correctly resolved")
    return True


def test_generated_code_consistency_with_instance_ids():
    """
    Test that generated strategy code has matching indicator calculations
    and condition references even when conditions use instance_ids.
    """
    print("\n=== Test: Generated code consistency with instance_ids ===")

    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    generator = StrategyGenerator(config)

    # Create a strategy where conditions reference instance_ids for
    # indicators that are NOT in the indicators list (simulating crossover mismatch)
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=0.8),
        ],
        entry_conditions=[
            # RSI_0 not present as indicator - should be auto-added
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='MACD', operator='cross_below', threshold=0, logic='OR'),
        ],
    )

    code = generator.generate_strategy_code(gene)

    # The generated code should have RSI indicator calculation
    assert 'ta.RSI(dataframe' in code, f"RSI calculation missing from generated code:\n{code}"
    # The generated code should have MACD indicator calculation
    assert 'ta.MACD(dataframe' in code, f"MACD calculation missing from generated code:\n{code}"
    # The generated code should reference rsi_ column in conditions
    assert "dataframe['rsi_" in code, f"RSI column reference missing from conditions in code:\n{code}"

    print("  ✓ Generated code has all necessary indicator calculations")
    return True


def test_crossover_with_instance_ids_produces_valid_code():
    """
    Test that crossover between strategies with instance_ids produces
    valid strategy code (no missing indicator columns).
    """
    print("\n=== Test: Crossover with instance_ids produces valid code ===")

    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    generator = StrategyGenerator(config)

    # Parent 1 with BBANDS + RSI (using instance_ids)
    parent1_gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}, weight=0.9, instance_id='BBANDS_0'),
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0, instance_id='RSI_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='BBANDS_0', operator='cross_below', threshold=0, logic='AND'),
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='BBANDS_0', operator='cross_above', threshold=0, logic='OR'),
        ],
    )

    # Parent 2 with SMA + CCI (using instance_ids)
    parent2_gene = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(type='SMA', parameters={'period': 20}, weight=0.8, instance_id='SMA_0'),
            IndicatorGene(type='CCI', parameters={'period': 20}, weight=0.7, instance_id='CCI_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='SMA_0', operator='cross_above', threshold=0, logic='AND'),
            ConditionGene(indicator='CCI_0', operator='<', threshold=-150, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='CCI_0', operator='>', threshold=150, logic='OR'),
        ],
    )

    parent1 = Individual(strategy_gene=parent1_gene)
    parent2 = Individual(strategy_gene=parent2_gene)

    # Run component crossover multiple times (it randomly swaps components)
    for trial in range(20):
        child1, child2 = crossover(
            parent1, parent2,
            generation=1, ind_id=100 + trial * 2,
            method='component', config=config
        )

        for label, child in [("child1", child1), ("child2", child2)]:
            code = generator.generate_strategy_code(child.strategy_gene)

            # For each condition in the child, verify the indicator is computed
            for cond in child.strategy_gene.entry_conditions + child.strategy_gene.exit_conditions:
                cond_ref = cond.indicator
                base_type = cond_ref.split('_')[0] if '_' in cond_ref else cond_ref

                if base_type == 'RSI':
                    assert 'ta.RSI(dataframe' in code, \
                        f"Trial {trial} {label}: RSI referenced in condition but not calculated"
                elif base_type == 'BBANDS':
                    assert 'ta.BBANDS(dataframe' in code, \
                        f"Trial {trial} {label}: BBANDS referenced in condition but not calculated"
                elif base_type == 'SMA':
                    assert 'ta.SMA(dataframe' in code, \
                        f"Trial {trial} {label}: SMA referenced in condition but not calculated"
                elif base_type == 'CCI':
                    assert 'ta.CCI(dataframe' in code, \
                        f"Trial {trial} {label}: CCI referenced in condition but not calculated"

    print("  ✓ All crossover children have consistent indicator code")
    return True


def test_mutation_remove_cleans_instance_id_conditions():
    """
    Test that removing an indicator during mutation also removes
    conditions referencing its instance_id.
    """
    print("\n=== Test: Mutation remove cleans instance_id conditions ===")

    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'ADX', 'CCI', 'STOCH'],
            'min_per_strategy': 1,
            'max_per_strategy': 5,
        },
        'strategy_constraints': {
            'timeframes': ['5m', '15m', '1h'],
        }
    }

    # Create strategy with RSI, MACD, CCI - conditions referencing instance_ids
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0, instance_id='RSI_0'),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=0.8, instance_id='MACD_0'),
            IndicatorGene(type='CCI', parameters={'period': 20}, weight=0.7, instance_id='CCI_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='CCI_0', operator='<', threshold=-150, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='MACD_0', operator='cross_below', threshold=0, logic='OR'),
        ],
    )

    individual = Individual(strategy_gene=gene)

    # Run mutation many times to trigger removes
    for trial in range(50):
        try:
            mutated = mutate_indicators(individual, mutation_rate=1.0, config=config)
        except ValueError:
            # Some mutations may fail validation
            continue

        # After mutation, check that all conditions reference present indicators
        present_types = {ind.type for ind in mutated.strategy_gene.indicators}
        present_ids = {ind.instance_id for ind in mutated.strategy_gene.indicators if ind.instance_id}

        for cond in mutated.strategy_gene.entry_conditions + mutated.strategy_gene.exit_conditions:
            base_type = cond.indicator.split('_')[0] if '_' in cond.indicator else cond.indicator
            assert base_type in present_types or cond.indicator in present_ids, \
                f"Trial {trial}: Condition references '{cond.indicator}' (base: {base_type}) " \
                f"but present types: {present_types}, present ids: {present_ids}"

    print("  ✓ Mutation correctly cleans up conditions for removed indicators")
    return True


def main():
    """Run all tests for the instance_id indicator fix."""
    print("=" * 70)
    print("Instance ID Indicator Fix Tests")
    print("=" * 70)

    tests = [
        test_ensure_indicators_extracts_base_type,
        test_ensure_indicators_handles_bbands_instance_id,
        test_generated_code_consistency_with_instance_ids,
        test_crossover_with_instance_ids_produces_valid_code,
        test_mutation_remove_cleans_instance_id_conditions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
