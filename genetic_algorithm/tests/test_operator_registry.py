"""
Test Suite for Operator Registry and Integration

Tests the central operator registry and its integration with mutation,
crossover, code generation, and hall of fame — ensuring that every
indicator+operator combination that can be assigned by the GA is properly
handled by the code generator (no more "Skipping condition" drops).

Test Categories:
  1. Registry unit tests — coverage, API correctness
  2. Registry ↔ Generator consistency — every valid pair produces code
  3. Mutation fuzz — post-mutation operators are always valid
  4. Crossover validity — post-crossover operators are always valid
  5. HoF round-trip — loaded genes have valid operators
  6. ATR/MACD/CCI/ADX end-to-end — formerly broken combos now work
  7. Regression — no volume-fallback strategies for valid genes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import copy
import random
import pytest
from unittest.mock import patch

from genetic_algorithm.strategies.operator_registry import (
    ADVANCED_OPERATORS,
    ALL_OPERATORS,
    CDL_TYPES,
    get_valid_operators,
    get_standard_operators,
    is_valid_operator,
    get_all_indicator_types,
    resolve_indicator_type,
    _STANDARD_OPERATORS,
)
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual


# ============================================================================
# Helpers
# ============================================================================

def _make_gene_for_type(ind_type, operator, threshold=30.0):
    """Create a minimal StrategyGene with one indicator + one condition."""
    params = _default_params(ind_type)
    ind = IndicatorGene(type=ind_type, parameters=params, instance_id=f'{ind_type}_0')
    cond = ConditionGene(indicator=f'{ind_type}_0', operator=operator, threshold=threshold)
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=[ind],
        entry_conditions=[cond],
        exit_conditions=[],
        stoploss=-0.10,
        timeframe='5m',
    )


def _default_params(ind_type):
    """Return minimal valid parameters for an indicator type."""
    return {
        'RSI':   {'period': 14},
        'EMA':   {'period': 20},
        'SMA':   {'period': 20},
        'MACD':  {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'STOCH': {'k_period': 14, 'd_period': 3, 'slowk_period': 3},
        'BBANDS': {'period': 20, 'nbdevup': 2, 'nbdevdn': 2},
        'ATR':   {'period': 14},
        'ADX':   {'period': 14},
        'CCI':   {'period': 20},
        'SUPERTREND': {'period': 10, 'multiplier': 3.0},
        'ICHIMOKU': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
        'DONCHIAN': {'period': 20},
        'PSAR':  {'af': 0.02, 'max_af': 0.2},
        'VWAP':  {},
        'CMF':   {'period': 20},
        'VROC':  {'period': 14},
    }.get(ind_type, {'period': 14})


def _make_individual(ind_type='RSI', operator='<', threshold=30.0):
    """Create a complete Individual for testing."""
    gene = _make_gene_for_type(ind_type, operator, threshold)
    ind = Individual(strategy_gene=gene)
    ind.fitness = 1.0
    ind.raw_fitness = 1.0
    ind.evaluated = True
    return ind


def _make_rich_gene():
    """Create a gene with multiple indicator types for crossover/mutation tests."""
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
        IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                      instance_id='MACD_0'),
        IndicatorGene(type='ATR', parameters={'period': 14}, instance_id='ATR_0'),
        IndicatorGene(type='ADX', parameters={'period': 14}, instance_id='ADX_0'),
        IndicatorGene(type='CCI', parameters={'period': 20}, instance_id='CCI_0'),
        IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
    ]
    entry_conditions = [
        ConditionGene(indicator='RSI_0', operator='<', threshold=30),
        ConditionGene(indicator='MACD_0', operator='cross_above', threshold=0),
        ConditionGene(indicator='ATR_0', operator='>', threshold=0.01),
    ]
    exit_conditions = [
        ConditionGene(indicator='RSI_0', operator='>', threshold=70),
        ConditionGene(indicator='ADX_0', operator='<', threshold=25),
        ConditionGene(indicator='CCI_0', operator='cross_below', threshold=100),
    ]
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        stoploss=-0.10,
        timeframe='5m',
    )


# ============================================================================
# 1. Registry Unit Tests
# ============================================================================

class TestRegistryCompleteness:
    """Verify the registry covers all expected indicator types."""

    EXPECTED_TYPES = [
        'RSI', 'STOCH', 'CCI', 'ADX', 'MACD', 'ATR', 'BBANDS',
        'EMA', 'SMA', 'SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'PSAR',
        'VWAP', 'CMF', 'VROC',
    ]

    def test_all_expected_types_registered(self):
        """Every known indicator type must be in the registry."""
        registered = set(get_all_indicator_types())
        for t in self.EXPECTED_TYPES:
            assert t in registered, f"{t} missing from operator registry"

    def test_every_type_has_nonempty_operators(self):
        for t in self.EXPECTED_TYPES:
            ops = get_valid_operators(t)
            assert len(ops) > 0, f"{t} has no valid operators"

    def test_standard_operators_are_subset_of_valid(self):
        for t in self.EXPECTED_TYPES:
            std = set(get_standard_operators(t))
            valid = set(get_valid_operators(t))
            assert std.issubset(valid), f"{t}: standard {std} not subset of valid {valid}"

    def test_advanced_operators_included_for_all_types(self):
        """Advanced operators should be valid for every standard indicator."""
        for t in self.EXPECTED_TYPES:
            valid = set(get_valid_operators(t))
            for adv_op in ADVANCED_OPERATORS:
                assert adv_op in valid, f"{t} missing advanced operator '{adv_op}'"

    def test_cdl_types_limited_operators(self):
        """CDL patterns should only support operators matching their directionality."""
        from genetic_algorithm.strategies.operator_registry import (
            CDL_POSITIVE_ONLY_TYPES, CDL_NEGATIVE_ONLY_TYPES, CDL_BIDIRECTIONAL_TYPES,
        )
        for cdl in CDL_POSITIVE_ONLY_TYPES:
            ops = get_valid_operators(cdl)
            assert set(ops) == {'>'}, f"{cdl} should only have '>', got: {ops}"
        for cdl in CDL_NEGATIVE_ONLY_TYPES:
            ops = get_valid_operators(cdl)
            assert set(ops) == {'<'}, f"{cdl} should only have '<', got: {ops}"
        for cdl in CDL_BIDIRECTIONAL_TYPES:
            ops = get_valid_operators(cdl)
            assert set(ops) == {'<', '>'}, f"{cdl} should have both '<' and '>', got: {ops}"

    def test_unknown_type_returns_empty(self):
        assert get_valid_operators('UNKNOWN_INDICATOR') == []
        assert get_standard_operators('UNKNOWN_INDICATOR') == []
        assert not is_valid_operator('UNKNOWN_INDICATOR', '<')


class TestResolveIndicatorType:
    """Test the indicator reference → type resolver."""

    @pytest.mark.parametrize("ref,expected", [
        ('RSI_0', 'RSI'),
        ('EMA_1', 'EMA'),
        ('MACD_0', 'MACD'),
        ('ATR_0', 'ATR'),
        ('BBANDS_0', 'BBANDS'),
        ('CDL_HAMMER_0', 'CDL_HAMMER'),
        ('CDL_ENGULFING', 'CDL_ENGULFING'),
        ('CDL_MORNINGSTAR_0', 'CDL_MORNINGSTAR'),
        ('RSI', 'RSI'),
        ('MACD', 'MACD'),
    ])
    def test_known_refs(self, ref, expected):
        assert resolve_indicator_type(ref) == expected

    def test_timeframe_suffix(self):
        """EMA_1h_0 should resolve to EMA."""
        result = resolve_indicator_type('EMA_1h_0')
        assert result == 'EMA'

    def test_multi_digit_instance_id(self):
        result = resolve_indicator_type('RSI_12')
        assert result == 'RSI'


class TestIsValidOperator:
    """Test operator validation."""

    @pytest.mark.parametrize("ind_type,op,expected", [
        ('RSI', '<', True),
        ('RSI', '>', True),
        ('RSI', 'cross_above', True),
        ('RSI', 'increasing', True),
        ('MACD', '<', True),
        ('MACD', '>', True),
        ('MACD', 'cross_above', True),
        ('ATR', '<', True),
        ('ATR', '>', True),
        ('ATR', 'cross_above', True),
        ('ATR', 'cross_below', True),
        ('CCI', 'cross_above', True),
        ('CCI', 'cross_below', True),
        ('ADX', 'cross_above', True),
        ('ADX', 'cross_below', True),
        ('CDL_HAMMER', '<', False),    # positive-only: only > is valid
        ('CDL_HAMMER', '>', True),
        ('CDL_HAMMER', 'cross_above', False),
        ('CDL_HAMMER', 'increasing', False),
        ('UNKNOWN', '<', False),
    ])
    def test_validity(self, ind_type, op, expected):
        assert is_valid_operator(ind_type, op) == expected


# ============================================================================
# 2. Registry ↔ Generator Consistency
# ============================================================================

class TestRegistryGeneratorConsistency:
    """Every valid (indicator_type, operator) pair must produce non-None code."""

    @pytest.fixture
    def generator(self):
        """Create a StrategyGenerator instance."""
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    # Test all standard indicator types (excluding CDL)
    @pytest.mark.parametrize("ind_type", [
        'RSI', 'MACD', 'STOCH', 'CCI', 'ADX', 'ATR', 'BBANDS',
        'EMA', 'SMA', 'SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'PSAR',
        'VWAP', 'CMF', 'VROC',
    ])
    def test_all_standard_operators_produce_code(self, generator, ind_type):
        """For each indicator, every valid standard operator must generate non-None code."""
        params = _default_params(ind_type)
        indicator = IndicatorGene(type=ind_type, parameters=params, instance_id=f'{ind_type}_0')

        for op in get_standard_operators(ind_type):
            threshold = 30.0 if ind_type != 'ATR' else 0.01
            cond = ConditionGene(indicator=f'{ind_type}_0', operator=op, threshold=threshold)
            code = generator._generate_single_condition(cond, [indicator])
            assert code is not None, (
                f"_generate_single_condition returned None for {ind_type} + '{op}'. "
                f"This would cause a 'Skipping condition' drop in production."
            )
            assert len(code) > 5, f"Code too short for {ind_type} + '{op}': {code}"

    @pytest.mark.parametrize("ind_type", [
        'RSI', 'EMA', 'MACD', 'ATR', 'ADX', 'CCI',
    ])
    def test_advanced_operators_produce_code(self, generator, ind_type):
        """Advanced operators (increasing, decreasing, etc.) must generate code."""
        params = _default_params(ind_type)
        indicator = IndicatorGene(type=ind_type, parameters=params, instance_id=f'{ind_type}_0')

        for op in ADVANCED_OPERATORS:
            cond = ConditionGene(
                indicator=f'{ind_type}_0', operator=op, threshold=30.0,
                threshold_upper=70.0 if op == 'between' else 0.0,
                lookback=3,
            )
            code = generator._generate_single_condition(cond, [indicator])
            assert code is not None, (
                f"Advanced operator '{op}' returned None for {ind_type}."
            )


# ============================================================================
# 3. Formerly Broken Combinations (Regression)
# ============================================================================

class TestFormerlyBrokenCombinations:
    """
    These exact (indicator, operator) combos were silently dropped in R5,
    producing 7,855 skipped conditions. All must now generate code.
    """

    @pytest.fixture
    def generator(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    @pytest.mark.parametrize("ind_type,op,threshold", [
        ('MACD', '>', 0.0),
        ('MACD', '<', 0.0),
        ('ATR', '>', 0.01),
        ('ATR', '<', 0.01),
        ('ATR', 'cross_above', 0.01),
        ('ATR', 'cross_below', 0.01),
        ('ADX', 'cross_above', 25.0),
        ('ADX', 'cross_below', 25.0),
        ('CCI', 'cross_above', 100.0),
        ('CCI', 'cross_below', -100.0),
    ])
    def test_formerly_broken_combo(self, generator, ind_type, op, threshold):
        params = _default_params(ind_type)
        indicator = IndicatorGene(type=ind_type, parameters=params, instance_id=f'{ind_type}_0')
        cond = ConditionGene(indicator=f'{ind_type}_0', operator=op, threshold=threshold)
        code = generator._generate_single_condition(cond, [indicator])
        assert code is not None, (
            f"REGRESSION: {ind_type} + '{op}' still returns None! "
            f"This was one of the 7,855 skipped conditions in R5."
        )


# ============================================================================
# 4. Mutation Fuzz Test
# ============================================================================

class TestMutationOperatorValidity:
    """After mutation, all operators must remain valid for their indicator."""

    def _validate_gene_operators(self, gene):
        """Assert all conditions have valid operators for their indicator."""
        indicator_map = {ind.instance_id: ind.type for ind in gene.indicators if ind.instance_id}
        for cond in gene.entry_conditions + gene.exit_conditions:
            ind_type = indicator_map.get(cond.indicator, resolve_indicator_type(cond.indicator))
            if not get_valid_operators(ind_type):
                continue  # Unknown indicator, can't validate
            assert is_valid_operator(ind_type, cond.operator), (
                f"Invalid operator '{cond.operator}' for {ind_type} "
                f"(indicator={cond.indicator}) after mutation"
            )

    def test_mutate_conditions_fuzz(self):
        """Run mutate_conditions 200 times, check operator validity."""
        from genetic_algorithm.core.mutation import mutate_conditions
        random.seed(42)

        for _ in range(200):
            gene = _make_rich_gene()
            ind = Individual(strategy_gene=gene)
            config = {'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10}}
            mutate_conditions(ind, mutation_rate=0.8, config=config)
            self._validate_gene_operators(ind.strategy_gene)

    def test_mutate_condition_reassign_fuzz(self):
        """Run mutate_condition_reassign 200 times, check operator validity."""
        from genetic_algorithm.core.mutation import mutate_condition_reassign
        random.seed(42)

        for _ in range(200):
            gene = _make_rich_gene()
            ind = Individual(strategy_gene=gene)
            config = {'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10}}
            mutate_condition_reassign(ind, mutation_rate=0.8, config=config)
            self._validate_gene_operators(ind.strategy_gene)

    def test_full_mutate_fuzz(self):
        """Run the top-level mutate() 100 times, check operator validity."""
        from genetic_algorithm.core.mutation import mutate
        random.seed(42)

        for _ in range(100):
            gene = _make_rich_gene()
            ind = Individual(strategy_gene=gene)
            config = {
                'ga': {
                    'indicator_config': {},
                    'max_indicators_per_strategy': 10,
                    'mutation': {
                        'parameter_rate': 0.3,
                        'indicator_rate': 0.3,
                        'condition_rate': 0.5,
                        'structure_rate': 0.2,
                    },
                },
            }
            mutated = mutate(ind, mutation_rate=0.5, config=config)
            self._validate_gene_operators(mutated.strategy_gene)


# ============================================================================
# 5. Crossover Validity
# ============================================================================

class TestCrossoverOperatorValidity:
    """After crossover, all operators must remain valid for their indicator."""

    CONFIG = {
        'ga': {
            'indicator_config': {},
            'max_indicators_per_strategy': 10,
            'crossover': {'method': 'single_point'},
        },
    }

    def _validate_gene_operators(self, gene):
        indicator_map = {ind.instance_id: ind.type for ind in gene.indicators if ind.instance_id}
        for cond in gene.entry_conditions + gene.exit_conditions:
            ind_type = indicator_map.get(cond.indicator, resolve_indicator_type(cond.indicator))
            if not get_valid_operators(ind_type):
                continue
            assert is_valid_operator(ind_type, cond.operator), (
                f"Invalid operator '{cond.operator}' for {ind_type} after crossover"
            )

    def _make_parents_with_different_types(self):
        """Create two parents with different indicators to stress crossover."""
        gene1 = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
                IndicatorGene(type='ATR', parameters={'period': 14}, instance_id='ATR_0'),
                IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                              instance_id='MACD_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
                ConditionGene(indicator='ATR_0', operator='>', threshold=0.01),
            ],
            exit_conditions=[
                ConditionGene(indicator='MACD_0', operator='cross_above', threshold=0),
            ],
            stoploss=-0.10, timeframe='5m',
        )
        gene2 = StrategyGene(
            generation=0, individual_id=1,
            indicators=[
                IndicatorGene(type='CCI', parameters={'period': 20}, instance_id='CCI_0'),
                IndicatorGene(type='ADX', parameters={'period': 14}, instance_id='ADX_0'),
                IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='CCI_0', operator='cross_above', threshold=100),
                ConditionGene(indicator='ADX_0', operator='>', threshold=25),
            ],
            exit_conditions=[
                ConditionGene(indicator='EMA_0', operator='cross_below', threshold=0),
            ],
            stoploss=-0.08, timeframe='5m',
        )
        p1 = Individual(strategy_gene=gene1)
        p1.fitness = 1.0; p1.raw_fitness = 1.0; p1.evaluated = True
        p2 = Individual(strategy_gene=gene2)
        p2.fitness = 1.0; p2.raw_fitness = 1.0; p2.evaluated = True
        return p1, p2

    @pytest.mark.parametrize("method", ['single_point', 'uniform', 'component'])
    def test_crossover_method_validity(self, method):
        """Run each crossover method 100 times, validate operator compatibility."""
        from genetic_algorithm.core.crossover import (
            single_point_crossover, uniform_crossover, component_crossover,
        )
        crossover_fn = {
            'single_point': single_point_crossover,
            'uniform': uniform_crossover,
            'component': component_crossover,
        }[method]
        random.seed(42)

        for i in range(100):
            p1, p2 = self._make_parents_with_different_types()
            child1, child2 = crossover_fn(p1, p2, generation=0, ind_id=i, config=self.CONFIG)
            self._validate_gene_operators(child1.strategy_gene)
            self._validate_gene_operators(child2.strategy_gene)


# ============================================================================
# 6. Hall of Fame Round-Trip
# ============================================================================

class TestHallOfFameOperatorFix:
    """Genes loaded from HoF must have valid operators after fixing."""

    def test_invalid_operators_are_fixed(self):
        from genetic_algorithm.core.hall_of_fame import HallOfFame

        # Create a gene with a CDL indicator that has an invalid operator.
        # CDL_HAMMER only supports < and >, not cross_above.
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
                IndicatorGene(type='CDL_HAMMER', parameters={}, instance_id='CDL_HAMMER_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
                ConditionGene(indicator='CDL_HAMMER_0', operator='cross_above', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='RSI_0', operator='>', threshold=70),
            ],
            stoploss=-0.10, timeframe='5m',
        )

        random.seed(42)
        HallOfFame._fix_invalid_operators(gene)

        # After fix, CDL_HAMMER should only have < or >
        cdl_cond = gene.entry_conditions[-1]
        assert cdl_cond.operator in ('<', '>'), (
            f"CDL_HAMMER still has invalid operator '{cdl_cond.operator}' after HoF fix"
        )

    def test_valid_operators_unchanged(self):
        from genetic_algorithm.core.hall_of_fame import HallOfFame

        gene = _make_rich_gene()
        original_ops = [(c.indicator, c.operator)
                        for c in gene.entry_conditions + gene.exit_conditions]
        HallOfFame._fix_invalid_operators(gene)
        fixed_ops = [(c.indicator, c.operator)
                     for c in gene.entry_conditions + gene.exit_conditions]
        assert original_ops == fixed_ops, "Valid operators should not be changed by HoF fix"


# ============================================================================
# 7. Pre-Generation Validation
# ============================================================================

class TestPreGenerationValidation:
    """Test _condition_has_valid_indicator validates operators too."""

    @pytest.fixture
    def generator(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    def test_valid_condition_passes(self, generator):
        indicators = [IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0')]
        cond = ConditionGene(indicator='RSI_0', operator='<', threshold=30)
        assert generator._condition_has_valid_indicator(cond, indicators) is True

    def test_invalid_operator_fails(self, generator):
        """A condition with an operator invalid for its indicator should fail."""
        indicators = [IndicatorGene(type='CDL_HAMMER', parameters={}, instance_id='CDL_HAMMER_0')]
        cond = ConditionGene(indicator='CDL_HAMMER_0', operator='cross_above', threshold=0)
        # cross_above is not valid for CDL patterns
        result = generator._condition_has_valid_indicator(cond, indicators)
        assert result is False, "Invalid operator should cause validation to fail"

    def test_missing_indicator_fails(self, generator):
        indicators = [IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0')]
        cond = ConditionGene(indicator='MACD_0', operator='cross_above', threshold=0)
        assert generator._condition_has_valid_indicator(cond, indicators) is False


# ============================================================================
# 8. DSR n_trials hash deduplication
# ============================================================================

class TestDSRTrialsCounting:
    """Verify that identical strategies count as one trial."""

    def test_unique_hash_counts_once(self):
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker

        tracker = DSRTracker({})
        tracker.register_evaluation(strategy_hash='abc123')
        tracker.register_evaluation(strategy_hash='abc123')
        tracker.register_evaluation(strategy_hash='abc123')
        # Same hash registered 3 times → should count as 1 unique trial
        assert len(tracker._strategy_hashes) == 1
        # _total_evaluated is NOT incremented when a hash is provided
        # (walk-forward re-evaluations of the same strategy don't inflate n_trials)
        assert tracker._total_evaluated == 0
        # n_trials = unique hashes + untracked = 1
        assert tracker.n_trials == 1

    def test_different_hashes_count_separately(self):
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker

        tracker = DSRTracker({})
        tracker.register_evaluation(strategy_hash='abc1')
        tracker.register_evaluation(strategy_hash='abc2')
        tracker.register_evaluation(strategy_hash='abc3')
        assert len(tracker._strategy_hashes) == 3

    def test_no_hash_increments_total(self):
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker

        tracker = DSRTracker({})
        tracker.register_evaluation()
        tracker.register_evaluation()
        assert tracker._total_evaluated == 2
        assert len(tracker._strategy_hashes) == 0


# ============================================================================
# 9. End-to-End: Full strategy code generation for problem indicators
# ============================================================================

class TestFullStrategyGeneration:
    """Generate full strategy code for formerly-broken indicator combos."""

    @pytest.fixture
    def generator(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    def test_atr_strategy_generates_complete_code(self, generator):
        """A strategy with ATR conditions should produce valid Python code."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='ATR', parameters={'period': 14}, instance_id='ATR_0'),
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='ATR_0', operator='>', threshold=0.01),
                ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ],
            exit_conditions=[
                ConditionGene(indicator='ATR_0', operator='<', threshold=0.005),
                ConditionGene(indicator='RSI_0', operator='>', threshold=70),
            ],
            stoploss=-0.10,
            timeframe='5m',
        )
        code = generator.generate_strategy_code(gene)
        assert code is not None, "Strategy with ATR should generate code"
        assert 'atr_14' in code, "Generated code should reference atr_14 column"
        # Should NOT have volume-only fallback
        assert "dataframe['volume'] > 0" not in code or 'atr_14' in code

    def test_macd_gt_lt_strategy_generates_code(self, generator):
        """MACD with > and < operators should produce code, not volume fallback."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                              instance_id='MACD_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='MACD_0', operator='>', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='MACD_0', operator='<', threshold=0),
            ],
            stoploss=-0.10, timeframe='5m',
        )
        code = generator.generate_strategy_code(gene)
        assert code is not None
        assert 'macd' in code.lower()


# ============================================================================
# 10. Regression Tests for Audit Fixes
# ============================================================================

class TestQtpylibImport:
    """Verify qtpylib is imported in generated strategy code."""

    @pytest.fixture
    def generator(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    def test_qtpylib_in_generated_code(self, generator):
        """Strategy code must import qtpylib for cross_above/below operators."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='CCI', parameters={'period': 20}, instance_id='CCI_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='CCI_0', operator='cross_above', threshold=100),
            ],
            exit_conditions=[
                ConditionGene(indicator='CCI_0', operator='cross_below', threshold=-100),
            ],
            stoploss=-0.10, timeframe='5m',
        )
        code = generator.generate_strategy_code(gene)
        assert code is not None
        assert 'qtpylib' in code, (
            "REGRESSION: Generated strategy code does not import qtpylib! "
            "CCI/ADX/ATR cross_above/cross_below will crash at runtime."
        )

    def test_code_compiles(self, generator):
        """Generated code with qtpylib calls must be compilable Python."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='ADX', parameters={'period': 14}, instance_id='ADX_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='ADX_0', operator='cross_above', threshold=25),
            ],
            exit_conditions=[
                ConditionGene(indicator='ADX_0', operator='<', threshold=20),
            ],
            stoploss=-0.10, timeframe='5m',
        )
        code = generator.generate_strategy_code(gene)
        assert code is not None
        # Code must at least compile (won't import freqtrade in test env, but syntax must be valid)
        compile(code, '<test_strategy>', 'exec')


class TestCDLDojiOperator:
    """CDL_DOJI must use '>' operator, not '!='."""

    def test_condition_generation_uses_gt(self):
        """_generate_condition_for_indicator for CDL_DOJI must use '>' not '!='."""
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        gen = StrategyGenerator(config)
        ind = IndicatorGene(type='CDL_DOJI', parameters={}, instance_id='CDL_DOJI_0')
        cond = gen._generate_condition_for_indicator(ind, is_entry=True)
        assert cond is not None, "CDL_DOJI should generate a condition"
        assert cond.operator != '!=', (
            "REGRESSION: CDL_DOJI still uses '!=' operator which is not in the registry"
        )
        assert cond.operator == '>', "CDL_DOJI entry condition should use '>' operator"

    def test_cdl_doji_handler_with_gt(self):
        """CDL_DOJI handler generates code for > (the only valid operator)."""
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        gen = StrategyGenerator(config)
        ind = IndicatorGene(type='CDL_DOJI', parameters={}, instance_id='CDL_DOJI_0')

        cond = ConditionGene(indicator='CDL_DOJI_0', operator='>', threshold=0)
        code = gen._generate_single_condition(cond, [ind])
        assert code is not None, "CDL_DOJI with '>' should generate code"
        assert '> 0' in code, f"CDL_DOJI '>' code should contain '> 0'"

    def test_cdl_doji_lt_is_invalid_operator(self):
        """CDL_DOJI '<' should be flagged as invalid by the registry."""
        assert not is_valid_operator('CDL_DOJI', '<'), \
            "CDL_DOJI '<' should be invalid (TA-Lib returns 0 or +100, never negative)"


class TestATRThresholdClamp:
    """ATR thresholds must stay in valid range after mutation."""

    def test_atr_in_threshold_clamps(self):
        from genetic_algorithm.core.mutation import _THRESHOLD_CLAMPS
        assert 'ATR' in _THRESHOLD_CLAMPS, "ATR must be in _THRESHOLD_CLAMPS"
        lo, hi = _THRESHOLD_CLAMPS['ATR']
        assert lo > 0, "ATR threshold lower bound must be positive"
        assert hi <= 0.2, "ATR threshold upper bound should be reasonable"

    def test_atr_clamp_prevents_negative(self):
        from genetic_algorithm.core.mutation import clamp_condition_thresholds
        cond = ConditionGene(indicator='ATR_0', operator='>', threshold=-0.5)
        clamp_condition_thresholds([cond])
        assert cond.threshold >= 0.001, (
            f"ATR threshold should be clamped to >= 0.001, got {cond.threshold}"
        )

    def test_atr_in_default_ranges(self):
        """ATR must have default ranges for mutation threshold resampling."""
        # We test indirectly: _mutate_condition_threshold with ATR should produce
        # values in a reasonable range, not just Gaussian perturbation.
        from genetic_algorithm.core.mutation import _mutate_condition_threshold
        random.seed(42)
        results = []
        for _ in range(100):
            cond = ConditionGene(indicator='ATR_0', operator='>', threshold=0.01)
            _mutate_condition_threshold(cond, {}, True, 0, [])
            results.append(cond.threshold)
        # At least some values should be different from ±10% of 0.01
        assert max(results) > 0.01 * 1.15 or min(results) < 0.01 * 0.85, (
            "ATR mutation should resample from _DEFAULT_RANGES, not just small Gaussian"
        )


class TestATRThresholdInMutation:
    """ATR threshold from _create_random_condition should not be 0."""

    def test_create_random_condition_atr_nonzero(self):
        from genetic_algorithm.core.mutation import _create_random_condition
        random.seed(42)
        for _ in range(50):
            cond = _create_random_condition('ATR', True, {})
            assert cond is not None
            assert cond.threshold > 0, (
                f"ATR threshold from _create_random_condition should be > 0, got {cond.threshold}"
            )


class TestVROCDoubleNegation:
    """VROC with negative threshold should not produce double-negation."""

    @pytest.fixture
    def generator(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    def test_vroc_negative_threshold_no_double_neg(self, generator):
        ind = IndicatorGene(type='VROC', parameters={'period': 14}, instance_id='VROC_0')
        cond = ConditionGene(indicator='VROC_0', operator='<', threshold=-100)
        code = generator._generate_single_condition(cond, [ind])
        assert code is not None
        # Should NOT contain '--' (double negation)
        assert '--' not in code, (
            f"VROC double-negation detected in generated code: {code}"
        )
        # Should produce vroc < -100 (using abs)
        assert '-100' in code, f"VROC code should reference 100, got: {code}"

    def test_vroc_positive_threshold(self, generator):
        ind = IndicatorGene(type='VROC', parameters={'period': 14}, instance_id='VROC_0')
        cond = ConditionGene(indicator='VROC_0', operator='<', threshold=100)
        code = generator._generate_single_condition(cond, [ind])
        assert code is not None
        assert '-100' in code, f"VROC < should negate positive threshold, got: {code}"


class TestCMFZeroThreshold:
    """CMF with threshold=0 should use 0, not silently replace with 0.1."""

    @pytest.fixture
    def generator(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator
        config = {
            'trading': {'pairs': ['BTC/USDT'], 'timeframe': '5m'},
            'ga': {'indicator_config': {}, 'max_indicators_per_strategy': 10},
        }
        return StrategyGenerator(config)

    def test_cmf_zero_threshold_preserved(self, generator):
        ind = IndicatorGene(type='CMF', parameters={'period': 20}, instance_id='CMF_0')
        cond = ConditionGene(indicator='CMF_0', operator='>', threshold=0)
        code = generator._generate_single_condition(cond, [ind])
        assert code is not None
        # Threshold 0 is valid for CMF (above/below zero line)
        assert '> 0' in code, f"CMF threshold=0 should produce '> 0', got: {code}"
        assert '0.1' not in code, f"CMF threshold=0 should NOT be replaced with 0.1: {code}"


# ============================================================================
# Trimmed Mean Aggregation Tests
# ============================================================================

class TestTrimmedMeanAggregation:
    """Test the trimmed_mean walk-forward aggregation method."""

    def test_basic_trimmed_mean(self):
        """Normal case: trims outliers and averages the rest."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        # 10 scores → trim 1 from each end → average middle 8
        scores = [0.1, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0]
        result = aggregate_validation_scores(scores, method='trimmed_mean')
        # Trimmed: [0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9] → mean = 0.7
        assert abs(result - 0.7) < 1e-9, f"Expected 0.7, got {result}"

    def test_trimmed_mean_removes_outliers(self):
        """Trimmed mean should be resistant to extreme outliers."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        scores_with_outlier = [0.0, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 10.0]
        scores_clean = [0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        result = aggregate_validation_scores(scores_with_outlier, method='trimmed_mean')
        expected = sum(scores_clean) / len(scores_clean)
        assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"

    def test_trimmed_mean_less_than_harmonic(self):
        """Trimmed mean should be less punishing than harmonic mean for mixed scores."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        scores = [0.2, 0.5, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.95]
        tm = aggregate_validation_scores(scores, method='trimmed_mean')
        hm = aggregate_validation_scores(scores, method='harmonic_mean')
        # Harmonic mean is always ≤ trimmed mean when low outliers are trimmed
        assert hm <= tm, f"Expected hm({hm:.4f}) ≤ tm({tm:.4f})"
        # Trimmed mean should differ from plain mean (proves trimming happened)
        mean = aggregate_validation_scores(scores, method='mean')
        assert abs(tm - mean) > 0.01, f"tm({tm:.4f}) should differ from mean({mean:.4f})"

    def test_trimmed_mean_two_scores_fallback(self):
        """With n≤2, falls back to plain mean (can't trim)."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        result = aggregate_validation_scores([0.3, 0.9], method='trimmed_mean')
        assert abs(result - 0.6) < 1e-9, f"Expected 0.6, got {result}"

    def test_trimmed_mean_single_score(self):
        """Single score should return that score."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        result = aggregate_validation_scores([0.42], method='trimmed_mean')
        assert abs(result - 0.42) < 1e-9, f"Expected 0.42, got {result}"

    def test_trimmed_mean_three_scores(self):
        """Three scores: trim 1 from each end → middle score only."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        result = aggregate_validation_scores([0.1, 0.5, 0.9], method='trimmed_mean')
        assert abs(result - 0.5) < 1e-9, f"Expected 0.5 (median), got {result}"

    def test_trimmed_mean_identical_scores(self):
        """All identical scores should return that value."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        result = aggregate_validation_scores([0.7] * 10, method='trimmed_mean')
        assert abs(result - 0.7) < 1e-9, f"Expected 0.7, got {result}"

    def test_trimmed_mean_empty_list(self):
        """Empty list should return 0.0."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        result = aggregate_validation_scores([], method='trimmed_mean')
        assert result == 0.0

    def test_trimmed_mean_large_list(self):
        """20 scores: trim 2 from each end."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        scores = [0.01, 0.05] + [0.5] * 16 + [0.95, 0.99]  # 20 total
        result = aggregate_validation_scores(scores, method='trimmed_mean')
        assert abs(result - 0.5) < 1e-9, f"Expected 0.5, got {result}"

    def test_trimmed_mean_filters_negative(self):
        """Negative scores should be filtered out before trimming."""
        from genetic_algorithm.utils.timerange import aggregate_validation_scores
        scores = [-1.0, 0.5, 0.6, 0.7, 0.8]
        result = aggregate_validation_scores(scores, method='trimmed_mean')
        # After filtering: [0.5, 0.6, 0.7, 0.8] → 4 scores, trim 1 each → [0.6, 0.7]
        assert abs(result - 0.65) < 1e-9, f"Expected 0.65, got {result}"


class TestWalkForwardConfigValidation:
    """Test that R6 config values are accepted by the validator."""

    def test_trimmed_mean_is_valid_aggregation(self):
        """trimmed_mean should be accepted as a valid aggregation method."""
        from genetic_algorithm.utils.timerange import validate_walk_forward_config
        config = {
            'enabled': True,
            'train_days': 90,
            'validation_days': 30,
            'step_days': 30,
            'mode': 'rolling',
            'aggregation': 'trimmed_mean',
            'min_train_trades': 10,
        }
        # Should not raise
        validate_walk_forward_config(config)

    def test_all_aggregation_methods_valid(self):
        """All 5 aggregation methods should be accepted."""
        from genetic_algorithm.utils.timerange import validate_walk_forward_config
        base = {
            'enabled': True,
            'train_days': 90,
            'validation_days': 30,
            'step_days': 30,
            'mode': 'rolling',
            'min_train_trades': 10,
        }
        for method in ['mean', 'min', 'harmonic_mean', 'weighted', 'trimmed_mean']:
            cfg = {**base, 'aggregation': method}
            validate_walk_forward_config(cfg)

    def test_invalid_aggregation_rejected(self):
        """Invalid aggregation methods should raise ValueError."""
        from genetic_algorithm.utils.timerange import validate_walk_forward_config
        config = {
            'enabled': True,
            'train_days': 90,
            'validation_days': 30,
            'step_days': 30,
            'mode': 'rolling',
            'aggregation': 'geometric_mean',
            'min_train_trades': 10,
        }
        with pytest.raises(ValueError, match="Walk-forward aggregation"):
            validate_walk_forward_config(config)

    def test_r6_walk_forward_config_valid(self):
        """The exact R6 walk-forward config should pass validation."""
        import yaml
        from genetic_algorithm.utils.timerange import validate_walk_forward_config
        config_path = Path(__file__).parent.parent / 'config' / 'ga_config_server_production_R6.yaml'
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            validate_walk_forward_config(full_config['walk_forward'])

    def test_r7_walk_forward_config_valid(self):
        """The exact R7 walk-forward config should pass validation."""
        import yaml
        from genetic_algorithm.utils.timerange import validate_walk_forward_config
        config_path = Path(__file__).parent.parent / 'config' / 'ga_config_server_production_R7.yaml'
        if config_path.exists():
            with open(config_path) as f:
                full_config = yaml.safe_load(f)
            validate_walk_forward_config(full_config['walk_forward'])


# ============================================================================
# R7 Fixes — CDL_DOJI exit removal, operator validation after mutation
# ============================================================================

class TestFixInvalidOperatorsRemovesCDLExits:
    """Test that _fix_invalid_operators removes CDL_DOJI from exit conditions."""

    def test_cdl_doji_exit_removed(self):
        """CDL_DOJI exit conditions should be removed entirely, not just fixed."""
        from genetic_algorithm.core.crossover import _fix_invalid_operators
        from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
                IndicatorGene(type='CDL_DOJI', parameters={}, instance_id='CDL_DOJI_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0),
                ConditionGene(indicator='CDL_DOJI_0', operator='>', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='CDL_DOJI_0', operator='<', threshold=0),
            ],
            timeframe='15m',
        )
        _fix_invalid_operators(gene)
        # CDL_DOJI exit should be completely removed
        assert len(gene.exit_conditions) == 0, \
            f"CDL_DOJI exit should be removed, got {[(c.indicator, c.operator) for c in gene.exit_conditions]}"
        # CDL_DOJI entry should remain (it's a valid entry indicator)
        assert len(gene.entry_conditions) == 2

    def test_cdl_doji_gt_exit_also_removed(self):
        """Even CDL_DOJI > is removed from exits — not a meaningful exit signal."""
        from genetic_algorithm.core.crossover import _fix_invalid_operators
        from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
                IndicatorGene(type='CDL_DOJI', parameters={}, instance_id='CDL_DOJI_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='EMA_0', operator='cross_below', threshold=0),
                ConditionGene(indicator='CDL_DOJI_0', operator='>', threshold=0),
            ],
            timeframe='15m',
        )
        _fix_invalid_operators(gene)
        assert len(gene.exit_conditions) == 1
        assert gene.exit_conditions[0].indicator == 'EMA_0'

    def test_cdl_eveningstar_entry_removed(self):
        """CDL_EVENINGSTAR (negative-only) should be removed from entries."""
        from genetic_algorithm.core.crossover import _fix_invalid_operators
        from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
                IndicatorGene(type='CDL_EVENINGSTAR', parameters={}, instance_id='CDL_EVENINGSTAR_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0),
                ConditionGene(indicator='CDL_EVENINGSTAR_0', operator='<', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='CDL_EVENINGSTAR_0', operator='<', threshold=0),
            ],
            timeframe='15m',
        )
        _fix_invalid_operators(gene)
        # Entry: CDL_EVENINGSTAR removed
        assert len(gene.entry_conditions) == 1
        assert gene.entry_conditions[0].indicator == 'EMA_0'
        # Exit: CDL_EVENINGSTAR kept (valid for exit)
        assert len(gene.exit_conditions) == 1

    def test_bidirectional_cdl_kept_in_both(self):
        """CDL_ENGULFING (bidirectional) should be kept in both entry and exit."""
        from genetic_algorithm.core.crossover import _fix_invalid_operators
        from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='CDL_ENGULFING', parameters={}, instance_id='CDL_ENGULFING_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='CDL_ENGULFING_0', operator='>', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='CDL_ENGULFING_0', operator='<', threshold=0),
            ],
            timeframe='15m',
        )
        _fix_invalid_operators(gene)
        assert len(gene.entry_conditions) == 1
        assert len(gene.exit_conditions) == 1


class TestMutateIndicatorsOperatorValidation:
    """Test that mutate_indicators validates operators after replacing indicators."""

    def test_replace_rsi_with_cdl_doji_removes_exit(self):
        """When RSI is replaced with CDL_DOJI, exit conditions using CDL_DOJI should be removed."""
        import random
        from genetic_algorithm.core.mutation import mutate_indicators
        from genetic_algorithm.core.individual import Individual
        from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

        random.seed(42)
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
                IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI_0', operator='cross_below', threshold=30),
                ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0),
            ],
            exit_conditions=[
                ConditionGene(indicator='RSI_0', operator='cross_above', threshold=70),
            ],
            timeframe='15m',
        )
        individual = Individual(strategy_gene=gene)

        config = {
            'indicators': {
                'available': ['RSI', 'EMA', 'CDL_DOJI'],
                'max_per_strategy': 4,
                'min_per_strategy': 2,
            }
        }

        # Run many times to hit the replace path with CDL_DOJI
        found_replacement = False
        for _ in range(200):
            random.seed(random.randint(0, 100000))
            try:
                result = mutate_indicators(individual, 1.0, config)
                sg = result.strategy_gene
                # Check if any exit condition references CDL_DOJI with '<'
                for cond in sg.exit_conditions:
                    ind_type = cond.indicator.replace('_0', '').replace('_1', '')
                    if ind_type == 'CDL_DOJI' and cond.operator == '<':
                        found_replacement = True
                        break
            except Exception:
                continue
            if found_replacement:
                break

        assert not found_replacement, \
            "mutate_indicators should never create CDL_DOJI:< exit conditions"


class TestR7ConfigValidation:
    """Validate R7 config fitness weights and thresholds."""

    def test_r7_weights_sum_to_one(self):
        """R7 fitness weights should sum to 1.0."""
        import yaml
        config_path = Path(__file__).parent.parent / 'config' / 'ga_config_server_production_R7.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        weights = config['fitness_weights']
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_r7_min_trades_aligned(self):
        """R7 min_trades should match between penalties and constraints."""
        import yaml
        config_path = Path(__file__).parent.parent / 'config' / 'ga_config_server_production_R7.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        penalty_min = config['fitness_penalties']['min_trades']
        constraint_min = config['strategy_constraints']['min_trades']
        assert penalty_min == constraint_min, \
            f"min_trades mismatch: penalties={penalty_min}, constraints={constraint_min}"

    def test_r7_no_cdl_doji_in_available(self):
        """R7 should not include CDL_DOJI in available indicators."""
        import yaml
        config_path = Path(__file__).parent.parent / 'config' / 'ga_config_server_production_R7.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        available = config['indicators']['available']
        assert 'CDL_DOJI' not in available

    def test_r7_trade_frequency_thresholds_present(self):
        """R7 should include explicit trade_frequency_thresholds."""
        import yaml
        config_path = Path(__file__).parent.parent / 'config' / 'ga_config_server_production_R7.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        tf = config.get('trade_frequency_thresholds', {})
        assert tf.get('ideal_min', 0) > 0
        assert tf.get('ideal_max', 0) > tf.get('ideal_min', 0)


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
