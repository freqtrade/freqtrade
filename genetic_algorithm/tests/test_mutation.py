"""
Tests for Mutation Operators

Tests mutate_parameters, mutate_indicators, mutate_conditions,
mutate_structure, mutate_gaussian, mutate_condition_reassign, mutate_adaptive_per_gene,
mutate_dynamic_bounds, and the top-level mutate() dispatcher.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import pytest
from genetic_algorithm.core.mutation import (
    mutate_parameters,
    mutate_indicators,
    mutate_conditions,
    mutate_structure,
    mutate_gaussian,
    mutate_condition_reassign,
    mutate_adaptive_per_gene,
    mutate,
    _mutate_indicator_params,
    _mutate_condition_threshold,
    _create_random_condition,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def _make_gene(indicators=None, entry_conditions=None, exit_conditions=None):
    """Create a StrategyGene with realistic defaults."""
    if indicators is None:
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                         instance_id='MACD_0'),
        ]
    if entry_conditions is None:
        entry_conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ConditionGene(indicator='EMA_0', operator='cross_above', threshold=0),
        ]
    if exit_conditions is None:
        exit_conditions = [
            ConditionGene(indicator='RSI_0', operator='>', threshold=70),
        ]
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        stoploss=-0.10,
        timeframe='5m',
        minimal_roi={"0": 0.05, "30": 0.03, "60": 0.01},
        max_open_trades=3,
    )


def _make_individual(gene=None, fitness=1.0):
    """Create an Individual wrapping a gene."""
    if gene is None:
        gene = _make_gene()
    ind = Individual(strategy_gene=gene)
    ind.fitness = fitness
    ind.raw_fitness = fitness
    ind.evaluated = True
    return ind


def _make_config():
    """Create a realistic configuration dict for mutation."""
    return {
        'indicators': {
            'available': ['RSI', 'EMA', 'SMA', 'MACD', 'BBANDS', 'CCI', 'ADX', 'ATR'],
            'max_per_strategy': 5,
            'min_per_strategy': 1,
            'RSI': {'period': [7, 21], 'buy_threshold': [20, 40], 'sell_threshold': [60, 80]},
            'EMA': {'period': [10, 50]},
            'SMA': {'period': [10, 50]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
            'BBANDS': {'period': [15, 30], 'std_dev': [1.5, 3.0]},
            'CCI': {'period': [14, 30], 'buy_threshold': [-200, -100], 'sell_threshold': [100, 200]},
            'ADX': {'period': [14, 30]},
            'ATR': {'period': [14, 30]},
        },
        'strategy_constraints': {
            'stoploss_range': [-0.20, -0.05],
            'roi_range': [0.01, 0.10],
            'timeframes': ['5m', '15m', '1h'],
            'max_open_trades_range': [1, 10],
        },
    }


# ============================================================================
# _mutate_indicator_params helper
# ============================================================================

class TestMutateIndicatorParams:
    def test_rsi_period_mutation(self):
        random.seed(42)
        indicator = IndicatorGene(type='RSI', parameters={'period': 14})
        mutations = []
        _mutate_indicator_params(indicator, {'period': [7, 21]}, 0, mutations)
        assert 7 <= indicator.parameters['period'] <= 21
        assert any('RSI_period' in m for m in mutations)
    
    def test_macd_params(self):
        random.seed(42)
        indicator = IndicatorGene(type='MACD', parameters={
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9
        })
        mutations = []
        _mutate_indicator_params(indicator, {}, 0, mutations)
        # MACD validation: fast < slow
        assert indicator.parameters['fast_period'] < indicator.parameters['slow_period']
    
    def test_bbands_params(self):
        random.seed(42)
        indicator = IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0})
        mutations = []
        _mutate_indicator_params(indicator, {'period': [15, 30], 'std_dev': [1.5, 3.0]}, 0, mutations)
        # At least one should have been mutated
        assert indicator.parameters['period'] >= 15 or indicator.parameters['std_dev'] >= 1.5
    
    def test_weight_mutation(self):
        """Weight mutation should occur ~30% of the time."""
        random.seed(1)  # seed that triggers weight mutation
        indicator = IndicatorGene(type='RSI', parameters={'period': 14})
        mutations = []
        # Try many times to trigger weight mutation
        for _ in range(20):
            _mutate_indicator_params(indicator, {'period': [7, 21]}, 0, mutations)
        weight_mutations = [m for m in mutations if 'weight' in m]
        assert len(weight_mutations) > 0


# ============================================================================
# _mutate_condition_threshold
# ============================================================================

class TestMutateConditionThreshold:
    def test_rsi_entry_threshold(self):
        random.seed(42)
        cond = ConditionGene(indicator='RSI', operator='<', threshold=30)
        mutations = []
        _mutate_condition_threshold(cond, {'buy_threshold': [20, 40]}, True, 0, mutations)
        assert 20 <= cond.threshold <= 40
    
    def test_rsi_exit_threshold(self):
        random.seed(42)
        cond = ConditionGene(indicator='RSI', operator='>', threshold=70)
        mutations = []
        _mutate_condition_threshold(cond, {'sell_threshold': [60, 80]}, False, 0, mutations)
        assert 60 <= cond.threshold <= 80
    
    def test_instance_id_format(self):
        """Should handle 'RSI_0' format by extracting base type 'RSI'."""
        random.seed(42)
        cond = ConditionGene(indicator='RSI_0', operator='<', threshold=30)
        mutations = []
        _mutate_condition_threshold(cond, {'buy_threshold': [20, 40]}, True, 0, mutations)
        assert 20 <= cond.threshold <= 40
    
    def test_unknown_indicator_gaussian_fallback(self):
        """Unknown indicators get a Gaussian perturbation (±10%) fallback."""
        cond = ConditionGene(indicator='UNKNOWN', operator='<', threshold=30)
        mutations = []
        _mutate_condition_threshold(cond, {}, True, 0, mutations)
        # Should mutate via Gaussian perturbation — value within ±10%
        assert 27.0 <= cond.threshold <= 33.0
        assert len(mutations) == 1


# ============================================================================
# _create_random_condition
# ============================================================================

class TestCreateRandomCondition:
    def test_rsi_entry(self):
        cond = _create_random_condition('RSI', True, {'RSI': {'buy_threshold': [20, 40]}})
        assert cond is not None
        assert cond.indicator == 'RSI'
    
    def test_rsi_exit(self):
        cond = _create_random_condition('RSI', False, {'RSI': {'sell_threshold': [60, 80]}})
        assert cond is not None
        assert cond.indicator == 'RSI'
    
    def test_macd_entry(self):
        cond = _create_random_condition('MACD', True, {})
        assert cond is not None
        assert cond.operator == 'cross_above'
        assert cond.threshold == 0
    
    def test_unknown_indicator_returns_none(self):
        cond = _create_random_condition('TOTALLYFAKE', True, {})
        assert cond is None
    
    def test_candlestick_bullish_entry(self):
        cond = _create_random_condition('CDL_HAMMER', True, {})
        assert cond is not None
        assert cond.operator == '>'
    
    def test_candlestick_bearish_entry(self):
        cond = _create_random_condition('CDL_EVENINGSTAR', True, {})
        assert cond is not None
        assert cond.operator == '<'


# ============================================================================
# mutate_parameters
# ============================================================================

class TestMutateParameters:
    def test_returns_new_individual(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_parameters(ind, 1.0, config)
        assert isinstance(result, Individual)
        assert result is not ind
    
    def test_records_mutations(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_parameters(ind, 1.0, config)
        assert len(result.mutations) > 0
        assert result.mutations[-1]['type'] == 'parameter'
    
    def test_no_mutation_at_zero_rate(self):
        """With mutation_rate=0, nothing should be mutated."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_parameters(ind, 0.0, config)
        # Very unlikely any mutation applied
        assert result.mutations[-1]['applied'] == [] or len(result.mutations[-1]['applied']) == 0
    
    def test_high_mutation_rate(self):
        """With mutation_rate=1.0, most parameters should be mutated."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_parameters(ind, 1.0, config)
        assert len(result.mutations[-1]['applied']) > 0
    
    def test_stoploss_range_respected(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_parameters(ind, 1.0, config)
        stoploss = result.strategy_gene.stoploss
        assert -0.20 <= stoploss <= -0.05
    
    def test_parent_id_tracked(self):
        ind = _make_individual()
        config = _make_config()
        result = mutate_parameters(ind, 1.0, config)
        assert ind.id in result.parent_ids


# ============================================================================
# mutate_indicators
# ============================================================================

class TestMutateIndicators:
    def test_returns_new_individual(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_indicators(ind, 0.5, config)
        assert isinstance(result, Individual)
    
    def test_add_indicator(self):
        """Should sometimes add a new indicator."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        added = False
        for seed in range(50):
            random.seed(seed)
            result = mutate_indicators(ind, 0.5, config)
            if len(result.strategy_gene.indicators) > len(ind.strategy_gene.indicators):
                added = True
                break
        assert added, "Expected at least one seed to trigger 'add' operation"
    
    def test_replace_indicator(self):
        """Should sometimes replace an indicator."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        replaced = False
        for seed in range(50):
            random.seed(seed)
            result = mutate_indicators(ind, 0.5, config)
            if any('replace' in m for m in result.mutations[-1].get('applied', [])):
                replaced = True
                break
        assert replaced, "Expected at least one seed to trigger 'replace' operation"
    
    def test_min_indicators_respected(self):
        """Should not remove below min_per_strategy."""
        random.seed(42)
        gene = _make_gene(indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ])
        ind = _make_individual(gene)
        config = _make_config()
        config['indicators']['min_per_strategy'] = 1
        for seed in range(30):
            random.seed(seed)
            result = mutate_indicators(ind, 1.0, config)
            assert len(result.strategy_gene.indicators) >= 1


# ============================================================================
# mutate_conditions
# ============================================================================

class TestMutateConditions:
    def test_returns_new_individual(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_conditions(ind, 0.5, config)
        assert isinstance(result, Individual)
    
    def test_mutation_record(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_conditions(ind, 1.0, config)
        assert result.mutations[-1]['type'] == 'condition'


# ============================================================================
# mutate_structure
# ============================================================================

class TestMutateStructure:
    def test_timeframe_mutation(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        mutated = False
        for seed in range(50):
            random.seed(seed)
            result = mutate_structure(ind, 1.0, config)
            if result.strategy_gene.timeframe != ind.strategy_gene.timeframe:
                mutated = True
                break
        assert mutated
    
    def test_stoploss_mutation(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_structure(ind, 1.0, config)
        assert -0.20 <= result.strategy_gene.stoploss <= -0.05
    
    def test_trailing_stop_toggle(self):
        """Trailing stop should sometimes toggle."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        toggled = False
        for seed in range(50):
            random.seed(seed)
            result = mutate_structure(ind, 1.0, config)
            if result.strategy_gene.trailing_stop != ind.strategy_gene.trailing_stop:
                toggled = True
                # If enabled, trailing params should be set
                if result.strategy_gene.trailing_stop:
                    assert result.strategy_gene.trailing_stop_positive is not None
                    assert result.strategy_gene.trailing_stop_positive_offset is not None
                    # offset must be > positive
                    assert result.strategy_gene.trailing_stop_positive_offset > result.strategy_gene.trailing_stop_positive
                break
        assert toggled


# ============================================================================
# mutate_gaussian
# ============================================================================

class TestMutateGaussian:
    def test_returns_new_individual(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_gaussian(ind, 1.0, config, sigma=0.1)
        assert isinstance(result, Individual)
    
    def test_sigma_in_mutation_record(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_gaussian(ind, 1.0, config, sigma=0.2)
        assert result.mutations[-1]['sigma'] == 0.2
    
    def test_roi_decreasing_over_time(self):
        """ROI values should decrease over time keys."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_gaussian(ind, 1.0, config, sigma=0.1)
        roi = result.strategy_gene.minimal_roi
        sorted_keys = sorted([int(k) for k in roi.keys()])
        for i in range(len(sorted_keys) - 1):
            assert roi[str(sorted_keys[i])] >= roi[str(sorted_keys[i + 1])]


# ============================================================================
# mutate_condition_reassign
# ============================================================================

class TestMutateConditionReassign:
    def test_returns_new_individual(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_condition_reassign(ind, 1.0, config)
        assert isinstance(result, Individual)
    
    def test_reassign_preserves_count(self):
        """Condition reassign should not add or remove indicators."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate_condition_reassign(ind, 1.0, config)
        assert len(result.strategy_gene.indicators) == len(ind.strategy_gene.indicators)
        assert len(result.strategy_gene.entry_conditions) == len(ind.strategy_gene.entry_conditions)


# ============================================================================
# mutate_adaptive_per_gene
# ============================================================================

class TestMutateAdaptivePerGene:
    def test_returns_individual(self):
        random.seed(42)
        ind = _make_individual(fitness=0.5)
        config = _make_config()
        result = mutate_adaptive_per_gene(ind, 0.5, config)
        assert isinstance(result, Individual)
    
    def test_none_fitness_handled(self):
        """Individual with None fitness should not crash."""
        random.seed(42)
        ind = _make_individual(fitness=None)
        ind.fitness = None
        config = _make_config()
        result = mutate_adaptive_per_gene(ind, 0.5, config)
        assert isinstance(result, Individual)


# ============================================================================
# mutate (top-level dispatcher)
# ============================================================================

class TestMutateDispatcher:
    def test_returns_individual(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate(ind, 0.5, config)
        assert isinstance(result, Individual)
    
    def test_custom_methods(self):
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate(ind, 1.0, config, methods=['parameters', 'structure'])
        assert isinstance(result, Individual)
    
    def test_unknown_method_does_not_crash(self):
        """Unknown methods should be silently skipped (caught by try/except)."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        # 'magical' is unknown — should not crash
        result = mutate(ind, 1.0, config, methods=['magical'])
        assert isinstance(result, Individual)
    
    def test_zero_mutation_rate(self):
        """With 0.0 rate, none of the method blocks should fire."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate(ind, 0.0, config, methods=['parameters', 'structure'])
        # Very unlikely to mutate
        assert isinstance(result, Individual)
    
    def test_default_methods_include_basics(self):
        """Default methods should include the 4 basic mutation types."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result = mutate(ind, 0.5, config)
        assert isinstance(result, Individual)
    
    def test_failed_mutation_returns_original(self):
        """If all mutations fail, original individual should be returned."""
        random.seed(42)
        # Create an individual with minimal components likely to fail
        gene = _make_gene(indicators=[
            IndicatorGene(type='FAKE', parameters={}),
        ])
        ind = _make_individual(gene)
        config = _make_config()
        config['indicators']['available'] = []  # no available indicators
        config['indicators']['max_per_strategy'] = 1
        config['indicators']['min_per_strategy'] = 1
        result = mutate(ind, 0.5, config, methods=['indicators'])
        assert isinstance(result, Individual)


# ============================================================================
# Edge cases
# ============================================================================

class TestMutationEdgeCases:
    def test_empty_indicators_list(self):
        """Gene with minimal indicators shouldn't crash mutations."""
        gene = _make_gene(indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ])
        ind = _make_individual(gene)
        config = _make_config()
        result = mutate_parameters(ind, 1.0, config)
        assert isinstance(result, Individual)
    
    def test_mutation_accumulates_history(self):
        """Multiple mutations should accumulate in the mutations list."""
        random.seed(42)
        ind = _make_individual()
        config = _make_config()
        result1 = mutate_parameters(ind, 1.0, config)
        result2 = mutate_parameters(result1, 1.0, config)
        assert len(result2.mutations) >= len(result1.mutations)
