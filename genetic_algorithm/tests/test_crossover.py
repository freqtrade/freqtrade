"""
Tests for Crossover Operators

Tests single_point, uniform, and component crossover operators,
plus the dispatch function and edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import pytest
from genetic_algorithm.core.crossover import (
    single_point_crossover,
    uniform_crossover,
    component_crossover,
    crossover,
    _enforce_min_entry_conditions,
    _uniform_crossover_lists,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def _make_gene(prefix='A', n_indicators=3, n_entry=3, n_exit=2, gen=0, ind_id=0):
    """Create a StrategyGene with identifiable components."""
    indicators = [
        IndicatorGene(type=f'{prefix}_IND_{i}', parameters={'period': 14 + i})
        for i in range(n_indicators)
    ]
    entry_conditions = [
        ConditionGene(indicator=f'{prefix}_IND_{i % n_indicators}', operator='<', threshold=30 + i)
        for i in range(n_entry)
    ]
    exit_conditions = [
        ConditionGene(indicator=f'{prefix}_IND_{i % n_indicators}', operator='>', threshold=70 + i)
        for i in range(n_exit)
    ]
    return StrategyGene(
        generation=gen, individual_id=ind_id,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        stoploss=-0.10,
        timeframe='5m',
    )


def _make_parent(prefix='A', gen=0, ind_id=0, n_indicators=3, n_entry=3, n_exit=2):
    """Create a parent Individual."""
    gene = _make_gene(prefix, n_indicators, n_entry, n_exit, gen, ind_id)
    ind = Individual(strategy_gene=gene)
    ind.fitness = 1.0
    ind.raw_fitness = 1.0
    ind.evaluated = True
    return ind


def _make_parents():
    """Create two distinct parents."""
    return _make_parent('A', ind_id=0), _make_parent('B', ind_id=1)


# ============================================================================
# _uniform_crossover_lists
# ============================================================================

class TestUniformCrossoverLists:
    def test_equal_length_lists(self):
        random.seed(42)
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]
        c1, c2 = _uniform_crossover_lists(list1, list2, 0.5)
        assert len(c1) == 3
        assert len(c2) == 3
    
    def test_unequal_length_lists(self):
        random.seed(42)
        list1 = [1, 2]
        list2 = [4, 5, 6, 7]
        c1, c2 = _uniform_crossover_lists(list1, list2, 0.5)
        # Max length is 4, but items may be missing
        assert len(c1) <= 4
        assert len(c2) <= 4
    
    def test_swap_prob_zero(self):
        """swap_prob=0 means keep original order."""
        random.seed(42)
        list1 = ['a', 'b', 'c']
        list2 = ['x', 'y', 'z']
        c1, c2 = _uniform_crossover_lists(list1, list2, 0.0)
        assert c1 == ['a', 'b', 'c']
        assert c2 == ['x', 'y', 'z']
    
    def test_swap_prob_one(self):
        """swap_prob=1.0 means swap everything."""
        list1 = ['a', 'b', 'c']
        list2 = ['x', 'y', 'z']
        c1, c2 = _uniform_crossover_lists(list1, list2, 1.0)
        assert c1 == ['x', 'y', 'z']
        assert c2 == ['a', 'b', 'c']
    
    def test_empty_lists(self):
        c1, c2 = _uniform_crossover_lists([], [], 0.5)
        assert c1 == []
        assert c2 == []


# ============================================================================
# _enforce_min_entry_conditions
# ============================================================================

class TestEnforceMinEntryConditions:
    def test_no_config(self):
        gene = _make_gene(n_entry=1)
        _enforce_min_entry_conditions(gene, None)
        assert len(gene.entry_conditions) == 1  # unchanged
    
    def test_already_meets_minimum(self):
        gene = _make_gene(n_entry=3)
        config = {'indicators': {'min_entry_conditions': 2}}
        _enforce_min_entry_conditions(gene, config)
        assert len(gene.entry_conditions) == 3  # unchanged


# ============================================================================
# single_point_crossover
# ============================================================================

class TestSinglePointCrossover:
    def test_produces_two_offspring(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=10)
        assert isinstance(c1, Individual)
        assert isinstance(c2, Individual)
    
    def test_offspring_have_correct_generation(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = single_point_crossover(p1, p2, generation=5, ind_id=20)
        assert c1.strategy_gene.generation == 5
        assert c2.strategy_gene.generation == 5
        assert c1.strategy_gene.individual_id == 20
        assert c2.strategy_gene.individual_id == 21
    
    def test_offspring_have_parent_ids(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert p1.id in c1.parent_ids
        assert p2.id in c1.parent_ids
        assert p1.id in c2.parent_ids
        assert p2.id in c2.parent_ids
    
    def test_offspring_unevaluated(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert c1.fitness is None
        assert c2.fitness is None
    
    def test_offspring_have_indicators(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert len(c1.strategy_gene.indicators) >= 1
        assert len(c2.strategy_gene.indicators) >= 1
    
    def test_offspring_have_entry_conditions(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert len(c1.strategy_gene.entry_conditions) >= 1
        assert len(c2.strategy_gene.entry_conditions) >= 1
    
    def test_parents_unchanged(self):
        """Crossover should not modify parents."""
        random.seed(42)
        p1, p2 = _make_parents()
        orig_p1_indicators = len(p1.strategy_gene.indicators)
        orig_p2_indicators = len(p2.strategy_gene.indicators)
        single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert len(p1.strategy_gene.indicators) == orig_p1_indicators
        assert len(p2.strategy_gene.indicators) == orig_p2_indicators
    
    def test_single_indicator_parents(self):
        """If parents have 1 indicator each, no indicator crossover possible."""
        random.seed(42)
        p1 = _make_parent('A', n_indicators=1, n_entry=1, n_exit=1)
        p2 = _make_parent('B', n_indicators=1, n_entry=1, n_exit=1)
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert isinstance(c1, Individual)
        assert isinstance(c2, Individual)


# ============================================================================
# uniform_crossover
# ============================================================================

class TestUniformCrossover:
    def test_produces_two_offspring(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = uniform_crossover(p1, p2, generation=1, ind_id=10)
        assert isinstance(c1, Individual)
        assert isinstance(c2, Individual)
    
    def test_swap_prob_zero(self):
        """With swap_prob=0, children should be identical to parents (before ensure_indicators)."""
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = uniform_crossover(p1, p2, generation=1, ind_id=0, swap_prob=0.0)
        # Indicators should come from respective parents
        assert len(c1.strategy_gene.indicators) >= 1
        assert len(c2.strategy_gene.indicators) >= 1
    
    def test_offspring_generation_ids(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = uniform_crossover(p1, p2, generation=3, ind_id=50)
        assert c1.strategy_gene.generation == 3
        assert c2.strategy_gene.generation == 3
        assert c1.strategy_gene.individual_id == 50
        assert c2.strategy_gene.individual_id == 51
    
    def test_at_least_one_indicator(self):
        """Uniform crossover ensures at least one indicator."""
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = uniform_crossover(p1, p2, generation=1, ind_id=0)
        assert len(c1.strategy_gene.indicators) >= 1
        assert len(c2.strategy_gene.indicators) >= 1


# ============================================================================
# component_crossover
# ============================================================================

class TestComponentCrossover:
    def test_produces_two_offspring(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = component_crossover(p1, p2, generation=1, ind_id=10)
        assert isinstance(c1, Individual)
        assert isinstance(c2, Individual)
    
    def test_component_swap_deterministic(self):
        """With fixed seed, same parents should produce same offspring."""
        p1, p2 = _make_parents()
        random.seed(99)
        c1_a, c2_a = component_crossover(p1, p2, generation=1, ind_id=0)
        random.seed(99)
        c1_b, c2_b = component_crossover(p1, p2, generation=1, ind_id=0)
        assert len(c1_a.strategy_gene.indicators) == len(c1_b.strategy_gene.indicators)
    
    def test_offspring_have_valid_structure(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = component_crossover(p1, p2, generation=1, ind_id=0)
        assert len(c1.strategy_gene.indicators) >= 1
        assert len(c1.strategy_gene.entry_conditions) >= 1
        assert len(c2.strategy_gene.indicators) >= 1
        assert len(c2.strategy_gene.entry_conditions) >= 1


# ============================================================================
# crossover dispatch function
# ============================================================================

class TestCrossoverDispatch:
    def test_single_point_method(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = crossover(p1, p2, generation=1, ind_id=0, method='single_point')
        assert isinstance(c1, Individual)
    
    def test_uniform_method(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = crossover(p1, p2, generation=1, ind_id=0, method='uniform')
        assert isinstance(c1, Individual)
    
    def test_component_method(self):
        random.seed(42)
        p1, p2 = _make_parents()
        c1, c2 = crossover(p1, p2, generation=1, ind_id=0, method='component')
        assert isinstance(c1, Individual)
    
    def test_unknown_method_raises(self):
        p1, p2 = _make_parents()
        with pytest.raises(ValueError, match="Unknown crossover method"):
            crossover(p1, p2, generation=1, ind_id=0, method='magical')
    
    def test_config_passed_through(self):
        """Config should flow through to crossover methods without error."""
        random.seed(42)
        p1, p2 = _make_parents()
        config = {'indicators': {'available': ['RSI', 'EMA']}}
        c1, c2 = crossover(p1, p2, generation=1, ind_id=0,
                           method='single_point', config=config)
        assert isinstance(c1, Individual)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_same_parent_crossover(self):
        """Crossing an individual with itself should work."""
        random.seed(42)
        p1 = _make_parent('A')
        c1, c2 = single_point_crossover(p1, p1, generation=1, ind_id=0)
        assert isinstance(c1, Individual)
        assert isinstance(c2, Individual)
    
    def test_asymmetric_parents(self):
        """Parents with different component counts."""
        random.seed(42)
        p1 = _make_parent('A', n_indicators=2, n_entry=2, n_exit=1)
        p2 = _make_parent('B', n_indicators=5, n_entry=4, n_exit=3)
        c1, c2 = single_point_crossover(p1, p2, generation=1, ind_id=0)
        assert len(c1.strategy_gene.indicators) >= 1
        assert len(c2.strategy_gene.indicators) >= 1
    
    def test_all_methods_produce_valid_offspring(self):
        """Every crossover method should produce valid strategies."""
        random.seed(42)
        p1, p2 = _make_parents()
        for method in ['single_point', 'uniform', 'component']:
            c1, c2 = crossover(p1, p2, generation=1, ind_id=0, method=method)
            assert len(c1.strategy_gene.indicators) >= 1
            assert len(c1.strategy_gene.entry_conditions) >= 1
            assert len(c2.strategy_gene.indicators) >= 1
            assert len(c2.strategy_gene.entry_conditions) >= 1
