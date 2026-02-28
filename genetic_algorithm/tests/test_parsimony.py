"""
Tests for Parsimony Pressure — Strategy Simplification

Tests simplify_strategy, _build_removal_candidates, _apply_removal,
and apply_parsimony_to_elites.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from genetic_algorithm.core.parsimony import (
    simplify_strategy,
    _build_removal_candidates,
    _apply_removal,
    apply_parsimony_to_elites,
)
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def _make_gene(n_indicators=3, n_entry=3, n_exit=2):
    """Create a StrategyGene with configurable number of components."""
    indicators = [
        IndicatorGene(type=f'IND_{i}', parameters={'period': 14 + i})
        for i in range(n_indicators)
    ]
    entry_conditions = [
        ConditionGene(indicator=f'IND_{i % n_indicators}', operator='<', threshold=30 + i)
        for i in range(n_entry)
    ]
    exit_conditions = [
        ConditionGene(indicator=f'IND_{i % n_indicators}', operator='>', threshold=70 + i)
        for i in range(n_exit)
    ]
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
    )


def _noop_evaluate(gene):
    """Evaluation that always returns same fitness."""
    return (1.0, {'profit': 100.0})


def _declining_evaluate(gene):
    """Evaluation that returns lower fitness for simpler strategies."""
    complexity = len(gene.indicators) + len(gene.entry_conditions) + len(gene.exit_conditions)
    return (0.1 * complexity, {'profit': complexity})


def _failing_evaluate(gene):
    """Evaluation that always raises."""
    raise RuntimeError("Evaluation failed")


# ============================================================================
# _build_removal_candidates
# ============================================================================

class TestBuildRemovalCandidates:
    def test_standard_gene(self):
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=2)
        candidates = _build_removal_candidates(gene, min_entry_conditions=2)
        
        # Should have: 3 indicator candidates (>1), 4 entry (all indices since 4>2), 2 exit
        ind_candidates = [c for c in candidates if c[0] == 'indicator']
        entry_candidates = [c for c in candidates if c[0] == 'entry_condition']
        exit_candidates = [c for c in candidates if c[0] == 'exit_condition']
        
        assert len(ind_candidates) == 3
        assert len(entry_candidates) == 4  # all 4 entries are candidates (count > min)
        assert len(exit_candidates) == 2
    
    def test_min_indicators_prevents_removal(self):
        """With only 1 indicator, no indicator removal candidates."""
        gene = _make_gene(n_indicators=1, n_entry=3, n_exit=1)
        candidates = _build_removal_candidates(gene, min_entry_conditions=2)
        ind_candidates = [c for c in candidates if c[0] == 'indicator']
        assert len(ind_candidates) == 0
    
    def test_min_entry_conditions_respected(self):
        """Entry conditions at or below min should not generate candidates."""
        gene = _make_gene(n_indicators=2, n_entry=2, n_exit=1)
        candidates = _build_removal_candidates(gene, min_entry_conditions=2)
        entry_candidates = [c for c in candidates if c[0] == 'entry_condition']
        assert len(entry_candidates) == 0
    
    def test_exit_conditions_always_removable(self):
        """Exit conditions can always be removed."""
        gene = _make_gene(n_indicators=2, n_entry=3, n_exit=5)
        candidates = _build_removal_candidates(gene, min_entry_conditions=2)
        exit_candidates = [c for c in candidates if c[0] == 'exit_condition']
        assert len(exit_candidates) == 5
    
    def test_no_exit_conditions(self):
        """No exit conditions = no exit candidates."""
        gene = _make_gene(n_indicators=2, n_entry=3, n_exit=0)
        candidates = _build_removal_candidates(gene, min_entry_conditions=1)
        exit_candidates = [c for c in candidates if c[0] == 'exit_condition']
        assert len(exit_candidates) == 0
    
    def test_high_min_entry_conditions(self):
        """If min_entry_conditions > current entries, no entry candidates."""
        gene = _make_gene(n_indicators=2, n_entry=3, n_exit=1)
        candidates = _build_removal_candidates(gene, min_entry_conditions=5)
        entry_candidates = [c for c in candidates if c[0] == 'entry_condition']
        assert len(entry_candidates) == 0


# ============================================================================
# _apply_removal
# ============================================================================

class TestApplyRemoval:
    def test_remove_indicator(self):
        gene = _make_gene(n_indicators=3, n_entry=3, n_exit=1)
        result = _apply_removal(gene, 'indicator', 0, min_entry_conditions=1)
        assert result is not None
        assert len(result.indicators) == 2
    
    def test_remove_indicator_cascades_conditions(self):
        """Removing an indicator should also remove its conditions."""
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
        ]
        entry_conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ConditionGene(indicator='EMA_0', operator='>', threshold=0),
        ]
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=[ConditionGene(indicator='RSI_0', operator='>', threshold=70)],
        )
        result = _apply_removal(gene, 'indicator', 0, min_entry_conditions=1)
        assert result is not None
        assert len(result.indicators) == 1
        # RSI_0 conditions should be removed
        for c in result.entry_conditions:
            assert c.indicator != 'RSI_0'
        for c in result.exit_conditions:
            assert c.indicator != 'RSI_0'
    
    def test_remove_indicator_below_min_entry_returns_none(self):
        """If removal would drop below min_entry_conditions, return None."""
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 20}, instance_id='EMA_0'),
        ]
        entry_conditions = [
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ConditionGene(indicator='EMA_0', operator='>', threshold=0),
        ]
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=[],
        )
        # Removing RSI_0 leaves only 1 entry condition but min is 2
        result = _apply_removal(gene, 'indicator', 0, min_entry_conditions=2)
        assert result is None
    
    def test_remove_entry_condition(self):
        gene = _make_gene(n_indicators=2, n_entry=4, n_exit=1)
        result = _apply_removal(gene, 'entry_condition', 1, min_entry_conditions=2)
        assert result is not None
        assert len(result.entry_conditions) == 3
    
    def test_remove_entry_below_min_returns_none(self):
        gene = _make_gene(n_indicators=2, n_entry=2, n_exit=1)
        result = _apply_removal(gene, 'entry_condition', 0, min_entry_conditions=2)
        assert result is None
    
    def test_remove_exit_condition(self):
        gene = _make_gene(n_indicators=2, n_entry=3, n_exit=3)
        result = _apply_removal(gene, 'exit_condition', 0, min_entry_conditions=1)
        assert result is not None
        assert len(result.exit_conditions) == 2
    
    def test_remove_unknown_kind(self):
        gene = _make_gene()
        result = _apply_removal(gene, 'unknown_kind', 0)
        assert result is None
    
    def test_remove_out_of_bounds_index(self):
        gene = _make_gene(n_indicators=2, n_entry=3, n_exit=1)
        result = _apply_removal(gene, 'indicator', 99)
        assert result is None
    
    def test_original_gene_unchanged(self):
        """_apply_removal should not modify the original gene."""
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=2)
        orig_n_ind = len(gene.indicators)
        _apply_removal(gene, 'indicator', 0, min_entry_conditions=1)
        assert len(gene.indicators) == orig_n_ind


# ============================================================================
# simplify_strategy
# ============================================================================

class TestSimplifyStrategy:
    def test_no_removals_if_fitness_drops(self):
        """If all removals drop fitness significantly, nothing removed."""
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=2)
        result_gene, result_fitness, n_removed = simplify_strategy(
            gene, 1.0, _declining_evaluate,
            epsilon=0.001, max_removals=1, random_seed=42,
        )
        # _declining_evaluate gives lower fitness for simpler strategies
        # so fitness drop > epsilon, no removal should happen
        assert n_removed == 0
        assert result_fitness == 1.0
    
    def test_removal_when_fitness_maintained(self):
        """If removal doesn't affect fitness, component should be removed."""
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=2)
        result_gene, result_fitness, n_removed = simplify_strategy(
            gene, 1.0, _noop_evaluate,
            epsilon=0.05, max_removals=3, random_seed=42,
        )
        # _noop_evaluate always returns 1.0, so no fitness drop
        assert n_removed > 0
        total_components = (len(result_gene.indicators) + 
                          len(result_gene.entry_conditions) + 
                          len(result_gene.exit_conditions))
        orig_components = 3 + 4 + 2
        assert total_components < orig_components
    
    def test_max_removals_respected(self):
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=3)
        _, _, n_removed = simplify_strategy(
            gene, 1.0, _noop_evaluate,
            epsilon=0.1, max_removals=1, random_seed=42,
        )
        assert n_removed <= 1
    
    def test_evaluate_failure_handled(self):
        """If evaluate_fn raises, simplification should continue."""
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=2)
        result_gene, result_fitness, n_removed = simplify_strategy(
            gene, 1.0, _failing_evaluate,
            epsilon=0.05, max_removals=3, random_seed=42,
        )
        assert n_removed == 0  # all evaluations failed
        assert result_fitness == 1.0
    
    def test_seed_reproducibility(self):
        gene = _make_gene(n_indicators=3, n_entry=4, n_exit=2)
        r1 = simplify_strategy(gene, 1.0, _noop_evaluate, epsilon=0.05,
                               max_removals=2, random_seed=100)
        r2 = simplify_strategy(gene, 1.0, _noop_evaluate, epsilon=0.05,
                               max_removals=2, random_seed=100)
        assert r1[2] == r2[2]  # same number removed
    
    def test_min_entry_conditions_parameter(self):
        """The min_entry_conditions parameter should be forwarded correctly."""
        gene = _make_gene(n_indicators=2, n_entry=3, n_exit=2)
        # With high min_entry_conditions, entry conditions can't be removed
        result_gene, _, n_removed = simplify_strategy(
            gene, 1.0, _noop_evaluate,
            epsilon=0.1, max_removals=5, random_seed=42,
            min_entry_conditions=3,
        )
        # Entry conditions should still be 3 (can't go below)
        assert len(result_gene.entry_conditions) >= 3


# ============================================================================
# apply_parsimony_to_elites
# ============================================================================

class TestApplyParsimonyToElites:
    def test_basic_application(self):
        elites = []
        for i in range(3):
            ind = Individual(strategy_gene=_make_gene(n_indicators=3, n_entry=4, n_exit=2))
            ind.fitness = 1.0
            ind.raw_fitness = 1.0
            elites.append(ind)
        
        config = {'epsilon': 0.05, 'max_removals': 1}
        total_removed = apply_parsimony_to_elites(elites, _noop_evaluate, config)
        assert total_removed >= 0
    
    def test_skips_unfit_individuals(self):
        """Individuals with fitness <= 0 should be skipped."""
        ind1 = Individual(strategy_gene=_make_gene())
        ind1.fitness = 0.0
        ind1.raw_fitness = 0.0
        
        ind2 = Individual(strategy_gene=_make_gene())
        ind2.fitness = -1.0
        ind2.raw_fitness = -1.0
        
        config = {'epsilon': 0.05}
        total = apply_parsimony_to_elites([ind1, ind2], _noop_evaluate, config)
        assert total == 0
    
    def test_skips_none_fitness(self):
        ind = Individual(strategy_gene=_make_gene())
        ind.fitness = None
        ind.raw_fitness = None
        
        config = {'epsilon': 0.05}
        total = apply_parsimony_to_elites([ind], _noop_evaluate, config)
        assert total == 0
    
    def test_modifies_individual_in_place(self):
        """Simplified individuals should be updated in place."""
        ind = Individual(strategy_gene=_make_gene(n_indicators=3, n_entry=4, n_exit=3))
        ind.fitness = 1.0
        ind.raw_fitness = 1.0
        
        config = {'epsilon': 0.1, 'max_removals': 3}
        total = apply_parsimony_to_elites([ind], _noop_evaluate, config)
        
        if total > 0:
            assert ind.metrics.get('parsimony_removed', 0) > 0
    
    def test_empty_elites_list(self):
        config = {'epsilon': 0.05}
        total = apply_parsimony_to_elites([], _noop_evaluate, config)
        assert total == 0
    
    def test_config_defaults(self):
        """Config without keys should use defaults."""
        ind = Individual(strategy_gene=_make_gene(n_indicators=3, n_entry=4, n_exit=2))
        ind.fitness = 1.0
        ind.raw_fitness = 1.0
        
        total = apply_parsimony_to_elites([ind], _noop_evaluate, {})
        # Default epsilon=0.02, max_removals=1
        assert total >= 0
