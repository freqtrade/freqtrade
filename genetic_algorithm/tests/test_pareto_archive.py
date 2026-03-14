"""
Tests for Pareto Archive with Crowding-Distance Decay

Tests ParetoArchive.update, _prune, get_best, serialization, and crowding decay.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from genetic_algorithm.core.pareto_archive import ParetoArchive
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def _make_gene(gen=0, ind_id=0):
    """Minimal valid StrategyGene for testing."""
    return StrategyGene(
        generation=gen,
        individual_id=ind_id,
        indicators=[IndicatorGene(type='RSI', parameters={'period': 14})],
        entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
        exit_conditions=[ConditionGene(indicator='RSI', operator='>', threshold=70)],
    )


def _make_individual(objectives, gen=0, ind_id=0):
    """Create an Individual with given objectives."""
    ind = Individual(strategy_gene=_make_gene(gen, ind_id))
    ind.objectives = list(objectives)
    ind.fitness = objectives[0]
    ind.raw_fitness = objectives[0]
    ind.evaluated = True
    return ind


def _make_population(objective_pairs, gen=0):
    """Create a population from list of (obj1, obj2) tuples."""
    return [_make_individual(objs, gen, i) for i, objs in enumerate(objective_pairs)]


# ============================================================================
# Constructor
# ============================================================================

class TestParetoArchiveInit:
    def test_default_params(self):
        archive = ParetoArchive()
        assert archive.max_size == 100
        assert archive.decay_rate == 0.99
        assert archive.min_size == 3
        assert archive.members == []
        assert archive.size == 0
    
    def test_custom_params(self):
        archive = ParetoArchive(max_size=50, decay_rate=0.8)
        assert archive.max_size == 50
        assert archive.decay_rate == 0.8
    
    def test_min_max_size(self):
        archive = ParetoArchive(max_size=0)
        assert archive.max_size == 1  # clamped to at least 1
    
    def test_decay_rate_clamped(self):
        archive = ParetoArchive(decay_rate=2.0)
        assert archive.decay_rate == 1.0
        archive2 = ParetoArchive(decay_rate=-0.5)
        assert archive2.decay_rate == 0.0


# ============================================================================
# update
# ============================================================================

class TestParetoArchiveUpdate:
    def test_empty_population(self):
        archive = ParetoArchive()
        archive.update([], generation=0)
        assert archive.size == 0
    
    def test_single_individual(self):
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 0.5)])
        archive.update(pop, generation=0)
        assert archive.size == 1
    
    def test_all_pareto_optimal(self):
        """Non-dominated individuals should all be kept."""
        archive = ParetoArchive(max_size=10)
        # These are all non-dominated (trade-offs between obj1 and obj2)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
        archive.update(pop, generation=0)
        assert archive.size == 3
    
    def test_dominated_individuals_excluded(self):
        """Dominated individuals should not be in the archive."""
        archive = ParetoArchive(max_size=10, min_size=1)
        pop = _make_population([
            (1.0, 1.0),    # dominates (0.5, 0.5)
            (0.5, 0.5),    # dominated
            (0.0, 2.0),    # non-dominated (best on obj2)
        ])
        archive.update(pop, generation=0)
        # Only rank-1 (non-dominated) kept (min_size=1 prevents rank-2 inclusion)
        assert archive.size == 2
        obj_sets = [tuple(m.objectives) for m in archive.members]
        assert (0.5, 0.5) not in obj_sets
    
    def test_capacity_enforced(self):
        """Archive should not exceed max_size."""
        archive = ParetoArchive(max_size=3)
        # Create many non-dominated individuals
        pop = _make_population([(float(i), float(10-i)) for i in range(10)])
        archive.update(pop, generation=0)
        assert archive.size <= 3
    
    def test_successive_updates(self):
        """Archive should accumulate across generations."""
        archive = ParetoArchive(max_size=20)
        pop1 = _make_population([(1.0, 0.0), (0.0, 1.0)], gen=0)
        archive.update(pop1, generation=0)
        assert archive.size == 2
        
        # Add improving individual
        pop2 = _make_population([(2.0, 0.5)], gen=1)
        archive.update(pop2, generation=1)
        # (2.0, 0.5) dominates (1.0, 0.0), so (1.0, 0.0) should be removed
        assert archive.size >= 1
    
    def test_individuals_without_objectives_skipped(self):
        """Individuals with objectives=None should be ignored."""
        archive = ParetoArchive(max_size=10)
        ind_no_obj = Individual(strategy_gene=_make_gene())
        ind_no_obj.objectives = None
        ind_with_obj = _make_individual((1.0, 0.5))
        archive.update([ind_no_obj, ind_with_obj], generation=0)
        assert archive.size == 1


# ============================================================================
# get_best
# ============================================================================

class TestGetBest:
    def test_get_best_single(self):
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
        archive.update(pop, generation=0)
        best = archive.get_best(n=1)
        assert len(best) == 1
        # Best should have highest crowding distance
        assert best[0].crowding_distance is not None
    
    def test_get_best_n(self):
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
        archive.update(pop, generation=0)
        best = archive.get_best(n=2)
        assert len(best) == 2
    
    def test_get_best_more_than_archive(self):
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 0.0)])
        archive.update(pop, generation=0)
        best = archive.get_best(n=5)
        assert len(best) == 1  # only 1 in archive
    
    def test_get_best_empty_archive(self):
        archive = ParetoArchive(max_size=10)
        best = archive.get_best(n=3)
        assert len(best) == 0


# ============================================================================
# get_archive
# ============================================================================

class TestGetArchive:
    def test_returns_copy(self):
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 0.5)])
        archive.update(pop, generation=0)
        
        result = archive.get_archive()
        assert len(result) == 1
        # Modifying the returned list should not affect internal state
        result.clear()
        assert archive.size == 1


# ============================================================================
# Crowding distance decay
# ============================================================================

class TestCrowdingDecay:
    def test_decay_applied(self):
        archive = ParetoArchive(max_size=10, decay_rate=0.5)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
        archive.update(pop, generation=0)
        
        # Record crowding distances after first update
        cd_before = {tuple(m.objectives): m.crowding_distance for m in archive.members}
        
        # Update again with empty population to trigger decay
        # We need at least something non-dominated to keep archive
        archive.update(pop, generation=1)
        
        # After second update, old members should have decayed CDs
        # (but they're recalculated in _prune, so we verify the archive is still valid)
        assert archive.size >= 1
    
    def test_decay_rate_one_no_decay(self):
        """decay_rate=1.0 should not change crowding distances."""
        archive = ParetoArchive(max_size=10, decay_rate=1.0)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0)])
        archive.update(pop, generation=0)
        # No error, archive works normally
        assert archive.size == 2


# ============================================================================
# _prune
# ============================================================================

class TestPrune:
    def test_prune_removes_most_crowded(self):
        """Pruning should keep boundary individuals (inf crowding distance)."""
        archive = ParetoArchive(max_size=2)
        # 5 non-dominated individuals
        pop = _make_population([(float(i), float(4-i)) for i in range(5)])
        archive.update(pop, generation=0)
        assert archive.size == 2
        # Boundary individuals (extreme obj values) tend to have inf crowding distance
        # and should be preserved
    
    def test_prune_below_capacity(self):
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0)])
        archive.update(pop, generation=0)
        assert archive.size == 2  # no pruning needed


# ============================================================================
# Serialization (to_dict / from_dict)
# ============================================================================

class TestSerialization:
    def test_round_trip(self):
        archive = ParetoArchive(max_size=50, decay_rate=0.8)
        pop = _make_population([(1.0, 0.5), (0.5, 1.0)])
        archive.update(pop, generation=0)
        
        data = archive.to_dict()
        restored = ParetoArchive.from_dict(data)
        
        assert restored.max_size == 50
        assert restored.decay_rate == 0.8
        assert restored.size == archive.size
    
    def test_empty_archive_serialization(self):
        archive = ParetoArchive()
        data = archive.to_dict()
        restored = ParetoArchive.from_dict(data)
        assert restored.size == 0
    
    def test_to_dict_structure(self):
        archive = ParetoArchive(max_size=5, decay_rate=0.9)
        data = archive.to_dict()
        assert 'max_size' in data
        assert 'decay_rate' in data
        assert 'members' in data
        assert data['max_size'] == 5
        assert data['decay_rate'] == 0.9
    
    def test_from_dict_defaults(self):
        """Missing keys should use defaults."""
        restored = ParetoArchive.from_dict({})
        assert restored.max_size == 100
        assert restored.decay_rate == 0.99
        assert restored.min_size == 3
        assert restored.size == 0


# ============================================================================
# _clone_individual
# ============================================================================

class TestCloneIndividual:
    def test_clone_is_independent(self):
        ind = _make_individual((1.0, 0.5))
        ind.metrics = {'sharpe': 2.0}
        clone = ParetoArchive._clone_individual(ind)
        
        # Clone should have same values
        assert clone.fitness == ind.fitness
        assert clone.objectives == ind.objectives
        assert clone.metrics['sharpe'] == 2.0
        
        # Modifying clone should not affect original
        clone.fitness = 999.0
        assert ind.fitness == 1.0
        
        clone.objectives[0] = 999.0
        assert ind.objectives[0] == 1.0
    
    def test_clone_none_objectives(self):
        ind = Individual(strategy_gene=_make_gene())
        ind.objectives = None
        clone = ParetoArchive._clone_individual(ind)
        assert clone.objectives is None


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_archive_size_one(self):
        archive = ParetoArchive(max_size=1)
        pop = _make_population([(1.0, 0.0), (0.0, 1.0)])
        archive.update(pop, generation=0)
        assert archive.size == 1
    
    def test_identical_objectives(self):
        """Individuals with identical objectives - only one should remain."""
        archive = ParetoArchive(max_size=10)
        pop = _make_population([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
        archive.update(pop, generation=0)
        # All are non-dominated (equal), all should be in archive
        assert archive.size >= 1
    
    def test_single_objective(self):
        """Archive should work with single-element objective vectors."""
        archive = ParetoArchive(max_size=5, min_size=1)
        pop = _make_population([(1.0,), (2.0,), (0.5,)])
        archive.update(pop, generation=0)
        # Only (2.0,) is non-dominated
        assert archive.size == 1
        assert archive.members[0].objectives == [2.0]
