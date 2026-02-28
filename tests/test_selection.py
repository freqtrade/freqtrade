"""
Tests for Selection Operators

Tests for tournament, roulette wheel, rank-based, and NSGA-II selection methods.
"""

import pytest
import random
from typing import List

from genetic_algorithm.core.selection import (
    tournament_selection,
    roulette_wheel_selection,
    rank_based_selection,
    nsga2_tournament_selection,
    select_parents,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.population import Population


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def simple_strategy_gene():
    """Create a minimal StrategyGene for testing."""
    return StrategyGene(
        generation=1,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70, logic='AND'),
        ],
        timeframe='5m',
        stoploss=-0.05,
    )


@pytest.fixture
def population_with_fitness(simple_strategy_gene) -> Population:
    """Create a population with varied fitness values."""
    population = Population(size=10)
    
    for i in range(10):
        gene = simple_strategy_gene.copy()
        gene.individual_id = i
        ind = Individual(strategy_gene=gene)
        ind.fitness = (i + 1) * 0.1  # 0.1, 0.2, ..., 1.0
        ind.raw_fitness = ind.fitness
        ind.evaluated = True
        population.add_individual(ind)
    
    return population


@pytest.fixture
def population_with_negative_fitness(simple_strategy_gene) -> Population:
    """Create a population with some negative fitness values."""
    population = Population(size=10)
    
    for i in range(10):
        gene = simple_strategy_gene.copy()
        gene.individual_id = i
        ind = Individual(strategy_gene=gene)
        ind.fitness = (i - 5) * 0.2  # -1.0, -0.8, ..., 0.8
        ind.raw_fitness = ind.fitness
        ind.evaluated = True
        population.add_individual(ind)
    
    return population


@pytest.fixture
def population_with_ties(simple_strategy_gene) -> Population:
    """Create a population with tied fitness values."""
    population = Population(size=6)
    
    # 3 individuals with low fitness, 3 with high fitness
    for i in range(6):
        gene = simple_strategy_gene.copy()
        gene.individual_id = i
        ind = Individual(strategy_gene=gene)
        ind.fitness = 0.3 if i < 3 else 0.7  # Tied groups
        ind.raw_fitness = ind.fitness
        ind.evaluated = True
        population.add_individual(ind)
    
    return population


@pytest.fixture
def population_single_individual(simple_strategy_gene) -> Population:
    """Create a population with only one individual."""
    population = Population(size=1)
    
    ind = Individual(strategy_gene=simple_strategy_gene)
    ind.fitness = 0.5
    ind.raw_fitness = 0.5
    ind.evaluated = True
    population.add_individual(ind)
    
    return population


@pytest.fixture
def population_with_objectives(simple_strategy_gene) -> Population:
    """Create a population with NSGA-II objectives."""
    population = Population(size=10)
    
    for i in range(10):
        gene = simple_strategy_gene.copy()
        gene.individual_id = i
        ind = Individual(strategy_gene=gene)
        ind.fitness = (i + 1) * 0.1
        # Objectives: [profit, drawdown (negated for max)]
        ind.objectives = [(i + 1) * 5.0, -0.1 - i * 0.02]
        ind.rank = i // 3  # Assign ranks 0, 0, 0, 1, 1, 1, 2, 2, 2, 3
        ind.crowding_distance = random.uniform(0.1, 2.0)
        ind.evaluated = True
        population.add_individual(ind)
    
    return population


# =============================================================================
# TOURNAMENT SELECTION TESTS
# =============================================================================

class TestTournamentSelection:
    """Tests for tournament selection."""
    
    def test_returns_individual(self, population_with_fitness):
        """Tournament selection returns an Individual."""
        random.seed(42)
        result = tournament_selection(population_with_fitness, tournament_size=3)
        assert isinstance(result, Individual)
        assert result in population_with_fitness.individuals
    
    def test_best_individual_wins_tournament_consistently(self, population_with_fitness):
        """With full population tournament, best should always win."""
        # Tournament size = population size means we always pick the best
        result = tournament_selection(population_with_fitness, tournament_size=10)
        best = max(population_with_fitness.individuals, key=lambda x: x.fitness)
        assert result.fitness == best.fitness
    
    def test_handles_ties(self, population_with_ties):
        """Tournament selection handles tied fitness values."""
        random.seed(42)
        result = tournament_selection(population_with_ties, tournament_size=3)
        assert result in population_with_ties.individuals
    
    def test_single_individual_population(self, population_single_individual):
        """Works with a single individual."""
        result = tournament_selection(population_single_individual, tournament_size=3)
        assert result == population_single_individual.individuals[0]
    
    def test_tournament_size_larger_than_population(self, population_single_individual):
        """Handles tournament size larger than population gracefully."""
        result = tournament_selection(population_single_individual, tournament_size=100)
        assert result == population_single_individual.individuals[0]
    
    def test_different_seeds_different_results(self, population_with_fitness):
        """Different random seeds can produce different selections."""
        random.seed(1)
        result1 = tournament_selection(population_with_fitness, tournament_size=2)
        random.seed(2)
        result2 = tournament_selection(population_with_fitness, tournament_size=2)
        # Not guaranteed to be different, but likely with small tournament
        # Just verify both are valid
        assert result1 in population_with_fitness.individuals
        assert result2 in population_with_fitness.individuals


# =============================================================================
# ROULETTE WHEEL SELECTION TESTS
# =============================================================================

class TestRouletteWheelSelection:
    """Tests for roulette wheel (fitness-proportionate) selection."""
    
    def test_returns_individual(self, population_with_fitness):
        """Roulette selection returns an Individual."""
        random.seed(42)
        result = roulette_wheel_selection(population_with_fitness)
        assert isinstance(result, Individual)
        assert result in population_with_fitness.individuals
    
    def test_handles_negative_fitness(self, population_with_negative_fitness):
        """Roulette selection handles negative fitness by shifting values."""
        random.seed(42)
        result = roulette_wheel_selection(population_with_negative_fitness)
        assert result in population_with_negative_fitness.individuals
    
    def test_empty_evaluated_raises_error(self, simple_strategy_gene):
        """Raises error when no individuals are evaluated."""
        population = Population(size=5)
        for i in range(5):
            gene = simple_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.fitness = None  # Not evaluated
            population.add_individual(ind)
        
        with pytest.raises(ValueError, match="No evaluated individuals"):
            roulette_wheel_selection(population)
    
    def test_single_individual(self, population_single_individual):
        """Works with single individual."""
        result = roulette_wheel_selection(population_single_individual)
        assert result == population_single_individual.individuals[0]
    
    def test_higher_fitness_selected_more_often(self, population_with_fitness):
        """Higher fitness individuals should be selected more frequently."""
        random.seed(42)
        selections = [roulette_wheel_selection(population_with_fitness) for _ in range(1000)]
        
        # Count selections by fitness
        high_fitness_count = sum(1 for s in selections if s.fitness > 0.5)
        low_fitness_count = sum(1 for s in selections if s.fitness <= 0.5)
        
        # Higher fitness should be selected more often
        assert high_fitness_count > low_fitness_count
    
    def test_all_zero_fitness(self, simple_strategy_gene):
        """Handles population where all fitness values are zero."""
        population = Population(size=5)
        for i in range(5):
            gene = simple_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.fitness = 0.0
            ind.evaluated = True
            population.add_individual(ind)
        
        # Should select randomly when all fitness is 0
        result = roulette_wheel_selection(population)
        assert result in population.individuals


# =============================================================================
# RANK-BASED SELECTION TESTS
# =============================================================================

class TestRankBasedSelection:
    """Tests for rank-based selection."""
    
    def test_returns_individual(self, population_with_fitness):
        """Rank selection returns an Individual."""
        random.seed(42)
        result = rank_based_selection(population_with_fitness)
        assert isinstance(result, Individual)
        assert result in population_with_fitness.individuals
    
    def test_handles_negative_fitness(self, population_with_negative_fitness):
        """Rank selection works with negative fitness values."""
        random.seed(42)
        result = rank_based_selection(population_with_negative_fitness)
        assert result in population_with_negative_fitness.individuals
    
    def test_empty_evaluated_raises_error(self, simple_strategy_gene):
        """Raises error when no individuals are evaluated."""
        population = Population(size=5)
        for i in range(5):
            gene = simple_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.fitness = None
            population.add_individual(ind)
        
        with pytest.raises(ValueError, match="No evaluated individuals"):
            rank_based_selection(population)
    
    def test_single_individual(self, population_single_individual):
        """Works with single individual."""
        result = rank_based_selection(population_single_individual)
        assert result == population_single_individual.individuals[0]
    
    def test_higher_rank_selected_more_often(self, population_with_fitness):
        """Higher ranked (better fitness) individuals selected more often."""
        random.seed(42)
        selections = [rank_based_selection(population_with_fitness) for _ in range(1000)]
        
        # Count selections by fitness ranking
        high_fitness_count = sum(1 for s in selections if s.fitness > 0.5)
        low_fitness_count = sum(1 for s in selections if s.fitness <= 0.5)
        
        # Higher ranked should be selected more often
        assert high_fitness_count > low_fitness_count


# =============================================================================
# NSGA-II TOURNAMENT SELECTION TESTS
# =============================================================================

class TestNSGA2TournamentSelection:
    """Tests for NSGA-II binary tournament selection."""
    
    def test_returns_individual(self, population_with_objectives):
        """NSGA-II selection returns an Individual."""
        random.seed(42)
        result = nsga2_tournament_selection(population_with_objectives, tournament_size=2)
        assert isinstance(result, Individual)
        assert result in population_with_objectives.individuals
    
    def test_prefers_lower_rank(self, population_with_objectives):
        """Should prefer individuals with lower Pareto rank."""
        random.seed(42)
        selections = [nsga2_tournament_selection(population_with_objectives, tournament_size=2) 
                      for _ in range(100)]
        
        # Most selections should be from rank 0 or 1
        low_rank_count = sum(1 for s in selections if s.rank <= 1)
        assert low_rank_count > 50  # More than half should be low rank
    
    def test_fallback_to_tournament_when_no_objectives(self, population_with_fitness):
        """Falls back to regular tournament when no objectives exist."""
        random.seed(42)
        result = nsga2_tournament_selection(population_with_fitness, tournament_size=2)
        assert result in population_with_fitness.individuals
    
    def test_single_individual(self, simple_strategy_gene):
        """Works with single individual with objectives."""
        population = Population(size=1)
        ind = Individual(strategy_gene=simple_strategy_gene)
        ind.objectives = [10.0, -0.1]
        ind.rank = 0
        ind.crowding_distance = 1.0
        ind.evaluated = True
        population.add_individual(ind)
        
        result = nsga2_tournament_selection(population, tournament_size=2)
        assert result == ind


# =============================================================================
# SELECT_PARENTS TESTS
# =============================================================================

class TestSelectParents:
    """Tests for the unified select_parents function."""
    
    def test_tournament_method(self, population_with_fitness):
        """select_parents works with tournament method."""
        random.seed(42)
        parents = select_parents(population_with_fitness, num_parents=4, method='tournament')
        assert len(parents) == 4
        assert all(p in population_with_fitness.individuals for p in parents)
    
    def test_roulette_method(self, population_with_fitness):
        """select_parents works with roulette method."""
        random.seed(42)
        parents = select_parents(population_with_fitness, num_parents=4, method='roulette')
        assert len(parents) == 4
        assert all(p in population_with_fitness.individuals for p in parents)
    
    def test_rank_method(self, population_with_fitness):
        """select_parents works with rank method."""
        random.seed(42)
        parents = select_parents(population_with_fitness, num_parents=4, method='rank')
        assert len(parents) == 4
        assert all(p in population_with_fitness.individuals for p in parents)
    
    def test_nsga2_method(self, population_with_objectives):
        """select_parents works with nsga2 method."""
        random.seed(42)
        parents = select_parents(population_with_objectives, num_parents=4, method='nsga2')
        assert len(parents) == 4
        assert all(p in population_with_objectives.individuals for p in parents)
    
    def test_unknown_method_raises_error(self, population_with_fitness):
        """Unknown selection method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown selection method"):
            select_parents(population_with_fitness, num_parents=2, method='unknown')
    
    def test_allow_duplicates_true(self, population_single_individual):
        """With allow_duplicates=True, can select same individual multiple times."""
        parents = select_parents(population_single_individual, num_parents=5, 
                                 method='tournament', allow_duplicates=True)
        assert len(parents) == 5
        # All should be the same individual
        assert all(p == population_single_individual.individuals[0] for p in parents)
    
    def test_allow_duplicates_false(self, population_with_fitness):
        """With allow_duplicates=False, tries to select unique parents."""
        random.seed(42)
        parents = select_parents(population_with_fitness, num_parents=5, 
                                 method='tournament', allow_duplicates=False)
        assert len(parents) == 5
        # Should have mostly unique parents
        unique_count = len(set(id(p) for p in parents))
        assert unique_count >= 3  # At least some uniqueness
    
    def test_tournament_size_kwarg(self, population_with_fitness):
        """Tournament size can be passed via kwargs."""
        # Full tournament = always select best
        parents = select_parents(population_with_fitness, num_parents=3, 
                                 method='tournament', tournament_size=10)
        best = max(population_with_fitness.individuals, key=lambda x: x.fitness)
        assert all(p.fitness == best.fitness for p in parents)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestSelectionEdgeCases:
    """Tests for edge cases across selection methods."""
    
    def test_all_same_fitness(self, simple_strategy_gene):
        """All methods work when all individuals have same fitness."""
        population = Population(size=5)
        for i in range(5):
            gene = simple_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.fitness = 0.5  # All same
            ind.evaluated = True
            population.add_individual(ind)
        
        random.seed(42)
        
        # All methods should work
        assert tournament_selection(population, 2) in population.individuals
        assert roulette_wheel_selection(population) in population.individuals
        assert rank_based_selection(population) in population.individuals
    
    def test_extreme_fitness_values(self, simple_strategy_gene):
        """Selection handles extreme fitness values."""
        population = Population(size=3)
        
        for i, fitness in enumerate([-1000.0, 0.0, 1000.0]):
            gene = simple_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.fitness = fitness
            ind.evaluated = True
            population.add_individual(ind)
        
        random.seed(42)
        
        # All methods should handle extreme values
        assert tournament_selection(population, 2) in population.individuals
        assert roulette_wheel_selection(population) in population.individuals
        assert rank_based_selection(population) in population.individuals
    
    def test_very_small_fitness_differences(self, simple_strategy_gene):
        """Selection works with very small fitness differences."""
        population = Population(size=5)
        
        for i in range(5):
            gene = simple_strategy_gene.copy()
            gene.individual_id = i
            ind = Individual(strategy_gene=gene)
            ind.fitness = 0.5 + i * 0.0001  # Very small differences
            ind.evaluated = True
            population.add_individual(ind)
        
        random.seed(42)
        
        # All methods should work
        assert tournament_selection(population, 2) in population.individuals
        assert roulette_wheel_selection(population) in population.individuals
        assert rank_based_selection(population) in population.individuals
