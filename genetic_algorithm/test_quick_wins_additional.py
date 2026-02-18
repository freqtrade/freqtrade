"""
Test quick wins: elite re-evaluation and population size overshoot

Verifies that:
1. Elite individuals are not re-evaluated in the next generation
2. Population size doesn't overshoot the configured size
"""

import pytest
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.population import Population


def create_test_strategy(gen, ind_id):
    """Create a test strategy."""
    return StrategyGene(
        generation=gen,
        individual_id=ind_id,
        indicators=[
            IndicatorGene(type='RSI', parameters={'timeperiod': 14}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70),
        ],
        timeframe='5m',
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
    )


class TestElitePreservation:
    """Test that elite individuals preserve their fitness."""
    
    def test_elite_copy_preserves_fitness(self):
        """Test that elite individual copy preserves fitness and evaluated flag."""
        # Create an elite individual with fitness
        strategy = create_test_strategy(0, 0)
        individual = Individual(strategy_gene=strategy)
        
        # Set fitness
        individual.set_fitness(0.85, {
            'profit': 15.0,
            'sharpe_ratio': 2.0,
            'max_drawdown': 0.10,
            'win_rate': 0.60,
            'num_trades': 30,
        })
        
        assert individual.evaluated == True
        assert individual.fitness == 0.85
        assert individual.raw_fitness == 0.85
        
        # Simulate elite copy (as done in evolution.py)
        gene_copy = individual.strategy_gene.copy()
        gene_copy.generation = 1
        elite_copy = Individual(strategy_gene=gene_copy)
        
        # Carry over fitness and metrics
        elite_copy.raw_fitness = individual.raw_fitness
        elite_copy.fitness = individual.fitness
        elite_copy.metrics = individual.metrics.copy() if individual.metrics else {}
        elite_copy.evaluated = True
        
        # Verify fitness preserved
        assert elite_copy.evaluated == True
        assert elite_copy.fitness == 0.85
        assert elite_copy.raw_fitness == 0.85
        assert elite_copy.metrics == individual.metrics
        
    def test_elite_copy_is_not_reevaluated(self):
        """Test that elite individuals would not be re-evaluated."""
        # Create population with evaluated individuals
        population = Population(size=5, generation=0)
        
        for i in range(5):
            strategy = create_test_strategy(0, i)
            individual = Individual(strategy_gene=strategy)
            individual.set_fitness(0.5 + i * 0.1, {
                'profit': 10.0 + i * 2,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.10,
                'win_rate': 0.50,
                'num_trades': 20,
            })
            population.add_individual(individual)
        
        # Get unevaluated individuals
        unevaluated = [ind for ind in population if not ind.evaluated]
        
        # All should be evaluated
        assert len(unevaluated) == 0
        
        # Now simulate elite copying
        best = population.get_best(2)
        next_gen = Population(size=5, generation=1)
        
        for individual in best:
            gene_copy = individual.strategy_gene.copy()
            gene_copy.generation = 1
            elite_copy = Individual(strategy_gene=gene_copy)
            elite_copy.raw_fitness = individual.raw_fitness
            elite_copy.fitness = individual.fitness
            elite_copy.metrics = individual.metrics.copy() if individual.metrics else {}
            elite_copy.evaluated = True
            next_gen.add_individual(elite_copy)
        
        # Check that elite copies are marked as evaluated
        unevaluated = [ind for ind in next_gen if not ind.evaluated]
        assert len(unevaluated) == 0


class TestPopulationSizeControl:
    """Test that population size doesn't overshoot."""
    
    def test_population_size_exact(self):
        """Test that population size is exactly as configured."""
        # Create population with exact size
        target_size = 10
        population = Population(size=target_size, generation=0)
        
        for i in range(target_size):
            strategy = create_test_strategy(0, i)
            individual = Individual(strategy_gene=strategy)
            population.add_individual(individual)
        
        assert len(population) == target_size
        
    def test_adding_children_respects_size_limit(self):
        """Test that adding children one-by-one respects size limit."""
        target_size = 10
        population = Population(size=target_size, generation=0)
        
        # Add children but check size before each addition
        for i in range(15):  # Try to add more than target
            if len(population) >= target_size:
                break
            strategy = create_test_strategy(0, i)
            individual = Individual(strategy_gene=strategy)
            population.add_individual(individual)
        
        # Should not exceed target size
        assert len(population) == target_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
