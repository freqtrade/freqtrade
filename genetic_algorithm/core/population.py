"""
Population Management

Manages a collection of individuals representing trading strategies.
"""

from typing import List, Optional
import random
from dataclasses import dataclass, field

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene


@dataclass
class PopulationStats:
    """Statistics for a population at a given generation."""
    
    generation: int
    size: int
    best_fitness: Optional[float] = None
    worst_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None


class Population:
    """
    Manages a population of strategy individuals.
    
    Provides methods for initialization, sorting, selection,
    and tracking population statistics.
    """
    
    def __init__(self, size: int, generation: int = 0):
        """
        Initialize a population.
        
        Args:
            size: Number of individuals in the population
            generation: Current generation number
        """
        self.size = size
        self.generation = generation
        self.individuals: List[Individual] = []
    
    def add_individual(self, individual: Individual):
        """Add an individual to the population."""
        self.individuals.append(individual)
    
    def remove_individual(self, individual: Individual):
        """Remove an individual from the population."""
        self.individuals.remove(individual)
    
    def sort_by_fitness(self, reverse: bool = True):
        """
        Sort population by fitness.
        
        Args:
            reverse: If True, sort descending (best first)
        """
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
                            reverse=reverse)
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """
        Get the top N individuals.
        
        Args:
            n: Number of top individuals to return
            
        Returns:
            List of top N individuals
        """
        self.sort_by_fitness(reverse=True)
        return self.individuals[:n]
    
    def get_worst(self, n: int = 1) -> List[Individual]:
        """
        Get the bottom N individuals.
        
        Args:
            n: Number of bottom individuals to return
            
        Returns:
            List of bottom N individuals
        """
        self.sort_by_fitness(reverse=False)
        return self.individuals[:n]
    
    def get_stats(self) -> PopulationStats:
        """
        Calculate population statistics.
        
        Returns:
            PopulationStats object with current statistics
        """
        evaluated = [ind for ind in self.individuals if ind.fitness is not None]
        
        if not evaluated:
            return PopulationStats(generation=self.generation, size=len(self.individuals))
        
        # Create sorted copy of fitnesses for median/min/max calculations
        fitnesses = [ind.fitness for ind in evaluated]
        fitnesses_sorted = sorted(fitnesses)
        n = len(fitnesses_sorted)
        
        avg_fitness = sum(fitnesses) / n
        stats = PopulationStats(
            generation=self.generation,
            size=len(self.individuals),
            best_fitness=fitnesses_sorted[-1],
            worst_fitness=fitnesses_sorted[0],
            avg_fitness=avg_fitness,
            median_fitness=fitnesses_sorted[n // 2],
        )
        
        # Calculate diversity (standard deviation of fitness)
        if n > 1:
            variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / n
            stats.diversity_score = variance ** 0.5
        
        return stats
    
    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return len(self.individuals)
    
    def __iter__(self):
        """Iterate over individuals."""
        return iter(self.individuals)
    
    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]
