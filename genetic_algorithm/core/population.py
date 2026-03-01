"""
Population Management

Manages a collection of individuals representing trading strategies.
"""

from typing import List, Optional
import random
from dataclasses import dataclass, field
import math

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene


@dataclass
class PopulationStats:
    """Statistics for a population at a given generation."""
    
    generation: int
    size: int
    best_fitness: Optional[float] = None  # Best shared fitness (after fitness sharing)
    worst_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    median_fitness: Optional[float] = None
    diversity_score: Optional[float] = None
    genetic_diversity: Optional[float] = None  # New: structural diversity
    best_raw_fitness: Optional[float] = None  # Best raw fitness (before fitness sharing)
    avg_raw_fitness: Optional[float] = None  # Average raw fitness
    # Holdout monitoring (populated by _run_holdout_monitoring when active)
    holdout_avg_degradation: Optional[float] = None
    holdout_best_degradation: Optional[float] = None
    holdout_num_evaluated: Optional[int] = None
    holdout_num_profitable: Optional[int] = None


def calculate_strategy_distance(ind1: Individual, ind2: Individual) -> float:
    """
    Calculate genetic distance between two strategies.
    
    Measures how different two strategies are based on:
    - Indicator types and parameters
    - Condition types and thresholds
    - Risk parameters
    
    Args:
        ind1: First individual
        ind2: Second individual
        
    Returns:
        Distance score (0 = identical, higher = more different)
    """
    gene1 = ind1.strategy_gene
    gene2 = ind2.strategy_gene
    
    distance = 0.0
    
    # Indicator type difference
    types1 = set(ind.type for ind in gene1.indicators)
    types2 = set(ind.type for ind in gene2.indicators)
    indicator_diff = len(types1.symmetric_difference(types2)) / max(len(types1), len(types2), 1)
    distance += indicator_diff * 0.3
    
    # Condition count difference
    entry_diff = abs(len(gene1.entry_conditions) - len(gene2.entry_conditions))
    exit_diff = abs(len(gene1.exit_conditions) - len(gene2.exit_conditions))
    condition_diff = (entry_diff + exit_diff) / 10.0  # Normalize
    distance += min(condition_diff, 1.0) * 0.2
    
    # Timeframe difference
    if gene1.timeframe != gene2.timeframe:
        distance += 0.2
    
    # Stoploss difference (normalized by actual range or clamped)
    # Stoploss values are negative (e.g., -0.05 to -0.25), so use absolute values
    stoploss_range = max(abs(gene1.stoploss), abs(gene2.stoploss), 0.20)
    stoploss_diff = abs(gene1.stoploss - gene2.stoploss) / stoploss_range
    distance += min(stoploss_diff, 1.0) * 0.15
    
    # Trailing stop difference
    if gene1.trailing_stop != gene2.trailing_stop:
        distance += 0.15
    
    return distance


def calculate_pairwise_distances(individuals: List[Individual]) -> List[List[float]]:
    """
    Calculate pairwise distances between all individuals.
    
    This is a helper function to avoid recomputing distances multiple times.
    Can be used by both fitness sharing and diversity calculations.
    
    Args:
        individuals: List of individuals to compute distances for
        
    Returns:
        n×n distance matrix where distances[i][j] is the distance between
        individuals[i] and individuals[j]
    """
    n = len(individuals)
    distances = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = calculate_strategy_distance(individuals[i], individuals[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    return distances


def apply_fitness_sharing(population: 'Population', sigma_share: float = 0.3, 
                         alpha: float = 1.0, distance_matrix: Optional[List[List[float]]] = None) -> None:
    """
    Apply fitness sharing to preserve diversity.
    
    Reduces fitness of individuals in crowded regions of the solution space,
    encouraging exploration of diverse strategies.
    Uses raw_fitness for calculation and stores result in fitness (shared_fitness).
    
    Args:
        population: Population to apply sharing to
        sigma_share: Sharing radius (how similar strategies must be to share)
        alpha: Sharing function exponent
        distance_matrix: Optional pre-computed distance matrix (n×n). 
                        If None, distances will be computed.
    """
    individuals = list(population.individuals)
    n = len(individuals)
    
    if n < 2:
        return
    
    # Use provided distance matrix or compute it
    if distance_matrix is None:
        distances = calculate_pairwise_distances(individuals)
    else:
        distances = distance_matrix
    
    # Calculate niche counts for each individual
    for i, individual in enumerate(individuals):
        if individual.raw_fitness is None or individual.raw_fitness <= 0:
            continue
        
        # Calculate sharing function
        niche_count = 0.0
        for j in range(n):
            dist = distances[i][j]
            if dist < sigma_share:
                # Sharing function: sh(d) = 1 - (d/sigma)^alpha
                sharing = 1.0 - (dist / sigma_share) ** alpha
                niche_count += sharing
        
        # Adjust fitness by niche count (shared fitness)
        if niche_count > 0:
            shared_fitness = individual.raw_fitness / niche_count
            individual.set_shared_fitness(shared_fitness)


def calculate_genetic_diversity(population: 'Population', 
                               distance_matrix: Optional[List[List[float]]] = None) -> float:
    """
    Calculate overall genetic diversity of the population.
    
    Higher diversity means more varied strategies, which is good for exploration.
    
    Args:
        population: Population to measure
        distance_matrix: Optional pre-computed distance matrix (n×n).
                        If None, distances will be computed.
        
    Returns:
        Diversity score (0 = no diversity, 1 = maximum diversity)
    """
    individuals = list(population.individuals)
    n = len(individuals)
    
    if n < 2:
        return 0.0
    
    # Use provided distance matrix or compute it
    if distance_matrix is None:
        distances = calculate_pairwise_distances(individuals)
    else:
        distances = distance_matrix
    
    # Calculate average pairwise distance
    total_distance = 0.0
    num_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += distances[i][j]
            num_pairs += 1
    
    if num_pairs == 0:
        return 0.0
    
    avg_distance = total_distance / num_pairs
    
    # Normalize to 0-1 range (assuming max distance ~2.0)
    diversity = min(1.0, avg_distance / 1.0)
    
    return diversity


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
    
    def get_all(self) -> List[Individual]:
        """Return all individuals in the population (list copy, same references)."""
        return list(self.individuals)

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
    
    def get_stats(self, distance_matrix: Optional[List[List[float]]] = None) -> PopulationStats:
        """
        Calculate population statistics.
        
        Args:
            distance_matrix: Optional pre-computed distance matrix for genetic diversity calculation
        
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
        
        # Calculate raw fitness stats (before fitness sharing)
        raw_fitnesses = [ind.raw_fitness for ind in evaluated if ind.raw_fitness is not None]
        best_raw_fitness = max(raw_fitnesses) if raw_fitnesses else None
        avg_raw_fitness = sum(raw_fitnesses) / len(raw_fitnesses) if raw_fitnesses else None
        
        stats = PopulationStats(
            generation=self.generation,
            size=len(self.individuals),
            best_fitness=fitnesses_sorted[-1],
            worst_fitness=fitnesses_sorted[0],
            avg_fitness=avg_fitness,
            median_fitness=fitnesses_sorted[n // 2],
            best_raw_fitness=best_raw_fitness,
            avg_raw_fitness=avg_raw_fitness,
        )
        
        # Calculate diversity (standard deviation of fitness)
        if n > 1:
            variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / n
            stats.diversity_score = variance ** 0.5
        
        # Calculate genetic diversity (structural differences)
        stats.genetic_diversity = calculate_genetic_diversity(self, distance_matrix=distance_matrix)
        
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
