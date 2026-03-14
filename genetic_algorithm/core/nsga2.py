"""
NSGA-II Multi-Objective Optimization

Implements Non-dominated Sorting Genetic Algorithm II (NSGA-II) for
evolving a portfolio of diverse trading strategies instead of a single best.

Key concepts:
- Pareto dominance: Strategy A dominates B if better in at least one objective 
  and not worse in any
- Non-dominated sorting: Assigns ranks to population (rank 1 = Pareto front)
- Crowding distance: Maintains diversity within same Pareto front

References:
- Deb, K., et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II"
  IEEE Transactions on Evolutionary Computation, 2002
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from functools import cmp_to_key

from genetic_algorithm.core.individual import Individual

logger = logging.getLogger(__name__)


def dominates(a: Individual, b: Individual) -> bool:
    """
    Check if individual 'a' Pareto-dominates individual 'b'.
    
    A dominates B if:
    - A is at least as good as B in ALL objectives
    - A is strictly better than B in at least ONE objective
    
    All objectives are assumed to be MAXIMIZED.
    
    Args:
        a: First individual
        b: Second individual
        
    Returns:
        True if a dominates b, False otherwise
    """
    if a.objectives is None or b.objectives is None:
        return False
    
    if len(a.objectives) != len(b.objectives):
        raise ValueError(f"Objective vectors must have same length: {len(a.objectives)} vs {len(b.objectives)}")
    
    at_least_as_good = True
    strictly_better = False
    
    for obj_a, obj_b in zip(a.objectives, b.objectives):
        if obj_a < obj_b:
            at_least_as_good = False
            break
        if obj_a > obj_b:
            strictly_better = True
    
    return at_least_as_good and strictly_better


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Fast non-dominated sorting algorithm from NSGA-II paper.
    
    Assigns each individual to a Pareto front (rank 1 = best front).
    
    Time complexity: O(M * N^2) where M = number of objectives, N = population size
    
    Args:
        population: List of evaluated individuals with objectives set
        
    Returns:
        List of fronts, where each front is a list of individuals.
        fronts[0] = Pareto front (rank 1), fronts[1] = rank 2, etc.
    """
    if not population:
        return []
    
    # Filter to only evaluated individuals
    evaluated = [ind for ind in population if ind.objectives is not None]
    
    if not evaluated:
        logger.warning("No individuals with objectives set for non-dominated sorting")
        return []
    
    n = len(evaluated)
    
    # S[p] = set of solutions that p dominates
    # n[p] = number of solutions that dominate p
    S: Dict[int, List[int]] = {i: [] for i in range(n)}
    domination_count: Dict[int, int] = {i: 0 for i in range(n)}
    
    # Compare all pairs
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(evaluated[i], evaluated[j]):
                S[i].append(j)  # i dominates j
            elif dominates(evaluated[j], evaluated[i]):
                domination_count[i] += 1  # j dominates i
    
    # First front: individuals that are not dominated by anyone
    fronts: List[List[Individual]] = []
    current_front_indices: List[int] = []
    
    for i in range(n):
        if domination_count[i] == 0:
            evaluated[i].rank = 1
            current_front_indices.append(i)
    
    if not current_front_indices:
        # Edge case: no individual is non-dominated (shouldn't happen in normal cases)
        # Assign all to rank 1
        logger.warning("No non-dominated individuals found, assigning all to rank 1")
        for ind in evaluated:
            ind.rank = 1
        return [evaluated]
    
    fronts.append([evaluated[i] for i in current_front_indices])
    
    # Generate subsequent fronts
    rank = 1
    while current_front_indices:
        next_front_indices: List[int] = []
        
        for i in current_front_indices:
            for j in S[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    evaluated[j].rank = rank + 1
                    next_front_indices.append(j)
        
        if next_front_indices:
            fronts.append([evaluated[j] for j in next_front_indices])
        
        current_front_indices = next_front_indices
        rank += 1
    
    logger.debug(f"Non-dominated sorting: {len(evaluated)} individuals → {len(fronts)} fronts "
                f"(front sizes: {[len(f) for f in fronts]})")
    
    return fronts


def crowding_distance_assignment(front: List[Individual]) -> None:
    """
    Calculate crowding distance for each individual in a front.
    
    Crowding distance measures how close an individual is to its neighbors.
    Individuals at the boundaries get infinite distance (always selected).
    
    This promotes diversity by preferring individuals in less crowded regions.
    
    Args:
        front: List of individuals in the same Pareto front
               (modified in place - sets crowding_distance attribute)
    """
    n = len(front)
    
    if n == 0:
        return
    
    # Initialize distances
    for ind in front:
        ind.crowding_distance = 0.0
    
    if n <= 2:
        # Boundary points get infinite distance
        for ind in front:
            ind.crowding_distance = float('inf')
        return
    
    # Get number of objectives
    num_objectives = len(front[0].objectives) if front[0].objectives else 0
    
    if num_objectives == 0:
        return
    
    # Calculate distance for each objective
    for obj_idx in range(num_objectives):
        # Sort by this objective
        sorted_front = sorted(front, key=lambda x: x.objectives[obj_idx] if x.objectives else 0)
        
        # Boundary points get infinite distance
        sorted_front[0].crowding_distance = float('inf')
        sorted_front[-1].crowding_distance = float('inf')
        
        # Objective range for normalization
        obj_min = sorted_front[0].objectives[obj_idx] if sorted_front[0].objectives else 0
        obj_max = sorted_front[-1].objectives[obj_idx] if sorted_front[-1].objectives else 0
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue  # Skip if all have same value for this objective
        
        # Calculate distance contribution for interior points
        for i in range(1, n - 1):
            if sorted_front[i].crowding_distance == float('inf'):
                continue  # Already at boundary
            
            prev_obj = sorted_front[i - 1].objectives[obj_idx] if sorted_front[i - 1].objectives else 0
            next_obj = sorted_front[i + 1].objectives[obj_idx] if sorted_front[i + 1].objectives else 0
            
            # Normalized distance contribution
            sorted_front[i].crowding_distance += (next_obj - prev_obj) / obj_range


def nsga2_tournament_selection(population: List[Individual], tournament_size: int = 2) -> Individual:
    """
    NSGA-II binary tournament selection.
    
    Selects winner based on:
    1. Lower rank (better Pareto front)
    2. If same rank, higher crowding distance (more diverse)
    
    Args:
        population: List of individuals to select from
        tournament_size: Number of individuals in tournament
        
    Returns:
        Selected individual
    """
    import random
    
    if not population:
        raise ValueError("Cannot select from empty population")
    
    tournament = random.sample(population, min(tournament_size, len(population)))
    
    # Compare using NSGA-II criteria
    def compare_nsga2(a: Individual, b: Individual) -> int:
        return a.nsga2_compare(b)
    
    # Return the best individual
    return max(tournament, key=cmp_to_key(lambda a, b: compare_nsga2(a, b)))


def nsga2_crowded_comparison_sort(population: List[Individual]) -> List[Individual]:
    """
    Sort population using NSGA-II crowded comparison operator.
    
    First by rank (ascending), then by crowding distance (descending).
    
    Args:
        population: List of individuals to sort
        
    Returns:
        Sorted list (best individuals first)
    """
    def compare(a: Individual, b: Individual) -> int:
        # Lower rank is better
        if a.rank < b.rank:
            return -1
        if a.rank > b.rank:
            return 1
        # Same rank: higher crowding distance is better
        if a.crowding_distance > b.crowding_distance:
            return -1
        if a.crowding_distance < b.crowding_distance:
            return 1
        return 0
    
    return sorted(population, key=cmp_to_key(compare))


def get_pareto_front(population: List[Individual]) -> List[Individual]:
    """
    Get the Pareto front (rank 1) from population.
    
    Args:
        population: List of individuals with objectives set
        
    Returns:
        List of non-dominated individuals
    """
    fronts = fast_non_dominated_sort(population)
    return fronts[0] if fronts else []


def calculate_hypervolume(pareto_front: List[Individual], reference_point: List[float]) -> float:
    """
    Calculate hypervolume indicator for a Pareto front.
    
    Hypervolume measures the "space" dominated by the Pareto front,
    bounded by a reference point. Higher is better.
    
    Supports 2D (staircase method) and N-dimensional (inclusion-exclusion algorithm).
    
    Args:
        pareto_front: List of non-dominated individuals
        reference_point: Reference point (should be dominated by all front points)
        
    Returns:
        Hypervolume value
    """
    if not pareto_front or not reference_point:
        return 0.0
    
    num_objectives = len(reference_point)
    
    # Filter valid points (dominate reference point in all objectives)
    valid_points = [
        ind.objectives for ind in pareto_front 
        if ind.objectives and len(ind.objectives) == num_objectives and
           all(ind.objectives[i] > reference_point[i] for i in range(num_objectives))
    ]
    
    if not valid_points:
        return 0.0
    
    if num_objectives == 2:
        return _hypervolume_2d(valid_points, reference_point)
    else:
        return _hypervolume_nd(valid_points, reference_point)


def _hypervolume_2d(points: List[List[float]], reference_point: List[float]) -> float:
    """
    Fast 2D hypervolume calculation using staircase method.
    
    Args:
        points: List of objective vectors (all dominating reference)
        reference_point: Reference point
        
    Returns:
        Hypervolume value
    """
    # Sort by first objective (descending)
    sorted_points = sorted(points, key=lambda x: -x[0])
    
    hv = 0.0
    prev_y = reference_point[1]
    
    for point in sorted_points:
        width = point[0] - reference_point[0]
        height = point[1] - prev_y
        if height > 0 and width > 0:
            hv += width * height
        prev_y = max(prev_y, point[1])
    
    return hv


def _hypervolume_nd(points: List[List[float]], reference_point: List[float]) -> float:
    """
    N-dimensional hypervolume using inclusion-exclusion algorithm.
    
    This is a simple recursive implementation suitable for small fronts (< 50 points)
    and low dimensions (2-4). For larger problems, consider using pymoo or pygmo.
    
    Args:
        points: List of objective vectors (all dominating reference)
        reference_point: Reference point
        
    Returns:
        Hypervolume value
    """
    n = len(points)
    num_obj = len(reference_point)
    
    if n == 0:
        return 0.0
    
    if n == 1:
        # Single point: hypervolume is the box from reference to point
        hv = 1.0
        for i in range(num_obj):
            hv *= max(0.0, points[0][i] - reference_point[i])
        return hv
    
    # Use HSO (Hypervolume by Slicing Objectives) approach for small N
    # Sort by first objective ascending so we sweep from reference upward;
    # each successive point extends the slice width positively.
    sorted_points = sorted(points, key=lambda x: x[0])
    
    hv = 0.0
    prev_slice = reference_point[0]
    
    for i, point in enumerate(sorted_points):
        # Current slice width in first objective
        slice_width = point[0] - prev_slice
        
        if slice_width > 0:
            # Calculate hypervolume of remaining objectives for points
            # that extend through this slice (first-objective >= point[0]).
            # Since sorted ascending, these are indices i onward.
            remaining_points = [p[1:] for p in sorted_points[i:]]
            remaining_ref = reference_point[1:]
            
            if len(remaining_ref) == 1:
                # Base case: 1D remaining
                slice_hv = max(max(p[0] for p in remaining_points) - remaining_ref[0], 0.0)
            else:
                # Recursive case: N-1 dimensions
                slice_hv = _hypervolume_nd(remaining_points, remaining_ref)
            
            hv += slice_width * slice_hv
        
        prev_slice = point[0]
    
    return hv


def extract_objectives_from_metrics(
    metrics: Dict[str, float], 
    objective_config: List[Dict[str, Any]],
    min_trades: int = 0,
) -> List[float]:
    """
    Extract objective values from metrics based on configuration.
    
    Transforms metrics to objectives where all objectives are to be MAXIMIZED.
    For "minimize" objectives, negates the value.
    
    If *min_trades* > 0 and the strategy produced fewer trades than that
    threshold, all objectives are set to worst-case values (0.0 for maximize,
    large negative for minimize).  This prevents statistically meaningless
    strategies (e.g. 1-3 trades with lucky profit) from dominating the
    Pareto front.
    
    Args:
        metrics: Dictionary of performance metrics
        objective_config: List of objective configurations, each with:
            - name: Metric name (e.g., 'profit', 'max_drawdown')
            - type: 'maximize' or 'minimize'
            - normalize: Optional normalization params
        min_trades: Minimum number of trades required for valid objectives.
                    Strategies below this get worst-case objective values.
            
    Returns:
        List of objective values (all to be maximized)
    """
    # ── Min-trades gate: penalize degenerate strategies ──
    num_trades = metrics.get('num_trades', metrics.get('trade_count', 0))
    if min_trades > 0 and num_trades < min_trades:
        # Return worst-case objectives so these individuals sink to the
        # bottom of NSGA-II ranking without being discarded entirely
        # (they can still mutate into something useful).
        worst_objectives = []
        for obj_cfg in objective_config:
            obj_type = obj_cfg.get('type', 'maximize')
            scale = obj_cfg.get('scale', 1.0)
            if obj_type == 'minimize':
                # For minimize objectives (converted to maximization via negation),
                # worst case = large penalty value (e.g. -1.0 after negation)
                worst_objectives.append(-1.0 / scale)
            else:
                worst_objectives.append(0.0)
        return worst_objectives

    objectives = []
    
    for obj_cfg in objective_config:
        name = obj_cfg['name']
        obj_type = obj_cfg.get('type', 'maximize')
        
        value = metrics.get(name, 0.0)
        
        # Convert to maximization
        if obj_type == 'minimize':
            value = -value
        elif obj_type == 'goldilocks':
            # For goldilocks objectives (trade frequency), calculate distance from target
            target = obj_cfg.get('target', 50)
            tolerance = obj_cfg.get('tolerance', 25)
            distance = abs(value - target)
            # Convert to maximization: closer to target = higher value
            value = max(0, 1.0 - distance / tolerance)
        
        # Optional normalization
        if 'scale' in obj_cfg:
            value = value / obj_cfg['scale']
        
        objectives.append(value)
    
    return objectives


# Default objective configuration for trading strategies
DEFAULT_OBJECTIVES = [
    {'name': 'profit', 'type': 'maximize', 'scale': 100.0},  # Normalize profit to ~0-1
    {'name': 'max_drawdown', 'type': 'minimize', 'scale': 1.0},  # -drawdown (lower is better)
    {'name': 'sharpe_ratio', 'type': 'maximize', 'scale': 3.0},  # Normalize sharpe to ~0-1
]
