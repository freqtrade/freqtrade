"""
Pareto Archive with Crowding-Distance Decay

Maintains an external archive of historically non-dominated individuals across
generations. The archive is updated each generation: new Pareto-optimal
individuals are added, individuals that become dominated are removed, and when
the archive exceeds its capacity the *most crowded* individuals are pruned.

Benefits:
- Preserves diversity across generations (the main population's Pareto front
  can shift dramatically between generations; the archive smooths this).
- Crowding-distance-based pruning naturally steers the archive towards the
  "knee" region of the Pareto front — the area of best trade-off — while
  still keeping boundary points.
- After evolution, the archive can be used as the final solution set (more
  stable than the last generation's Pareto front alone).

Integration: instantiated in ``evolve()`` when NSGA-II mode is active,
updated once per generation.
"""

import logging
import copy
from typing import List, Dict, Any, Optional

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.nsga2 import (
    dominates,
    fast_non_dominated_sort,
    crowding_distance_assignment,
)

logger = logging.getLogger(__name__)


class ParetoArchive:
    """
    External Pareto archive with crowding-distance decay.

    Keeps up to ``max_size`` non-dominated individuals. When new members
    push the archive over capacity, members with the *smallest* crowding
    distance (most redundant) are removed first.
    """

    def __init__(self, max_size: int = 100, decay_rate: float = 0.99, min_size: int = 3):
        """
        Args:
            max_size: Maximum archive capacity.
            decay_rate: Multiplicative decay applied to crowding distances
                each generation. A value < 1.0 lets long-lived archive
                members gradually lose their "novelty bonus", making room
                for fresh solutions. Set to 1.0 to disable decay.
                Default 0.99 (was 0.95 — reduced to prevent premature
                archive collapse observed in benchmarks).
            min_size: Minimum archive floor. If the rank-1 Pareto front is
                smaller than this, rank-2 members are included to prevent
                archive collapse to a single individual.
        """
        self.max_size = max(1, max_size)
        self.min_size = max(1, min_size)
        self.decay_rate = max(0.0, min(1.0, decay_rate))
        self.members: List[Individual] = []
        self._generation_added: Dict[int, int] = {}  # id(ind) -> generation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, population: List[Individual], generation: int = 0) -> None:
        """
        Merge the current population into the archive.

        Steps:
        1. Apply crowding-distance decay to existing members.
        2. Pool existing archive + new candidates.
        3. Run non-dominated sort on the pool.
        4. Keep only rank-1 (Pareto front).
        5. If over capacity, prune lowest-crowding members.

        Args:
            population: Current generation's population.
            generation: The current generation number (for logging).
        """
        # Step 1: decay existing members' crowding distances
        if self.decay_rate < 1.0:
            for m in self.members:
                if m.crowding_distance is not None and m.crowding_distance != float('inf'):
                    m.crowding_distance *= self.decay_rate

        # Step 2: build a candidate pool
        # Deep-copy new individuals so archive is decoupled from the population
        new_candidates = []
        for ind in population:
            if ind.objectives is None:
                continue
            clone = self._clone_individual(ind)
            new_candidates.append(clone)

        pool = list(self.members) + new_candidates

        if not pool:
            return

        # Step 3: non-dominated sort
        fronts = fast_non_dominated_sort(pool)
        if not fronts:
            return

        # Step 4: keep rank-1 (and rank-2 if needed to meet min_size)
        rank1 = fronts[0]
        candidates = list(rank1)
        if len(candidates) < self.min_size and len(fronts) > 1:
            # Include rank-2 members to prevent archive collapse
            rank2 = fronts[1]
            candidates.extend(rank2)
            logger.info(
                f"[ARCHIVE] rank-1 too small ({len(rank1)}), "
                f"added {len(rank2)} rank-2 members (min_size={self.min_size})"
            )

        # Step 5: prune to capacity
        self.members = self._prune(candidates)

        # Track generation for new entrants
        for m in new_candidates:
            if m in self.members:
                self._generation_added[id(m)] = generation

        logger.info(
            f"[ARCHIVE] gen={generation}: archive size={len(self.members)}, "
            f"Pareto front={len(rank1)}, pool={len(pool)}"
        )

    def get_archive(self) -> List[Individual]:
        """Return a copy of the current archive members."""
        return list(self.members)

    def get_best(self, n: int = 1) -> List[Individual]:
        """
        Return the *n* archive members with the highest crowding distance
        (the most "unique" solutions on the Pareto front).
        """
        sorted_members = sorted(
            self.members,
            key=lambda ind: ind.crowding_distance if ind.crowding_distance is not None else 0.0,
            reverse=True,
        )
        return sorted_members[:n]

    @property
    def size(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize archive for checkpoint storage."""
        return {
            'max_size': self.max_size,
            'min_size': self.min_size,
            'decay_rate': self.decay_rate,
            'members': [m.to_dict() for m in self.members],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParetoArchive':
        """Restore archive from checkpoint data."""
        archive = cls(
            max_size=data.get('max_size', 100),
            decay_rate=data.get('decay_rate', 0.99),
            min_size=data.get('min_size', 3),
        )
        for m_data in data.get('members', []):
            archive.members.append(Individual.from_dict(m_data))
        return archive

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune(self, members: List[Individual]) -> List[Individual]:
        """
        Prune a set of members to ``max_size`` using crowding distance.

        Recalculates crowding distances, then iteratively removes the
        member with the smallest crowding distance until within capacity.
        """
        if len(members) <= self.max_size:
            crowding_distance_assignment(members)
            return members

        # Recalculate crowding distances for the full set
        crowding_distance_assignment(members)

        # Iteratively remove worst-crowding member
        remaining = list(members)
        while len(remaining) > self.max_size:
            # Find member with smallest crowding distance
            worst_idx = 0
            worst_cd = remaining[0].crowding_distance if remaining[0].crowding_distance is not None else 0.0
            for i, m in enumerate(remaining[1:], 1):
                cd = m.crowding_distance if m.crowding_distance is not None else 0.0
                if cd < worst_cd:
                    worst_cd = cd
                    worst_idx = i
            remaining.pop(worst_idx)

            # Recalculate crowding distances after removal
            if len(remaining) > self.max_size:
                crowding_distance_assignment(remaining)

        return remaining

    @staticmethod
    def _clone_individual(ind: Individual) -> Individual:
        """Create a deep clone of an Individual for archive storage."""
        clone = Individual(strategy_gene=ind.strategy_gene.copy())
        clone.fitness = ind.fitness
        clone.raw_fitness = ind.raw_fitness
        clone.objectives = list(ind.objectives) if ind.objectives else None
        clone.rank = ind.rank
        clone.crowding_distance = ind.crowding_distance
        clone.metrics = dict(ind.metrics) if ind.metrics else {}
        clone.evaluated = ind.evaluated
        return clone
