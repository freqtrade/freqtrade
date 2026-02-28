"""
Parsimony Pressure — Strategy Simplification

After each generation, attempts to simplify elite strategies by removing
one indicator or condition at a time.  If removing a component does **not**
drop fitness by more than a configurable ε (epsilon), the simpler version
replaces the original.

Benefits:
- Reduces overfitting by forcing strategies to justify every component
- Smaller strategies are faster to backtest
- Results in more interpretable trading rules

Integration point: called from ``create_next_generation()`` in evolution.py
on the elite subset *after* they've been copied into the next generation.
"""

import copy
import logging
import random
from typing import Callable, Dict, Any, Optional, Tuple, List

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

logger = logging.getLogger(__name__)


def simplify_strategy(
    gene: StrategyGene,
    original_fitness: float,
    evaluate_fn: Callable[[StrategyGene], Tuple[float, Dict[str, Any]]],
    epsilon: float = 0.02,
    max_removals: int = 1,
    random_seed: Optional[int] = None,
    min_entry_conditions: int = 2,
) -> Tuple[StrategyGene, float, int]:
    """
    Try to simplify a strategy by removing components.

    Each call attempts to remove up to *max_removals* components.  For each
    candidate removal the strategy is re-evaluated; if the fitness drop is
    within *epsilon*, the simpler version is kept and the next removal
    candidate is tried on the already-simplified version.

    Removal candidates (in random order):
    1. An indicator (and all conditions that reference it)
    2. An entry condition
    3. An exit condition

    Args:
        gene: The strategy gene to simplify.
        original_fitness: Fitness of the *original* strategy (avoids a
            redundant evaluation when it has already been scored).
        evaluate_fn: ``(StrategyGene) -> (fitness, metrics)`` callable.
        epsilon: Maximum acceptable relative fitness drop (0.02 = 2%).
        max_removals: Maximum number of components to try removing.
        random_seed: Optional seed for reproducibility.

    Returns:
        ``(simplified_gene, new_fitness, n_removed)``
    """
    rng = random.Random(random_seed)
    current = gene.copy()
    current_fitness = original_fitness
    total_removed = 0

    for _ in range(max_removals):
        candidates = _build_removal_candidates(current, min_entry_conditions=min_entry_conditions)
        if not candidates:
            break

        # Shuffle so we don't always try the same order
        rng.shuffle(candidates)

        removed_this_round = False
        for kind, index in candidates:
            trial = _apply_removal(current, kind, index, min_entry_conditions=min_entry_conditions)
            if trial is None:
                continue

            try:
                trial_fitness, _ = evaluate_fn(trial)
            except Exception as e:
                logger.debug(f"[PARSIMONY] Evaluation failed during simplification: {e}")
                continue

            fitness_drop = (current_fitness - trial_fitness) / max(abs(current_fitness), 1e-9)

            if fitness_drop <= epsilon:
                logger.debug(
                    f"[PARSIMONY] Removed {kind}[{index}]: "
                    f"fitness {current_fitness:.4f} → {trial_fitness:.4f} "
                    f"(drop {fitness_drop:.2%} ≤ ε={epsilon:.2%})"
                )
                current = trial
                current_fitness = trial_fitness
                total_removed += 1
                removed_this_round = True
                break  # restart candidate list from new simplified gene

        if not removed_this_round:
            break  # no candidate was removable → done

    if total_removed > 0:
        logger.info(
            f"[PARSIMONY] Simplified strategy: removed {total_removed} component(s), "
            f"complexity {gene.calculate_complexity()} → {current.calculate_complexity()}"
        )

    return current, current_fitness, total_removed


def _build_removal_candidates(
    gene: StrategyGene,
    min_entry_conditions: int = 2,
) -> List[Tuple[str, int]]:
    """
    Build a list of ``(kind, index)`` tuples representing removable components.

    Skips removals that would make the strategy invalid (e.g. removing the
    last indicator or dropping below min_entry_conditions).
    """
    candidates: List[Tuple[str, int]] = []

    # Indicators (only if > 1 remain)
    if len(gene.indicators) > 1:
        for i in range(len(gene.indicators)):
            candidates.append(('indicator', i))

    # Entry conditions (only if above min_entry_conditions)
    if len(gene.entry_conditions) > min_entry_conditions:
        for i in range(len(gene.entry_conditions)):
            candidates.append(('entry_condition', i))

    # Exit conditions (can remove all of them — stoploss/ROI still exist)
    for i in range(len(gene.exit_conditions)):
        candidates.append(('exit_condition', i))

    return candidates


def _apply_removal(
    gene: StrategyGene, kind: str, index: int,
    min_entry_conditions: int = 2,
) -> Optional[StrategyGene]:
    """
    Return a *copy* of ``gene`` with the specified component removed, or
    ``None`` if the removal would leave the strategy in an invalid state.
    """
    trial = gene.copy()

    try:
        if kind == 'indicator':
            removed_ind = trial.indicators[index]
            ref = removed_ind.instance_id or removed_ind.type

            # Remove the indicator
            del trial.indicators[index]

            # Also remove conditions that reference this indicator
            trial.entry_conditions = [
                c for c in trial.entry_conditions if c.indicator != ref
            ]
            trial.exit_conditions = [
                c for c in trial.exit_conditions if c.indicator != ref
            ]

            # Must still have at least min_entry_conditions
            if len(trial.entry_conditions) < min_entry_conditions:
                return None

        elif kind == 'entry_condition':
            del trial.entry_conditions[index]
            if len(trial.entry_conditions) < min_entry_conditions:
                return None

        elif kind == 'exit_condition':
            del trial.exit_conditions[index]

        else:
            return None

    except IndexError:
        return None

    return trial


def apply_parsimony_to_elites(
    elites: list,
    evaluate_fn: Callable[[StrategyGene], Tuple[float, Dict[str, Any]]],
    config: Dict[str, Any],
) -> int:
    """
    Apply parsimony pressure to a list of elite ``Individual`` objects.

    Mutates elites in-place: if a simplified version is accepted, the
    elite's ``strategy_gene``, ``fitness``, and ``raw_fitness`` are updated.

    Args:
        elites: List of ``Individual`` objects (must have ``.strategy_gene``,
            ``.fitness``, ``.raw_fitness``).
        evaluate_fn: ``(StrategyGene) -> (fitness, metrics)``.
        config: Parsimony config section. Keys:
            - epsilon (float): max relative fitness drop (default 0.02)
            - max_removals (int): per-strategy removal cap (default 1)

    Returns:
        Total number of components removed across all elites.
    """
    epsilon = config.get('epsilon', 0.02)
    max_removals = config.get('max_removals', 1)
    min_entry_conditions = config.get('min_entry_conditions', 2)
    total_removed = 0

    for ind in elites:
        base_fitness = ind.raw_fitness if ind.raw_fitness is not None else ind.fitness
        if base_fitness is None or base_fitness <= 0:
            continue  # skip unfit individuals

        simplified, new_fitness, n_removed = simplify_strategy(
            ind.strategy_gene,
            base_fitness,
            evaluate_fn,
            epsilon=epsilon,
            max_removals=max_removals,
            min_entry_conditions=min_entry_conditions,
        )

        if n_removed > 0:
            ind.strategy_gene = simplified
            ind.raw_fitness = new_fitness
            ind.fitness = new_fitness
            ind.metrics = ind.metrics or {}
            ind.metrics['parsimony_removed'] = n_removed
            total_removed += n_removed

    if total_removed > 0:
        logger.info(f"[PARSIMONY] Removed {total_removed} total component(s) from elites")

    return total_removed
