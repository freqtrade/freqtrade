"""
Crossover Operators

Implements various crossover strategies for combining two parent
strategies to create offspring.
"""

import copy
import random
from typing import Tuple

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.core.mutation import clamp_condition_thresholds


def _deduplicate_indicators(gene: StrategyGene) -> None:
    """Remove duplicate indicators (same type + params) keeping the first."""
    seen = set()
    deduped = []
    for ind in gene.indicators:
        key = (ind.type, str(sorted(ind.parameters.items())) if ind.parameters else '')
        if key not in seen:
            seen.add(key)
            deduped.append(ind)
    gene.indicators = deduped


def _enforce_max_indicators(gene: StrategyGene, config: dict) -> None:
    """Trim indicators to max_per_strategy, removing lowest-weight first.
    
    After trimming, removes any conditions that reference indicators
    no longer present (orphaned conditions).
    """
    max_indicators = (config or {}).get('indicators', {}).get('max_per_strategy', 6)
    if len(gene.indicators) <= max_indicators:
        return
    
    # Sort by weight descending (higher weight = more important = keep)
    # Indicators without an explicit weight default to 1.0
    gene.indicators.sort(key=lambda ind: getattr(ind, 'weight', 1.0), reverse=True)
    gene.indicators = gene.indicators[:max_indicators]
    
    # Build set of remaining indicator references.
    # Conditions may reference indicators by instance_id ('RSI_0') or by bare
    # type ('RSI'), so we must match against both to avoid orphaning valid
    # conditions when instance_id is None or when conditions predate instance_id
    # assignment.
    remaining_refs = set()
    for ind in gene.indicators:
        if ind.instance_id:
            remaining_refs.add(ind.instance_id)
        remaining_refs.add(ind.type)
    
    # Remove orphaned conditions (reference indicators we just trimmed)
    gene.entry_conditions = [c for c in gene.entry_conditions if c.indicator in remaining_refs]
    gene.exit_conditions = [c for c in gene.exit_conditions if c.indicator in remaining_refs]
    
    # Ensure at least one entry condition remains
    if not gene.entry_conditions and gene.indicators:
        from genetic_algorithm.core.mutation import _create_random_condition
        indicator_config = (config or {}).get('indicators', {})
        ind = gene.indicators[0]
        try:
            new_cond = _create_random_condition(ind.type, True, indicator_config)
            if new_cond:
                new_cond.indicator = ind.instance_id or ind.type
                gene.entry_conditions = [new_cond]
        except Exception:
            pass


def _deduplicate_conditions(conditions: list) -> list:
    """Remove exact-duplicate conditions and prune subsumed pairs.
    
    Two conditions on the same indicator with the same operator where one
    threshold is strictly tighter than the other: keep only the tighter one.
    E.g. 'vroc < -117' AND 'vroc < -200' → keep 'vroc < -200'.
    """
    # 1. Exact dedup
    seen = set()
    unique = []
    for c in conditions:
        key = (c.indicator, c.operator, round(c.threshold, 6), c.logic)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    # 2. Subsumption pruning for '<' / '>' operators
    #    Only safe when ALL conditions in a group share AND logic.
    #    Under OR logic, the *looser* condition dominates, which inverts
    #    the selection — skipping subsumption avoids silent signal loss.
    result = []
    by_ind_op = {}  # (indicator, operator) → list of conditions
    for c in unique:
        if c.operator in ('<', 'less_than'):
            by_ind_op.setdefault((c.indicator, '<'), []).append(c)
        elif c.operator in ('>', 'greater_than'):
            by_ind_op.setdefault((c.indicator, '>'), []).append(c)
        else:
            result.append(c)  # keep non-comparable operators as-is
    
    for (ind, op), conds in by_ind_op.items():
        # If any condition uses OR logic, subsumption is unsafe — keep all
        if any(getattr(c, 'logic', 'AND') == 'OR' for c in conds):
            result.extend(conds)
            continue

        if op == '<':
            # For AND logic: x < A AND x < B → keep min(A, B)
            keeper = min(conds, key=lambda c: c.threshold)
        else:
            # For AND logic: x > A AND x > B → keep max(A, B)
            keeper = max(conds, key=lambda c: c.threshold)
        result.append(keeper)
    
    return result


def _enforce_min_entry_conditions(gene: StrategyGene, config: dict) -> None:
    """Ensure a strategy gene has at least min_entry_conditions entry conditions
    AND at least min_exit_conditions exit conditions.
    
    If the gene has fewer, generates additional random conditions from
    the available indicators.
    """
    if not config:
        return
    indicator_config = config.get('indicators', {})
    min_entry = indicator_config.get('min_entry_conditions', 2)
    min_exit = indicator_config.get('min_exit_conditions', 1)
    
    # Enforce entry conditions
    if len(gene.entry_conditions) < min_entry:
        _top_up_conditions(gene, min_entry - len(gene.entry_conditions), 
                          is_entry=True, indicator_config=indicator_config)
    
    # Enforce exit conditions
    if len(gene.exit_conditions) < min_exit:
        _top_up_conditions(gene, min_exit - len(gene.exit_conditions),
                          is_entry=False, indicator_config=indicator_config)


def _top_up_conditions(gene: StrategyGene, needed: int, is_entry: bool, 
                       indicator_config: dict) -> None:
    """Add random conditions to meet the minimum requirement."""
    from genetic_algorithm.core.mutation import _create_random_condition
    available_indicators = gene.indicators
    if not available_indicators:
        return
    
    conditions = gene.entry_conditions if is_entry else gene.exit_conditions
    attempts = 0
    added = 0
    while added < needed and attempts < 10:
        ind = random.choice(available_indicators)
        ind_type = ind.type  # _create_random_condition expects the type (e.g. 'RSI')
        ind_ref = ind.instance_id or ind_type  # conditions reference instance_id
        try:
            new_cond = _create_random_condition(ind_type, is_entry, indicator_config)
            if new_cond:
                # Update the condition to reference instance_id, not bare type
                new_cond.indicator = ind_ref
                existing_keys = {(c.indicator, c.operator, str(c.threshold)) for c in conditions}
                new_key = (new_cond.indicator, new_cond.operator, str(new_cond.threshold))
                if new_key not in existing_keys:
                    conditions.append(new_cond)
                    added += 1
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create random condition for {ind_ref}: {e}")
        attempts += 1


def single_point_crossover(parent1: Individual, parent2: Individual, 
                          generation: int, ind_id: int,
                          config: dict = None) -> Tuple[Individual, Individual]:
    """
    Single-point crossover.
    
    Split both parents at a random point and swap the second parts.
    
    Note: After crossover, `ensure_indicators_for_conditions()` is called to
    ensure all conditions have their required indicators. This may ADD indicators
    that were not present in either parent if conditions reference missing 
    indicator types. This is intentional to maintain strategy validity.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        config: Optional config dict for indicator setup
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Crossover indicators
    if len(parent1.strategy_gene.indicators) > 1 and len(parent2.strategy_gene.indicators) > 1:
        point = random.randint(1, min(len(parent1.strategy_gene.indicators),
                                     len(parent2.strategy_gene.indicators)) - 1)
        child1_gene.indicators = ([copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators[:point]] + 
                                 [copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators[point:]])
        child2_gene.indicators = ([copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators[:point]] + 
                                 [copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators[point:]])
    
    # Crossover entry conditions
    if len(parent1.strategy_gene.entry_conditions) > 1 and len(parent2.strategy_gene.entry_conditions) > 1:
        point = random.randint(1, min(len(parent1.strategy_gene.entry_conditions),
                                     len(parent2.strategy_gene.entry_conditions)) - 1)
        child1_gene.entry_conditions = ([copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions[:point]] + 
                                       [copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions[point:]])
        child2_gene.entry_conditions = ([copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions[:point]] + 
                                       [copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions[point:]])
    
    # Crossover exit conditions
    if (len(parent1.strategy_gene.exit_conditions) > 1 and 
        len(parent2.strategy_gene.exit_conditions) > 1):
        point = random.randint(1, min(len(parent1.strategy_gene.exit_conditions),
                                     len(parent2.strategy_gene.exit_conditions)) - 1)
        child1_gene.exit_conditions = ([copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions[:point]] + 
                                      [copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions[point:]])
        child2_gene.exit_conditions = ([copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions[:point]] + 
                                      [copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions[point:]])
    
    # Randomly inherit scalar parameters
    for attr in ['timeframe', 'stoploss']:
        if random.random() < 0.5:
            setattr(child1_gene, attr, getattr(parent2.strategy_gene, attr))
            setattr(child2_gene, attr, getattr(parent1.strategy_gene, attr))
    
    # Swap informative_timeframes along with timeframe for consistency
    if random.random() < 0.5:
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if random.random() < 0.5:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Ensure all indicators referenced in conditions are calculated
    if config:
        indicator_config = config.get('indicators', {})
        child1_gene.ensure_indicators_for_conditions(indicator_config)
        child2_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after crossover to avoid ID conflicts
    child1_gene.assign_instance_ids()
    child2_gene.assign_instance_ids()
    
    # Enforce minimum entry conditions
    _enforce_min_entry_conditions(child1_gene, config)
    _enforce_min_entry_conditions(child2_gene, config)
    
    # Post-crossover quality: dedup indicators, dedup/prune conditions, clamp thresholds
    for g in (child1_gene, child2_gene):
        _deduplicate_indicators(g)
        _enforce_max_indicators(g, config)
        g.entry_conditions = _deduplicate_conditions(g.entry_conditions)
        g.exit_conditions = _deduplicate_conditions(g.exit_conditions)
        clamp_condition_thresholds(g.entry_conditions)
        clamp_condition_thresholds(g.exit_conditions)
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def _uniform_crossover_lists(parent1_list, parent2_list, swap_prob):
    """Helper for uniform crossover on lists."""
    child1_list, child2_list = [], []
    max_len = max(len(parent1_list), len(parent2_list))
    
    for i in range(max_len):
        if random.random() < swap_prob:
            if i < len(parent2_list):
                child1_list.append(copy.deepcopy(parent2_list[i]))
            if i < len(parent1_list):
                child2_list.append(copy.deepcopy(parent1_list[i]))
        else:
            if i < len(parent1_list):
                child1_list.append(copy.deepcopy(parent1_list[i]))
            if i < len(parent2_list):
                child2_list.append(copy.deepcopy(parent2_list[i]))
    
    return child1_list, child2_list


def uniform_crossover(parent1: Individual, parent2: Individual,
                     generation: int, ind_id: int,
                     swap_prob: float = 0.5,
                     config: dict = None) -> Tuple[Individual, Individual]:
    """
    Uniform crossover.
    
    Each component is randomly inherited from either parent.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual  
        generation: Generation number for offspring
        ind_id: Starting individual ID
        swap_prob: Probability of swapping each component
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Uniform crossover for indicators
    child1_indicators, child2_indicators = _uniform_crossover_lists(
        parent1.strategy_gene.indicators, parent2.strategy_gene.indicators, swap_prob)
    
    # Ensure at least one indicator in each child
    child1_gene.indicators = child1_indicators if child1_indicators else [parent1.strategy_gene.indicators[0]]
    child2_gene.indicators = child2_indicators if child2_indicators else [parent2.strategy_gene.indicators[0]]
    
    # Uniform crossover for entry conditions
    child1_entry, child2_entry = _uniform_crossover_lists(
        parent1.strategy_gene.entry_conditions, parent2.strategy_gene.entry_conditions, swap_prob)
    
    # Ensure at least one entry condition in each child
    child1_gene.entry_conditions = child1_entry if child1_entry else [parent1.strategy_gene.entry_conditions[0]]
    child2_gene.entry_conditions = child2_entry if child2_entry else [parent2.strategy_gene.entry_conditions[0]]
    
    # Uniform crossover for exit conditions
    child1_gene.exit_conditions, child2_gene.exit_conditions = _uniform_crossover_lists(
        parent1.strategy_gene.exit_conditions, parent2.strategy_gene.exit_conditions, swap_prob)
    
    # Randomly inherit scalar parameters
    for attr in ['timeframe', 'stoploss']:
        if random.random() < swap_prob:
            val1 = getattr(parent2.strategy_gene, attr)
            val2 = getattr(parent1.strategy_gene, attr)
            setattr(child1_gene, attr, val1)
            setattr(child2_gene, attr, val2)
    
    if random.random() < swap_prob:
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if random.random() < swap_prob:
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
    
    # Swap regime specialization fields (Phase 1B)
    if random.random() < swap_prob:
        child1_gene.preferred_regime = parent2.strategy_gene.preferred_regime
        child1_gene.regime_mode = parent2.strategy_gene.regime_mode
        child2_gene.preferred_regime = parent1.strategy_gene.preferred_regime
        child2_gene.regime_mode = parent1.strategy_gene.regime_mode
    
    if random.random() < swap_prob:
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Ensure all indicators referenced in conditions are calculated
    if config:
        indicator_config = config.get('indicators', {})
        child1_gene.ensure_indicators_for_conditions(indicator_config)
        child2_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after crossover to avoid ID conflicts
    child1_gene.assign_instance_ids()
    child2_gene.assign_instance_ids()
    
    # Enforce minimum entry conditions
    _enforce_min_entry_conditions(child1_gene, config)
    _enforce_min_entry_conditions(child2_gene, config)
    
    # Post-crossover quality: dedup indicators, dedup/prune conditions, clamp thresholds
    for g in (child1_gene, child2_gene):
        _deduplicate_indicators(g)
        _enforce_max_indicators(g, config)
        g.entry_conditions = _deduplicate_conditions(g.entry_conditions)
        g.exit_conditions = _deduplicate_conditions(g.exit_conditions)
        clamp_condition_thresholds(g.entry_conditions)
        clamp_condition_thresholds(g.exit_conditions)
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def component_crossover(parent1: Individual, parent2: Individual,
                       generation: int, ind_id: int,
                       config: dict = None) -> Tuple[Individual, Individual]:
    """
    Component-based crossover.
    
    Exchange entire indicator sets or rule sets between parents.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        
    Returns:
        Tuple of two offspring individuals
    """
    # Create copies of parent genes
    child1_gene = parent1.strategy_gene.copy()
    child2_gene = parent2.strategy_gene.copy()
    
    # Swap components based on random decisions
    swaps = {
        'indicators': random.random() < 0.5,
        'entry': random.random() < 0.5,
        'exit': random.random() < 0.5,
        'risk': random.random() < 0.5,
    }
    
    if swaps['indicators']:
        child1_gene.indicators, child2_gene.indicators = [copy.deepcopy(ind) for ind in parent2.strategy_gene.indicators], [copy.deepcopy(ind) for ind in parent1.strategy_gene.indicators]
        # Swap informative_timeframes with indicators for consistency
        child1_gene.informative_timeframes = list(parent2.strategy_gene.informative_timeframes)
        child2_gene.informative_timeframes = list(parent1.strategy_gene.informative_timeframes)
    
    if swaps['entry']:
        child1_gene.entry_conditions, child2_gene.entry_conditions = [copy.deepcopy(cond) for cond in parent2.strategy_gene.entry_conditions], [copy.deepcopy(cond) for cond in parent1.strategy_gene.entry_conditions]
    
    if swaps['exit']:
        child1_gene.exit_conditions, child2_gene.exit_conditions = [copy.deepcopy(cond) for cond in parent2.strategy_gene.exit_conditions], [copy.deepcopy(cond) for cond in parent1.strategy_gene.exit_conditions]
    
    if swaps['risk']:
        # Swap all risk parameters (stoploss, ROI, trailing stop)
        child1_gene.stoploss = parent2.strategy_gene.stoploss
        child1_gene.minimal_roi = parent2.strategy_gene.minimal_roi.copy()
        child1_gene.trailing_stop = parent2.strategy_gene.trailing_stop
        # Include regime specialization in risk swap (Phase 1B)
        child1_gene.preferred_regime = parent2.strategy_gene.preferred_regime
        child1_gene.regime_mode = parent2.strategy_gene.regime_mode
        
        child2_gene.stoploss = parent1.strategy_gene.stoploss
        child2_gene.minimal_roi = parent1.strategy_gene.minimal_roi.copy()
        child2_gene.trailing_stop = parent1.strategy_gene.trailing_stop
        child2_gene.preferred_regime = parent1.strategy_gene.preferred_regime
        child2_gene.regime_mode = parent1.strategy_gene.regime_mode
    
    # Update generation and IDs
    child1_gene.generation = generation
    child1_gene.individual_id = ind_id
    child2_gene.generation = generation
    child2_gene.individual_id = ind_id + 1
    
    # Ensure all indicators referenced in conditions are calculated
    if config:
        indicator_config = config.get('indicators', {})
        child1_gene.ensure_indicators_for_conditions(indicator_config)
        child2_gene.ensure_indicators_for_conditions(indicator_config)
    
    # Reassign instance IDs after crossover to avoid ID conflicts
    child1_gene.assign_instance_ids()
    child2_gene.assign_instance_ids()
    
    # Enforce minimum entry conditions
    _enforce_min_entry_conditions(child1_gene, config)
    _enforce_min_entry_conditions(child2_gene, config)
    
    # Post-crossover quality: dedup indicators, dedup/prune conditions, clamp thresholds
    for g in (child1_gene, child2_gene):
        _deduplicate_indicators(g)
        _enforce_max_indicators(g, config)
        g.entry_conditions = _deduplicate_conditions(g.entry_conditions)
        g.exit_conditions = _deduplicate_conditions(g.exit_conditions)
        clamp_condition_thresholds(g.entry_conditions)
        clamp_condition_thresholds(g.exit_conditions)
    
    return (Individual(strategy_gene=child1_gene, parent_ids=[parent1.id, parent2.id]),
            Individual(strategy_gene=child2_gene, parent_ids=[parent1.id, parent2.id]))


def crossover(parent1: Individual, parent2: Individual,
             generation: int, ind_id: int,
             method: str = 'single_point',
             config: dict = None,
             **kwargs) -> Tuple[Individual, Individual]:
    """
    Perform crossover using specified method.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        generation: Generation number for offspring
        ind_id: Starting individual ID
        method: Crossover method ('single_point', 'uniform', 'component')
        config: Configuration dictionary with indicator settings
        **kwargs: Additional arguments for crossover method
        
    Returns:
        Tuple of two offspring individuals
    """
    crossover_methods = {
        'single_point': single_point_crossover,
        'uniform': uniform_crossover,
        'component': component_crossover,
    }
    
    if method not in crossover_methods:
        raise ValueError(f"Unknown crossover method: {method}")
    
    # Pass config to crossover methods
    kwargs['config'] = config
    return crossover_methods[method](parent1, parent2, generation, ind_id, **kwargs)
