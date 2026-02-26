"""
Dynamic Parameter Ranges

Allows indicator parameter ranges (min, max) to be evolved alongside the parameters
themselves. This lets the GA discover tighter or wider search spaces per indicator
instance over successive generations.

Storage:
--------
Each `IndicatorGene` can optionally hold a `param_bounds` dict:

    {
      "period": [5, 30],        # original static range
      "period_evolved": [8, 22] # tightened by evolution
    }

If `param_bounds` is absent, mutation falls back to config-defined ranges
(backward compatible).

Mutation operators:
------------------
- `shift_bounds`: move min/max together (shift the search region)
- `expand_bounds`: widen by adding to max and subtracting from min
- `contract_bounds`: tighten the range (converge toward known good values)

Integration:
-----------
- `initialise_bounds()` is called when generating a new indicator to seed
  bounds from config.
- `mutate_bounds()` may be invoked by the main `mutate()` function to
  perturb bounds at a low probability.
- `sample_from_bounds()` is used by parameter mutation and crossover to
  sample new parameter values respecting evolved bounds.
"""

from typing import Dict, Any, Tuple, Optional
import random
import math


def initialise_bounds(
    indicator_type: str,
    parameters: Dict[str, Any],
    indicator_config: Dict[str, Any],
) -> Dict[str, Tuple[float, float]]:
    """
    Seed evolable bounds from static config for a new indicator.

    Args:
        indicator_type: e.g. 'RSI', 'MACD'
        parameters: current parameter values on the indicator
        indicator_config: indicator-specific config with default ranges

    Returns:
        Dict mapping param_name -> (min, max)
    """
    cfg = indicator_config.get(indicator_type, {})
    bounds: Dict[str, Tuple[float, float]] = {}

    for param_name, param_value in parameters.items():
        # Try to get the static range from config
        default_range = cfg.get(param_name)
        if default_range and len(default_range) == 2:
            bounds[param_name] = tuple(default_range)
        else:
            # Fallback: use a symmetric window around current value
            window = max(1, abs(param_value) * 0.5)
            bounds[param_name] = (param_value - window, param_value + window)

    return bounds


def mutate_bounds(
    bounds: Dict[str, Tuple[float, float]],
    parameters: Dict[str, Any],
    mutation_strength: float = 0.1,
    rng: Optional[random.Random] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Apply random mutation to parameter bounds.

    Randomly chooses a perturbation type:
    - shift: move min and max together (translate the search window)
    - expand: widen the range
    - contract: narrow the range toward the current parameter value

    Args:
        bounds: current bounds dict
        parameters: current parameter values (used for contract)
        mutation_strength: controls magnitude of change (0.0–1.0)
        rng: optional Random instance

    Returns:
        New mutated bounds dict
    """
    rng = rng or random.Random()
    new_bounds = dict(bounds)

    for pname, (lo, hi) in bounds.items():
        if rng.random() > 0.5:
            # skip this param with 50% prob to limit changes
            continue

        span = hi - lo
        delta = span * mutation_strength
        op = rng.choice(['shift', 'expand', 'contract'])

        if op == 'shift':
            shift = rng.uniform(-delta, delta)
            new_lo = lo + shift
            new_hi = hi + shift
        elif op == 'expand':
            new_lo = lo - rng.uniform(0, delta * 0.5)
            new_hi = hi + rng.uniform(0, delta * 0.5)
        else:  # contract
            cur_val = parameters.get(pname)
            if cur_val is None:
                cur_val = (lo + hi) / 2
            # pull lo/hi toward current value
            new_lo = lo + (cur_val - lo) * mutation_strength * rng.random()
            new_hi = hi - (hi - cur_val) * mutation_strength * rng.random()

        # Ensure valid (min <= max)
        if new_lo > new_hi:
            new_lo, new_hi = new_hi, new_lo

        # Ensure minimum span (10% of original or 1) to prevent collapse
        min_span = max(1, span * 0.1)
        if (new_hi - new_lo) < min_span:
            mid = (new_lo + new_hi) / 2
            new_lo = mid - min_span / 2
            new_hi = mid + min_span / 2

        new_bounds[pname] = (new_lo, new_hi)

    return new_bounds


def sample_from_bounds(
    param_name: str,
    bounds: Optional[Dict[str, Tuple[float, float]]],
    fallback_range: Tuple[float, float],
    is_int: bool = True,
    rng: Optional[random.Random] = None,
) -> Any:
    """
    Sample a new parameter value respecting evolved bounds.

    Args:
        param_name: the parameter to sample
        bounds: evolved bounds dict (may be None)
        fallback_range: (min, max) to use if bounds missing
        is_int: if True, return an int; otherwise float
        rng: optional Random

    Returns:
        Sampled value
    """
    rng = rng or random.Random()

    if bounds and param_name in bounds:
        lo, hi = bounds[param_name]
    else:
        lo, hi = fallback_range

    if is_int:
        return rng.randint(int(math.floor(lo)), int(math.ceil(hi)))
    return rng.uniform(lo, hi)


def crossover_bounds(
    bounds1: Optional[Dict[str, Tuple[float, float]]],
    bounds2: Optional[Dict[str, Tuple[float, float]]],
    rng: Optional[random.Random] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Uniform crossover of two bounds dicts.

    For each param, randomly picks the bounds from either parent.

    Args:
        bounds1, bounds2: parent bounds (may be None)
        rng: Random instance

    Returns:
        Child bounds dict
    """
    rng = rng or random.Random()
    all_keys = set()
    if bounds1:
        all_keys.update(bounds1.keys())
    if bounds2:
        all_keys.update(bounds2.keys())

    child: Dict[str, Tuple[float, float]] = {}
    for k in all_keys:
        b1 = bounds1.get(k) if bounds1 else None
        b2 = bounds2.get(k) if bounds2 else None
        if b1 and b2:
            child[k] = rng.choice([b1, b2])
        elif b1:
            child[k] = b1
        elif b2:
            child[k] = b2

    return child
