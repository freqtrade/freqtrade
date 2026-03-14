"""
GA Config Validator

Validates GA configuration at startup to catch misconfigurations early,
before spending time on evolution that would fail or produce bad results.

Returns a list of errors (critical, must fix) and warnings (suboptimal, but runs).
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def validate_ga_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Validate a GA configuration dictionary.
    
    Args:
        config: Full GA config dict (top-level keys: genetic_algorithm, backtesting, etc.)
        
    Returns:
        Tuple of (errors, warnings). Errors are fatal; warnings are informational.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # --- genetic_algorithm section ---
    ga = config.get('genetic_algorithm')
    if not ga:
        errors.append("Missing 'genetic_algorithm' section")
    else:
        _validate_int_range(ga, 'population_size', 2, 10000, errors)
        _validate_int_range(ga, 'generations', 1, 100000, errors)
        _validate_float_range(ga, 'mutation_rate', 0, 1, errors)
        _validate_float_range(ga, 'crossover_rate', 0, 1, errors)
        
        ps = ga.get('population_size', 0)
        es = ga.get('elite_size', 0)
        if isinstance(ps, int) and isinstance(es, int) and es >= ps:
            errors.append(f"elite_size ({es}) must be < population_size ({ps})")
        
        ts = ga.get('tournament_size', 3)
        if isinstance(ts, int) and isinstance(ps, int) and ts > ps:
            errors.append(f"tournament_size ({ts}) must be <= population_size ({ps})")
        
        mode = ga.get('mode', 'single_objective')
        if mode not in ('single_objective', 'nsga2'):
            errors.append(f"mode must be 'single_objective' or 'nsga2', got '{mode}'")
    
    # --- backtesting section ---
    bt = config.get('backtesting')
    if not bt:
        errors.append("Missing 'backtesting' section")
    else:
        pairs = bt.get('pairs', [])
        if not pairs:
            errors.append("backtesting.pairs must be a non-empty list")
        
        timerange = bt.get('timerange', '')
        if not timerange:
            warnings.append("backtesting.timerange is empty — will use all available data")
        elif '-' not in timerange:
            errors.append(f"backtesting.timerange format should be 'YYYYMMDD-YYYYMMDD', got '{timerange}'")
        
        datadir = bt.get('datadir', '')
        if not datadir:
            warnings.append("backtesting.datadir not set — will use FreqTrade default")
        
        fee = bt.get('fee', 0.001)
        if isinstance(fee, (int, float)) and fee > 0.05:
            warnings.append(f"backtesting.fee={fee} seems high (>5%). Typical range: 0.0005-0.002")
        
        fee_noise = bt.get('fee_noise_std', 0)
        if isinstance(fee_noise, (int, float)) and fee_noise > 0.01:
            warnings.append(f"backtesting.fee_noise_std={fee_noise} is large. Typical: 0.0001-0.0005")
    
    # --- fitness_weights section ---
    fw = config.get('fitness_weights', {})
    if fw:
        total = sum(v for v in fw.values() if isinstance(v, (int, float)))
        if abs(total - 1.0) > 0.10:
            warnings.append(f"fitness_weights sum to {total:.3f} (expected ~1.0, will be auto-normalized)")
        for key, val in fw.items():
            if isinstance(val, (int, float)) and val < 0:
                errors.append(f"fitness_weights.{key}={val} is negative (must be >= 0)")
    
    # --- walk_forward section ---
    wf = config.get('walk_forward', {})
    if wf.get('enabled'):
        td = wf.get('train_days')
        vd = wf.get('validation_days')
        if td is not None and isinstance(td, (int, float)) and td < 7:
            errors.append(f"walk_forward.train_days={td} is too short (minimum 7)")
        if vd is not None and isinstance(vd, (int, float)) and vd < 1:
            errors.append(f"walk_forward.validation_days={vd} is too short (minimum 1)")
    
    # --- strategy_constraints section ---
    sc = config.get('strategy_constraints', {})
    sl = sc.get('stoploss_range', [-0.20, -0.05])
    if isinstance(sl, list) and len(sl) == 2:
        if sl[0] > sl[1]:
            errors.append(f"strategy_constraints.stoploss_range[0] ({sl[0]}) must be <= [1] ({sl[1]})")
        if sl[1] > 0:
            warnings.append(f"strategy_constraints.stoploss_range upper bound ({sl[1]}) is positive. "
                          "Stoploss should normally be negative")
    
    # --- monte_carlo section ---
    mc = config.get('monte_carlo', {})
    if mc.get('enabled'):
        nperms = mc.get('num_permutations', 100)
        if isinstance(nperms, int) and nperms > 1000:
            warnings.append(f"monte_carlo.num_permutations={nperms} — this will slow down evaluation")
    
    # --- indicators section ---
    ind = config.get('indicators', {})
    max_ind = ind.get('max_per_strategy', 6)
    min_ind = ind.get('min_per_strategy', 2)
    if isinstance(max_ind, int) and isinstance(min_ind, int) and min_ind > max_ind:
        errors.append(f"indicators.min_per_strategy ({min_ind}) > max_per_strategy ({max_ind})")
    
    return errors, warnings


def _validate_int_range(section: Dict, key: str, min_val: int, max_val: int, 
                        errors: List[str]) -> None:
    """Validate integer config value is in range."""
    val = section.get(key)
    if val is None:
        errors.append(f"Missing required key: {key}")
    elif not isinstance(val, int):
        errors.append(f"{key} must be an integer, got {type(val).__name__}")
    elif val < min_val or val > max_val:
        errors.append(f"{key}={val} must be between {min_val} and {max_val}")


def _validate_float_range(section: Dict, key: str, min_val: float, max_val: float,
                          errors: List[str]) -> None:
    """Validate float config value is in range."""
    val = section.get(key)
    if val is None:
        return  # Optional
    if not isinstance(val, (int, float)):
        errors.append(f"{key} must be a number, got {type(val).__name__}")
    elif val < min_val or val > max_val:
        errors.append(f"{key}={val} must be between {min_val} and {max_val}")


def validate_and_log(config: Dict[str, Any]) -> bool:
    """
    Validate config and log results.
    
    Returns:
        True if valid (no errors), False if invalid.
    """
    errors, warnings = validate_ga_config(config)
    
    for w in warnings:
        logger.warning(f"[CONFIG] Warning: {w}")
    
    if errors:
        for e in errors:
            logger.error(f"[CONFIG] Error: {e}")
        return False
    
    logger.info(f"[CONFIG] Validation passed ({len(warnings)} warnings)")
    return True
