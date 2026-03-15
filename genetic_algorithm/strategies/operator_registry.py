"""
Operator Registry

Central source of truth for which operators are valid for each indicator type.
All GA components (mutation, crossover, condition generation, code generation)
reference this registry to prevent invalid indicator+operator combinations.

The 8 supported operators:
  - '<'             : value below threshold
  - '>'             : value above threshold
  - 'cross_above'   : value crosses above reference/threshold
  - 'cross_below'   : value crosses below reference/threshold
  - 'increasing'    : value increasing over lookback period
  - 'decreasing'    : value decreasing over lookback period
  - 'between'       : value between threshold and threshold_upper
  - 'value_above_ago': value above its own value N bars ago
"""

from typing import Dict, FrozenSet, List

# Advanced operators work generically via _resolve_primary_column for ANY
# indicator that has a resolvable primary column.
ADVANCED_OPERATORS = frozenset({'increasing', 'decreasing', 'between', 'value_above_ago'})

# All 8 operator names
ALL_OPERATORS = frozenset({
    '<', '>', 'cross_above', 'cross_below',
    'increasing', 'decreasing', 'between', 'value_above_ago',
})

# Full set of standard (non-advanced) operators per indicator.
# Advanced operators are implicitly valid for every indicator listed here.
_STANDARD_OPERATORS: Dict[str, FrozenSet[str]] = {
    # --- Oscillators ---
    'RSI':   frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'STOCH': frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'CCI':   frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'ADX':   frozenset({'<', '>', 'cross_above', 'cross_below'}),

    # --- Signal-line indicators ---
    'MACD':  frozenset({'<', '>', 'cross_above', 'cross_below'}),

    # --- Volatility ---
    'ATR':   frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'BBANDS': frozenset({'<', '>', 'cross_above', 'cross_below'}),

    # --- Moving averages ---
    'EMA':   frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'SMA':   frozenset({'<', '>', 'cross_above', 'cross_below'}),

    # --- Trend indicators ---
    'SUPERTREND': frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'ICHIMOKU':   frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'DONCHIAN':   frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'PSAR':       frozenset({'<', '>', 'cross_above', 'cross_below'}),

    # --- Volume / price indicators ---
    'VWAP': frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'CMF':  frozenset({'<', '>', 'cross_above', 'cross_below'}),
    'VROC': frozenset({'<', '>', 'cross_above', 'cross_below'}),
}

# Candlestick patterns — bidirectional patterns output -100/0/+100
_CDL_STANDARD_OPERATORS = frozenset({'<', '>'})

# Unidirectional positive-only patterns (only output 0 or +100, NEVER negative)
# CDL_DOJI is indecision — TA-Lib returns 0 or +100 only, so '<' is always false.
_CDL_POSITIVE_ONLY_OPERATORS = frozenset({'>'})

# Patterns that only output positive values (0 or +100)
CDL_POSITIVE_ONLY_TYPES = frozenset({
    'CDL_DOJI', 'CDL_HAMMER', 'CDL_MORNINGSTAR', 'CDL_PIERCING',
    'CDL_3WHITESOLDIERS',
})

# Patterns that only output negative values (0 or -100)
CDL_NEGATIVE_ONLY_TYPES = frozenset({
    'CDL_EVENINGSTAR', 'CDL_SHOOTINGSTAR', 'CDL_DARKCLOUD', 'CDL_3BLACKCROWS',
})
_CDL_NEGATIVE_ONLY_OPERATORS = frozenset({'<'})

# Bidirectional patterns output -100/0/+100
CDL_BIDIRECTIONAL_TYPES = frozenset({
    'CDL_ENGULFING', 'CDL_HARAMI',
})

# All known candlestick pattern types
CDL_TYPES = CDL_POSITIVE_ONLY_TYPES | CDL_NEGATIVE_ONLY_TYPES | CDL_BIDIRECTIONAL_TYPES


def get_valid_operators(indicator_type: str) -> List[str]:
    """
    Return list of all valid operators for the given indicator type.

    Includes both standard operators (specific to the indicator) and
    advanced operators (generic, work for any numeric column).

    Args:
        indicator_type: e.g. 'RSI', 'MACD', 'ATR', 'CDL_HAMMER'

    Returns:
        Sorted list of valid operator strings.
        Empty list if indicator type is unknown.
    """
    if indicator_type in _STANDARD_OPERATORS:
        return sorted(_STANDARD_OPERATORS[indicator_type] | ADVANCED_OPERATORS)
    if indicator_type in CDL_POSITIVE_ONLY_TYPES:
        return sorted(_CDL_POSITIVE_ONLY_OPERATORS)
    if indicator_type in CDL_NEGATIVE_ONLY_TYPES:
        return sorted(_CDL_NEGATIVE_ONLY_OPERATORS)
    if indicator_type in CDL_BIDIRECTIONAL_TYPES:
        return sorted(_CDL_STANDARD_OPERATORS)
    return []


def get_standard_operators(indicator_type: str) -> List[str]:
    """
    Return only the standard (non-advanced) operators for the indicator.
    Useful for mutation when you want to pick a "typical" operator.
    """
    if indicator_type in _STANDARD_OPERATORS:
        return sorted(_STANDARD_OPERATORS[indicator_type])
    if indicator_type in CDL_POSITIVE_ONLY_TYPES:
        return sorted(_CDL_POSITIVE_ONLY_OPERATORS)
    if indicator_type in CDL_NEGATIVE_ONLY_TYPES:
        return sorted(_CDL_NEGATIVE_ONLY_OPERATORS)
    if indicator_type in CDL_BIDIRECTIONAL_TYPES:
        return sorted(_CDL_STANDARD_OPERATORS)
    return []


def is_valid_operator(indicator_type: str, operator: str) -> bool:
    """
    Check whether the given operator is valid for the indicator type.

    Returns True for known indicator+operator combos, False otherwise.
    """
    if indicator_type in _STANDARD_OPERATORS:
        return operator in (_STANDARD_OPERATORS[indicator_type] | ADVANCED_OPERATORS)
    if indicator_type in CDL_POSITIVE_ONLY_TYPES:
        return operator in _CDL_POSITIVE_ONLY_OPERATORS
    if indicator_type in CDL_NEGATIVE_ONLY_TYPES:
        return operator in _CDL_NEGATIVE_ONLY_OPERATORS
    if indicator_type in CDL_BIDIRECTIONAL_TYPES:
        return operator in _CDL_STANDARD_OPERATORS
    return False


def get_all_indicator_types() -> List[str]:
    """Return all indicator types known to the registry (excluding CDL)."""
    return sorted(_STANDARD_OPERATORS.keys())


def resolve_indicator_type(indicator_ref: str) -> str:
    """
    Extract the base indicator type from a condition's indicator reference.

    Handles instance ID formats like 'RSI_0', 'EMA_1h_0', 'CDL_HAMMER_0'.

    Args:
        indicator_ref: The indicator field from a ConditionGene

    Returns:
        Base indicator type string (e.g. 'RSI', 'CDL_HAMMER')
    """
    # CDL patterns: 'CDL_HAMMER_0' → 'CDL_HAMMER'
    if indicator_ref.startswith('CDL_'):
        parts = indicator_ref.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return indicator_ref

    # Regular indicators: 'RSI_0' → 'RSI', 'EMA_1h_0' → 'EMA_1h' → further handled
    if '_' in indicator_ref:
        parts = indicator_ref.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
            # Handle timeframe suffixes like 'EMA_1h' → 'EMA'
            for known_type in sorted(_STANDARD_OPERATORS.keys(), key=len, reverse=True):
                if base == known_type or base.startswith(known_type + '_'):
                    return known_type
            return base
        return indicator_ref

    return indicator_ref
