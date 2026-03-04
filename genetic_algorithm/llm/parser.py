"""
LLM Response Parser

Parses raw LLM JSON responses into validated :class:`StrategyGene` objects.
Handles retries with error feedback so the model can self-correct invalid
output on subsequent attempts.
"""

import logging
import random
import re
from typing import Dict, Any, List, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.llm.prompts import INDICATOR_REFERENCE

logger = logging.getLogger(__name__)

# Operators supported by the condition evaluator
_VALID_OPERATORS = frozenset({
    '<', '>', 'cross_above', 'cross_below',
    'increasing', 'decreasing', 'between', 'value_above_ago',
})


class StrategyParser:
    """
    Converts LLM JSON output into a validated :class:`StrategyGene`.

    Validation rules (mirrors ``StrategyGene.__post_init__``):

    * At least one indicator with a known type.
    * At least one entry condition referencing a valid indicator.
    * Conditions referencing unknown indicators are silently removed.
    * Stoploss is clamped to the configured range.
    * Minimal-ROI keys are coerced to strings and values to floats.

    Retry feedback:
        Call :meth:`parse_with_feedback` instead of :meth:`parse` to receive
        a human-readable error message when parsing fails; this message can be
        appended to the next LLM prompt so the model can self-correct.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full GA config dict (used for ``indicators.available``,
                ``strategy_constraints``, and ``indicators.min_*_conditions``).
        """
        self.config = config
        self.available_indicators: List[str] = (
            config.get('indicators', {}).get('available', [])
        )
        self.min_entry = config.get('indicators', {}).get('min_entry_conditions', 2)
        self.min_exit = config.get('indicators', {}).get('min_exit_conditions', 1)
        sl_range = config.get('strategy_constraints', {}).get('stoploss_range', [-0.20, -0.05])
        self.sl_min: float = sl_range[0]
        self.sl_max: float = sl_range[1]
        self.available_timeframes: List[str] = (
            config.get('strategy_constraints', {}).get('timeframes', ['5m', '15m', '1h'])
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        data: Dict[str, Any],
        generation: int,
        individual_id: int,
    ) -> Optional[StrategyGene]:
        """
        Parse a dict (already JSON-decoded) into a :class:`StrategyGene`.

        Args:
            data: Decoded JSON dict from the LLM.
            generation: Generation number to embed in the gene.
            individual_id: Individual ID to embed in the gene.

        Returns:
            :class:`StrategyGene` on success, ``None`` on unrecoverable failure.
        """
        gene, _ = self._parse_inner(data, generation, individual_id)
        return gene

    def parse_with_feedback(
        self,
        data: Dict[str, Any],
        generation: int,
        individual_id: int,
    ) -> tuple:
        """
        Parse and return ``(StrategyGene | None, error_message | None)``.

        The error message (when not ``None``) is suitable for inclusion in
        a follow-up LLM prompt so the model can self-correct.

        Args:
            data: Decoded JSON dict from the LLM.
            generation: Generation number.
            individual_id: Individual ID.

        Returns:
            Tuple of ``(gene, error_msg)``.  If parsing succeeds, ``error_msg``
            is ``None``.  If it fails, ``gene`` is ``None`` and ``error_msg``
            is a descriptive string.
        """
        return self._parse_inner(data, generation, individual_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_inner(
        self,
        data: Dict[str, Any],
        generation: int,
        individual_id: int,
    ) -> tuple:
        """Core parsing logic; returns ``(gene_or_None, error_or_None)``."""
        fixes: List[str] = []

        try:
            # ---- Indicators ----------------------------------------
            indicators = self._parse_indicators(data.get('indicators', []), fixes)
            if not indicators:
                msg = ("No valid indicators found. "
                       f"Use types from: {', '.join(self.available_indicators)}.")
                return None, msg

            valid_refs = {ind.instance_id for ind in indicators}
            valid_refs.update(ind.type for ind in indicators)

            # ---- Entry conditions -----------------------------------
            entry_conditions = self._parse_conditions(
                data.get('entry_conditions', []), valid_refs, 'entry'
            )
            if len(entry_conditions) < self.min_entry:
                self._add_default_conditions(
                    indicators, entry_conditions,
                    self.min_entry - len(entry_conditions), is_entry=True,
                )
                fixes.append(
                    f"Added default entry conditions (need ≥ {self.min_entry})."
                )

            if not entry_conditions:
                msg = (f"No valid entry conditions. "
                       f"Reference indicators by instance_id "
                       f"({', '.join(ind.instance_id for ind in indicators)}) "
                       f"using operators: {', '.join(sorted(_VALID_OPERATORS))}.")
                return None, msg

            # ---- Exit conditions ------------------------------------
            exit_conditions = self._parse_conditions(
                data.get('exit_conditions', []), valid_refs, 'exit'
            )
            if len(exit_conditions) < self.min_exit:
                self._add_default_conditions(
                    indicators, exit_conditions,
                    self.min_exit - len(exit_conditions), is_entry=False,
                )
                fixes.append(
                    f"Added default exit conditions (need ≥ {self.min_exit})."
                )

            # ---- Risk parameters -----------------------------------
            timeframe = data.get('timeframe', '15m')
            if timeframe not in self.available_timeframes:
                timeframe = random.choice(self.available_timeframes)
                fixes.append(f"Timeframe replaced with '{timeframe}'.")

            stoploss = self._safe_float(data.get('stoploss', -0.10))
            stoploss = max(self.sl_min, min(self.sl_max, stoploss))

            minimal_roi = data.get('minimal_roi', {"0": 0.05, "30": 0.03, "60": 0.01})
            if not isinstance(minimal_roi, dict):
                minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}
            minimal_roi = {str(k): float(v) for k, v in minimal_roi.items()}

            max_open_trades = max(1, min(10, int(data.get('max_open_trades', 3))))

            trailing_stop = bool(data.get('trailing_stop', False))
            tsp = self._optional_float(data.get('trailing_stop_positive'))
            tspo = self._optional_float(data.get('trailing_stop_positive_offset'))

            # ---- Build gene ----------------------------------------
            gene = StrategyGene(
                generation=generation,
                individual_id=individual_id,
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                timeframe=timeframe,
                stoploss=stoploss,
                minimal_roi=minimal_roi,
                max_open_trades=max_open_trades,
                trailing_stop=trailing_stop,
                trailing_stop_positive=tsp,
                trailing_stop_positive_offset=tspo,
            )

            if fixes:
                logger.info("Parser applied %d fix(es): %s", len(fixes), '; '.join(fixes))

            return gene, None

        except Exception as exc:  # pragma: no cover – unexpected errors
            logger.error("StrategyParser unexpected error: %s", exc)
            return None, f"Unexpected parse error: {exc}"

    def _parse_indicators(
        self,
        raw: List[Any],
        fixes: List[str],
    ) -> List[IndicatorGene]:
        """Parse and validate indicator list; skip unknown types."""
        indicators: List[IndicatorGene] = []
        used_ids: set = set()

        for item in raw:
            if not isinstance(item, dict):
                continue
            ind_type = item.get('type', '')
            if ind_type not in self.available_indicators:
                logger.debug("Unknown indicator type '%s', skipping.", ind_type)
                fixes.append(f"Removed unknown indicator '{ind_type}'.")
                continue

            # Ensure unique instance_id
            instance_id = item.get('instance_id', '')
            if not instance_id or instance_id in used_ids:
                counter = 0
                while f"{ind_type}_{counter}" in used_ids:
                    counter += 1
                instance_id = f"{ind_type}_{counter}"
                fixes.append(f"Auto-assigned instance_id '{instance_id}'.")
            used_ids.add(instance_id)

            params = item.get('parameters', {})
            if not isinstance(params, dict):
                params = {}

            indicators.append(IndicatorGene(
                type=ind_type,
                parameters=params,
                weight=float(item.get('weight', 1.0)),
                instance_id=instance_id,
                timeframe=item.get('timeframe'),
            ))

        return indicators

    def _parse_conditions(
        self,
        raw: List[Any],
        valid_refs: set,
        condition_type: str,
    ) -> List[ConditionGene]:
        """Parse and validate condition list; skip invalid entries."""
        conditions: List[ConditionGene] = []

        for item in raw:
            if not isinstance(item, dict):
                continue
            indicator = item.get('indicator', '')
            operator = item.get('operator', '')

            if indicator not in valid_refs:
                logger.debug(
                    "Skipping %s condition: unknown indicator ref '%s'.",
                    condition_type, indicator,
                )
                continue
            if operator not in _VALID_OPERATORS:
                logger.debug(
                    "Skipping %s condition: unknown operator '%s'.",
                    condition_type, operator,
                )
                continue

            conditions.append(ConditionGene(
                indicator=indicator,
                operator=operator,
                threshold=self._safe_float(item.get('threshold', 0)),
                logic=item.get('logic', 'AND'),
                threshold_upper=self._safe_float(item.get('threshold_upper', 0)),
                lookback=max(1, int(item.get('lookback', 3))),
            ))

        return conditions

    def _add_default_conditions(
        self,
        indicators: List[IndicatorGene],
        conditions: List[ConditionGene],
        needed: int,
        is_entry: bool,
    ) -> None:
        """Append sensible default conditions for underused indicators."""
        referenced = {c.indicator for c in conditions}
        unreferenced = [
            ind for ind in indicators
            if ind.instance_id not in referenced and ind.type not in referenced
        ]
        if not unreferenced:
            unreferenced = list(indicators)

        for ind in unreferenced[:needed]:
            ref = INDICATOR_REFERENCE.get(ind.type, {})
            template = (
                ref.get('typical_entry', {'operator': '>', 'threshold': '50'})
                if is_entry
                else ref.get('typical_exit', {'operator': '<', 'threshold': '50'})
            )
            threshold_str = str(template.get('threshold', '50'))
            numbers = re.findall(r'-?\d+(?:\.\d+)?', threshold_str)
            threshold = float(numbers[0]) if numbers else 50.0
            operator = template.get('operator', '>' if is_entry else '<')

            conditions.append(ConditionGene(
                indicator=ind.instance_id or ind.type,
                operator=operator,
                threshold=threshold,
                logic='AND',
            ))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Coerce a value to float, returning *default* on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        """Return ``float(value)`` or ``None`` if value is falsy/invalid."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
