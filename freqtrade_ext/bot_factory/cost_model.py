from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_SCENARIO_NAMES = ("best", "normal", "stress")
_TOTAL_COST_COMPONENT_FIELDS = (
    "fee_bps_entry",
    "fee_bps_exit",
    "spread_bps",
    "slippage_bps_entry",
    "slippage_bps_exit",
    "adverse_selection_bps",
    "stress_multiplier",
)


@dataclass(frozen=True)
class CostModelContext:
    pair: str | None = None
    timeframe: str | None = None
    order_type: str | None = None
    liquidity_tier: str | None = None
    volatility_regime: str | None = None


@dataclass(frozen=True)
class CostScenario:
    scenario_name: str
    fee_bps_entry: float
    fee_bps_exit: float
    spread_bps: float
    slippage_bps_entry: float
    slippage_bps_exit: float
    adverse_selection_bps: float
    no_fill_rate: float
    partial_fill_rate: float
    exit_taker_rate: float
    stress_multiplier: float = 1.0
    total_cost_bps_override: float | None = None
    pair: str | None = None
    timeframe: str | None = None
    order_type: str | None = None
    liquidity_tier: str | None = None
    volatility_regime: str | None = None

    @property
    def total_cost_bps(self) -> float:
        if self.total_cost_bps_override is not None:
            return round(max(0.0, float(self.total_cost_bps_override)), 6)
        base_cost = (
            float(self.fee_bps_entry)
            + float(self.fee_bps_exit)
            + float(self.spread_bps)
            + float(self.slippage_bps_entry)
            + float(self.slippage_bps_exit)
            + float(self.adverse_selection_bps)
        )
        return round(max(0.0, base_cost * max(0.0, float(self.stress_multiplier))), 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "fee_bps_entry": float(self.fee_bps_entry),
            "fee_bps_exit": float(self.fee_bps_exit),
            "spread_bps": float(self.spread_bps),
            "slippage_bps_entry": float(self.slippage_bps_entry),
            "slippage_bps_exit": float(self.slippage_bps_exit),
            "adverse_selection_bps": float(self.adverse_selection_bps),
            "no_fill_rate": float(self.no_fill_rate),
            "partial_fill_rate": float(self.partial_fill_rate),
            "exit_taker_rate": float(self.exit_taker_rate),
            "stress_multiplier": float(self.stress_multiplier),
            "total_cost_bps": self.total_cost_bps,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "order_type": self.order_type,
            "liquidity_tier": self.liquidity_tier,
            "volatility_regime": self.volatility_regime,
        }


def default_cost_scenarios(*, normal_cost_bps: float = 12.0) -> dict[str, CostScenario]:
    normal_cost_bps = max(0.0, float(normal_cost_bps))
    return {
        "best": CostScenario(
            scenario_name="best",
            fee_bps_entry=2.0,
            fee_bps_exit=2.0,
            spread_bps=1.0,
            slippage_bps_entry=0.5,
            slippage_bps_exit=0.5,
            adverse_selection_bps=0.0,
            no_fill_rate=0.03,
            partial_fill_rate=0.08,
            exit_taker_rate=0.25,
            stress_multiplier=1.0,
        ),
        "normal": CostScenario(
            scenario_name="normal",
            fee_bps_entry=3.0,
            fee_bps_exit=3.0,
            spread_bps=2.0,
            slippage_bps_entry=2.0,
            slippage_bps_exit=2.0,
            adverse_selection_bps=0.0,
            no_fill_rate=0.08,
            partial_fill_rate=0.15,
            exit_taker_rate=0.5,
            stress_multiplier=1.0,
            total_cost_bps_override=normal_cost_bps,
        ),
        "stress": CostScenario(
            scenario_name="stress",
            fee_bps_entry=3.0,
            fee_bps_exit=3.0,
            spread_bps=2.0,
            slippage_bps_entry=2.0,
            slippage_bps_exit=2.0,
            adverse_selection_bps=1.0,
            no_fill_rate=0.18,
            partial_fill_rate=0.3,
            exit_taker_rate=0.75,
            stress_multiplier=1.5,
        ),
    }


def cost_scenarios_from_spec(
    spec: Mapping[str, Any],
    *,
    context: CostModelContext | None = None,
) -> dict[str, dict[str, Any]]:
    context = context or cost_context_from_spec(spec)
    normal_cost_bps = _float_or_none(spec.get("all_in_cost_bps"))
    defaults = default_cost_scenarios(
        normal_cost_bps=12.0 if normal_cost_bps is None else normal_cost_bps
    )
    model = spec.get("cost_model")
    if isinstance(model, Mapping):
        normal_override = _float_or_none(model.get("all_in_cost_bps"))
        if normal_override is not None:
            defaults["normal"] = _scenario_from_mapping(
                {
                    **defaults["normal"].to_dict(),
                    "total_cost_bps": normal_override,
                },
                fallback=defaults["normal"],
                context=context,
            )
        _apply_scenario_mappings(defaults, model.get("scenarios"), context=context)
        selected = _select_override(model, context)
        if selected is not None:
            selected_normal_override = _float_or_none(selected.get("all_in_cost_bps"))
            if selected_normal_override is not None:
                defaults["normal"] = _scenario_from_mapping(
                    {
                        **defaults["normal"].to_dict(),
                        "total_cost_bps": selected_normal_override,
                    },
                    fallback=defaults["normal"],
                    context=context,
                )
            _apply_scenario_mappings(
                defaults,
                selected.get("scenarios"),
                context=context,
            )
    return {name: defaults[name].to_dict() for name in _SCENARIO_NAMES}


def cost_context_from_spec(spec: Mapping[str, Any]) -> CostModelContext:
    return CostModelContext(
        pair=_string_or_none(spec.get("pair") or spec.get("target_symbol")),
        timeframe=_string_or_none(spec.get("timeframe")),
        order_type=_string_or_none(spec.get("order_type") or spec.get("entry_order_type")),
        liquidity_tier=_string_or_none(spec.get("liquidity_tier")),
        volatility_regime=_string_or_none(spec.get("volatility_regime")),
    )


def _select_override(
    model: Mapping[str, Any], context: CostModelContext
) -> Mapping[str, Any] | None:
    raw = model.get("overrides")
    if not isinstance(raw, list):
        return None
    matches: list[tuple[int, int, Mapping[str, Any]]] = []
    selector_fields = (
        "pair",
        "timeframe",
        "order_type",
        "liquidity_tier",
        "volatility_regime",
    )
    for item in raw:
        if isinstance(item, Mapping) and _override_matches(item, context):
            specificity = sum(
                1
                for field in selector_fields
                if _string_or_none(item.get(field)) is not None
            )
            matches.append((specificity, len(matches), item))
    if not matches:
        return None
    return max(matches, key=lambda match: (match[0], match[1]))[2]


def _override_matches(item: Mapping[str, Any], context: CostModelContext) -> bool:
    for field in (
        "pair",
        "timeframe",
        "order_type",
        "liquidity_tier",
        "volatility_regime",
    ):
        expected = _string_or_none(item.get(field))
        actual = getattr(context, field)
        if expected is not None and _normalize(expected) != _normalize(actual):
            return False
    return True


def _apply_scenario_mappings(
    scenarios: dict[str, CostScenario],
    raw_scenarios: Any,
    *,
    context: CostModelContext,
) -> None:
    if not isinstance(raw_scenarios, list):
        return
    for item in raw_scenarios:
        if not isinstance(item, Mapping):
            continue
        name = _scenario_name(item.get("scenario_name"))
        if name in _SCENARIO_NAMES:
            scenarios[name] = _scenario_from_mapping(
                item,
                fallback=scenarios[name],
                context=context,
            )


def _scenario_from_mapping(
    data: Mapping[str, Any],
    *,
    fallback: CostScenario,
    context: CostModelContext,
) -> CostScenario:
    return CostScenario(
        scenario_name=_scenario_name(data.get("scenario_name")) or fallback.scenario_name,
        fee_bps_entry=_coalesce_float(data.get("fee_bps_entry"), fallback.fee_bps_entry),
        fee_bps_exit=_coalesce_float(data.get("fee_bps_exit"), fallback.fee_bps_exit),
        spread_bps=_coalesce_float(data.get("spread_bps"), fallback.spread_bps),
        slippage_bps_entry=_coalesce_float(
            data.get("slippage_bps_entry"), fallback.slippage_bps_entry
        ),
        slippage_bps_exit=_coalesce_float(
            data.get("slippage_bps_exit"), fallback.slippage_bps_exit
        ),
        adverse_selection_bps=(
            _coalesce_float(
                data.get("adverse_selection_bps"),
                fallback.adverse_selection_bps,
            )
        ),
        no_fill_rate=_coalesce_float(data.get("no_fill_rate"), fallback.no_fill_rate),
        partial_fill_rate=_coalesce_float(
            data.get("partial_fill_rate"), fallback.partial_fill_rate
        ),
        exit_taker_rate=_coalesce_float(
            data.get("exit_taker_rate"), fallback.exit_taker_rate
        ),
        stress_multiplier=_coalesce_float(
            data.get("stress_multiplier"), fallback.stress_multiplier
        ),
        total_cost_bps_override=_total_cost_override_from_mapping(data, fallback),
        pair=_string_or_none(data.get("pair")) or context.pair,
        timeframe=_string_or_none(data.get("timeframe")) or context.timeframe,
        order_type=_string_or_none(data.get("order_type")) or context.order_type,
        liquidity_tier=_string_or_none(data.get("liquidity_tier")) or context.liquidity_tier,
        volatility_regime=(
            _string_or_none(data.get("volatility_regime")) or context.volatility_regime
        ),
    )


def _total_cost_override_from_mapping(
    data: Mapping[str, Any], fallback: CostScenario
) -> float | None:
    parsed = _float_or_none(data.get("total_cost_bps"))
    if parsed is not None:
        return parsed
    has_price_component_override = any(
        _float_or_none(data.get(field)) is not None
        for field in _TOTAL_COST_COMPONENT_FIELDS
    )
    if has_price_component_override:
        return None
    return fallback.total_cost_bps_override


def _scenario_name(value: Any) -> str | None:
    text = _normalize(value)
    return text if text in _SCENARIO_NAMES else None


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce_float(value: Any, fallback: float) -> float:
    parsed = _float_or_none(value)
    return float(fallback) if parsed is None else parsed
