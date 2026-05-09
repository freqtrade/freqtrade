from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_LOGIC_VARIANTS = {
    "amihud_illiquidity_premium",
    "bipower_jump_decay",
    "calendar_turnover_seasonality",
    "crowding_unwind_reaccumulation",
    "cross_asset_cointegration_spread",
    "cross_asset_correlation_recovery",
    "cross_asset_lead_lag",
    "downside_liquidity_shock_reversal",
    "directional_change_overshoot",
    "entropy_regime_transition",
    "fractal_long_memory_regime",
    "funding_pressure_carry",
    "intraday_session_liquidity_reclaim",
    "liquidity_recovery_horizon",
    "market_beta_drawdown_carry",
    "mark_discount_reclaim_continuation",
    "mark_fair_value_momentum_lag",
    "mark_price_dislocation_reclaim",
    "mean_reversion_pullback",
    "microstructure_spread_reversion",
    "range_quarticity_vol_of_vol_state",
    "realized_skewness_tail_shape",
    "regime_state_reentry",
    "semivariance_asymmetry_regime",
    "signed_volume_imbalance_accumulation",
    "trend_continuation",
    "variance_ratio_regime_switch",
    "volatility_breakout",
}
ML_GENERATOR_MODES = {"freqai", "hybrid_ml"}
LOW_ENTRY_SIGNAL_RATIO = 0.001
DEFAULT_ENTRY_EDGE_WINDOW_COUNT = 4
DEFAULT_ENTRY_EDGE_MIN_PROFITABLE_WINDOWS_RATIO = 0.5
LOCAL_CROWDING_OPEN_INTEREST_PATH = Path(
    "data/market_structure/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet"
)
LOCAL_CROWDING_LONG_SHORT_RATIO_PATH = Path(
    "data/market_structure/bybit/futures/BTC_USDT_USDT-1h-long_short_ratio.parquet"
)

COMPONENT_DESCRIPTIONS = {
    "price_impact_premium": "Amihud-style price impact is above its local baseline, proxying an illiquidity premium regime.",
    "illiquidity_releasing": "Amihud-style price impact is declining on the current closed candle.",
    "not_extreme_impact": "Amihud-style price impact is elevated but not in an extreme stress tail.",
    "price_resilience": "Price holds above the rolling midpoint while illiquidity releases.",
    "positive_illiquidity_drift": "Closed-candle drift over the illiquidity lookback is positive.",
    "volume_floor": "Closed-candle volume is above the generated minimum participation threshold.",
    "positive_jump_detected": "A recent closed-candle positive jump event was detected from realized jump variation.",
    "jump_dominates_continuous_variance": "Estimated jump variation is large relative to its local baseline.",
    "continuous_variance_decaying": "Bipower variation indicates the continuous variance component is decaying.",
    "post_jump_drift_positive": "Closed-candle drift after the jump remains positive.",
    "not_overextended_after_jump": "Price has not overextended beyond the recent rolling high after the jump.",
    "calendar_risk_window": "Candle is in the theory-defined Monday or Thursday UTC day-of-week risk window.",
    "weekend_discount_context": "Recent weekend turnover baseline is at or below the weekday turnover baseline.",
    "turnover_recovery": "Closed-candle turnover is recovering relative to its rolling calendar turnover baseline.",
    "positive_calendar_drift": "Closed-candle drift over the calendar lookback is positive.",
    "open_interest_unwinding": "Validated local open interest is contracting over the generated closed-candle unwind window.",
    "short_account_reaccumulation": "Validated local long/short account ratio is washed out enough to proxy reduced long-side crowding.",
    "price_above_sma": "Close is back above the generated 12h moving-average reacceptance anchor.",
    "volume_participation_floor": "Closed-candle volume is not withdrawn relative to its generated local baseline.",
    "eth_positive_lead": "ETH's previous closed-candle return is above its local lead-return baseline.",
    "btc_lag_discount": "ETH's lead return exceeds the simultaneous BTC return, proxying a cross-asset lag setup.",
    "spread_not_extreme": "The ETH-BTC lead spread is positive without being an extreme local outlier.",
    "btc_resilience": "BTC holds above the rolling price-range midpoint after the cross-asset lead signal.",
    "positive_cross_asset_drift": "BTC drift over the generated cross-asset lookback is positive.",
    "btc_discount_to_eth": "BTC/ETH log-price ratio is below its local equilibrium baseline.",
    "spread_reversion_turn": "The BTC/ETH ratio z-score has started reverting upward from discount.",
    "eth_market_support": "ETH has positive drift over the generated cointegration lookback.",
    "cointegration_spread_not_extreme": "The BTC/ETH ratio z-score is away from equilibrium without being an extreme local outlier.",
    "correlation_breakdown": "BTC/ETH return correlation baseline is locally low after a cross-asset regime disconnect.",
    "correlation_recovery": "BTC/ETH return correlation is rising back above its local baseline.",
    "btc_relative_recovery": "BTC closed-candle return is improving relative to ETH after the correlation break.",
    "downside_shock": "Close has fallen enough over the lookback window relative to normalized ATR.",
    "rsi_washout": "RSI reached the generated downside washout threshold within the lookback window.",
    "quiet_volume": "Volume is below the generated local volume threshold, proxying a liquidity-provision regime.",
    "local_low_reclaim": "Close has reclaimed the previous rolling low after the downside shock.",
    "directional_change_confirmed": "Closed-candle directional-change state is bullish and recent enough for event-time evaluation.",
    "overshoot_persisted": "The bullish overshoot has moved beyond the directional-change threshold and persisted for multiple candles without extreme extension.",
    "event_time_trend_positive": "Event-time trend since the directional-change state is positive for the long-only setup.",
    "adverse_reversal_absent": "Price has not reversed far enough from the event-time high to invalidate the overshoot.",
    "turnover_controlled": "Turnover is above the generated participation floor without entering an extreme local churn regime.",
    "range_quarticity_state_decay": "OHLC range-quarticity stress has recently been elevated and is now decaying.",
    "post_stress_stabilization": "Range volatility-of-volatility and current candle range have stabilized after local stress.",
    "participation_present": "Closed-candle participation remains above the generated floor during stabilization.",
    "range_not_reexpanding": "The range-quarticity stress proxy is below its recent stress peak.",
    "positive_stabilization_drift": "Closed-candle drift is not deteriorating during the stabilization window.",
    "low_directional_entropy": "Directional entropy is compressed relative to its rolling baseline.",
    "efficiency_expanding": "Range efficiency is above its rolling baseline.",
    "positive_entropy_drift": "Closed-candle drift over the entropy lookback is positive.",
    "midline_hold": "Close is above the rolling price-range midpoint.",
    "range_not_extended": "Close has not exceeded the previous rolling high, avoiding a range-extension chase.",
    "persistent_memory_regime": "Rolling Hurst proxy indicates persistent long-memory behavior.",
    "efficient_path": "Fractal path efficiency is above its rolling baseline.",
    "positive_fractal_drift": "Closed-candle drift over the fractal lookback is positive.",
    "not_range_extension": "Close has not exceeded the previous rolling high, avoiding a range-extension chase.",
    "variance_ratio_expansion": "Rolling variance ratio is at or above its local baseline, proxying a non-random-walk regime.",
    "positive_autocorr_regime": "Closed-candle returns show positive local first-lag autocorrelation.",
    "positive_regime_drift": "Closed-candle drift over the generated regime lookback is positive.",
    "controlled_regime_return": "Variance-ratio regime drift is not overextended relative to local ATR.",
    "midline_resilience": "Close holds above the rolling price-range midpoint in the variance-ratio regime.",
    "negative_funding_pressure": "Recent perpetual funding pressure is negative, proxying short-side payment pressure.",
    "funding_pressure_releasing": "Funding pressure is becoming less negative rather than worsening.",
    "price_resilience": "Price holds above the rolling price-range midpoint while funding pressure releases.",
    "not_positive_crowding": "Funding has not turned into positive long-crowding pressure.",
    "moderate_drawdown": "BTC is in a moderate drawdown from its recent high, preserving beta-carry upside without severe stress.",
    "volatility_budget": "Realized volatility is inside the generated local risk budget.",
    "positive_candle_reentry": "The current closed candle recovered above its open after the drawdown.",
    "beta_resilience": "Price holds above the generated rolling midpoint in the drawdown-control regime.",
    "participation_floor": "Closed-candle volume is above the generated minimum participation floor.",
    "not_overheated": "RSI has not reached the generated exit threshold, avoiding overheated beta exposure.",
    "mark_discount_pressure": "Traded futures close is below the locally merged mark-price fair value by the generated dislocation threshold.",
    "mark_gap_reclaiming": "The last-vs-mark dislocation is contracting on the current closed candle.",
    "six_candle_discount_reclaim": "The mark discount is closing over the last six closed candles.",
    "short_return_nonnegative": "The last three closed candles have non-negative return.",
    "mark_price_support": "The closed mark-price trend is not deteriorating beyond the generated support threshold.",
    "discount_not_extreme": "The last-vs-mark discount is large enough to matter but not in an extreme stress tail.",
    "mark_fair_value_momentum": "The 4h mark-price fair-value anchor has positive closed-candle momentum.",
    "traded_price_lag": "Traded futures price has not yet followed the mark-price fair-value move.",
    "range_budget": "The current closed-candle range is inside the fixed fair-value lag risk budget.",
    "event_cooldown": "The generated entry set preserves local event cooldown sampling instead of firing on every matching candle.",
    "spread_pressure": "Roll-style spread proxy is elevated relative to its local baseline.",
    "spread_compressing": "The Roll-style spread proxy is declining on the current closed candle.",
    "hl_spread_normalizing": "High-low spread proxy is not excessively wide relative to its local baseline.",
    "positive_recovery": "Closed-candle return is positive while the microstructure spread proxy compresses.",
    "state_stability": "The closed-candle negative-return frequency is not above its local regime baseline.",
    "volatility_state_budget": "Realized regime volatility is inside the generated local state budget.",
    "trendline_support": "Close holds above the generated slow regime trendline.",
    "closed_candle_reentry": "The current closed candle recovered above its open inside the positive regime state.",
    "drawdown_state_intact": "Drawdown from the local regime high has not broken the generated state boundary.",
    "low_realized_skewness": "Rolling realized skewness is below its local baseline, avoiding lottery-like positive skew.",
    "kurtosis_risk_premium": "Rolling realized kurtosis is above its local baseline, proxying compensated tail-shape risk.",
    "lottery_tail_cooling": "The rolling maximum 5-minute return is not elevated relative to its local baseline.",
    "positive_tail_shape_drift": "Closed-candle drift over the higher-moment lookback is positive.",
    "good_volatility_dominance": "Upside realized semivariance dominates downside realized semivariance.",
    "bad_volatility_decay": "Downside realized semivariance is below its rolling baseline.",
    "positive_semivariance_drift": "Closed-candle drift over the semivariance lookback is positive.",
    "session_window": "Candle is inside the generated UTC liquidity session window.",
    "weekday_liquidity": "Candle is on a weekday, avoiding the lower-activity weekend regime.",
    "prior_vwap_discount": "Previous close was below the same-day session VWAP.",
    "vwap_reclaim": "Close crossed back above the same-day session VWAP.",
    "controlled_atr": "ATR is not excessively expanded relative to its local mean.",
    "recent_liquidity_stress": "A volatility or illiquidity stress episode occurred inside the generated recovery horizon.",
    "liquidity_normalizing": "Illiquidity, range, and price-turn recovery components indicate normalization after stress.",
    "participation_recovered": "Closed-candle participation has recovered relative to its local baseline.",
    "below_recovery_anchor": "Price remains below the generated recovery anchor, preserving room for recovery.",
    "recovery_turn": "The current closed candle has turned upward after stress.",
    "controlled_cost_proxy": "High-low spread proxy is controlled relative to its local baseline.",
    "positive_signed_imbalance": "Rolling candle-direction signed volume imbalance is positive enough to proxy buy-side pressure.",
    "close_location_accumulation": "Closes are persistently in the upper part of their candle ranges.",
    "upper_close_location": "The current close is in the upper half of the candle range.",
    "mid_reclaim": "Close crossed back above the rolling price-range midpoint.",
    "not_breakout_chase": "Close has not exceeded the previous rolling high, avoiding a breakout chase.",
    "controlled_range": "Current candle range is not excessively expanded relative to local range.",
    "pullback_seen": "RSI has reached the generated pullback threshold within the lookback window.",
    "rsi_recovered": "RSI crossed back above the generated recovery threshold on this candle.",
    "trend_filter": "Fast EMA is compatible with the generated long-only trend filter.",
    "volume_filter": "Closed-candle volume is above the generated local volume threshold.",
    "ml_filter": "Generated FreqAI target prediction is above threshold, or always true for rule-only modes.",
    "volume_positive": "Closed-candle volume is positive.",
    "momentum_confirmed": "RSI crossed above the generated momentum confirmation threshold.",
    "atr_floor": "ATR is at or above its local ATR mean.",
    "breakout_filter": "Close is above the previous rolling high.",
    "atr_expansion": "ATR is above its local ATR mean.",
}


@dataclass(frozen=True)
class CandidateSignalDiagnosticsInputs:
    root_dir: Path
    generated_metadata_path: Path
    ohlcv_path: Path
    informative_ohlcv_path: Path | None = None
    funding_rate_path: Path | None = None
    freqai_predictions_dir: Path | None = None
    output_root: Path = Path("registry/strategies/diagnostics")
    diagnostics_id: str | None = None
    timerange: str | None = None
    entry_edge_hold_candles: int | None = None
    entry_edge_all_in_cost_bps: float | None = None
    entry_edge_min_profitable_windows_ratio: float = (
        DEFAULT_ENTRY_EDGE_MIN_PROFITABLE_WINDOWS_RATIO
    )
    reviewer_notes: list[str] = field(default_factory=list)


def diagnose_candidate_signals(inputs: CandidateSignalDiagnosticsInputs) -> dict[str, Any]:
    root = inputs.root_dir.resolve()
    metadata_path = _resolve_inside(inputs.generated_metadata_path, root)
    ohlcv_path = _resolve_inside(inputs.ohlcv_path, root)
    informative_ohlcv_path = (
        _resolve_inside(inputs.informative_ohlcv_path, root)
        if inputs.informative_ohlcv_path is not None
        else None
    )
    funding_rate_path = (
        _resolve_inside(inputs.funding_rate_path, root)
        if inputs.funding_rate_path is not None
        else None
    )
    freqai_predictions_dir = (
        _resolve_inside(inputs.freqai_predictions_dir, root)
        if inputs.freqai_predictions_dir is not None
        else None
    )
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    diagnostics_id = inputs.diagnostics_id or _diagnostics_id(generated_at)
    strategy_name = str(metadata.get("strategy_name") or metadata.get("strategy_class_name") or "unknown")
    candidate_id = str(metadata.get("candidate_id") or "unknown_candidate")
    logic_variant = str(metadata.get("strategy_logic_variant") or "mean_reversion_pullback")
    generator_mode = str(metadata.get("generator_mode") or "rule_based")
    parameters = _parameter_defaults(metadata)
    ml_target_column = _ml_target_column(generator_mode, metadata.get("target_definition"))

    diagnostics: dict[str, Any] = {
        "generated_at": generated_at,
        "factory": "candidate_signal_diagnostics",
        "diagnostics_id": diagnostics_id,
        "status": "completed",
        "strategy_name": strategy_name,
        "candidate_id": candidate_id,
        "generated_metadata_path": _rel(metadata_path, root),
        "ohlcv_path": _rel(ohlcv_path, root),
        "informative_ohlcv_path": (
            _rel(informative_ohlcv_path, root)
            if informative_ohlcv_path is not None
            else None
        ),
        "funding_rate_path": (
            _rel(funding_rate_path, root) if funding_rate_path is not None else None
        ),
        "freqai_predictions_dir": (
            _rel(freqai_predictions_dir, root) if freqai_predictions_dir is not None else None
        ),
        "timerange": inputs.timerange,
        "generator_mode": generator_mode,
        "strategy_logic_variant": logic_variant,
        "parameter_defaults": parameters,
        "prediction_merge": {
            "requested": freqai_predictions_dir is not None,
            "target_column": ml_target_column,
            "prediction_file_count": 0,
            "prediction_row_count": 0,
            "matched_row_count": 0,
            "target_column_present_after_merge": False,
        },
        "checks": [],
        "diagnosis_codes": [],
        "reviewer_notes": list(inputs.reviewer_notes),
        "safety_scope": {
            "historical_only": True,
            "freqtrade_trade_started": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
            "prediction_artifacts_read_only": True,
            "informative_ohlcv_artifacts_read_only": True,
            "funding_rate_artifacts_read_only": True,
            "local_artifacts_source_of_truth": True,
        },
    }

    checks = diagnostics["checks"]
    checks.append(_check("metadata_file_present", metadata_path.is_file()))
    checks.append(_check("ohlcv_file_present", ohlcv_path.is_file()))
    if informative_ohlcv_path is not None:
        checks.append(_check("informative_ohlcv_file_present", informative_ohlcv_path.is_file()))
    if funding_rate_path is not None:
        checks.append(_check("funding_rate_file_present", funding_rate_path.is_file()))
    if freqai_predictions_dir is not None:
        checks.append(_check("freqai_predictions_dir_present", freqai_predictions_dir.is_dir()))
    checks.append(_check("logic_variant_supported", logic_variant in SUPPORTED_LOGIC_VARIANTS))
    if any(check["status"] == "fail" for check in checks):
        diagnostics["status"] = "blocked"
        diagnostics["blockers"] = [check["name"] for check in checks if check["status"] == "fail"]
        return diagnostics

    dataframe = _load_ohlcv(ohlcv_path)
    informative_dataframe = (
        _load_ohlcv(informative_ohlcv_path)
        if informative_ohlcv_path is not None
        else None
    )
    funding_dataframe = _load_ohlcv(funding_rate_path) if funding_rate_path is not None else None
    original_rows = len(dataframe)
    dataframe = _apply_timerange(dataframe, inputs.timerange)
    if informative_dataframe is not None:
        informative_dataframe = _apply_timerange(informative_dataframe, inputs.timerange)
        dataframe, informative_ohlcv_merge = _merge_informative_ohlcv(
            dataframe,
            informative_dataframe=informative_dataframe,
            root=root,
            informative_ohlcv_path=informative_ohlcv_path,
            base_timeframe=str(metadata.get("timeframe") or "5m"),
        )
    else:
        informative_ohlcv_merge = {
            "requested": False,
            "informative_ohlcv_row_count": 0,
            "matched_row_count": 0,
            "eth_log_return_column_present_after_merge": False,
        }
    diagnostics["informative_ohlcv_merge"] = informative_ohlcv_merge
    if funding_dataframe is not None:
        dataframe, funding_merge = _merge_funding_rate(
            dataframe,
            funding_dataframe=funding_dataframe,
            root=root,
            funding_rate_path=funding_rate_path,
            base_timeframe=str(metadata.get("timeframe") or "5m"),
        )
    else:
        funding_merge = {
            "requested": False,
            "funding_rate_row_count": 0,
            "matched_row_count": 0,
            "funding_rate_column_present_after_merge": False,
        }
    diagnostics["funding_rate_merge"] = funding_merge
    if logic_variant == "crowding_unwind_reaccumulation":
        dataframe, structural_data_merge = _merge_local_crowding_context(
            dataframe,
            root=root,
            base_timeframe=str(metadata.get("timeframe") or "5m"),
        )
        diagnostics["structural_data_merge"] = structural_data_merge
        checks.extend(
            [
                _check(
                    "crowding_open_interest_file_present",
                    bool(structural_data_merge["open_interest"]["exists"]),
                ),
                _check(
                    "crowding_long_short_ratio_file_present",
                    bool(structural_data_merge["long_short_ratio"]["exists"]),
                ),
                _check(
                    "crowding_open_interest_column_present_after_merge",
                    bool(structural_data_merge["open_interest_column_present_after_merge"]),
                ),
                _check(
                    "crowding_long_short_ratio_column_present_after_merge",
                    bool(structural_data_merge["long_short_ratio_column_present_after_merge"]),
                ),
            ]
        )
    else:
        diagnostics["structural_data_merge"] = {"requested": False}
    diagnostics["row_count"] = len(dataframe)
    diagnostics["source_row_count"] = original_rows
    checks.append(_check("ohlcv_has_rows", len(dataframe) > 0))
    checks.append(_check("required_ohlcv_columns_present", _required_columns_present(dataframe)))
    if any(check["status"] == "fail" for check in checks):
        diagnostics["status"] = "blocked"
        diagnostics["blockers"] = [check["name"] for check in checks if check["status"] == "fail"]
        return diagnostics

    if freqai_predictions_dir is not None:
        dataframe, prediction_merge = _merge_freqai_predictions(
            dataframe,
            freqai_predictions_dir=freqai_predictions_dir,
            root=root,
            target_column=ml_target_column,
        )
        diagnostics["prediction_merge"] = prediction_merge

    dataframe = _with_indicators(dataframe, parameters)
    if logic_variant == "crowding_unwind_reaccumulation":
        dataframe = _with_crowding_features(dataframe)
    ml_target_column_present = (
        None if ml_target_column is None else ml_target_column in dataframe.columns
    )
    component_masks = _entry_component_masks(
        dataframe,
        logic_variant=logic_variant,
        generator_mode=generator_mode,
        target_definition=metadata.get("target_definition"),
        prediction_threshold=metadata.get("prediction_threshold"),
        parameters=parameters,
    )
    component_order = list(component_masks)
    entry_mask = _combine(component_masks.values(), dataframe.index)
    cumulative: dict[str, Any] = {}
    running = pd.Series(True, index=dataframe.index)
    for name in component_order:
        running = running & component_masks[name]
        cumulative[name] = _count(running)

    individual = {name: _count(mask) for name, mask in component_masks.items()}
    all_except = {
        name: _count(_combine([mask for other, mask in component_masks.items() if other != name], dataframe.index))
        for name in component_order
    }
    first_zero = next((name for name in component_order if cumulative[name] == 0), None)
    rarest_component = min(individual, key=lambda name: individual[name]) if individual else None
    entry_count = _count(entry_mask)
    generated_entry_edge = _entry_edge_diagnostics(
        dataframe,
        entry_mask,
        hold_candles=_entry_edge_hold_candles(inputs.entry_edge_hold_candles, parameters),
        all_in_cost_bps=_entry_edge_cost_bps(inputs.entry_edge_all_in_cost_bps, metadata),
        min_profitable_windows_ratio=inputs.entry_edge_min_profitable_windows_ratio,
    )
    diagnosis_codes = _diagnosis_codes(
        entry_count=entry_count,
        row_count=len(dataframe),
        generator_mode=generator_mode,
        ml_target_column_present=ml_target_column_present,
        generated_entry_edge=generated_entry_edge,
    )
    bottleneck_components = sorted(
        (
            {
                "name": name,
                "individual_count": individual[name],
                "cumulative_count": cumulative[name],
                "all_except_count": all_except[name],
                "description": COMPONENT_DESCRIPTIONS.get(name, ""),
            }
            for name in component_order
        ),
        key=lambda item: (-int(item["all_except_count"]), int(item["individual_count"]), item["name"]),
    )
    component_counts = {
        name: {
            "description": COMPONENT_DESCRIPTIONS.get(name, ""),
            "individual_count": individual[name],
            "individual_ratio": _ratio(individual[name], len(dataframe)),
            "cumulative_count": cumulative[name],
            "cumulative_ratio": _ratio(cumulative[name], len(dataframe)),
            "all_except_count": all_except[name],
            "all_except_ratio": _ratio(all_except[name], len(dataframe)),
        }
        for name in component_order
    }
    diagnostics.update(
        {
            "entry_count": entry_count,
            "entry_signal_count": entry_count,
            "entry_signal_ratio": _ratio(entry_count, len(dataframe)),
            "zero_entry_signal": entry_count == 0,
            "component_order": component_order,
            "component_counts": component_counts,
            "components": component_counts,
            "bottleneck_components": bottleneck_components,
            "first_zero_component": first_zero,
            "rarest_component": rarest_component,
            "ml_filter": {
                "generator_mode": generator_mode,
                "target_column": ml_target_column,
                "target_column_present": ml_target_column_present,
            },
            "generated_entry_edge": generated_entry_edge,
            "entry_edge": generated_entry_edge,
            "diagnosis_codes": diagnosis_codes,
            "diagnosis": {
                "primary_zero_component": first_zero,
                "rarest_component": rarest_component,
                "rarest_component_count": individual.get(rarest_component) if rarest_component else None,
                "codes": diagnosis_codes,
                "message": _diagnosis_message(entry_count, first_zero, rarest_component),
            },
        }
    )
    return diagnostics


def write_signal_diagnostics_artifacts(
    diagnostics: dict[str, Any], *, root_dir: Path, output_root: Path
) -> tuple[Path, Path]:
    root = root_dir.resolve()
    strategy = _safe_path_component(str(diagnostics.get("strategy_name") or "unknown_strategy"))
    candidate = _safe_path_component(str(diagnostics.get("candidate_id") or "unknown_candidate"))
    diagnostics_id = _safe_path_component(str(diagnostics.get("diagnostics_id") or "diagnostics"))
    out_dir = _resolve_inside(output_root, root) / strategy / candidate / diagnostics_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "signal_diagnostics.json"
    report_path = out_dir / "signal_diagnostics_report.md"
    json_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(diagnostics), encoding="utf-8")
    return json_path, report_path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _parameter_defaults(metadata: dict[str, Any]) -> dict[str, float]:
    raw = metadata.get("parameter_defaults")
    if not isinstance(raw, dict):
        raw = {}
    defaults: dict[str, float] = {
        "buy_rsi_window": 14,
        "buy_pullback_lookback": 5,
        "buy_rsi_pullback": 32,
        "buy_rsi_recovery": 42,
        "buy_ema_fast": 12,
        "buy_ema_slow": 48,
        "buy_volume_window": 24,
        "buy_volume_factor": 1.0,
        "sell_rsi_exit": 65,
        "sell_timeout_candles": 96,
    }
    for key, default in list(defaults.items()):
        try:
            defaults[key] = float(raw.get(key, default))
        except (TypeError, ValueError):
            defaults[key] = float(default)
    return defaults


def _merge_freqai_predictions(
    dataframe: pd.DataFrame,
    *,
    freqai_predictions_dir: Path,
    root: Path,
    target_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_files = _prediction_files(freqai_predictions_dir)
    summary: dict[str, Any] = {
        "requested": True,
        "predictions_dir": _rel(freqai_predictions_dir, root),
        "target_column": target_column,
        "prediction_file_count": len(prediction_files),
        "prediction_files": [_rel(path, root) for path in prediction_files],
        "prediction_row_count": 0,
        "matched_row_count": 0,
        "target_column_present_after_merge": False,
        "errors": [],
    }
    if not prediction_files or "date" not in dataframe.columns:
        return dataframe, summary

    frames: list[pd.DataFrame] = []
    for prediction_file in prediction_files:
        try:
            prediction_frame = _load_prediction_file(prediction_file)
        except Exception as exc:  # pragma: no cover - defensive artifact diagnostics.
            summary["errors"].append(f"{_rel(prediction_file, root)}: {exc}")
            continue
        if "date" not in prediction_frame.columns:
            summary["errors"].append(f"{_rel(prediction_file, root)}: missing date column")
            continue
        frames.append(prediction_frame)
    if not frames:
        return dataframe, summary

    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    predictions = predictions.sort_values("date").drop_duplicates("date", keep="last")
    prediction_columns = [column for column in predictions.columns if column != "date"]
    if not prediction_columns:
        return dataframe, summary

    base = dataframe.copy()
    base["date"] = pd.to_datetime(base["date"], utc=True)
    base = base.drop(columns=[column for column in prediction_columns if column in base.columns])
    merged = base.merge(predictions[["date", *prediction_columns]], on="date", how="left")
    summary["prediction_row_count"] = int(len(predictions))
    if target_column and target_column in merged.columns:
        summary["target_column_present_after_merge"] = True
        summary["matched_row_count"] = int(merged[target_column].notna().sum())
    return merged, summary


def _prediction_files(predictions_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*_prediction.feather", "*.feather", "*.csv"):
        files.extend(sorted(predictions_dir.glob(pattern)))
    unique: dict[Path, None] = {}
    for path in files:
        unique[path] = None
    return list(unique)


def _load_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_feather(path)


def _load_ohlcv(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataframe = pd.read_parquet(path)
    elif suffix == ".csv":
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported OHLCV input format: {path}")
    if "date" in dataframe.columns:
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
        dataframe = dataframe.sort_values("date").reset_index(drop=True)
    return dataframe


def _merge_funding_rate(
    dataframe: pd.DataFrame,
    *,
    funding_dataframe: pd.DataFrame,
    root: Path,
    funding_rate_path: Path,
    base_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = {
        "requested": True,
        "funding_rate_path": _rel(funding_rate_path, root),
        "funding_rate_row_count": int(len(funding_dataframe)),
        "matched_row_count": 0,
        "funding_rate_column_present_after_merge": False,
    }
    if dataframe.empty or funding_dataframe.empty or "date" not in dataframe.columns:
        return dataframe, summary
    if "date" not in funding_dataframe.columns or "open" not in funding_dataframe.columns:
        return dataframe, summary

    funding = funding_dataframe.copy()
    funding["date"] = pd.to_datetime(funding["date"], utc=True)
    funding["funding_rate_raw"] = funding["open"].astype(float)
    funding["funding_rate_mean_raw"] = (
        funding["funding_rate_raw"].rolling(6, min_periods=1).mean()
    )
    funding["funding_rate_abs_mean_raw"] = (
        funding["funding_rate_raw"].abs().rolling(6, min_periods=1).mean()
    )
    base_minutes = _timeframe_to_minutes(base_timeframe)
    funding["date_merge"] = (
        funding["date"] + pd.to_timedelta(8 * 60 - base_minutes, unit="m")
    )

    base = dataframe.copy()
    base["date"] = pd.to_datetime(base["date"], utc=True)
    merged = pd.merge_ordered(
        base,
        funding[
            [
                "date_merge",
                "funding_rate_raw",
                "funding_rate_mean_raw",
                "funding_rate_abs_mean_raw",
            ]
        ],
        fill_method="ffill",
        left_on="date",
        right_on="date_merge",
        how="left",
    ).drop(columns=["date_merge"])
    merged["funding_rate"] = merged["funding_rate_raw"].fillna(0.0)
    merged["funding_rate_mean"] = merged["funding_rate_mean_raw"].fillna(0.0)
    merged["funding_rate_abs_mean"] = merged["funding_rate_abs_mean_raw"].fillna(0.0)
    summary["matched_row_count"] = int(merged["funding_rate_raw"].notna().sum())
    summary["funding_rate_column_present_after_merge"] = True
    return merged, summary


def _merge_informative_ohlcv(
    dataframe: pd.DataFrame,
    *,
    informative_dataframe: pd.DataFrame,
    root: Path,
    informative_ohlcv_path: Path,
    base_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    is_mark_price = "mark" in informative_ohlcv_path.stem.lower()
    summary = {
        "requested": True,
        "informative_ohlcv_path": _rel(informative_ohlcv_path, root),
        "informative_ohlcv_row_count": int(len(informative_dataframe)),
        "matched_row_count": 0,
        "eth_log_return_column_present_after_merge": False,
        "mark_close_column_present_after_merge": False,
    }
    if dataframe.empty or informative_dataframe.empty or "date" not in dataframe.columns:
        return dataframe, summary
    if "date" not in informative_dataframe.columns or "close" not in informative_dataframe.columns:
        return dataframe, summary

    informative = informative_dataframe.copy()
    informative["date"] = pd.to_datetime(informative["date"], utc=True)
    if is_mark_price:
        informative["mark_close_raw"] = informative["close"].astype(float)
        informative["mark_log_return_raw"] = np.log(
            informative["mark_close_raw"] / informative["mark_close_raw"].shift(1)
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        informative["mark_price_return_bps_raw"] = (
            informative["mark_close_raw"] / informative["mark_close_raw"].shift(1) - 1.0
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 10000.0
        base_minutes = _timeframe_to_minutes(base_timeframe)
        informative["date_merge"] = (
            informative["date"] + pd.to_timedelta(4 * 60 - base_minutes, unit="m")
        )
        base = dataframe.copy()
        base["date"] = pd.to_datetime(base["date"], utc=True)
        merged = pd.merge_ordered(
            base,
            informative[
                ["date_merge", "mark_close_raw", "mark_log_return_raw", "mark_price_return_bps_raw"]
            ],
            fill_method="ffill",
            left_on="date",
            right_on="date_merge",
            how="left",
        ).drop(columns=["date_merge"])
        summary["matched_row_count"] = int(merged["mark_close_raw"].notna().sum())
        summary["mark_close_column_present_after_merge"] = True
        merged["mark_close"] = merged["mark_close_raw"].fillna(merged["close"])
        merged["mark_log_return"] = merged["mark_log_return_raw"].fillna(0.0)
        merged["mark_price_return_bps"] = merged["mark_price_return_bps_raw"].fillna(0.0)
        return merged, summary

    informative["eth_close_raw"] = informative["close"].astype(float)
    informative["eth_log_return_raw"] = np.log(
        informative["eth_close_raw"] / informative["eth_close_raw"].shift(1)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if "volume" in informative.columns:
        informative["eth_volume_raw"] = informative["volume"].astype(float)
    else:
        informative["eth_volume_raw"] = 0.0

    base = dataframe.copy()
    base["date"] = pd.to_datetime(base["date"], utc=True)
    merged = base.merge(
        informative[["date", "eth_close_raw", "eth_log_return_raw", "eth_volume_raw"]],
        on="date",
        how="left",
    )
    summary["matched_row_count"] = int(merged["eth_close_raw"].notna().sum())
    summary["eth_log_return_column_present_after_merge"] = True
    merged["eth_close"] = merged["eth_close_raw"].fillna(0.0)
    merged["eth_log_return"] = merged["eth_log_return_raw"].fillna(0.0)
    merged["eth_volume"] = merged["eth_volume_raw"].fillna(0.0)
    return merged, summary


def _merge_local_crowding_context(
    dataframe: pd.DataFrame,
    *,
    root: Path,
    base_timeframe: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    open_interest_path = root / LOCAL_CROWDING_OPEN_INTEREST_PATH
    long_short_ratio_path = root / LOCAL_CROWDING_LONG_SHORT_RATIO_PATH
    open_interest, open_interest_summary = _load_structural_context(
        open_interest_path,
        root=root,
        columns=["open_interest"],
    )
    long_short_ratio, long_short_ratio_summary = _load_structural_context(
        long_short_ratio_path,
        root=root,
        columns=["long_account_ratio", "short_account_ratio", "long_short_ratio"],
    )
    summary: dict[str, Any] = {
        "requested": True,
        "context_merge_semantics": "closed_context_candle_availability_v1",
        "open_interest": open_interest_summary,
        "long_short_ratio": long_short_ratio_summary,
        "open_interest_column_present_after_merge": False,
        "long_short_ratio_column_present_after_merge": False,
    }
    merged = dataframe.copy()
    base_minutes = _timeframe_to_minutes(base_timeframe)
    if not open_interest.empty:
        merged, matched = _merge_structural_context(
            merged,
            open_interest,
            columns=["open_interest"],
            base_minutes=base_minutes,
        )
        summary["open_interest"]["matched_row_count"] = matched
        summary["open_interest_column_present_after_merge"] = "open_interest" in merged.columns
    if not long_short_ratio.empty:
        merged, matched = _merge_structural_context(
            merged,
            long_short_ratio,
            columns=["long_account_ratio", "short_account_ratio", "long_short_ratio"],
            base_minutes=base_minutes,
        )
        summary["long_short_ratio"]["matched_row_count"] = matched
        summary["long_short_ratio_column_present_after_merge"] = (
            "long_short_ratio" in merged.columns
        )
    return merged, summary


def _load_structural_context(
    path: Path,
    *,
    root: Path,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "path": _rel(path, root),
        "exists": path.is_file(),
        "row_count": 0,
        "required_columns": list(columns),
        "required_columns_present": False,
        "matched_row_count": 0,
        "error": None,
    }
    if not path.is_file():
        return pd.DataFrame(columns=["date", *columns]), summary
    try:
        if path.suffix.lower() == ".csv":
            context = pd.read_csv(path)
        else:
            context = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive artifact diagnostics.
        summary["error"] = str(exc)
        return pd.DataFrame(columns=["date", *columns]), summary
    summary["row_count"] = int(len(context))
    if "date" not in context.columns or not set(columns).issubset(context.columns):
        return pd.DataFrame(columns=["date", *columns]), summary
    summary["required_columns_present"] = True
    selected = context[["date", *columns]].copy()
    selected["date"] = pd.to_datetime(selected["date"], utc=True, errors="coerce")
    selected = selected.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected, summary


def _merge_structural_context(
    dataframe: pd.DataFrame,
    context: pd.DataFrame,
    *,
    columns: list[str],
    base_minutes: int,
) -> tuple[pd.DataFrame, int]:
    if dataframe.empty or context.empty or "date" not in dataframe.columns:
        return dataframe, 0
    base = dataframe.copy()
    base["date"] = pd.to_datetime(base["date"], utc=True)
    context_frame = context.copy()
    context_minutes = _infer_candle_minutes(context_frame["date"], fallback=60)
    context_frame["date_merge"] = context_frame["date"] + pd.to_timedelta(
        context_minutes - base_minutes,
        unit="m",
    )
    merged = pd.merge_asof(
        base.sort_values("date"),
        context_frame.sort_values("date_merge")[["date_merge", *columns]],
        left_on="date",
        right_on="date_merge",
        direction="backward",
    ).drop(columns=["date_merge"], errors="ignore")
    matched = int(merged[columns[0]].notna().sum()) if columns[0] in merged.columns else 0
    return merged, matched


def _infer_candle_minutes(date_series: pd.Series, *, fallback: int) -> int:
    dates = pd.to_datetime(date_series, utc=True, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return fallback
    diffs = dates.diff().dropna()
    if diffs.empty:
        return fallback
    minutes = int(round(float(diffs.median().total_seconds()) / 60.0))
    return max(minutes, 1)


def _timeframe_to_minutes(value: str) -> int:
    text = str(value or "5m").strip().lower()
    try:
        amount = int(text[:-1])
    except ValueError:
        return 5
    unit = text[-1:]
    if unit == "m":
        return max(amount, 1)
    if unit == "h":
        return max(amount * 60, 1)
    if unit == "d":
        return max(amount * 1440, 1)
    return 5


def _apply_timerange(dataframe: pd.DataFrame, timerange: str | None) -> pd.DataFrame:
    if not timerange or "date" not in dataframe.columns:
        return dataframe.copy()
    start_raw, end_raw = timerange.split("-", 1)
    start = pd.Timestamp(start_raw, tz="UTC")
    end = pd.Timestamp(end_raw, tz="UTC")
    mask = (dataframe["date"] >= start) & (dataframe["date"] < end)
    return dataframe.loc[mask].reset_index(drop=True)


def _required_columns_present(dataframe: pd.DataFrame) -> bool:
    return {"open", "high", "low", "close", "volume"}.issubset(dataframe.columns)


def _with_indicators(dataframe: pd.DataFrame, parameters: dict[str, float]) -> pd.DataFrame:
    df = dataframe.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    rsi_window = int(parameters["buy_rsi_window"])
    fast = int(parameters["buy_ema_fast"])
    slow = int(parameters["buy_ema_slow"])
    volume_window = int(parameters["buy_volume_window"])
    lookback = int(parameters["buy_pullback_lookback"])
    df["rsi"] = _rsi(close, rsi_window)
    df["ema_fast"] = close.ewm(span=fast, adjust=False, min_periods=1).mean()
    df["ema_slow"] = close.ewm(span=slow, adjust=False, min_periods=1).mean()
    df["volume_mean"] = volume.rolling(volume_window, min_periods=1).mean()
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.rolling(14, min_periods=1).mean()
    df["atr_mean"] = df["atr"].rolling(24, min_periods=1).mean()
    df["rolling_high"] = close.rolling(lookback, min_periods=1).max()
    df["rolling_low"] = close.rolling(lookback, min_periods=1).min()
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], utc=True)
    else:
        date_series = pd.Series(df.index, index=df.index)
    df["hour_utc"] = date_series.dt.hour
    df["weekday"] = date_series.dt.dayofweek
    session_key = date_series.dt.strftime("%Y-%m-%d")
    typical_price = (high + low + close) / 3.0
    cumulative_pv = (typical_price * volume).groupby(session_key).cumsum()
    cumulative_volume = volume.groupby(session_key).cumsum().replace(0, 1)
    df["session_vwap"] = cumulative_pv / cumulative_volume
    candle_direction = ((close > df["open"].astype(float)).astype(int)
                        - (close < df["open"].astype(float)).astype(int))
    df["signed_volume"] = volume * candle_direction
    rolling_signed_volume = df["signed_volume"].rolling(lookback, min_periods=1).sum()
    rolling_volume = volume.rolling(lookback, min_periods=1).sum().replace(0, 1)
    df["signed_volume_imbalance"] = rolling_signed_volume / rolling_volume
    candle_range = (high - low).replace(0, 1e-9)
    df["close_location_value"] = ((close - low) - (high - close)) / candle_range
    df["close_location_mean"] = df["close_location_value"].rolling(
        lookback, min_periods=1
    ).mean()
    df["range_pct"] = candle_range / close.replace(0, 1)
    df["range_pct_mean"] = df["range_pct"].rolling(24, min_periods=1).mean()
    df["rolling_mid"] = (df["rolling_high"] + df["rolling_low"]) / 2.0
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    if "funding_rate_mean" not in df.columns:
        df["funding_rate_mean"] = 0.0
    if "funding_rate_abs_mean" not in df.columns:
        df["funding_rate_abs_mean"] = 0.0
    df["funding_rate"] = df["funding_rate"].astype(float).fillna(0.0)
    df["funding_rate_mean"] = df["funding_rate_mean"].astype(float).fillna(0.0)
    df["funding_rate_abs_mean"] = df["funding_rate_abs_mean"].astype(float).fillna(0.0)
    df["funding_pressure"] = df["funding_rate"].rolling(12, min_periods=1).mean()
    df["funding_pressure_delta"] = df["funding_pressure"].diff().fillna(0.0)
    if "mark_close" not in df.columns:
        df["mark_close"] = close
    if "mark_log_return" not in df.columns:
        df["mark_log_return"] = 0.0
    if "mark_price_return_bps" not in df.columns:
        df["mark_price_return_bps"] = 0.0
    df["mark_close"] = df["mark_close"].astype(float).fillna(close).replace(0, np.nan)
    df["mark_log_return"] = df["mark_log_return"].astype(float).fillna(0.0)
    df["mark_price_return_bps"] = df["mark_price_return_bps"].astype(float).fillna(0.0)
    df["mark_price_gap"] = ((close - df["mark_close"]) / df["mark_close"]).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["mark_price_gap_delta"] = df["mark_price_gap"].diff().fillna(0.0)
    df["mark_price_gap_delta_6"] = (df["mark_price_gap"] - df["mark_price_gap"].shift(6)).fillna(
        0.0
    )
    df["return_3"] = (close / close.shift(3) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(
        0.0
    )
    df["traded_lag_return_bps"] = (close / close.shift(lookback) - 1.0).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0) * 10000.0
    volume_std = volume.rolling(volume_window, min_periods=1).std().replace(0, np.nan)
    df["volume_zscore"] = ((volume - df["volume_mean"]) / volume_std).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["mark_price_gap_mean"] = df["mark_price_gap"].rolling(volume_window, min_periods=1).mean()
    df["mark_price_gap_abs_mean"] = df["mark_price_gap"].abs().rolling(
        volume_window, min_periods=1
    ).mean().replace(0, 1e-9)
    df["mark_price_trend"] = (df["mark_close"] / df["mark_close"].shift(lookback) - 1.0).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    direction_up_probability = (
        (close.diff().fillna(0.0) > 0).astype(float)
        .rolling(lookback, min_periods=1)
        .mean()
        .clip(0.001, 0.999)
    )
    df["direction_entropy"] = -(
        direction_up_probability * np.log(direction_up_probability)
        + (1.0 - direction_up_probability) * np.log(1.0 - direction_up_probability)
    )
    df["direction_entropy_baseline"] = df["direction_entropy"].rolling(
        volume_window, min_periods=1
    ).mean()
    range_sum = candle_range.rolling(lookback, min_periods=1).sum().replace(0, 1e-9)
    df["range_efficiency"] = (close - close.shift(lookback)).abs() / range_sum
    df["range_efficiency_mean"] = df["range_efficiency"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["entropy_drift"] = close / close.shift(lookback) - 1.0
    log_return = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["log_return"] = log_return
    dc_threshold = (
        (df["atr"] / close.replace(0, np.nan))
        .rolling(lookback, min_periods=1)
        .mean()
        .clip(lower=0.0025, upper=0.0300)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0060)
    dc_high = close.rolling(lookback, min_periods=1).max()
    dc_low = close.rolling(lookback, min_periods=1).min()
    pullback_from_high = (
        close / dc_high.replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    rebound_from_low = (
        close / dc_low.replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["directional_change_threshold"] = dc_threshold
    df["directional_change_state"] = np.select(
        [rebound_from_low >= dc_threshold, pullback_from_high <= -dc_threshold],
        [1.0, -1.0],
        default=0.0,
    )
    df["directional_change_state"] = (
        df["directional_change_state"].replace(0.0, np.nan).ffill().fillna(0.0)
    )
    df["directional_change_event"] = (
        (df["directional_change_state"] != df["directional_change_state"].shift(1))
        & (df["directional_change_state"] != 0.0)
    ).astype(int)
    df["bar_index"] = np.arange(len(df), dtype=float)
    df["directional_change_event_index"] = np.where(
        df["directional_change_event"] > 0,
        df["bar_index"],
        np.nan,
    )
    df["directional_change_event_index"] = df["directional_change_event_index"].ffill()
    df["directional_change_event_age"] = (
        df["bar_index"] - df["directional_change_event_index"]
    ).replace([np.inf, -np.inf], lookback + 1).fillna(lookback + 1)
    df["directional_change_extreme"] = np.where(
        df["directional_change_state"] > 0.0,
        dc_low,
        np.where(df["directional_change_state"] < 0.0, dc_high, close),
    )
    bullish_overshoot = close / df["directional_change_extreme"].replace(0, np.nan) - 1.0
    bearish_overshoot = df["directional_change_extreme"] / close.replace(0, np.nan) - 1.0
    df["overshoot_return"] = np.where(
        df["directional_change_state"] >= 0.0,
        bullish_overshoot,
        bearish_overshoot,
    )
    df["overshoot_return"] = (
        df["overshoot_return"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    df["overshoot_ratio"] = (
        df["overshoot_return"] / dc_threshold.replace(0, np.nan)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["overshoot_length"] = df["directional_change_event_age"]
    event_time_window = max(3, lookback // 3)
    df["event_time_trend"] = (
        log_return.rolling(event_time_window, min_periods=1).sum()
        * df["directional_change_state"]
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    bullish_reversal = close / dc_high.replace(0, np.nan) - 1.0
    bearish_reversal = dc_low / close.replace(0, np.nan) - 1.0
    df["adverse_reversal_distance"] = np.where(
        df["directional_change_state"] >= 0.0,
        bullish_reversal,
        bearish_reversal,
    )
    df["adverse_reversal_distance"] = (
        df["adverse_reversal_distance"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    df["turnover_proxy"] = (
        volume / df["volume_mean"].replace(0, 1e-9)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    range_state_window = volume_window
    range_state_lookback = lookback
    range_min_periods = min(range_state_lookback, 8)
    df["ohlc_range"] = df["range_pct"]
    safe_high = df["high"].astype(float).replace(0, np.nan)
    safe_low = df["low"].astype(float).replace(0, np.nan)
    df["range_return"] = np.log(safe_high / safe_low).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["range_quarticity_proxy"] = df["range_return"].pow(4).rolling(
        range_state_lookback, min_periods=range_min_periods
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["range_quarticity_mean"] = df["range_quarticity_proxy"].rolling(
        range_state_window * 2, min_periods=range_state_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-12)
    df["range_quarticity_ratio"] = (
        df["range_quarticity_proxy"] / df["range_quarticity_mean"]
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["range_quarticity_delta"] = df["range_quarticity_ratio"].diff().fillna(0.0)
    df["range_volatility"] = df["range_pct"].rolling(
        range_state_lookback, min_periods=range_min_periods
    ).std().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["range_volatility_mean"] = df["range_volatility"].rolling(
        range_state_window * 2, min_periods=range_state_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
    df["range_vol_of_vol_state"] = (
        df["range_volatility"] / df["range_volatility_mean"]
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    range_decay_window = max(3, range_state_lookback // 3)
    df["range_state_decay"] = (
        df["range_quarticity_ratio"]
        / df["range_quarticity_ratio"].shift(range_decay_window).replace(0, np.nan)
    ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["range_stress_ratio"] = (
        df["range_quarticity_ratio"] * df["range_vol_of_vol_state"]
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["range_stress_recent"] = df["range_stress_ratio"].rolling(
        range_state_lookback, min_periods=1
    ).max().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["participation_recovery"] = (
        volume / df["volume_mean"].replace(0, 1e-9)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stabilization_window = max(3, range_state_lookback // 3)
    df["stabilization_drift"] = (
        close / close.shift(stabilization_window) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    jump_min_periods = min(lookback, 12)
    abs_log_return = log_return.abs()
    df["realized_variance_fast"] = (
        log_return.pow(2).rolling(lookback, min_periods=jump_min_periods).sum()
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["bipower_variation"] = (
        (np.pi / 2.0)
        * (abs_log_return * abs_log_return.shift(1))
        .rolling(lookback, min_periods=jump_min_periods)
        .sum()
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["jump_variation"] = (
        df["realized_variance_fast"] - df["bipower_variation"]
    ).clip(lower=0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["jump_variation_mean"] = df["jump_variation"].rolling(
        lookback * 2, min_periods=jump_min_periods
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
    df["jump_variation_ratio"] = (
        df["jump_variation"] / df["jump_variation_mean"]
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    decay_shift = max(2, lookback // 3)
    df["continuous_variance_decay"] = (
        df["bipower_variation"] / df["bipower_variation"].shift(decay_shift).replace(0, 1e-9)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["positive_jump_event"] = (
        (log_return > 0.0)
        & (df["jump_variation_ratio"] > 1.25)
        & (df["jump_variation"] > 0.0)
    ).astype(int)
    post_jump_window = max(3, lookback // 4)
    df["post_jump_drift"] = (
        close / close.shift(post_jump_window) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    follow_through_window = max(3, lookback // 3)
    df["jump_follow_through"] = (
        close / close.rolling(follow_through_window, min_periods=3).max().shift(1) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["jump_overextension"] = (
        close / df["rolling_high"].shift(1).replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    variance_min_periods = min(lookback, 8)
    one_step_variance = log_return.rolling(
        lookback, min_periods=variance_min_periods
    ).var().replace(0, 1e-12)
    multi_step_return = np.log(close / close.shift(lookback)).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    multi_step_variance = multi_step_return.rolling(
        lookback, min_periods=variance_min_periods
    ).var()
    df["variance_ratio"] = (
        multi_step_variance / (one_step_variance * lookback)
    ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    df["variance_ratio_mean"] = df["variance_ratio"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["variance_ratio_delta"] = df["variance_ratio"].diff().fillna(0.0)
    df["return_autocorr"] = log_return.rolling(
        lookback, min_periods=variance_min_periods
    ).corr(log_return.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["autocorr_mean"] = df["return_autocorr"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["regime_drift"] = close / close.shift(lookback) - 1.0
    normalized_atr = (
        df["atr"] / close.replace(0, 1)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0).rolling(
        lookback, min_periods=1
    ).mean().replace(0, 1e-9)
    df["normalized_regime_return"] = (
        df["regime_drift"] / normalized_atr
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    beta_risk_window = volume_window * 4
    beta_volatility_min_periods = min(volume_window, 8)
    df["realized_volatility"] = log_return.rolling(
        volume_window, min_periods=beta_volatility_min_periods
    ).std().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["realized_volatility_mean"] = df["realized_volatility"].rolling(
        beta_risk_window, min_periods=volume_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
    df["market_beta_high"] = close.rolling(beta_risk_window, min_periods=1).max()
    df["market_beta_drawdown"] = (
        close / df["market_beta_high"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["market_beta_drift"] = (close / close.shift(volume_window) - 1.0).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    regime_fast_window = lookback
    regime_state_window = volume_window
    regime_slow_window = regime_state_window * 2
    regime_min_periods = min(regime_state_window, 8)
    df["regime_return_fast"] = (close / close.shift(regime_fast_window) - 1.0).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["regime_return_slow"] = (close / close.shift(regime_slow_window) - 1.0).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["regime_negative_frequency"] = (
        (log_return < 0.0)
        .astype(float)
        .rolling(regime_state_window, min_periods=regime_min_periods)
        .mean()
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["regime_negative_frequency_mean"] = df[
        "regime_negative_frequency"
    ].rolling(
        regime_slow_window, min_periods=regime_state_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.5).replace(0, 1e-9)
    df["regime_volatility"] = log_return.rolling(
        regime_state_window, min_periods=regime_min_periods
    ).std().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["regime_volatility_mean"] = df["regime_volatility"].rolling(
        regime_slow_window, min_periods=regime_state_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
    df["regime_trendline"] = close.rolling(
        regime_slow_window, min_periods=regime_state_window
    ).mean()
    df["regime_high"] = close.rolling(
        regime_slow_window, min_periods=regime_state_window
    ).max()
    # Reset pandas block fragmentation before adding the broad diagnostic feature set.
    df = df.copy()
    df["regime_drawdown"] = (
        close / df["regime_high"].replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if "eth_close" not in df.columns:
        df["eth_close"] = 0.0
    if "eth_log_return" not in df.columns:
        df["eth_log_return"] = 0.0
    if "eth_volume" not in df.columns:
        df["eth_volume"] = 0.0
    df["eth_close"] = df["eth_close"].astype(float).fillna(0.0)
    df["eth_log_return"] = df["eth_log_return"].astype(float).fillna(0.0)
    df["eth_volume"] = df["eth_volume"].astype(float).fillna(0.0)
    df["btc_log_return"] = log_return
    df["eth_lead_return"] = df["eth_log_return"].shift(1).fillna(0.0)
    df["eth_lead_return_mean"] = df["eth_lead_return"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["eth_btc_return_spread"] = df["eth_lead_return"] - df["btc_log_return"]
    df["eth_btc_spread_mean"] = df["eth_btc_return_spread"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["eth_btc_spread_abs_mean"] = df["eth_btc_return_spread"].abs().rolling(
        volume_window, min_periods=1
    ).mean()
    safe_eth_close = df["eth_close"].replace(0, np.nan)
    safe_btc_close = close.replace(0, np.nan)
    df["btc_eth_log_ratio"] = np.log(safe_btc_close / safe_eth_close).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["btc_eth_ratio_mean"] = df["btc_eth_log_ratio"].rolling(
        volume_window, min_periods=1
    ).mean()
    btc_eth_ratio_std = df["btc_eth_log_ratio"].rolling(
        volume_window, min_periods=2
    ).std().replace(0, 1e-9)
    df["btc_eth_ratio_zscore"] = (
        (df["btc_eth_log_ratio"] - df["btc_eth_ratio_mean"]) / btc_eth_ratio_std
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["btc_eth_ratio_zscore_delta"] = df["btc_eth_ratio_zscore"].diff().fillna(0.0)
    df["eth_regime_drift"] = (
        df["eth_close"] / df["eth_close"].shift(lookback) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    corr_min_periods = min(volume_window, 8)
    df["btc_eth_return_corr"] = log_return.rolling(
        volume_window, min_periods=corr_min_periods
    ).corr(df["eth_log_return"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["btc_eth_corr_mean"] = df["btc_eth_return_corr"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["btc_eth_corr_delta"] = df["btc_eth_return_corr"].diff().fillna(0.0)
    df["btc_eth_relative_return"] = log_return - df["eth_log_return"]
    df["btc_eth_relative_return_mean"] = df["btc_eth_relative_return"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["cross_asset_drift"] = close / close.shift(lookback) - 1.0

    def _hurst_rs_proxy(values: np.ndarray) -> float:
        if len(values) < 8:
            return np.nan
        centered = values - np.nanmean(values)
        cumulative = np.nancumsum(centered)
        value_range = np.nanmax(cumulative) - np.nanmin(cumulative)
        value_std = np.nanstd(values)
        if value_range <= 0 or value_std <= 1e-12:
            return 0.5
        return float(np.clip(np.log(value_range / value_std) / np.log(len(values)), 0.0, 1.0))

    hurst_min_periods = min(lookback, 8)
    df["hurst_proxy"] = log_return.rolling(lookback, min_periods=hurst_min_periods).apply(
        _hurst_rs_proxy, raw=True
    )
    df["hurst_baseline"] = df["hurst_proxy"].rolling(volume_window, min_periods=1).mean()
    path_length = close.diff().abs().rolling(lookback, min_periods=1).sum().replace(0, 1e-9)
    df["fractal_efficiency"] = (close - close.shift(lookback)).abs() / path_length
    df["fractal_efficiency_mean"] = df["fractal_efficiency"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["fractal_drift"] = close / close.shift(lookback) - 1.0
    upside_squared_return = log_return.clip(lower=0.0).pow(2)
    downside_squared_return = log_return.clip(upper=0.0).pow(2)
    df["upside_semivariance"] = upside_squared_return.rolling(
        lookback, min_periods=1
    ).mean()
    df["downside_semivariance"] = downside_squared_return.rolling(
        lookback, min_periods=1
    ).mean()
    df["downside_semivariance_mean"] = df["downside_semivariance"].rolling(
        volume_window, min_periods=1
    ).mean()
    semivariance_total = (df["upside_semivariance"] + df["downside_semivariance"]).replace(
        0, 1e-12
    )
    df["semivariance_balance"] = (
        df["upside_semivariance"] - df["downside_semivariance"]
    ) / semivariance_total
    df["semivariance_balance_mean"] = df["semivariance_balance"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["semivariance_drift"] = close / close.shift(lookback) - 1.0
    higher_moment_min_periods = min(lookback, 4)
    df["realized_skewness"] = log_return.rolling(
        lookback, min_periods=higher_moment_min_periods
    ).skew().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["realized_kurtosis"] = log_return.rolling(
        lookback, min_periods=higher_moment_min_periods
    ).kurt().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["realized_skewness_mean"] = df["realized_skewness"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["realized_kurtosis_mean"] = df["realized_kurtosis"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["max_return"] = log_return.rolling(lookback, min_periods=1).max()
    df["max_return_mean"] = df["max_return"].rolling(volume_window, min_periods=1).mean()
    df["min_return"] = log_return.rolling(lookback, min_periods=1).min()
    df["tail_shape_drift"] = close / close.shift(lookback) - 1.0
    df["calendar_turnover_ratio"] = (
        volume / df["volume_mean"].replace(0, 1)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["calendar_turnover_ratio_mean"] = df["calendar_turnover_ratio"].rolling(
        volume_window, min_periods=1
    ).mean()
    weekend_mask = df["weekday"].isin([5, 6])
    weekday_mask = df["weekday"] < 5
    df["weekend_turnover_baseline"] = (
        df["calendar_turnover_ratio"]
        .where(weekend_mask)
        .rolling(288, min_periods=1)
        .mean()
        .ffill()
        .fillna(1.0)
    )
    df["weekday_turnover_baseline"] = (
        df["calendar_turnover_ratio"]
        .where(weekday_mask)
        .rolling(288, min_periods=1)
        .mean()
        .ffill()
        .fillna(1.0)
    )
    df["calendar_drift"] = close / close.shift(lookback) - 1.0
    spread_min_periods = min(lookback, 8)
    spread_baseline_window = volume_window * 2
    return_autocovariance = (log_return * log_return.shift(1)).rolling(
        lookback, min_periods=spread_min_periods
    ).mean()
    df["return_autocovariance"] = return_autocovariance.replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["roll_spread_proxy"] = (
        2.0 * np.sqrt((-return_autocovariance).clip(lower=0.0))
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["roll_spread_mean"] = df["roll_spread_proxy"].rolling(
        spread_baseline_window, min_periods=volume_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
    df["roll_spread_delta"] = df["roll_spread_proxy"].diff().fillna(0.0)
    df["hl_spread_proxy"] = df["range_pct"].replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    df["hl_spread_mean"] = df["hl_spread_proxy"].rolling(
        spread_baseline_window, min_periods=volume_window
    ).mean().replace([np.inf, -np.inf], 0.0).fillna(0.0).replace(0, 1e-9)
    short_noise = log_return.abs().rolling(lookback, min_periods=1).mean()
    long_noise = log_return.abs().rolling(
        spread_baseline_window, min_periods=lookback
    ).mean().replace(0, 1e-9)
    df["microstructure_noise_ratio"] = (
        short_noise / long_noise
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["dollar_volume"] = (close.abs() * volume.abs()).replace(0, 1e-9)
    df["amihud_illiquidity"] = (
        log_return.abs() / df["dollar_volume"] * 1e9
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["amihud_illiquidity_mean"] = df["amihud_illiquidity"].rolling(
        volume_window, min_periods=1
    ).mean()
    df["amihud_illiquidity_delta"] = df["amihud_illiquidity"].diff().fillna(0.0)
    df["illiquidity_drift"] = close / close.shift(lookback) - 1.0
    df["amihud_illiquidity_ratio"] = (
        df["amihud_illiquidity"] / df["amihud_illiquidity_mean"].replace(0, 1e-9)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["range_recovery_ratio"] = (
        df["range_pct"] / df["range_pct_mean"].replace(0, 1e-9)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["volume_recovery_ratio"] = (
        volume / df["volume_mean"].replace(0, 1e-9)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    liquidity_stress_flag = (
        (df["amihud_illiquidity_ratio"] > 1.35)
        | (df["range_recovery_ratio"] > 1.30)
    ).astype(float)
    df["liquidity_stress_recent"] = liquidity_stress_flag.rolling(
        lookback, min_periods=1
    ).max()
    liquidity_normalized = (
        (df["amihud_illiquidity_ratio"] <= 1.05)
        & (df["range_recovery_ratio"] <= 1.15)
    ).astype(float)
    participation_recovered = (
        df["volume_recovery_ratio"] >= float(parameters["buy_volume_factor"])
    ).astype(float)
    price_recovery_turn = (close.diff().fillna(0.0) > 0.0).astype(float)
    df["liquidity_recovery_score"] = (
        liquidity_normalized + participation_recovered + price_recovery_turn
    )
    df["liquidity_recovery_anchor"] = (
        df["rolling_mid"] + close.rolling(volume_window, min_periods=1).mean()
    ) / 2.0
    df["recovery_horizon_return"] = (
        close / close.shift(lookback) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def _with_crowding_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    for column in [
        "open_interest",
        "long_account_ratio",
        "short_account_ratio",
        "long_short_ratio",
    ]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    interest = df["open_interest"].astype(float)
    ratio = df["long_short_ratio"].astype(float)
    df["open_interest_delta_pct_288"] = (
        interest / interest.shift(288).replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 100.0
    ratio_mean = ratio.rolling(864, min_periods=864).mean()
    ratio_std = ratio.rolling(864, min_periods=864).std().replace(0, np.nan)
    df["long_short_ratio_zscore_864"] = (
        (ratio - ratio_mean) / ratio_std
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sma = close.rolling(144, min_periods=144).mean()
    df["sma_distance_bps_144"] = (
        close / sma.replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 10000.0
    volume_mean = volume.rolling(288, min_periods=288).mean()
    volume_std = volume.rolling(288, min_periods=288).std().replace(0, np.nan)
    df["volume_zscore_288"] = (
        (volume - volume_mean) / volume_std
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def _entry_component_masks(
    dataframe: pd.DataFrame,
    *,
    logic_variant: str,
    generator_mode: str,
    target_definition: Any,
    prediction_threshold: Any,
    parameters: dict[str, float],
) -> dict[str, pd.Series]:
    volume_positive = _bool(dataframe["volume"].astype(float) > 0)
    volume_filter = _bool(
        dataframe["volume"].astype(float)
        > dataframe["volume_mean"].astype(float) * parameters["buy_volume_factor"]
    )
    quiet_volume = _bool(
        dataframe["volume"].astype(float)
        < dataframe["volume_mean"].astype(float) * parameters["buy_volume_factor"]
    )
    ml_filter = _ml_filter(dataframe, generator_mode, target_definition, prediction_threshold)
    if logic_variant == "amihud_illiquidity_premium":
        return {
            "price_impact_premium": _bool(
                dataframe["amihud_illiquidity"] > dataframe["amihud_illiquidity_mean"]
            ),
            "illiquidity_releasing": _bool(dataframe["amihud_illiquidity_delta"] < 0.0),
            "not_extreme_impact": _bool(
                dataframe["amihud_illiquidity"]
                <= dataframe["amihud_illiquidity_mean"] * 3.0
            ),
            "price_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "positive_illiquidity_drift": _bool(dataframe["illiquidity_drift"] > 0.0),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.5),
            "volume_floor": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "bipower_jump_decay":
        jump_window = max(2, int(float(parameters["buy_pullback_lookback"])) // 4)
        return {
            "positive_jump_detected": _bool(
                dataframe["positive_jump_event"].rolling(jump_window, min_periods=1).max()
                > 0
            ),
            "jump_dominates_continuous_variance": _bool(
                dataframe["jump_variation_ratio"] > 1.25
            ),
            "continuous_variance_decaying": _bool(
                dataframe["continuous_variance_decay"] < 0.95
            ),
            "post_jump_drift_positive": _bool(dataframe["post_jump_drift"] > 0.0),
            "not_overextended_after_jump": _bool(
                dataframe["jump_overextension"] < 0.018
            ),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "directional_change_overshoot":
        event_age_limit = max(3, int(float(parameters["sell_timeout_candles"])))
        return {
            "directional_change_confirmed": _bool(
                (dataframe["directional_change_state"] > 0.0)
                & (dataframe["directional_change_event_age"] >= 1.0)
                & (dataframe["directional_change_event_age"] <= event_age_limit)
            ),
            "overshoot_persisted": _bool(
                (dataframe["overshoot_ratio"] >= 1.05)
                & (dataframe["overshoot_ratio"] <= 4.0)
                & (dataframe["overshoot_length"] >= 2.0)
            ),
            "event_time_trend_positive": _bool(dataframe["event_time_trend"] > 0.0),
            "adverse_reversal_absent": _bool(
                dataframe["adverse_reversal_distance"]
                >= -dataframe["directional_change_threshold"] * 0.90
            ),
            "turnover_controlled": _bool(
                (dataframe["turnover_proxy"] >= parameters["buy_volume_factor"])
                & (dataframe["turnover_proxy"] <= 4.0)
            ),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "range_quarticity_vol_of_vol_state":
        return {
            "range_quarticity_state_decay": _bool(
                (dataframe["range_stress_recent"] >= 1.05)
                & (dataframe["range_state_decay"] <= 1.05)
                & (dataframe["range_quarticity_delta"] <= 0.25)
            ),
            "post_stress_stabilization": _bool(
                (dataframe["range_vol_of_vol_state"] <= 1.35)
                & (dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.60)
            ),
            "participation_present": _bool(
                dataframe["participation_recovery"] >= parameters["buy_volume_factor"]
            ),
            "range_not_reexpanding": _bool(
                dataframe["range_stress_ratio"]
                <= dataframe["range_stress_recent"] * 0.98
            ),
            "positive_stabilization_drift": _bool(
                dataframe["stabilization_drift"] > -0.002
            ),
            "turnover_controlled": _bool(
                (dataframe["turnover_proxy"] >= parameters["buy_volume_factor"])
                & (dataframe["turnover_proxy"] <= 4.0)
            ),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "calendar_turnover_seasonality":
        return {
            "calendar_risk_window": _bool(dataframe["weekday"].isin([0, 3])),
            "weekend_discount_context": _bool(
                dataframe["weekend_turnover_baseline"]
                <= dataframe["weekday_turnover_baseline"]
            ),
            "turnover_recovery": _bool(
                dataframe["calendar_turnover_ratio"]
                >= dataframe["calendar_turnover_ratio_mean"] * parameters["buy_volume_factor"]
            ),
            "positive_calendar_drift": _bool(dataframe["calendar_drift"] > 0.0),
            "midline_hold": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.4),
            "not_breakout_chase": _bool(
                dataframe["close"] <= dataframe["rolling_high"].shift(1)
            ),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "cross_asset_lead_lag":
        return {
            "eth_positive_lead": _bool(
                dataframe["eth_lead_return"] > dataframe["eth_lead_return_mean"]
            ),
            "btc_lag_discount": _bool(dataframe["eth_btc_return_spread"] > 0.0),
            "spread_not_extreme": _bool(
                dataframe["eth_btc_return_spread"]
                <= dataframe["eth_btc_spread_mean"] + dataframe["eth_btc_spread_abs_mean"] * 2.0
            ),
            "btc_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "positive_cross_asset_drift": _bool(dataframe["cross_asset_drift"] > 0.0),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.5),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "cross_asset_cointegration_spread":
        return {
            "btc_discount_to_eth": _bool(dataframe["btc_eth_ratio_zscore"] < -0.50),
            "spread_reversion_turn": _bool(dataframe["btc_eth_ratio_zscore_delta"] > 0.0),
            "eth_market_support": _bool(dataframe["eth_regime_drift"] > 0.0),
            "btc_resilience": _bool(dataframe["close"] > dataframe["rolling_low"].shift(1)),
            "cointegration_spread_not_extreme": _bool(
                dataframe["btc_eth_ratio_zscore"].abs() <= 2.5
            ),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.5),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "cross_asset_correlation_recovery":
        return {
            "correlation_breakdown": _bool(dataframe["btc_eth_corr_mean"] < 0.35),
            "correlation_recovery": _bool(
                (dataframe["btc_eth_return_corr"] > dataframe["btc_eth_corr_mean"])
                & (dataframe["btc_eth_corr_delta"] > 0.0)
            ),
            "btc_relative_recovery": _bool(
                dataframe["btc_eth_relative_return"] > dataframe["btc_eth_relative_return_mean"]
            ),
            "eth_market_support": _bool(dataframe["eth_regime_drift"] > 0.0),
            "btc_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.5),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "downside_liquidity_shock_reversal":
        lookback = int(parameters["buy_pullback_lookback"])
        lookback_return = dataframe["close"].astype(float) / dataframe["close"].shift(lookback) - 1.0
        normalized_atr = (
            dataframe["atr"].astype(float) / dataframe["close"].astype(float)
        ).rolling(lookback, min_periods=1).mean()
        return {
            "downside_shock": _bool(lookback_return <= -(normalized_atr * 1.5)),
            "rsi_washout": _bool(
                dataframe["rsi"].rolling(lookback, min_periods=1).min()
                <= parameters["buy_rsi_pullback"]
            ),
            "rsi_recovered": _crossed_above(dataframe["rsi"], parameters["buy_rsi_recovery"]),
            "quiet_volume": quiet_volume,
            "local_low_reclaim": _bool(dataframe["close"] > dataframe["rolling_low"].shift(1)),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "entropy_regime_transition":
        return {
            "low_directional_entropy": _bool(
                dataframe["direction_entropy"]
                <= dataframe["direction_entropy_baseline"] * 0.85
            ),
            "efficiency_expanding": _bool(
                dataframe["range_efficiency"] > dataframe["range_efficiency_mean"]
            ),
            "positive_entropy_drift": _bool(dataframe["entropy_drift"] > 0.0),
            "midline_hold": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "range_not_extended": _bool(
                dataframe["close"] <= dataframe["rolling_high"].shift(1)
            ),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "fractal_long_memory_regime":
        return {
            "persistent_memory_regime": _bool(dataframe["hurst_proxy"] > 0.52),
            "efficient_path": _bool(
                dataframe["fractal_efficiency"]
                > dataframe["fractal_efficiency_mean"] * 1.05
            ),
            "positive_fractal_drift": _bool(dataframe["fractal_drift"] > 0.0),
            "midline_hold": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "not_range_extension": _bool(
                dataframe["close"] <= dataframe["rolling_high"].shift(1)
            ),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "variance_ratio_regime_switch":
        return {
            "variance_ratio_expansion": _bool(
                dataframe["variance_ratio"] >= dataframe["variance_ratio_mean"] * 0.98
            ),
            "positive_autocorr_regime": _bool(
                (dataframe["return_autocorr"] > 0.0)
                & (dataframe["return_autocorr"] >= dataframe["autocorr_mean"])
            ),
            "positive_regime_drift": _bool(dataframe["regime_drift"] > 0.0),
            "controlled_regime_return": _bool(
                dataframe["normalized_regime_return"] <= 2.5
            ),
            "midline_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.5),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "crowding_unwind_reaccumulation":
        return {
            "open_interest_unwinding": _bool(
                dataframe["open_interest_delta_pct_288"] <= -0.75
            ),
            "short_account_reaccumulation": _bool(
                dataframe["long_short_ratio_zscore_864"] <= -0.75
            ),
            "price_above_sma": _bool(dataframe["sma_distance_bps_144"] >= 0.0),
            "volume_participation_floor": _bool(
                dataframe["volume_zscore_288"] >= -0.25
            ),
            "price_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(
                dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.4
            ),
            "not_overheated": _bool(dataframe["rsi"] < parameters["sell_rsi_exit"]),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "funding_pressure_carry":
        return {
            "negative_funding_pressure": _bool(dataframe["funding_pressure"] < 0.0),
            "funding_pressure_releasing": _bool(dataframe["funding_pressure_delta"] > 0.0),
            "price_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "not_positive_crowding": _bool(
                dataframe["funding_rate"]
                <= dataframe["funding_rate_mean"] + dataframe["funding_rate_abs_mean"] * 0.25
            ),
            "controlled_range": _bool(
                dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.4
            ),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "market_beta_drawdown_carry":
        return {
            "moderate_drawdown": _bool(
                (dataframe["market_beta_drawdown"] <= -0.005)
                & (dataframe["market_beta_drawdown"] >= -0.055)
            ),
            "volatility_budget": _bool(
                dataframe["realized_volatility"]
                <= dataframe["realized_volatility_mean"] * 1.45
            ),
            "positive_candle_reentry": _bool(dataframe["close"] > dataframe["open"]),
            "beta_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "participation_floor": volume_filter,
            "not_overheated": _bool(dataframe["rsi"] < parameters["sell_rsi_exit"]),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "mark_price_dislocation_reclaim":
        return {
            "mark_discount_pressure": _bool(dataframe["mark_price_gap"] <= -0.006),
            "mark_gap_reclaiming": _bool(dataframe["mark_price_gap_delta"] > 0.0),
            "mark_price_support": _bool(dataframe["mark_price_trend"] > -0.005),
            "discount_not_extreme": _bool(dataframe["mark_price_gap"] >= -0.035),
            "price_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.6),
            "participation_floor": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "mark_discount_reclaim_continuation":
        return {
            "mark_discount_pressure": _bool(dataframe["mark_price_gap"] <= -0.0005),
            "six_candle_discount_reclaim": _bool(dataframe["mark_price_gap_delta_6"] >= 0.0001),
            "short_return_nonnegative": _bool(dataframe["return_3"] >= 0.0),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "mark_fair_value_momentum_lag":
        raw_event = (
            _bool(dataframe["mark_price_return_bps"] >= 25.0)
            & _bool(dataframe["traded_lag_return_bps"] <= 0.0)
            & _bool(dataframe["range_pct"] <= 0.008)
            & _bool(dataframe["volume_zscore"] >= -1.0)
            & _bool(ml_filter)
            & _bool(volume_positive)
        )
        return {
            "mark_fair_value_momentum": _bool(dataframe["mark_price_return_bps"] >= 25.0),
            "traded_price_lag": _bool(dataframe["traded_lag_return_bps"] <= 0.0),
            "range_budget": _bool(dataframe["range_pct"] <= 0.008),
            "participation_floor": _bool(dataframe["volume_zscore"] >= -1.0),
            "event_cooldown": _cooldown_mask(
                raw_event,
                cooldown_candles=int(float(parameters["buy_pullback_lookback"])),
            ),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "microstructure_spread_reversion":
        return {
            "spread_pressure": _bool(
                dataframe["roll_spread_proxy"] > dataframe["roll_spread_mean"] * 1.20
            ),
            "spread_compressing": _bool(dataframe["roll_spread_delta"] < 0.0),
            "hl_spread_normalizing": _bool(
                dataframe["hl_spread_proxy"] <= dataframe["hl_spread_mean"] * 1.50
            ),
            "price_resilience": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "positive_recovery": _bool(dataframe["log_return"] > 0.0),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.8),
            "participation_floor": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "regime_state_reentry":
        return {
            "positive_regime_drift": _bool(
                (dataframe["regime_return_fast"] > 0.0)
                & (dataframe["regime_return_slow"] > 0.0)
            ),
            "state_stability": _bool(
                dataframe["regime_negative_frequency"]
                <= dataframe["regime_negative_frequency_mean"] * 1.10
            ),
            "volatility_state_budget": _bool(
                dataframe["regime_volatility"] <= dataframe["regime_volatility_mean"] * 1.60
            ),
            "trendline_support": _bool(dataframe["close"] > dataframe["regime_trendline"]),
            "closed_candle_reentry": _bool(dataframe["close"] > dataframe["open"]),
            "drawdown_state_intact": _bool(dataframe["regime_drawdown"] >= -0.030),
            "participation_floor": volume_filter,
            "not_overheated": _bool(dataframe["rsi"] < parameters["sell_rsi_exit"]),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "realized_skewness_tail_shape":
        return {
            "low_realized_skewness": _bool(
                dataframe["realized_skewness"] < dataframe["realized_skewness_mean"]
            ),
            "kurtosis_risk_premium": _bool(
                dataframe["realized_kurtosis"] > dataframe["realized_kurtosis_mean"]
            ),
            "lottery_tail_cooling": _bool(
                dataframe["max_return"] <= dataframe["max_return_mean"] * 1.10
            ),
            "positive_tail_shape_drift": _bool(dataframe["tail_shape_drift"] > 0.0),
            "midline_hold": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.5),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "semivariance_asymmetry_regime":
        return {
            "good_volatility_dominance": _bool(dataframe["semivariance_balance"] > 0.05),
            "bad_volatility_decay": _bool(
                dataframe["downside_semivariance"]
                < dataframe["downside_semivariance_mean"] * 0.95
            ),
            "positive_semivariance_drift": _bool(dataframe["semivariance_drift"] > 0.0),
            "midline_hold": _bool(dataframe["close"] > dataframe["rolling_mid"]),
            "controlled_range": _bool(dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.4),
            "not_range_extension": _bool(
                dataframe["close"] <= dataframe["rolling_high"].shift(1)
            ),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "intraday_session_liquidity_reclaim":
        return {
            "session_window": _bool(dataframe["hour_utc"].between(13, 20)),
            "weekday_liquidity": _bool(dataframe["weekday"] < 5),
            "prior_vwap_discount": _bool(
                dataframe["close"].shift(1) < dataframe["session_vwap"].shift(1)
            ),
            "vwap_reclaim": _crossed_above(dataframe["close"], dataframe["session_vwap"]),
            "volume_filter": volume_filter,
            "controlled_atr": _bool(dataframe["atr"] <= dataframe["atr_mean"] * 1.5),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "liquidity_recovery_horizon":
        return {
            "recent_liquidity_stress": _bool(dataframe["liquidity_stress_recent"] > 0.0),
            "liquidity_normalizing": _bool(dataframe["liquidity_recovery_score"] >= 2.0),
            "participation_recovered": _bool(
                dataframe["volume_recovery_ratio"] >= float(parameters["buy_volume_factor"])
            ),
            "below_recovery_anchor": _bool(
                dataframe["close"] < dataframe["liquidity_recovery_anchor"]
            ),
            "recovery_turn": _bool(dataframe["close"] > dataframe["close"].shift(1)),
            "controlled_cost_proxy": _bool(
                dataframe["hl_spread_proxy"] <= dataframe["hl_spread_mean"] * 1.15
            ),
            "positive_recovery_horizon": _bool(dataframe["recovery_horizon_return"] > -0.015),
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "signed_volume_imbalance_accumulation":
        return {
            "positive_signed_imbalance": _bool(
                dataframe["signed_volume_imbalance"].astype(float) > 0.18
            ),
            "close_location_accumulation": _bool(
                dataframe["close_location_mean"].astype(float) > 0.20
            ),
            "upper_close_location": _bool(
                dataframe["close_location_value"].astype(float) > 0.0
            ),
            "mid_reclaim": _crossed_above(dataframe["close"], dataframe["rolling_mid"]),
            "not_breakout_chase": _bool(dataframe["close"] <= dataframe["rolling_high"].shift(1)),
            "controlled_range": _bool(
                dataframe["range_pct"] <= dataframe["range_pct_mean"] * 1.6
            ),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "trend_continuation":
        return {
            "trend_filter": _bool(dataframe["ema_fast"] > dataframe["ema_slow"]),
            "momentum_confirmed": _crossed_above(dataframe["rsi"], parameters["buy_rsi_recovery"]),
            "atr_floor": _bool(dataframe["atr"] >= dataframe["atr_mean"]),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    if logic_variant == "volatility_breakout":
        return {
            "breakout_filter": _bool(dataframe["close"] > dataframe["rolling_high"].shift(1)),
            "atr_expansion": _bool(dataframe["atr"] > dataframe["atr_mean"]),
            "volume_filter": volume_filter,
            "ml_filter": ml_filter,
            "volume_positive": volume_positive,
        }
    return {
        "pullback_seen": _bool(
            dataframe["rsi"].rolling(int(parameters["buy_pullback_lookback"]), min_periods=1).min()
            <= parameters["buy_rsi_pullback"]
        ),
        "rsi_recovered": _crossed_above(dataframe["rsi"], parameters["buy_rsi_recovery"]),
        "trend_filter": _bool(dataframe["ema_fast"] >= dataframe["ema_slow"]),
        "volume_filter": volume_filter,
        "ml_filter": ml_filter,
        "volume_positive": volume_positive,
    }


def _ml_filter(
    dataframe: pd.DataFrame, generator_mode: str, target_definition: Any, prediction_threshold: Any
) -> pd.Series:
    if generator_mode not in ML_GENERATOR_MODES:
        return pd.Series(True, index=dataframe.index)
    target_column = _ml_target_column(generator_mode, target_definition)
    threshold = float(prediction_threshold or 0.0)
    if target_column is None or target_column not in dataframe.columns:
        return pd.Series(False, index=dataframe.index)
    return _bool(dataframe[target_column].astype(float) > threshold)


def _ml_target_column(generator_mode: str, target_definition: Any) -> str | None:
    if generator_mode not in ML_GENERATOR_MODES:
        return None
    target_name = str(target_definition or "future_return")
    return f"&-{target_name}"


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss_safe = average_loss.where(average_loss != 0)
    rs = average_gain / average_loss_safe
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    return rsi.fillna(50.0).astype(float)


def _crossed_above(series: pd.Series, threshold: float) -> pd.Series:
    return _bool((series.shift(1) <= threshold) & (series > threshold))


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _cooldown_mask(mask: pd.Series, *, cooldown_candles: int) -> pd.Series:
    values = _bool(mask).to_numpy()
    allowed = np.zeros(len(values), dtype=bool)
    next_allowed_index = 0
    cooldown = max(1, int(cooldown_candles))
    for row_index in np.flatnonzero(values):
        row_index = int(row_index)
        if row_index < next_allowed_index:
            continue
        allowed[row_index] = True
        next_allowed_index = row_index + cooldown
    return pd.Series(allowed, index=mask.index)


def _combine(masks: Any, index: pd.Index) -> pd.Series:
    combined = pd.Series(True, index=index)
    for mask in masks:
        combined = combined & _bool(mask)
    return combined


def _count(mask: pd.Series) -> int:
    return int(_bool(mask).sum())


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _entry_edge_hold_candles(input_value: int | None, parameters: dict[str, float]) -> int:
    if input_value is not None:
        return max(1, int(input_value))
    return max(1, int(float(parameters.get("sell_timeout_candles", 1))))


def _entry_edge_cost_bps(input_value: float | None, metadata: dict[str, Any]) -> float:
    if input_value is not None:
        return max(0.0, float(input_value))
    overrides = metadata.get("parameter_overrides")
    if isinstance(overrides, dict):
        try:
            return max(0.0, float(overrides.get("all_in_cost_bps", 0.0)))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _entry_edge_diagnostics(
    dataframe: pd.DataFrame,
    entry_mask: pd.Series,
    *,
    hold_candles: int,
    all_in_cost_bps: float,
    min_profitable_windows_ratio: float,
    window_count: int = DEFAULT_ENTRY_EDGE_WINDOW_COUNT,
) -> dict[str, Any]:
    hold = max(1, int(hold_candles))
    cost = max(0.0, float(all_in_cost_bps))
    windows = max(1, int(window_count))
    minimum_ratio = max(0.0, min(1.0, float(min_profitable_windows_ratio)))
    summary: dict[str, Any] = {
        "status": "not_applicable",
        "hold_candles": hold,
        "all_in_cost_bps": _round_float(cost),
        "min_profitable_windows_ratio": _round_float(minimum_ratio),
        "window_count": windows,
        "sample_count": 0,
        "expected_edge_bps": None,
        "median_edge_bps": None,
        "net_edge_bps": None,
        "win_rate_before_cost": None,
        "win_rate_after_cost": None,
        "profitable_windows_ratio": 0.0,
        "windows_with_entries": 0,
        "windows": [],
    }
    if dataframe.empty or "close" not in dataframe.columns:
        return summary

    close = dataframe["close"].astype(float)
    forward_return_bps = ((close.shift(-hold) / close) - 1.0).replace(
        [np.inf, -np.inf], np.nan
    ) * 10000.0
    eligible_mask = _bool(entry_mask) & forward_return_bps.notna()
    event_returns = forward_return_bps.loc[eligible_mask].astype(float)
    summary["sample_count"] = int(len(event_returns))
    summary["windows"] = _entry_edge_windows(
        dataframe,
        eligible_mask,
        forward_return_bps,
        hold_candles=hold,
        all_in_cost_bps=cost,
        window_count=windows,
    )
    summary["windows_with_entries"] = sum(
        1 for window in summary["windows"] if int(window.get("sample_count") or 0) > 0
    )
    profitable_windows = sum(
        1
        for window in summary["windows"]
        if int(window.get("sample_count") or 0) > 0
        and (window.get("net_edge_bps") or 0.0) > 0.0
    )
    summary["profitable_windows_ratio"] = _round_float(profitable_windows / windows)
    if event_returns.empty:
        return summary

    net_returns = event_returns - cost
    expected_edge = float(event_returns.mean())
    net_edge = float(net_returns.mean())
    summary.update(
        {
            "status": (
                "pass"
                if net_edge > 0.0
                and float(summary["profitable_windows_ratio"]) >= minimum_ratio
                else "fail"
            ),
            "expected_edge_bps": _round_float(expected_edge),
            "median_edge_bps": _round_float(float(event_returns.median())),
            "net_edge_bps": _round_float(net_edge),
            "win_rate_before_cost": _round_float(float((event_returns > 0.0).mean())),
            "win_rate_after_cost": _round_float(float((net_returns > 0.0).mean())),
        }
    )
    return summary


def _entry_edge_windows(
    dataframe: pd.DataFrame,
    eligible_mask: pd.Series,
    forward_return_bps: pd.Series,
    *,
    hold_candles: int,
    all_in_cost_bps: float,
    window_count: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    row_indices = np.array_split(np.arange(len(dataframe)), max(1, int(window_count)))
    for index, indices in enumerate(row_indices, start=1):
        if len(indices) == 0:
            windows.append(_entry_edge_window_summary(index, pd.Series(dtype=float), None, None, all_in_cost_bps))
            continue
        window_mask = eligible_mask.iloc[indices]
        returns = forward_return_bps.iloc[indices].loc[window_mask].astype(float)
        start_date = _date_value(dataframe.iloc[int(indices[0])].get("date"))
        end_date = _date_value(dataframe.iloc[int(indices[-1])].get("date"))
        windows.append(
            _entry_edge_window_summary(
                index,
                returns,
                start_date,
                end_date,
                all_in_cost_bps,
                hold_candles=hold_candles,
            )
        )
    return windows


def _entry_edge_window_summary(
    index: int,
    returns: pd.Series,
    start_date: str | None,
    end_date: str | None,
    all_in_cost_bps: float,
    *,
    hold_candles: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "window": int(index),
        "start": start_date,
        "end": end_date,
        "hold_candles": hold_candles,
        "sample_count": int(len(returns)),
        "expected_edge_bps": None,
        "net_edge_bps": None,
        "win_rate_after_cost": None,
    }
    if returns.empty:
        return summary
    net_returns = returns - all_in_cost_bps
    summary.update(
        {
            "expected_edge_bps": _round_float(float(returns.mean())),
            "net_edge_bps": _round_float(float(net_returns.mean())),
            "win_rate_after_cost": _round_float(float((net_returns > 0.0).mean())),
        }
    )
    return summary


def _date_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _diagnosis_message(entry_count: int, first_zero: str | None, rarest: str | None) -> str:
    if entry_count > 0:
        return "Entry conditions produced signals; inspect counts before promotion."
    if first_zero:
        return f"Entry conditions produced zero signals; cumulative evaluation first reached zero at {first_zero}."
    if rarest:
        return f"Entry conditions produced zero signals; rarest individual condition was {rarest}."
    return "Entry conditions produced zero signals, but no component masks were available."


def _diagnosis_codes(
    *,
    entry_count: int,
    row_count: int,
    generator_mode: str,
    ml_target_column_present: bool | None,
    generated_entry_edge: dict[str, Any],
) -> list[str]:
    codes: list[str] = []
    if entry_count == 0:
        codes.append("ZERO_ENTRY_SIGNALS")
    elif _ratio(entry_count, row_count) < LOW_ENTRY_SIGNAL_RATIO:
        codes.append("LOW_ENTRY_SIGNALS")
    edge_status = str(generated_entry_edge.get("status") or "")
    if entry_count > 0 and edge_status == "not_applicable":
        codes.append("GENERATED_ENTRY_EDGE_UNAVAILABLE")
    if edge_status == "fail":
        if (generated_entry_edge.get("net_edge_bps") or 0.0) <= 0.0:
            codes.append("GENERATED_ENTRY_EDGE_NEGATIVE_AFTER_COST")
        if (
            generated_entry_edge.get("profitable_windows_ratio") or 0.0
        ) < (generated_entry_edge.get("min_profitable_windows_ratio") or 0.0):
            codes.append("GENERATED_ENTRY_EDGE_WINDOW_FRAGILE")
    if generator_mode in ML_GENERATOR_MODES and ml_target_column_present is False:
        codes.append("ML_FILTER_UNAVAILABLE")
    return codes


def _check(name: str, passed: bool) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail"}


def _render_report(diagnostics: dict[str, Any]) -> str:
    lines = [
        "# Candidate Signal Diagnostics",
        "",
        f"- diagnostics_id: {diagnostics.get('diagnostics_id')}",
        f"- strategy: {diagnostics.get('strategy_name')}",
        f"- candidate_id: {diagnostics.get('candidate_id')}",
        f"- status: {diagnostics.get('status')}",
        f"- logic_variant: {diagnostics.get('strategy_logic_variant')}",
        f"- row_count: {diagnostics.get('row_count')}",
        f"- entry_signal_count: {diagnostics.get('entry_signal_count')}",
        f"- zero_entry_signal: {diagnostics.get('zero_entry_signal')}",
        f"- diagnosis_codes: {', '.join(diagnostics.get('diagnosis_codes') or [])}",
        "",
        "## Diagnosis",
        "",
        f"- {diagnostics.get('diagnosis', {}).get('message')}",
        "",
        "## Generated Entry Edge",
        "",
    ]
    entry_edge = diagnostics.get("generated_entry_edge") or {}
    lines.append(f"- status: {entry_edge.get('status')}")
    lines.append(f"- hold_candles: {entry_edge.get('hold_candles')}")
    lines.append(f"- sample_count: {entry_edge.get('sample_count')}")
    lines.append(f"- all_in_cost_bps: {entry_edge.get('all_in_cost_bps')}")
    lines.append(f"- expected_edge_bps: {entry_edge.get('expected_edge_bps')}")
    lines.append(f"- net_edge_bps: {entry_edge.get('net_edge_bps')}")
    lines.append(
        "- profitable_windows_ratio: "
        f"{entry_edge.get('profitable_windows_ratio')}"
    )
    lines.extend([
        "",
        "## Prediction Merge",
        "",
    ])
    prediction_merge = diagnostics.get("prediction_merge") or {}
    lines.append(f"- requested: {prediction_merge.get('requested')}")
    lines.append(f"- target_column: {prediction_merge.get('target_column')}")
    lines.append(f"- prediction_file_count: {prediction_merge.get('prediction_file_count')}")
    lines.append(f"- prediction_row_count: {prediction_merge.get('prediction_row_count')}")
    lines.append(f"- matched_row_count: {prediction_merge.get('matched_row_count')}")
    lines.append(
        "- target_column_present_after_merge: "
        f"{prediction_merge.get('target_column_present_after_merge')}"
    )
    if prediction_merge.get("errors"):
        lines.append(f"- errors: {'; '.join(prediction_merge.get('errors') or [])}")
    lines.extend(
        [
            "",
            "## Components",
            "",
        ]
    )
    for name, counts in diagnostics.get("component_counts", {}).items():
        lines.append(
            "- "
            f"{name}: individual={counts.get('individual_count')}, "
            f"cumulative={counts.get('cumulative_count')}, "
            f"all_except={counts.get('all_except_count')} - "
            f"{counts.get('description')}"
        )
    if diagnostics.get("bottleneck_components"):
        lines.extend(["", "## Bottlenecks", ""])
        for item in diagnostics["bottleneck_components"][:5]:
            lines.append(
                "- "
                f"{item.get('name')}: individual={item.get('individual_count')}, "
                f"cumulative={item.get('cumulative_count')}, "
                f"all_except={item.get('all_except_count')}"
            )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Diagnostic only; no backtest, paper, dry-run, live, exchange order, or process-control command is started.",
            "- Local JSON and Markdown artifacts remain the source of truth.",
            "",
        ]
    )
    return "\n".join(lines)


def _diagnostics_id(generated_at: str) -> str:
    parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_inside(path: Path, root: Path) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must resolve inside the workspace: {path}") from exc
    return resolved


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _safe_path_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "unknown"
