#!/usr/bin/env python3
"""Generate second-pass strategies from autonomous smoke failures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "iterative_research_strategies.py"
ITERATION_REGISTRY = AGENT_ROOT / "experiments/iterative_strategy_registry.json"
ITERATION_EXPERIMENT = AGENT_ROOT / "experiments/iterative_strategy_experiment.json"
ITERATION_LEDGER = AGENT_ROOT / "experiments/iterative_hypothesis_ledger.md"
REPORT_INDEX = AGENT_ROOT / "reports/agent_report_index.json"


BASE_TO_ITERATED: dict[str, dict[str, Any]] = {
    "AutoTrendPullbackContinuationStrategy": {
        "class_name": "IterAutoTrendPullbackContinuationV2Strategy",
        "family": "iterated-trend-pullback-continuation",
        "hypothesis": "The first trend-pullback version traded too little and had weak PF; V2 lowers leverage, widens RSI, and allows smaller resume moves while keeping trend and pullback confirmation.",
        "risk_notes": "Still a trend-following continuation test; must beat the original smoke and survive stress-cost checks.",
        "source": [
            'class IterAutoTrendPullbackContinuationV2Strategy(BtcEthFuturesRegime10xOneMinuteStrategy):',
            '    """Failure-driven V2 of AutoTrendPullbackContinuationStrategy."""',
            '    minimal_roi = {"180": 0.0, "45": 0.004, "0": 0.010}',
            "    stoploss = -0.009",
            "    startup_candle_count = 2400",
            "    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:",
            "        return min(2.0, max_leverage)",
            "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_indicators(dataframe, metadata)",
            '        dataframe["iter_ema_fast"] = ta.EMA(dataframe, timeperiod=120)',
            '        dataframe["iter_ema_mid"] = ta.EMA(dataframe, timeperiod=480)',
            '        dataframe["iter_ema_slow"] = ta.EMA(dataframe, timeperiod=2880)',
            '        dataframe["iter_rsi"] = ta.RSI(dataframe, timeperiod=45)',
            '        dataframe["iter_ret_10m"] = dataframe["close"] / dataframe["close"].shift(10) - 1.0',
            '        dataframe["iter_pullback_long"] = dataframe["low"].rolling(300).min() <= dataframe["iter_ema_mid"] * 1.0025',
            '        dataframe["iter_pullback_short"] = dataframe["high"].rolling(300).max() >= dataframe["iter_ema_mid"] * 0.9975',
            "        return dataframe",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_up_regime"] & (dataframe["close"] > dataframe["iter_ema_slow"]) & dataframe["iter_pullback_long"] & (dataframe["close"] > dataframe["iter_ema_fast"]) & (dataframe["iter_ret_10m"] > 0.00025) & dataframe["iter_rsi"].between(43, 69) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "iter_trend_pullback_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["iter_ema_slow"]) & dataframe["iter_pullback_short"] & (dataframe["close"] < dataframe["iter_ema_fast"]) & (dataframe["iter_ret_10m"] < -0.00025) & dataframe["iter_rsi"].between(31, 57) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "iter_trend_pullback_short")',
            "        return dataframe",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[((dataframe["close"] < dataframe["iter_ema_mid"]) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "iter_trend_pullback_long_exit")',
            '        dataframe.loc[((dataframe["close"] > dataframe["iter_ema_mid"]) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "iter_trend_pullback_short_exit")',
            "        return dataframe",
        ],
    },
    "AutoRangeMeanReversionStrategy": {
        "class_name": "IterAutoRangeMeanReversionV2Strategy",
        "family": "iterated-range-mean-reversion",
        "hypothesis": "The first range strategy had too few losing trades; V2 narrows leverage and requires less extreme bands but exits faster at the mean.",
        "risk_notes": "Mean-reversion remains highly fee-sensitive and must be rejected unless stress-cost results improve.",
        "source": [
            'class IterAutoRangeMeanReversionV2Strategy(BtcEthFuturesRegime10xOneMinuteStrategy):',
            '    """Failure-driven V2 of AutoRangeMeanReversionStrategy."""',
            '    minimal_roi = {"90": 0.0, "20": 0.0025, "0": 0.005}',
            "    stoploss = -0.0055",
            "    startup_candle_count = 2400",
            "    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:",
            "        return min(1.0, max_leverage)",
            "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_indicators(dataframe, metadata)",
            '        bb = ta.BBANDS(dataframe, timeperiod=360)',
            '        dataframe["iter_bb_upper"] = bb["upperband"]',
            '        dataframe["iter_bb_middle"] = bb["middleband"]',
            '        dataframe["iter_bb_lower"] = bb["lowerband"]',
            '        dataframe["iter_bb_width"] = (dataframe["iter_bb_upper"] - dataframe["iter_bb_lower"]) / dataframe["iter_bb_middle"]',
            '        dataframe["iter_rsi"] = ta.RSI(dataframe, timeperiod=60)',
            '        dataframe["iter_vol_floor"] = dataframe["volume"].rolling(360).mean() * 0.45',
            "        return dataframe",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["range_regime"] & ~dataframe["high_vol_regime"] & dataframe["iter_bb_width"].between(0.0015, 0.025) & (dataframe["close"] < dataframe["iter_bb_lower"]) & (dataframe["iter_rsi"] < 38) & (dataframe["volume"] > dataframe["iter_vol_floor"])), ["enter_long", "enter_tag"]] = (1, "iter_range_mr_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["range_regime"] & ~dataframe["high_vol_regime"] & dataframe["iter_bb_width"].between(0.0015, 0.025) & (dataframe["close"] > dataframe["iter_bb_upper"]) & (dataframe["iter_rsi"] > 62) & (dataframe["volume"] > dataframe["iter_vol_floor"])), ["enter_short", "enter_tag"]] = (1, "iter_range_mr_short")',
            "        return dataframe",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[((dataframe["close"] >= dataframe["iter_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "iter_range_mr_long_exit")',
            '        dataframe.loc[((dataframe["close"] <= dataframe["iter_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "iter_range_mr_short_exit")',
            "        return dataframe",
        ],
    },
    "AutoVolatilitySqueezeBreakoutStrategy": {
        "class_name": "IterAutoVolatilitySqueezeBreakoutV2Strategy",
        "family": "iterated-volatility-squeeze-breakout",
        "hypothesis": "The first breakout strategy overtraded with very poor PF; V2 is a lower-leverage stricter breakout requiring ADX, volume expansion, and trend alignment.",
        "risk_notes": "This is a salvage attempt for the failed breakout family; high trade count plus poor PF should trigger fast rejection.",
        "source": [
            'class IterAutoVolatilitySqueezeBreakoutV2Strategy(BtcEthFuturesRegime10xOneMinuteStrategy):',
            '    """Failure-driven V2 of AutoVolatilitySqueezeBreakoutStrategy."""',
            '    minimal_roi = {"240": 0.0, "90": 0.006, "0": 0.014}',
            "    stoploss = -0.008",
            "    startup_candle_count = 2400",
            "    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:",
            "        return min(1.0, max_leverage)",
            "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_indicators(dataframe, metadata)",
            '        dataframe["iter_high_12h"] = dataframe["high"].rolling(720).max().shift(1)',
            '        dataframe["iter_low_12h"] = dataframe["low"].rolling(720).min().shift(1)',
            '        dataframe["iter_vol_mean"] = dataframe["volume"].rolling(1440).mean()',
            '        dataframe["iter_bb_width_ma"] = dataframe["bb_width"].rolling(1440).mean()',
            '        dataframe["iter_squeeze_recent"] = (dataframe["bb_width"] < dataframe["iter_bb_width_ma"] * 0.60).rolling(180).max().fillna(0).astype(bool)',
            "        return dataframe",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["iter_squeeze_recent"] & dataframe["trend_up_regime"] & (dataframe["adx"] >= 28) & (dataframe["close"] > dataframe["iter_high_12h"]) & (dataframe["volume"] > dataframe["iter_vol_mean"] * 1.35)), ["enter_long", "enter_tag"]] = (1, "iter_squeeze_breakout_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["iter_squeeze_recent"] & dataframe["trend_down_regime"] & (dataframe["adx"] >= 28) & (dataframe["close"] < dataframe["iter_low_12h"]) & (dataframe["volume"] > dataframe["iter_vol_mean"] * 1.35)), ["enter_short", "enter_tag"]] = (1, "iter_squeeze_breakout_short")',
            "        return dataframe",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[((dataframe["close"] < dataframe["iter_high_12h"]) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "iter_squeeze_long_exit")',
            '        dataframe.loc[((dataframe["close"] > dataframe["iter_low_12h"]) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "iter_squeeze_short_exit")',
            "        return dataframe",
        ],
    },
    "AutoShortOnlyFailedBounceStrategy": {
        "class_name": "IterAutoShortOnlyFailedBounceV2Strategy",
        "family": "iterated-short-only-failed-bounce",
        "hypothesis": "The first short-only failed-bounce idea traded almost never; V2 widens the rejection zone while preserving bearish regime alignment.",
        "risk_notes": "Short-only logic must be checked against bull windows before any promotion.",
        "source": [
            'class IterAutoShortOnlyFailedBounceV2Strategy(BtcEthFuturesRegime10xOneMinuteStrategy):',
            '    """Failure-driven V2 of AutoShortOnlyFailedBounceStrategy."""',
            '    minimal_roi = {"240": 0.0, "60": 0.005, "0": 0.012}',
            "    stoploss = -0.008",
            "    startup_candle_count = 2400",
            "    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:",
            "        return min(2.0, max_leverage)",
            "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_indicators(dataframe, metadata)",
            '        dataframe["iter_ema_fast"] = ta.EMA(dataframe, timeperiod=180)',
            '        dataframe["iter_ema_mid"] = ta.EMA(dataframe, timeperiod=960)',
            '        dataframe["iter_rsi"] = ta.RSI(dataframe, timeperiod=45)',
            '        dataframe["iter_reject"] = (dataframe["high"] >= dataframe["iter_ema_fast"] * 0.996) & (dataframe["close"] < dataframe["iter_ema_fast"] * 1.001)',
            "        return dataframe",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["iter_ema_mid"]) & dataframe["iter_reject"] & dataframe["iter_rsi"].between(34, 62) & (dataframe["minus_di"] > dataframe["plus_di"] * 1.03) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "iter_failed_bounce_short")',
            "        return dataframe",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[((dataframe["close"] > dataframe["iter_ema_fast"] * 1.002) | dataframe["range_regime"] | (dataframe["plus_di"] > dataframe["minus_di"])) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "iter_failed_bounce_short_exit")',
            "        return dataframe",
        ],
    },
    "AutoMicroMomentumConfirmationStrategy": {
        "class_name": "IterAutoMicroMomentumConfirmationV2Strategy",
        "family": "iterated-micro-momentum-confirmation",
        "hypothesis": "The first micro-momentum version was too strict and still low PF; V2 lowers leverage and uses 5/15/45 minute confirmation with ADX direction.",
        "risk_notes": "Short-horizon confirmation can overfit; promote only after walk-forward checks.",
        "source": [
            'class IterAutoMicroMomentumConfirmationV2Strategy(BtcEthFuturesRegime10xOneMinuteStrategy):',
            '    """Failure-driven V2 of AutoMicroMomentumConfirmationStrategy."""',
            '    minimal_roi = {"120": 0.0, "30": 0.0035, "0": 0.008}',
            "    stoploss = -0.0065",
            "    startup_candle_count = 2400",
            "    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:",
            "        return min(1.0, max_leverage)",
            "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_indicators(dataframe, metadata)",
            '        dataframe["iter_ret_5m"] = dataframe["close"] / dataframe["close"].shift(5) - 1.0',
            '        dataframe["iter_ret_15m"] = dataframe["close"] / dataframe["close"].shift(15) - 1.0',
            '        dataframe["iter_ret_45m"] = dataframe["close"] / dataframe["close"].shift(45) - 1.0',
            '        dataframe["iter_rsi"] = ta.RSI(dataframe, timeperiod=45)',
            '        dataframe["iter_micro_up"] = (dataframe["iter_ret_5m"] > 0) & (dataframe["iter_ret_15m"] > 0.0002) & (dataframe["iter_ret_45m"] > 0.0007)',
            '        dataframe["iter_micro_down"] = (dataframe["iter_ret_5m"] < 0) & (dataframe["iter_ret_15m"] < -0.0002) & (dataframe["iter_ret_45m"] < -0.0007)',
            "        return dataframe",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_up_regime"] & dataframe["iter_micro_up"] & (dataframe["plus_di"] > dataframe["minus_di"]) & dataframe["iter_rsi"].between(46, 70) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "iter_micro_confirm_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & dataframe["iter_micro_down"] & (dataframe["minus_di"] > dataframe["plus_di"]) & dataframe["iter_rsi"].between(30, 56) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "iter_micro_confirm_short")',
            "        return dataframe",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[((dataframe["iter_ret_15m"] < -0.00025) | dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "iter_micro_long_exit")',
            '        dataframe.loc[((dataframe["iter_ret_15m"] > 0.00025) | dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "iter_micro_short_exit")',
            "        return dataframe",
        ],
    },
    "AutoDefensiveFlatFilterStrategy": {
        "class_name": "IterAutoDefensiveFlatFilterV2Strategy",
        "family": "iterated-defensive-flat-filter",
        "hypothesis": "The defensive baseline had strong PF but too few trades; V2 modestly widens calm-market filters while keeping 1x leverage.",
        "risk_notes": "This is a capital-preservation baseline; judge it on stable PF, low drawdown, and enough trades.",
        "source": [
            'class IterAutoDefensiveFlatFilterV2Strategy(BtcEthFuturesRegime10xOneMinuteStrategy):',
            '    """Failure-driven V2 of AutoDefensiveFlatFilterStrategy."""',
            '    minimal_roi = {"240": 0.0, "75": 0.0035, "0": 0.007}',
            "    stoploss = -0.0055",
            "    startup_candle_count = 2400",
            "    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:",
            "        return min(1.0, max_leverage)",
            "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_indicators(dataframe, metadata)",
            '        dataframe["iter_ema_mid"] = ta.EMA(dataframe, timeperiod=960)',
            '        dataframe["iter_rsi"] = ta.RSI(dataframe, timeperiod=90)',
            '        dataframe["iter_volume_ok"] = dataframe["volume"] > dataframe["volume"].rolling(960).mean() * 0.4',
            '        dataframe["iter_calm"] = (dataframe["atr_pct"] < 0.008) & (dataframe["realized_vol_24h"] < 0.055)',
            "        return dataframe",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["iter_calm"] & dataframe["trend_up_regime"] & (dataframe["close"] > dataframe["iter_ema_mid"]) & dataframe["iter_rsi"].between(46, 66) & dataframe["iter_volume_ok"]), ["enter_long", "enter_tag"]] = (1, "iter_defensive_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["iter_calm"] & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["iter_ema_mid"]) & dataframe["iter_rsi"].between(34, 56) & dataframe["iter_volume_ok"]), ["enter_short", "enter_tag"]] = (1, "iter_defensive_short")',
            "        return dataframe",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            '        dataframe.loc[((~dataframe["iter_calm"]) | dataframe["range_regime"] | (dataframe["close"] < dataframe["iter_ema_mid"])) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "iter_defensive_long_exit")',
            '        dataframe.loc[((~dataframe["iter_calm"]) | dataframe["range_regime"] | (dataframe["close"] > dataframe["iter_ema_mid"])) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "iter_defensive_short_exit")',
            "        return dataframe",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Use a specific agent report JSON.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timerange", default="20240101-20260622")
    parser.add_argument("--smoke-timerange", default="20260101-20260201")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_report_path() -> Path:
    index = load_json(REPORT_INDEX)
    for item in [index.get("latest_report"), *index.get("reports", [])]:
        if not item:
            continue
        path = REPO_ROOT / item["path"]
        if not path.exists():
            continue
        report = load_json(path)
        if selected_failures(report):
            return path
    raise SystemExit("No autonomous failure report found. Run autonomous smoke first.")


def selected_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for result in report.get("results", []):
        strategy = result.get("strategy")
        if strategy in BASE_TO_ITERATED and result.get("classification") in {"rejected", "needs_review"}:
            selected.append(result)
    return selected


def source_header(generated_at: str) -> list[str]:
    return [
        '"""Failure-driven iterated research strategies.',
        "",
        "Do not edit by hand. Re-generate with user_data/strategy_research/strategy_iteration_engine.py.",
        "These classes are research-only and must not be promoted to dry-run or live without manual approval.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from datetime import datetime",
        "from pathlib import Path",
        "",
        "from pandas import DataFrame",
        "import talib.abstract as ta",
        "",
        "sys.path.append(str(Path(__file__).resolve().parents[1]))",
        "",
        "from btc_eth_risk_controlled_strategies import BtcEthFuturesRegime10xOneMinuteStrategy",
        "",
        "",
        f"GENERATED_AT_UTC = {generated_at!r}",
        "",
    ]


def build_source(items: list[dict[str, Any]]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = source_header(generated_at)
    for item in items:
        template = BASE_TO_ITERATED[item["strategy"]]
        lines.extend(template["source"])
        lines.append("")
        lines.append("")
    return "\n".join(lines)


def build_registry(items: list[dict[str, Any]], report_path: Path) -> dict[str, Any]:
    entries = []
    for item in items:
        template = BASE_TO_ITERATED[item["strategy"]]
        entries.append(
            {
                "name": template["class_name"],
                "family": template["family"],
                "source": "strategy_iteration_engine",
                "iterated_from": item["strategy"],
                "previous_classification": item.get("classification"),
                "previous_trades": item.get("trades"),
                "previous_total_profit_pct": item.get("total_profit_pct"),
                "previous_profit_factor": item.get("profit_factor"),
                "previous_reasons": item.get("reasons", []),
                "hypothesis": template["hypothesis"],
                "risk_notes": template["risk_notes"],
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "input_report": str(report_path.relative_to(REPO_ROOT)),
        "generated_strategy_file": str(GENERATED_FILE.relative_to(REPO_ROOT)),
        "research_mode": "failure_driven_iteration",
        "strategies": entries,
    }


def build_experiment(registry: dict[str, Any], timerange: str, smoke_timerange: str) -> dict[str, Any]:
    return {
        "id": "iterative_strategy_lab",
        "title": "Failure-driven autonomous strategy iterations",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": ["1m"],
        "timeranges": [timerange],
        "matrix": {
            "timeranges": [
                {"name": "smoke", "label": "Recent smoke", "timerange": smoke_timerange},
                {"name": "full", "label": "Full sample", "timerange": timerange},
            ]
        },
        "fee": 0.0005,
        "strategies": [item["name"] for item in registry["strategies"]],
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False,
        },
        "notes": [
            "Generated from failed autonomous smoke results.",
            "Each V2 strategy records the original failure reasons.",
            "Promotion requires outperforming the original smoke plus matrix and bias checks.",
        ],
    }


def ledger_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Iterative Strategy Hypothesis Ledger",
        "",
        f"- Generated UTC: `{registry['generated_at_utc']}`",
        f"- Input report: `{registry['input_report']}`",
        f"- Strategy file: `{registry['generated_strategy_file']}`",
        "",
        "| Iterated Strategy | From | Previous Result | Previous Trades | Previous PF | Failure Reasons | New Hypothesis |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for item in registry["strategies"]:
        lines.append(
            "| {name} | {iterated_from} | {previous_classification} | {previous_trades} | {previous_profit_factor} | {reasons} | {hypothesis} |".format(
                reasons=", ".join(item.get("previous_reasons", [])),
                **item,
            )
        )
    lines.extend(
        [
            "",
            "## Iteration Rules",
            "",
            "- Too few trades: widen entry tolerance without removing the market-regime gate.",
            "- Weak profit factor or negative return: lower leverage, tighten confirmation, or exit faster.",
            "- Very high trade count with poor PF: make the hypothesis stricter rather than adding leverage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve() if args.report else latest_report_path()
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    report = load_json(report_path)
    failures = selected_failures(report)
    if not failures:
        raise SystemExit("No supported autonomous failures found for iteration.")

    source = build_source(failures)
    registry = build_registry(failures, report_path)
    experiment = build_experiment(registry, args.timerange, args.smoke_timerange)
    if args.dry_run:
        print(source)
        print(json.dumps(registry, indent=2, ensure_ascii=False))
        print(json.dumps(experiment, indent=2, ensure_ascii=False))
        return

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(source, encoding="utf-8")
    ITERATION_REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    ITERATION_EXPERIMENT.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    ITERATION_LEDGER.write_text(ledger_markdown(registry), encoding="utf-8")
    print(f"Wrote {GENERATED_FILE.relative_to(REPO_ROOT)}")
    print(f"Wrote {ITERATION_REGISTRY.relative_to(REPO_ROOT)}")
    print(f"Wrote {ITERATION_EXPERIMENT.relative_to(REPO_ROOT)}")
    print(f"Wrote {ITERATION_LEDGER.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
