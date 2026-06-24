#!/usr/bin/env python3
"""Generate autonomous strategy hypotheses as isolated Freqtrade strategies.

This module is deliberately deterministic. The agent is allowed to invent and
test research hypotheses, but it must do so through auditable blueprints rather
than opaque code generation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "autonomous_research_strategies.py"
GENERATED_REGISTRY = AGENT_ROOT / "experiments/autonomous_strategy_registry.json"
GENERATED_EXPERIMENT = AGENT_ROOT / "experiments/autonomous_strategy_experiment.json"
HYPOTHESIS_LEDGER = AGENT_ROOT / "experiments/autonomous_hypothesis_ledger.md"
REPORT_INDEX = AGENT_ROOT / "reports/agent_report_index.json"
IMPROVEMENT_QUEUE = AGENT_ROOT / "agent_iterations/improvement_queue.json"
RETIRED_FAMILY_JSON = AGENT_ROOT / "experiments/retired_seed_family_ledger.json"
RETIRED_FAMILY_MD = AGENT_ROOT / "experiments/retired_seed_family_ledger.md"


@dataclass(frozen=True)
class Blueprint:
    class_name: str
    family: str
    regime: str
    direction: str
    leverage: float
    roi: dict[str, float]
    stoploss: float
    hypothesis: str
    risk_notes: str
    indicator_block: list[str]
    entry_block: list[str]
    exit_block: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timerange", default="20240101-20260622")
    parser.add_argument("--smoke-timerange", default="20260101-20260201")
    return parser.parse_args()


def common_imports(generated_at: str) -> list[str]:
    return [
        '"""Autonomous research strategies generated from auditable blueprints.',
        "",
        "Do not edit by hand. Re-generate with user_data/strategy_research/autonomous_strategy_lab.py.",
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


def roi_literal(roi: dict[str, float]) -> str:
    pieces = [f"{key!r}: {value!r}" for key, value in roi.items()]
    return "{" + ", ".join(pieces) + "}"


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_report() -> dict[str, object]:
    index = load_json(REPORT_INDEX)
    latest = index.get("latest_report") if isinstance(index.get("latest_report"), dict) else {}
    if not latest:
        latest = index.get("latest_dashboard_refresh") if isinstance(index.get("latest_dashboard_refresh"), dict) else {}
    path_value = latest.get("path") if isinstance(latest, dict) else None
    if not path_value:
        return {}
    return load_json(REPO_ROOT / str(path_value))


def num(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def needs_seed_followups(report: dict[str, object]) -> bool:
    results = report.get("results")
    experiment = report.get("experiment")
    if not isinstance(results, list) or not isinstance(experiment, dict):
        return False
    strategy_groups = experiment.get("strategy_groups")
    if not isinstance(strategy_groups, dict):
        return False
    seed_names = set(strategy_groups.get("autonomous_seed") or [])
    seed_results = [item for item in results if isinstance(item, dict) and item.get("strategy") in seed_names]
    if not seed_results:
        return False
    rejected = sum(1 for item in seed_results if item.get("classification") == "rejected")
    too_few = sum(1 for item in seed_results if "too few trades" in ";".join(item.get("reasons", [])))
    high_trade_negative = any(
        num(item.get("trades")) >= 200 and (num(item.get("total_profit_pct")) < 0 or num(item.get("profit_factor")) < 1)
        for item in seed_results
    )
    return rejected == len(seed_results) or too_few >= max(3, len(seed_results) // 2) or high_trade_negative


def needs_anti_edge_followups(report: dict[str, object]) -> bool:
    results = report.get("results")
    if not isinstance(results, list):
        return False
    registry = load_json(GENERATED_REGISTRY)
    followup_names = {
        item.get("name")
        for item in registry.get("strategies", [])
        if isinstance(item, dict) and item.get("source_type") == "seed_followup"
    }
    if not followup_names:
        return False
    followup_results = [item for item in results if isinstance(item, dict) and item.get("strategy") in followup_names]
    if not followup_results:
        return False
    negative = sum(
        1
        for item in followup_results
        if num(item.get("total_profit_pct")) < 0 or num(item.get("profit_factor")) < 1
    )
    enough_sample_negative = any(
        num(item.get("trades")) >= 200 and (num(item.get("total_profit_pct")) < 0 or num(item.get("profit_factor")) < 1)
        for item in followup_results
    )
    return negative >= max(2, len(followup_results) // 2) or enough_sample_negative


def anti_edge_family_failed() -> bool:
    queue = load_json(IMPROVEMENT_QUEUE)
    for item in queue.get("items", []):
        if isinstance(item, dict) and item.get("issue_id") == "anti_edge_family_failed" and item.get("status") == "open":
            return True
    return False


def class_block(blueprint: Blueprint) -> list[str]:
    tag = blueprint.family.lower().replace("-", "_")[:36]
    lines = [
        "",
        f"class {blueprint.class_name}(BtcEthFuturesRegime10xOneMinuteStrategy):",
        f"    \"\"\"Autonomous research hypothesis: {blueprint.family}.\"\"\"",
        "",
        f"    minimal_roi = {roi_literal(blueprint.roi)}",
        f"    stoploss = {blueprint.stoploss!r}",
        "    startup_candle_count = 2400",
        "",
        "    def leverage(",
        "        self,",
        "        pair: str,",
        "        current_time: datetime,",
        "        current_rate: float,",
        "        proposed_leverage: float,",
        "        max_leverage: float,",
        "        entry_tag: str | None,",
        "        side: str,",
        "        **kwargs,",
        "    ) -> float:",
        f"        return min({blueprint.leverage!r}, max_leverage)",
        "",
        "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe = super().populate_indicators(dataframe, metadata)",
    ]
    lines.extend(f"        {line}" for line in blueprint.indicator_block)
    lines.extend(
        [
            "        return dataframe",
            "",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        ]
    )
    lines.extend(f"        {line}" for line in blueprint.entry_block)
    lines.extend(
        [
            "        return dataframe",
            "",
            "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        ]
    )
    lines.extend(f"        {line}" for line in blueprint.exit_block)
    lines.extend(
        [
            "        return dataframe",
            "",
            f"    # Research tag: {tag}",
        ]
    )
    return lines


def blueprints() -> list[Blueprint]:
    return [
        Blueprint(
            class_name="AutoTrendPullbackContinuationStrategy",
            family="trend-pullback-continuation",
            regime="trend",
            direction="long_short",
            leverage=3.0,
            roi={"180": 0.0, "45": 0.006, "0": 0.014},
            stoploss=-0.012,
            hypothesis="Trend entries should wait for a pullback to a medium EMA and a resume candle, not fire immediately on slow momentum.",
            risk_notes="Designed to test entry timing improvement versus the older always-enter-on-signal logic.",
            indicator_block=[
                'dataframe["auto_ema_fast"] = ta.EMA(dataframe, timeperiod=180)',
                'dataframe["auto_ema_mid"] = ta.EMA(dataframe, timeperiod=720)',
                'dataframe["auto_ema_slow"] = ta.EMA(dataframe, timeperiod=4320)',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=60)',
                'dataframe["auto_ret_15m"] = dataframe["close"] / dataframe["close"].shift(15) - 1.0',
                'dataframe["auto_pullback_long"] = dataframe["low"].rolling(180).min() <= dataframe["auto_ema_mid"] * 1.001',
                'dataframe["auto_pullback_short"] = dataframe["high"].rolling(180).max() >= dataframe["auto_ema_mid"] * 0.999',
            ],
            entry_block=[
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["trend_up_regime"]',
                '& (dataframe["close"] > dataframe["auto_ema_slow"])',
                '& dataframe["auto_pullback_long"]',
                '& (dataframe["close"] > dataframe["auto_ema_fast"])',
                '& (dataframe["auto_ret_15m"] > 0.0006)',
                '& dataframe["auto_rsi"].between(46, 66)',
                '& (dataframe["volume"] > 0)',
                "),",
                '["enter_long", "enter_tag"],',
                '] = (1, "auto_trend_pullback_long")',
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["trend_down_regime"]',
                '& (dataframe["close"] < dataframe["auto_ema_slow"])',
                '& dataframe["auto_pullback_short"]',
                '& (dataframe["close"] < dataframe["auto_ema_fast"])',
                '& (dataframe["auto_ret_15m"] < -0.0006)',
                '& dataframe["auto_rsi"].between(34, 54)',
                '& (dataframe["volume"] > 0)',
                "),",
                '["enter_short", "enter_tag"],',
                '] = (1, "auto_trend_pullback_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] < dataframe["auto_ema_mid"]) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_trend_pullback_long_exit")',
                'dataframe.loc[((dataframe["close"] > dataframe["auto_ema_mid"]) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_trend_pullback_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoRangeMeanReversionStrategy",
            family="range-mean-reversion",
            regime="range",
            direction="long_short",
            leverage=2.0,
            roi={"120": 0.0, "30": 0.003, "0": 0.007},
            stoploss=-0.007,
            hypothesis="Range regimes may support small mean-reversion trades if high volatility and trend regimes are explicitly blocked.",
            risk_notes="Expected to be fee-sensitive; must pass stress-cost matrix before promotion.",
            indicator_block=[
                'bb = ta.BBANDS(dataframe, timeperiod=720)',
                'dataframe["auto_bb_upper"] = bb["upperband"]',
                'dataframe["auto_bb_middle"] = bb["middleband"]',
                'dataframe["auto_bb_lower"] = bb["lowerband"]',
                'dataframe["auto_bb_width"] = (dataframe["auto_bb_upper"] - dataframe["auto_bb_lower"]) / dataframe["auto_bb_middle"]',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=120)',
                'dataframe["auto_volume_floor"] = dataframe["volume"].rolling(720).mean() * 0.55',
            ],
            entry_block=[
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["range_regime"]',
                '& ~dataframe["high_vol_regime"]',
                '& dataframe["auto_bb_width"].between(0.002, 0.030)',
                '& (dataframe["close"] < dataframe["auto_bb_lower"])',
                '& (dataframe["auto_rsi"] < 34)',
                '& (dataframe["volume"] > dataframe["auto_volume_floor"])',
                "),",
                '["enter_long", "enter_tag"],',
                '] = (1, "auto_range_mr_long")',
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["range_regime"]',
                '& ~dataframe["high_vol_regime"]',
                '& dataframe["auto_bb_width"].between(0.002, 0.030)',
                '& (dataframe["close"] > dataframe["auto_bb_upper"])',
                '& (dataframe["auto_rsi"] > 66)',
                '& (dataframe["volume"] > dataframe["auto_volume_floor"])',
                "),",
                '["enter_short", "enter_tag"],',
                '] = (1, "auto_range_mr_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] >= dataframe["auto_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_range_mr_long_exit")',
                'dataframe.loc[((dataframe["close"] <= dataframe["auto_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_range_mr_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoVolatilitySqueezeBreakoutStrategy",
            family="volatility-squeeze-breakout",
            regime="compression_expansion",
            direction="long_short",
            leverage=3.0,
            roi={"180": 0.0, "60": 0.008, "0": 0.018},
            stoploss=-0.010,
            hypothesis="Compression followed by volume-confirmed expansion may improve breakout timing versus slow trend filters.",
            risk_notes="Breakout systems can whipsaw; requires regime and cost robustness.",
            indicator_block=[
                'dataframe["auto_high_6h"] = dataframe["high"].rolling(360).max().shift(1)',
                'dataframe["auto_low_6h"] = dataframe["low"].rolling(360).min().shift(1)',
                'dataframe["auto_vol_mean"] = dataframe["volume"].rolling(720).mean()',
                'dataframe["auto_bb_width_ma"] = dataframe["bb_width"].rolling(720).mean()',
                'dataframe["auto_squeeze"] = dataframe["bb_width"] < dataframe["auto_bb_width_ma"] * 0.72',
                'dataframe["auto_squeeze_recent"] = dataframe["auto_squeeze"].rolling(120).max().fillna(0).astype(bool)',
            ],
            entry_block=[
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["auto_squeeze_recent"]',
                '& ~dataframe["high_vol_regime"]',
                '& (dataframe["close"] > dataframe["auto_high_6h"])',
                '& (dataframe["plus_di"] > dataframe["minus_di"])',
                '& (dataframe["volume"] > dataframe["auto_vol_mean"] * 1.15)',
                "),",
                '["enter_long", "enter_tag"],',
                '] = (1, "auto_squeeze_breakout_long")',
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["auto_squeeze_recent"]',
                '& ~dataframe["high_vol_regime"]',
                '& (dataframe["close"] < dataframe["auto_low_6h"])',
                '& (dataframe["minus_di"] > dataframe["plus_di"])',
                '& (dataframe["volume"] > dataframe["auto_vol_mean"] * 1.15)',
                "),",
                '["enter_short", "enter_tag"],',
                '] = (1, "auto_squeeze_breakout_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] < dataframe["auto_high_6h"]) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_squeeze_long_exit")',
                'dataframe.loc[((dataframe["close"] > dataframe["auto_low_6h"]) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_squeeze_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoShortOnlyFailedBounceStrategy",
            family="short-only-failed-bounce",
            regime="bear_trend",
            direction="short_only",
            leverage=3.0,
            roi={"240": 0.0, "60": 0.007, "0": 0.016},
            stoploss=-0.010,
            hypothesis="Prior tests suggest shorts were cleaner; failed bounce entries may capture bearish continuation without immediate chase entries.",
            risk_notes="Short-only edge can decay in bull markets; must be segmented by market regime.",
            indicator_block=[
                'dataframe["auto_ema_fast"] = ta.EMA(dataframe, timeperiod=240)',
                'dataframe["auto_ema_mid"] = ta.EMA(dataframe, timeperiod=1440)',
                'dataframe["auto_bounce_high"] = dataframe["high"].rolling(240).max()',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=60)',
                'dataframe["auto_reject"] = (dataframe["high"] >= dataframe["auto_ema_fast"] * 0.999) & (dataframe["close"] < dataframe["auto_ema_fast"])',
            ],
            entry_block=[
                "dataframe.loc[",
                "(",
                'dataframe["risk_allowed"]',
                '& dataframe["trend_down_regime"]',
                '& (dataframe["close"] < dataframe["auto_ema_mid"])',
                '& dataframe["auto_reject"]',
                '& dataframe["auto_rsi"].between(38, 58)',
                '& (dataframe["minus_di"] > dataframe["plus_di"])',
                '& (dataframe["volume"] > 0)',
                "),",
                '["enter_short", "enter_tag"],',
                '] = (1, "auto_failed_bounce_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] > dataframe["auto_ema_fast"]) | dataframe["range_regime"] | (dataframe["plus_di"] > dataframe["minus_di"])) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_failed_bounce_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoMicroMomentumConfirmationStrategy",
            family="micro-momentum-confirmation",
            regime="trend",
            direction="long_short",
            leverage=2.0,
            roi={"120": 0.0, "30": 0.004, "0": 0.010},
            stoploss=-0.008,
            hypothesis="A slow regime signal should require short-horizon confirmation from several consecutive micro returns.",
            risk_notes="May overfit short noise; requires out-of-sample windows.",
            indicator_block=[
                'dataframe["auto_ret_3m"] = dataframe["close"] / dataframe["close"].shift(3) - 1.0',
                'dataframe["auto_ret_10m"] = dataframe["close"] / dataframe["close"].shift(10) - 1.0',
                'dataframe["auto_ret_30m"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=30)',
                'dataframe["auto_micro_up"] = (dataframe["auto_ret_3m"] > 0) & (dataframe["auto_ret_10m"] > 0.0004) & (dataframe["auto_ret_30m"] > 0.001)',
                'dataframe["auto_micro_down"] = (dataframe["auto_ret_3m"] < 0) & (dataframe["auto_ret_10m"] < -0.0004) & (dataframe["auto_ret_30m"] < -0.001)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_up_regime"] & dataframe["auto_micro_up"] & dataframe["auto_rsi"].between(48, 68) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_micro_confirm_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & dataframe["auto_micro_down"] & dataframe["auto_rsi"].between(32, 52) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_micro_confirm_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["auto_ret_10m"] < -0.0004) | dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_micro_long_exit")',
                'dataframe.loc[((dataframe["auto_ret_10m"] > 0.0004) | dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_micro_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoDefensiveFlatFilterStrategy",
            family="defensive-flat-filter",
            regime="capital_preservation",
            direction="long_short",
            leverage=1.0,
            roi={"240": 0.0, "90": 0.004, "0": 0.009},
            stoploss=-0.006,
            hypothesis="The agent needs a low-leverage defensive baseline that trades only when volatility, trend, and liquidity filters are all clean.",
            risk_notes="This is a benchmark-quality baseline, not an aggressive profit candidate.",
            indicator_block=[
                'dataframe["auto_ema_mid"] = ta.EMA(dataframe, timeperiod=1440)',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=120)',
                'dataframe["auto_volume_ok"] = dataframe["volume"] > dataframe["volume"].rolling(1440).mean() * 0.5',
                'dataframe["auto_calm"] = (dataframe["atr_pct"] < 0.006) & (dataframe["realized_vol_24h"] < 0.045)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["auto_calm"] & dataframe["trend_up_regime"] & (dataframe["close"] > dataframe["auto_ema_mid"]) & dataframe["auto_rsi"].between(48, 62) & dataframe["auto_volume_ok"]), ["enter_long", "enter_tag"]] = (1, "auto_defensive_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["auto_calm"] & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["auto_ema_mid"]) & dataframe["auto_rsi"].between(38, 52) & dataframe["auto_volume_ok"]), ["enter_short", "enter_tag"]] = (1, "auto_defensive_short")',
            ],
            exit_block=[
                'dataframe.loc[((~dataframe["auto_calm"]) | dataframe["range_regime"] | (dataframe["close"] < dataframe["auto_ema_mid"])) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_defensive_long_exit")',
                'dataframe.loc[((~dataframe["auto_calm"]) | dataframe["range_regime"] | (dataframe["close"] > dataframe["auto_ema_mid"])) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_defensive_short_exit")',
            ],
        ),
    ]


def seed_followup_blueprints(report: dict[str, object]) -> list[Blueprint]:
    if not needs_seed_followups(report):
        return []
    return [
        Blueprint(
            class_name="AutoSampleFloorTrendProbeStrategy",
            family="sample-floor-trend-probe",
            regime="trend",
            direction="long_short",
            leverage=2.0,
            roi={"120": 0.0, "30": 0.003, "0": 0.008},
            stoploss=-0.008,
            hypothesis="When trend seed variants are too sparse, loosen to a two-condition trend probe: slow regime plus 24h direction.",
            risk_notes="Research-only sample-floor probe; reject if higher sample size still has negative expectancy.",
            indicator_block=[
                'dataframe["auto_ret_10m"] = dataframe["close"] / dataframe["close"].shift(10) - 1.0',
                'dataframe["auto_volume_ok"] = dataframe["volume"] > 0',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_up_regime"] & (dataframe["ret_24h"] > 0.001) & dataframe["auto_volume_ok"]), ["enter_long", "enter_tag"]] = (1, "auto_sample_trend_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & (dataframe["ret_24h"] < -0.001) & dataframe["auto_volume_ok"]), ["enter_short", "enter_tag"]] = (1, "auto_sample_trend_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["auto_ret_10m"] < -0.0005) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_sample_trend_long_exit")',
                'dataframe.loc[((dataframe["auto_ret_10m"] > 0.0005) | dataframe["range_regime"] | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_sample_trend_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoSampleFloorRangeProbeStrategy",
            family="sample-floor-range-probe",
            regime="range",
            direction="long_short",
            leverage=1.5,
            roi={"90": 0.0, "20": 0.0025, "0": 0.006},
            stoploss=-0.006,
            hypothesis="Range mean-reversion was too sparse; test a looser Bollinger mean-reversion probe with broad RSI bands.",
            risk_notes="Cost-sensitive probe; promotion requires fee stress because targets are small.",
            indicator_block=[
                'bb = ta.BBANDS(dataframe, timeperiod=480)',
                'dataframe["auto_bb_upper"] = bb["upperband"]',
                'dataframe["auto_bb_middle"] = bb["middleband"]',
                'dataframe["auto_bb_lower"] = bb["lowerband"]',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=60)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["range_regime"] & (dataframe["close"] < dataframe["auto_bb_lower"]) & (dataframe["auto_rsi"] < 42) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_sample_range_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["range_regime"] & (dataframe["close"] > dataframe["auto_bb_upper"]) & (dataframe["auto_rsi"] > 58) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_sample_range_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] >= dataframe["auto_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_sample_range_long_exit")',
                'dataframe.loc[((dataframe["close"] <= dataframe["auto_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_sample_range_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoSampleFloorMicroProbeStrategy",
            family="sample-floor-micro-probe",
            regime="micro_momentum",
            direction="long_short",
            leverage=2.0,
            roi={"90": 0.0, "20": 0.003, "0": 0.007},
            stoploss=-0.007,
            hypothesis="Micro momentum should be evaluated with enough trades before adding slow regime filters back.",
            risk_notes="Designed to measure signal edge; can be noisy and must pass cost stress.",
            indicator_block=[
                'dataframe["auto_ret_5m"] = dataframe["close"] / dataframe["close"].shift(5) - 1.0',
                'dataframe["auto_ret_15m"] = dataframe["close"] / dataframe["close"].shift(15) - 1.0',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=30)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["auto_ret_5m"] > 0.0003) & (dataframe["auto_ret_15m"] > 0.0008) & dataframe["auto_rsi"].between(44, 72) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_sample_micro_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["auto_ret_5m"] < -0.0003) & (dataframe["auto_ret_15m"] < -0.0008) & dataframe["auto_rsi"].between(28, 56) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_sample_micro_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["auto_ret_5m"] < -0.0002) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_sample_micro_long_exit")',
                'dataframe.loc[((dataframe["auto_ret_5m"] > 0.0002) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_sample_micro_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoInverseSqueezeFadeStrategy",
            family="inverse-squeeze-fade",
            regime="compression_expansion",
            direction="long_short",
            leverage=1.5,
            roi={"120": 0.0, "30": 0.003, "0": 0.008},
            stoploss=-0.008,
            hypothesis="The high-trade squeeze breakout seed was strongly negative; test whether post-breakout fade has the opposite edge.",
            risk_notes="This is an explicit inverse test, not a promotion candidate without out-of-sample confirmation.",
            indicator_block=[
                'dataframe["auto_high_6h"] = dataframe["high"].rolling(360).max().shift(1)',
                'dataframe["auto_low_6h"] = dataframe["low"].rolling(360).min().shift(1)',
                'dataframe["auto_mid_2h"] = ta.EMA(dataframe, timeperiod=120)',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=45)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["close"] > dataframe["auto_high_6h"]) & (dataframe["auto_rsi"] > 62) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_inverse_squeeze_short")',
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["close"] < dataframe["auto_low_6h"]) & (dataframe["auto_rsi"] < 38) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_inverse_squeeze_long")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] >= dataframe["auto_mid_2h"]) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_inverse_squeeze_long_exit")',
                'dataframe.loc[((dataframe["close"] <= dataframe["auto_mid_2h"]) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_inverse_squeeze_short_exit")',
            ],
        ),
    ]


def anti_edge_followup_blueprints(report: dict[str, object]) -> list[Blueprint]:
    if not needs_anti_edge_followups(report):
        return []
    return [
        Blueprint(
            class_name="AutoAntiEdgeMicroFadeStrategy",
            family="anti-edge-micro-fade",
            regime="micro_momentum",
            direction="long_short",
            leverage=1.5,
            roi={"60": 0.0, "15": 0.0025, "0": 0.006},
            stoploss=-0.006,
            hypothesis="If sample-floor micro momentum is negative, fading short-horizon extension may capture the opposite edge.",
            risk_notes="Explicit anti-edge test; reject quickly if cost-adjusted PF stays below 1.",
            indicator_block=[
                'dataframe["auto_ret_5m"] = dataframe["close"] / dataframe["close"].shift(5) - 1.0',
                'dataframe["auto_ret_15m"] = dataframe["close"] / dataframe["close"].shift(15) - 1.0',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=30)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["auto_ret_5m"] > 0.0005) & (dataframe["auto_ret_15m"] > 0.0012) & (dataframe["auto_rsi"] > 62) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_anti_micro_short")',
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["auto_ret_5m"] < -0.0005) & (dataframe["auto_ret_15m"] < -0.0012) & (dataframe["auto_rsi"] < 38) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_anti_micro_long")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["auto_ret_5m"] > 0.0) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_anti_micro_long_exit")',
                'dataframe.loc[((dataframe["auto_ret_5m"] < 0.0) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_anti_micro_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoCostAwareFastExitProbeStrategy",
            family="cost-aware-fast-exit-probe",
            regime="micro_momentum",
            direction="long_short",
            leverage=1.0,
            roi={"45": 0.0, "10": 0.002, "0": 0.005},
            stoploss=-0.0045,
            hypothesis="If follow-up probes are negative, shorten holding time and use smaller leverage to separate signal edge from exit drag.",
            risk_notes="Cost-aware diagnostic; useful only if trade count remains high and drawdown falls.",
            indicator_block=[
                'dataframe["auto_ret_3m"] = dataframe["close"] / dataframe["close"].shift(3) - 1.0',
                'dataframe["auto_ret_10m"] = dataframe["close"] / dataframe["close"].shift(10) - 1.0',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=24)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["auto_ret_3m"] > 0.0002) & (dataframe["auto_ret_10m"] > 0.0006) & dataframe["auto_rsi"].between(45, 68) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_cost_fast_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["auto_ret_3m"] < -0.0002) & (dataframe["auto_ret_10m"] < -0.0006) & dataframe["auto_rsi"].between(32, 55) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_cost_fast_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["auto_ret_3m"] < 0.0) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_cost_fast_long_exit")',
                'dataframe.loc[((dataframe["auto_ret_3m"] > 0.0) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_cost_fast_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoRangeBreakoutFadeStrategy",
            family="range-breakout-fade",
            regime="range",
            direction="long_short",
            leverage=1.5,
            roi={"90": 0.0, "20": 0.0025, "0": 0.006},
            stoploss=-0.006,
            hypothesis="If direct range and breakout probes are negative, test fading range-edge breaks back toward the mean.",
            risk_notes="Requires cost stress because it may trade frequently around bands.",
            indicator_block=[
                'bb = ta.BBANDS(dataframe, timeperiod=360)',
                'dataframe["auto_bb_upper"] = bb["upperband"]',
                'dataframe["auto_bb_middle"] = bb["middleband"]',
                'dataframe["auto_bb_lower"] = bb["lowerband"]',
                'dataframe["auto_rsi"] = ta.RSI(dataframe, timeperiod=45)',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["range_regime"] & (dataframe["close"] > dataframe["auto_bb_upper"]) & (dataframe["auto_rsi"] > 60) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_range_fade_short")',
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["range_regime"] & (dataframe["close"] < dataframe["auto_bb_lower"]) & (dataframe["auto_rsi"] < 40) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_range_fade_long")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] >= dataframe["auto_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_range_fade_long_exit")',
                'dataframe.loc[((dataframe["close"] <= dataframe["auto_bb_middle"]) | ~dataframe["range_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_range_fade_short_exit")',
            ],
        ),
    ]


def context_feature_blueprints() -> list[Blueprint]:
    return [
        Blueprint(
            class_name="AutoContextDailyTrendPullbackStrategy",
            family="context-daily-trend-pullback",
            regime="higher_timeframe_context",
            direction="long_short",
            leverage=2.0,
            roi={"240": 0.0, "60": 0.006, "0": 0.014},
            stoploss=-0.010,
            hypothesis="After retiring simple 1m OHLCV seeds, use 24h/72h context plus a local pullback/resume trigger.",
            risk_notes="Context-feature replacement for retired simple OHLCV seeds; requires matrix and walk-forward checks.",
            indicator_block=[
                'dataframe["ctx_ema_4h"] = ta.EMA(dataframe, timeperiod=240)',
                'dataframe["ctx_ema_24h"] = ta.EMA(dataframe, timeperiod=1440)',
                'dataframe["ctx_ret_6h"] = dataframe["close"] / dataframe["close"].shift(360) - 1.0',
                'dataframe["ctx_ret_24h"] = dataframe["close"] / dataframe["close"].shift(1440) - 1.0',
                'dataframe["ctx_pullback_long"] = dataframe["low"].rolling(90).min() <= dataframe["ctx_ema_4h"] * 1.001',
                'dataframe["ctx_pullback_short"] = dataframe["high"].rolling(90).max() >= dataframe["ctx_ema_4h"] * 0.999',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["close"] > dataframe["ctx_ema_24h"]) & (dataframe["ctx_ret_24h"] > 0.004) & dataframe["ctx_pullback_long"] & (dataframe["ctx_ret_6h"] > 0.001) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_context_daily_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & (dataframe["close"] < dataframe["ctx_ema_24h"]) & (dataframe["ctx_ret_24h"] < -0.004) & dataframe["ctx_pullback_short"] & (dataframe["ctx_ret_6h"] < -0.001) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_context_daily_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["close"] < dataframe["ctx_ema_4h"]) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_context_daily_long_exit")',
                'dataframe.loc[((dataframe["close"] > dataframe["ctx_ema_4h"]) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_context_daily_short_exit")',
            ],
        ),
        Blueprint(
            class_name="AutoContextVolatilityExpansionStrategy",
            family="context-volatility-expansion",
            regime="volatility_context",
            direction="long_short",
            leverage=1.5,
            roi={"180": 0.0, "45": 0.005, "0": 0.012},
            stoploss=-0.009,
            hypothesis="Use volatility expansion only when 24h direction and 7d context agree, avoiding isolated 1m breakout noise.",
            risk_notes="Tests whether context filters can salvage expansion logic after simple squeeze breakout failed.",
            indicator_block=[
                'dataframe["ctx_ret_24h"] = dataframe["close"] / dataframe["close"].shift(1440) - 1.0',
                'dataframe["ctx_ret_7d"] = dataframe["close"] / dataframe["close"].shift(10080) - 1.0',
                'dataframe["ctx_bb_width_ma"] = dataframe["bb_width"].rolling(1440).mean()',
                'dataframe["ctx_expanding"] = dataframe["bb_width"] > dataframe["ctx_bb_width_ma"] * 1.15',
                'dataframe["ctx_ret_30m"] = dataframe["close"] / dataframe["close"].shift(30) - 1.0',
            ],
            entry_block=[
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["ctx_expanding"] & (dataframe["ctx_ret_7d"] > 0.01) & (dataframe["ctx_ret_24h"] > 0.002) & (dataframe["ctx_ret_30m"] > 0.0008) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "auto_context_vol_long")',
                'dataframe.loc[(dataframe["risk_allowed"] & dataframe["ctx_expanding"] & (dataframe["ctx_ret_7d"] < -0.01) & (dataframe["ctx_ret_24h"] < -0.002) & (dataframe["ctx_ret_30m"] < -0.0008) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "auto_context_vol_short")',
            ],
            exit_block=[
                'dataframe.loc[((dataframe["ctx_ret_30m"] < -0.0004) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_long", "exit_tag"]] = (1, "auto_context_vol_long_exit")',
                'dataframe.loc[((dataframe["ctx_ret_30m"] > 0.0004) | dataframe["high_vol_regime"]) & (dataframe["volume"] > 0), ["exit_short", "exit_tag"]] = (1, "auto_context_vol_short_exit")',
            ],
        ),
    ]


def build_source(items: list[Blueprint]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = common_imports(generated_at)
    for item in items:
        lines.extend(class_block(item))
    return "\n".join(lines) + "\n"


def registry_payload(items: list[Blueprint]) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    followup_count = sum(1 for item in items if item.family.startswith(("sample-floor", "inverse-")))
    anti_edge_count = sum(1 for item in items if item.family.startswith(("anti-edge", "cost-aware", "range-breakout-fade")))
    return {
        "generated_at_utc": generated_at,
        "generated_strategy_file": str(GENERATED_FILE.relative_to(REPO_ROOT)),
        "research_mode": "autonomous_blueprint_generation",
        "base_seed_count": len(items) - followup_count - anti_edge_count,
        "followup_seed_count": followup_count,
        "anti_edge_seed_count": anti_edge_count,
        "strategies": [
            {
                "name": item.class_name,
                "family": f"autonomous-{item.family}",
                "source": "autonomous_strategy_lab",
                "source_type": (
                    "anti_edge_followup"
                    if item.family.startswith(("anti-edge", "cost-aware", "range-breakout-fade"))
                    else "seed_followup"
                    if item.family.startswith(("sample-floor", "inverse-"))
                    else "base_seed"
                ),
                "regime": item.regime,
                "direction": item.direction,
                "leverage_cap": item.leverage,
                "hypothesis": item.hypothesis,
                "risk_notes": item.risk_notes,
            }
            for item in items
        ],
        "blueprints": [asdict(item) for item in items],
    }


def experiment_payload(items: list[Blueprint], timerange: str, smoke_timerange: str) -> dict[str, object]:
    return {
        "id": "autonomous_strategy_lab",
        "title": "Autonomous strategy lab generated hypotheses",
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
        "strategies": [item.class_name for item in items],
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False,
        },
        "notes": [
            "Generated from deterministic blueprints, not external executable code.",
            "Smoke timerange is used by the full cycle before expensive long-sample promotion.",
            "Dry-run/live promotion requires manual approval plus recursive/lookahead checks.",
        ],
    }


def ledger_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Autonomous Strategy Hypothesis Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Strategy file: `{payload['generated_strategy_file']}`",
        "",
        "| Strategy | Family | Regime | Direction | Leverage | Hypothesis |",
        "|---|---|---|---|---:|---|",
    ]
    for item in payload["strategies"]:  # type: ignore[index]
        lines.append(
            "| {name} | {family} | {regime} | {direction} | {leverage_cap} | {hypothesis} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Every generated strategy is research-only.",
            "- External code is not imported or executed.",
            "- Old failed high-leverage logic is not promoted; new hypotheses focus on entry timing, regime separation, and cost-aware selectivity.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_retired_family_ledger(enabled: bool, items: list[Blueprint]) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "retired": enabled,
        "retired_family": "simple_1m_ohlcv_seed_family" if enabled else None,
        "replacement_families": [item.family for item in items],
        "reason": "base, sample-floor, and anti-edge 1m OHLCV seed variants failed" if enabled else "not retired",
    }
    RETIRED_FAMILY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Retired Seed Family Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Retired: `{payload['retired']}`",
        f"- Retired family: `{payload['retired_family']}`",
        f"- Reason: {payload['reason']}",
        "",
        "## Replacement Families",
        "",
    ]
    for family in payload["replacement_families"]:
        lines.append(f"- `{family}`")
    RETIRED_FAMILY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = latest_report()
    retire_simple_seed = anti_edge_family_failed()
    if retire_simple_seed:
        items = context_feature_blueprints()
    else:
        items = blueprints() + seed_followup_blueprints(report) + anti_edge_followup_blueprints(report)
    source = build_source(items)
    registry = registry_payload(items)
    experiment = experiment_payload(items, args.timerange, args.smoke_timerange)
    if args.dry_run:
        print(source)
        print(json.dumps(registry, indent=2, ensure_ascii=False))
        print(json.dumps(experiment, indent=2, ensure_ascii=False))
        return

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(source, encoding="utf-8")
    GENERATED_REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    GENERATED_EXPERIMENT.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    HYPOTHESIS_LEDGER.write_text(ledger_markdown(registry), encoding="utf-8")
    write_retired_family_ledger(retire_simple_seed, items)
    print(f"Wrote {GENERATED_FILE.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATED_REGISTRY.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATED_EXPERIMENT.relative_to(REPO_ROOT)}")
    print(f"Wrote {HYPOTHESIS_LEDGER.relative_to(REPO_ROOT)}")
    print(f"Wrote {RETIRED_FAMILY_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {RETIRED_FAMILY_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
