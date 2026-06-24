#!/usr/bin/env python3
"""Generate sample-expansion variants for promising but under-sampled strategies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "sample_expansion_strategies.py"
REGISTRY_PATH = AGENT_ROOT / "experiments/sample_expansion_strategy_registry.json"
EXPERIMENT_PATH = AGENT_ROOT / "experiments/sample_expansion_experiment.json"
LEDGER_PATH = AGENT_ROOT / "sample_expansion/latest_sample_expansion_plan.md"


VARIANTS = [
    {
        "name": "SampleAutoDefensiveFlatFilterBalancedStrategy",
        "base": "AutoDefensiveFlatFilterStrategy",
        "family": "sample-expansion-defensive-flat",
        "bucket": "cost_pressure",
        "body": [
            '    minimal_roi = {"180": 0.0, "60": 0.0035, "0": 0.007}',
            "    stoploss = -0.0065",
            "",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_entry_trend(dataframe, metadata)",
            '        relaxed_calm = (dataframe["atr_pct"] < 0.009) & (dataframe["realized_vol_24h"] < 0.070)',
            '        dataframe.loc[(dataframe["risk_allowed"] & relaxed_calm & dataframe["trend_up_regime"] & (dataframe["close"] > dataframe["auto_ema_mid"]) & dataframe["auto_rsi"].between(44, 68) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "sample_defensive_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & relaxed_calm & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["auto_ema_mid"]) & dataframe["auto_rsi"].between(30, 58) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "sample_defensive_short")',
            "        return dataframe",
        ],
    },
    {
        "name": "SampleIterDefensiveFlatFilterBalancedStrategy",
        "base": "IterAutoDefensiveFlatFilterV2Strategy",
        "family": "sample-expansion-defensive-flat",
        "bucket": "cost_pressure",
        "body": [
            '    minimal_roi = {"180": 0.0, "60": 0.003, "0": 0.006}',
            "    stoploss = -0.006",
            "",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_entry_trend(dataframe, metadata)",
            '        relaxed_calm = (dataframe["atr_pct"] < 0.010) & (dataframe["realized_vol_24h"] < 0.075)',
            '        dataframe.loc[(dataframe["risk_allowed"] & relaxed_calm & dataframe["trend_up_regime"] & (dataframe["close"] > dataframe["iter_ema_mid"]) & dataframe["iter_rsi"].between(43, 70) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "sample_iter_defensive_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & relaxed_calm & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["iter_ema_mid"]) & dataframe["iter_rsi"].between(30, 60) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "sample_iter_defensive_short")',
            "        return dataframe",
        ],
    },
    {
        "name": "SampleAutoTrendPullbackBalancedStrategy",
        "base": "AutoTrendPullbackContinuationStrategy",
        "family": "sample-expansion-trend-pullback",
        "bucket": "trend",
        "body": [
            '    minimal_roi = {"180": 0.0, "45": 0.0045, "0": 0.010}',
            "    stoploss = -0.009",
            "",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_entry_trend(dataframe, metadata)",
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_up_regime"] & (dataframe["close"] > dataframe["auto_ema_slow"]) & dataframe["auto_pullback_long"] & (dataframe["auto_ret_15m"] > 0.0) & dataframe["auto_rsi"].between(42, 70) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "sample_trend_pullback_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & (dataframe["close"] < dataframe["auto_ema_slow"]) & dataframe["auto_pullback_short"] & (dataframe["auto_ret_15m"] < 0.0) & dataframe["auto_rsi"].between(30, 58) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "sample_trend_pullback_short")',
            "        return dataframe",
        ],
    },
    {
        "name": "SampleAutoMicroMomentumBalancedStrategy",
        "base": "AutoMicroMomentumConfirmationStrategy",
        "family": "sample-expansion-micro-momentum",
        "bucket": "short_momentum",
        "body": [
            '    minimal_roi = {"120": 0.0, "30": 0.003, "0": 0.007}',
            "    stoploss = -0.0065",
            "",
            "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
            "        dataframe = super().populate_entry_trend(dataframe, metadata)",
            '        micro_up = (dataframe["auto_ret_3m"] > 0) & (dataframe["auto_ret_10m"] > 0.00015)',
            '        micro_down = (dataframe["auto_ret_3m"] < 0) & (dataframe["auto_ret_10m"] < -0.00015)',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_up_regime"] & micro_up & dataframe["auto_rsi"].between(44, 72) & (dataframe["volume"] > 0)), ["enter_long", "enter_tag"]] = (1, "sample_micro_long")',
            '        dataframe.loc[(dataframe["risk_allowed"] & dataframe["trend_down_regime"] & micro_down & dataframe["auto_rsi"].between(28, 58) & (dataframe["volume"] > 0)), ["enter_short", "enter_tag"]] = (1, "sample_micro_short")',
            "        return dataframe",
        ],
    },
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_source() -> str:
    lines = [
        '"""Sample-expansion research strategies.',
        "",
        "Generated by user_data/strategy_research/generate_sample_expansion_strategies.py.",
        "Research-only: do not promote without full validation.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from pandas import DataFrame",
        "",
        "from autonomous_research_strategies import (",
        "    AutoDefensiveFlatFilterStrategy,",
        "    AutoMicroMomentumConfirmationStrategy,",
        "    AutoTrendPullbackContinuationStrategy,",
        ")",
        "from iterative_research_strategies import IterAutoDefensiveFlatFilterV2Strategy",
        "",
        f"GENERATED_AT_UTC = {utc_stamp()!r}",
        "",
    ]
    for item in VARIANTS:
        lines.extend([f"class {item['name']}({item['base']}):", f'    """{item["family"]}: broaden entry sample while preserving risk_allowed."""'])
        lines.extend(item["body"])
        lines.append("")
    return "\n".join(lines) + "\n"


def registry_payload() -> dict[str, Any]:
    return {
        "generated_at_utc": utc_stamp(),
        "generated_strategy_file": rel(GENERATED_FILE),
        "research_mode": "sample_expansion_generation",
        "trigger": "too_few_trades_after_family_diversity",
        "strategies": [
            {
                "name": item["name"],
                "base_strategy": item["base"],
                "family": item["family"],
                "bucket": item["bucket"],
                "source": "sample_expansion_generator",
                "hypothesis": "A lightly broadened entry can produce enough trades to judge edge without removing risk_allowed controls.",
                "risk_notes": "Research-only low-leverage futures variant; must pass cost, matrix, bias, and walk-forward gates.",
            }
            for item in VARIANTS
        ],
    }


def experiment_payload() -> dict[str, Any]:
    return {
        "id": "sample_expansion_strategy_lab",
        "title": "Sample expansion for promising under-sampled strategy families",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": ["1m"],
        "timeranges": ["20240101-20260622"],
        "matrix": {
            "timeranges": [
                {"name": "smoke", "label": "Recent smoke", "timerange": "20260101-20260622"},
                {"name": "full", "label": "Full local sample", "timerange": "20240101-20260622"},
            ]
        },
        "fee": 0.0005,
        "strategies": [item["name"] for item in VARIANTS],
        "strategy_groups": {
            "sample_expansion": [item["name"] for item in VARIANTS],
        },
        "checks": {"backtesting": True, "recursive_analysis": False, "lookahead_analysis": False},
        "notes": [
            "Targets strategies rejected mainly for too few trades.",
            "Broadens entry thresholds but keeps risk_allowed and low leverage.",
            "Promotion still requires matrix, walk-forward, cost, recursive, and lookahead checks.",
        ],
    }


def write_ledger(payload: dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sample Expansion Plan",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Strategy file: `{payload['generated_strategy_file']}`",
        f"- Experiment: `{rel(EXPERIMENT_PATH)}`",
        "",
        "| Strategy | Base | Bucket | Family |",
        "|---|---|---|---|",
    ]
    for item in payload["strategies"]:
        lines.append("| {name} | {base_strategy} | {bucket} | {family} |".format(**item))
    LEDGER_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    payload = registry_payload()
    GENERATED_FILE.write_text(build_source(), encoding="utf-8")
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    EXPERIMENT_PATH.write_text(json.dumps(experiment_payload(), indent=2, ensure_ascii=False), encoding="utf-8")
    write_ledger(payload)
    print(f"Wrote {rel(GENERATED_FILE)}")
    print(f"Wrote {rel(REGISTRY_PATH)}")
    print(f"Wrote {rel(EXPERIMENT_PATH)}")
    print(f"Wrote {rel(LEDGER_PATH)}")


if __name__ == "__main__":
    main()
