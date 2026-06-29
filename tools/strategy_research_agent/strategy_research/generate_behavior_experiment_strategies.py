#!/usr/bin/env python3
"""Generate Freqtrade strategy variants from behavior-driven experiment plans."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
PLAN_PATH = AGENT_ROOT / "behavior_experiments/latest_behavior_experiment_plan.json"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "behavior_experiment_strategies.py"
REGISTRY_PATH = AGENT_ROOT / "experiments/behavior_experiment_strategy_registry.json"
EXPERIMENT_PATH = AGENT_ROOT / "experiments/behavior_experiment_strategy_experiment.json"
LEDGER_PATH = AGENT_ROOT / "experiments/behavior_experiment_hypothesis_ledger.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=9)
    parser.add_argument("--timerange", default="20240101-20260622")
    parser.add_argument("--smoke-timerange", default="20260101-20260201")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_class_name(value: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise ValueError(f"Unsafe strategy class name: {value}")
    return value


def camel(value: str) -> str:
    return "".join(piece[:1].upper() + piece[1:] for piece in re.split(r"[^A-Za-z0-9]+", value) if piece)


def generated_name(base: str, experiment_id: str) -> str:
    return f"Behavior{base}{camel(experiment_id)}Strategy"


def source_header(generated_at: str, base_names: list[str]) -> list[str]:
    imports = ", ".join(base_names)
    return [
        '"""Behavior-driven research strategy variants.',
        "",
        "Do not edit by hand. Re-generate with user_data/strategy_research/generate_behavior_experiment_strategies.py.",
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
        "",
        "sys.path.append(str(Path(__file__).resolve().parents[1]))",
        "",
        f"from btc_eth_risk_controlled_strategies import {imports}",
        "",
        "",
        f"GENERATED_AT_UTC = {generated_at!r}",
        "",
    ]


def filter_method(lines: list[str]) -> list[str]:
    body = [
        "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe = super().populate_entry_trend(dataframe, metadata)",
        '        if "enter_short" not in dataframe.columns:',
        '            dataframe["enter_short"] = 0',
        '        if "enter_long" not in dataframe.columns:',
        '            dataframe["enter_long"] = 0',
        '        if "enter_tag" not in dataframe.columns:',
        '            dataframe["enter_tag"] = None',
    ]
    body.extend(f"        {line}" for line in lines)
    body.append("        return dataframe")
    return body


def variant_source(plan: dict[str, Any], class_name: str) -> list[str]:
    experiment = plan["experiment_id"]
    lines = [
        "",
        f"class {class_name}({safe_class_name(plan['strategy'])}):",
        f"    \"\"\"Behavior experiment: {experiment}.\"\"\"",
        "",
    ]
    if experiment == "stop_loss_and_invalidation_sweep":
        lines.extend(
            [
                '    minimal_roi = {"180": 0.0, "45": 0.010, "0": 0.018}',
                "    stoploss = -0.095",
                "",
            ]
        )
        lines.extend(
            filter_method(
                [
                    'strong_confirm = (dataframe["minus_di"] > dataframe["plus_di"] * 1.22) & (dataframe["ret_24h"] < -0.0045) & (dataframe["ret_72h"] < -0.012)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & ~strong_confirm, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "behavior_stop_invalidation_short"',
                ]
            )
        )
    elif experiment == "cooldown_after_loss_cluster":
        lines.extend(
            filter_method(
                [
                    'bad_micro_regime = ((dataframe["range_regime"]) | (dataframe["high_vol_regime"]) | (dataframe["plus_di"] > dataframe["minus_di"] * 1.03)).rolling(360).max().fillna(0).astype(bool)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & bad_micro_regime, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "behavior_cooldown_proxy_short"',
                ]
            )
        )
    elif experiment == "short_only_regime_split":
        lines.extend(
            filter_method(
                [
                    'bearish_only = (dataframe["trend_down_regime"]) & (dataframe["ret_24h"] < -0.006) & (dataframe["ret_72h"] < -0.018) & (dataframe["contract_score"] < -0.025)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & ~bearish_only, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "behavior_short_regime_split"',
                ]
            )
        )
    elif experiment.startswith("disable_dragging_pair_"):
        lines.extend(
            filter_method(
                [
                    'if metadata.get("pair") == "BTC/USDT:USDT":',
                    "    dataframe[[\"enter_long\", \"enter_short\", \"enter_tag\"]] = (0, 0, None)",
                    "    return dataframe",
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "behavior_pair_filtered_short"',
                ]
            )
        )
    elif experiment == "entry_timing_confirmation_sweep":
        lines.extend(
            filter_method(
                [
                    'confirm = (dataframe["close"] < dataframe["close"].shift(3)) & (dataframe["ret_5m"] < -0.0012) & (dataframe["ret_24h"] < -0.004)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & ~confirm, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "behavior_entry_timing_short"',
                ]
            )
        )
    else:
        lines.extend(
            [
                "    pass",
            ]
        )
    lines.append("")
    return lines


def selected_plans(plan_payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return plan_payload.get("plans", [])[:limit]


def build_source(plans: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bases = sorted({safe_class_name(plan["strategy"]) for plan in plans})
    lines = source_header(generated_at, bases)
    registry = []
    for plan in plans:
        class_name = generated_name(safe_class_name(plan["strategy"]), plan["experiment_id"])
        lines.extend(variant_source(plan, class_name))
        registry.append(
            {
                "name": class_name,
                "base_strategy": plan["strategy"],
                "experiment_id": plan["experiment_id"],
                "family": f"behavior-{plan['experiment_id']}",
                "source": "behavior_experiment_plan",
                "hypothesis": plan["hypothesis"],
                "change_set": plan["change_set"],
                "success_gate": plan["success_gate"],
                "risk_notes": plan["risk_note"],
            }
        )
    return "\n".join(lines) + "\n", registry


def build_experiment(registry: list[dict[str, Any]], timerange: str, smoke_timerange: str) -> dict[str, Any]:
    return {
        "id": "behavior_experiment_strategy_lab",
        "title": "Behavior-driven strategy experiment variants",
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
        "strategies": [item["name"] for item in registry],
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False,
        },
        "notes": [
            "Generated from trade behavior diagnostics.",
            "Variants are isolated subclasses of existing candidate strategies.",
            "Promotion requires behavior improvement plus matrix, walk-forward, cost, and bias checks.",
        ],
    }


def ledger_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Behavior Experiment Strategy Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source plan: `{payload['source_plan']}`",
        f"- Strategy file: `{payload['generated_strategy_file']}`",
        "",
        "| Strategy | Base | Experiment | Hypothesis | Success Gate |",
        "|---|---|---|---|---|",
    ]
    for item in payload["strategies"]:
        lines.append(
            "| {name} | {base_strategy} | {experiment_id} | {hypothesis} | {success_gate} |".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    plan_payload = load_json(PLAN_PATH)
    plans = selected_plans(plan_payload, args.limit)
    if not plans:
        raise SystemExit("No behavior experiment plans found. Run plan_behavior_experiments.py first.")
    source, registry_entries = build_source(plans)
    registry_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_plan": str(PLAN_PATH.relative_to(REPO_ROOT)),
        "generated_strategy_file": str(GENERATED_FILE.relative_to(REPO_ROOT)),
        "research_mode": "behavior_driven_strategy_generation",
        "strategies": registry_entries,
    }
    experiment = build_experiment(registry_entries, args.timerange, args.smoke_timerange)
    if args.dry_run:
        print(source)
        print(json.dumps(registry_payload, indent=2, ensure_ascii=False))
        print(json.dumps(experiment, indent=2, ensure_ascii=False))
        return

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(source, encoding="utf-8")
    REGISTRY_PATH.write_text(json.dumps(registry_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    EXPERIMENT_PATH.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    LEDGER_PATH.write_text(ledger_markdown(registry_payload), encoding="utf-8")
    print(f"Wrote {GENERATED_FILE.relative_to(REPO_ROOT)}")
    print(f"Wrote {REGISTRY_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {EXPERIMENT_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {LEDGER_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
