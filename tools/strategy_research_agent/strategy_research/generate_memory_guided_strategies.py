#!/usr/bin/env python3
"""Generate isolated strategy variants from memory-guided hypotheses."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
PLAN_PATH = AGENT_ROOT / "experiments/memory_guided_hypothesis_plan.json"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "memory_guided_research_strategies.py"
REGISTRY_PATH = AGENT_ROOT / "experiments/memory_guided_strategy_registry.json"
EXPERIMENT_PATH = AGENT_ROOT / "experiments/memory_guided_strategy_experiment.json"
LEDGER_PATH = AGENT_ROOT / "experiments/memory_guided_strategy_ledger.md"
VERIFICATION_ONLY_BLOCKERS = {"bias_checks_missing", "lookahead_or_recursive_unverified"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--timerange", default="20240101-20260622")
    parser.add_argument("--smoke-timerange", default="20260101-20260201")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe_class_name(value: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise ValueError(f"Unsafe strategy class name: {value}")
    return value


def camel(value: str) -> str:
    return "".join(piece[:1].upper() + piece[1:] for piece in re.split(r"[^A-Za-z0-9]+", value) if piece)


def generated_name(base: str, blocker: str) -> str:
    return f"Memory{base}{camel(blocker)}Strategy"


def source_header(generated_at: str, btc_bases: list[str], translated_bases: list[str]) -> list[str]:
    lines = [
        '"""Memory-guided research strategy variants.',
        "",
        "Do not edit by hand. Re-generate with user_data/strategy_research/generate_memory_guided_strategies.py.",
        "These classes are research-only and must not be promoted to dry-run or live without manual approval.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        "from pandas import DataFrame",
        "",
        "sys.path.append(str(Path(__file__).resolve().parents[1]))",
        "sys.path.append(str(Path(__file__).resolve().parent))",
        "",
    ]
    if btc_bases:
        lines.append(f"from btc_eth_risk_controlled_strategies import {', '.join(btc_bases)}")
    if translated_bases:
        lines.append(f"from source_translated_strategies import {', '.join(translated_bases)}")
    lines.extend(["", "", f"GENERATED_AT_UTC = {generated_at!r}", ""])
    return lines


def filter_method(lines: list[str]) -> list[str]:
    body = [
        "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe = super().populate_entry_trend(dataframe, metadata)",
        '        dataframe["memory_ret_5m"] = dataframe["ret_5m"] if "ret_5m" in dataframe.columns else dataframe["close"] / dataframe["close"].shift(5) - 1.0',
        '        dataframe["memory_ret_24h"] = dataframe["ret_24h"] if "ret_24h" in dataframe.columns else dataframe["close"] / dataframe["close"].shift(1440) - 1.0',
        '        dataframe["memory_ret_72h"] = dataframe["ret_72h"] if "ret_72h" in dataframe.columns else dataframe["close"] / dataframe["close"].shift(4320) - 1.0',
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


def variant_source(item: dict[str, Any], class_name: str) -> list[str]:
    blocker = item["blocker"]
    base = safe_class_name(item["strategy"])
    lines = [
        "",
        f"class {class_name}({base}):",
        f"    \"\"\"Memory-guided variant for blocker: {blocker}.\"\"\"",
        "",
        '    minimal_roi = {"0": 1.20, "180": 1.50, "360": 1.00}',
        "    stoploss = -0.60",
        "",
        "    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, side, **kwargs):",
        "        return min(50.0, max_leverage)",
        "",
    ]
    if blocker in {"cost_evidence_missing", "cost_not_estimated", "negative_after_cost", "stress_cost_failure"}:
        lines.extend(
            filter_method(
                [
                    'move_floor = dataframe["memory_ret_5m"].abs().rolling(60).mean().fillna(0) * 1.15',
                    'short_edge = (dataframe["memory_ret_5m"] < -move_floor) & (dataframe["memory_ret_24h"] < -0.004)',
                    'long_edge = (dataframe["memory_ret_5m"] > move_floor) & (dataframe["memory_ret_24h"] > 0.004)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & ~short_edge, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_long", 0).fillna(0).astype(bool) & ~long_edge, ["enter_long", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "memory_cost_edge_short"',
                    'dataframe.loc[dataframe.get("enter_long", 0).fillna(0).astype(bool), "enter_tag"] = "memory_cost_edge_long"',
                ]
            )
        )
    elif blocker in {"matrix_not_robust", "fragile_matrix", "matrix_not_tested"}:
        lines.extend(
            filter_method(
                [
                    'short_regime = dataframe["trend_down_regime"] & ~dataframe["range_regime"] & (dataframe["memory_ret_72h"] < -0.012)',
                    'long_regime = dataframe["trend_up_regime"] & ~dataframe["range_regime"] & (dataframe["memory_ret_72h"] > 0.012)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & ~short_regime, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_long", 0).fillna(0).astype(bool) & ~long_regime, ["enter_long", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "memory_regime_short"',
                    'dataframe.loc[dataframe.get("enter_long", 0).fillna(0).astype(bool), "enter_tag"] = "memory_regime_long"',
                ]
            )
        )
    elif blocker in {"walk_forward_not_passed", "loss_exit_quality", "weak_profit_factor"}:
        lines.extend(
            filter_method(
                [
                    'bad_state = dataframe["high_vol_regime"] | dataframe["range_regime"]',
                    'short_confirm = ~bad_state & (dataframe["close"] < dataframe["close"].shift(3)) & (dataframe["memory_ret_5m"] < -0.001)',
                    'long_confirm = ~bad_state & (dataframe["close"] > dataframe["close"].shift(3)) & (dataframe["memory_ret_5m"] > 0.001)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool) & ~short_confirm, ["enter_short", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_long", 0).fillna(0).astype(bool) & ~long_confirm, ["enter_long", "enter_tag"]] = (0, None)',
                    'dataframe.loc[dataframe.get("enter_short", 0).fillna(0).astype(bool), "enter_tag"] = "memory_walk_forward_short"',
                    'dataframe.loc[dataframe.get("enter_long", 0).fillna(0).astype(bool), "enter_tag"] = "memory_walk_forward_long"',
                ]
            )
        )
    else:
        lines.extend(["    pass"])
    lines.append("")
    return lines


def selected_hypotheses(plan: dict[str, Any], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generated = []
    skipped = []
    for item in plan.get("hypotheses", [])[:limit]:
        if item.get("blocker") in VERIFICATION_ONLY_BLOCKERS:
            skipped.append({"hypothesis_id": item.get("hypothesis_id"), "strategy": item.get("strategy"), "reason": "verification_only_blocker"})
        else:
            generated.append(item)
    return generated, skipped


def build_source(hypotheses: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bases = sorted({safe_class_name(item["strategy"]) for item in hypotheses})
    translated = [item for item in bases if item.startswith("SourceTranslated")]
    btc = [item for item in bases if item not in translated]
    lines = source_header(generated_at, btc, translated)
    registry = []
    for item in hypotheses:
        base = safe_class_name(item["strategy"])
        class_name = generated_name(base, item["blocker"])
        lines.extend(variant_source(item, class_name))
        registry.append(
            {
                "name": class_name,
                "base_strategy": item["strategy"],
                "hypothesis_id": item["hypothesis_id"],
                "family": f"memory-guided-{item['blocker']}",
                "source": "memory_guided_hypothesis_plan",
                "hypothesis": item["objective"],
                "blocker": item["blocker"],
                "change_set": item["proposed_changes"],
                "success_gate": item["success_gate"],
                "risk_notes": item["risk_notes"],
            }
        )
    return "\n".join(lines) + "\n", registry


def build_experiment(registry: list[dict[str, Any]], timerange: str, smoke_timerange: str) -> dict[str, Any]:
    memory_strategies = [item["name"] for item in registry]
    strategies = list(dict.fromkeys(memory_strategies))
    return {
        "id": "memory_guided_strategy_lab",
        "title": "Memory-guided strategy variants",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": ["15m"],
        "timeranges": [timerange],
        "matrix": {
            "timeranges": [
                {"name": "smoke", "label": "Recent smoke", "timerange": smoke_timerange},
                {"name": "full", "label": "Full sample", "timerange": timerange},
            ]
        },
        "fee": 0.0005,
        "strategies": strategies,
        "strategy_groups": {
            "memory_guided": memory_strategies,
        },
        "checks": {"backtesting": True, "recursive_analysis": False, "lookahead_analysis": False},
        "notes": [
            "Generated only from memory-guided hypotheses after research memory and knowledge graph are rebuilt.",
            "Generated classes explicitly lock ROI, stoploss, and 50x leverage to the current futures risk policy.",
            "Verification-only blockers are skipped by the code generator.",
            "Promotion requires scorecard, matrix, walk-forward, cost, and bias checks.",
        ],
    }


def ledger_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Memory-Guided Strategy Ledger",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Generated strategies: `{len(payload['strategies'])}`",
        f"- Skipped plans: `{len(payload['skipped_hypotheses'])}`",
        f"- Strategy file: `{payload['generated_strategy_file']}`",
        "",
        "| Strategy | Base | Blocker | Hypothesis | Success Gate |",
        "|---|---|---|---|---|",
    ]
    for item in payload["strategies"]:
        lines.append(
            "| {name} | {base_strategy} | {blocker} | {hypothesis} | {success_gate} |".format(**item)
        )
    if payload["skipped_hypotheses"]:
        lines.extend(["", "## Skipped Hypotheses", "", "| ID | Strategy | Reason |", "|---|---|---|"])
        for item in payload["skipped_hypotheses"]:
            lines.append("| {hypothesis_id} | {strategy} | {reason} |".format(**item))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    plan = load_json(PLAN_PATH)
    hypotheses, skipped = selected_hypotheses(plan, args.limit)
    if not hypotheses:
        raise SystemExit("No actionable memory-guided hypotheses found. Run plan_memory_guided_hypotheses.py first.")
    source, registry_entries = build_source(hypotheses)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_plan": rel(PLAN_PATH),
        "generated_strategy_file": rel(GENERATED_FILE),
        "research_mode": "memory_guided_strategy_generation",
        "strategies": registry_entries,
        "skipped_hypotheses": skipped,
    }
    experiment = build_experiment(registry_entries, args.timerange, args.smoke_timerange)
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(source, encoding="utf-8")
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    EXPERIMENT_PATH.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    LEDGER_PATH.write_text(ledger_markdown(payload), encoding="utf-8")
    print(f"Wrote {rel(GENERATED_FILE)}")
    print(f"Wrote {rel(REGISTRY_PATH)}")
    print(f"Wrote {rel(EXPERIMENT_PATH)}")
    print(f"Wrote {rel(LEDGER_PATH)}")


if __name__ == "__main__":
    main()
