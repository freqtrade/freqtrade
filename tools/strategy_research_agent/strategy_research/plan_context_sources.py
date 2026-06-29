#!/usr/bin/env python3
"""Plan non-pure-OHLCV context-source experiments after seed-family retirement."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
OUTPUT_DIR = AGENT_ROOT / "context_sources"
PLAN_JSON = OUTPUT_DIR / "latest_context_source_plan.json"
PLAN_MD = OUTPUT_DIR / "latest_context_source_plan.md"
REGISTRY_PATH = OUTPUT_DIR / "context_source_strategy_registry.json"
EXPERIMENT_PATH = AGENT_ROOT / "experiments/context_source_experiment.json"
RETIRED_LEDGER = AGENT_ROOT / "experiments/retired_seed_family_ledger.json"
AUX_CONVERSION = AGENT_ROOT / "cost_audits/latest_freqtrade_futures_aux_conversion.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aux_status() -> dict[str, Any]:
    payload = load_json(AUX_CONVERSION)
    files = payload.get("converted_files", [])
    return {
        "available": bool(files),
        "source": rel(AUX_CONVERSION) if files else None,
        "files": files,
        "funding_pairs": sorted({item.get("pair") for item in files if item.get("candle_type") == "funding_rate"}),
        "mark_pairs": sorted({item.get("pair") for item in files if item.get("candle_type") == "mark"}),
        "latest_aux_utc": max((item.get("last_utc") or "" for item in files), default=None),
    }


def build_payload() -> dict[str, Any]:
    retired = load_json(RETIRED_LEDGER)
    aux = aux_status()
    strategies = [
        "BtcEthFuturesBtcLead0mEthPullbackShortStrategy",
        "BtcEthFuturesBtcLead15mEthPullbackShortStrategy",
        "BtcEthFuturesBtcLead60mEthPullbackShortStrategy",
        "BtcEthFuturesBtcLead240mEthPullbackShortStrategy",
    ]
    strategy_registry = [
        {
            "name": name,
            "family": "context-btc-eth-lead-lag",
            "source": "context_source_planner",
            "source_type": "local_translated_strategy",
            "hypothesis": "Use lagged BTC bearish continuation as a simple context source for ETH pullback shorts.",
            "risk_notes": (
                "Research-only futures short strategy. Requires cost, regime, recursive, lookahead, "
                "and walk-forward validation before any dry-run promotion."
            ),
        }
        for name in strategies
    ]
    experiment = {
        "id": "context_source_lead_lag_lab",
        "title": "Context-source BTC lead-lag and aux-data research",
        "profile_ref": "strategy_registry.json",
        "timeframes": ["1m"],
        "timeranges": ["20240101-20260622"],
        "matrix": {
            "timeranges": [
                {"name": "recent", "label": "Recent post-2026 smoke", "timerange": "20260101-20260622"},
                {"name": "full", "label": "Full local sample", "timerange": "20240101-20260622"},
            ]
        },
        "fee": 0.0005,
        "strategies": strategies,
        "strategy_groups": {
            "btc_eth_lead_lag": strategies,
        },
        "checks": {"backtesting": True, "recursive_analysis": False, "lookahead_analysis": False},
        "notes": [
            "Generated because simple 1m OHLCV seed family was retired.",
            "Uses BTC signal timing as a cross-asset context source for ETH shorts.",
            "Funding/mark aux data availability is recorded for follow-up cost-aware planning.",
        ],
    }
    return {
        "generated_at_utc": utc_stamp(),
        "trigger": "context_seed_negative_expectancy",
        "retired_family": retired.get("retired_family"),
        "aux_status": aux,
        "research_tracks": [
            {
                "id": "btc_eth_lead_lag",
                "status": "ready",
                "experiment": rel(EXPERIMENT_PATH),
                "registry": rel(REGISTRY_PATH),
                "hypothesis": "BTC bearish regime may lead ETH pullback shorts by a lagged window.",
                "success_gate": "At least one lag window beats ETH self-only baseline with enough trades and acceptable drawdown.",
            },
            {
                "id": "funding_mark_cost_context",
                "status": "data_available" if aux["available"] else "blocked_missing_aux_data",
                "experiment": None,
                "hypothesis": "Funding/mark data should shape cost-aware promotion and later strategy features.",
                "success_gate": "Aux data is present through the tested timerange or the gap is recorded before promotion.",
            },
        ],
        "experiment": experiment,
        "strategy_registry": {
            "generated_at_utc": utc_stamp(),
            "source": "context_source_planner",
            "strategies": strategy_registry,
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Context Source Plan",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Trigger: `{payload['trigger']}`",
        f"- Retired family: `{payload.get('retired_family')}`",
        f"- Aux available: `{payload['aux_status']['available']}`",
        f"- Latest aux UTC: `{payload['aux_status'].get('latest_aux_utc')}`",
        "",
        "## Research Tracks",
        "",
        "| Track | Status | Hypothesis | Success Gate |",
        "|---|---|---|---|",
    ]
    for item in payload["research_tracks"]:
        lines.append("| {id} | {status} | {hypothesis} | {success_gate} |".format(**item))
    lines.extend(
        [
            "",
            "## Experiment",
            "",
            f"- Path: `{rel(EXPERIMENT_PATH)}`",
            f"- Registry: `{rel(REGISTRY_PATH)}`",
            f"- Strategies: `{len(payload['experiment']['strategies'])}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(PLAN_MD, payload)
    REGISTRY_PATH.write_text(json.dumps(payload["strategy_registry"], indent=2, ensure_ascii=False), encoding="utf-8")
    EXPERIMENT_PATH.write_text(json.dumps(payload["experiment"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {rel(PLAN_JSON)}")
    print(f"Wrote {rel(PLAN_MD)}")
    print(f"Wrote {rel(REGISTRY_PATH)}")
    print(f"Wrote {rel(EXPERIMENT_PATH)}")


if __name__ == "__main__":
    main()
