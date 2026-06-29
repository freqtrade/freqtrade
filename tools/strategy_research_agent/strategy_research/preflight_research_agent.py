#!/usr/bin/env python3
"""Preflight checks for the local strategy research agent."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DEFAULT_CONFIG = AGENT_ROOT / "agent_config.json"
DEFAULT_REGISTRY = AGENT_ROOT / "strategy_registry.json"
WORKFLOW_GATE = AGENT_ROOT / "enforce_agent_workflow_gate.py"
LEVERAGE_SOURCE_PATHS = [
    REPO_ROOT / "user_data/strategies",
    AGENT_ROOT,
]
LEVERAGE_METHOD_RE = re.compile(r"def\s+leverage\s*\([^)]*\)\s*->\s*float:\s*(.*?)(?=\n    def |\nclass |\Z)", re.DOTALL)
LEVERAGE_RETURN_RE = re.compile(
    r"return\s+(?:min\(\s*)?([0-9]+(?:\.[0-9]+)?)(?:\s*,\s*max_leverage\s*\))?"
)


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Write machine-readable preflight output.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def add(checks: list[Check], name: str, status: str, detail: str) -> None:
    checks.append(Check(name=name, status=status, detail=detail))


def pair_data_path(pair: str, timeframe: str) -> Path:
    if ":" in pair:
        stem = pair.replace("/", "_").replace(":", "_")
        return REPO_ROOT / f"user_data/data/binance/futures/{stem}-{timeframe}-futures.feather"
    stem = pair.replace("/", "_")
    return REPO_ROOT / f"user_data/data/binance/{stem}-{timeframe}.feather"


def check_python(checks: list[Check]) -> None:
    python_path = REPO_ROOT / ".venv/bin/python"
    freqtrade_path = REPO_ROOT / ".venv/bin/freqtrade"
    if python_path.exists():
        add(checks, "python", "ok", rel(python_path))
    else:
        add(checks, "python", "fail", f"Missing {rel(python_path)}")
    if freqtrade_path.exists():
        completed = subprocess.run(
            [str(freqtrade_path), "--version"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        first_line = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
        status = "ok" if completed.returncode == 0 else "fail"
        add(checks, "freqtrade", status, first_line or f"exit={completed.returncode}")
    else:
        add(checks, "freqtrade", "fail", f"Missing {rel(freqtrade_path)}")


def check_agent_config(checks: list[Check]) -> dict[str, Any] | None:
    if not DEFAULT_CONFIG.exists():
        add(checks, "agent_config", "fail", f"Missing {rel(DEFAULT_CONFIG)}")
        return None
    try:
        config = load_json(DEFAULT_CONFIG)
    except json.JSONDecodeError as exc:
        add(checks, "agent_config", "fail", f"Invalid JSON: {exc}")
        return None

    agent = config.get("agent", {})
    unsafe_flags = [
        key
        for key in ["live_trading_allowed", "may_modify_live_config", "may_read_private_api_keys", "may_run_external_code"]
        if agent.get(key) is not False
    ]
    if unsafe_flags:
        add(checks, "safety_flags", "fail", "Unsafe agent flags: " + ", ".join(unsafe_flags))
    else:
        add(checks, "safety_flags", "ok", "Research-only flags are locked down.")
    return config


def check_fixed_risk_policy(checks: list[Check], config: dict[str, Any] | None) -> None:
    if not config:
        return
    scope = config.get("research_scope", {})
    analysis = config.get("analysis", {})
    policy = config.get("risk_policy", {})
    expected_roi = {"0": 0.30, "120": 1.00, "240": 0.60}

    if scope.get("allowed_markets") == ["Binance USDT-M futures"]:
        add(checks, "risk_policy:market_scope", "ok", "Futures-only market scope.")
    else:
        add(checks, "risk_policy:market_scope", "fail", f"Unexpected markets: {scope.get('allowed_markets')}")

    if analysis.get("default_leverage") == 50 and analysis.get("default_leverage_grid") == [50]:
        add(checks, "risk_policy:leverage", "ok", "default_leverage=50 and grid=[50].")
    else:
        add(
            checks,
            "risk_policy:leverage",
            "fail",
            f"default_leverage={analysis.get('default_leverage')} grid={analysis.get('default_leverage_grid')}",
        )

    if policy.get("market_type") == "futures" and policy.get("margin_mode") == "isolated":
        add(checks, "risk_policy:margin", "ok", "futures isolated.")
    else:
        add(checks, "risk_policy:margin", "fail", f"market_type={policy.get('market_type')} margin={policy.get('margin_mode')}")

    if policy.get("default_leverage") == 50:
        add(checks, "risk_policy:default_leverage", "ok", "risk_policy default_leverage=50.")
    else:
        add(checks, "risk_policy:default_leverage", "fail", f"default_leverage={policy.get('default_leverage')}")

    if policy.get("minimal_roi") == expected_roi:
        add(checks, "risk_policy:minimal_roi", "ok", "minimal_roi fixed at 0:0.30, 120:1.00, 240:0.60.")
    else:
        add(checks, "risk_policy:minimal_roi", "fail", f"minimal_roi={policy.get('minimal_roi')}")

    if policy.get("stoploss") == -0.60:
        add(checks, "risk_policy:stoploss", "ok", "stoploss=-0.60.")
    else:
        add(checks, "risk_policy:stoploss", "fail", f"stoploss={policy.get('stoploss')}")


def check_strategy_leverage_overrides(checks: list[Check]) -> None:
    offenders: list[str] = []
    scanned = 0
    for root in LEVERAGE_SOURCE_PATHS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            matches: list[str] = []
            for method in LEVERAGE_METHOD_RE.findall(text):
                matches.extend(LEVERAGE_RETURN_RE.findall(method))
            if not matches:
                continue
            scanned += 1
            for value in matches:
                if float(value) != 50.0:
                    offenders.append(f"{rel(path)} returns {value}x")
    if offenders:
        add(checks, "risk_policy:strategy_leverage_overrides", "fail", "; ".join(offenders[:20]))
    else:
        add(checks, "risk_policy:strategy_leverage_overrides", "ok", f"All scanned leverage overrides are 50x ({scanned} files).")


def check_workflow_gate(checks: list[Check]) -> None:
    if not WORKFLOW_GATE.exists():
        add(checks, "strategy_agent_gate", "fail", f"Missing {rel(WORKFLOW_GATE)}")
        return
    completed = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/python"), str(WORKFLOW_GATE), "--json"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        add(checks, "strategy_agent_gate", "fail", completed.stdout[-2000:].strip())
        return
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        add(checks, "strategy_agent_gate", "fail", f"Invalid gate output: {exc}")
        return
    required = payload.get("must_load_before_research", [])
    add(checks, "strategy_agent_gate", "ok", f"Loaded {len(required)} fixed workflow artifacts.")


def check_registry(checks: list[Check]) -> dict[str, Any] | None:
    if not DEFAULT_REGISTRY.exists():
        add(checks, "strategy_registry", "fail", f"Missing {rel(DEFAULT_REGISTRY)}")
        return None
    try:
        registry = load_json(DEFAULT_REGISTRY)
    except json.JSONDecodeError as exc:
        add(checks, "strategy_registry", "fail", f"Invalid JSON: {exc}")
        return None
    strategies = registry.get("strategies", [])
    profile = registry.get("profile", {})
    if strategies:
        add(checks, "strategy_registry", "ok", f"{len(strategies)} registered research strategies.")
    else:
        add(checks, "strategy_registry", "warn", "No registered research strategies.")
    if profile.get("config") and (REPO_ROOT / profile["config"]).exists():
        add(checks, "freqtrade_config", "ok", profile["config"])
    else:
        add(checks, "freqtrade_config", "fail", f"Missing profile config: {profile.get('config')}")
    return registry


def check_data(checks: list[Check], registry: dict[str, Any] | None) -> None:
    if not registry:
        return
    profile = registry.get("profile", {})
    timeframe = profile.get("timeframe", "1m")
    pairs = profile.get("pairs", [])
    for pair in pairs:
        path = pair_data_path(pair, timeframe)
        if not path.exists():
            add(checks, f"data:{pair}:{timeframe}", "fail", f"Missing {rel(path)}")
            continue
        try:
            frame = pd.read_feather(path, columns=["date"])
            dates = pd.to_datetime(frame["date"], utc=True).sort_values()
            first = dates.iloc[0].isoformat() if len(dates) else "empty"
            last = dates.iloc[-1].isoformat() if len(dates) else "empty"
            status = "ok" if len(dates) >= 10000 else "warn"
            add(checks, f"data:{pair}:{timeframe}", status, f"{len(dates)} rows, {first} -> {last}")
        except Exception as exc:  # noqa: BLE001 - preflight should surface local data problems.
            add(checks, f"data:{pair}:{timeframe}", "fail", f"{rel(path)}: {exc}")


def check_outputs(checks: list[Check]) -> None:
    paths = {
        "reports": AGENT_ROOT / "reports/agent_report_index.json",
        "dashboard": AGENT_ROOT / "dashboard/index.html",
        "assessment": AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.md",
        "matrix_summary": AGENT_ROOT / "matrix_summaries/latest_matrix_summary.md",
        "context_source_plan": AGENT_ROOT / "context_sources/latest_context_source_plan.md",
        "manual_trade_playbook": AGENT_ROOT / "manual_playbook/latest_manual_trade_playbook.md",
        "manual_direction_plan": AGENT_ROOT / "manual_playbook/latest_manual_direction_plan.md",
        "manual_entry_confirmation_plan": AGENT_ROOT / "manual_playbook/latest_manual_entry_confirmation_plan.md",
        "manual_abstention_plan": AGENT_ROOT / "manual_playbook/latest_manual_abstention_plan.md",
        "manual_strong_confirmation_plan": AGENT_ROOT / "manual_playbook/latest_manual_strong_confirmation_plan.md",
        "multi_timeframe_kline_plan": AGENT_ROOT / "manual_playbook/latest_multi_timeframe_kline_plan.md",
        "manual_research_review": AGENT_ROOT / "manual_playbook/latest_manual_research_review.md",
        "walk_forward_summary": AGENT_ROOT / "walk_forward_summaries/latest_walk_forward_summary.md",
        "promotion_report": AGENT_ROOT / "promotion_reports/latest_promotion_report.md",
        "research_agenda": AGENT_ROOT / "research_agendas/latest_research_agenda.md",
        "agenda_run": AGENT_ROOT / "agenda_runs/latest_agenda_run.md",
        "trade_behavior": AGENT_ROOT / "trade_behavior/latest_trade_behavior.md",
        "behavior_experiments": AGENT_ROOT / "behavior_experiments/latest_behavior_experiment_plan.md",
        "behavior_variant_ledger": AGENT_ROOT / "experiments/behavior_experiment_hypothesis_ledger.md",
        "failure_attribution": AGENT_ROOT / "failure_attribution/latest_failure_attribution.md",
        "mature_researcher": AGENT_ROOT / "mature_researcher/latest_researcher_decision.md",
        "mature_researcher_queue": AGENT_ROOT / "mature_researcher/latest_response_queue.md",
        "agent_iteration_review": AGENT_ROOT / "agent_iterations/latest_iteration_review.md",
        "agent_improvement_queue": AGENT_ROOT / "agent_iterations/improvement_queue.json",
        "strategy_lineage": AGENT_ROOT / "strategy_library/latest_strategy_lineage.md",
        "research_memory": AGENT_ROOT / "research_memory/latest_research_memory.md",
        "factor_research": AGENT_ROOT / "factors/latest_factor_research.md",
        "factor_strategy_plan": AGENT_ROOT / "factors/latest_factor_strategy_plan.md",
        "event_study": AGENT_ROOT / "event_studies/latest_event_study.md",
        "memory_guided_hypotheses": AGENT_ROOT / "experiments/memory_guided_hypothesis_ledger.md",
        "memory_guided_strategy_ledger": AGENT_ROOT / "experiments/memory_guided_strategy_ledger.md",
        "source_discovery": AGENT_ROOT / "source_discovery/latest_source_discovery.md",
    }
    for name, path in paths.items():
        if path.exists():
            add(checks, name, "ok", rel(path))
        else:
            add(checks, name, "warn", f"Not generated yet: {rel(path)}")


def check_git_cleanliness(checks: list[Check]) -> None:
    completed = subprocess.run(
        ["git", "status", "--short", "--", "tools/strategy_research_agent", "docs/personal-live-trading-checklist.md"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        add(checks, "git_scope", "warn", completed.stdout.strip() or "Could not inspect git status.")
        return
    dirty_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if dirty_lines:
        add(checks, "git_scope", "warn", f"{len(dirty_lines)} versioned research files have local changes.")
    else:
        add(checks, "git_scope", "ok", "Versioned research files are clean.")


def main() -> int:
    args = parse_args()
    checks: list[Check] = []
    check_python(checks)
    config = check_agent_config(checks)
    check_fixed_risk_policy(checks, config)
    check_strategy_leverage_overrides(checks)
    check_workflow_gate(checks)
    registry = check_registry(checks)
    check_data(checks, registry)
    check_outputs(checks)
    check_git_cleanliness(checks)

    counts = {status: sum(1 for item in checks if item.status == status) for status in ["ok", "warn", "fail"]}
    payload = {
        "status": "fail" if counts["fail"] or (args.strict and counts["warn"]) else "ok",
        "counts": counts,
        "checks": [item.__dict__ for item in checks],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("Strategy Research Agent preflight")
        print(f"Status: {payload['status']}  ok={counts['ok']} warn={counts['warn']} fail={counts['fail']}")
        for item in checks:
            print(f"[{item.status.upper():4}] {item.name}: {item.detail}")

    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
