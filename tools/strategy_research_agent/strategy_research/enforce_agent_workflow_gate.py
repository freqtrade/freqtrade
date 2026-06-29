#!/usr/bin/env python3
"""Enforce the fixed Strategy Agent workflow before research starts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
RUNTIME_RULES = AGENT_ROOT / "consolidation/agent_operating_rules.json"
DEFAULT_RULES = AGENT_ROOT / "consolidation/agent_operating_rules.default.json"
REQUIRED_GATES = [
    "event_study_edge_check",
    "freqtrade_backtesting",
    "recursive_analysis",
    "lookahead_analysis",
    "regime_matrix",
    "fee_slippage_stress",
    "walk_forward_validation",
    "promotion_gate",
]


@dataclass
class GateCheck:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Write machine-readable gate output.")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: list[GateCheck], name: str, status: str, detail: str) -> None:
    checks.append(GateCheck(name=name, status=status, detail=detail))


def resolve_rules_path(checks: list[GateCheck]) -> Path | None:
    if RUNTIME_RULES.exists():
        add(checks, "operating_rules", "ok", rel(RUNTIME_RULES))
        return RUNTIME_RULES
    if DEFAULT_RULES.exists():
        add(checks, "operating_rules", "ok", f"{rel(DEFAULT_RULES)} (versioned default)")
        return DEFAULT_RULES
    add(checks, "operating_rules", "fail", f"Missing {rel(RUNTIME_RULES)} and {rel(DEFAULT_RULES)}")
    return None


def validate_artifact(path_text: str, checks: list[GateCheck]) -> None:
    path = REPO_ROOT / path_text
    if not path.exists():
        add(checks, f"load:{path_text}", "fail", "missing")
        return
    if path.suffix == ".json":
        try:
            payload = read_json(path)
        except json.JSONDecodeError as exc:
            add(checks, f"load:{path_text}", "fail", f"invalid JSON: {exc}")
            return
        generated = payload.get("generated_at_utc") or payload.get("generated_at")
        detail = f"json ok"
        if generated:
            detail += f", generated={generated}"
        add(checks, f"load:{path_text}", "ok", detail)
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        add(checks, f"load:{path_text}", "fail", "empty markdown/text")
        return
    add(checks, f"load:{path_text}", "ok", f"text ok, chars={len(text)}")


def validate_rules(path: Path, checks: list[GateCheck]) -> dict[str, Any]:
    try:
        rules = read_json(path)
    except json.JSONDecodeError as exc:
        add(checks, "operating_rules_json", "fail", f"invalid JSON: {exc}")
        return {}
    if rules.get("research_only") is not True:
        add(checks, "research_only", "fail", "research_only must be true")
    else:
        add(checks, "research_only", "ok", "research-only boundary locked")
    must_load = rules.get("must_load_before_research")
    if not isinstance(must_load, list) or not must_load:
        add(checks, "must_load_before_research", "fail", "missing or empty list")
    else:
        add(checks, "must_load_before_research", "ok", f"{len(must_load)} required artifacts")
        for path_text in must_load:
            validate_artifact(str(path_text), checks)
    gates = set(rules.get("required_gates", []))
    missing_gates = [gate for gate in REQUIRED_GATES if gate not in gates]
    if missing_gates:
        add(checks, "required_gates", "fail", "missing " + ", ".join(missing_gates))
    else:
        add(checks, "required_gates", "ok", ", ".join(REQUIRED_GATES))
    prompt_contract = rules.get("prompt_contract", [])
    has_load_rule = any("Load knowledge graph context" in str(rule) for rule in prompt_contract)
    if has_load_rule:
        add(checks, "prompt_contract", "ok", "knowledge/memory/consolidation load rule present")
    else:
        add(checks, "prompt_contract", "fail", "missing mandatory load-before-strategy contract")
    return rules


def build_payload() -> dict[str, Any]:
    checks: list[GateCheck] = []
    rules_path = resolve_rules_path(checks)
    rules: dict[str, Any] = {}
    if rules_path:
        rules = validate_rules(rules_path, checks)
    failed = [check for check in checks if check.status == "fail"]
    return {
        "status": "fail" if failed else "ok",
        "rules_path": rel(rules_path) if rules_path else None,
        "must_load_before_research": rules.get("must_load_before_research", []),
        "checks": [check.__dict__ for check in checks],
    }


def print_text(payload: dict[str, Any]) -> None:
    print("Strategy Agent Gate")
    print(f"Status: {payload['status']}")
    print(f"Rules: {payload.get('rules_path') or 'missing'}")
    for index, path_text in enumerate(payload.get("must_load_before_research", []), start=1):
        print(f"{index}. loaded {path_text}")
    for check in payload["checks"]:
        print(f"[{check['status'].upper():4}] {check['name']}: {check['detail']}")


def main() -> int:
    args = parse_args()
    payload = build_payload()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_text(payload)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
