#!/usr/bin/env python3
"""Convert factor research evidence into guarded strategy hypotheses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
FACTOR_JSON = AGENT_ROOT / "factors/latest_factor_research.json"
OUTPUT_DIR = AGENT_ROOT / "factors"
LATEST_JSON = OUTPUT_DIR / "latest_factor_strategy_plan.json"
LATEST_MD = OUTPUT_DIR / "latest_factor_strategy_plan.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> dict[str, Any]:
    factor_report = load_json(FACTOR_JSON)
    candidates = factor_report.get("edge_candidates", [])
    hypotheses = []
    for index, item in enumerate(candidates, start=1):
        hypotheses.append(
            {
                "hypothesis_id": f"factor_edge_{index:03d}",
                "status": "ready_for_event_study",
                "pair": item["pair"],
                "timeframe": item["timeframe"],
                "side": item["side"],
                "factor": item["factor"],
                "source": "factor_research",
                "evidence": {
                    "sample": item["sample"],
                    "mean_after_fee_pct": item["mean_after_fee_pct"],
                    "win_rate": item["win_rate"],
                    "mean_mfe_pct": item["mean_mfe_pct"],
                    "mean_mae_pct": item["mean_mae_pct"],
                },
                "strategy_generation_gate": "Do not generate a concrete strategy until this factor is converted into an event definition and passes event-study validation.",
            }
        )
    return {
        "generated_at_utc": now_utc(),
        "research_only": True,
        "factor_report": rel(FACTOR_JSON) if factor_report else None,
        "hypotheses": hypotheses,
        "summary": {
            "factor_candidates": len(candidates),
            "strategy_hypotheses": len(hypotheses),
            "verdict": "ready_for_event_study" if hypotheses else "no_factor_edge_to_synthesize",
        },
        "blocked_reason": None if hypotheses else "No factor row met sample, after-fee expectancy, and win-rate gates.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Factor-To-Strategy Plan",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Factor report: `{payload.get('factor_report')}`",
        f"- Verdict: `{payload['summary']['verdict']}`",
        "",
        "## Hypotheses",
        "",
        "| ID | Pair | TF | Factor | Side | Sample | Mean After Fee % | Win Rate | Status |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for item in payload["hypotheses"]:
        evidence = item["evidence"]
        lines.append(
            "| {hypothesis_id} | {pair} | {timeframe} | {factor} | {side} | {sample} | {mean_after_fee_pct} | {win_rate} | {status} |".format(
                sample=evidence["sample"],
                mean_after_fee_pct=evidence["mean_after_fee_pct"],
                win_rate=evidence["win_rate"],
                **item,
            )
        )
    if not payload["hypotheses"]:
        lines.append("| none | - | - | - | - | 0 | 0 | 0 | blocked |")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- This is a Strategy Agent sub-flow, not a separate Agent.",
            "- Passing factor rows become event-study hypotheses first, not immediate strategy classes.",
            "- If this plan is blocked, strategy generation should redesign factors or run negative controls instead of generating another class from theory.",
        ]
    )
    if payload.get("blocked_reason"):
        lines.extend(["", f"Blocked reason: {payload['blocked_reason']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    timestamp = payload["generated_at_utc"]
    json_path = OUTPUT_DIR / f"factor_strategy_plan_{timestamp}.json"
    md_path = OUTPUT_DIR / f"factor_strategy_plan_{timestamp}.md"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(text, encoding="utf-8")
    LATEST_JSON.write_text(text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(md_path)}")
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")


if __name__ == "__main__":
    main()
