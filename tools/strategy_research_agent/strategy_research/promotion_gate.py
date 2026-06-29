#!/usr/bin/env python3
"""Evaluate research strategies for promotion readiness.

The gate is intentionally conservative. Passing it does not start dry-run or
live trading; it only records that the strategy has enough evidence for manual
review before a dry-run decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
CANDIDATE_DIR = AGENT_ROOT / "candidates"
WATCHLIST_DIR = AGENT_ROOT / "watchlist"
PROMOTION_DIR = AGENT_ROOT / "promotion_candidates"
BLOCK_DIR = AGENT_ROOT / "promotion_blocks"
REPORT_DIR = AGENT_ROOT / "promotion_reports"
LATEST_REPORT_JSON = REPORT_DIR / "latest_promotion_report.json"
LATEST_REPORT_MD = REPORT_DIR / "latest_promotion_report.md"


@dataclass
class GateVerdict:
    strategy: str
    verdict: str
    ready_for_manual_dryrun_review: bool
    blocks: list[str]
    evidence: dict[str, Any]
    next_actions: list[str]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_pool() -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for directory in [CANDIDATE_DIR, WATCHLIST_DIR]:
        for path in sorted(directory.glob("*.json")):
            payload = load_json(path)
            if not payload:
                continue
            strategy = payload.get("strategy") or payload.get("name")
            if not strategy:
                continue
            payload["_pool"] = directory.name
            payload["_path"] = rel(path)
            pool[strategy] = payload
    return pool


def index_by(items: list[dict[str, Any]], key: str = "strategy") -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items if item.get(key)}


def latest_scorecards() -> dict[str, dict[str, Any]]:
    payload = load_json(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json") or {}
    return index_by(payload.get("scorecards", []))


def latest_walk_forward() -> dict[str, dict[str, Any]]:
    payload = load_json(AGENT_ROOT / "walk_forward_summaries/latest_walk_forward_summary.json") or {}
    return index_by(payload.get("strategy_summary", []))


def latest_matrix() -> dict[str, dict[str, Any]]:
    payload = load_json(AGENT_ROOT / "matrix_summaries/latest_matrix_summary.json") or {}
    return index_by(payload.get("strategy_summary", []))


def latest_costs() -> dict[str, dict[str, Any]]:
    payload = load_json(AGENT_ROOT / "cost_adjustments/latest_trade_cost_estimate.json") or {}
    return index_by(payload.get("estimates", []))


def next_actions(blocks: list[str]) -> list[str]:
    actions = []
    if "not_in_candidate_pool" in blocks:
        actions.append("先通过 smoke/full-sample gates 进入 research_candidate，再考虑晋级。")
    if "scorecard_not_promotable" in blocks:
        actions.append("提高评分卡证据，补齐收益、PF、回撤、成本和矩阵韧性。")
    if "walk_forward_not_passed" in blocks:
        actions.append("继续重构策略，直到跨固定窗口稳定，而不是单窗口好看。")
    if "matrix_not_robust" in blocks:
        actions.append("按市场状态拆分或收紧规则，减少 regime/cost 矩阵脆弱性。")
    if "negative_after_cost" in blocks or "cost_evidence_missing" in blocks:
        actions.append("导出交易明细并纳入 funding、手续费和滑点校正。")
    if "bias_checks_missing" in blocks:
        actions.append("运行 recursive-analysis 与 lookahead-analysis 后再进入 dry-run 评审。")
    if "too_few_trades" in blocks:
        actions.append("扩大样本内有效交易数，避免统计上不可评估。")
    if not actions:
        actions.append("进入人工 dry-run 评审，不自动改配置、不自动实盘。")
    return actions[:5]


def evaluate_strategy(
    strategy: str,
    pool_item: dict[str, Any],
    scorecard: dict[str, Any] | None,
    walk_forward: dict[str, Any] | None,
    matrix: dict[str, Any] | None,
    costs: dict[str, Any] | None,
) -> GateVerdict:
    blocks: list[str] = []
    evidence: dict[str, Any] = {
        "pool": pool_item.get("_pool"),
        "pool_path": pool_item.get("_path"),
        "classification": pool_item.get("classification"),
        "trades": pool_item.get("trades"),
        "total_profit_pct": pool_item.get("total_profit_pct"),
        "profit_factor": pool_item.get("profit_factor"),
        "max_drawdown_pct": pool_item.get("max_drawdown_pct"),
        "recursive_analysis": pool_item.get("recursive_analysis"),
        "lookahead_analysis": pool_item.get("lookahead_analysis"),
        "scorecard": scorecard,
        "walk_forward": walk_forward,
        "matrix": matrix,
        "costs": costs,
    }

    if pool_item.get("_pool") != "candidates" or pool_item.get("classification") not in {"research_candidate", "dryrun_candidate"}:
        blocks.append("not_in_candidate_pool")
    if int(pool_item.get("trades") or 0) < 100:
        blocks.append("too_few_trades")
    if not scorecard or scorecard.get("tier") != "promotable_research_candidate":
        blocks.append("scorecard_not_promotable")
    if not walk_forward or walk_forward.get("verdict") != "walk_forward_candidate":
        blocks.append("walk_forward_not_passed")
    if not matrix or matrix.get("verdict") != "robust_candidate":
        blocks.append("matrix_not_robust")
    if not costs:
        blocks.append("cost_evidence_missing")
    elif numeric(costs.get("adjusted_profit_pct")) is None or numeric(costs.get("adjusted_profit_pct")) <= 0:
        blocks.append("negative_after_cost")
    recursive = pool_item.get("recursive_analysis") or {}
    lookahead = pool_item.get("lookahead_analysis") or {}
    if recursive.get("status") != "ok" or lookahead.get("status") != "ok":
        blocks.append("bias_checks_missing")

    if blocks:
        verdict = "blocked"
        ready = False
    else:
        verdict = "ready_for_manual_dryrun_review"
        ready = True
    return GateVerdict(
        strategy=strategy,
        verdict=verdict,
        ready_for_manual_dryrun_review=ready,
        blocks=blocks,
        evidence=evidence,
        next_actions=next_actions(blocks),
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Promotion Gate Report",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Ready: `{payload['ready_count']}`",
        f"- Blocked: `{payload['blocked_count']}`",
        "",
        "## Verdicts",
        "",
        "| Strategy | Verdict | Ready | Blocks | Next Actions |",
        "|---|---|---:|---|---|",
    ]
    for item in payload["verdicts"]:
        lines.append(
            "| {strategy} | {verdict} | {ready} | {blocks} | {actions} |".format(
                strategy=item["strategy"],
                verdict=item["verdict"],
                ready=item["ready_for_manual_dryrun_review"],
                blocks=", ".join(item.get("blocks", [])),
                actions="; ".join(item.get("next_actions", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Promotion Policy",
            "",
            "- Passing this gate does not start dry-run or live trading.",
            "- A strategy must pass scorecard, walk-forward, matrix, cost, and bias gates.",
            "- API keys, live config changes, and Freqtrade startup remain manual approval steps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload() -> dict[str, Any]:
    pool = load_pool()
    scorecards = latest_scorecards()
    walk_forward = latest_walk_forward()
    matrix = latest_matrix()
    costs = latest_costs()
    verdicts = [
        evaluate_strategy(
            strategy,
            item,
            scorecards.get(strategy),
            walk_forward.get(strategy),
            matrix.get(strategy),
            costs.get(strategy),
        )
        for strategy, item in sorted(pool.items())
    ]
    ready = [item for item in verdicts if item.ready_for_manual_dryrun_review]
    blocked = [item for item in verdicts if not item.ready_for_manual_dryrun_review]
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "ready_count": len(ready),
        "blocked_count": len(blocked),
        "verdicts": [item.__dict__ for item in verdicts],
    }


def write_outputs(payload: dict[str, Any]) -> None:
    PROMOTION_DIR.mkdir(parents=True, exist_ok=True)
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for old in PROMOTION_DIR.glob("*.json"):
        old.unlink()
    for old in BLOCK_DIR.glob("*.json"):
        old.unlink()
    for item in payload["verdicts"]:
        target_dir = PROMOTION_DIR if item["ready_for_manual_dryrun_review"] else BLOCK_DIR
        (target_dir / f"{item['strategy']}.json").write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")

    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"promotion_report_{timestamp}.json"
    md_path = REPORT_DIR / f"promotion_report_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LATEST_REPORT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Ready: {payload['ready_count']}; Blocked: {payload['blocked_count']}")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)


if __name__ == "__main__":
    main()
