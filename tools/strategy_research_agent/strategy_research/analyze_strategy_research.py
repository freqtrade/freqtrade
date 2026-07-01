#!/usr/bin/env python3
"""Build strategy scorecards and failure diagnostics from local research artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_DIR = AGENT_ROOT / "strategy_assessments"
POOL_DIRS = [
    AGENT_ROOT / "candidates",
    AGENT_ROOT / "watchlist",
    AGENT_ROOT / "rejected",
]


@dataclass
class Scorecard:
    strategy: str
    tier: str
    score: int
    base_return_pct: float | None
    adjusted_return_pct: float | None
    market_change_pct: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    trades: int | None
    matrix_verdict: str | None
    positive_matrix_runs: int | None
    stress_negative_runs: int | None
    too_few_trade_runs: int | None
    primary_failures: list[str]
    next_actions: list[str]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def latest_report_path() -> Path | None:
    index = load_json(AGENT_ROOT / "reports/agent_report_index.json")
    if not index:
        return None
    latest = index.get("latest_report") or index.get("latest_dashboard_refresh")
    if not latest:
        return None
    return REPO_ROOT / latest["path"]


def load_pool_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for directory in POOL_DIRS:
        for path in directory.glob("*.json"):
            item = load_json(path)
            if not item:
                continue
            strategy = item.get("strategy") or item.get("name")
            if not strategy:
                continue
            current = metrics.setdefault(strategy, {})
            current.update(item)
            current["pool"] = directory.name
            current["pool_path"] = rel_path(path)
    return metrics


def index_by_strategy(items: list[dict[str, Any]], key: str = "strategy") -> dict[str, dict[str, Any]]:
    return {item[key]: item for item in items if item.get(key)}


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_points(score: int, failures: list[str], condition: bool, points: int, failure: str) -> int:
    if condition:
        return score + points
    failures.append(failure)
    return score


def tier(score: int, failures: list[str]) -> str:
    hard_failures = {
        "negative_after_cost",
        "fragile_matrix",
        "too_few_matrix_trades",
        "stress_cost_failure",
        "lookahead_or_recursive_unverified",
    }
    if score >= 80 and not hard_failures.intersection(failures):
        return "promotable_research_candidate"
    if score >= 60:
        return "watchlist"
    if score >= 40:
        return "needs_redesign"
    return "reject_or_archive"


def next_actions_for(failures: list[str]) -> list[str]:
    actions: list[str] = []
    if "too_few_matrix_trades" in failures:
        actions.append("放宽或重构入场确认，让 90 天窗口内交易数足够评估。")
    if "stress_cost_failure" in failures or "negative_after_cost" in failures:
        actions.append("提高单笔期望值，减少低边际交易；优先测试更严格的入场质量而不是提高杠杆。")
    if "cost_not_estimated" in failures:
        actions.append("先导出交易明细并运行 funding/滑点成本校正，再决定是否继续研究。")
    if "matrix_not_tested" in failures:
        actions.append("先纳入 bull/bear/range/high-vol 与 base/stress fee 矩阵，补齐韧性证据。")
    if "underperforms_market" in failures:
        actions.append("加入基准超额收益门槛，避免只得到低波动但跑输持有的策略。")
    if "weak_profit_factor" in failures:
        actions.append("分解进出场标签，检查亏损主要来自止损、信号失效还是震荡误入场。")
    if "fragile_matrix" in failures:
        actions.append("按 bull/bear/range/high-vol 分别建模，不要用同一套规则硬吃所有状态。")
    if "lookahead_or_recursive_unverified" in failures:
        actions.append("对进入 dry-run 候选前的版本运行 recursive-analysis 和 lookahead-analysis。")
    if not actions:
        actions.append("进入更长 dry-run，对比真实成交、滑点和回测差异。")
    return actions[:4]


def score_strategy(
    strategy: str,
    base: dict[str, Any],
    matrix: dict[str, Any] | None,
    costs: dict[str, Any] | None,
) -> Scorecard:
    failures: list[str] = []
    score = 0

    base_return = numeric(base.get("total_profit_pct"))
    adjusted_return = numeric(costs.get("adjusted_profit_pct")) if costs else None
    market_change = numeric(base.get("market_change_pct"))
    profit_factor = numeric(base.get("profit_factor"))
    max_drawdown = numeric(base.get("max_drawdown_pct"))
    trades = int(base["trades"]) if base.get("trades") is not None else None

    matrix_verdict = matrix.get("verdict") if matrix else None
    positive_runs = int(matrix["positive_runs"]) if matrix and matrix.get("positive_runs") is not None else None
    stress_negative = int(matrix["stress_negative_runs"]) if matrix and matrix.get("stress_negative_runs") is not None else None
    too_few = int(matrix["too_few_trade_runs"]) if matrix and matrix.get("too_few_trade_runs") is not None else None

    score = add_points(score, failures, base_return is not None and base_return > 0, 12, "negative_or_missing_return")
    score = add_points(score, failures, profit_factor is not None and profit_factor >= 1.2, 12, "weak_profit_factor")
    score = add_points(score, failures, max_drawdown is not None and max_drawdown <= 5, 10, "drawdown_too_high")
    score = add_points(score, failures, trades is not None and trades >= 50, 10, "too_few_full_sample_trades")
    score = add_points(
        score,
        failures,
        market_change is not None and base_return is not None and base_return >= market_change,
        10,
        "underperforms_market",
    )
    if costs:
        score = add_points(
            score,
            failures,
            adjusted_return is not None and adjusted_return > 0,
            12,
            "negative_after_cost",
        )
    else:
        failures.append("cost_not_estimated")

    if matrix:
        score = add_points(
            score,
            failures,
            matrix_verdict in {"watchlist", "robust_candidate"},
            12,
            "fragile_matrix",
        )
        score = add_points(
            score,
            failures,
            too_few is not None and too_few == 0,
            8,
            "too_few_matrix_trades",
        )
        score = add_points(
            score,
            failures,
            stress_negative is not None and stress_negative == 0,
            8,
            "stress_cost_failure",
        )
    else:
        failures.append("matrix_not_tested")
    score = add_points(
        score,
        failures,
        bool(base.get("recursive_analysis")) and bool(base.get("lookahead_analysis")),
        6,
        "lookahead_or_recursive_unverified",
    )

    return Scorecard(
        strategy=strategy,
        tier=tier(score, failures),
        score=score,
        base_return_pct=base_return,
        adjusted_return_pct=adjusted_return,
        market_change_pct=market_change,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown,
        trades=trades,
        matrix_verdict=matrix_verdict,
        positive_matrix_runs=positive_runs,
        stress_negative_runs=stress_negative,
        too_few_trade_runs=too_few,
        primary_failures=failures,
        next_actions=next_actions_for(failures),
    )


def build_payload() -> dict[str, Any]:
    latest_report = load_json(latest_report_path()) if latest_report_path() else {}
    pool_metrics = load_pool_metrics()
    matrix_payload = load_json(AGENT_ROOT / "matrix_summaries/latest_matrix_summary.json") or {}
    cost_payload = load_json(AGENT_ROOT / "cost_adjustments/latest_trade_cost_estimate.json") or {}

    matrix_by_strategy = index_by_strategy(matrix_payload.get("strategy_summary", []))
    costs_by_strategy = index_by_strategy(cost_payload.get("estimates", []))

    for item in latest_report.get("candidate_pool", []) + latest_report.get("watchlist_pool", []) + latest_report.get("rejected_pool", []):
        strategy = item.get("strategy") or item.get("name")
        if strategy:
            pool_metrics.setdefault(strategy, {}).update(item)

    scorecards = [
        asdict(score_strategy(strategy, base, matrix_by_strategy.get(strategy), costs_by_strategy.get(strategy)))
        for strategy, base in sorted(pool_metrics.items())
    ]

    failure_counts = Counter()
    for card in scorecards:
        failure_counts.update(card["primary_failures"])

    diagnostics_by_strategy = []
    matrix_rows = defaultdict(list)
    for row in matrix_payload.get("rows", []):
        if row.get("strategy"):
            matrix_rows[row["strategy"]].append(row)
    for card in scorecards:
        strategy = card["strategy"]
        rows = matrix_rows.get(strategy, [])
        worst_rows = sorted(
            rows,
            key=lambda item: (
                numeric(item.get("return_pct")) if numeric(item.get("return_pct")) is not None else 999,
                numeric(item.get("profit_factor")) if numeric(item.get("profit_factor")) is not None else 999,
            ),
        )[:3]
        diagnostics_by_strategy.append(
            {
                "strategy": strategy,
                "tier": card["tier"],
                "score": card["score"],
                "primary_failures": card["primary_failures"],
                "next_actions": card["next_actions"],
                "worst_matrix_rows": worst_rows,
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "market_profile": "crypto / Binance USDT-M futures / BTC-ETH only",
        "score_definition": {
            "max_score": 100,
            "dimensions": [
                "positive return",
                "profit factor",
                "drawdown",
                "trade count",
                "market outperformance",
                "cost-adjusted return",
                "matrix robustness",
                "matrix trade sufficiency",
                "stress-cost survival",
                "bias-analysis verification",
            ],
        },
        "scorecards": sorted(scorecards, key=lambda item: (-item["score"], item["strategy"])),
        "failure_summary": [
            {"failure": failure, "count": count}
            for failure, count in failure_counts.most_common()
        ],
        "diagnostics": diagnostics_by_strategy,
        "source_artifacts": {
            "latest_report": rel_path(latest_report_path()) if latest_report_path() else None,
            "matrix_summary": rel_path(AGENT_ROOT / "matrix_summaries/latest_matrix_summary.json"),
            "trade_cost_estimate": rel_path(AGENT_ROOT / "cost_adjustments/latest_trade_cost_estimate.json"),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strategy Scorecards and Failure Diagnostics",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Market profile: `{payload['market_profile']}`",
        "",
        "## Scorecards",
        "",
        "| Strategy | Tier | Score | Base % | Adjusted % | Market % | PF | DD % | Trades | Matrix | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in payload["scorecards"]:
        lines.append(
            "| {strategy} | {tier} | {score} | {base_return_pct} | {adjusted_return_pct} | {market_change_pct} | {profit_factor} | {max_drawdown_pct} | {trades} | {matrix_verdict} | {failures} |".format(
                failures=", ".join(item["primary_failures"]),
                **item,
            )
        )
    lines.extend(["", "## Failure Summary", "", "| Failure | Count |", "|---|---:|"])
    for item in payload["failure_summary"]:
        lines.append("| {failure} | {count} |".format(**item))
    lines.extend(["", "## Next Actions", ""])
    for item in payload["diagnostics"]:
        lines.append(f"### {item['strategy']}")
        for action in item["next_actions"]:
            lines.append(f"- {action}")
        if item["worst_matrix_rows"]:
            lines.append("- Worst matrix rows:")
            for row in item["worst_matrix_rows"]:
                lines.append(
                    "  - {experiment}/{regime}: return {return_pct}%, PF {profit_factor}, trades {trades}, reasons {reasons}".format(
                        experiment=row.get("experiment"),
                        regime=row.get("regime"),
                        return_pct=row.get("return_pct"),
                        profit_factor=row.get("profit_factor"),
                        trades=row.get("trades"),
                        reasons=", ".join(row.get("reasons", [])),
                    )
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"strategy_assessment_{timestamp}.json"
    md_path = REPORT_DIR / f"strategy_assessment_{timestamp}.md"
    latest_json = REPORT_DIR / "latest_strategy_assessment.json"
    latest_md = REPORT_DIR / "latest_strategy_assessment.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {rel_path(json_path)}")
    print(f"Wrote {rel_path(md_path)}")
    print(f"Wrote {rel_path(latest_json)}")
    print(f"Wrote {rel_path(latest_md)}")


if __name__ == "__main__":
    main()
