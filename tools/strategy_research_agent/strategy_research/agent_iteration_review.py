#!/usr/bin/env python3
"""Review one research iteration and turn agent weaknesses into upgrade work.

This module makes the strategy agent inspect its own research behavior after an
experiment run.  It does not promote strategies or trade; it records what the
agent tried, what improved, where the research process is weak, and which
agent-upgrade items should be handled next.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_DIR = AGENT_ROOT / "agent_iterations"
LATEST_REVIEW_JSON = REPORT_DIR / "latest_iteration_review.json"
LATEST_REVIEW_MD = REPORT_DIR / "latest_iteration_review.md"
IMPROVEMENT_QUEUE_JSON = REPORT_DIR / "improvement_queue.json"
IMPROVEMENT_QUEUE_MD = REPORT_DIR / "improvement_queue.md"


@dataclass
class AgentIssue:
    issue_id: str
    priority: int
    status: str
    diagnosis: str
    evidence: list[str]
    proposed_upgrade: str
    next_action: str
    success_gate: str


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
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit else rows


def load_pool(name: str) -> list[dict[str, Any]]:
    directory = AGENT_ROOT / name
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        item = load_json(path)
        if item:
            item["_path"] = rel(path)
            rows.append(item)
    return rows


def latest_agent_report() -> tuple[dict[str, Any], str | None]:
    index = load_json(AGENT_ROOT / "reports/agent_report_index.json")
    latest = index.get("latest_report") or index.get("latest_dashboard_refresh") or {}
    report_path = latest.get("path")
    if not report_path:
        return {}, None
    path = REPO_ROOT / report_path
    return load_json(path), report_path


def num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_results(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", [])
    classifications: dict[str, int] = {}
    for item in results:
        classification = item.get("classification", "unknown")
        classifications[classification] = classifications.get(classification, 0) + 1
    unique_strategies = sorted({item.get("strategy", "") for item in results if item.get("strategy")})
    grouped: dict[str, list[dict[str, Any]]] = {}
    strategy_groups = report.get("experiment", {}).get("strategy_groups", {})
    for group, names in strategy_groups.items():
        name_set = set(names)
        grouped[group] = [item for item in results if item.get("strategy") in name_set]
    autonomous_registry = load_json(AGENT_ROOT / "experiments/autonomous_strategy_registry.json")
    seed_followup_names = {
        item.get("name")
        for item in autonomous_registry.get("strategies", [])
        if item.get("source_type") == "seed_followup"
    }
    anti_edge_names = {
        item.get("name")
        for item in autonomous_registry.get("strategies", [])
        if item.get("source_type") == "anti_edge_followup"
    }
    if seed_followup_names:
        grouped["seed_followup"] = [item for item in results if item.get("strategy") in seed_followup_names]
    if anti_edge_names:
        grouped["anti_edge_followup"] = [item for item in results if item.get("strategy") in anti_edge_names]
    best = sorted(
        results,
        key=lambda item: (num(item.get("total_profit_pct")), num(item.get("profit_factor"))),
        reverse=True,
    )[:5]
    return {
        "result_count": len(results),
        "unique_strategy_count": len(unique_strategies),
        "classifications": classifications,
        "groups": {
            name: {
                "count": len(items),
                "rejected": sum(1 for item in items if item.get("classification") == "rejected"),
                "too_few_trades": sum(1 for item in items if "too few trades" in ";".join(item.get("reasons", []))),
                "negative_expectancy": sum(
                    1
                    for item in items
                    if num(item.get("total_profit_pct")) < 0 or num(item.get("profit_factor")) < 1
                ),
                "max_trades": max((num(item.get("trades")) for item in items), default=0),
            }
            for name, items in grouped.items()
        },
        "best_results": [
            {
                "strategy": item.get("strategy"),
                "classification": item.get("classification"),
                "total_profit_pct": item.get("total_profit_pct"),
                "profit_factor": item.get("profit_factor"),
                "max_drawdown_pct": item.get("max_drawdown_pct"),
                "trades": item.get("trades"),
            }
            for item in best
        ],
    }


def build_issues(
    report: dict[str, Any],
    pools: dict[str, list[dict[str, Any]]],
    mature_queue: dict[str, Any],
    execution_history: list[dict[str, Any]],
    walk_forward: dict[str, Any],
    promotion: dict[str, Any],
) -> list[AgentIssue]:
    summary = summarize_results(report)
    result_count = summary["result_count"]
    rejected_count = summary["classifications"].get("rejected", 0)
    autonomous_group = summary.get("groups", {}).get("autonomous_seed", {})
    followup_group = summary.get("groups", {}).get("seed_followup", {})
    anti_edge_group = summary.get("groups", {}).get("anti_edge_followup", {})
    retired_seed = load_json(AGENT_ROOT / "experiments/retired_seed_family_ledger.json")
    candidate_count = len(pools["candidate"]) + len(pools["watchlist"])
    safe_queue_count = num(mature_queue.get("safe_count"))
    cooldown_skips = [
        item
        for item in mature_queue.get("queue", [])
        if item.get("skip_reason") or num(item.get("attempts_24h")) > 0
    ]
    history_keys = [item.get("key") for item in execution_history if item.get("key")]
    duplicate_history = len(history_keys) - len(set(history_keys))
    issues: list[AgentIssue] = []

    if (
        result_count
        and rejected_count == result_count
        and autonomous_group
        and autonomous_group.get("count", 0) > 0
        and not followup_group
        and not retired_seed.get("retired")
    ):
        issues.append(
            AgentIssue(
                issue_id="seed_family_underpowered",
                priority=106,
                status="open",
                diagnosis="自主 seed 策略族已经加入研究闭环，但本轮 seed 仍全部被拒，问题从缺少探索转为 seed 家族本身样本/edge 不足。",
                evidence=[
                    f"autonomous_seed_count={autonomous_group.get('count')}",
                    f"autonomous_seed_rejected={autonomous_group.get('rejected')}",
                    f"autonomous_seed_too_few_trades={autonomous_group.get('too_few_trades')}",
                    f"autonomous_seed_negative_expectancy={autonomous_group.get('negative_expectancy')}",
                    f"autonomous_seed_max_trades={autonomous_group.get('max_trades')}",
                ],
                proposed_upgrade="增加 seed-family 诊断生成器：对样本不足的 seed 自动放宽单个条件；对高交易数负期望 seed 自动生成反向/禁用/退出改造实验。",
                next_action="实现 seed-family follow-up planner，让 --research-iteration 读取 seed 组结果并生成下一代 targeted seed variants。",
                success_gate="下一轮 seed 组至少出现一个可评估样本量的正 PF 变体，或把高交易数负期望 seed 明确降级并生成反向实验。",
            )
        )

    if (
        result_count
        and retired_seed.get("retired")
        and autonomous_group
        and autonomous_group.get("count", 0) > 0
        and autonomous_group.get("negative_expectancy", 0) == autonomous_group.get("count", 0)
    ):
        issues.append(
            AgentIssue(
                issue_id="context_seed_negative_expectancy",
                priority=110,
                status="open",
                diagnosis="旧 1m OHLCV seed 家族已退休，替代的 context-feature seed 仍为负期望；下一步不应继续堆简单 K 线特征。",
                evidence=[
                    f"retired_family={retired_seed.get('retired_family')}",
                    f"context_seed_count={autonomous_group.get('count')}",
                    f"context_seed_negative_expectancy={autonomous_group.get('negative_expectancy')}",
                    f"context_seed_max_trades={autonomous_group.get('max_trades')}",
                ],
                proposed_upgrade="把研究预算转向非 OHLCV 信息源或更明确的多周期验证，例如 funding/mark/盘口代理特征、BTC->ETH lead-lag、4h/1h context 分层。",
                next_action="实现 context-source scout/planner，生成外部特征或跨资产 lead-lag 的可审计实验，而不是继续扩写 1m OHLCV seed。",
                success_gate="下一轮产生一个非纯 1m OHLCV 的实验族，或明确记录缺少所需外部数据。",
            )
        )

    if (
        result_count
        and followup_group
        and followup_group.get("count", 0) > 0
        and followup_group.get("negative_expectancy", 0) >= max(2, followup_group.get("count", 0) // 2)
        and not anti_edge_group
    ):
        issues.append(
            AgentIssue(
                issue_id="followup_negative_expectancy",
                priority=108,
                status="open",
                diagnosis="seed follow-up 已经补足样本探索，但多数 follow-up 变体仍是负期望，说明问题从样本不足转为信号方向/成本/退出质量问题。",
                evidence=[
                    f"seed_followup_count={followup_group.get('count')}",
                    f"seed_followup_negative_expectancy={followup_group.get('negative_expectancy')}",
                    f"seed_followup_too_few_trades={followup_group.get('too_few_trades')}",
                    f"seed_followup_max_trades={followup_group.get('max_trades')}",
                ],
                proposed_upgrade="增加 anti-edge 研究分支：对负期望 follow-up 自动生成反向、禁用方向、退出缩短和成本压力对照实验。",
                next_action="在 autonomous_strategy_lab 中为高样本负期望 follow-up 增加 inverse/cost-aware exit variants。",
                success_gate="下一轮至少明确一个 follow-up 家族是可反向利用、可通过退出改善，或应被降级为失败模式。",
            )
        )

    if (
        result_count
        and anti_edge_group
        and anti_edge_group.get("count", 0) > 0
        and anti_edge_group.get("negative_expectancy", 0) == anti_edge_group.get("count", 0)
    ):
        issues.append(
            AgentIssue(
                issue_id="anti_edge_family_failed",
                priority=112,
                status="open",
                diagnosis="base seed、sample-floor、anti-edge 三层 1m OHLCV 变体均失败，简单 K 线方向/反向/快退出研究已经进入低收益区域。",
                evidence=[
                    f"anti_edge_count={anti_edge_group.get('count')}",
                    f"anti_edge_negative_expectancy={anti_edge_group.get('negative_expectancy')}",
                    f"anti_edge_max_trades={anti_edge_group.get('max_trades')}",
                    f"autonomous_seed_negative_expectancy={autonomous_group.get('negative_expectancy')}",
                ],
                proposed_upgrade="加入策略族降级机制：把失败的 1m OHLCV seed 家族标记为 retired，并把下一轮研究预算转向新信息源、跨周期特征或非 OHLCV 微结构特征。",
                next_action="实现 retired_family ledger，让 autonomous generator 跳过已证伪的 seed 家族，并生成 higher-timeframe/context-feature 实验。",
                success_gate="下一轮不再继续扩写同一批 1m OHLCV seed，而是生成可审计的新研究族或明确需要外部特征数据。",
            )
        )

    if result_count and rejected_count == result_count and not autonomous_group:
        issues.append(
            AgentIssue(
                issue_id="no_candidate_yield",
                priority=100,
                status="open",
                diagnosis="本轮策略研究没有产生新的可保留候选，Agent 的假设质量或筛选方向需要升级。",
                evidence=[
                    f"results={result_count}",
                    f"rejected={rejected_count}",
                    f"candidate_plus_watchlist_pool={candidate_count}",
                ],
                proposed_upgrade="增加更强的新策略族生成器，并要求每轮至少覆盖趋势、均值回归、短动量、成本压力四类不同假设。",
                next_action="实现或刷新一个 strategy-family generator，再跑 --research-iteration 验证候选产出是否改善。",
                success_gate="下一轮至少产生 1 个 watchlist/research_candidate，或明确证明全部新族在成本压力下失败。",
            )
        )

    if result_count and autonomous_group and autonomous_group.get("too_few_trades", 0) >= max(3, autonomous_group.get("count", 0) // 2):
        issues.append(
            AgentIssue(
                issue_id="sample_floor_failure",
                priority=94,
                status="open",
                diagnosis="多数 seed 策略仍因交易太少被拒，Agent 需要把样本量作为策略生成阶段的一等目标。",
                evidence=[
                    f"autonomous_seed_too_few_trades={autonomous_group.get('too_few_trades')}",
                    f"autonomous_seed_count={autonomous_group.get('count')}",
                ],
                proposed_upgrade="为 seed 生成器加入 sample-floor variants：每个稀疏 seed 自动生成一个只放宽单一入场条件的版本，并保留原始版本作对照。",
                next_action="在 autonomous_strategy_lab 中增加 sample-floor blueprint 或 follow-up variant generator。",
                success_gate="下一轮至少一半 seed 变体达到最低可评估交易数。",
            )
        )

    if summary["unique_strategy_count"] and summary["unique_strategy_count"] <= max(2, result_count // 8):
        issues.append(
            AgentIssue(
                issue_id="low_experiment_diversity",
                priority=92,
                status="open",
                diagnosis="实验多样性偏低，Agent 可能在少数策略上做局部微调，而不是探索新的研究空间。",
                evidence=[
                    f"results={result_count}",
                    f"unique_strategies={summary['unique_strategy_count']}",
                ],
                proposed_upgrade="给研究循环加入 diversity budget，限制同一 root strategy 的连续实验数量，并强制补足未覆盖策略族。",
                next_action="在 agenda/response queue 里加入 root-strategy 配额与探索/利用比例。",
                success_gate="下一轮 unique_strategy_count/result_count 显著提升，且不牺牲完整回测证据。",
            )
        )

    if cooldown_skips or duplicate_history > 0:
        issues.append(
            AgentIssue(
                issue_id="repeated_action_pressure",
                priority=88,
                status="open",
                diagnosis="Agent 仍有重复执行压力，需要继续把执行记忆用于更上游的实验规划。",
                evidence=[
                    f"cooldown_or_recent_queue_items={len(cooldown_skips)}",
                    f"duplicate_history_keys={duplicate_history}",
                    f"safe_queue_count={safe_queue_count}",
                ],
                proposed_upgrade="把 response execution memory 前移到 hypothesis/agenda 生成阶段，生成时就避开冷却中的动作。",
                next_action="让 hypothesis planner 读取 response_execution_history.jsonl 并生成替代实验。",
                success_gate="队列中 cooldown skip 项减少，同时 safe_queue_count 保持为正。",
            )
        )

    if not walk_forward:
        issues.append(
            AgentIssue(
                issue_id="missing_walk_forward_evidence",
                priority=82,
                status="open",
                diagnosis="本轮 review 缺少 walk-forward 证据，候选策略无法判断是否只适配单一窗口。",
                evidence=["walk_forward_summary=missing"],
                proposed_upgrade="把 walk-forward 缺口变成 research-iteration 的默认检查项，候选出现后自动安排验证。",
                next_action="运行 --walk-forward 或让 --research-iteration 在发现候选时自动排队 walk-forward。",
                success_gate="latest_iteration_review 能引用最新 walk-forward summary。",
            )
        )

    if not promotion:
        issues.append(
            AgentIssue(
                issue_id="missing_promotion_gate",
                priority=80,
                status="open",
                diagnosis="本轮 review 缺少 promotion gate 证据，无法严格区分研究候选与 dry-run 候选。",
                evidence=["promotion_report=missing"],
                proposed_upgrade="把 promotion gate 作为研究闭环的强制收尾步骤。",
                next_action="运行 --promotion-gate 或让 --research-iteration 默认刷新 promotion report。",
                success_gate="latest_iteration_review 能引用最新 promotion report 并列出阻塞项。",
            )
        )

    if not issues:
        issues.append(
            AgentIssue(
                issue_id="continue_hypothesis_expansion",
                priority=50,
                status="monitor",
                diagnosis="本轮没有发现明显流程故障，下一步应扩大高质量假设覆盖面。",
                evidence=[
                    f"results={result_count}",
                    f"candidate_plus_watchlist_pool={candidate_count}",
                    f"safe_queue_count={safe_queue_count}",
                ],
                proposed_upgrade="继续增加独立策略族与压力测试，不改变安全边界。",
                next_action="跑下一轮 --research-iteration，并关注候选池质量是否改善。",
                success_gate="候选池质量提高且没有新增 lookahead/recursive/cost gate 风险。",
            )
        )

    return sorted(issues, key=lambda item: (-item.priority, item.issue_id))


def build_payload() -> dict[str, Any]:
    report, report_path = latest_agent_report()
    pools = {
        "candidate": load_pool("candidates"),
        "watchlist": load_pool("watchlist"),
        "rejected": load_pool("rejected"),
    }
    mature_queue = load_json(AGENT_ROOT / "mature_researcher/latest_response_queue.json")
    execution_history = load_jsonl(AGENT_ROOT / "mature_researcher/response_execution_history.jsonl", limit=200)
    walk_forward = load_json(AGENT_ROOT / "walk_forward_summaries/latest_walk_forward_summary.json")
    promotion = load_json(AGENT_ROOT / "promotion_reports/latest_promotion_report.json")
    issues = build_issues(report, pools, mature_queue, execution_history, walk_forward, promotion)
    summary = summarize_results(report)
    return {
        "generated_at_utc": utc_stamp(),
        "source_report": report_path,
        "strategy_results": summary,
        "pool_sizes": {name: len(items) for name, items in pools.items()},
        "mature_queue": {
            "queue_count": mature_queue.get("queue_count"),
            "safe_count": mature_queue.get("safe_count"),
            "cooldown_hours": mature_queue.get("cooldown_hours"),
        },
        "execution_history_rows_reviewed": len(execution_history),
        "walk_forward_source": rel(AGENT_ROOT / "walk_forward_summaries/latest_walk_forward_summary.json")
        if walk_forward
        else None,
        "promotion_source": rel(AGENT_ROOT / "promotion_reports/latest_promotion_report.json") if promotion else None,
        "agent_issues": [asdict(item) for item in issues],
        "improvement_queue": {
            "generated_at_utc": utc_stamp(),
            "open_count": sum(1 for item in issues if item.status == "open"),
            "items": [asdict(item) for item in issues],
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    results = payload["strategy_results"]
    lines = [
        "# Agent Research Iteration Review",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source report: `{payload.get('source_report')}`",
        f"- Results reviewed: `{results['result_count']}`",
        f"- Unique strategies: `{results['unique_strategy_count']}`",
        f"- Pool sizes: `candidate={payload['pool_sizes']['candidate']}`, `watchlist={payload['pool_sizes']['watchlist']}`, `rejected={payload['pool_sizes']['rejected']}`",
        f"- Mature queue: `queue={payload['mature_queue'].get('queue_count')}`, `safe={payload['mature_queue'].get('safe_count')}`, `cooldown_h={payload['mature_queue'].get('cooldown_hours')}`",
        f"- Execution history rows reviewed: `{payload['execution_history_rows_reviewed']}`",
        "",
        "## Best Strategy Results",
        "",
        "| Strategy | Class | Return % | PF | DD % | Trades |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in results["best_results"]:
        lines.append(
            "| {strategy} | {classification} | {total_profit_pct} | {profit_factor} | {max_drawdown_pct} | {trades} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Agent Issues",
            "",
            "| Priority | Issue | Status | Diagnosis | Proposed Upgrade | Success Gate |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for item in payload["agent_issues"]:
        lines.append(
            "| {priority} | {issue_id} | {status} | {diagnosis} | {proposed_upgrade} | {success_gate} |".format(
                **item
            )
        )
    lines.extend(["", "## Next Actions", ""])
    for item in payload["agent_issues"]:
        evidence = "; ".join(item.get("evidence", []))
        lines.append(f"- `{item['issue_id']}`: {item['next_action']} Evidence: {evidence}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = payload["generated_at_utc"]
    review_json = REPORT_DIR / f"iteration_review_{stamp}.json"
    review_md = REPORT_DIR / f"iteration_review_{stamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    review_json.write_text(json_text, encoding="utf-8")
    LATEST_REVIEW_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(review_md, payload)
    LATEST_REVIEW_MD.write_text(review_md.read_text(encoding="utf-8"), encoding="utf-8")

    queue_text = json.dumps(payload["improvement_queue"], indent=2, ensure_ascii=False)
    IMPROVEMENT_QUEUE_JSON.write_text(queue_text, encoding="utf-8")
    write_markdown(IMPROVEMENT_QUEUE_MD, payload)
    print(f"Wrote {rel(review_json)}")
    print(f"Wrote {rel(review_md)}")
    print(f"Wrote {rel(LATEST_REVIEW_JSON)}")
    print(f"Wrote {rel(LATEST_REVIEW_MD)}")
    print(f"Wrote {rel(IMPROVEMENT_QUEUE_JSON)}")
    print(f"Wrote {rel(IMPROVEMENT_QUEUE_MD)}")


def main() -> None:
    write_outputs(build_payload())


if __name__ == "__main__":
    main()
