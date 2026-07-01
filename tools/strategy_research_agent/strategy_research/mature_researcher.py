#!/usr/bin/env python3
"""Build a senior-researcher decision plan from the latest strategy evidence.

The output is deliberately action-oriented: it tells the local strategy agent
what problem it is seeing, what evidence supports that diagnosis, which safe
experiments to run next, and which promotion gates must stay closed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_DIR = AGENT_ROOT / "mature_researcher"
LATEST_REPORT_JSON = REPORT_DIR / "latest_researcher_decision.json"
LATEST_REPORT_MD = REPORT_DIR / "latest_researcher_decision.md"


@dataclass
class Decision:
    priority: int
    strategy: str
    diagnosis: str
    confidence: str
    evidence: list[str]
    response_plan: list[str]
    next_experiments: list[str]
    next_command: str
    success_gate: str
    promotion_block: str


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latest_agent_report() -> dict[str, Any]:
    index = load_json(AGENT_ROOT / "reports/agent_report_index.json")
    latest = index.get("latest_report") or {}
    path = REPO_ROOT / latest.get("path", "")
    return load_json(path)


def index_by_strategy(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["strategy"]: item for item in items if item.get("strategy")}


def num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_report_result(item: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    trades = num(item.get("trades"))
    daily_trades = num(item.get("daily_trades"))
    profit = num(item.get("total_profit_pct"))
    pf = num(item.get("profit_factor"))
    dd = num(item.get("max_drawdown_pct"))
    long_profit = num(item.get("long_profit_pct"))
    short_profit = num(item.get("short_profit_pct"))
    fee = num(item.get("fee"), 0.0005)

    if trades >= 300 and daily_trades >= 10 and profit < 0 and pf < 0.8:
        tags.append("high_frequency_negative_expectancy")
    if trades >= 300 and pf < 0.7 and long_profit < 0 and short_profit < 0:
        tags.append("two_sided_signal_failure")
    if trades < 50:
        tags.append("insufficient_sample")
    if profit < 0 and pf < 1:
        tags.append("negative_expectancy")
    if dd >= 20:
        tags.append("excessive_drawdown")
    if fee >= 0.001:
        tags.append("stress_cost_sensitive")
    if item.get("classification") == "research_candidate" and profit > 0 and pf >= 1:
        tags.append("candidate_needs_robustness")
    return tags


def behavior_tags(behavior: dict[str, Any] | None) -> list[str]:
    if not behavior:
        return []
    tags: list[str] = []
    win_rate = num(behavior.get("win_rate_pct"))
    total_profit = num(behavior.get("total_profit_abs"))
    pf = num(behavior.get("profit_factor"))
    avg_mfe = behavior.get("avg_mfe_pct")
    avg_mae = behavior.get("avg_mae_pct")
    if win_rate >= 55 and total_profit < 0:
        tags.append("payoff_asymmetry")
    if pf and pf < 0.8:
        tags.append("poor_trade_expectancy")
    if avg_mfe is not None and avg_mae is not None and num(avg_mae) > num(avg_mfe):
        tags.append("bad_entry_timing")
    if num(behavior.get("max_consecutive_losses")) >= 5:
        tags.append("loss_cluster")
    if num(behavior.get("stop_loss_trades")) > 0 and num(behavior.get("stop_loss_profit_abs")) < 0:
        tags.append("stop_loss_damage")
    return tags


def failure_tags(failure: dict[str, Any] | None) -> list[str]:
    if not failure:
        return []
    return [item.get("mode", "") for item in failure.get("failure_modes", []) if item.get("mode")]


def build_decision(
    result: dict[str, Any],
    behavior: dict[str, Any] | None,
    attribution: dict[str, Any] | None,
    scorecard: dict[str, Any] | None,
) -> Decision:
    strategy = result.get("strategy", "")
    tags = set(classify_report_result(result))
    tags.update(behavior_tags(behavior))
    tags.update(failure_tags(attribution))
    failures = set(scorecard.get("primary_failures", []) if scorecard else [])

    evidence = [
        f"return_pct={result.get('total_profit_pct')}",
        f"profit_factor={result.get('profit_factor')}",
        f"max_drawdown_pct={result.get('max_drawdown_pct')}",
        f"trades={result.get('trades')}",
        f"daily_trades={result.get('daily_trades')}",
        f"long_profit_pct={result.get('long_profit_pct')}",
        f"short_profit_pct={result.get('short_profit_pct')}",
    ]
    if behavior:
        evidence.extend(
            [
                f"win_rate_pct={behavior.get('win_rate_pct')}",
                f"avg_mfe_pct={behavior.get('avg_mfe_pct')}",
                f"avg_mae_pct={behavior.get('avg_mae_pct')}",
                f"max_consecutive_losses={behavior.get('max_consecutive_losses')}",
            ]
        )

    if "high_frequency_negative_expectancy" in tags:
        return Decision(
            priority=100,
            strategy=strategy,
            diagnosis="高频负期望：交易频率已经足够，问题不是单数少，而是每笔交易平均没有 edge。",
            confidence="high",
            evidence=evidence,
            response_plan=[
                "停止用更高杠杆放大该信号。",
                "在固定 50x 合约口径下验证信号本身是否能形成净 edge。",
                "自动测试反向信号、确认入场、成交费压力和短持仓退出。",
                "若 fee stress 后仍 PF<1，整类 K 线 scalping 降级为研究失败样本。",
            ],
            next_experiments=[
                "inverse_signal_retest",
                "fixed_50x_signal_edge_check",
                "fee_sensitivity_grid",
                "entry_delay_confirmation",
                "time_stop_exit_grid",
            ],
            next_command="user_data/strategy_research/start_manual_research.sh --mature-researcher",
            success_gate="Fixed-50x base-fee and stress-fee adjusted return remains positive, drawdown stays within family gate, and sample size clears the current gate.",
            promotion_block="Do not promote high-frequency strategies until net expectancy is positive after fee/slippage stress.",
        )

    if "two_sided_signal_failure" in tags:
        return Decision(
            priority=95,
            strategy=strategy,
            diagnosis="多空双侧信号失败：long 和 short 都亏，说明不是方向偏差，而是入场/退出逻辑本身无效。",
            confidence="high",
            evidence=evidence,
            response_plan=[
                "拆分 long-only 与 short-only，不允许继续混在一个总体收益里看。",
                "对每个方向分别做 regime matrix。",
                "自动生成反向、延迟、过滤低波动三组诊断实验。",
            ],
            next_experiments=[
                "long_short_lane_split",
                "regime_matrix_by_direction",
                "inverse_by_direction",
                "volatility_floor_filter",
            ],
            next_command="user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses",
            success_gate="At least one isolated direction has positive adjusted return and no hostile regime cluster.",
            promotion_block="Do not promote combined long/short strategies when both lanes are independently negative.",
        )

    if "bad_entry_timing" in tags:
        return Decision(
            priority=90,
            strategy=strategy,
            diagnosis="入场时机问题：平均不利波动大于有利波动，信号触发点太早或确认不足。",
            confidence="medium",
            evidence=evidence,
            response_plan=[
                "测试信号后 3/5/10 根 K 线延迟入场。",
                "要求价格先朝有利方向移动再进场。",
                "对比 MFE 损失与 MAE 改善，不只看总收益。",
            ],
            next_experiments=[
                "entry_delay_confirmation",
                "price_moves_in_favor_before_entry",
                "mfe_mae_before_after_comparison",
            ],
            next_command="user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses",
            success_gate="avg_mae_pct falls and PF improves without reducing trades below the evaluation floor.",
            promotion_block="Do not promote strategies whose entries show persistent adverse excursion dominance.",
        )

    if "cost_sensitivity" in tags or "negative_after_cost" in failures or "stress_cost_failure" in failures:
        return Decision(
            priority=86,
            strategy=strategy,
            diagnosis="成本敏感：名义收益可能被手续费、滑点或 funding 吃掉。",
            confidence="medium",
            evidence=evidence,
            response_plan=[
                "自动跑 base fee / high fee / slippage stress 三档。",
                "降低交易频率或提高单笔目标，不通过 stress cost 就不晋级。",
                "对 scalping 单独标注需要 maker/taker 成交假设。",
            ],
            next_experiments=[
                "fee_sensitivity_grid",
                "slippage_stress_grid",
                "min_edge_per_trade_filter",
            ],
            next_command="user_data/strategy_research/start_manual_research.sh --promotion-gate",
            success_gate="Adjusted return remains positive after stress cost and funding estimate.",
            promotion_block="Do not promote cost-negative strategies even if base-fee backtest is positive.",
        )

    if "insufficient_sample" in tags:
        return Decision(
            priority=75,
            strategy=strategy,
            diagnosis="样本不足：交易太少，当前结果不够评价策略质量。",
            confidence="medium",
            evidence=evidence,
            response_plan=[
                "先放宽一个条件，不同时放宽多个条件。",
                "保持最多 3 个确认条件，避免为了增加交易数引入复杂过拟合。",
                "扩大到固定 walk-forward 窗口后再判断。",
            ],
            next_experiments=[
                "single_condition_relaxation",
                "timeframe_breadth_check",
                "walk_forward_sample_floor",
            ],
            next_command="user_data/strategy_research/start_manual_research.sh --memory-guided-hypotheses",
            success_gate="Trade count clears sample floor while PF and drawdown do not deteriorate.",
            promotion_block="Do not promote thin-sample strategies from one favorable slice.",
        )

    if "candidate_needs_robustness" in tags:
        return Decision(
            priority=70,
            strategy=strategy,
            diagnosis="候选策略需要稳健性复核：已有正收益，但还不能等同于可 dry-run。",
            confidence="medium",
            evidence=evidence,
            response_plan=[
                "补齐 recursive-analysis 与 lookahead-analysis。",
                "跑 regime matrix、walk-forward、stress fee。",
                "只有全部通过才进入人工 dry-run 评审。",
            ],
            next_experiments=[
                "recursive_analysis",
                "lookahead_analysis",
                "regime_matrix",
                "walk_forward_validation",
                "stress_cost_validation",
            ],
            next_command="user_data/strategy_research/start_manual_research.sh --promotion-gate",
            success_gate="Promotion gate reports ready_for_manual_dryrun_review=true.",
            promotion_block="Research candidate is not live permission.",
        )

    return Decision(
        priority=40,
        strategy=strategy,
        diagnosis="证据不足以形成强诊断：先刷新行为分析、失败归因和稳健性矩阵。",
        confidence="low",
        evidence=evidence,
        response_plan=[
            "刷新 trade behavior、failure attribution、strategy lineage。",
            "不要在诊断不清时直接优化参数。",
        ],
        next_experiments=["refresh_behavior_evidence", "refresh_failure_attribution"],
        next_command="user_data/strategy_research/start_manual_research.sh --trade-behavior",
        success_gate="A dominant failure mode is identified or strategy remains stable across matrix checks.",
        promotion_block="Do not promote strategies without a clear evidence trail.",
    )


def build_payload() -> dict[str, Any]:
    agent_report = latest_agent_report()
    behavior = load_json(AGENT_ROOT / "trade_behavior/latest_trade_behavior.json")
    failure = load_json(AGENT_ROOT / "failure_attribution/latest_failure_attribution.json")
    assessment = load_json(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json")
    promotion = load_json(AGENT_ROOT / "promotion_reports/latest_promotion_report.json")

    behaviors = index_by_strategy(behavior.get("summaries", []))
    attributions = index_by_strategy(failure.get("attributions", []))
    scorecards = index_by_strategy(assessment.get("scorecards", []))
    decisions = [
        build_decision(
            result,
            behaviors.get(result.get("strategy", "")),
            attributions.get(result.get("strategy", "")),
            scorecards.get(result.get("strategy", "")),
        )
        for result in agent_report.get("results", [])
        if result.get("strategy")
    ]
    decisions = sorted(decisions, key=lambda item: (-item.priority, item.strategy))
    ready = [
        item.get("strategy")
        for item in promotion.get("verdicts", [])
        if item.get("ready_for_manual_dryrun_review")
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "decision_count": len(decisions),
        "top_decisions": [asdict(item) for item in decisions[:12]],
        "decisions": [asdict(item) for item in decisions],
        "ready_for_manual_dryrun_review": ready,
        "global_policy": [
            "Never use leverage to rescue a negative-expectancy signal.",
            "Do not promote from a single sample, single regime, or base-fee-only result.",
            "High-frequency strategies require explicit fee/slippage stress and trade-behavior diagnostics.",
            "Order-book, market-making, and microstructure ideas require data beyond OHLCV before live claims.",
        ],
        "source_artifacts": {
            "agent_report_index": rel(AGENT_ROOT / "reports/agent_report_index.json"),
            "trade_behavior": rel(AGENT_ROOT / "trade_behavior/latest_trade_behavior.json"),
            "failure_attribution": rel(AGENT_ROOT / "failure_attribution/latest_failure_attribution.json"),
            "strategy_assessment": rel(AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json"),
            "promotion_report": rel(AGENT_ROOT / "promotion_reports/latest_promotion_report.json"),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Mature Researcher Decision Plan",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Decisions: `{payload['decision_count']}`",
        f"- Ready for manual dry-run review: `{', '.join(payload['ready_for_manual_dryrun_review']) or 'none'}`",
        "",
        "## Global Policy",
        "",
    ]
    for item in payload["global_policy"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Top Decisions",
            "",
            "| Priority | Strategy | Diagnosis | Confidence | Next Command | Success Gate | Promotion Block |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for item in payload["top_decisions"]:
        lines.append(
            "| {priority} | {strategy} | {diagnosis} | {confidence} | `{next_command}` | {success_gate} | {promotion_block} |".format(
                **item
            )
        )
    lines.extend(["", "## Evidence And Response Plans", ""])
    for item in payload["top_decisions"]:
        lines.append(f"### {item['strategy']}")
        lines.append(f"- Diagnosis: {item['diagnosis']}")
        lines.append(f"- Confidence: `{item['confidence']}`")
        lines.append(f"- Success gate: {item['success_gate']}")
        lines.append(f"- Promotion block: {item['promotion_block']}")
        lines.append("- Evidence:")
        for evidence in item["evidence"]:
            lines.append(f"  - {evidence}")
        lines.append("- Response plan:")
        for step in item["response_plan"]:
            lines.append(f"  - {step}")
        lines.append("- Next experiments:")
        for experiment in item["next_experiments"]:
            lines.append(f"  - {experiment}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"researcher_decision_{timestamp}.json"
    md_path = REPORT_DIR / f"researcher_decision_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_REPORT_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_REPORT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Decisions: {payload['decision_count']}")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)


if __name__ == "__main__":
    main()
