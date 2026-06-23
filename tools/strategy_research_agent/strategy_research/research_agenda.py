#!/usr/bin/env python3
"""Build the next research agenda from promotion gate blockers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
REPORT_DIR = AGENT_ROOT / "research_agendas"
LATEST_REPORT_JSON = REPORT_DIR / "latest_research_agenda.json"
LATEST_REPORT_MD = REPORT_DIR / "latest_research_agenda.md"


@dataclass
class AgendaItem:
    priority: int
    strategy: str
    blocker: str
    objective: str
    hypothesis: str
    next_command: str
    success_gate: str
    risk_note: str


BLOCKER_PLAYBOOK: dict[str, dict[str, Any]] = {
    "not_in_candidate_pool": {
        "priority": 20,
        "objective": "先进入 research_candidate 候选池",
        "hypothesis": "观察池策略只有先通过 smoke/full-sample gates，后续稳健性验证才有意义。",
        "next_command": "user_data/strategy_research/start_manual_research.sh --iterate-smoke",
        "success_gate": "classification becomes research_candidate with positive adjusted return.",
        "risk_note": "不要因为单次小样本盈利跳过候选池规则。",
    },
    "too_few_trades": {
        "priority": 35,
        "objective": "提高有效交易数并保持 PF/回撤不恶化",
        "hypothesis": "当前入场条件过窄；放宽确认窗口或增加 regime-specific 变体可能提升样本量。",
        "next_command": "./.venv/bin/python user_data/strategy_research/strategy_iteration_engine.py",
        "success_gate": ">=100 trades while adjusted return stays positive and max drawdown remains bounded.",
        "risk_note": "交易数增加不能靠过度高频噪音换来。",
    },
    "scorecard_not_promotable": {
        "priority": 40,
        "objective": "提升评分卡到 promotable_research_candidate",
        "hypothesis": "策略需要同时改善收益、PF、回撤、成本校正和矩阵韧性，而不是只优化单一指标。",
        "next_command": "./.venv/bin/python user_data/strategy_research/analyze_strategy_research.py",
        "success_gate": "scorecard tier becomes promotable_research_candidate.",
        "risk_note": "不要用过拟合参数只抬高全样本收益。",
    },
    "walk_forward_not_passed": {
        "priority": 50,
        "objective": "跨固定时间窗稳定",
        "hypothesis": "把策略按 bull/bear/range 或波动状态拆分，可能减少单窗口依赖。",
        "next_command": "user_data/strategy_research/start_manual_research.sh --walk-forward",
        "success_gate": "walk-forward verdict becomes walk_forward_candidate.",
        "risk_note": "单个窗口表现优秀不能抵消跨窗口失效。",
    },
    "matrix_not_robust": {
        "priority": 55,
        "objective": "通过市场状态与成本矩阵",
        "hypothesis": "当前规则在部分 regime 或 stress fee 下脆弱；需要状态过滤、退出提速或降杠杆。",
        "next_command": "user_data/strategy_research/run_full_research_cycle.sh --skip-aux-fetch",
        "success_gate": "matrix verdict becomes robust_candidate with no stress-negative cluster.",
        "risk_note": "不要只看 base fee；stress cost 不通过就不能晋级。",
    },
    "cost_evidence_missing": {
        "priority": 60,
        "objective": "补齐交易级成本证据",
        "hypothesis": "没有 funding、手续费和滑点校正时，合约策略收益不可采信。",
        "next_command": "./.venv/bin/python user_data/strategy_research/estimate_trade_costs.py",
        "success_gate": "latest cost estimate contains this strategy with positive adjusted return.",
        "risk_note": "缺成本证据时禁止进入 dry-run 评审。",
    },
    "negative_after_cost": {
        "priority": 65,
        "objective": "把成本后收益修回正值",
        "hypothesis": "减少持仓时间、降低换手或提高单笔 edge，可能抵消 funding/slippage 侵蚀。",
        "next_command": "user_data/strategy_research/run_full_research_cycle.sh --skip-aux-fetch",
        "success_gate": "adjusted_profit_pct > 0 after funding and slippage estimates.",
        "risk_note": "成本后亏损的策略不能靠高杠杆掩盖。",
    },
    "bias_checks_missing": {
        "priority": 70,
        "objective": "补齐 recursive/lookahead 偏差检查",
        "hypothesis": "策略必须证明没有未来函数和递归指标漂移，才能进入 dry-run 复核。",
        "next_command": "./.venv/bin/python user_data/strategy_research/run_research_agent.py --run-recursive --run-lookahead",
        "success_gate": "recursive_analysis and lookahead_analysis both return ok.",
        "risk_note": "偏差检查缺失时，任何回测收益都只能当作未验证。",
    },
}


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


def build_items(promotion_report: dict[str, Any]) -> list[AgendaItem]:
    items: list[AgendaItem] = []
    for verdict in promotion_report.get("verdicts", []):
        strategy = verdict.get("strategy", "")
        for blocker in verdict.get("blocks", []):
            play = BLOCKER_PLAYBOOK.get(blocker)
            if not play:
                continue
            items.append(
                AgendaItem(
                    priority=play["priority"],
                    strategy=strategy,
                    blocker=blocker,
                    objective=play["objective"],
                    hypothesis=play["hypothesis"],
                    next_command=play["next_command"],
                    success_gate=play["success_gate"],
                    risk_note=play["risk_note"],
                )
            )
    return sorted(items, key=lambda item: (-item.priority, item.strategy, item.blocker))


def build_payload() -> dict[str, Any]:
    promotion_path = AGENT_ROOT / "promotion_reports/latest_promotion_report.json"
    promotion_report = load_json(promotion_path)
    items = build_items(promotion_report)
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_promotion_report": rel(promotion_path),
        "agenda_count": len(items),
        "ready_count": promotion_report.get("ready_count", 0),
        "blocked_count": promotion_report.get("blocked_count", 0),
        "top_priorities": [item.__dict__ for item in items[:12]],
        "items": [item.__dict__ for item in items],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Research Agenda",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Source promotion report: `{payload['source_promotion_report']}`",
        f"- Ready strategies: `{payload['ready_count']}`",
        f"- Blocked strategies: `{payload['blocked_count']}`",
        f"- Agenda items: `{payload['agenda_count']}`",
        "",
        "## Top Priorities",
        "",
        "| Priority | Strategy | Blocker | Objective | Next Command | Success Gate |",
        "|---:|---|---|---|---|---|",
    ]
    for item in payload["top_priorities"]:
        lines.append(
            "| {priority} | {strategy} | {blocker} | {objective} | `{next_command}` | {success_gate} |".format(
                **item
            )
        )
    lines.extend(["", "## Research Notes", ""])
    for item in payload["top_priorities"]:
        lines.append(f"### {item['strategy']} / {item['blocker']}")
        lines.append(f"- Hypothesis: {item['hypothesis']}")
        lines.append(f"- Risk: {item['risk_note']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"research_agenda_{timestamp}.json"
    md_path = REPORT_DIR / f"research_agenda_{timestamp}.md"
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    LATEST_REPORT_JSON.write_text(json_text, encoding="utf-8")
    write_markdown(md_path, payload)
    LATEST_REPORT_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {LATEST_REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"Agenda items: {payload['agenda_count']}")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)


if __name__ == "__main__":
    main()
