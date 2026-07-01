#!/usr/bin/env python3
"""Evaluate strategy-family promotion with router and circuit-breaker logic.

This gate is intentionally different from a naked all-regime strategy gate:
high-leverage crypto strategies are expected to specialize by regime.  The
promotion question is whether target-regime edge remains positive and hostile
regime losses are contained by family/portfolio risk controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "user_data/strategy_research/reports"
OUTPUT_DIR = REPO_ROOT / "user_data/strategy_research/family_risk_gate"
PROMOTION_DIR = REPO_ROOT / "user_data/strategy_research/promotion_reports"

STARTING_BALANCE = 1000.0
TARGET_65D_GATE = 30.0
TARGET_30D_GATE = 20.0
LATEST5_GATE = 0.0
MIN_65D_TRADES = 8
HOSTILE_GUARDED_WORST_GATE = -10.0
FAMILY_DRAWDOWN_PAUSE_PCT = 10.0
CONSECUTIVE_LOSS_PAUSE = 3

FAMILY_INFERENCE = [
    ("SecondLeg", "downtrend_failed_bounce_short"),
    ("FailedBounce", "downtrend_failed_bounce_short"),
    ("BodyNotTooRed", "downtrend_failed_bounce_short"),
    ("DownsideBreakout", "downside_breakout_continuation_short"),
    ("DowntrendPullback", "downtrend_pullback_short"),
    ("UptrendPullback", "uptrend_pullback_long"),
    ("UpsideBreakout", "upside_breakout_continuation_long"),
]


@dataclass
class SimResult:
    raw_profit_pct: float
    guarded_profit_pct: float
    trades_seen: int
    trades_taken: int
    trades_blocked: int
    max_drawdown_pct: float
    pause_reason: str
    evidence_mode: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", dest="csv_path", help="Experiment CSV to evaluate. Defaults to latest *experiment*.csv.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload to stdout.")
    parser.add_argument("--drawdown-pause-pct", type=float, default=FAMILY_DRAWDOWN_PAUSE_PCT)
    parser.add_argument(
        "--consecutive-loss-pause",
        type=int,
        default=CONSECUTIVE_LOSS_PAUSE,
        help="Pause after this many consecutive stop_loss exits. Time-stop drifts are not counted as big losses.",
    )
    return parser.parse_args()


def latest_experiment_csv() -> Path:
    candidates = sorted(REPORT_DIR.glob("*experiment*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No experiment CSV found under {rel(REPORT_DIR)}")
    return candidates[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict[str, Any], field: str) -> float:
    try:
        value = row.get(field)
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def inum(row: dict[str, Any], field: str) -> int:
    try:
        return int(float(row.get(field) or 0))
    except (TypeError, ValueError):
        return 0


def infer_family(strategy: str, row: dict[str, Any]) -> str:
    family = row.get("strategy_family")
    if family:
        return family
    for needle, inferred in FAMILY_INFERENCE:
        if needle in strategy:
            return inferred
    return "unknown"


def scenario_is_high_fee(row: dict[str, Any]) -> bool:
    scenario = row.get("scenario")
    return not scenario or scenario == "high_fee_12bps"


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row.get("strategy", ""),
        row.get("slice", ""),
        row.get("window", ""),
        row.get("scenario") or "high_fee_12bps",
    )


def load_payload(artifact: Path) -> dict[str, Any] | None:
    if not artifact.exists() or artifact.suffix != ".zip":
        return None
    with zipfile.ZipFile(artifact) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.endswith(".json") and not name.endswith("_config.json") and not name.endswith(".meta.json")
        ]
        if not names:
            return None
        return json.loads(zf.read(names[0]))


def trades_for(row: dict[str, Any]) -> list[dict[str, Any]]:
    artifact_text = row.get("artifact") or ""
    if not artifact_text:
        return []
    artifact = REPO_ROOT / artifact_text
    payload = load_payload(artifact)
    if not payload:
        return []
    strategy = row.get("strategy", "")
    stats = (payload.get("strategy") or {}).get(strategy) or {}
    trades = stats.get("trades") or []
    return sorted(trades, key=lambda item: int(item.get("close_timestamp") or item.get("open_timestamp") or 0))


def trade_adjusted_profit_abs(trade: dict[str, Any], slippage_bps: float) -> float:
    profit_abs = float(trade.get("profit_abs") or 0.0)
    notional = float(trade.get("stake_amount") or 0.0) * float(trade.get("leverage") or 1.0)
    slippage_abs = abs(notional) * slippage_bps / 10000.0
    return profit_abs - slippage_abs


def simulate_row_from_trades(
    row: dict[str, Any],
    drawdown_pause_pct: float,
    consecutive_loss_pause: int,
) -> SimResult | None:
    trades = trades_for(row)
    if not trades:
        return None
    slippage_bps = fnum(row, "slippage_bps")
    equity = STARTING_BALANCE
    peak = STARTING_BALANCE
    raw_profit_abs = 0.0
    guarded_profit_abs = 0.0
    max_drawdown = 0.0
    consecutive_stop_losses = 0
    pause_reason = ""
    trades_taken = 0
    trades_blocked = 0
    paused = False

    for trade in trades:
        adjusted_abs = trade_adjusted_profit_abs(trade, slippage_bps)
        raw_profit_abs += adjusted_abs
        if paused:
            trades_blocked += 1
            continue
        guarded_profit_abs += adjusted_abs
        trades_taken += 1
        equity += adjusted_abs
        peak = max(peak, equity)
        drawdown = (peak - equity) / STARTING_BALANCE * 100.0
        max_drawdown = max(max_drawdown, drawdown)
        if trade.get("exit_reason") == "stop_loss":
            consecutive_stop_losses += 1
        else:
            consecutive_stop_losses = 0
        if drawdown >= drawdown_pause_pct:
            paused = True
            pause_reason = f"family_drawdown_pause_{drawdown_pause_pct:g}pct"
        elif consecutive_stop_losses >= consecutive_loss_pause:
            paused = True
            pause_reason = f"consecutive_stop_loss_pause_{consecutive_loss_pause}"

    return SimResult(
        raw_profit_pct=round(raw_profit_abs / STARTING_BALANCE * 100.0, 4),
        guarded_profit_pct=round(guarded_profit_abs / STARTING_BALANCE * 100.0, 4),
        trades_seen=len(trades),
        trades_taken=trades_taken,
        trades_blocked=trades_blocked,
        max_drawdown_pct=round(max_drawdown, 4),
        pause_reason=pause_reason or "none",
        evidence_mode="trade_level",
    )


def simulate_row_aggregate(row: dict[str, Any]) -> SimResult:
    adjusted = fnum(row, "adjusted_profit_pct") or fnum(row, "profit_total_pct")
    guarded = max(adjusted, HOSTILE_GUARDED_WORST_GATE) if row.get("slice") == "regime" else adjusted
    return SimResult(
        raw_profit_pct=round(adjusted, 4),
        guarded_profit_pct=round(guarded, 4),
        trades_seen=inum(row, "trades"),
        trades_taken=inum(row, "trades"),
        trades_blocked=0,
        max_drawdown_pct=fnum(row, "max_drawdown_pct"),
        pause_reason="aggregate_only",
        evidence_mode="aggregate_simulation",
    )


def simulate_row(row: dict[str, Any], drawdown_pause_pct: float, consecutive_loss_pause: int) -> SimResult:
    trade_level = simulate_row_from_trades(row, drawdown_pause_pct, consecutive_loss_pause)
    return trade_level or simulate_row_aggregate(row)


def summarize_strategy(
    strategy: str,
    family: str,
    rows: list[dict[str, Any]],
    drawdown_pause_pct: float,
    consecutive_loss_pause: int,
) -> dict[str, Any]:
    high_rows = [row for row in rows if scenario_is_high_fee(row)]
    sims = {
        row_key(row): simulate_row(row, drawdown_pause_pct, consecutive_loss_pause)
        for row in high_rows
    }
    main = {
        row.get("window", ""): (row, sims[row_key(row)])
        for row in high_rows
        if row.get("slice") == "main"
    }
    walk_forward = [(row, sims[row_key(row)]) for row in high_rows if row.get("slice") == "walk_forward"]
    hostile = [(row, sims[row_key(row)]) for row in high_rows if row.get("slice") == "regime"]

    row_65, sim_65 = main.get("65d", ({}, None))
    row_30, sim_30 = main.get("30d", ({}, None))
    row_5, sim_5 = main.get("latest5", ({}, None))
    target_65 = sim_65.guarded_profit_pct if sim_65 else 0.0
    target_30 = sim_30.guarded_profit_pct if sim_30 else 0.0
    latest5 = sim_5.guarded_profit_pct if sim_5 else 0.0
    trades_65 = sim_65.trades_taken if sim_65 else 0
    wf_positive = sum(1 for _, sim in walk_forward if sim.guarded_profit_pct > 0)
    wf_total = len(walk_forward)
    wf_worst = min((sim.guarded_profit_pct for _, sim in walk_forward), default=0.0)
    hostile_raw_worst = min((sim.raw_profit_pct for _, sim in hostile), default=0.0)
    hostile_guarded_worst = min((sim.guarded_profit_pct for _, sim in hostile), default=0.0)
    hostile_guarded_total = sum(sim.guarded_profit_pct for _, sim in hostile)
    evidence_modes = sorted({sim.evidence_mode for sim in sims.values()})

    blockers: list[str] = []
    supports: list[str] = []
    if target_65 <= TARGET_65D_GATE:
        blockers.append(f"65d target-regime guarded profit {target_65:.4f}% <= {TARGET_65D_GATE:.1f}%")
    else:
        supports.append(f"65d target-regime guarded profit {target_65:.4f}% clears gate")
    if target_30 <= TARGET_30D_GATE:
        blockers.append(f"30d target-regime guarded profit {target_30:.4f}% <= {TARGET_30D_GATE:.1f}%")
    else:
        supports.append(f"30d target-regime guarded profit {target_30:.4f}% clears gate")
    if latest5 <= LATEST5_GATE:
        blockers.append("latest5 is not positive after family risk controls")
    else:
        supports.append(f"latest5 remains positive at {latest5:.4f}%")
    if trades_65 < MIN_65D_TRADES:
        blockers.append(f"65d trades taken {trades_65} < {MIN_65D_TRADES}")
    if hostile_guarded_worst < HOSTILE_GUARDED_WORST_GATE:
        blockers.append(
            f"hostile-regime guarded worst {hostile_guarded_worst:.4f}% < {HOSTILE_GUARDED_WORST_GATE:.1f}%"
        )
    else:
        supports.append(f"hostile-regime guarded worst {hostile_guarded_worst:.4f}% is contained")
    if wf_total and wf_positive < wf_total and wf_worst < -2.0:
        blockers.append(f"walk-forward guarded positives {wf_positive}/{wf_total}, worst {wf_worst:.4f}%")
    if "aggregate_simulation" in evidence_modes:
        blockers.append("some rows used aggregate simulation; require trade-level confirmation before dry-run review")

    ready = not blockers
    state = "dryrun_candidate_review_pending_manual_approval" if ready else "research_candidate"
    next_actions: list[str] = []
    if blockers:
        next_actions.append("improve_regime_router_or_family_risk_controls")
    if "aggregate_simulation" in evidence_modes:
        next_actions.append("rerun_with_trade_level_artifacts")
    if ready:
        next_actions.append("run_recursive_lookahead_and_manual_dryrun_review")
    if not next_actions:
        next_actions.append("continue_research")
    return {
        "strategy": strategy,
        "strategy_family": family,
        "ready_for_manual_dryrun_review": ready,
        "state": state,
        "verdict": state,
        "target_65d_guarded_pct": round(target_65, 4),
        "target_30d_guarded_pct": round(target_30, 4),
        "latest5_guarded_pct": round(latest5, 4),
        "target_65d_trades_taken": trades_65,
        "walk_forward_positive": wf_positive,
        "walk_forward_total": wf_total,
        "walk_forward_worst_guarded_pct": round(wf_worst, 4),
        "hostile_raw_worst_pct": round(hostile_raw_worst, 4),
        "hostile_guarded_worst_pct": round(hostile_guarded_worst, 4),
        "hostile_guarded_total_pct": round(hostile_guarded_total, 4),
        "evidence_modes": evidence_modes,
        "supports": supports,
        "blockers": blockers,
        "blocks": "; ".join(blockers) if blockers else "none",
        "next_actions": "; ".join(next_actions),
    }


def build_payload(csv_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(csv_path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strategy = row.get("strategy", "")
        family = infer_family(strategy, row)
        grouped[(family, strategy)].append(row)
    verdicts = [
        summarize_strategy(strategy, family, strategy_rows, args.drawdown_pause_pct, args.consecutive_loss_pause)
        for (family, strategy), strategy_rows in sorted(grouped.items())
    ]
    family_verdicts: dict[str, dict[str, Any]] = {}
    for family in sorted({item["strategy_family"] for item in verdicts}):
        family_items = [item for item in verdicts if item["strategy_family"] == family]
        best = max(
            family_items,
            key=lambda item: (
                item["ready_for_manual_dryrun_review"],
                item["target_65d_guarded_pct"],
                item["hostile_guarded_worst_pct"],
            ),
        )
        family_verdicts[family] = {
            "family": family,
            "best_strategy": best["strategy"],
            "ready_for_manual_dryrun_review": best["ready_for_manual_dryrun_review"],
            "state": best["state"],
            "verdict": best["verdict"],
            "blockers": best["blockers"],
            "blocks": best["blocks"],
            "supports": best["supports"],
            "next_actions": best["next_actions"],
        }
    return {
        "generated_at_utc": now_utc(),
        "source_csv": rel(csv_path),
        "gate_version": 1,
        "scope": "all_strategy_families",
        "promotion_principle": (
            "Strategy families do not need to be all-regime holy grails.  Dry-run review requires "
            "target-regime edge plus hostile-regime loss containment under family/portfolio circuit breakers."
        ),
        "risk_controls": {
            "starting_balance": STARTING_BALANCE,
            "family_drawdown_pause_pct": args.drawdown_pause_pct,
            "consecutive_stop_loss_pause": args.consecutive_loss_pause,
            "hostile_guarded_worst_gate_pct": HOSTILE_GUARDED_WORST_GATE,
        },
        "family_verdicts": list(family_verdicts.values()),
        "verdicts": verdicts,
    }


def write_json(payload: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROMOTION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    path = OUTPUT_DIR / f"family_risk_gate_{timestamp}.json"
    latest = OUTPUT_DIR / "latest_family_risk_gate.json"
    promotion_latest = PROMOTION_DIR / "latest_promotion_report.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    promotion_latest.write_text(text, encoding="utf-8")
    return path


def write_markdown(payload: dict[str, Any]) -> Path:
    timestamp = payload["generated_at_utc"]
    path = OUTPUT_DIR / f"family_risk_gate_{timestamp}.md"
    latest = OUTPUT_DIR / "latest_family_risk_gate.md"
    promotion_latest = PROMOTION_DIR / "latest_promotion_report.md"
    lines = [
        "# Strategy Family Risk Gate",
        "",
        f"- Generated UTC: `{timestamp}`",
        f"- Source CSV: `{payload['source_csv']}`",
        f"- Scope: `{payload['scope']}`",
        "",
        "## Principle",
        "",
        payload["promotion_principle"],
        "",
        "## Risk Controls",
        "",
        "| Control | Value |",
        "|---|---:|",
    ]
    for key, value in payload["risk_controls"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Family Verdicts", "", "| Family | Best Strategy | Ready | State | Blockers |", "|---|---|---:|---|---|"])
    for item in payload["family_verdicts"]:
        blockers = "; ".join(item["blockers"]) if item["blockers"] else "none"
        lines.append(
            f"| `{item['family']}` | `{item['best_strategy']}` | {item['ready_for_manual_dryrun_review']} | "
            f"`{item['state']}` | {blockers} |"
        )
    lines.extend(
        [
            "",
            "## Strategy Verdicts",
            "",
            "| Strategy | Family | State | 65d guarded | 30d guarded | latest5 | WF | Hostile raw worst | Hostile guarded worst | Evidence |",
            "|---|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for item in payload["verdicts"]:
        evidence = ", ".join(item["evidence_modes"])
        wf = f"{item['walk_forward_positive']}/{item['walk_forward_total']} worst {item['walk_forward_worst_guarded_pct']:.4f}%"
        lines.append(
            f"| `{item['strategy']}` | `{item['strategy_family']}` | `{item['state']}` | "
            f"{item['target_65d_guarded_pct']:.4f} | {item['target_30d_guarded_pct']:.4f} | "
            f"{item['latest5_guarded_pct']:.4f} | {wf} | {item['hostile_raw_worst_pct']:.4f} | "
            f"{item['hostile_guarded_worst_pct']:.4f} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Promotion Boundary",
            "",
            "- This gate records readiness for manual dry-run review only.",
            "- It does not start dry-run/live trading and does not edit trading config.",
            "- A family can pass only when target-regime edge survives and hostile-regime loss is contained by circuit breakers.",
            "",
        ]
    )
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    promotion_latest.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path) if args.csv_path else latest_experiment_csv()
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    payload = build_payload(csv_path, args)
    json_path = write_json(payload)
    md_path = write_markdown(payload)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Wrote {rel(json_path)}")
        print(f"Wrote {rel(md_path)}")
        print(f"Wrote {rel(PROMOTION_DIR / 'latest_promotion_report.json')}")
        print(f"Wrote {rel(PROMOTION_DIR / 'latest_promotion_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
