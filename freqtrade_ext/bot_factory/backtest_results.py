from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class BacktestMetrics:
    strategy_name: str
    total_return: float
    total_return_pct: float
    cagr: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown_pct: float | None
    profit_factor: float | None
    win_rate: float
    average_win: float | None
    average_loss: float | None
    trade_count: int
    expectancy: float | None
    fee_paid: float | None
    backtest_start: str | None
    backtest_end: str | None
    generated_at: str


@dataclass
class GateThresholds:
    min_trades: int = 200
    min_profit_factor: float = 1.25
    max_drawdown_pct: float = 15.0
    min_sortino: float | None = 1.2


def load_backtest_result(path: Path) -> dict[str, Any]:
    if path.suffix == ".zip":
        return _load_backtest_zip(path)

    result = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(result, dict) and isinstance(result.get("latest_backtest"), str):
        return load_backtest_result(path.parent / result["latest_backtest"])
    return result


def select_strategy_result(result: dict[str, Any], strategy_name: str | None = None) -> dict[str, Any]:
    strategies = result.get("strategy", {})
    if not strategies:
        raise ValueError("Backtest JSON does not contain a 'strategy' section.")

    if strategy_name:
        if strategy_name not in strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found in backtest JSON.")
        return strategies[strategy_name]

    if len(strategies) != 1:
        names = ", ".join(strategies)
        raise ValueError(f"Multiple strategies found. Specify --strategy. Available: {names}")
    return next(iter(strategies.values()))


def strategy_name_from_result(result: dict[str, Any], fallback: str | None = None) -> str:
    strategies = result.get("strategy", {})
    if fallback:
        return fallback
    if len(strategies) == 1:
        return next(iter(strategies))
    return "unknown_strategy"


def summarize(result: dict[str, Any], strategy_name: str | None = None) -> BacktestMetrics:
    selected = select_strategy_result(result, strategy_name)
    selected_name = strategy_name_from_result(result, strategy_name)
    trades = selected.get("trades", []) or []
    profit_ratios = [float(t.get("profit_ratio", 0.0) or 0.0) for t in trades]
    wins = [p for p in profit_ratios if p > 0]
    losses = [p for p in profit_ratios if p < 0]

    trade_count = int(selected.get("total_trades", len(trades)) or len(trades))
    win_rate = (len(wins) / trade_count) if trade_count else 0.0
    fee_paid = _sum_fees(trades)

    max_dd = selected.get("max_drawdown_account", selected.get("max_relative_drawdown"))
    max_dd_pct = float(max_dd) * 100 if max_dd is not None else None

    return BacktestMetrics(
        strategy_name=selected_name,
        total_return=float(selected.get("profit_total", 0.0) or 0.0),
        total_return_pct=_total_return_pct(selected),
        cagr=_optional_float(selected.get("cagr")),
        sharpe=_optional_float(selected.get("sharpe")),
        sortino=_optional_float(selected.get("sortino")),
        calmar=_optional_float(selected.get("calmar")),
        max_drawdown_pct=max_dd_pct,
        profit_factor=_optional_float(selected.get("profit_factor")),
        win_rate=win_rate,
        average_win=mean(wins) if wins else None,
        average_loss=mean(losses) if losses else None,
        trade_count=trade_count,
        expectancy=mean(profit_ratios) if profit_ratios else None,
        fee_paid=fee_paid,
        backtest_start=selected.get("backtest_start"),
        backtest_end=selected.get("backtest_end"),
        generated_at=datetime.now(UTC).isoformat(),
    )


def write_metrics(metrics: BacktestMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2, ensure_ascii=False), encoding="utf-8")


def write_result_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def write_trades_csv(result: dict[str, Any], path: Path, strategy_name: str | None = None) -> None:
    selected = select_strategy_result(result, strategy_name)
    trades = selected.get("trades", []) or []
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trades:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for trade in trades for key in trade.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)


def load_gate_thresholds(path: Path | None = None) -> GateThresholds:
    if path is None:
        return GateThresholds()

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Gate config must be a JSON object: {path}")
    rules = payload.get("backtest_rules", payload)
    if not isinstance(rules, dict):
        raise ValueError(f"Gate config 'backtest_rules' must be a JSON object: {path}")

    allowed = {field.name for field in GateThresholds.__dataclass_fields__.values()}
    unknown = sorted(set(rules) - allowed)
    if unknown:
        raise ValueError(f"Unknown gate threshold key(s): {', '.join(unknown)}")

    values = asdict(GateThresholds())
    values.update(rules)
    return GateThresholds(**values)


def write_report(
    metrics: BacktestMetrics,
    path: Path,
    thresholds: GateThresholds | None = None,
    reviewer_notes: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or GateThresholds()
    gate = evaluate_initial_gate(metrics, thresholds)
    lines = [
        f"# Backtest Report: {metrics.strategy_name}",
        "",
        "## Summary",
        "",
        f"- Total return: {metrics.total_return:.6f} ({metrics.total_return_pct:.2f}%)",
        f"- Trade count: {metrics.trade_count}",
        f"- Win rate: {metrics.win_rate:.2%}",
        f"- Profit factor: {_fmt(metrics.profit_factor)}",
        f"- Max drawdown: {_fmt(metrics.max_drawdown_pct)}%",
        f"- CAGR: {_fmt(metrics.cagr)}",
        f"- Sharpe: {_fmt(metrics.sharpe)}",
        f"- Sortino: {_fmt(metrics.sortino)}",
        f"- Calmar: {_fmt(metrics.calmar)}",
        f"- Expectancy: {_fmt(metrics.expectancy)}",
        f"- Fee paid: {_fmt(metrics.fee_paid)}",
        f"- Period: {metrics.backtest_start or 'unknown'} to {metrics.backtest_end or 'unknown'}",
        "",
        "## Initial Gate",
        "",
        f"- Recommendation: {gate['recommendation']}",
        f"- Promotion recommendation: {promotion_recommendation(gate)}",
        "",
        "## Gate Checks",
        "",
    ]
    for check in gate["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(f"- {status}: {check['name']} ({check['actual']} vs {check['rule']})")
    lines.extend(
        [
            "",
            "## Gate Thresholds",
            "",
            f"- Minimum trades: {thresholds.min_trades}",
            f"- Minimum profit factor: {thresholds.min_profit_factor}",
            f"- Maximum drawdown pct: {thresholds.max_drawdown_pct}",
            f"- Minimum sortino: {_fmt(thresholds.min_sortino)}",
            "",
            "## Reviewer Notes",
            "",
        ]
    )
    if reviewer_notes:
        lines.extend(f"- {note}" for note in reviewer_notes)
    else:
        lines.append("- No human reviewer notes recorded.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is generated from a backtest only.",
            "- It is not approval for paper trading or live trading.",
            "- Production promotion requires walk-forward, paper trading, and human approval.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_initial_gate(
    metrics: BacktestMetrics, thresholds: GateThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    checks = [
        {
            "name": "min_trades",
            "actual": metrics.trade_count,
            "rule": f">= {thresholds.min_trades}",
            "pass": metrics.trade_count >= thresholds.min_trades,
        },
        {
            "name": "min_profit_factor",
            "actual": _fmt(metrics.profit_factor),
            "rule": f">= {thresholds.min_profit_factor}",
            "pass": metrics.profit_factor is not None
            and metrics.profit_factor >= thresholds.min_profit_factor,
        },
        {
            "name": "max_drawdown_pct",
            "actual": _fmt(metrics.max_drawdown_pct),
            "rule": f"<= {thresholds.max_drawdown_pct}",
            "pass": metrics.max_drawdown_pct is not None
            and metrics.max_drawdown_pct <= thresholds.max_drawdown_pct,
        },
    ]
    if thresholds.min_sortino is not None:
        checks.append(
            {
                "name": "min_sortino",
                "actual": _fmt(metrics.sortino),
                "rule": f">= {thresholds.min_sortino}",
                "pass": metrics.sortino is not None and metrics.sortino >= thresholds.min_sortino,
            }
        )
    recommendation = "pass" if all(check["pass"] for check in checks) else "fail"
    return {"recommendation": recommendation, "checks": checks, "thresholds": asdict(thresholds)}


def promotion_recommendation(gate: dict[str, Any]) -> str:
    if gate.get("recommendation") == "pass":
        return "eligible_for_walk_forward_review"
    return "retry_with_modification"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_fees(trades: list[dict[str, Any]]) -> float | None:
    if not trades:
        return None
    total = 0.0
    found = False
    for trade in trades:
        for key in ("fee_open_cost", "fee_close_cost"):
            value = trade.get(key)
            if value is not None:
                try:
                    total += float(value)
                    found = True
                except (TypeError, ValueError):
                    pass
    return total if found else None


def _total_return_pct(selected: dict[str, Any]) -> float:
    pct = _optional_float(selected.get("profit_total_pct"))
    if pct is not None:
        return pct
    total = _optional_float(selected.get("profit_total"))
    return (total * 100.0) if total is not None else 0.0


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _load_backtest_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json") or name.endswith("_config.json"):
                continue

            result = json.loads(archive.read(name))
            if isinstance(result, dict) and "strategy" in result:
                return result

    raise ValueError(f"Backtest zip does not contain a result JSON with a 'strategy' section: {path}")
