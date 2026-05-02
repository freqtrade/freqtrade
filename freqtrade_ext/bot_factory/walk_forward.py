from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    timerange: str
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WalkForwardRules:
    min_pass_rate: float = 0.7
    min_profitable_windows_ratio: float = 0.6
    max_drawdown_pct_any_window: float = 20.0
    max_single_window_profit_dependency: float = 0.4


def parse_window_specs(specs: Sequence[str]) -> list[WalkForwardWindow]:
    return [parse_window_spec(spec, index) for index, spec in enumerate(specs, start=1)]


def parse_window_spec(spec: str, index: int = 1) -> WalkForwardWindow:
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Window specs with train/test boundaries must use "
                "TRAIN_START:TRAIN_END:TEST_START:TEST_END."
            )
        train_start, train_end, test_start, test_end = [_date_token(part) for part in parts]
        if not (_parse_date(train_start) < _parse_date(train_end) <= _parse_date(test_start)):
            raise ValueError(f"Invalid train/test window order: {spec}")
        if not _parse_date(test_start) < _parse_date(test_end):
            raise ValueError(f"Invalid test window order: {spec}")
        return WalkForwardWindow(
            index=index,
            timerange=f"{train_start}-{test_end}",
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    parts = spec.split("-")
    if len(parts) != 2:
        raise ValueError("Fixed windows must use START-END, for example 20250101-20250107.")
    start, end = [_date_token(part) for part in parts]
    if not _parse_date(start) < _parse_date(end):
        raise ValueError(f"Invalid fixed window order: {spec}")
    return WalkForwardWindow(
        index=index,
        timerange=f"{start}-{end}",
        test_start=start,
        test_end=end,
    )


def generate_rolling_windows(
    *,
    start: str,
    end: str,
    train_days: int,
    test_days: int,
    step_days: int,
) -> list[WalkForwardWindow]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, and step_days must be positive.")

    start_date = _parse_date(_date_token(start))
    end_date = _parse_date(_date_token(end))
    windows: list[WalkForwardWindow] = []
    train_start = start_date
    index = 1

    while True:
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > end_date:
            break
        windows.append(
            WalkForwardWindow(
                index=index,
                timerange=f"{_fmt_date(train_start)}-{_fmt_date(test_end)}",
                train_start=_fmt_date(train_start),
                train_end=_fmt_date(train_end),
                test_start=_fmt_date(test_start),
                test_end=_fmt_date(test_end),
            )
        )
        train_start += timedelta(days=step_days)
        index += 1

    return windows


def window_run_id(prefix: str, window: WalkForwardWindow) -> str:
    if window.train_start and window.train_end and window.test_start and window.test_end:
        suffix = (
            f"train_{window.train_start}_{window.train_end}_"
            f"test_{window.test_start}_{window.test_end}"
        )
    else:
        suffix = window.timerange.replace("-", "_")
    return f"{prefix}_{window.index:02d}_{suffix}"


def aggregate_walk_forward_results(
    window_results: Sequence[dict[str, Any]],
    rules: WalkForwardRules | None = None,
) -> dict[str, Any]:
    rules = rules or WalkForwardRules()
    total_windows = len(window_results)
    completed = [
        result
        for result in window_results
        if result.get("status") == "completed" and isinstance(result.get("metrics"), dict)
    ]
    failed = total_windows - len(completed)
    pass_count = sum(1 for result in completed if result.get("gate_recommendation") == "pass")
    profitable = [
        result
        for result in completed
        if _metric_float(result["metrics"], "total_return") is not None
        and _metric_float(result["metrics"], "total_return") > 0
    ]

    pass_rate = (pass_count / len(completed)) if completed else 0.0
    profitable_ratio = (len(profitable) / len(completed)) if completed else 0.0
    total_return = sum(_metric_float(result["metrics"], "total_return") or 0.0 for result in completed)
    total_return_pct = sum(
        _metric_float(result["metrics"], "total_return_pct") or 0.0 for result in completed
    )
    drawdowns = [
        value
        for value in (_metric_float(result["metrics"], "max_drawdown_pct") for result in completed)
        if value is not None
    ]
    max_drawdown = max(drawdowns) if drawdowns else None
    positive_returns = [
        _metric_float(result["metrics"], "total_return") or 0.0 for result in profitable
    ]
    profit_dependency = _single_window_profit_dependency(positive_returns)

    checks = [
        {
            "name": "all_windows_completed",
            "actual": f"{len(completed)}/{total_windows}",
            "rule": "all windows complete",
            "pass": failed == 0 and total_windows > 0,
        },
        {
            "name": "min_pass_rate",
            "actual": pass_rate,
            "rule": f">= {rules.min_pass_rate}",
            "pass": pass_rate >= rules.min_pass_rate,
        },
        {
            "name": "min_profitable_windows_ratio",
            "actual": profitable_ratio,
            "rule": f">= {rules.min_profitable_windows_ratio}",
            "pass": profitable_ratio >= rules.min_profitable_windows_ratio,
        },
        {
            "name": "max_drawdown_pct_any_window",
            "actual": max_drawdown,
            "rule": f"<= {rules.max_drawdown_pct_any_window}",
            "pass": max_drawdown is not None
            and max_drawdown <= rules.max_drawdown_pct_any_window,
        },
        {
            "name": "max_single_window_profit_dependency",
            "actual": profit_dependency,
            "rule": f"<= {rules.max_single_window_profit_dependency}",
            "pass": profit_dependency is not None
            and profit_dependency <= rules.max_single_window_profit_dependency,
        },
    ]
    recommendation = "pass" if all(check["pass"] for check in checks) else "fail"

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "2",
        "status": "completed" if failed == 0 else "completed_with_failed_windows",
        "recommendation": recommendation,
        "rules": asdict(rules),
        "summary": {
            "window_count": total_windows,
            "completed_windows": len(completed),
            "failed_windows": failed,
            "pass_rate": pass_rate,
            "profitable_windows_ratio": profitable_ratio,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct_any_window": max_drawdown,
            "max_single_window_profit_dependency": profit_dependency,
        },
        "checks": checks,
        "windows": list(window_results),
        "safety_scope": {
            "command": "freqtrade backtesting only",
            "paper_trading": False,
            "dry_run_trading": False,
            "live_trading": False,
            "exchange_order_placement": False,
        },
    }


def write_walk_forward_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def write_walk_forward_report(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = metrics["summary"]
    lines = [
        "# Walk-Forward Report",
        "",
        "## Summary",
        "",
        f"- Recommendation: {metrics['recommendation']}",
        f"- Windows: {summary['completed_windows']}/{summary['window_count']} completed",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Profitable windows ratio: {summary['profitable_windows_ratio']:.2%}",
        f"- Total return: {summary['total_return']:.6f} ({summary['total_return_pct']:.2f}%)",
        f"- Max drawdown in any window: {_fmt(summary['max_drawdown_pct_any_window'])}%",
        "- Max single-window profit dependency: "
        f"{_fmt(summary['max_single_window_profit_dependency'])}",
        "",
        "## Gate Checks",
        "",
    ]
    for check in metrics["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(f"- {status}: {check['name']} ({_fmt(check['actual'])} vs {check['rule']})")

    lines.extend(["", "## Windows", ""])
    for result in metrics["windows"]:
        window = result["window"]
        metrics_payload = result.get("metrics") or {}
        lines.append(
            "- Window {index}: {timerange} | status={status} | gate={gate} | "
            "return={return_pct}% | trades={trades}".format(
                index=window["index"],
                timerange=window["timerange"],
                status=result.get("status"),
                gate=result.get("gate_recommendation", "n/a"),
                return_pct=_fmt(metrics_payload.get("total_return_pct")),
                trades=_fmt(metrics_payload.get("trade_count")),
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report is generated from historical backtests only.",
            "- Passing walk-forward gates does not authorize paper trading or live trading.",
            "- FreqAI labels are backtest labels, not live trading instructions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _date_token(value: str) -> str:
    token = value.strip().replace("-", "")
    _parse_date(token)
    return token


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _fmt_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _single_window_profit_dependency(positive_returns: Sequence[float]) -> float | None:
    total_positive = sum(positive_returns)
    if total_positive <= 0:
        return None
    return max(positive_returns) / total_positive


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
