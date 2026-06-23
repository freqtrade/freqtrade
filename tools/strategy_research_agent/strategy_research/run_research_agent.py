#!/usr/bin/env python3
"""Run the local strategy researcher agent.

This orchestrator is intentionally research-only:
- it never reads live API keys,
- it never modifies Freqtrade live/dry-run config,
- it never runs external source code directly.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from run_strategy_research import BacktestMetrics, classify, metrics_to_dict, parse_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DEFAULT_CONFIG = AGENT_ROOT / "agent_config.json"
DEFAULT_REGISTRY = AGENT_ROOT / "strategy_registry.json"
DEFAULT_EXPERIMENT = AGENT_ROOT / "experiments/btc_eth_futures_core_matrix.json"
GENERATED_VARIANT_REGISTRY = AGENT_ROOT / "experiments/generated_variant_registry.json"
SOURCE_TRANSLATED_REGISTRY = AGENT_ROOT / "experiments/source_translated_registry.json"
AUTONOMOUS_STRATEGY_REGISTRY = AGENT_ROOT / "experiments/autonomous_strategy_registry.json"
ITERATIVE_STRATEGY_REGISTRY = AGENT_ROOT / "experiments/iterative_strategy_registry.json"
LOOKAHEAD_CONFIG_OVERRIDE = AGENT_ROOT / "config_lookahead_pricing_override.json"


@dataclass
class DataAudit:
    pair: str
    timeframe: str
    market_type: str
    path: str
    exists: bool
    rows: int | None = None
    first_utc: str | None = None
    last_utc: str | None = None
    expected_step_seconds: int | None = None
    gaps: int | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--strategy", action="append", help="Run only the named strategy. Repeatable.")
    parser.add_argument("--timerange", help="Override experiment timerange.")
    parser.add_argument("--timeframe", help="Override experiment timeframe.")
    parser.add_argument("--fee", type=float, help="Override experiment fee.")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands and write a report.")
    parser.add_argument("--skip-backtests", action="store_true", help="Only audit data and refresh dashboard.")
    parser.add_argument("--run-recursive", action="store_true", help="Run recursive-analysis after backtesting.")
    parser.add_argument("--run-lookahead", action="store_true", help="Run lookahead-analysis after backtesting.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_workspace(config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, value in config["workspace"].items():
        path = REPO_ROOT / value
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    return paths


def timeframe_seconds(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def pair_data_path(pair: str, timeframe: str) -> tuple[str, Path]:
    if ":" in pair:
        stem = pair.replace("/", "_").replace(":", "_")
        return "futures", REPO_ROOT / f"user_data/data/binance/futures/{stem}-{timeframe}-futures.feather"
    stem = pair.replace("/", "_")
    return "spot", REPO_ROOT / f"user_data/data/binance/{stem}-{timeframe}.feather"


def audit_one_dataset(pair: str, timeframe: str) -> DataAudit:
    market_type, path = pair_data_path(pair, timeframe)
    rel_path = str(path.relative_to(REPO_ROOT))
    if not path.exists():
        return DataAudit(pair, timeframe, market_type, rel_path, exists=False)

    try:
        frame = pd.read_feather(path)
        date_col = "date" if "date" in frame.columns else frame.columns[0]
        dates = pd.to_datetime(frame[date_col], utc=True).sort_values()
        expected = timeframe_seconds(timeframe)
        deltas = dates.diff().dt.total_seconds().dropna()
        gaps = int((deltas > expected).sum())
        return DataAudit(
            pair=pair,
            timeframe=timeframe,
            market_type=market_type,
            path=rel_path,
            exists=True,
            rows=int(len(frame)),
            first_utc=dates.iloc[0].isoformat() if len(dates) else None,
            last_utc=dates.iloc[-1].isoformat() if len(dates) else None,
            expected_step_seconds=expected,
            gaps=gaps,
        )
    except Exception as exc:  # noqa: BLE001 - report data audit failures without hiding them.
        return DataAudit(pair, timeframe, market_type, rel_path, exists=True, error=str(exc))


def audit_data(profile: dict[str, Any], timeframes: list[str]) -> list[DataAudit]:
    audits: list[DataAudit] = []
    for pair in profile.get("pairs", []):
        for timeframe in timeframes:
            audits.append(audit_one_dataset(pair, timeframe))
    return audits


def research_env() -> dict[str, str]:
    env = os.environ.copy()
    offline_path = str(REPO_ROOT / "user_data/offline_exchange")
    env["PYTHONPATH"] = (
        offline_path
        if not env.get("PYTHONPATH")
        else f"{offline_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    return env


def freqtrade_command(
    command_name: str,
    config_path: str,
    strategy: str,
    timeframe: str,
    timerange: str,
    fee: float,
    strategy_path: str | None = None,
) -> list[str]:
    command = [
        str(REPO_ROOT / ".venv/bin/freqtrade"),
        command_name,
        "-c",
        config_path,
        "--strategy",
        strategy,
        "--timeframe",
        timeframe,
        "--timerange",
        timerange,
        "--fee",
        str(fee),
    ]
    if strategy_path:
        command.extend(["--strategy-path", strategy_path])
    return command


def run_backtest_command(
    config_path: str,
    strategy: str,
    timeframe: str,
    timerange: str,
    fee: float,
    dry_run: bool,
    strategy_path: str | None,
) -> BacktestMetrics:
    command = freqtrade_command(
        "backtesting",
        config_path,
        strategy,
        timeframe,
        timerange,
        fee,
        strategy_path,
    )
    command.extend(["--cache", "none", "--export", "none"])
    if dry_run:
        return BacktestMetrics(strategy=strategy, status="dry_run", command=command)

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=research_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lowered = completed.stdout.lower()
    if completed.returncode != 0 or "configuration error" in lowered or " - error -" in lowered:
        return BacktestMetrics(
            strategy=strategy,
            status="failed",
            command=command,
            error=completed.stdout[-4000:],
        )
    return parse_metrics(strategy, completed.stdout, command)


def run_analysis_command(
    command_name: str,
    config_path: str,
    strategy: str,
    timeframe: str,
    timerange: str,
    fee: float,
    dry_run: bool,
    strategy_path: str | None,
) -> dict[str, Any]:
    command = freqtrade_command(
        command_name,
        config_path,
        strategy,
        timeframe,
        timerange,
        fee,
        strategy_path,
    )
    if command_name == "recursive-analysis":
        fee_index = command.index("--fee")
        del command[fee_index : fee_index + 2]
    if command_name == "lookahead-analysis":
        config_index = command.index("-c")
        command[config_index + 2 : config_index + 2] = ["-c", str(LOOKAHEAD_CONFIG_OVERRIDE.relative_to(REPO_ROOT))]
        command.append("--allow-limit-orders")

    if dry_run:
        return {"status": "dry_run", "command": command}

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=research_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_tail = completed.stdout[-4000:]
    lowered = completed.stdout.lower()
    status = "ok" if completed.returncode == 0 else "failed"
    if "configuration error" in lowered or " - error -" in lowered:
        status = "failed"
    if completed.returncode == 0 and "no bias detected" in lowered:
        status = "ok"
    elif completed.returncode == 0 and (
        "bias detected" in lowered
        or "has bias" in lowered
        or "failed" in lowered
    ):
        status = "needs_review"
    return {
        "status": status,
        "command": command,
        "returncode": completed.returncode,
        "output_tail": output_tail,
    }


def strategy_metadata(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = {item["name"]: item for item in registry.get("strategies", [])}
    for path in [
        GENERATED_VARIANT_REGISTRY,
        SOURCE_TRANSLATED_REGISTRY,
        AUTONOMOUS_STRATEGY_REGISTRY,
        ITERATIVE_STRATEGY_REGISTRY,
    ]:
        if path.exists():
            generated = load_json(path)
            metadata.update({item["name"]: item for item in generated.get("strategies", [])})
    return metadata


def selected_strategies(experiment: dict[str, Any], args: argparse.Namespace) -> list[str]:
    names = args.strategy or experiment["strategies"]
    return list(dict.fromkeys(names))


def selected_timeranges(experiment: dict[str, Any], args: argparse.Namespace, default_timerange: str) -> list[dict[str, Any]]:
    if args.timerange:
        return [{"name": "override", "label": "CLI override", "timerange": args.timerange}]
    matrix = experiment.get("matrix", {})
    if matrix.get("timeranges"):
        return matrix["timeranges"]
    return [{"name": "full", "label": "Full sample", "timerange": experiment.get("timeranges", [default_timerange])[0]}]


def write_candidate_files(paths: dict[str, Path], result: dict[str, Any]) -> None:
    classification = result["classification"]
    candidate_path = paths["candidate_dir"] / f"{result['strategy']}.json"
    watchlist_path = paths["watchlist_dir"] / f"{result['strategy']}.json"
    rejected_path = paths["rejected_dir"] / f"{result['strategy']}.json"

    if classification in {"research_candidate", "dryrun_candidate"}:
        target = candidate_path
        if watchlist_path.exists():
            watchlist_path.unlink()
        if rejected_path.exists():
            rejected_path.unlink()
    elif classification in {"needs_more_data", "needs_review"}:
        target = watchlist_path
        if candidate_path.exists():
            candidate_path.unlink()
        if rejected_path.exists():
            rejected_path.unlink()
    elif classification == "rejected":
        target = rejected_path
        if candidate_path.exists():
            candidate_path.unlink()
        if watchlist_path.exists():
            watchlist_path.unlink()
    else:
        if candidate_path.exists():
            candidate_path.unlink()
        if watchlist_path.exists():
            watchlist_path.unlink()
        if rejected_path.exists():
            rejected_path.unlink()
        return
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_reports(report_dir: Path, limit: int = 20) -> list[Path]:
    return sorted(report_dir.glob("agent_research_*.json"), reverse=True)[:limit]


def check_status(value: Any) -> str:
    if not value:
        return ""
    return str(value.get("status") or "")


def load_pool(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item_path in sorted(path.glob("*.json")):
        try:
            item = load_json(item_path)
        except json.JSONDecodeError:
            continue
        item["_path"] = str(item_path.relative_to(REPO_ROOT))
        items.append(item)
    return items


def load_source_reviews() -> list[dict[str, Any]]:
    review_dir = AGENT_ROOT / "sources/reviews"
    reviews: list[dict[str, Any]] = []
    for path in sorted(review_dir.glob("*.review.json")):
        try:
            review = load_json(path)
        except json.JSONDecodeError:
            continue
        review["_path"] = str(path.relative_to(REPO_ROOT))
        reviews.append(review)
    return reviews


def load_latest_matrix_summary() -> dict[str, Any] | None:
    path = AGENT_ROOT / "matrix_summaries/latest_matrix_summary.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_cost_audit() -> dict[str, Any] | None:
    path = AGENT_ROOT / "cost_audits/latest_futures_cost_audit.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_trade_cost_estimate() -> dict[str, Any] | None:
    path = AGENT_ROOT / "cost_adjustments/latest_trade_cost_estimate.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_aux_conversion() -> dict[str, Any] | None:
    path = AGENT_ROOT / "cost_audits/latest_freqtrade_futures_aux_conversion.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_ohlcv_update() -> dict[str, Any] | None:
    path = AGENT_ROOT / "data_updates/latest_ohlcv_1m_update.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_strategy_assessment() -> dict[str, Any] | None:
    path = AGENT_ROOT / "strategy_assessments/latest_strategy_assessment.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_walk_forward_summary() -> dict[str, Any] | None:
    path = AGENT_ROOT / "walk_forward_summaries/latest_walk_forward_summary.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_promotion_report() -> dict[str, Any] | None:
    path = AGENT_ROOT / "promotion_reports/latest_promotion_report.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_research_agenda() -> dict[str, Any] | None:
    path = AGENT_ROOT / "research_agendas/latest_research_agenda.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_agenda_run() -> dict[str, Any] | None:
    path = AGENT_ROOT / "agenda_runs/latest_agenda_run.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_trade_behavior() -> dict[str, Any] | None:
    path = AGENT_ROOT / "trade_behavior/latest_trade_behavior.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def load_latest_behavior_experiment_plan() -> dict[str, Any] | None:
    path = AGENT_ROOT / "behavior_experiments/latest_behavior_experiment_plan.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except json.JSONDecodeError:
        return None
    payload["_path"] = str(path.relative_to(REPO_ROOT))
    return payload


def render_dashboard(paths: dict[str, Path], payload: dict[str, Any]) -> Path:
    dashboard_path = paths["dashboard_dir"] / "index.html"
    rows = []
    for item in payload["results"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['strategy'])}</td>"
            f"<td>{html.escape(item.get('regime', ''))}</td>"
            f"<td>{html.escape(item['classification'])}</td>"
            f"<td>{item.get('trades')}</td>"
            f"<td>{item.get('total_profit_pct')}</td>"
            f"<td>{item.get('max_drawdown_pct')}</td>"
            f"<td>{item.get('profit_factor')}</td>"
            f"<td>{html.escape('; '.join(item.get('reasons', [])))}</td>"
            "</tr>"
        )
    audit_rows = []
    for item in payload["data_audit"]:
        audit_rows.append(
            "<tr>"
            f"<td>{html.escape(item['pair'])}</td>"
            f"<td>{html.escape(item['timeframe'])}</td>"
            f"<td>{html.escape(item['market_type'])}</td>"
            f"<td>{item['exists']}</td>"
            f"<td>{item.get('rows')}</td>"
            f"<td>{item.get('gaps')}</td>"
            f"<td>{html.escape(item.get('first_utc') or '')}</td>"
            f"<td>{html.escape(item.get('last_utc') or '')}</td>"
            "</tr>"
        )
    ohlcv_update_rows = []
    ohlcv_update = payload.get("ohlcv_update") or {}
    for item in ohlcv_update.get("results", []):
        ohlcv_update_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('symbol', ''))}</td>"
            f"<td>{html.escape(item.get('status', ''))}</td>"
            f"<td>{item.get('rows')}</td>"
            f"<td>{html.escape(item.get('first_utc') or '')}</td>"
            f"<td>{html.escape(item.get('last_utc') or '')}</td>"
            f"<td>{item.get('gaps')}</td>"
            f"<td>{item.get('archives')}</td>"
            "</tr>"
        )
    candidate_rows = []
    for item in payload.get("candidate_pool", []):
        candidate_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', item.get('name', '')))}</td>"
            f"<td>{html.escape(item.get('classification', ''))}</td>"
            f"<td>{item.get('total_profit_pct')}</td>"
            f"<td>{item.get('max_drawdown_pct')}</td>"
            f"<td>{item.get('profit_factor')}</td>"
            f"<td>{html.escape(item.get('_path', ''))}</td>"
            "</tr>"
        )
    rejected_rows = []
    for item in payload.get("rejected_pool", []):
        rejected_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', item.get('name', '')))}</td>"
            f"<td>{html.escape('; '.join(item.get('reasons', [])))}</td>"
            f"<td>{item.get('total_profit_pct')}</td>"
            f"<td>{html.escape(item.get('_path', ''))}</td>"
            "</tr>"
        )
    watchlist_rows = []
    for item in payload.get("watchlist_pool", []):
        watchlist_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', item.get('name', '')))}</td>"
            f"<td>{html.escape(item.get('classification', ''))}</td>"
            f"<td>{item.get('total_profit_pct')}</td>"
            f"<td>{item.get('max_drawdown_pct')}</td>"
            f"<td>{item.get('profit_factor')}</td>"
            f"<td>{html.escape('; '.join(item.get('reasons', [])))}</td>"
            "</tr>"
        )
    source_rows = []
    for item in payload.get("source_reviews", []):
        source = item.get("source", {})
        source_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('id', ''))}</td>"
            f"<td>{html.escape(item.get('status', ''))}</td>"
            f"<td>{html.escape(item.get('inferred_strategy_family', ''))}</td>"
            f"<td>{html.escape(', '.join(item.get('detected_indicators', [])))}</td>"
            f"<td>{html.escape(source.get('title') or '')}</td>"
            "</tr>"
        )
    matrix_rows = []
    matrix_summary = payload.get("matrix_summary") or {}
    for item in matrix_summary.get("strategy_summary", []):
        matrix_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{html.escape(item.get('verdict', ''))}</td>"
            f"<td>{item.get('runs')}</td>"
            f"<td>{item.get('positive_runs')}</td>"
            f"<td>{item.get('too_few_trade_runs')}</td>"
            f"<td>{item.get('stress_negative_runs')}</td>"
            f"<td>{item.get('min_return_pct')}</td>"
            f"<td>{item.get('max_return_pct')}</td>"
            f"<td>{item.get('min_profit_factor')}</td>"
            "</tr>"
        )
    funding_rows = []
    mark_rows = []
    cost_audit = payload.get("cost_audit") or {}
    for item in cost_audit.get("funding", []):
        funding_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('pair', ''))}</td>"
            f"<td>{item.get('rows')}</td>"
            f"<td>{html.escape(item.get('first_utc') or '')}</td>"
            f"<td>{html.escape(item.get('last_utc') or '')}</td>"
            f"<td>{item.get('gaps')}</td>"
            f"<td>{item.get('mean_rate_pct')}</td>"
            f"<td>{item.get('sum_rate_pct')}</td>"
            "</tr>"
        )
    for item in cost_audit.get("mark_price", []):
        mark_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('pair', ''))}</td>"
            f"<td>{item.get('rows')}</td>"
            f"<td>{item.get('gaps')}</td>"
            f"<td>{item.get('mean_abs_basis_bps')}</td>"
            f"<td>{item.get('p95_abs_basis_bps')}</td>"
            f"<td>{item.get('max_abs_basis_bps')}</td>"
            "</tr>"
        )
    aux_conversion_rows = []
    aux_conversion = payload.get("aux_conversion") or {}
    for item in aux_conversion.get("converted_files", []):
        aux_conversion_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('pair', ''))}</td>"
            f"<td>{html.escape(item.get('candle_type', ''))}</td>"
            f"<td>{html.escape(item.get('timeframe', ''))}</td>"
            f"<td>{item.get('rows')}</td>"
            f"<td>{html.escape(item.get('first_utc') or '')}</td>"
            f"<td>{html.escape(item.get('last_utc') or '')}</td>"
            "</tr>"
        )
    trade_cost_rows = []
    trade_cost_estimate = payload.get("trade_cost_estimate") or {}
    for item in trade_cost_estimate.get("estimates", []):
        trade_cost_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{item.get('trades')}</td>"
            f"<td>{item.get('base_profit_pct')}</td>"
            f"<td>{item.get('funding_pct_of_start_balance')}</td>"
            f"<td>{item.get('slippage_pct_of_start_balance')}</td>"
            f"<td>{item.get('adjusted_profit_pct')}</td>"
            f"<td>{item.get('funding_events')}</td>"
            f"<td>{item.get('funding_coverage_missing_trades')}</td>"
            "</tr>"
        )
    scorecard_rows = []
    failure_rows = []
    assessment = payload.get("strategy_assessment") or {}
    for item in assessment.get("scorecards", []):
        scorecard_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{html.escape(item.get('tier', ''))}</td>"
            f"<td>{item.get('score')}</td>"
            f"<td>{item.get('base_return_pct')}</td>"
            f"<td>{item.get('adjusted_return_pct')}</td>"
            f"<td>{item.get('profit_factor')}</td>"
            f"<td>{item.get('max_drawdown_pct')}</td>"
            f"<td>{item.get('trades')}</td>"
            f"<td>{html.escape(', '.join(item.get('primary_failures', [])))}</td>"
            "</tr>"
        )
    for item in assessment.get("failure_summary", []):
        failure_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('failure', ''))}</td>"
            f"<td>{item.get('count')}</td>"
            "</tr>"
        )
    walk_forward_rows = []
    walk_forward = payload.get("walk_forward_summary") or {}
    for item in walk_forward.get("strategy_summary", []):
        walk_forward_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{html.escape(item.get('verdict', ''))}</td>"
            f"<td>{item.get('windows')}</td>"
            f"<td>{item.get('positive_windows')}</td>"
            f"<td>{item.get('negative_windows')}</td>"
            f"<td>{item.get('total_trades')}</td>"
            f"<td>{item.get('median_return_pct')}</td>"
            f"<td>{item.get('median_profit_factor')}</td>"
            f"<td>{item.get('worst_drawdown_pct')}</td>"
            f"<td>{html.escape(', '.join(item.get('reasons', [])))}</td>"
            "</tr>"
        )
    promotion_rows = []
    promotion_report = payload.get("promotion_report") or {}
    for item in promotion_report.get("verdicts", []):
        promotion_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{html.escape(item.get('verdict', ''))}</td>"
            f"<td>{item.get('ready_for_manual_dryrun_review')}</td>"
            f"<td>{html.escape(', '.join(item.get('blocks', [])))}</td>"
            f"<td>{html.escape('; '.join(item.get('next_actions', [])))}</td>"
            "</tr>"
        )
    agenda_rows = []
    research_agenda = payload.get("research_agenda") or {}
    for item in research_agenda.get("top_priorities", []):
        agenda_rows.append(
            "<tr>"
            f"<td>{item.get('priority')}</td>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{html.escape(item.get('blocker', ''))}</td>"
            f"<td>{html.escape(item.get('objective', ''))}</td>"
            f"<td><code>{html.escape(item.get('next_command', ''))}</code></td>"
            f"<td>{html.escape(item.get('success_gate', ''))}</td>"
            "</tr>"
        )
    agenda_run = payload.get("agenda_run") or {}
    agenda_run_item = agenda_run.get("selected_item") or {}
    agenda_run_rows = []
    if agenda_run:
        agenda_run_rows.append(
            "<tr>"
            f"<td>{html.escape(agenda_run.get('status', ''))}</td>"
            f"<td>{html.escape(agenda_run.get('mode', ''))}</td>"
            f"<td>{html.escape(agenda_run_item.get('strategy', ''))}</td>"
            f"<td>{html.escape(agenda_run_item.get('blocker', ''))}</td>"
            f"<td><code>{html.escape(agenda_run.get('command') or '')}</code></td>"
            f"<td>{agenda_run.get('returncode')}</td>"
            "</tr>"
        )
    trade_behavior_rows = []
    trade_behavior = payload.get("trade_behavior") or {}
    for item in trade_behavior.get("summaries", []):
        trade_behavior_rows.append(
            "<tr>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{item.get('trades')}</td>"
            f"<td>{item.get('win_rate_pct')}</td>"
            f"<td>{item.get('total_profit_abs')}</td>"
            f"<td>{item.get('profit_factor')}</td>"
            f"<td>{item.get('payoff_ratio')}</td>"
            f"<td>{item.get('avg_duration_min')}</td>"
            f"<td>{item.get('long_trades')}/{item.get('short_trades')}</td>"
            f"<td>{item.get('stop_loss_trades')}</td>"
            f"<td>{item.get('avg_mfe_pct')}</td>"
            f"<td>{item.get('avg_mae_pct')}</td>"
            f"<td>{html.escape('; '.join(item.get('diagnostics', [])))}</td>"
            "</tr>"
        )
    behavior_experiment_rows = []
    behavior_experiments = payload.get("behavior_experiments") or {}
    for item in behavior_experiments.get("plans", []):
        behavior_experiment_rows.append(
            "<tr>"
            f"<td>{item.get('priority')}</td>"
            f"<td>{html.escape(item.get('strategy', ''))}</td>"
            f"<td>{html.escape(item.get('experiment_id', ''))}</td>"
            f"<td>{html.escape(item.get('hypothesis', ''))}</td>"
            f"<td>{html.escape(item.get('success_gate', ''))}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strategy Research Agent</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f7f8fa; }}
    header {{ padding: 24px 32px 16px; background: #ffffff; border-bottom: 1px solid #e5e7eb; }}
    main {{ padding: 24px 32px 40px; display: grid; gap: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    .meta {{ color: #5b6472; font-size: 14px; }}
    section {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #eef0f3; text-align: left; white-space: nowrap; }}
    th {{ color: #344054; background: #fafafa; font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <h1>Strategy Research Agent</h1>
    <div class="meta">Generated UTC: {html.escape(payload['generated_at_utc'])} | Experiment: {html.escape(payload['experiment']['id'])}</div>
  </header>
  <main>
    <section>
      <h2>策略结果</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Regime</th><th>Class</th><th>Trades</th><th>Return %</th><th>DD %</th><th>PF</th><th>Notes</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>数据检查</h2>
      <table>
        <thead><tr><th>Pair</th><th>TF</th><th>Market</th><th>Exists</th><th>Rows</th><th>Gaps</th><th>First</th><th>Last</th></tr></thead>
        <tbody>{''.join(audit_rows)}</tbody>
      </table>
      <table>
        <thead><tr><th>Symbol</th><th>Update</th><th>Rows</th><th>First</th><th>Last</th><th>Gaps</th><th>Archives</th></tr></thead>
        <tbody>{''.join(ohlcv_update_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>候选池</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Class</th><th>Return %</th><th>DD %</th><th>PF</th><th>Path</th></tr></thead>
        <tbody>{''.join(candidate_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>策略评分与失败归因</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Tier</th><th>Score</th><th>Base %</th><th>Adjusted %</th><th>PF</th><th>DD %</th><th>Trades</th><th>Failures</th></tr></thead>
        <tbody>{''.join(scorecard_rows)}</tbody>
      </table>
      <table>
        <thead><tr><th>Failure</th><th>Count</th></tr></thead>
        <tbody>{''.join(failure_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Walk-Forward 稳健性</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Verdict</th><th>Windows</th><th>Positive</th><th>Negative</th><th>Trades</th><th>Median %</th><th>Median PF</th><th>Worst DD %</th><th>Reasons</th></tr></thead>
        <tbody>{''.join(walk_forward_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>晋级闸门</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Verdict</th><th>Ready</th><th>Blocks</th><th>Next Actions</th></tr></thead>
        <tbody>{''.join(promotion_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>研究议程</h2>
      <table>
        <thead><tr><th>Priority</th><th>Strategy</th><th>Blocker</th><th>Objective</th><th>Next Command</th><th>Success Gate</th></tr></thead>
        <tbody>{''.join(agenda_rows)}</tbody>
      </table>
      <table>
        <thead><tr><th>Status</th><th>Mode</th><th>Strategy</th><th>Blocker</th><th>Command</th><th>Return Code</th></tr></thead>
        <tbody>{''.join(agenda_run_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>交易行为分析</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Trades</th><th>Win %</th><th>Profit Abs</th><th>PF</th><th>Payoff</th><th>Avg Dur</th><th>Long/Short</th><th>Stop Losses</th><th>MFE %</th><th>MAE %</th><th>Diagnostics</th></tr></thead>
        <tbody>{''.join(trade_behavior_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>行为驱动实验计划</h2>
      <table>
        <thead><tr><th>Priority</th><th>Strategy</th><th>Experiment</th><th>Hypothesis</th><th>Success Gate</th></tr></thead>
        <tbody>{''.join(behavior_experiment_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>淘汰池</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Reason</th><th>Return %</th><th>Path</th></tr></thead>
        <tbody>{''.join(rejected_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>观察池</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Class</th><th>Return %</th><th>DD %</th><th>PF</th><th>Notes</th></tr></thead>
        <tbody>{''.join(watchlist_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>来源审查</h2>
      <table>
        <thead><tr><th>Source</th><th>Status</th><th>Family</th><th>Indicators</th><th>Title</th></tr></thead>
        <tbody>{''.join(source_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>市场状态与成本韧性</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Verdict</th><th>Runs</th><th>Positive</th><th>Too Few</th><th>Stress Negative</th><th>Min Return %</th><th>Max Return %</th><th>Min PF</th></tr></thead>
        <tbody>{''.join(matrix_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>合约成本数据</h2>
      <table>
        <thead><tr><th>Funding Pair</th><th>Rows</th><th>First</th><th>Last</th><th>Gaps</th><th>Mean %</th><th>Sum %</th></tr></thead>
        <tbody>{''.join(funding_rows)}</tbody>
      </table>
      <table>
        <thead><tr><th>Mark Pair</th><th>Rows</th><th>Gaps</th><th>Mean abs bps</th><th>P95 abs bps</th><th>Max abs bps</th></tr></thead>
        <tbody>{''.join(mark_rows)}</tbody>
      </table>
      <table>
        <thead><tr><th>Pair</th><th>Candle Type</th><th>Timeframe</th><th>Rows</th><th>First</th><th>Last</th></tr></thead>
        <tbody>{''.join(aux_conversion_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>策略级成本校正</h2>
      <table>
        <thead><tr><th>Strategy</th><th>Trades</th><th>Base %</th><th>Funding %</th><th>Slippage %</th><th>Adjusted %</th><th>Funding Events</th><th>Coverage Missing</th></tr></thead>
        <tbody>{''.join(trade_cost_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    dashboard_path.write_text(page, encoding="utf-8")
    return dashboard_path


def write_report(paths: dict[str, Path], payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    report_dir = paths["report_dir"]
    timestamp = payload["generated_at_utc"]
    json_path = report_dir / f"agent_research_{timestamp}.json"
    md_path = report_dir / f"agent_research_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Strategy Research Agent Report",
        "",
        f"- Generated UTC: `{timestamp}`",
        f"- Experiment: `{payload['experiment']['id']}`",
        f"- Timerange: `{payload['run']['timerange']}`",
        f"- Timeframe: `{payload['run']['timeframe']}`",
        f"- Fee: `{payload['run']['fee']}`",
        f"- Dry run: `{payload['run']['dry_run']}`",
        "",
        "## Strategy Results",
        "",
        "| Strategy | Regime | Timerange | Class | Trades | Return | DD | PF | Recursive | Lookahead | Notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["results"]:
        lines.append(
            "| {strategy} | {regime} | {timerange} | {classification} | {trades} | {ret}% | {dd}% | {pf} | {recursive} | {lookahead} | {notes} |".format(
                strategy=item["strategy"],
                regime=item.get("regime", ""),
                timerange=item.get("timerange", ""),
                classification=item["classification"],
                trades=item.get("trades"),
                ret=item.get("total_profit_pct"),
                dd=item.get("max_drawdown_pct"),
                pf=item.get("profit_factor"),
                recursive=check_status(item.get("recursive_analysis")),
                lookahead=check_status(item.get("lookahead_analysis")),
                notes="; ".join(item.get("reasons", [])),
            )
        )
    lines.extend(["", "## Data Audit", "", "| Pair | TF | Type | Exists | Rows | Gaps | First | Last |", "|---|---:|---|---:|---:|---:|---|---|"])
    for item in payload["data_audit"]:
        lines.append(
            "| {pair} | {timeframe} | {market_type} | {exists} | {rows} | {gaps} | {first} | {last} |".format(
                pair=item["pair"],
                timeframe=item["timeframe"],
                market_type=item["market_type"],
                exists=item["exists"],
                rows=item.get("rows"),
                gaps=item.get("gaps"),
                first=item.get("first_utc"),
                last=item.get("last_utc"),
            )
        )
    ohlcv_update = payload.get("ohlcv_update") or {}
    if ohlcv_update.get("results"):
        lines.extend(
            [
                "",
                "## OHLCV 1m Update",
                "",
                "| Symbol | Status | Rows | First | Last | Gaps | Archives |",
                "|---|---|---:|---|---|---:|---:|",
            ]
        )
        for item in ohlcv_update["results"]:
            lines.append(
                "| {symbol} | {status} | {rows} | {first_utc} | {last_utc} | {gaps} | {archives} |".format(
                    **item
                )
            )
    if payload.get("experiment", {}).get("cost_model"):
        cost = payload["experiment"]["cost_model"]
        lines.extend(
            [
                "",
                "## Cost Model",
                "",
                f"- Fee: `{cost.get('fee')}`",
                f"- Slippage included: `{cost.get('slippage_included')}`",
                f"- Funding included: `{cost.get('funding_included')}`",
                f"- Mark price included: `{cost.get('mark_price_included')}`",
            ]
        )
        for note in cost.get("notes", []):
            lines.append(f"- {note}")
    lines.extend(["", "## Candidate Pool", "", "| Strategy | Class | Return | DD | PF |", "|---|---:|---:|---:|---:|"])
    for item in payload.get("candidate_pool", []):
        lines.append(
            "| {strategy} | {classification} | {ret}% | {dd}% | {pf} |".format(
                strategy=item.get("strategy", item.get("name")),
                classification=item.get("classification"),
                ret=item.get("total_profit_pct"),
                dd=item.get("max_drawdown_pct"),
                pf=item.get("profit_factor"),
            )
        )
    lines.extend(["", "## Watchlist", "", "| Strategy | Class | Return | DD | PF | Notes |", "|---|---:|---:|---:|---:|---|"])
    for item in payload.get("watchlist_pool", []):
        lines.append(
            "| {strategy} | {classification} | {ret}% | {dd}% | {pf} | {notes} |".format(
                strategy=item.get("strategy", item.get("name")),
                classification=item.get("classification"),
                ret=item.get("total_profit_pct"),
                dd=item.get("max_drawdown_pct"),
                pf=item.get("profit_factor"),
                notes="; ".join(item.get("reasons", [])),
            )
        )
    assessment = payload.get("strategy_assessment") or {}
    if assessment.get("scorecards"):
        lines.extend(
            [
                "",
                "## Strategy Scorecards",
                "",
                "| Strategy | Tier | Score | Base % | Adjusted % | Market % | PF | DD % | Trades | Failures |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in assessment["scorecards"]:
            lines.append(
                "| {strategy} | {tier} | {score} | {base_return_pct} | {adjusted_return_pct} | {market_change_pct} | {profit_factor} | {max_drawdown_pct} | {trades} | {failures} |".format(
                    failures=", ".join(item.get("primary_failures", [])),
                    **item,
                )
            )
        lines.extend(["", "## Failure Diagnostics", "", "| Failure | Count |", "|---|---:|"])
        for item in assessment.get("failure_summary", []):
            lines.append("| {failure} | {count} |".format(**item))
        lines.extend(["", "## Research Next Actions", ""])
        for item in assessment.get("diagnostics", []):
            lines.append(f"### {item.get('strategy')}")
            for action in item.get("next_actions", []):
                lines.append(f"- {action}")
            lines.append("")
    walk_forward = payload.get("walk_forward_summary") or {}
    if walk_forward.get("strategy_summary"):
        lines.extend(
            [
                "",
                "## Walk-Forward Validation",
                "",
                "| Strategy | Verdict | Windows | Positive | Negative | Trades | Median Return | Median PF | Worst DD | Reasons |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in walk_forward["strategy_summary"]:
            row = dict(item)
            row["reasons"] = ", ".join(item.get("reasons", []))
            lines.append(
                "| {strategy} | {verdict} | {windows} | {positive_windows} | {negative_windows} | {total_trades} | {median_return_pct}% | {median_profit_factor} | {worst_drawdown_pct}% | {reasons} |".format(
                    **row,
                )
            )
    promotion_report = payload.get("promotion_report") or {}
    if promotion_report.get("verdicts"):
        lines.extend(
            [
                "",
                "## Promotion Gate",
                "",
                "| Strategy | Verdict | Ready | Blocks | Next Actions |",
                "|---|---|---:|---|---|",
            ]
        )
        for item in promotion_report["verdicts"]:
            row = dict(item)
            row["blocks"] = ", ".join(item.get("blocks", []))
            row["next_actions"] = "; ".join(item.get("next_actions", []))
            lines.append(
                "| {strategy} | {verdict} | {ready_for_manual_dryrun_review} | {blocks} | {next_actions} |".format(
                    **row
                )
            )
    research_agenda = payload.get("research_agenda") or {}
    if research_agenda.get("top_priorities"):
        lines.extend(
            [
                "",
                "## Research Agenda",
                "",
                "| Priority | Strategy | Blocker | Objective | Next Command | Success Gate |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for item in research_agenda["top_priorities"]:
            lines.append(
                "| {priority} | {strategy} | {blocker} | {objective} | `{next_command}` | {success_gate} |".format(
                    **item
                )
            )
    agenda_run = payload.get("agenda_run") or {}
    if agenda_run:
        item = agenda_run.get("selected_item") or {}
        lines.extend(
            [
                "",
                "## Agenda Run",
                "",
                "| Status | Mode | Strategy | Blocker | Command | Return Code |",
                "|---|---|---|---|---|---:|",
                "| {status} | {mode} | {strategy} | {blocker} | `{command}` | {returncode} |".format(
                    status=agenda_run.get("status"),
                    mode=agenda_run.get("mode"),
                    strategy=item.get("strategy", ""),
                    blocker=item.get("blocker", ""),
                    command=agenda_run.get("command") or "",
                    returncode=agenda_run.get("returncode"),
                ),
            ]
        )
    trade_behavior = payload.get("trade_behavior") or {}
    if trade_behavior.get("summaries"):
        lines.extend(
            [
                "",
                "## Trade Behavior",
                "",
                "| Strategy | Trades | Win % | Profit Abs | PF | Payoff | Avg Dur | Long/Short | Stop Losses | MFE % | MAE % | Diagnostics |",
                "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for item in trade_behavior["summaries"]:
            row = dict(item)
            row["diagnostics"] = "; ".join(item.get("diagnostics", []))
            lines.append(
                "| {strategy} | {trades} | {win_rate_pct} | {total_profit_abs} | {profit_factor} | {payoff_ratio} | {avg_duration_min} | {long_trades}/{short_trades} | {stop_loss_trades} | {avg_mfe_pct} | {avg_mae_pct} | {diagnostics} |".format(
                    **row
                )
            )
    behavior_experiments = payload.get("behavior_experiments") or {}
    if behavior_experiments.get("plans"):
        lines.extend(
            [
                "",
                "## Behavior-Driven Experiment Plan",
                "",
                "| Priority | Strategy | Experiment | Hypothesis | Success Gate |",
                "|---:|---|---|---|---|",
            ]
        )
        for item in behavior_experiments["plans"]:
            lines.append(
                "| {priority} | {strategy} | {experiment_id} | {hypothesis} | {success_gate} |".format(
                    **item
                )
            )
    lines.extend(["", "## Source Reviews", "", "| Source | Status | Family | Indicators |", "|---|---|---|---|"])
    for item in payload.get("source_reviews", []):
        lines.append(
            "| {source} | {status} | {family} | {indicators} |".format(
                source=item.get("id"),
                status=item.get("status"),
                family=item.get("inferred_strategy_family"),
                indicators=", ".join(item.get("detected_indicators", [])),
            )
        )
    matrix_summary = payload.get("matrix_summary") or {}
    if matrix_summary.get("strategy_summary"):
        lines.extend(
            [
                "",
                "## Matrix Robustness",
                "",
                "| Strategy | Verdict | Runs | Positive | Too Few Trades | Stress Negative | Min Return | Max Return | Min PF |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in matrix_summary["strategy_summary"]:
            lines.append(
                "| {strategy} | {verdict} | {runs} | {positive_runs} | {too_few_trade_runs} | {stress_negative_runs} | {min_return_pct}% | {max_return_pct}% | {min_profit_factor} |".format(
                    **item
                )
            )
    cost_audit = payload.get("cost_audit") or {}
    if cost_audit.get("funding"):
        lines.extend(
            [
                "",
                "## Futures Cost Data",
                "",
                "| Pair | Funding Rows | Funding Last | Funding Gaps | Mean Funding % | Sum Funding % |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for item in cost_audit["funding"]:
            lines.append(
                "| {pair} | {rows} | {last_utc} | {gaps} | {mean_rate_pct} | {sum_rate_pct} |".format(
                    **item
                )
            )
        lines.extend(
            [
                "",
                "| Pair | Mark Rows | Mark Last | Mark Gaps | Mean abs basis bps | P95 abs basis bps |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for item in cost_audit.get("mark_price", []):
            lines.append(
                "| {pair} | {rows} | {last_utc} | {gaps} | {mean_abs_basis_bps} | {p95_abs_basis_bps} |".format(
                    **item
                )
            )
    aux_conversion = payload.get("aux_conversion") or {}
    if aux_conversion.get("converted_files"):
        lines.extend(
            [
                "",
                "## Freqtrade Futures Aux Conversion",
                "",
                "| Pair | Candle Type | Timeframe | Rows | First | Last |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for item in aux_conversion["converted_files"]:
            lines.append(
                "| {pair} | {candle_type} | {timeframe} | {rows} | {first_utc} | {last_utc} |".format(
                    **item
                )
            )
    trade_cost_estimate = payload.get("trade_cost_estimate") or {}
    if trade_cost_estimate.get("estimates"):
        lines.extend(
            [
                "",
                "## Trade Cost Adjustment",
                "",
                f"- Round-trip slippage bps: `{trade_cost_estimate.get('slippage_bps_round_trip')}`",
                "",
                "| Strategy | Trades | Base % | Funding % | Slippage % | Adjusted % | Funding Events | Coverage Missing |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in trade_cost_estimate["estimates"]:
            lines.append(
                "| {strategy} | {trades} | {base_profit_pct} | {funding_pct_of_start_balance} | {slippage_pct_of_start_balance} | {adjusted_profit_pct} | {funding_events} | {funding_coverage_missing_trades} |".format(
                    **item
                )
            )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- This agent is research-only.",
            "- Live trading, API key access, leverage increases, and live config changes require manual approval.",
            "- External source code may be read and translated, but must not be installed or executed by default.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dashboard_path = render_dashboard(paths, payload)
    return json_path, md_path, dashboard_path


def update_index(paths: dict[str, Path], latest_payload: dict[str, Any]) -> Path:
    report_dir = paths["report_dir"]
    index_path = report_dir / "agent_report_index.json"
    reports = []
    for path in latest_reports(report_dir):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        reports.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "generated_at_utc": payload.get("generated_at_utc"),
                "experiment": payload.get("experiment", {}).get("id"),
                "result_count": len(payload.get("results", [])),
            }
        )
    index_payload = {
        "updated_at_utc": latest_payload["generated_at_utc"],
        "latest_report": next((item for item in reports if item["result_count"] > 0), reports[0] if reports else None),
        "latest_dashboard_refresh": reports[0] if reports else None,
        "reports": reports,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    registry = load_json(args.registry)
    experiment = load_json(args.experiment)
    paths = ensure_workspace(config)

    if not config["agent"]["live_trading_allowed"]:
        print("Safety: live trading is disabled for this research agent.")

    profile = registry["profile"]
    timeframes = [args.timeframe] if args.timeframe else experiment.get("timeframes", [profile["timeframe"]])
    timeranges = selected_timeranges(experiment, args, profile["timerange"])
    if len(timeframes) != 1:
        raise SystemExit("This runner currently expects one timeframe per run.")
    timeframe = timeframes[0]
    run_timerange_label = ",".join(item["timerange"] for item in timeranges)
    fee = args.fee if args.fee is not None else experiment.get("fee", profile["fee"])
    strategy_path = experiment.get("strategy_path")

    audits = audit_data(profile, timeframes)
    meta = strategy_metadata(registry)
    thresholds = registry["thresholds"]
    results: list[dict[str, Any]] = []

    if not args.skip_backtests:
        for strategy in selected_strategies(experiment, args):
            if strategy not in meta:
                raise SystemExit(f"Strategy not found in registry: {strategy}")
            for timerange_item in timeranges:
                timerange = timerange_item["timerange"]
                metrics = run_backtest_command(
                    profile["config"],
                    strategy,
                    timeframe,
                    timerange,
                    fee,
                    args.dry_run,
                    strategy_path,
                )
                classification, reasons = classify(metrics, thresholds)
                result = metrics_to_dict(metrics, classification, reasons)
                result.update(
                    {
                        "family": meta[strategy].get("family"),
                        "source": meta[strategy].get("source"),
                        "hypothesis": meta[strategy].get("hypothesis"),
                        "risk_notes": meta[strategy].get("risk_notes"),
                        "regime": timerange_item.get("name"),
                        "regime_label": timerange_item.get("label"),
                        "timerange": timerange,
                        "btc_return_pct": timerange_item.get("btc_return_pct"),
                        "regime_realized_vol_pct": timerange_item.get("realized_vol_pct"),
                        "fee": fee,
                        "recursive_analysis": None,
                        "lookahead_analysis": None,
                    }
                )
                if args.run_recursive:
                    result["recursive_analysis"] = run_analysis_command(
                        "recursive-analysis",
                        profile["config"],
                        strategy,
                        timeframe,
                        timerange,
                        fee,
                        args.dry_run,
                        strategy_path,
                    )
                if args.run_lookahead:
                    result["lookahead_analysis"] = run_analysis_command(
                        "lookahead-analysis",
                        profile["config"],
                        strategy,
                        timeframe,
                        timerange,
                        fee,
                        args.dry_run,
                        strategy_path,
                    )
                results.append(result)
                if len(timeranges) == 1:
                    write_candidate_files(paths, result)
                print(f"{strategy} [{timerange_item.get('name')}]: {classification} ({'; '.join(reasons)})")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at_utc": timestamp,
        "agent": config["agent"],
        "experiment": experiment,
        "run": {
            "timeframe": timeframe,
            "timerange": run_timerange_label,
            "fee": fee,
            "dry_run": args.dry_run,
            "skip_backtests": args.skip_backtests,
            "run_recursive": args.run_recursive,
            "run_lookahead": args.run_lookahead,
        },
        "profile": profile,
        "thresholds": thresholds,
        "data_audit": [asdict(item) for item in audits],
        "ohlcv_update": load_latest_ohlcv_update(),
        "candidate_pool": load_pool(paths["candidate_dir"]),
        "watchlist_pool": load_pool(paths["watchlist_dir"]),
        "rejected_pool": load_pool(paths["rejected_dir"]),
        "source_reviews": load_source_reviews(),
        "matrix_summary": load_latest_matrix_summary(),
        "cost_audit": load_latest_cost_audit(),
        "trade_cost_estimate": load_latest_trade_cost_estimate(),
        "aux_conversion": load_latest_aux_conversion(),
        "strategy_assessment": load_latest_strategy_assessment(),
        "walk_forward_summary": load_latest_walk_forward_summary(),
        "promotion_report": load_latest_promotion_report(),
        "research_agenda": load_latest_research_agenda(),
        "agenda_run": load_latest_agenda_run(),
        "trade_behavior": load_latest_trade_behavior(),
        "behavior_experiments": load_latest_behavior_experiment_plan(),
        "results": results,
    }
    json_path, md_path, dashboard_path = write_report(paths, payload)
    index_path = update_index(paths, payload)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {dashboard_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {index_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
