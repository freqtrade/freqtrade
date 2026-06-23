#!/usr/bin/env python3
"""Audit futures funding-rate and mark-price coverage for BTC/ETH research."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
AUX_ROOT = REPO_ROOT / "user_data/data/binance/futures_aux"
REPORT_DIR = REPO_ROOT / "user_data/strategy_research/cost_audits"
PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT"]


@dataclass
class FundingAudit:
    pair: str
    exists: bool
    rows: int | None = None
    first_utc: str | None = None
    last_utc: str | None = None
    gaps: int | None = None
    mean_rate_pct: float | None = None
    median_rate_pct: float | None = None
    positive_rate_pct: float | None = None
    sum_rate_pct: float | None = None
    always_long_funding_pct: float | None = None
    always_short_funding_pct: float | None = None


@dataclass
class MarkAudit:
    pair: str
    exists: bool
    rows: int | None = None
    first_utc: str | None = None
    last_utc: str | None = None
    gaps: int | None = None
    overlap_rows: int | None = None
    mean_basis_bps: float | None = None
    mean_abs_basis_bps: float | None = None
    p95_abs_basis_bps: float | None = None
    max_abs_basis_bps: float | None = None


def read_feather(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values("date")


def count_gaps(dates: pd.Series, expected_seconds: int, tolerance_seconds: int = 1) -> int:
    deltas = dates.diff().dt.total_seconds().dropna()
    return int((deltas > expected_seconds + tolerance_seconds).sum())


def audit_funding(pair: str) -> FundingAudit:
    path = AUX_ROOT / "funding_rate" / f"{pair}.feather"
    if not path.exists():
        return FundingAudit(pair, False)
    frame = read_feather(path)
    rates = frame["funding_rate"].astype(float)
    sum_rate = float(rates.sum() * 100)
    return FundingAudit(
        pair=pair,
        exists=True,
        rows=int(len(frame)),
        first_utc=frame["date"].min().isoformat(),
        last_utc=frame["date"].max().isoformat(),
        gaps=count_gaps(frame["date"], 8 * 3600, tolerance_seconds=60),
        mean_rate_pct=round(float(rates.mean() * 100), 5),
        median_rate_pct=round(float(rates.median() * 100), 5),
        positive_rate_pct=round(float((rates > 0).mean() * 100), 2),
        sum_rate_pct=round(sum_rate, 4),
        always_long_funding_pct=round(-sum_rate, 4),
        always_short_funding_pct=round(sum_rate, 4),
    )


def audit_mark(pair: str) -> MarkAudit:
    mark_path = AUX_ROOT / "mark_price" / f"{pair}-1m.feather"
    ohlcv_path = REPO_ROOT / "user_data/data/binance/futures" / f"{pair}-1m-futures.feather"
    if not mark_path.exists():
        return MarkAudit(pair, False)
    mark = read_feather(mark_path)
    audit = MarkAudit(
        pair=pair,
        exists=True,
        rows=int(len(mark)),
        first_utc=mark["date"].min().isoformat(),
        last_utc=mark["date"].max().isoformat(),
        gaps=count_gaps(mark["date"], 60),
    )
    if not ohlcv_path.exists():
        return audit
    ohlcv = read_feather(ohlcv_path)[["date", "close"]].rename(columns={"close": "last_close"})
    merged = mark[["date", "close"]].rename(columns={"close": "mark_close"}).merge(
        ohlcv,
        on="date",
        how="inner",
    )
    basis_bps = (merged["mark_close"] / merged["last_close"] - 1.0) * 10000
    audit.overlap_rows = int(len(merged))
    audit.mean_basis_bps = round(float(basis_bps.mean()), 4)
    audit.mean_abs_basis_bps = round(float(basis_bps.abs().mean()), 4)
    audit.p95_abs_basis_bps = round(float(basis_bps.abs().quantile(0.95)), 4)
    audit.max_abs_basis_bps = round(float(basis_bps.abs().max()), 4)
    return audit


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Futures Cost Data Audit",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        "- Market profile: `crypto / Binance USDT-M futures`",
        "",
        "## Funding Rate",
        "",
        "| Pair | Rows | First | Last | Gaps | Mean % | Positive % | Sum % | Always Long % | Always Short % |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["funding"]:
        lines.append(
            "| {pair} | {rows} | {first_utc} | {last_utc} | {gaps} | {mean_rate_pct} | {positive_rate_pct} | {sum_rate_pct} | {always_long_funding_pct} | {always_short_funding_pct} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Mark Price Basis",
            "",
            "| Pair | Rows | First | Last | Gaps | Overlap | Mean bps | Mean abs bps | P95 abs bps | Max abs bps |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["mark_price"]:
        lines.append(
            "| {pair} | {rows} | {first_utc} | {last_utc} | {gaps} | {overlap_rows} | {mean_basis_bps} | {mean_abs_basis_bps} | {p95_abs_basis_bps} | {max_abs_basis_bps} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Funding signs are reported from the perspective of the raw funding rate: positive rates are typically paid by longs and received by shorts.",
            "- `always_long_funding_pct` and `always_short_funding_pct` are simple always-in-market reference sums, not strategy-specific funding PnL.",
            "- Current Freqtrade backtests still use OHLCV candles; mark price and funding are audited as separate realism checks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "funding": [asdict(audit_funding(pair)) for pair in PAIRS],
        "mark_price": [asdict(audit_mark(pair)) for pair in PAIRS],
    }
    timestamp = payload["generated_at_utc"]
    json_path = REPORT_DIR / f"futures_cost_audit_{timestamp}.json"
    md_path = REPORT_DIR / f"futures_cost_audit_{timestamp}.md"
    latest_json = REPORT_DIR / "latest_futures_cost_audit.json"
    latest_md = REPORT_DIR / "latest_futures_cost_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {latest_json.relative_to(REPO_ROOT)}")
    print(f"Wrote {latest_md.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
