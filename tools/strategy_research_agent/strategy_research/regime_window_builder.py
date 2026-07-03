#!/usr/bin/env python3
"""Build data-derived market-regime windows for strategy research.

The research agent must not treat hand-picked dates as real market regimes.
This module reads local Binance USDT-M BTC/ETH futures OHLCV, computes
trend/volatility/range evidence, labels daily regimes, and publishes a
versioned manifest that later gates can load.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PAIRS = ["BTC_USDT_USDT", "ETH_USDT_USDT"]
REGIME_LABELS = ["bull", "bear", "range", "high_vol"]
MANIFEST_VERSION = 1
MIN_WINDOW_DAYS = 45
WINDOW_LENGTHS = [60, 75, 90]
LEGACY_REGIME_TOKENS = [
    "bull_home",
    "range_home",
    "bear_home",
    "bear_hostile",
    "high_vol_hostile",
    "bull_20241022_20250120",
    "range_20240507_20240805",
    "bear_20251222_20260322",
    "high_vol_20260118_20260418",
    "20241022-20250120",
    "20240507-20240805",
    "20251222-20260322",
    "20260118-20260418",
]

FAMILY_HOME_LABELS = {
    "downtrend_failed_bounce_short": {"bear"},
    "downtrend_pullback_short": {"bear"},
    "downside_breakout_continuation_short": {"bear", "high_vol"},
    "uptrend_failed_pullback_long": {"bull"},
    "uptrend_pullback_long": {"bull"},
    "upside_breakout_continuation_long": {"bull", "high_vol"},
    "range_upper_reversion_short": {"range"},
    "range_lower_reversion_long": {"range"},
    "volatility_compression_breakout": {"range"},
}


@dataclass
class RegimeCheck:
    name: str
    status: str
    detail: str


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for start in candidates:
        for path in [start, *start.parents]:
            if (path / "user_data/data/binance/futures").exists() and (path / "pyproject.toml").exists():
                return path
    raise RuntimeError("Could not locate freqtrade repo root with user_data/data/binance/futures.")


REPO_ROOT = find_repo_root()
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DATA_DIR = REPO_ROOT / "user_data/data/binance/futures"
OUTPUT_DIR = AGENT_ROOT / "regime_windows"
LATEST_JSON = OUTPUT_DIR / "latest_regime_windows.json"
LATEST_MD = OUTPUT_DIR / "latest_regime_windows.md"
QUARANTINE_JSON = OUTPUT_DIR / "regime_inference_quarantine.json"
QUARANTINE_MD = OUTPUT_DIR / "regime_inference_quarantine.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_feather(path, columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.sort_values("date").drop_duplicates("date").set_index("date")


def resample_to_1h(frame: pd.DataFrame) -> pd.DataFrame:
    hourly = frame.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return hourly.dropna(subset=["open", "high", "low", "close"])


def load_pair_1h(pair: str) -> tuple[pd.DataFrame, str]:
    preferred = DATA_DIR / f"{pair}-1h-futures.feather"
    if preferred.exists():
        return read_ohlcv(preferred), rel(preferred)
    for timeframe in ["15m", "5m", "1m"]:
        path = DATA_DIR / f"{pair}-{timeframe}-futures.feather"
        if path.exists():
            return resample_to_1h(read_ohlcv(path)), f"{rel(path)} resampled_to_1h"
    raise FileNotFoundError(f"Missing 1h/15m/5m/1m futures data for {pair}")


def daily_features(pair: str) -> tuple[pd.DataFrame, str]:
    hourly, source = load_pair_1h(pair)
    daily = hourly.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    daily = daily.dropna(subset=["open", "high", "low", "close"]).copy()
    daily["ret_1d"] = daily["close"].pct_change()
    daily["ret_30d"] = daily["close"].pct_change(30)
    daily["ret_60d"] = daily["close"].pct_change(60)
    daily["ema30"] = daily["close"].ewm(span=30, adjust=False).mean()
    daily["ema60"] = daily["close"].ewm(span=60, adjust=False).mean()
    daily["ema120"] = daily["close"].ewm(span=120, adjust=False).mean()
    daily["ema30_120_gap"] = daily["ema30"] / daily["ema120"] - 1.0

    prev_close = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr_pct"] = true_range.ewm(alpha=1 / 14, adjust=False).mean() / daily["close"]
    daily["realized_vol_30d"] = daily["ret_1d"].rolling(30, min_periods=20).std() * math.sqrt(365)
    daily["realized_vol_pctile"] = daily["realized_vol_30d"].rank(pct=True)
    daily["atr_pctile"] = daily["atr_pct"].rank(pct=True)
    mid = daily["close"].rolling(20, min_periods=15).mean()
    std = daily["close"].rolling(20, min_periods=15).std()
    daily["bb_width"] = (4 * std) / mid
    daily["bb_width_pctile"] = daily["bb_width"].rank(pct=True)
    daily["trend_efficiency_30d"] = (
        (daily["close"] - daily["close"].shift(30)).abs()
        / daily["close"].diff().abs().rolling(30, min_periods=20).sum()
    )
    daily = daily.add_prefix(pair.lower().replace("_usdt_usdt", "") + "_")
    return daily, source


def combined_features() -> tuple[pd.DataFrame, dict[str, str]]:
    sources: dict[str, str] = {}
    frames = []
    for pair in PAIRS:
        frame, source = daily_features(pair)
        sources[pair] = source
        frames.append(frame)
    data = pd.concat(frames, axis=1).dropna(subset=["btc_close", "eth_close"]).copy()
    data["combined_ret_30d"] = data[["btc_ret_30d", "eth_ret_30d"]].mean(axis=1)
    data["combined_ret_60d"] = data[["btc_ret_60d", "eth_ret_60d"]].mean(axis=1)
    data["combined_ema_gap"] = data[["btc_ema30_120_gap", "eth_ema30_120_gap"]].mean(axis=1)
    data["combined_vol_pctile"] = data[["btc_realized_vol_pctile", "eth_realized_vol_pctile"]].mean(axis=1)
    data["combined_atr_pctile"] = data[["btc_atr_pctile", "eth_atr_pctile"]].mean(axis=1)
    data["combined_bb_width_pctile"] = data[["btc_bb_width_pctile", "eth_bb_width_pctile"]].mean(axis=1)
    data["combined_trend_efficiency"] = data[["btc_trend_efficiency_30d", "eth_trend_efficiency_30d"]].mean(axis=1)
    data["direction_agreement_60d"] = (
        (data["btc_ret_60d"] > 0) & (data["eth_ret_60d"] > 0)
        | ((data["btc_ret_60d"] < 0) & (data["eth_ret_60d"] < 0))
    ).astype(float)
    return data, sources


def label_daily(row: pd.Series) -> str:
    high_vol = row["combined_vol_pctile"] >= 0.82 or row["combined_atr_pctile"] >= 0.82
    bull = (
        row["combined_ret_60d"] >= 0.16
        and row["combined_ema_gap"] >= 0.035
        and row["combined_trend_efficiency"] >= 0.25
        and row["direction_agreement_60d"] >= 1.0
    )
    bear = (
        row["combined_ret_60d"] <= -0.14
        and row["combined_ema_gap"] <= -0.030
        and row["combined_trend_efficiency"] >= 0.22
        and row["direction_agreement_60d"] >= 1.0
    )
    range_bound = (
        abs(row["combined_ret_60d"]) <= 0.12
        and abs(row["combined_ema_gap"]) <= 0.040
        and row["combined_trend_efficiency"] <= 0.36
        and row["combined_vol_pctile"] <= 0.72
    )
    if high_vol:
        return "high_vol"
    if bull:
        return "bull"
    if bear:
        return "bear"
    if range_bound:
        return "range"
    return "mixed"


def score_window(label: str, window: pd.DataFrame) -> float:
    share = float((window["daily_label"] == label).mean())
    ret = float(window["combined_ret_60d"].iloc[-1])
    ema_gap = float(window["combined_ema_gap"].mean())
    vol = float(window["combined_vol_pctile"].mean())
    trend = float(window["combined_trend_efficiency"].mean())
    if label == "bull":
        directional = max(ret, 0) + max(ema_gap, 0) + trend
    elif label == "bear":
        directional = max(-ret, 0) + max(-ema_gap, 0) + trend
    elif label == "range":
        directional = max(0.0, 0.40 - trend) + max(0.0, 0.12 - abs(ret))
    elif label == "high_vol":
        directional = vol + float(window["combined_atr_pctile"].mean())
    else:
        directional = 0.0
    return share * 10.0 + directional


def summarize_window(label: str, data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    window = data.loc[start:end]
    counts = window["daily_label"].value_counts().to_dict()
    days = int(len(window))
    label_share = float(counts.get(label, 0) / days) if days else 0.0
    confidence = "high" if label_share >= 0.70 else "medium" if label_share >= 0.55 else "low"
    evidence = {
        "btc_return_pct": round((window["btc_close"].iloc[-1] / window["btc_close"].iloc[0] - 1.0) * 100.0, 4),
        "eth_return_pct": round((window["eth_close"].iloc[-1] / window["eth_close"].iloc[0] - 1.0) * 100.0, 4),
        "combined_ret_30d_last_pct": round(float(window["combined_ret_30d"].iloc[-1]) * 100.0, 4),
        "combined_ret_60d_last_pct": round(float(window["combined_ret_60d"].iloc[-1]) * 100.0, 4),
        "combined_ema_gap_avg_pct": round(float(window["combined_ema_gap"].mean()) * 100.0, 4),
        "realized_vol_percentile_avg": round(float(window["combined_vol_pctile"].mean()), 4),
        "atr_percentile_avg": round(float(window["combined_atr_pctile"].mean()), 4),
        "bb_width_percentile_avg": round(float(window["combined_bb_width_pctile"].mean()), 4),
        "trend_efficiency_avg": round(float(window["combined_trend_efficiency"].mean()), 4),
        "btc_eth_direction_agreement_avg": round(float(window["direction_agreement_60d"].mean()), 4),
        "label_share": round(label_share, 4),
        "daily_label_counts": {str(key): int(value) for key, value in counts.items()},
    }
    return {
        "label": label,
        "name": f"{label}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}",
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "timerange": f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}",
        "days": days,
        "confidence": confidence,
        "evidence": evidence,
    }


def select_windows(data: pd.DataFrame) -> list[dict[str, Any]]:
    clean = data.dropna(
        subset=[
            "combined_ret_60d",
            "combined_ema_gap",
            "combined_vol_pctile",
            "combined_atr_pctile",
            "combined_trend_efficiency",
        ]
    ).copy()
    clean["daily_label"] = clean.apply(label_daily, axis=1)
    windows: list[dict[str, Any]] = []
    used_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for label in REGIME_LABELS:
        best: tuple[float, pd.Timestamp, pd.Timestamp] | None = None
        for length in WINDOW_LENGTHS:
            if len(clean) < length:
                continue
            for end_idx in range(length - 1, len(clean)):
                window = clean.iloc[end_idx - length + 1 : end_idx + 1]
                if len(window) < MIN_WINDOW_DAYS:
                    continue
                share = float((window["daily_label"] == label).mean())
                if share < 0.35:
                    continue
                start = window.index[0]
                end = window.index[-1]
                score = score_window(label, window)
                for used_start, used_end in used_ranges:
                    overlap = max(pd.Timedelta(0), min(end, used_end) - max(start, used_start)).days
                    if overlap > length * 0.6:
                        score -= 2.0
                if best is None or score > best[0]:
                    best = (score, start, end)
        if best is None:
            windows.append(
                {
                    "label": label,
                    "name": f"{label}_data_insufficient",
                    "status": "data_insufficient",
                    "reason": "No 60-90 day segment had enough indicator support for this label.",
                    "confidence": "none",
                    "evidence": {},
                }
            )
            continue
        _, start, end = best
        row = summarize_window(label, clean, start, end)
        row["status"] = "active"
        windows.append(row)
        used_ranges.append((start, end))
    return windows


def family_window_roles(windows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    active = [window for window in windows if window.get("status") == "active"]
    roles: dict[str, dict[str, list[str]]] = {}
    for family, home_labels in FAMILY_HOME_LABELS.items():
        home = [window["name"] for window in active if window.get("label") in home_labels]
        hostile = [window["name"] for window in active if window.get("label") not in home_labels]
        roles[family] = {"home": home, "hostile": hostile}
    return roles


def scan_legacy_sources() -> list[dict[str, Any]]:
    roots = [
        AGENT_ROOT / "reports",
        AGENT_ROOT / "event_studies",
        AGENT_ROOT / "research_memory",
        AGENT_ROOT / "failure_attribution",
        AGENT_ROOT / "mature_researcher",
    ]
    matches: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir() or path.suffix not in {".json", ".md", ".csv"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            tokens = [token for token in LEGACY_REGIME_TOKENS if token in text]
            if not tokens:
                continue
            matches.append(
                {
                    "path": rel(path),
                    "tokens": sorted(set(tokens)),
                    "quarantine_reason": "legacy_hardcoded_regime_window",
                    "new_status": "needs_regime_relabel",
                    "allowed_use": "raw_date_range_backtest_only",
                }
            )
    return matches


def build_payload() -> dict[str, Any]:
    data, sources = combined_features()
    windows = select_windows(data)
    active_labels = {window["label"] for window in windows if window.get("status") == "active"}
    payload = {
        "generated_at_utc": now_utc(),
        "manifest_version": MANIFEST_VERSION,
        "method": "data_derived_btc_eth_futures_ohlcv",
        "research_only": True,
        "data_sources": sources,
        "feature_timeframe": "1h",
        "daily_rows": int(len(data)),
        "date_range": {
            "start": data.index.min().strftime("%Y-%m-%d"),
            "end": data.index.max().strftime("%Y-%m-%d"),
        },
        "label_definitions": {
            "bull": "positive 60d BTC/ETH aligned trend, positive EMA gap, directional efficiency",
            "bear": "negative 60d BTC/ETH aligned trend, negative EMA gap, directional efficiency",
            "range": "low 60d direction, flat EMA gap, low trend efficiency, non-extreme volatility",
            "high_vol": "realized volatility or ATR percentile in the top regime bucket",
            "mixed": "not used as home/hostile window candidate",
        },
        "windows": windows,
        "family_window_roles": family_window_roles(windows),
        "quality": {
            "active_label_count": len(active_labels),
            "missing_labels": sorted(set(REGIME_LABELS) - active_labels),
            "legacy_hardcoded_windows_allowed": False,
        },
    }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_manifest_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Data-Derived Regime Windows",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Method: `{payload['method']}`",
        f"- Data range: `{payload['date_range']['start']}` -> `{payload['date_range']['end']}`",
        f"- Daily rows: `{payload['daily_rows']}`",
        "",
        "## Windows",
        "",
        "| Label | Name | Timerange | Days | Confidence | BTC Ret % | ETH Ret % | Vol Pctile | Trend Eff | Label Share |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in payload["windows"]:
        ev = item.get("evidence", {})
        lines.append(
            "| {label} | `{name}` | `{timerange}` | {days} | {confidence} | {btc} | {eth} | {vol} | {trend} | {share} |".format(
                label=item.get("label", ""),
                name=item.get("name", ""),
                timerange=item.get("timerange") or item.get("status", ""),
                days=item.get("days", 0),
                confidence=item.get("confidence", ""),
                btc=ev.get("btc_return_pct", ""),
                eth=ev.get("eth_return_pct", ""),
                vol=ev.get("realized_vol_percentile_avg", ""),
                trend=ev.get("trend_efficiency_avg", ""),
                share=ev.get("label_share", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- These windows replace legacy hand-picked bull/range/bear/high-vol dates for active regime gates.",
            "- Old reports may still describe raw date-range backtests, but their regime inference is quarantined until relabeled.",
            "- Strategy-family home/hostile roles are derived from the family contract and this manifest, not from hardcoded names.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quarantine(matches: list[dict[str, Any]], generated_at: str) -> None:
    payload = {
        "generated_at_utc": generated_at,
        "quarantine_version": 1,
        "status": "active",
        "reason": "Legacy reports and memory used hand-picked regime labels before data-derived regime windows existed.",
        "legacy_tokens": LEGACY_REGIME_TOKENS,
        "entries": matches,
        "policy": {
            "old_backtest_rows_allowed": True,
            "old_regime_inference_allowed": False,
            "required_new_status": "needs_regime_relabel",
        },
    }
    write_json(QUARANTINE_JSON, payload)
    lines = [
        "# Regime Inference Quarantine",
        "",
        f"- Generated UTC: `{generated_at}`",
        f"- Entries: `{len(matches)}`",
        "",
        "Old reports remain raw date-range evidence only. Their bull/range/bear/high-vol interpretation is not valid Agent memory until relabeled by the data-derived manifest.",
        "",
        "| Path | Tokens | Status | Allowed Use |",
        "|---|---|---|---|",
    ]
    for item in matches[:200]:
        lines.append(
            "| `{path}` | {tokens} | `{status}` | `{allowed}` |".format(
                path=item["path"],
                tokens=", ".join(f"`{token}`" for token in item["tokens"][:6]),
                status=item["new_status"],
                allowed=item["allowed_use"],
            )
        )
    if len(matches) > 200:
        lines.append(f"| ... | ... | ... | {len(matches) - 200} additional entries omitted from markdown |")
    QUARANTINE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_regime_manifest(path: Path = LATEST_JSON) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing regime manifest: {rel(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def validate_manifest(payload: dict[str, Any]) -> list[RegimeCheck]:
    checks: list[RegimeCheck] = []
    def add(name: str, status: str, detail: str) -> None:
        checks.append(RegimeCheck(name, status, detail))

    if payload.get("manifest_version") != MANIFEST_VERSION:
        add("manifest_version", "fail", f"expected {MANIFEST_VERSION}, got {payload.get('manifest_version')}")
    else:
        add("manifest_version", "ok", str(MANIFEST_VERSION))
    if payload.get("method") != "data_derived_btc_eth_futures_ohlcv":
        add("method", "fail", str(payload.get("method")))
    else:
        add("method", "ok", payload["method"])
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        add("windows", "fail", "missing windows")
    else:
        active = [item for item in windows if item.get("status") == "active"]
        labels = {item.get("label") for item in active}
        missing = sorted(set(REGIME_LABELS) - labels)
        status = "warn" if missing else "ok"
        add("windows:labels", status, f"active={sorted(labels)} missing={missing}")
        for item in active:
            evidence = item.get("evidence", {})
            required = [
                "btc_return_pct",
                "eth_return_pct",
                "combined_ret_60d_last_pct",
                "realized_vol_percentile_avg",
                "trend_efficiency_avg",
                "label_share",
            ]
            missing_evidence = [key for key in required if key not in evidence]
            if missing_evidence:
                add(f"window:{item.get('name')}", "fail", "missing evidence " + ", ".join(missing_evidence))
    failed = [check for check in checks if check.status == "fail"]
    if failed:
        raise ValueError("; ".join(f"{check.name}: {check.detail}" for check in failed))
    return checks


def check_manifest_status(path: Path = LATEST_JSON) -> list[RegimeCheck]:
    if not path.exists():
        return [RegimeCheck("regime_manifest", "fail", f"Missing {rel(path)}")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = validate_manifest(payload)
    except Exception as exc:  # noqa: BLE001 - gate must surface local manifest problems.
        return [RegimeCheck("regime_manifest", "fail", f"{type(exc).__name__}: {exc}")]
    checks.insert(0, RegimeCheck("regime_manifest", "ok", rel(path)))
    return checks


def write_outputs(payload: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = payload["generated_at_utc"]
    json_path = OUTPUT_DIR / f"regime_windows_{timestamp}.json"
    md_path = OUTPUT_DIR / f"regime_windows_{timestamp}.md"
    write_json(json_path, payload)
    write_json(LATEST_JSON, payload)
    write_manifest_markdown(md_path, payload)
    LATEST_MD.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    matches = scan_legacy_sources()
    write_quarantine(matches, timestamp)
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate the latest manifest without rebuilding.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload/checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only:
        checks = check_manifest_status()
        payload = {"status": "fail" if any(c.status == "fail" for c in checks) else "ok", "checks": [c.__dict__ for c in checks]}
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            for check in checks:
                print(f"[{check.status.upper():4}] {check.name}: {check.detail}")
        return 0 if payload["status"] == "ok" else 1

    payload = build_payload()
    json_path, md_path = write_outputs(payload)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Wrote {rel(json_path)}")
        print(f"Wrote {rel(md_path)}")
        print(f"Wrote {rel(LATEST_JSON)}")
        print(f"Wrote {rel(LATEST_MD)}")
        print(f"Wrote {rel(QUARANTINE_JSON)}")
        print(f"Wrote {rel(QUARANTINE_MD)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
