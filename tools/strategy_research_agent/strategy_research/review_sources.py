#!/usr/bin/env python3
"""Review quarantined external sources and create isolated translation drafts."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
SOURCE_REGISTRY = AGENT_ROOT / "source_registry.json"
REVIEW_DIR = AGENT_ROOT / "sources/reviews"
DRAFT_DIR = AGENT_ROOT / "sources/translation_drafts"


INDICATOR_KEYWORDS = {
    "rsi": "RSI",
    "relative strength index": "RSI",
    "ema": "EMA",
    "exponential moving average": "EMA",
    "sma": "SMA",
    "moving average": "MA",
    "bollinger": "Bollinger Bands",
    "bbands": "Bollinger Bands",
    "adx": "ADX",
    "atr": "ATR",
    "macd": "MACD",
    "volume": "Volume",
    "breakout": "Breakout",
    "pullback": "Pullback",
    "mean reversion": "Mean Reversion",
    "trend": "Trend",
    "momentum": "Momentum",
}

RISK_PATTERNS = {
    "future_leakage": [
        r"\bshift\s*\(\s*-\d+",
        r"future\s+(data|candle|price)",
        r"repaint",
    ],
    "execution_risk": [
        r"guaranteed",
        r"no\s+loss",
        r"martingale",
        r"grid\s+doubling",
        r"50x",
        r"100x",
        r"no\s+stop",
    ],
    "code_execution_risk": [
        r"os\.system",
        r"subprocess",
        r"eval\s*\(",
        r"exec\s*\(",
        r"pip\s+install",
        r"curl\s+.*\|\s*(bash|sh)",
    ],
    "overfit_story": [
        r"best\s+settings",
        r"optimized\s+for",
        r"only\s+works\s+on",
        r"perfect\s+backtest",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", action="append", help="Review only the selected source id. Repeatable.")
    parser.add_argument("--max-chars", type=int, default=20000)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def strip_html(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def read_source_text(entry: dict[str, Any], max_chars: int) -> tuple[str, str | None]:
    snapshot = entry.get("snapshot")
    if not snapshot:
        return "", "no_snapshot"
    path = REPO_ROOT / snapshot["path"]
    if not path.exists():
        return "", "snapshot_missing"
    raw = path.read_bytes()[:max_chars]
    content_type = (snapshot.get("content_type") or "").lower()
    if "html" in content_type or path.suffix in {"", ".snapshot", ".html"}:
        return strip_html(raw), None
    return raw.decode("utf-8", errors="replace"), None


def detect_indicators(text: str) -> list[str]:
    lowered = text.lower()
    found = {label for keyword, label in INDICATOR_KEYWORDS.items() if keyword in lowered}
    return sorted(found)


def detect_risks(text: str) -> dict[str, list[str]]:
    risks: dict[str, list[str]] = {}
    lowered = text.lower()
    for family, patterns in RISK_PATTERNS.items():
        hits = []
        for pattern in patterns:
            if re.search(pattern, lowered):
                hits.append(pattern)
        if hits:
            risks[family] = hits
    return risks


def infer_strategy_family(indicators: list[str], text: str) -> str:
    lowered = text.lower()
    if "mean reversion" in lowered or "Bollinger Bands" in indicators:
        return "mean_reversion"
    if "pullback" in lowered:
        return "trend_pullback"
    if "breakout" in lowered:
        return "breakout"
    if {"EMA", "MA"} & set(indicators) or "trend" in lowered or "momentum" in lowered:
        return "trend_following"
    return "unknown"


def review_status(entry: dict[str, Any], text: str, risks: dict[str, list[str]], missing_reason: str | None) -> str:
    if entry.get("trust_level") == "D":
        return "rejected_source"
    if missing_reason:
        return "needs_snapshot"
    if risks.get("code_execution_risk") or risks.get("future_leakage"):
        return "blocked_pending_manual_review"
    if risks.get("execution_risk"):
        return "needs_manual_risk_review"
    if not text:
        return "needs_snapshot"
    return "approved_for_translation_draft"


def build_review(entry: dict[str, Any], max_chars: int) -> dict[str, Any]:
    text, missing_reason = read_source_text(entry, max_chars)
    indicators = detect_indicators(text)
    risks = detect_risks(text)
    family = infer_strategy_family(indicators, text)
    status = review_status(entry, text, risks, missing_reason)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    excerpt = text[:1000]
    return {
        "id": entry["id"],
        "reviewed_at_utc": now,
        "source": {
            "title": entry.get("title"),
            "location": entry.get("location"),
            "kind": entry.get("kind"),
            "trust_level": entry.get("trust_level"),
            "license": entry.get("license"),
            "snapshot": entry.get("snapshot"),
        },
        "status": status,
        "missing_reason": missing_reason,
        "detected_indicators": indicators,
        "inferred_strategy_family": family,
        "risk_flags": risks,
        "summary": {
            "hypothesis": strategy_hypothesis(family, indicators),
            "market_fit": "BTC/ETH spot or USDT-M futures only until local tests prove otherwise.",
            "translation_boundary": "Translate idea into isolated local code; do not run or import external source code.",
            "excerpt": excerpt,
        },
        "required_reviews": [
            "license_review",
            "future_leakage_review",
            "overfit_story_review",
            "local_btc_eth_reproducibility_check",
        ],
    }


def strategy_hypothesis(family: str, indicators: list[str]) -> str:
    if family == "trend_pullback":
        return "A trend filter plus pullback/resume confirmation may improve entry timing versus entering on the first regime signal."
    if family == "trend_following":
        return "A trend-following filter may avoid countertrend trades when BTC/ETH direction is persistent."
    if family == "mean_reversion":
        return "A mean-reversion setup may work only in range regimes and should be disabled during strong trends."
    if family == "breakout":
        return "A breakout setup may work during volatility expansion but must be tested for false breakouts and fee drag."
    if indicators:
        return f"Detected {', '.join(indicators)}. Treat as a generic indicator idea until reviewed."
    return "No clear strategy hypothesis detected from the available source text."


def build_translation_draft(review: dict[str, Any]) -> dict[str, Any]:
    family = review["inferred_strategy_family"]
    indicators = review["detected_indicators"]
    allowed = review["status"] == "approved_for_translation_draft"
    return {
        "id": f"{review['id']}-draft",
        "source_review_id": review["id"],
        "status": "draft_ready" if allowed else "blocked",
        "strategy_family": family,
        "detected_indicators": indicators,
        "target_market": "BTC/ETH USDT-M futures research sandbox",
        "target_timeframes": ["1m", "5m", "15m", "1h"],
        "default_risk": {
            "max_live_permission": "none",
            "initial_leverage_cap": 1.0,
            "requires_stoploss": True,
            "requires_recursive_analysis": True,
            "requires_lookahead_analysis": True,
        },
        "implementation_plan": [
            "Create a research-only Freqtrade strategy under user_data/strategies/research_generated.",
            "Use local indicators only; do not import external code.",
            "Backtest on BTC/ETH with fees and benchmark comparison.",
            "Keep initial classification no higher than research_candidate.",
        ],
        "blocked_reason": None if allowed else review["status"],
    }


def update_registry_status(registry: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    statuses = {item["id"]: item["status"] for item in reviews}
    for entry in registry.get("sources", []):
        if entry.get("id") in statuses:
            entry["review_status"] = statuses[entry["id"]]
            entry["last_reviewed_at_utc"] = next(
                item["reviewed_at_utc"] for item in reviews if item["id"] == entry["id"]
            )


def main() -> None:
    args = parse_args()
    registry = load_json(SOURCE_REGISTRY)
    selected = set(args.source_id or [])
    reviews: list[dict[str, Any]] = []
    for entry in registry.get("sources", []):
        if selected and entry["id"] not in selected:
            continue
        if not selected and entry.get("kind") == "internal_strategy_library":
            continue
        review = build_review(entry, args.max_chars)
        reviews.append(review)
        review_path = REVIEW_DIR / f"{entry['id']}.review.json"
        draft_path = DRAFT_DIR / f"{entry['id']}.draft.json"
        write_json(review_path, review)
        write_json(draft_path, build_translation_draft(review))
        print(f"{entry['id']}: {review['status']} -> {review_path.relative_to(REPO_ROOT)}")

    update_registry_status(registry, reviews)
    SOURCE_REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
