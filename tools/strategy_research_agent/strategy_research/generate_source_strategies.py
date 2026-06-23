#!/usr/bin/env python3
"""Generate isolated Freqtrade strategies from reviewed translation drafts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
DRAFT_DIR = AGENT_ROOT / "sources/translation_drafts"
GENERATED_DIR = REPO_ROOT / "user_data/strategies/research_generated"
GENERATED_FILE = GENERATED_DIR / "source_translated_strategies.py"
GENERATED_REGISTRY = AGENT_ROOT / "experiments/source_translated_registry.json"
GENERATED_EXPERIMENT = AGENT_ROOT / "experiments/source_translated_experiment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-id", action="append", help="Generate only the selected draft id. Repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_drafts(selected: set[str]) -> list[dict[str, Any]]:
    drafts = []
    for path in sorted(DRAFT_DIR.glob("*.draft.json")):
        with path.open("r", encoding="utf-8") as handle:
            draft = json.load(handle)
        if selected and draft["id"] not in selected:
            continue
        if draft.get("status") != "draft_ready":
            continue
        drafts.append(draft)
    return drafts


def safe_class_piece(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value)
    text = "".join(word[:1].upper() + word[1:] for word in words)
    return text or "SourceDraft"


def class_name(draft: dict[str, Any]) -> str:
    return f"SourceTranslated{safe_class_piece(draft['source_review_id'])}Strategy"


def build_trend_pullback_class(cls: str, draft: dict[str, Any]) -> list[str]:
    tag = draft["source_review_id"][:40].replace("-", "_")
    return [
        "",
        f"class {cls}(BtcEthFuturesRegime10xOneMinuteStrategy):",
        f"    \"\"\"Research-only source-translated strategy from {draft['source_review_id']}.\"\"\"",
        "",
        "    minimal_roi = {\"0\": 0.012, \"60\": 0.006, \"240\": 0}",
        "    stoploss = -0.012",
        "    startup_candle_count = 2400",
        "",
        "    def leverage(",
        "        self,",
        "        pair: str,",
        "        current_time: datetime,",
        "        current_rate: float,",
        "        proposed_leverage: float,",
        "        max_leverage: float,",
        "        entry_tag: str | None,",
        "        side: str,",
        "        **kwargs,",
        "    ) -> float:",
        "        return min(1.0, max_leverage)",
        "",
        "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe = super().populate_indicators(dataframe, metadata)",
        "        dataframe[\"src_ema_fast\"] = ta.EMA(dataframe, timeperiod=20)",
        "        dataframe[\"src_ema_mid\"] = ta.EMA(dataframe, timeperiod=60)",
        "        dataframe[\"src_ema_slow\"] = ta.EMA(dataframe, timeperiod=240)",
        "        dataframe[\"src_rsi\"] = ta.RSI(dataframe, timeperiod=14)",
        "        dataframe[\"src_ret_5m\"] = dataframe[\"close\"] / dataframe[\"close\"].shift(5) - 1.0",
        "        dataframe[\"src_pullback_long\"] = dataframe[\"low\"].rolling(30).min() <= dataframe[\"src_ema_mid\"] * 1.0015",
        "        dataframe[\"src_pullback_short\"] = dataframe[\"high\"].rolling(30).max() >= dataframe[\"src_ema_mid\"] * 0.9985",
        "        dataframe[\"src_resume_long\"] = (",
        "            (dataframe[\"close\"] > dataframe[\"src_ema_fast\"])",
        "            & (dataframe[\"close\"] > dataframe[\"open\"])",
        "            & (dataframe[\"src_rsi\"].between(45, 65))",
        "            & (dataframe[\"src_ret_5m\"] > 0.0005)",
        "        )",
        "        dataframe[\"src_resume_short\"] = (",
        "            (dataframe[\"close\"] < dataframe[\"src_ema_fast\"])",
        "            & (dataframe[\"close\"] < dataframe[\"open\"])",
        "            & (dataframe[\"src_rsi\"].between(35, 55))",
        "            & (dataframe[\"src_ret_5m\"] < -0.0005)",
        "        )",
        "        return dataframe",
        "",
        "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe.loc[",
        "            (",
        "                dataframe[\"risk_allowed\"]",
        "                & dataframe[\"trend_up_regime\"]",
        "                & (dataframe[\"close\"] > dataframe[\"src_ema_slow\"])",
        "                & dataframe[\"src_pullback_long\"]",
        "                & dataframe[\"src_resume_long\"]",
        "                & (dataframe[\"volume\"] > 0)",
        "            ),",
        "            [\"enter_long\", \"enter_tag\"],",
        f"        ] = (1, \"src_{tag}_long\")",
        "",
        "        dataframe.loc[",
        "            (",
        "                dataframe[\"risk_allowed\"]",
        "                & dataframe[\"trend_down_regime\"]",
        "                & (dataframe[\"close\"] < dataframe[\"src_ema_slow\"])",
        "                & dataframe[\"src_pullback_short\"]",
        "                & dataframe[\"src_resume_short\"]",
        "                & (dataframe[\"volume\"] > 0)",
        "            ),",
        "            [\"enter_short\", \"enter_tag\"],",
        f"        ] = (1, \"src_{tag}_short\")",
        "        return dataframe",
        "",
    ]


def build_mean_reversion_class(cls: str, draft: dict[str, Any]) -> list[str]:
    tag = draft["source_review_id"][:40].replace("-", "_")
    return [
        "",
        f"class {cls}(BtcEthFuturesRegime10xOneMinuteStrategy):",
        f"    \"\"\"Research-only source-translated mean-reversion strategy from {draft['source_review_id']}.\"\"\"",
        "",
        "    minimal_roi = {\"0\": 0.006, \"90\": 0.003, \"240\": 0}",
        "    stoploss = -0.008",
        "    startup_candle_count = 2400",
        "",
        "    def leverage(",
        "        self,",
        "        pair: str,",
        "        current_time: datetime,",
        "        current_rate: float,",
        "        proposed_leverage: float,",
        "        max_leverage: float,",
        "        entry_tag: str | None,",
        "        side: str,",
        "        **kwargs,",
        "    ) -> float:",
        "        return min(1.0, max_leverage)",
        "",
        "    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe = super().populate_indicators(dataframe, metadata)",
        "        bb = ta.BBANDS(dataframe, timeperiod=120)",
        "        dataframe[\"src_bb_upper\"] = bb[\"upperband\"]",
        "        dataframe[\"src_bb_middle\"] = bb[\"middleband\"]",
        "        dataframe[\"src_bb_lower\"] = bb[\"lowerband\"]",
        "        dataframe[\"src_bb_width\"] = (dataframe[\"src_bb_upper\"] - dataframe[\"src_bb_lower\"]) / dataframe[\"src_bb_middle\"]",
        "        dataframe[\"src_rsi\"] = ta.RSI(dataframe, timeperiod=14)",
        "        dataframe[\"src_volume_mean\"] = dataframe[\"volume\"].rolling(120).mean()",
        "        dataframe[\"src_range_ok\"] = (",
        "            dataframe[\"range_regime\"]",
        "            & ~dataframe[\"high_vol_regime\"]",
        "            & (dataframe[\"src_bb_width\"].between(0.0015, 0.025))",
        "        )",
        "        return dataframe",
        "",
        "    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe.loc[",
        "            (",
        "                dataframe[\"risk_allowed\"]",
        "                & dataframe[\"src_range_ok\"]",
        "                & (dataframe[\"close\"] < dataframe[\"src_bb_lower\"])",
        "                & (dataframe[\"src_rsi\"] < 32)",
        "                & (dataframe[\"volume\"] > dataframe[\"src_volume_mean\"] * 0.6)",
        "            ),",
        "            [\"enter_long\", \"enter_tag\"],",
        f"        ] = (1, \"src_{tag}_mr_long\")",
        "",
        "        dataframe.loc[",
        "            (",
        "                dataframe[\"risk_allowed\"]",
        "                & dataframe[\"src_range_ok\"]",
        "                & (dataframe[\"close\"] > dataframe[\"src_bb_upper\"])",
        "                & (dataframe[\"src_rsi\"] > 68)",
        "                & (dataframe[\"volume\"] > dataframe[\"src_volume_mean\"] * 0.6)",
        "            ),",
        "            [\"enter_short\", \"enter_tag\"],",
        f"        ] = (1, \"src_{tag}_mr_short\")",
        "        return dataframe",
        "",
        "    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:",
        "        dataframe.loc[",
        "            (",
        "                ((dataframe[\"close\"] >= dataframe[\"src_bb_middle\"]) | ~dataframe[\"src_range_ok\"])",
        "                & (dataframe[\"volume\"] > 0)",
        "            ),",
        "            [\"exit_long\", \"exit_tag\"],",
        f"        ] = (1, \"src_{tag}_mr_long_exit\")",
        "",
        "        dataframe.loc[",
        "            (",
        "                ((dataframe[\"close\"] <= dataframe[\"src_bb_middle\"]) | ~dataframe[\"src_range_ok\"])",
        "                & (dataframe[\"volume\"] > 0)",
        "            ),",
        "            [\"exit_short\", \"exit_tag\"],",
        f"        ] = (1, \"src_{tag}_mr_short_exit\")",
        "        return dataframe",
        "",
    ]


def build_source(drafts: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        '"""Auto-generated source-translated strategy drafts.',
        "",
        "Do not edit by hand. Re-generate with user_data/strategy_research/generate_source_strategies.py.",
        "These classes are research-only and must not be promoted to live without manual approval.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import sys",
        "from datetime import datetime",
        "from pathlib import Path",
        "",
        "from pandas import DataFrame",
        "import talib.abstract as ta",
        "",
        "sys.path.append(str(Path(__file__).resolve().parents[1]))",
        "",
        "from btc_eth_risk_controlled_strategies import BtcEthFuturesRegime10xOneMinuteStrategy",
        "",
        "",
        f"GENERATED_AT_UTC = {generated_at!r}",
        "",
    ]
    entries: list[dict[str, Any]] = []
    for draft in drafts:
        cls = class_name(draft)
        family = draft.get("strategy_family")
        if family == "trend_pullback":
            lines.extend(build_trend_pullback_class(cls, draft))
        elif family == "mean_reversion":
            lines.extend(build_mean_reversion_class(cls, draft))
        else:
            continue
        entries.append(
            {
                "name": cls,
                "family": f"source-translated-{family}",
                "source": "source_translation_draft",
                "source_review_id": draft["source_review_id"],
                "hypothesis": f"Reviewed {family} idea translated into isolated BTC/ETH futures research code.",
                "risk_notes": "Generated from a reviewed source draft. Max initial class is research_candidate; live and dry-run promotion require manual approval plus recursive/lookahead checks.",
            }
        )
    return "\n".join(lines) + "\n", entries


def main() -> None:
    args = parse_args()
    drafts = load_drafts(set(args.draft_id or []))
    if not drafts:
        raise SystemExit("No draft_ready translation drafts found.")
    source, entries = build_source(drafts)
    if not entries:
        raise SystemExit("No supported draft families found for code generation.")
    if args.dry_run:
        print(source)
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FILE.write_text(source, encoding="utf-8")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_strategy_file": str(GENERATED_FILE.relative_to(REPO_ROOT)),
        "strategies": entries,
    }
    GENERATED_REGISTRY.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment = {
        "id": "source_translated_strategies",
        "title": "Reviewed source-translated strategy drafts",
        "profile_ref": "strategy_registry.json",
        "strategy_path": "user_data/strategies/research_generated",
        "timeframes": ["1m"],
        "timeranges": ["20240101-20260622"],
        "fee": 0.0005,
        "strategies": [item["name"] for item in entries],
        "checks": {
            "backtesting": True,
            "recursive_analysis": False,
            "lookahead_analysis": False,
        },
        "notes": [
            "Source-translated strategies are isolated and research-only.",
            "External code is not imported or executed.",
            "Dry-run/live promotion requires manual approval.",
        ],
    }
    GENERATED_EXPERIMENT.write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {GENERATED_FILE.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATED_REGISTRY.relative_to(REPO_ROOT)}")
    print(f"Wrote {GENERATED_EXPERIMENT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
