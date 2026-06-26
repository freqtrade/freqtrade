#!/usr/bin/env python3
"""Query the local price-action knowledge cards with simple keyword matching."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = REPO_ROOT / "user_data/strategy_research/knowledge"
CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="Search words, for example: breakout pullback crypto")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    return parser.parse_args()


def load_cards() -> list[dict[str, Any]]:
    cards = []
    for path in sorted(CARDS_DIR.glob("*.json")):
        cards.append(json.loads(path.read_text(encoding="utf-8")))
    return cards


def tokenize(value: str) -> list[str]:
    return [item.lower() for item in re.findall(r"[\w\u4e00-\u9fff]+", value)]


def score_card(card: dict[str, Any], terms: list[str]) -> int:
    haystack = " ".join(
        [
            card.get("id", ""),
            card.get("title", ""),
            card.get("category", ""),
            card.get("strategy_family", ""),
            " ".join(card.get("concepts", [])),
            card.get("knowledge", ""),
            card.get("strategy_hypothesis", ""),
            json.dumps(card.get("freqtrade_translation", {}), ensure_ascii=False),
            json.dumps(card.get("source_quality", {}), ensure_ascii=False),
            json.dumps(card.get("risk_notes", []), ensure_ascii=False),
            json.dumps(card.get("avoid_rules", []), ensure_ascii=False),
        ]
    ).lower()
    return sum(haystack.count(term) for term in terms)


def main() -> None:
    args = parse_args()
    terms = tokenize(" ".join(args.query))
    cards = load_cards()
    matches = []
    for card in cards:
        score = score_card(card, terms)
        if score > 0:
            matches.append((score, card))
    matches.sort(key=lambda item: (-item[0], item[1]["id"]))
    selected = [card for _, card in matches[: args.limit]]
    if args.json:
        print(json.dumps({"query": " ".join(args.query), "matches": selected}, indent=2, ensure_ascii=False))
        return
    print(f"# Price Action Knowledge Query: {' '.join(args.query)}")
    print("")
    if not selected:
        print("No matching cards. Build the knowledge base first or add more cards.")
        return
    for card in selected:
        print(f"## {card['title']}")
        print("")
        print(f"- Card: `{card['id']}`")
        print(f"- Category: `{card.get('category', 'unknown')}`")
        strategy_family = (card.get("freqtrade_translation") or {}).get("strategy_family", "unknown")
        print(f"- Strategy family: `{strategy_family}`")
        print(f"- Concepts: `{', '.join(card.get('concepts', []))}`")
        source_quality = card.get("source_quality") or {}
        print(f"- Source quality: `{source_quality.get('level', 'unknown')}`")
        print(f"- Knowledge: {card.get('knowledge')}")
        print(f"- Strategy hypothesis: {card.get('strategy_hypothesis')}")
        rules = (card.get("freqtrade_translation") or {}).get("entry_rules", [])
        if rules:
            print(f"- Freqtrade rules: {'; '.join(rules)}")
        risk_notes = card.get("risk_notes") or []
        if risk_notes:
            print(f"- Risk notes: {'; '.join(risk_notes[:2])}")
        print("")


if __name__ == "__main__":
    main()
