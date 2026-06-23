#!/usr/bin/env python3
"""Register an external strategy source in the isolated research inbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
SOURCE_REGISTRY = AGENT_ROOT / "source_registry.json"
SOURCE_INBOX = AGENT_ROOT / "sources/inbox"
MAX_FETCH_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="External source URL.")
    parser.add_argument("--title", required=True, help="Human-readable source title.")
    parser.add_argument(
        "--kind",
        default="web_article",
        choices=["web_article", "github_repo", "github_file", "paper", "research_report", "forum_post"],
    )
    parser.add_argument("--author", default="unknown")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--trust-level", default="C", choices=["B", "C", "D"])
    parser.add_argument("--fetch", action="store_true", help="Fetch and store a bounded raw snapshot.")
    return parser.parse_args()


def load_registry() -> dict[str, Any]:
    with SOURCE_REGISTRY.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_registry(registry: dict[str, Any]) -> None:
    SOURCE_REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value[:80] or "external-source"


def source_id(url: str, title: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{slugify(title)}-{digest}"


def fetch_snapshot(url: str, source_key: str) -> dict[str, Any]:
    SOURCE_INBOX.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "local-strategy-researcher/0.1"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - bounded user-requested fetch.
        raw = response.read(MAX_FETCH_BYTES + 1)
        truncated = len(raw) > MAX_FETCH_BYTES
        raw = raw[:MAX_FETCH_BYTES]
        content_type = response.headers.get("content-type", "unknown")
    snapshot_path = SOURCE_INBOX / f"{source_key}.snapshot"
    snapshot_path.write_bytes(raw)
    return {
        "path": str(snapshot_path.relative_to(REPO_ROOT)),
        "bytes": len(raw),
        "truncated": truncated,
        "content_type": content_type,
    }


def main() -> None:
    args = parse_args()
    registry = load_registry()
    key = source_id(args.url, args.title)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    snapshot = fetch_snapshot(args.url, key) if args.fetch else None
    entry = {
        "id": key,
        "trust_level": args.trust_level,
        "kind": args.kind,
        "title": args.title,
        "location": args.url,
        "author": args.author,
        "license": args.license,
        "status": "quarantined_for_review",
        "registered_at_utc": now,
        "snapshot": snapshot,
        "allowed_actions": [
            "read",
            "summarize",
            "translate_to_isolated_strategy"
        ],
        "blocked_actions": [
            "install_dependencies",
            "run_external_code",
            "import_external_code",
            "live_trade",
            "modify_live_config"
        ],
        "review_requirements": [
            "license_review",
            "future_leakage_review",
            "overfit_story_review",
            "local_btc_eth_reproducibility_check"
        ],
        "notes": "External sources are untrusted. They may inspire isolated research strategies but must not be executed directly.",
    }

    existing = [item for item in registry.get("sources", []) if item.get("id") != key]
    existing.append(entry)
    registry["sources"] = existing
    save_registry(registry)
    print(f"Registered {key}")
    if snapshot:
        print(f"Snapshot {snapshot['path']} ({snapshot['bytes']} bytes)")


if __name__ == "__main__":
    main()
