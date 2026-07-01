#!/usr/bin/env python3
"""Build an external source discovery queue for strategy research."""

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
DISCOVERY_DIR = AGENT_ROOT / "source_discovery"
LATEST_JSON = DISCOVERY_DIR / "latest_source_discovery.json"
LATEST_MD = DISCOVERY_DIR / "latest_source_discovery.md"
MAX_FETCH_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", help="Register an external source URL. Repeatable.")
    parser.add_argument("--title", action="append", help="Title for each --url. Repeatable and positional by order.")
    parser.add_argument("--kind", default="github_file", choices=["web_article", "github_repo", "github_file", "paper", "research_report", "forum_post"])
    parser.add_argument("--author", default="unknown")
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--trust-level", default="C", choices=["B", "C", "D"])
    parser.add_argument("--fetch", action="store_true", help="Fetch bounded snapshots for newly registered URLs.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sources": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value[:80] or "external-source"


def source_id(url: str, title: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{slugify(title)}-{digest}"


def fetch_snapshot(url: str, source_key: str) -> dict[str, Any]:
    SOURCE_INBOX.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "local-strategy-researcher/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - bounded user-requested fetch.
        raw = response.read(MAX_FETCH_BYTES + 1)
        truncated = len(raw) > MAX_FETCH_BYTES
        raw = raw[:MAX_FETCH_BYTES]
        content_type = response.headers.get("content-type", "unknown")
    snapshot_path = SOURCE_INBOX / f"{source_key}.snapshot"
    snapshot_path.write_bytes(raw)
    return {
        "path": rel(snapshot_path),
        "bytes": len(raw),
        "truncated": truncated,
        "content_type": content_type,
    }


def register_urls(registry: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    urls = args.url or []
    titles = args.title or []
    registered = []
    if titles and len(titles) != len(urls):
        raise SystemExit("--title must be repeated once per --url when provided.")
    existing = {item.get("id"): item for item in registry.get("sources", [])}
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index, url in enumerate(urls):
        title = titles[index] if titles else url.rsplit("/", 1)[-1] or url
        key = source_id(url, title)
        snapshot = fetch_snapshot(url, key) if args.fetch else existing.get(key, {}).get("snapshot")
        entry = {
            "id": key,
            "trust_level": args.trust_level,
            "kind": args.kind,
            "title": title,
            "location": url,
            "author": args.author,
            "license": args.license,
            "status": "quarantined_for_review",
            "registered_at_utc": existing.get(key, {}).get("registered_at_utc", now),
            "snapshot": snapshot,
            "allowed_actions": ["read", "summarize", "translate_to_isolated_strategy"],
            "blocked_actions": ["install_dependencies", "run_external_code", "import_external_code", "live_trade", "modify_live_config"],
            "review_requirements": ["license_review", "future_leakage_review", "overfit_story_review", "local_btc_eth_reproducibility_check"],
            "notes": "External sources are untrusted. They may inspire isolated research strategies but must not be executed directly.",
        }
        if existing.get(key, {}).get("review_status"):
            entry["review_status"] = existing[key]["review_status"]
            entry["last_reviewed_at_utc"] = existing[key].get("last_reviewed_at_utc")
        existing[key] = entry
        registered.append(entry)
    registry["sources"] = list(existing.values())
    return registered


def recommended_action(entry: dict[str, Any]) -> tuple[str, str]:
    if entry.get("kind") == "internal_strategy_library":
        return "internal", "No external-source action required."
    if not entry.get("snapshot"):
        command = (
            "./.venv/bin/python user_data/strategy_research/ingest_source.py "
            f"--url {json.dumps(entry.get('location'))} --title {json.dumps(entry.get('title'))} "
            f"--kind {entry.get('kind', 'web_article')} --fetch"
        )
        return "fetch_snapshot", command
    if not entry.get("review_status"):
        return "review_source", f"./.venv/bin/python user_data/strategy_research/review_sources.py --source-id {entry['id']}"
    if entry.get("review_status") == "approved_for_translation_draft":
        return "translate_to_strategy", "./.venv/bin/python user_data/strategy_research/generate_source_strategies.py"
    if entry.get("review_status") in {"needs_snapshot", "blocked_pending_manual_review", "needs_manual_risk_review"}:
        return "manual_review", f"Inspect user_data/strategy_research/sources/reviews/{entry['id']}.review.json"
    return "hold", "No automatic action."


def build_payload(registry: dict[str, Any], registered: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    action_counts: dict[str, int] = {}
    for entry in registry.get("sources", []):
        action, command = recommended_action(entry)
        action_counts[action] = action_counts.get(action, 0) + 1
        if action == "internal":
            continue
        candidates.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "kind": entry.get("kind"),
                "trust_level": entry.get("trust_level"),
                "status": entry.get("status"),
                "review_status": entry.get("review_status"),
                "has_snapshot": bool(entry.get("snapshot")),
                "snapshot_bytes": (entry.get("snapshot") or {}).get("bytes"),
                "recommended_action": action,
                "next_command": command,
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "registered_this_run": [item.get("id") for item in registered],
        "candidate_count": len(candidates),
        "action_summary": [{"action": key, "count": action_counts[key]} for key in sorted(action_counts)],
        "candidates": sorted(candidates, key=lambda item: (item["recommended_action"], item["id"] or "")),
        "safety_policy": {
            "allowed": ["read", "summarize", "translate_to_isolated_strategy"],
            "blocked": ["install_dependencies", "run_external_code", "import_external_code", "live_trade", "modify_live_config"],
            "max_snapshot_bytes": MAX_FETCH_BYTES,
        },
        "source_artifacts": {"source_registry": rel(SOURCE_REGISTRY)},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# External Source Discovery",
        "",
        f"- Generated UTC: `{payload['generated_at_utc']}`",
        f"- Candidates: `{payload['candidate_count']}`",
        f"- Registered this run: `{', '.join(payload['registered_this_run']) or 'none'}`",
        "",
        "## Action Summary",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    for item in payload["action_summary"]:
        lines.append(f"| {item['action']} | {item['count']} |")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Source | Kind | Trust | Snapshot | Review | Action | Next Command |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for item in payload["candidates"]:
        lines.append(
            "| {id} | {kind} | {trust_level} | {has_snapshot} | {review_status} | {recommended_action} | `{next_command}` |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Safety Policy",
            "",
            "- External sources are always untrusted until reviewed.",
            "- Snapshots are bounded and stored locally; external code is never imported or executed.",
            "- Translation must produce isolated local research strategies before any backtest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    registry = load_json(SOURCE_REGISTRY)
    registered = register_urls(registry, args)
    if registered:
        save_json(SOURCE_REGISTRY, registry)
    payload = build_payload(registry, registered)
    save_json(LATEST_JSON, payload)
    write_markdown(LATEST_MD, payload)
    print(f"Wrote {rel(LATEST_JSON)}")
    print(f"Wrote {rel(LATEST_MD)}")
    if registered:
        print("Registered " + ", ".join(item["id"] for item in registered))


if __name__ == "__main__":
    main()
