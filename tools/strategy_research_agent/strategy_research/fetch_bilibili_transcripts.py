#!/usr/bin/env python3
"""Fetch Bilibili AI subtitles for the local price-action course manifest.

This script uses the user's authenticated browser cookies only to request
subtitle metadata and subtitle JSON. It does not download video files.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yt_dlp.cookies import extract_cookies_from_browser


REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = REPO_ROOT / "user_data/strategy_research/knowledge"
BILIBILI_ROOT = KNOWLEDGE_ROOT / "raw_sources/bilibili"
MANIFEST = BILIBILI_ROOT / "bilibili_price_action_course_manifest.json"
TRANSCRIPTS_DIR = BILIBILI_ROOT / "transcripts"
PRIVATE_DIR = BILIBILI_ROOT / "private"
REPORT_JSON = BILIBILI_ROOT / "bilibili_transcript_fetch_report.json"
REPORT_MD = BILIBILI_ROOT / "bilibili_transcript_fetch_report.md"
PLAYER_API = "https://api.bilibili.com/x/player/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="chrome", help="Browser for yt-dlp cookie extraction.")
    parser.add_argument("--profile", help="Optional browser profile name, for example 'Default' or 'Profile 1'.")
    parser.add_argument("--cookies", help="Optional Netscape cookies.txt file.")
    parser.add_argument("--limit", type=int, help="Fetch only the first N pages.")
    parser.add_argument("--pages", help="Comma-separated page numbers to fetch, for example 40,41,44.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per page when subtitle URL is temporarily absent.")
    parser.add_argument("--language", default="ai-zh")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {rel(MANIFEST)}. Run build_price_action_knowledge_base.py first.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_cookiejar(args: argparse.Namespace) -> http.cookiejar.CookieJar:
    if args.cookies:
        cookie_path = Path(args.cookies).expanduser()
        jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
        jar.load(ignore_discard=True, ignore_expires=True)
        return jar
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    return extract_cookies_from_browser(args.browser, profile=args.profile)


def opener_from_cookiejar(jar: http.cookiejar.CookieJar) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def request_json(opener: urllib.request.OpenerDirector, url: str, referer: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 local-strategy-researcher/0.1",
            "Referer": referer,
        },
    )
    with opener.open(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def sanitize(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:90] or "untitled"


def format_ts(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def subtitle_to_srt(subtitle: dict[str, Any]) -> str:
    lines: list[str] = []
    body = subtitle.get("body") or []
    for index, item in enumerate(body, start=1):
        content = str(item.get("content", "")).replace("\r", " ").strip()
        if not content:
            continue
        lines.append(str(index))
        lines.append(f"{format_ts(float(item.get('from', 0)))} --> {format_ts(float(item.get('to', item.get('from', 0))))}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def subtitle_to_text(subtitle: dict[str, Any]) -> str:
    body = subtitle.get("body") or []
    return "\n".join(str(item.get("content", "")).strip() for item in body if str(item.get("content", "")).strip()) + "\n"


def subtitle_url_from_player(player: dict[str, Any], language: str) -> str | None:
    subtitles = ((player.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
    preferred = [item for item in subtitles if item.get("lan") == language]
    selected = preferred[0] if preferred else (subtitles[0] if subtitles else None)
    if not selected:
        return None
    url = selected.get("subtitle_url")
    if not url:
        return None
    if url.startswith("//"):
        return "https:" + url
    return url


def fetch_page(opener: urllib.request.OpenerDirector, manifest: dict[str, Any], page: dict[str, Any], language: str) -> dict[str, Any]:
    bvid = manifest["bvid"]
    page_no = int(page["page"])
    title = str(page.get("part") or f"p{page_no}")
    referer = f"https://www.bilibili.com/video/{bvid}/?p={page_no}"
    player_url = f"{PLAYER_API}?bvid={bvid}&cid={page['cid']}"
    player = request_json(opener, player_url, referer)
    subtitle_url = subtitle_url_from_player(player, language)
    result: dict[str, Any] = {
        "page": page_no,
        "cid": page.get("cid"),
        "title": title,
        "status": "missing_subtitle",
        "language": language,
        "subtitle_url_present": bool(subtitle_url),
        "files": {},
    }
    if not subtitle_url:
        cached = sorted(TRANSCRIPTS_DIR.glob(f"p{page_no:03d}_*.txt"))
        if cached:
            result["status"] = "cached"
            result["files"] = {"txt": rel(cached[0])}
            result["line_count"] = len([line for line in cached[0].read_text(encoding="utf-8").splitlines() if line.strip()])
            result["note"] = "Subtitle API did not return a URL this run, but a previous local transcript exists."
            return result
        result["need_login_subtitle"] = bool((player.get("data") or {}).get("need_login_subtitle"))
        result["subtitle_count"] = len((((player.get("data") or {}).get("subtitle") or {}).get("subtitles") or []))
        return result

    subtitle = request_json(opener, subtitle_url, referer)
    stem = f"p{page_no:03d}_{sanitize(title)}"
    json_path = TRANSCRIPTS_DIR / f"{stem}.json"
    srt_path = TRANSCRIPTS_DIR / f"{stem}.srt"
    txt_path = TRANSCRIPTS_DIR / f"{stem}.txt"
    json_path.write_text(json.dumps(subtitle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    srt_path.write_text(subtitle_to_srt(subtitle), encoding="utf-8")
    txt_path.write_text(subtitle_to_text(subtitle), encoding="utf-8")
    result["status"] = "fetched"
    result["line_count"] = len(subtitle.get("body") or [])
    result["files"] = {"json": rel(json_path), "srt": rel(srt_path), "txt": rel(txt_path)}
    return result


def write_report(report: dict[str, Any]) -> None:
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Bilibili Transcript Fetch Report",
        "",
        f"- Generated UTC: `{report['generated_at_utc']}`",
        f"- Pages total: `{report['pages_total']}`",
        f"- Fetched: `{report['fetched_count']}`",
        f"- Cached from previous runs: `{report['cached_count']}`",
        f"- Missing subtitle: `{report['missing_count']}`",
        "",
        "| Page | Status | Lines | Title | TXT |",
        "|---:|---|---:|---|---|",
    ]
    for item in report["pages"]:
        txt = (item.get("files") or {}).get("txt", "")
        lines.append(f"| {item['page']} | {item['status']} | {item.get('line_count', 0)} | {item['title']} | `{txt}` |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    pages = manifest.get("pages") or []
    if args.pages:
        selected_pages = {int(item.strip()) for item in args.pages.split(",") if item.strip()}
        pages = [page for page in pages if int(page.get("page", 0)) in selected_pages]
    if args.limit:
        pages = pages[: args.limit]
    jar = load_cookiejar(args)
    opener = opener_from_cookiejar(jar)
    results = []
    for page in pages:
        result = None
        for attempt in range(1, max(args.retries, 1) + 1):
            try:
                result = fetch_page(opener, manifest, page, args.language)
            except Exception as exc:  # noqa: BLE001 - write page-level failure and keep going.
                result = {
                    "page": page.get("page"),
                    "cid": page.get("cid"),
                    "title": page.get("part"),
                    "status": "error",
                    "error": str(exc),
                    "files": {},
                }
            result["attempts"] = attempt
            if result["status"] != "missing_subtitle":
                break
        results.append(result)
        print(f"p{result['page']:03d}: {result['status']} {result.get('title', '')}")
    report = {
        "generated_at_utc": now_utc(),
        "manifest": rel(MANIFEST),
        "pages_total": len(results),
        "fetched_count": sum(1 for item in results if item["status"] == "fetched"),
        "cached_count": sum(1 for item in results if item["status"] == "cached"),
        "missing_count": sum(1 for item in results if item["status"] == "missing_subtitle"),
        "error_count": sum(1 for item in results if item["status"] == "error"),
        "pages": results,
    }
    write_report(report)
    print(f"Wrote {rel(REPORT_MD)}")


if __name__ == "__main__":
    main()
