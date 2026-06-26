#!/usr/bin/env python3
"""Build a local price-action knowledge layer for the strategy researcher.

The builder only stores public metadata, bounded public web snapshots, and
short original knowledge cards. It intentionally does not download paid books,
paid videos, or pirated PDFs.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
KNOWLEDGE_ROOT = AGENT_ROOT / "knowledge"
RAW_SOURCES = KNOWLEDGE_ROOT / "raw_sources"
SNAPSHOTS = RAW_SOURCES / "web_snapshots"
BILIBILI_DIR = RAW_SOURCES / "bilibili"
BOOKS_DIR = RAW_SOURCES / "books_to_add"
CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards"
INDEX_DIR = KNOWLEDGE_ROOT / "index"
REPORT_MD = KNOWLEDGE_ROOT / "latest_price_action_knowledge_report.md"
REPORT_JSON = KNOWLEDGE_ROOT / "latest_price_action_knowledge_report.json"

BILIBILI_BVID = "BV1G2AgzkELu"
BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_PLAYER_API = "https://api.bilibili.com/x/player/v2"
MAX_SNAPSHOT_BYTES = 900_000


PUBLIC_WEB_SOURCES = [
    {
        "id": "brooks_trading_course_books",
        "title": "Al Brooks Price Action Trading Books",
        "url": "https://www.brookstradingcourse.com/price-action-trading-books/",
        "author": "Brooks Trading Course",
        "kind": "official_book_page",
        "license": "public webpage metadata; books are copyrighted",
    },
    {
        "id": "brooks_trading_course_home",
        "title": "Brooks Trading Course public overview",
        "url": "https://www.brookstradingcourse.com/",
        "author": "Brooks Trading Course",
        "kind": "official_course_page",
        "license": "public webpage metadata; paid course content is copyrighted",
    },
    {
        "id": "investopedia_price_action_intro",
        "title": "An Introduction to Price Action Trading Strategies",
        "url": "https://www.investopedia.com/articles/active-trading/110714/introduction-price-action-trading-strategies.asp",
        "author": "Investopedia",
        "kind": "web_article",
        "license": "public webpage snapshot for local research",
    },
    {
        "id": "investopedia_price_action_definition",
        "title": "Price Action: What It Is and How Stock Traders Use It",
        "url": "https://www.investopedia.com/terms/p/price-action.asp",
        "author": "Investopedia",
        "kind": "web_article",
        "license": "public webpage snapshot for local research",
    },
    {
        "id": "binance_academy_support_resistance",
        "title": "The Basics of Support and Resistance Explained",
        "url": "https://www.binance.com/en/academy/articles/the-basics-of-support-and-resistance-explained",
        "author": "Binance Academy",
        "kind": "crypto_web_article",
        "license": "public webpage snapshot for local research",
    },
    {
        "id": "coinmarketcap_support_resistance_zones",
        "title": "Technical Analysis 101: How to Find Support and Resistance Zones",
        "url": "https://coinmarketcap.com/academy/article/technical-analysis-101-how-to-find-support-and-resistance-zones",
        "author": "CoinMarketCap Academy",
        "kind": "crypto_web_article",
        "license": "public webpage snapshot for local research",
    },
    {
        "id": "kraken_technical_analysis_intro",
        "title": "A brief introduction to technical analysis",
        "url": "https://www.kraken.com/learn/introduction-to-technical-analysis",
        "author": "Kraken Learn",
        "kind": "crypto_web_article",
        "license": "public webpage snapshot for local research",
    },
    {
        "id": "phemex_crypto_price_action",
        "title": "How to Read Price Action in The Crypto Markets?",
        "url": "https://phemex.com/academy/crypto-price-action-trading",
        "author": "Phemex Academy",
        "kind": "crypto_web_article",
        "license": "public webpage snapshot for local research",
    },
]


STARTER_CARDS = [
    {
        "id": "pa_signal_bar_requires_context",
        "title": "信号K线必须放在背景里判断",
        "concepts": ["signal_bar", "context", "entry_confirmation"],
        "source_refs": ["bilibili_price_action_course", "brooks_trading_course_books"],
        "knowledge": "单根K线本身不是策略。先判断趋势、震荡、关键位置和最近买卖压力，再把信号K线当作入场触发器。",
        "strategy_hypothesis": "把方向判断和入场触发拆开：趋势/区间过滤只给方向，具体开仓必须等待1到2根短周期K线确认恢复或失败。",
        "freqtrade_translation": {
            "features": ["ema_slope", "range_position", "recent_swing_high_low", "signal_bar_body_ratio"],
            "entry_rules": ["方向过滤 + 信号K线 + 下一根K线确认，最多三个条件"],
            "avoid": ["只因出现锤子线/吞没线/大阳线就直接进场"],
        },
        "risk_notes": ["高杠杆下不要把信号K线的低点/高点机械当极近止损；要换算为杠杆后账户风险。"],
    },
    {
        "id": "pa_market_cycle_trend_range_transition",
        "title": "市场周期：趋势、震荡和过渡",
        "concepts": ["market_cycle", "trend", "trading_range", "regime"],
        "source_refs": ["bilibili_price_action_course", "investopedia_price_action_intro"],
        "knowledge": "策略必须先区分趋势行情、震荡行情和趋势转震荡的过渡段。同一入场形态在不同市场周期里含义不同。",
        "strategy_hypothesis": "Agent 生成策略时先选择行情类型，再选择策略族：趋势用回调恢复/突破延续，震荡用区间边界反转，过渡段降低频率或观望。",
        "freqtrade_translation": {
            "features": ["adx", "ema_spread", "atr_percentile", "range_width", "close_position_in_range"],
            "entry_rules": ["regime == trend 时禁用均值回归", "regime == range 时禁用追突破"],
            "avoid": ["同一套参数同时吃趋势和震荡"],
        },
        "risk_notes": ["行情切换是短周期策略回撤的高发区，必须保留冷却和最大回撤暂停。"],
    },
    {
        "id": "pa_breakout_needs_close_and_follow_through",
        "title": "突破需要收盘确认和后续跟进",
        "concepts": ["breakout", "follow_through", "false_breakout"],
        "source_refs": ["bilibili_price_action_course", "binance_academy_support_resistance"],
        "knowledge": "突破不是刺破价位。更可靠的突破通常需要收在关键位之外，并出现后续跟进K线或量能确认。",
        "strategy_hypothesis": "不要在触及阻力/支撑瞬间入场；测试收盘突破、回踩不破、或下一根继续推动三类确认。",
        "freqtrade_translation": {
            "features": ["prior_high_low", "close_break_distance", "volume_zscore", "followthrough_return"],
            "entry_rules": ["close > resistance + buffer", "next candle does not close back inside range"],
            "avoid": ["wick-only breakout", "breakout after stretched move without pullback"],
        },
        "risk_notes": ["假突破会快速反抽，高杠杆策略要限制追突破距离和滑点。"],
    },
    {
        "id": "pa_failed_breakout_as_reversal_seed",
        "title": "失败突破可以作为反向种子",
        "concepts": ["failed_breakout", "reversal", "trap"],
        "source_refs": ["bilibili_price_action_course", "coinmarketcap_support_resistance_zones"],
        "knowledge": "关键位突破后如果迅速收回区间，说明追突破的一方被套，反向移动可能更快。",
        "strategy_hypothesis": "设计假突破反转策略：先识别刺破关键位，再要求收回区间和反向小动量确认。",
        "freqtrade_translation": {
            "features": ["wick_outside_range", "close_back_inside_range", "reversal_body_ratio"],
            "entry_rules": ["刺破 + 收回 + 反向确认，不超过三个条件"],
            "avoid": ["在趋势极强时硬做反转"],
        },
        "risk_notes": ["反转策略必须有趋势过滤；强趋势里失败突破容易变成小回调。"],
    },
    {
        "id": "pa_pullback_resume_entry",
        "title": "趋势回调后的恢复入场",
        "concepts": ["pullback", "trend_resume", "entry_timing"],
        "source_refs": ["bilibili_price_action_course", "kraken_technical_analysis_intro"],
        "knowledge": "趋势策略不应一有方向就追。更好的位置通常是回调到动态支撑/压力附近，然后等待恢复。",
        "strategy_hypothesis": "用高一级时间框架定方向，低一级时间框架等待回调到均线/前低前高附近，再用恢复K线入场。",
        "freqtrade_translation": {
            "features": ["higher_tf_ema_slope", "pullback_to_ema", "resume_candle", "rsi_midzone"],
            "entry_rules": ["高周期方向 + 回调位置 + 恢复确认"],
            "avoid": ["离均线太远追单", "连续大K后追入"],
        },
        "risk_notes": ["回调策略交易次数会少，必须用多窗口样本验证，不要凭一段行情晋级。"],
    },
    {
        "id": "pa_support_resistance_are_zones",
        "title": "支撑阻力是区域，不是一根线",
        "concepts": ["support", "resistance", "zone", "liquidity"],
        "source_refs": ["binance_academy_support_resistance", "coinmarketcap_support_resistance_zones", "kraken_technical_analysis_intro"],
        "knowledge": "支撑阻力应被视为价格区域。短周期噪音和交易所盘口会导致针刺，机械按单一价格判断容易被洗掉。",
        "strategy_hypothesis": "把关键位转换成ATR或近期波动率宽度的区域，再测试区域内反应，而不是点位触发。",
        "freqtrade_translation": {
            "features": ["atr_zone_width", "swing_cluster", "touch_count", "rejection_from_zone"],
            "entry_rules": ["进入区域 + 拒绝形态 + 反向确认"],
            "avoid": ["一触线就买卖"],
        },
        "risk_notes": ["止损也要放在区域外并考虑杠杆后亏损，不能只看裸价格距离。"],
    },
    {
        "id": "pa_order_type_changes_strategy_meaning",
        "title": "订单类型会改变策略含义",
        "concepts": ["stop_order", "limit_order", "execution", "slippage"],
        "source_refs": ["bilibili_price_action_course"],
        "knowledge": "Stop order 更像突破/动量确认，limit order 更像回调/均值回归。信号相同但订单类型不同，策略实际暴露不同。",
        "strategy_hypothesis": "Freqtrade 回测里虽然不是盘口级撮合，也要把策略意图写清楚：追随突破、回调挂单、还是反转接刀。",
        "freqtrade_translation": {
            "features": ["entry_type_tag", "expected_slippage_bps", "spread_proxy"],
            "entry_rules": ["动量策略用确认后进场", "均值策略用区域反应后进场"],
            "avoid": ["用市价追所有信号"],
        },
        "risk_notes": ["短周期合约策略手续费和滑点可能吃掉大部分 edge，必须做压力测试。"],
    },
    {
        "id": "pa_crypto_needs_fee_and_24h_regime_adjustment",
        "title": "加密货币价格行为要额外处理手续费和24小时 regime",
        "concepts": ["crypto", "fees", "funding", "sessionless_market"],
        "source_refs": ["phemex_crypto_price_action", "investopedia_price_action_definition"],
        "knowledge": "传统价格行为多来自股票/期货盘中经验，加密货币是24小时交易，波动聚集、资金费率、周末流动性和交易所差异都会影响形态可靠性。",
        "strategy_hypothesis": "每个价格行为策略都必须加手续费/滑点压力测试、时间段切片和 BTC/ETH 分资产验证。",
        "freqtrade_translation": {
            "features": ["hour_of_day", "day_of_week", "funding_window", "fee_stress"],
            "entry_rules": ["形态策略必须跨时间段验证", "费用压力下PF仍需过线"],
            "avoid": ["把股票盘中形态不加验证地搬到50x合约"],
        },
        "risk_notes": ["如果裸信号收益接近0，高杠杆只会放大噪音和费用，不会创造 edge。"],
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    full_url = url
    if params:
        full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0 local-strategy-researcher/0.1"})
    with urllib.request.urlopen(request, timeout=25) as response:  # noqa: S310 - user-requested bounded fetch.
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> tuple[bytes, str, bool, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 local-strategy-researcher/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:  # noqa: S310 - user-requested bounded fetch.
            raw = response.read(MAX_SNAPSHOT_BYTES + 1)
            return raw[:MAX_SNAPSHOT_BYTES], response.headers.get("content-type", "unknown"), len(raw) > MAX_SNAPSHOT_BYTES, None
    except (urllib.error.URLError, TimeoutError) as exc:
        fallback = subprocess.run(
            [
                "curl",
                "-L",
                "--max-time",
                "25",
                "-A",
                "Mozilla/5.0 local-strategy-researcher/0.1",
                url,
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        raw = fallback.stdout[:MAX_SNAPSHOT_BYTES]
        if raw:
            return raw, "unknown", len(fallback.stdout) > MAX_SNAPSHOT_BYTES, f"urllib_failed_then_curl_ok: {exc}"
        return b"", "unknown", False, f"{exc}; curl_stderr={fallback.stderr.decode('utf-8', errors='replace')[:500]}"


def html_to_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_bilibili_manifest() -> dict[str, Any]:
    view = request_json(BILIBILI_VIEW_API, {"bvid": BILIBILI_BVID})
    if view.get("code") != 0:
        raise RuntimeError(f"Bilibili view API failed: {view.get('code')} {view.get('message')}")
    data = view["data"]
    pages = data.get("pages") or []
    subtitle_status = []
    for page in pages:
        player = request_json(BILIBILI_PLAYER_API, {"bvid": BILIBILI_BVID, "cid": page["cid"]})
        player_data = player.get("data") or {}
        subtitles = (player_data.get("subtitle") or {}).get("subtitles") or []
        subtitle_status.append(
            {
                "page": page.get("page"),
                "cid": page.get("cid"),
                "part": page.get("part"),
                "need_login_subtitle": bool(player_data.get("need_login_subtitle")),
                "subtitle_count": len(subtitles),
                "subtitle_urls": [item.get("subtitle_url") for item in subtitles if item.get("subtitle_url")],
            }
        )
    manifest = {
        "id": "bilibili_price_action_course",
        "fetched_at_utc": now_utc(),
        "source_type": "bilibili_multi_part_video",
        "url": f"https://www.bilibili.com/video/{BILIBILI_BVID}/",
        "bvid": BILIBILI_BVID,
        "title": data.get("title"),
        "owner": (data.get("owner") or {}).get("name"),
        "aid": data.get("aid"),
        "page_count": len(pages),
        "duration_seconds": data.get("duration"),
        "rights": data.get("rights"),
        "subtitle_access": {
            "public_subtitle_pages": sum(1 for item in subtitle_status if item["subtitle_count"] > 0),
            "login_required_pages": sum(1 for item in subtitle_status if item["need_login_subtitle"]),
            "note": "Only public subtitle metadata is fetched. Add legally obtained transcripts under raw_sources/bilibili/transcripts/ if needed.",
        },
        "pages": pages,
        "subtitle_status": subtitle_status,
        "copyright_policy": {
            "stored": ["metadata", "page list", "public subtitle metadata if available"],
            "not_stored": ["video files", "paid content", "full copyrighted transcripts unless provided locally with permission"],
        },
    }
    write_json(BILIBILI_DIR / "bilibili_price_action_course_manifest.json", manifest)
    return manifest


def fetch_public_sources() -> list[dict[str, Any]]:
    results = []
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    for source in PUBLIC_WEB_SOURCES:
        raw, content_type, truncated, error = fetch_bytes(source["url"])
        snapshot_path = SNAPSHOTS / f"{source['id']}.snapshot"
        text_path = SNAPSHOTS / f"{source['id']}.txt"
        if raw:
            snapshot_path.write_bytes(raw)
            text_path.write_text(html_to_text(raw)[:120_000] + "\n", encoding="utf-8")
        results.append(
            {
                **source,
                "fetched_at_utc": now_utc(),
                "fetch_status": "ok" if raw else "failed",
                "error": error if error else (None if raw else "empty_response"),
                "snapshot": rel(snapshot_path) if raw else None,
                "text_extract": rel(text_path) if raw else None,
                "bytes": len(raw),
                "content_type": content_type,
                "truncated": truncated,
            }
        )
    write_json(RAW_SOURCES / "public_web_sources_manifest.json", {"sources": results})
    return results


def build_books_manifest() -> dict[str, Any]:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": "al_brooks_books_manual_import",
        "updated_at_utc": now_utc(),
        "status": "waiting_for_licensed_local_files",
        "books": [
            {
                "title": "Trading Price Action Trends",
                "author": "Al Brooks",
                "publisher": "Wiley",
                "status": "copyrighted_not_downloaded",
            },
            {
                "title": "Trading Price Action Trading Ranges",
                "author": "Al Brooks",
                "publisher": "Wiley",
                "status": "copyrighted_not_downloaded",
            },
            {
                "title": "Trading Price Action Reversals",
                "author": "Al Brooks",
                "publisher": "Wiley",
                "status": "copyrighted_not_downloaded",
            },
            {
                "title": "Reading Price Charts Bar by Bar",
                "author": "Al Brooks",
                "publisher": "Wiley",
                "status": "copyrighted_not_downloaded",
            },
        ],
        "local_import_instruction": "Put legally owned PDFs/TXT/EPUB exports here, then summarize into short knowledge cards. Do not commit book files.",
    }
    write_json(BOOKS_DIR / "al_brooks_books_manifest.json", manifest)
    readme = """# Books To Add

这里放你合法拥有的书籍文本/PDF/EPUB 导出，例如 Al Brooks 的 Wiley 价格行为学书。

当前脚本不会从非官方 PDF 站下载书籍，也不会把整本书内容提交进仓库。建议流程：

1. 把你自己拥有的文件放到本目录。
2. 后续只抽取短知识卡、概念、量化规则和来源定位。
3. 不把大段原文、整章翻译或盗版 PDF 写入 Git。
"""
    (BOOKS_DIR / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def write_cards() -> list[dict[str, Any]]:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    cards = []
    for card in STARTER_CARDS:
        payload = {
            **card,
            "version": 1,
            "created_at_utc": now_utc(),
            "copyright_note": "Original local summary for strategy research. Not a verbatim excerpt.",
            "agent_use": {
                "when_to_retrieve": card["concepts"],
                "must_turn_into_testable_rule": True,
                "must_backtest_before_candidate": True,
            },
        }
        write_json(CARDS_DIR / f"{card['id']}.json", payload)
        cards.append(payload)
    return cards


def build_index(cards: list[dict[str, Any]], bilibili: dict[str, Any], web_sources: list[dict[str, Any]], books: dict[str, Any]) -> dict[str, Any]:
    concept_index: dict[str, list[str]] = {}
    for card in cards:
        for concept in card["concepts"]:
            concept_index.setdefault(concept, []).append(card["id"])
    index = {
        "generated_at_utc": now_utc(),
        "knowledge_root": rel(KNOWLEDGE_ROOT),
        "card_count": len(cards),
        "concept_index": concept_index,
        "sources": {
            "bilibili": {
                "title": bilibili.get("title"),
                "page_count": bilibili.get("page_count"),
                "manifest": rel(BILIBILI_DIR / "bilibili_price_action_course_manifest.json"),
            },
            "public_web": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "url": item["url"],
                    "fetch_status": item["fetch_status"],
                    "text_extract": item["text_extract"],
                }
                for item in web_sources
            ],
            "books": {
                "status": books["status"],
                "manifest": rel(BOOKS_DIR / "al_brooks_books_manifest.json"),
            },
        },
    }
    write_json(INDEX_DIR / "price_action_knowledge_index.json", index)
    return index


def write_report(index: dict[str, Any], bilibili: dict[str, Any], web_sources: list[dict[str, Any]], books: dict[str, Any]) -> None:
    lines = [
        "# Price Action Knowledge Base",
        "",
        f"- Generated UTC: `{index['generated_at_utc']}`",
        f"- Knowledge root: `{index['knowledge_root']}`",
        f"- Knowledge cards: `{index['card_count']}`",
        "",
        "## Bilibili Course",
        "",
        f"- Title: `{bilibili.get('title')}`",
        f"- Owner: `{bilibili.get('owner')}`",
        f"- Pages: `{bilibili.get('page_count')}`",
        f"- Duration seconds: `{bilibili.get('duration_seconds')}`",
        f"- Public subtitle pages: `{bilibili['subtitle_access']['public_subtitle_pages']}`",
        f"- Login-required subtitle pages: `{bilibili['subtitle_access']['login_required_pages']}`",
        f"- Manifest: `{index['sources']['bilibili']['manifest']}`",
        "",
        "## Public Web Sources",
        "",
        "| Source | Status | Bytes | Text Extract |",
        "|---|---|---:|---|",
    ]
    for item in web_sources:
        lines.append(f"| {item['id']} | {item['fetch_status']} | {item['bytes']} | `{item['text_extract']}` |")
    lines.extend(
        [
            "",
            "## Books",
            "",
            f"- Status: `{books['status']}`",
            f"- Manifest: `{rel(BOOKS_DIR / 'al_brooks_books_manifest.json')}`",
            "- Al Brooks/Wiley books are copyrighted; this builder records bibliographic targets and waits for legally owned local files.",
            "",
            "## Concept Index",
            "",
            "| Concept | Cards |",
            "|---|---|",
        ]
    )
    for concept, card_ids in sorted(index["concept_index"].items()):
        lines.append(f"| {concept} | {', '.join(card_ids)} |")
    lines.extend(
        [
            "",
            "## Agent Usage",
            "",
            "1. Query cards before generating a strategy hypothesis.",
            "2. Convert at most 1-3 retrieved concepts into testable Freqtrade rules.",
            "3. Backtest and write the result into research memory before reusing the idea.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORT_JSON, {"index": index, "report": rel(REPORT_MD)})


def main() -> None:
    for path in [KNOWLEDGE_ROOT, RAW_SOURCES, SNAPSHOTS, BILIBILI_DIR, CARDS_DIR, INDEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    bilibili = build_bilibili_manifest()
    web_sources = fetch_public_sources()
    books = build_books_manifest()
    cards = write_cards()
    index = build_index(cards, bilibili, web_sources, books)
    write_report(index, bilibili, web_sources, books)
    print(f"Wrote {rel(REPORT_MD)}")
    print(f"Wrote {rel(INDEX_DIR / 'price_action_knowledge_index.json')}")


if __name__ == "__main__":
    main()
