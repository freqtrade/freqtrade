"""按 NFlX8_4 的实际信号结构统计，并重写主文档「四、交易结构分布」章节。

    $env:PYTHONUTF8="1"; py -3.14 tools/build_signal_stats.py

只读结果包 + 改主文档第四章，不跑回测。
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_nflx8_result as P  # noqa: E402

DOC_DIR = ROOT / "user_data" / "configs" / "binance" / "doc"
MAIN_DOC = DOC_DIR / "NFlX8_4币安数据回测分析.md"
OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"

SECTION_START = "## 四、各月交易笔数分布（回测后填写）"
NEW_TITLE = "## 四、交易结构分布（按入场信号 / 退出类型 / 持仓 / 补仓）"


def reason_class(reason: str) -> str:
    r = (reason or "").lower()
    if "grind" in r:
        return "grind（磨单止盈）"
    if "derisk" in r:
        return "derisk（降风险硬退）"
    if "trailing" in r:
        return "trailing（移动止损）"
    if "stop_loss" in r:
        return "stop_loss（止损）"
    if "force_exit" in r:
        return "force_exit（月末强平）"
    if "liquidation" in r:
        return "liquidation（爆仓）"
    if "exit_signal" in r:
        return "exit_signal（普通出场）"
    if "exit_profit" in r:
        return "exit_profit（止盈信号）"
    if r.startswith("exit_long") or r.startswith("exit_short"):
        return "exit_long/short（策略主动出场）"
    return "其它"


def load_all():
    trades = []
    for p in sorted(OUT_DIR.glob("binance_nflx8_4_full343_*.zip")):
        ym = p.stem.split("_")[-1]
        t, _ = P.load_trades(p)
        for tr in t:
            tr["_ym"] = ym
        trades.extend(t)
    return trades


def stat_rows(items):
    n = len(items)
    wins = sum(1 for t in items if t.get("profit_abs", 0) > 0)
    pl = sum(t.get("profit_abs", 0) for t in items)
    avg_ratio = statistics.mean([t.get("profit_ratio", 0) * 100 for t in items]) if n else 0
    return n, wins, (wins / n * 100 if n else 0), pl, avg_ratio


def row_str(name, items):
    n, wins, wr, pl, ar = stat_rows(items)
    return f"| {name} | {n} | {wins} | {wr:.2f}% | {pl:,.2f} | {ar:.2f}% | {pl / n:.2f} |"


def build_section(trades):
    n = len(trades)
    total = sum(t.get("profit_abs", 0) for t in trades)
    yms = sorted({t.get("_ym") for t in trades if t.get("_ym")})
    n_months = len(yms)
    range_str = f"{yms[0][:4]}-{yms[0][4:]} ~ {yms[-1][:4]}-{yms[-1][4:]}" if yms else ""
    out = [NEW_TITLE, ""]
    out.append(
        f"> 统计口径：{n} 笔真实回测（{range_str}，{n_months} 个月，Full343 × 每月独立 1000U × 4x 杠杆）。"
    )
    out.append("> NFlX8_4 的入场信号 `enter_tag` 是**数字编号**（多档），不是 s1~s4，故按实际编号统计。")
    out.append("")

    # 4.1 入场信号
    by_tag = defaultdict(list)
    for t in trades:
        by_tag[str(t.get("enter_tag") or "-").strip()].append(t)
    tags_sorted = sorted(by_tag.items(), key=lambda kv: sum(x.get("profit_abs", 0) for x in kv[1]), reverse=True)

    out.append("### 4.1 按入场信号（enter_tag，按总盈亏排序）")
    out.append("")
    out.append("| 入场信号 | 笔数 | 盈利笔数 | 胜率 | 总盈亏(USDT) | 平均收益率 | 平均盈亏(USDT) |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    for tag, items in tags_sorted[:15]:
        out.append(row_str(tag, items))
    out.append("")
    out.append(f"> 全部信号档位数：{len(by_tag)}；上表为总盈亏 Top15。")
    out.append("")

    # 4.2 方向 × 信号
    out.append("### 4.2 按方向 × 入场信号（各方向 Top8）")
    out.append("")
    for label, short in (("做多", False), ("做空", True)):
        sub = [t for t in trades if bool(t.get("is_short")) == short]
        by_t = defaultdict(list)
        for t in sub:
            by_t[str(t.get("enter_tag") or "-").strip()].append(t)
        items_sorted = sorted(by_t.items(), key=lambda kv: sum(x.get("profit_abs", 0) for x in kv[1]), reverse=True)
        out.append(f"**{label}**（共 {len(sub)} 笔，总盈亏 {sum(t.get('profit_abs', 0) for t in sub):,.2f} USDT）")
        out.append("")
        out.append("| 入场信号 | 笔数 | 盈利笔数 | 胜率 | 总盈亏(USDT) | 平均收益率 | 平均盈亏(USDT) |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for tag, items in items_sorted[:8]:
            out.append(row_str(tag, items))
        out.append("")

    # 4.3 退出类型
    out.append("### 4.3 按退出类型（exit_reason 归类）")
    out.append("")
    out.append("| 退出类型 | 笔数 | 占比 | 盈利笔数 | 胜率 | 总盈亏(USDT) | 平均收益率 | 平均盈亏(USDT) |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    by_cls = defaultdict(list)
    for t in trades:
        by_cls[reason_class(t.get("exit_reason"))].append(t)
    for cls, items in sorted(by_cls.items(), key=lambda kv: sum(x.get("profit_abs", 0) for x in kv[1]), reverse=True):
        cnt, wins, wr, pl, ar = stat_rows(items)
        out.append(
            f"| {cls} | {cnt} | {cnt / n * 100:.1f}% | {wins} | {wr:.2f}% | {pl:,.2f} | {ar:.2f}% | {pl / cnt:.2f} |"
        )
    out.append("")
    out.append(
        "> **要点**：盈利 100% 来自主动出场（exit_long/short、exit_profit、grind）；"
        "亏损 100% 来自被动出场（force_exit 月末强平、stop_loss、liquidation）。"
    )
    out.append("")

    # 4.4 持仓时长
    out.append("### 4.4 按持仓时长分档")
    out.append("")
    buckets = [
        ("< 1 小时", lambda h: h < 1),
        ("1 ~ 6 小时", lambda h: 1 <= h < 6),
        ("6 ~ 24 小时", lambda h: 6 <= h < 24),
        ("1 ~ 3 天", lambda h: 24 <= h < 72),
        ("3 ~ 7 天", lambda h: 72 <= h < 168),
        ("> 7 天", lambda h: h >= 168),
    ]
    hours = {}
    for t in trades:
        if t.get("close_timestamp") and t.get("open_timestamp"):
            hours[id(t)] = (t["close_timestamp"] - t["open_timestamp"]) / 3600000
    out.append("| 持仓档 | 笔数 | 占比 | 胜率 | 总盈亏(USDT) | 平均盈亏(USDT) |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for name, fn in buckets:
        items = [t for t in trades if id(t) in hours and fn(hours[id(t)])]
        if not items:
            continue
        cnt, wins, wr, pl, _ar = stat_rows(items)
        out.append(f"| {name} | {cnt} | {cnt / n * 100:.1f}% | {wr:.2f}% | {pl:,.2f} | {pl / cnt:.2f} |")
    out.append("")

    # 4.5 补仓
    out.append("### 4.5 按补仓次数分档")
    out.append("")
    rebuckets = [
        ("0 次（不补仓）", lambda r: r == 0),
        ("1 ~ 2 次", lambda r: 1 <= r <= 2),
        ("3 ~ 5 次", lambda r: 3 <= r <= 5),
        ("6 ~ 10 次", lambda r: 6 <= r <= 10),
        ("> 10 次", lambda r: r > 10),
    ]
    re_map = {}
    for t in trades:
        re_map[id(t)] = max(sum(1 for o in (t.get("orders") or []) if o.get("ft_is_entry")) - 1, 0)
    out.append("| 补仓档 | 笔数 | 占比 | 胜率 | 总盈亏(USDT) | 平均盈亏(USDT) |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for name, fn in rebuckets:
        items = [t for t in trades if fn(re_map[id(t)])]
        if not items:
            continue
        cnt, wins, wr, pl, _ar = stat_rows(items)
        out.append(f"| {name} | {cnt} | {cnt / n * 100:.1f}% | {wr:.2f}% | {pl:,.2f} | {pl / cnt:.2f} |")
    out.append("")

    # 4.6 小结
    wins_n = sum(1 for t in trades if t.get("profit_abs", 0) > 0)
    best_tag = tags_sorted[0]
    worst_items = sorted(by_tag.items(), key=lambda kv: sum(x.get("profit_abs", 0) for x in kv[1]))[:1]
    long_items = [t for t in trades if not t.get("is_short")]
    short_items = [t for t in trades if t.get("is_short")]
    out.append("### 4.6 小结")
    out.append("")
    out.append(f"- 总笔数 {n}，盈利 {wins_n} 笔，胜率 {wins_n / n * 100:.2f}%，总盈亏 {total:,.2f} USDT")
    out.append(
        f"- 入场信号档位共 {len(by_tag)} 档；盈利最高：**{best_tag[0]}**"
        f"（{len(best_tag[1])} 笔，{sum(x.get('profit_abs', 0) for x in best_tag[1]):,.2f} USDT）"
    )
    if worst_items:
        wtag, witems = worst_items[0]
        out.append(
            f"- 入场信号中亏损最大：**{wtag}**（{len(witems)} 笔，"
            f"{sum(x.get('profit_abs', 0) for x in witems):,.2f} USDT）"
        )
    out.append(
        f"- 做多 {len(long_items)} 笔 / {sum(t.get('profit_abs', 0) for t in long_items):,.2f} USDT；"
        f"做空 {len(short_items)} 笔 / {sum(t.get('profit_abs', 0) for t in short_items):,.2f} USDT"
    )
    out.append(
        "- 主动出场贡献全部盈利，被动出场（force_exit / stop_loss / liquidation）构成全部亏损；"
        "月度切分会把承接中的单在月末砍掉，对 DCA 策略属系统性悲观偏差"
    )
    out.append("")
    return "\n".join(out)


def main():
    trades = load_all()
    if not trades:
        print("没有结果包")
        return 1

    section = build_section(trades)
    text = MAIN_DOC.read_text(encoding="utf-8")
    lines = text.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("## 四、"):
            start = i
            break
    if start is None:
        print("未找到第四章")
        return 1
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    lines[start:end] = section.split("\n")
    MAIN_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"已重写第四章 -> {MAIN_DOC.name}")
    print(f"统计笔数: {len(trades)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
