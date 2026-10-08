"""汇总 NFlX8_4 所有已完成月份的结果，并回填主文档的「全局汇总」与明细的「汇总小结」。

用法：
    $env:PYTHONUTF8="1"; py -3.14 tools/summarize_all.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "user_data" / "configs" / "binance" / "doc"
MAIN_DOC = DOC_DIR / "NFlX8_4币安数据回测分析.md"
DETAIL_DOC = DOC_DIR / "NFlX8_4每月回测明细.md"
OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"
START_WALLET = 1000.0


def load_summaries():
    """优先用已缓存的 *_summary.json，没有就直接解析结果包。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import parse_nflx8_result as P

    data = {}
    for p in sorted(OUT_DIR.glob("binance_nflx8_4_full343_*.zip")):
        ym = p.stem.split("_")[-1]
        cache = OUT_DIR / f"{ym}_summary.json"
        if cache.exists():
            data[ym] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            trades, raw = P.load_trades(p)
            data[ym] = P.summarize(trades, raw)
    return data


def fmt(v, nd=2):
    return f"{v:,.{nd}f}"


def main():
    data = load_summaries()
    if not data:
        print("没有找到任何 *_summary.json")
        return 1

    months = sorted(data)
    n_months = len(months)
    total_trades = sum(d["交易数"] for d in data.values())
    total_wins = sum(d["盈利笔数"] for d in data.values())
    total_profit = sum(d["总利润(USDT)"] for d in data.values())
    win_months = sum(1 for d in data.values() if d["总利润(USDT)"] > 0)
    loss_months = [ym for ym, d in data.items() if d["总利润(USDT)"] <= 0]
    long_pl = sum(d["多头盈亏(USDT)"] for d in data.values())
    short_pl = sum(d["空头盈亏(USDT)"] for d in data.values())
    monthly_ret = total_profit / (START_WALLET * n_months) * 100
    worst = min(data.items(), key=lambda kv: kv[1]["总利润(USDT)"])
    best = max(data.items(), key=lambda kv: kv[1]["总利润(USDT)"])
    avg_dd = None

    print(f"已完成月份: {n_months} ({months[0]} ~ {months[-1]})")
    print(f"总交易数: {total_trades}")
    print(f"总盈利笔数: {total_wins}  整体胜率: {total_wins / total_trades * 100:.2f}%")
    print(f"总利润: {total_profit:.2f} USDT")
    print(f"加权月化收益率: {monthly_ret:.2f}%")
    print(f"盈利月份: {win_months}/{n_months}   亏损月份: {loss_months or '无'}")
    print(f"多头累计: {long_pl:.2f}  空头累计: {short_pl:.2f}")

    # ---- 主文档：三、全局汇总
    txt = MAIN_DOC.read_text(encoding="utf-8")
    lines = txt.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("## 三、全局汇总"):
            start = i
            break
    if start is not None:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
        new_block = [
            f"- 覆盖月份数：{n_months}（{months[0][:4]}-{months[0][4:]} ~ "
            f"{months[-1][:4]}-{months[-1][4:]}）",
            f"- 总交易数：{total_trades} 笔",
            f"- 总盈利笔数：{total_wins} 笔，整体胜率：{total_wins / total_trades * 100:.2f}%",
            f"- 总利润（各月独立累加，因每月起 1000U）：{fmt(total_profit)} USDT",
            f"- 加权月化收益率（总利润 / (1000 × 月数)）：{monthly_ret:.2f}%",
            f"- 多头累计盈亏：{fmt(long_pl)} USDT；空头累计盈亏：{fmt(short_pl)} USDT",
            f"- 盈利月份数：{win_months}/{n_months}"
            + (f"（亏损月份：{', '.join(loss_months)}）" if loss_months else "（无亏损月份）"),
            f"- 最差单月：{worst[0]}（{fmt(worst[1]['总利润(USDT)'])} USDT）；"
            f"最优单月：{best[0]}（{fmt(best[1]['总利润(USDT)'])} USDT）",
        ]
        # 保留原有说明引用块（以 "> 说明" 开头的行）
        keep = [l for l in lines[start + 1 : end] if l.startswith("> 说明")]
        lines[start + 1 : end] = new_block + ([""] + keep if keep else [])
        MAIN_DOC.write_text("\n".join(lines), encoding="utf-8")
        print(f"[主文档] 已更新全局汇总 -> {MAIN_DOC.name}")

    # ---- 明细：汇总小结表
    dtxt = DETAIL_DOC.read_text(encoding="utf-8")
    by_year = {}
    for ym, d in data.items():
        by_year.setdefault(ym[:4], []).append((ym, d))
    year_rows = []
    for year in sorted(by_year):
        items = sorted(by_year[year])
        wins = sum(1 for _ym, d in items if d["总利润(USDT)"] > 0)
        profit = sum(d["总利润(USDT)"] for _ym, d in items)
        avg_ret = profit / (START_WALLET * len(items)) * 100
        worst_y = min(items, key=lambda kv: kv[1]["总利润(USDT)"])
        best_y = max(items, key=lambda kv: kv[1]["总利润(USDT)"])
        year_rows.append(
            f"| {year} 年 | {wins}/{len(items)} | {avg_ret:.2f}% | {worst_y[0]} "
            f"（{worst_y[1]['总利润(USDT)']:.2f}） | {best_y[0]}（{best_y[1]['总利润(USDT)']:.2f}） "
            f"| - | - |"
        )
    year_rows.append(
        f"| **全周期** | {win_months}/{n_months} | {monthly_ret:.2f}% | {worst[0]} "
        f"（{worst[1]['总利润(USDT)']:.2f}） | {best[0]}（{best[1]['总利润(USDT)']:.2f}） | - | - |"
    )

    dlines = dtxt.split("\n")
    s = None
    for i, line in enumerate(dlines):
        if line.strip().startswith("| 周期 | 盈利月数 |"):
            s = i
            break
    if s is not None:
        e = s + 2
        while e < len(dlines) and dlines[e].lstrip().startswith("|"):
            e += 1
        dlines[s + 2 : e] = year_rows
        DETAIL_DOC.write_text("\n".join(dlines), encoding="utf-8")
        print(f"[明细] 已更新汇总小结 -> {DETAIL_DOC.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
