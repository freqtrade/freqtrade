"""只读统计：NFlX8_4 已回测月份的逐笔数据分布（不跑回测、不改文档）。

    $env:PYTHONUTF8="1"; py -3.14 tools/analyze_trades.py

输出：币种集中度 / 退出类型分布 / 多空结构 / 持仓时长 / 极端单笔。
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_nflx8_result as P  # noqa: E402

OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"


def reason_class(reason: str) -> str:
    r = (reason or "").lower()
    if "grind" in r:
        return "grind(磨单止盈)"
    if "derisk" in r:
        return "derisk(降风险硬退)"
    if "trailing" in r:
        return "trailing(移动止损)"
    if "stop_loss" in r:
        return "stop_loss(止损)"
    if "force_exit" in r:
        return "force_exit(月末强平)"
    if "exit_signal" in r:
        return "exit_signal(普通出场信号)"
    if "exit_profit" in r:
        return "exit_profit(止盈信号)"
    if r.startswith("exit_long") or r.startswith("exit_short"):
        return "exit_long/short(策略出场)"
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


def main():
    trades = load_all()
    if not trades:
        print("没有结果包")
        return 1

    n = len(trades)
    total = sum(t.get("profit_abs", 0) for t in trades)
    wins = [t for t in trades if t.get("profit_abs", 0) > 0]
    losses = [t for t in trades if t.get("profit_abs", 0) <= 0]

    print(f"总笔数: {n}   总盈亏: {total:,.2f} USDT")
    print(f"盈利笔: {len(wins)}   亏损笔: {len(losses)}   胜率: {len(wins) / n * 100:.2f}%")
    print(f"平均单笔盈利: {statistics.mean([t['profit_abs'] for t in wins]):.2f}    "
          f"平均单笔亏损: {statistics.mean([t['profit_abs'] for t in losses]):.2f}")

    # ---------- 1. 币种集中度
    print("\n===== 1. 币种集中度 =====")
    by_pair = defaultdict(float)
    by_pair_n = defaultdict(int)
    for t in trades:
        by_pair[t["pair"]] += t.get("profit_abs", 0)
        by_pair_n[t["pair"]] += 1
    pairs = sorted(by_pair.items(), key=lambda kv: kv[1], reverse=True)
    print(f"参与交易的币种数: {len(pairs)}")
    print(f"盈利币种数: {sum(1 for _p, v in pairs if v > 0)}  "
          f"亏损币种数: {sum(1 for _p, v in pairs if v <= 0)}")

    for k in (5, 10, 20):
        s = sum(v for _p, v in pairs[:k])
        print(f"  利润 Top{k} 币合计: {s:,.2f} USDT  占总利润 {s / total * 100:.1f}%")

    print("\n利润 Top10 币:")
    print(f"  {'交易对':<20}{'笔数':>6}{'盈亏USDT':>12}")
    for p, v in pairs[:10]:
        print(f"  {p:<20}{by_pair_n[p]:>6}{v:>12,.2f}")
    print("\n亏损 Top5 币:")
    for p, v in pairs[-5:]:
        print(f"  {p:<20}{by_pair_n[p]:>6}{v:>12,.2f}")

    # ---------- 2. 退出类型
    print("\n===== 2. 退出类型分布（大类）=====")
    by_cls = defaultdict(list)
    for t in trades:
        by_cls[reason_class(t.get("exit_reason"))].append(t)
    print(f"  {'类型':<26}{'笔数':>6}{'占比':>8}{'胜率':>9}{'总盈亏':>14}{'平均':>10}")
    for cls, items in sorted(by_cls.items(), key=lambda kv: sum(x["profit_abs"] for x in kv[1]), reverse=True):
        s = sum(x.get("profit_abs", 0) for x in items)
        w = sum(1 for x in items if x.get("profit_abs", 0) > 0)
        print(f"  {cls:<26}{len(items):>6}{len(items) / n * 100:>7.1f}%"
              f"{w / len(items) * 100:>8.1f}%{s:>14,.2f}{s / len(items):>10,.2f}")

    print("\n退出原因细分 Top12（按总盈亏）:")
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[(t.get("exit_reason") or "-").split(" (")[0].strip()].append(t)
    print(f"  {'exit_reason':<48}{'笔数':>6}{'总盈亏':>14}")
    for r, items in sorted(by_reason.items(), key=lambda kv: sum(x["profit_abs"] for x in kv[1]), reverse=True)[:12]:
        s = sum(x.get("profit_abs", 0) for x in items)
        print(f"  {r:<48}{len(items):>6}{s:>14,.2f}")

    # ---------- 3. 多空结构
    print("\n===== 3. 多空结构 =====")
    for label, short in (("做多", False), ("做空", True)):
        items = [t for t in trades if bool(t.get("is_short")) == short]
        if not items:
            continue
        s = sum(t.get("profit_abs", 0) for t in items)
        w = sum(1 for t in items if t.get("profit_abs", 0) > 0)
        print(f"  {label}: {len(items)} 笔，胜率 {w / len(items) * 100:.2f}%，"
              f"总盈亏 {s:,.2f}，平均 {s / len(items):.2f}")

    # ---------- 4. 持仓时长 & 补仓
    print("\n===== 4. 持仓时长 / 补仓 =====")
    durs = [
        (t["close_timestamp"] - t["open_timestamp"]) / 3600000
        for t in trades if t.get("close_timestamp") and t.get("open_timestamp")
    ]
    if durs:
        durs_sorted = sorted(durs)
        print(f"  平均持仓: {statistics.mean(durs):.1f} 小时   中位: {statistics.median(durs):.1f} 小时   "
              f"最长: {max(durs):.1f} 小时")
        print(f"  持仓 > 24h 的笔数: {sum(1 for d in durs if d > 24)}   "
              f"> 72h: {sum(1 for d in durs if d > 72)}")
    reentries = [
        max(sum(1 for o in (t.get("orders") or []) if o.get("ft_is_entry")) - 1, 0)
        for t in trades
    ]
    print(f"  总补仓次数: {sum(reentries)}   有补仓的笔数: {sum(1 for r in reentries if r > 0)}   "
          f"单笔最多补仓: {max(reentries)}")

    # ---------- 5. 极端单笔
    print("\n===== 5. 极端单笔 =====")
    worst = sorted(trades, key=lambda t: t.get("profit_abs", 0))[:5]
    best = sorted(trades, key=lambda t: t.get("profit_abs", 0), reverse=True)[:5]
    print("  最大亏损 5 笔:")
    for t in worst:
        print(f"    {t['_ym']} {t['pair']:<18} {t.get('exit_reason', '')[:38]:<38} "
              f"{t.get('profit_abs', 0):>9,.2f}  {t.get('profit_ratio', 0) * 100:>7.2f}%")
    print("  最大盈利 5 笔:")
    for t in best:
        print(f"    {t['_ym']} {t['pair']:<18} {t.get('exit_reason', '')[:38]:<38} "
              f"{t.get('profit_abs', 0):>9,.2f}  {t.get('profit_ratio', 0) * 100:>7.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
