"""批量回填：把所有已跑完的月份重新写入两个 NFlX8_4 文档。

会重算逐笔明细（含「初始投入 / 补仓投入」两列）与月度汇总，
并把主文档对应年份汇总表该月列一并刷新（含 2026 年新跑出来的月份）。

用法：
    $env:PYTHONUTF8="1"; py -3.14 tools/refill_all_months.py
    $env:PYTHONUTF8="1"; py -3.14 tools/refill_all_months.py 202601 202602   # 只重填指定月
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import update_docs_from_result as U  # noqa: E402

OUT_DIR = U.OUT_DIR


def available_months():
    zips = sorted(OUT_DIR.glob("binance_nflx8_4_full343_*.zip"))
    out = []
    for z in zips:
        m = re.search(r"_(\d{6})\.zip$", z.name)
        if m:
            out.append(m.group(1))
    return out


def main():
    months = sys.argv[1:] or available_months()
    if not months:
        raise SystemExit("没有任何回测结果包")

    total = {
        "交易数": 0,
        "总利润(USDT)": 0.0,
        "总投入(USDT)": 0.0,
        "初始投入(USDT)": 0.0,
        "补仓投入(USDT)": 0.0,
        "总补仓次数": 0,
        "做多笔数": 0,
        "做空笔数": 0,
        "多头盈亏(USDT)": 0.0,
        "空头盈亏(USDT)": 0.0,
        "盈利月份数": 0,
    }
    month_rows = []
    for ym in months:
        print(f"\n===== {ym} =====")
        summ = U.process(ym)
        for k in (
            "交易数",
            "总利润(USDT)",
            "总投入(USDT)",
            "初始投入(USDT)",
            "补仓投入(USDT)",
            "总补仓次数",
            "做多笔数",
            "做空笔数",
            "多头盈亏(USDT)",
            "空头盈亏(USDT)",
        ):
            total[k] += summ[k]
        if summ["总利润(USDT)"] > 0:
            total["盈利月份数"] += 1
        month_rows.append((ym, summ))

    print("\n================ 全部月份汇总 ================")
    n_months = len(months)
    print(f"月份数：{n_months}（{months[0]} ~ {months[-1]}）")
    print(f"总交易数：{total['交易数']}")
    print(f"总利润：{total['总利润(USDT)']:.2f} USDT")
    print(
        f"总投入：{total['总投入(USDT)']:.2f} USDT｜初始：{total['初始投入(USDT)']:.2f}｜"
        f"补仓：{total['补仓投入(USDT)']:.2f}（占 "
        f"{total['补仓投入(USDT)'] / total['总投入(USDT)'] * 100:.2f}%）"
    )
    print(f"总补仓次数：{total['总补仓次数']}")
    print(f"做多：{total['做多笔数']}｜做空：{total['做空笔数']}")
    print(
        f"多头盈亏：{total['多头盈亏(USDT)']:.2f}｜空头盈亏：{total['空头盈亏(USDT)']:.2f}"
    )
    print(f"盈利月份：{total['盈利月份数']}/{n_months}")
    print(
        f"加权月化收益率：{total['总利润(USDT)'] / (1000.0 * n_months) * 100:.2f}%"
    )

    print("\n--- 逐月 投入 / 补仓 ---")
    print("| 月份 | 交易数 | 总投入 | 初始投入 | 补仓投入 | 补仓次数 | 利润 |")
    for ym, s in month_rows:
        print(
            f"| {ym} | {s['交易数']} | {s['总投入(USDT)']:.2f} | {s['初始投入(USDT)']:.2f} | "
            f"{s['补仓投入(USDT)']:.2f} | {s['总补仓次数']} | {s['总利润(USDT)']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
