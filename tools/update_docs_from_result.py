"""把某个月的回测结果回填进 NFlX8_4 的两个文档。

用法：
    $env:PYTHONUTF8="1"; py -3.14 tools/update_docs_from_result.py 202401

做的事：
  1) 读 user_data/backtest_results/NFlX8/binance_nflx8_4_full343_<YYYYMM>.zip
  2) 更新主文档 NFlX8_4币安数据回测分析.md：对应年份汇总表的该月列
  3) 更新明细 NFlX8_4每月回测明细.md：该月章节的逐笔表 + 本月小结
"""

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_nflx8_result as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "user_data" / "configs" / "binance" / "doc"
MAIN_DOC = DOC_DIR / "NFlX8_4币安数据回测分析.md"
DETAIL_DOC = DOC_DIR / "NFlX8_4每月回测明细.md"
OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"
LOG_DIR = OUT_DIR / "logs"

START_WALLET = 1000.0


def days_of(y, m):
    s = date(y, m, 1)
    e = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return (e - s).days


def parse_drawdown_from_log(ym: str):
    """从回测日志里抓账户级最大回撤百分比。

    兼容新旧 freqtrade 日志：
    - 旧版：`Absolute drawdown (wallet balance) │ X USDT (Y%)` 即真实最大回撤。
    - 新版：该字段含义变为「低于初始钱包的回撤」(常为 0%)，真实值在
      `Max % of account underwater │ Y%`（无 balance 后缀那一行）。
    统一优先取标准「Max % of account underwater │ Y%」。
    """
    log = LOG_DIR / f"{ym}.log"
    if not log.exists():
        return "—"
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Max % of account underwater\s*│\s*([\d.]+)%", text)
    if m:
        return f"{float(m.group(1)):.2f}%"
    m2 = re.search(r"Absolute drawdown\s*│\s*[-\d.]+ USDT \(([\d.]+)%\)", text)
    return f"{float(m2.group(1)):.2f}%" if m2 else "—"


# ------------------------------------------------------------------ 主文档
def update_main(ym: str, values: dict):
    y, m = int(ym[:4]), int(ym[4:])
    text = MAIN_DOC.read_text(encoding="utf-8")
    lines = text.split("\n")

    # 定位 "### <年> 年" 段
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"### {y} 年":
            start = i
            break
    if start is None:
        print(f"[主文档] 未找到 '### {y} 年' 段")
        return False

    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("### ") or s.startswith("## "):
            end = j
            break

    # 表头行
    header_idx = None
    for j in range(start, end):
        if lines[j].lstrip().startswith("| 指标 |"):
            header_idx = j
            break
    if header_idx is None:
        print("[主文档] 未找到指标表头")
        return False

    header_cells = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    col = None
    for idx, name in enumerate(header_cells[1:], start=1):
        if name == f"{m:02d}":
            col = idx
            break
    if col is None:
        print(f"[主文档] 表头中没有 {m:02d} 月列")
        return False

    for j in range(header_idx, end):
        line = lines[j]
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        name = cells[0]
        if name in values and len(cells) > col:
            cells[col] = values[name]
            lines[j] = "| " + " | ".join(cells) + " |"

    MAIN_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"[主文档] 已更新 {y} 年 {m:02d} 月列 -> {MAIN_DOC.name}")
    return True


# ------------------------------------------------------------------ 明细文档
def update_detail(ym: str, trades_md_rows: list[str], summary_line: list[str]):
    y, m = int(ym[:4]), int(ym[4:])
    title_prefix = f"## {y}-{m:02d}月"
    text = DETAIL_DOC.read_text(encoding="utf-8")
    lines = text.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.startswith(title_prefix):
            start = i
            break
    if start is None:
        print(f"[明细] 未找到 {title_prefix} 章节")
        return False

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break

    # 标题笔数
    lines[start] = re.sub(r"（共.*?）", f"（共 {summary_line['trades']} 笔）", lines[start])

    # 找到表头/分隔行，重建数据行
    header_idx = None
    for j in range(start, end):
        if lines[j].lstrip().startswith("| 序号 |"):
            header_idx = j
            break
    if header_idx is None:
        print("[明细] 未找到逐笔表头")
        return False

    # 表头/分隔行一并升级为最新版（含 初始投入 / 补仓投入 两列）
    lines[header_idx] = trades_md_rows[0]
    lines[header_idx + 1] = trades_md_rows[1]

    # 表头之后到下一个空行/非表格行之间的内容全部替换为新数据行
    k = header_idx + 2  # 跳过表头 + 分隔行
    while k < end and lines[k].lstrip().startswith("|"):
        k += 1
    lines[header_idx + 2 : k] = trades_md_rows[2:]

    # 小结行：只处理本章节内「**本月小结**」之后的连续 bullet，绝不越界到下一章节
    sum_start = None
    for j in range(header_idx, len(lines)):
        if lines[j].startswith("**本月小结**"):
            sum_start = j
            break
    limit = len(lines)
    if sum_start is not None:
        for j in range(sum_start + 1, len(lines)):
            if lines[j].startswith("## "):
                limit = j
                break

    j = (sum_start + 1) if sum_start is not None else header_idx
    while j < limit:
        s = lines[j]
        if not s.startswith("- "):
            if s.strip() == "":
                j += 1
                continue
            break
        if s.startswith("- 交易数："):
            lines[j] = summary_line["line1"]
        elif s.startswith("- 总利润："):
            lines[j] = summary_line["line2"]
        elif s.startswith("- 多头盈亏："):
            lines[j] = summary_line["line3"]
            # 投入行紧跟在多空盈亏之后
            if summary_line.get("line4"):
                if j + 1 < len(lines) and lines[j + 1].startswith("- 总投入："):
                    lines[j + 1] = summary_line["line4"]
                else:
                    lines.insert(j + 1, summary_line["line4"])
                j += 1
        elif s.startswith("- 总投入：") and summary_line.get("line4"):
            lines[j] = summary_line["line4"]
        j += 1

    DETAIL_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"[明细] 已更新 {y}-{m:02d} 章节（{summary_line['trades']} 笔）-> {DETAIL_DOC.name}")
    return True


def process(ym: str):
    """回填单月：主文档汇总表 + 明细章节。"""
    y, m = int(ym[:4]), int(ym[4:])

    zip_path = OUT_DIR / f"binance_nflx8_4_full343_{ym}.zip"
    if not zip_path.exists():
        raise SystemExit(f"找不到结果包: {zip_path}")

    trades, _raw = P.load_trades(zip_path)
    summ = P.summarize(trades, _raw)
    dd = parse_drawdown_from_log(ym)
    n = summ["交易数"]
    days = days_of(y, m)

    values = {
        "总收益率": f"{summ['总利润(USDT)'] / START_WALLET * 100:.2f}%",
        "总利润(USDT)": f"{summ['总利润(USDT)']:.2f}",
        "胜率": summ["胜率"],
        "最大回撤(账户)": dd,
        "交易数": str(n),
        "做多笔数": str(summ["做多笔数"]),
        "做空笔数": str(summ["做空笔数"]),
        "多头盈亏(USDT)": f"{summ['多头盈亏(USDT)']:.2f}",
        "空头盈亏(USDT)": f"{summ['空头盈亏(USDT)']:.2f}",
        "总补仓次数": str(summ["总补仓次数"]),
        "日均单数(笔/天)": f"{n / days:.2f}",
    }
    print("回填值:", values)

    update_main(ym, values)

    rows = P.trades_markdown(trades).split("\n")  # 含表头两行
    summary_line = {
        "trades": n,
        "line1": f"- 交易数：{n}｜做多：{summ['做多笔数']}｜做空：{summ['做空笔数']}｜补仓次数：{summ['总补仓次数']}",
        "line2": (
            f"- 总利润：{summ['总利润(USDT)']:.2f} USDT｜总收益率："
            f"{summ['总利润(USDT)'] / START_WALLET * 100:.2f}%｜胜率：{summ['胜率']}｜最大回撤：{dd}"
        ),
        "line3": f"- 多头盈亏：{summ['多头盈亏(USDT)']:.2f} USDT｜空头盈亏：{summ['空头盈亏(USDT)']:.2f} USDT",
        "line4": (
            f"- 总投入：{summ['总投入(USDT)']:.2f} USDT（保证金，4x 杠杆）｜初始投入："
            f"{summ['初始投入(USDT)']:.2f} USDT｜补仓投入：{summ['补仓投入(USDT)']:.2f} USDT"
            f"（占 {summ['补仓占比']}）"
        ),
    }
    update_detail(ym, rows, summary_line)
    print("完成。")
    return summ


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    process(sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
