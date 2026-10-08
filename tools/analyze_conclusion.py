"""为 NFlX8_4 主文档「五、结论」重算关键指标。

只读取回测结果包 + BTC 日线数据，输出可直接回填进结论章节的精确数字。
用法：
    $env:PYTHONUTF8="1"; py -3.14 tools/analyze_conclusion.py
"""

import sys
from pathlib import Path

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_nflx8_result as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "user_data" / "backtest_results" / "NFlX8"
BTC_FEATHER = ROOT / "user_data" / "data" / "binance" / "futures" / "BTC_USDT_USDT-1d-futures.feather"
START_WALLET = 1000.0


def btc_monthly_change():
    """返回 {YYYYMM: 百分比}：当月首日开盘 → 次月首日开盘。"""
    df = pd.read_feather(BTC_FEATHER)
    # 兼容两种列名
    date_col = None
    for c in ("date", "open_date", "open_datetime", "timestamp"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # 常见：第一列是 datetime 索引/列
        date_col = df.columns[0]
    df = df.copy()
    # 统一转成 datetime（避免 tz 错误）
    dt = pd.to_datetime(df[date_col], utc=True)
    df["_dt"] = dt.dt.tz_convert("UTC")
    df = df.sort_values("_dt")
    df["ym"] = df["_dt"].dt.strftime("%Y%m")
    # 取每月首日（最小 datetime）那行的开盘价
    open_col = "open" if "open" in df.columns else df.columns[1]
    first = df.groupby("ym").first().reset_index()
    first["_d"] = pd.to_datetime(first["_dt"])
    first = first.sort_values("_d")
    vals = dict(zip(first["ym"], first[open_col].astype(float)))
    out = {}
    months = sorted(vals)
    for m in months:
        y, mo = int(m[:4]), int(m[4:])
        if mo == 12:
            ny, nmo = y + 1, 1
        else:
            ny, nmo = y, mo + 1
        nm = f"{ny:04d}{nmo:02d}"
        if nm in vals and vals[m] != 0:
            out[m] = (vals[nm] / vals[m] - 1) * 100
    return out


def main():
    btc = btc_monthly_change()

    months = sorted(p.stem.split("_")[-1] for p in OUT_DIR.glob("binance_nflx8_4_full343_*.zip"))
    print(f"结果包月份数: {len(months)} ({months[0]} ~ {months[-1]})")

    # 校验 BTC 计算：挑几个有文档对照的月份
    for check in ("202401", "202402", "202404", "202606"):
        if check in btc:
            print(f"  [校验] BTC {check}: {btc[check]:+.2f}%")

    total_profit = 0.0
    win_months = 0
    down_months = []          # BTC 下跌的月份
    down_profit_months = []   # BTC 下跌且策略盈利
    long_n = short_n = 0
    long_pl = short_pl = 0.0
    rows = []
    for ym in months:
        trades, _ = P.load_trades(OUT_DIR / f"binance_nflx8_4_full343_{ym}.zip")
        s = P.summarize(trades, None)
        pf = s["总利润(USDT)"]
        total_profit += pf
        if pf > 0:
            win_months += 1
        chg = btc.get(ym)
        is_down = (chg is not None and chg < 0)
        if is_down:
            down_months.append(ym)
            if pf > 0:
                down_profit_months.append(ym)
        long_n += s["做多笔数"]
        short_n += s["做空笔数"]
        long_pl += s["多头盈亏(USDT)"]
        short_pl += s["空头盈亏(USDT)"]
        rows.append((ym, chg, pf))

    n = len(months)
    print(f"\n盈利月份: {win_months}/{n}")
    print(f"加权月化收益率: {total_profit / (START_WALLET * n) * 100:.2f}%")
    print(f"BTC 下跌月份数: {len(down_months)} -> {', '.join(down_months)}")
    print(f"  其中策略仍盈利: {len(down_profit_months)} -> {', '.join(down_profit_months)}")
    print(f"做多: {long_n} 笔 / {long_pl:,.2f} USDT")
    print(f"做空: {short_n} 笔 / {short_pl:,.2f} USDT")

    # 直接遍历所有 trades 统计 exit_reason 归类
    from build_signal_stats import reason_class  # noqa: E402
    cnt = {}
    pl = {}
    raw_other = []
    for ym in months:
        trades, _ = P.load_trades(OUT_DIR / f"binance_nflx8_4_full343_{ym}.zip")
        for t in trades:
            cls = reason_class(t.get("exit_reason"))
            cnt[cls] = cnt.get(cls, 0) + 1
            pl[cls] = pl.get(cls, 0.0) + t.get("profit_abs", 0)
            if cls == "其它":
                raw_other.append(t.get("exit_reason"))
    print("\n退出类型分布:")
    for cls in sorted(pl, key=lambda k: pl[k]):
        print(f"  {cls}: {cnt[cls]} 笔 / {pl[cls]:+,.2f} USDT")
    risk_keys = (
        "liquidation（爆仓）",
        "trailing（移动止损）",
        "stop_loss（止损）",
        "derisk（降风险硬退）",
    )
    risk_n = sum(cnt.get(cls, 0) for cls in risk_keys)
    risk_p = sum(pl.get(cls, 0.0) for cls in risk_keys)
    print(f"\n风险类(爆仓+移动止损+硬止损+derisk): {risk_n} 笔 / {risk_p:+,.2f} USDT")
    print(f"  其中爆仓: {cnt.get('liquidation（爆仓）', 0)} 笔 / {pl.get('liquidation（爆仓）', 0.0):+,.2f} USDT")
    print(f"  其中硬止损stop_loss: {cnt.get('stop_loss（止损）', 0)} 笔 / {pl.get('stop_loss（止损）', 0.0):+,.2f} USDT")
    print(f"  其中移动止损trailing: {cnt.get('trailing（移动止损）', 0)} 笔 / {pl.get('trailing（移动止损）', 0.0):+,.2f} USDT")
    print(f"  其中derisk: {cnt.get('derisk（降风险硬退）', 0)} 笔 / {pl.get('derisk（降风险硬退）', 0.0):+,.2f} USDT")
    print(f"  '其它' 原始 exit_reason 样本: {sorted(set(raw_other))[:10]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
